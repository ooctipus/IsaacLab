# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared vector updates trained against the frozen-contact physics residual."""

import numpy as np
import torch
import warp as wp
from experiments.neio import SolverNeIO
from experiments.neio.deformable import NeIODeformable
from experiments.neio.network import NeIONetwork, group_sum, history_features
from experiments.neio.optimizer import NeIOOptimizer
from experiments.neio.rigid import quat_rotate, quat_rotate_inv


def quaternion_product(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def rotation_quaternion(rotation):
    angle = rotation.norm(dim=-1, keepdim=True)
    return torch.cat((0.5 * torch.sinc(angle / (2 * torch.pi)) * rotation, torch.cos(0.5 * angle)), -1)


def quaternion_energy_gradient(q, world_torque_gradient):
    """Tangent quaternion cotangent whose pullback is the world rotation derivative."""
    vector = q[..., 3:] * world_torque_gradient + torch.linalg.cross(world_torque_gradient, q[..., :3])
    return 2 * torch.cat((vector, -(q[..., :3] * world_torque_gradient).sum(-1, keepdim=True)), -1)


def new_network(device, width=96):
    return NeIONetwork(9, width, 3, 1, initial_scale=1e-4).to(device)


class SharedOptimizer(NeIODeformable):
    """Adapt the existing particle optimizer protocol to shared particle/body vector nodes.

    This is the study's experimental representation, not the port's rigid architecture.
    The network is supplied by the composition root and never constructed by a scene.
    """

    def __init__(self, network, kind, learn=False, history="clear"):
        NeIOOptimizer.__init__(self, None)
        self.network = network
        self.kind = kind
        self.learn = learn
        self.history_at_step_start = history
        self.group_by = "world"
        self.inference = "torch"

    def bind(self, solver):
        if self.kind == "particle":
            super().bind(solver)
            return
        NeIOOptimizer.bind(self, solver)
        model = solver.model
        device = next(self.network.parameters()).device
        n = model.body_count
        free = wp.to_torch(model.body_inv_mass) > 0
        world = wp.to_torch(model.body_world).long()
        _, group = torch.unique(world, sorted=True, return_inverse=True)
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        valid = (parent >= 0) & (child >= 0)
        pairs = np.stack((parent[valid], child[valid]), -1)
        pairs = np.concatenate((2 * pairs, 2 * pairs + 1, np.stack((2 * np.arange(n), 2 * np.arange(n) + 1), -1)))
        self._set_graph(
            torch.as_tensor(pairs, device=device),
            2 * n,
            group.repeat_interleave(2),
            free.repeat_interleave(2),
            world.repeat_interleave(2),
        )
        self.kinematic = ~free  # the SolverNeIO interface masks body rows, not vector nodes
        mass = wp.to_torch(model.body_mass)
        inertia = wp.to_torch(model.body_inertia)
        # Radius of gyration: torque/length and length*angle preserve virtual work.
        self.length = (inertia.diagonal(dim1=-2, dim2=-1).sum(-1) / mass.clamp_min(1e-20)).sqrt().clamp_min(1e-4)

    def initialize(self, data):
        if self.kind == "particle":
            super().initialize(data)

    def _network(self, features):
        graph = self.graph
        direction, scale = self.network(
            features,
            graph.source,
            graph.target,
            graph.inv_degree,
            graph.group,
            graph.inv_group_size,
            self.log_scale.clone(),
        )
        # Bound the recurrent scale during training and inference alike. Longer
        # sweep budgets must not exponentiate an unbounded controller state.
        return direction, scale.clamp(np.log(1e-10), np.log(1e-2))

    def iterate(self, data, clear_history):
        with torch.set_grad_enabled(self.learn):
            if self.kind == "particle":
                result = super().iterate(data, clear_history)
                self.before_displacement = wp.to_torch(self.solver.particle_displacements).clone().detach()
            else:
                if clear_history:
                    self._apply_history_option()
                self.rotation = data.body_q[:, 3:].clone().detach()
                gradient = -torch.stack(
                    (
                        quat_rotate_inv(self.rotation, data.body_forces),
                        quat_rotate_inv(self.rotation, data.body_torques) / self.length[:, None],
                    ),
                    1,
                )
                gradient = gradient.flatten(0, 1).masked_fill(~self.graph.free[:, None], 0).detach()
                rms = self._gradient_rms(gradient)
                features = history_features(
                    gradient, self.last_gradient, self.last_update, None, rms, self.log_scale.exp()[self.graph.group]
                )
                direction, self._new_log_scale = self._network(features)
                local = (self._new_log_scale.exp()[self.graph.group] * direction).masked_fill(
                    ~self.graph.free[:, None], 0
                )
                local = local.reshape(-1, 2, 3)
                result = torch.cat(
                    (
                        quat_rotate(self.rotation, local[:, 0]),
                        quat_rotate(self.rotation, local[:, 1] / self.length[:, None]),
                    ),
                    -1,
                )
                self._gradient = gradient
                self._proposed_vectors = local.flatten(0, 1)
            # A solved force balance is a fixed point, including isolated rigid
            # bodies before contact. Network biases must not move a zero residual.
            active = self._gradient_rms(self._gradient) > 1e-7
            result = result * (active if self.kind == "particle" else active[::2])
            self.proposal = result
            return result

    def commit(self, applied):
        with torch.no_grad():
            if self.kind == "particle":
                super().commit(applied)
            else:
                local = torch.stack(
                    (
                        quat_rotate_inv(self.rotation, applied[:, :3]),
                        quat_rotate_inv(self.rotation, applied[:, 3:]) * self.length[:, None],
                    ),
                    1,
                )
                self._store_history(self._gradient, local.flatten(0, 1), self._proposed_vectors)


class StudySolver(SolverNeIO):
    """Retain Newton's step lifecycle; train only the update graph at each iteration."""

    def __init__(self, model, network, *, learn=False, history="clear", **kwargs):
        if kwargs.get("rigid_soft_enable_dat"):
            raise ValueError("The training bridge does not support rigid-soft DAT")
        self.learn = learn
        self.failed_updates = 0
        self.residuals = []
        self.losses = []
        super().__init__(
            model,
            rigid_optimizer=SharedOptimizer(network, "rigid", learn, history),
            particle_optimizer=SharedOptimizer(network, "particle", learn, history),
            **kwargs,
        )

    def _solve_rigid_body_iteration(self, state_in, state_out, control, contacts, dt):
        self.step_arguments = state_in, state_out, control, contacts, dt
        super()._solve_rigid_body_iteration(*self.step_arguments)

    def _vector_gradients(self):
        gradients = []
        if self.particle_optimizer is not None:
            gradients.append(-wp.to_torch(self.particle_forces).clone())
        if self.rigid_optimizer is not None:
            length = self.rigid_optimizer.length[:, None]
            vectors = torch.stack((wp.to_torch(self.body_forces), wp.to_torch(self.body_torques) / length), 1)
            gradients.append(-vectors.flatten(0, 1).clone())
        return gradients

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        if not self.learn:
            return
        self._evaluate_forces(*self.step_arguments)
        optimizers = [o for o in (self.particle_optimizer, self.rigid_optimizer) if o is not None]
        final_gradients = self._vector_gradients()
        if not all(bool(torch.isfinite(g).all()) for g in final_gradients):
            self.failed_updates += 1
            return
        # L = mean_world log(1 + ||g_after||² / ||g_before||²). The reference
        # norm is fixed for this local update; the target is zero physics residual.
        vectors, losses = [], []
        for optimizer, gradient in zip(optimizers, final_gradients):
            graph = optimizer.graph
            initial = optimizer._gradient.detach().square().sum(-1)
            before = group_sum(initial, graph.group, graph.group_count).clamp_min(1e-12)
            after = group_sum(gradient.square().sum(-1), graph.group, graph.group_count)
            losses.append(torch.log1p(after / before).mean())
            vectors.append(2 * gradient / (before + after)[graph.group, None] / graph.group_count / len(optimizers))
        self.losses.append(float(torch.stack(losses).mean()))
        self.residuals.append(float(torch.stack([g.square().mean() for g in final_gradients]).sum().sqrt()))

        # For the energy-gradient oracle, dL/dx = H dL/dg. Central differences
        # supply that Hessian-vector product without differentiating collisions.
        # All poses, forces and contacts return to the actual post-update state.
        peak = max(float(v.abs().max()) for v in vectors)
        epsilon = 1e-5 / max(peak, 1e-20)
        particle, body = self.particle_optimizer, self.rigid_optimizer
        particle_q = wp.to_torch(state_in.particle_q).clone() if particle is not None else None
        body_q = wp.to_torch(state_in.body_q).clone() if body is not None else None
        if body is not None:
            com = body_q[:, :3] + quat_rotate(body_q[:, 3:], wp.to_torch(self.model.body_com))
            direction = vectors[-1].reshape(-1, 2, 3)
        probes = []
        for sign in (1, -1):
            if particle is not None:
                wp.to_torch(state_in.particle_q).copy_(particle_q + sign * epsilon * vectors[0])
            if body is not None:
                rotation = quaternion_product(
                    rotation_quaternion(sign * epsilon * direction[:, 1] / body.length[:, None]), body_q[:, 3:]
                )
                position = (
                    com + sign * epsilon * direction[:, 0] - quat_rotate(rotation, wp.to_torch(self.model.body_com))
                )
                wp.to_torch(state_in.body_q).copy_(torch.cat((position, rotation), -1))
            self._evaluate_forces(*self.step_arguments)
            probes.append(self._vector_gradients())
        if particle is not None:
            wp.to_torch(state_in.particle_q).copy_(particle_q)
            wp.to_torch(self.particle_forces).copy_(-final_gradients[0])
        if body is not None:
            wp.to_torch(state_in.body_q).copy_(body_q)
            g = final_gradients[-1].reshape(-1, 2, 3)
            wp.to_torch(self.body_forces).copy_(-g[:, 0])
            wp.to_torch(self.body_torques).copy_(-g[:, 1] * body.length[:, None])
        derivatives = [(plus - minus) / (2 * epsilon) for plus, minus in zip(*probes)]
        outputs, gradients = [], []
        with torch.enable_grad():
            if particle is not None:
                candidate = (particle.before_displacement + particle.proposal) * wp.to_torch(self.truncation_ts)[
                    :, None
                ]
                if self.particle_enable_self_contact:
                    bound = self._self_contact_query_radius * self.dat_conservative_bound_relaxation * 0.5
                    candidate = candidate * (bound / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp(max=1)
                outputs.append(candidate)
                gradients.append(derivatives[0])
            if body is not None:
                q = quaternion_product(rotation_quaternion(body.proposal[:, 3:]), body.rotation)
                derivative = derivatives[-1].reshape(-1, 2, 3)
                outputs.extend((body.proposal[:, :3], q))
                gradients.extend(
                    (derivative[:, 0], quaternion_energy_gradient(q.detach(), derivative[:, 1] * body.length[:, None]))
                )
            valid = all(bool(torch.isfinite(g).all()) for g in gradients)
            valid &= all(bool(torch.isfinite(x).all()) for x in outputs)
            if valid:
                torch.autograd.backward(outputs, [g.detach() / self.iterations for g in gradients])
            else:
                self.failed_updates += 1
