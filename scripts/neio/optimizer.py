# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared vector updates trained against the frozen-contact physics residual."""

import numpy as np
import torch
import warp as wp
from experiments.neio import SolverNeIO, kernels
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


def new_network(device, width=96):
    return NeIONetwork(12, width, 3, 1, initial_scale=1e-4).to(device)


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
        graph, model, network = self.graph, self.solver.model, self.network
        dt, unit = self.solver._data.dt, self.solver.length_unit_m
        if self.kind == "particle":
            gradient = wp.to_torch(self.solver.particle_forces)
            mass = wp.to_torch(model.particle_mass)
        else:
            gradient = torch.stack(
                (wp.to_torch(self.solver.body_forces), wp.to_torch(self.solver.body_torques) / self.length[:, None]), 1
            ).flatten(0, 1)
            mass = wp.to_torch(model.body_mass).repeat_interleave(2)
        gradient = gradient.masked_fill(~graph.free[:, None], 0)
        # Normalized direction alone erases the force and inertial scales needed
        # to distinguish stiff cables from cloth. Context uses physical units;
        # no scene identity, target solution, or separate task network is supplied.
        rms = self._gradient_rms(gradient).detach() * unit
        context = torch.cat(
            (
                torch.log10(rms.clamp_min(1e-20)) / 10,
                torch.log10((mass / dt**2).clamp_min(1e-20))[:, None] / 10,
                self.log_scale[graph.group] / 10,
            ),
            -1,
        )
        hidden = network.trunk(torch.cat((features, context), -1), graph.source, graph.target, graph.inv_degree)
        direction = torch.tanh(network.direction_head(hidden))
        pooled = group_sum(hidden, graph.group, graph.group_count) * graph.inv_group_size
        # Allow a large scale correction in the first sweep, with the same final
        # physical bound at every training and inference sweep budget.
        scale = self.log_scale.clone() + network.scale_head(pooled)
        return direction, scale.clamp(np.log(1e-10), np.log(1e-2))

    def iterate(self, data, clear_history):
        with torch.set_grad_enabled(self.learn):
            unit = self.solver.length_unit_m
            if self.kind == "particle":
                # The learned scale and stored updates are in meters, including
                # native examples whose simulation coordinates are centimeters.
                result = super().iterate(data, clear_history) / unit
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
                        quat_rotate(self.rotation, local[:, 0] / unit),
                        quat_rotate(self.rotation, local[:, 1] / (self.length[:, None] * unit)),
                    ),
                    -1,
                )
                self._gradient = gradient
                self._proposed_vectors = local.flatten(0, 1)
            # A solved force balance is a fixed point, including isolated rigid
            # bodies before contact. Network biases must not move a zero residual.
            self.active = self._gradient_rms(self._gradient) > 1e-7
            result = result * (self.active if self.kind == "particle" else self.active[::2])
            self.proposal = result
            return result

    def commit(self, applied):
        with torch.no_grad():
            unit = self.solver.length_unit_m
            if self.kind == "particle":
                super().commit(applied * unit)
            else:
                local = torch.stack(
                    (
                        quat_rotate_inv(self.rotation, applied[:, :3]) * unit,
                        quat_rotate_inv(self.rotation, applied[:, 3:]) * self.length[:, None] * unit,
                    ),
                    1,
                )
                self._store_history(self._gradient, local.flatten(0, 1), self._proposed_vectors)


class StudySolver(SolverNeIO):
    """Retain Newton's step lifecycle; train only the update graph at each iteration."""

    def __init__(self, model, network, *, learn=False, history="clear", length_unit_m=1.0, **kwargs):
        if kwargs.get("rigid_soft_enable_dat"):
            raise ValueError("The training bridge does not support rigid-soft DAT")
        if learn:
            wp.set_module_options({"enable_backward": True}, module=kernels)
        self.learn = learn
        self.length_unit_m = length_unit_m
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
        # Differentiate the actual frozen-contact force map. Its Jacobian can be
        # nonsymmetric (normal-load-dependent friction), so an H v approximation
        # cannot replace the required J^T v residual derivative.
        particle, body = self.particle_optimizer, self.rigid_optimizer
        arrays = []
        if particle is not None:
            arrays.extend((state_in.particle_q, self.particle_forces))
        if body is not None:
            arrays.extend((state_in.body_q, self.body_forces, self.body_torques))
        for array in arrays:
            array.requires_grad = True
            array.grad.zero_()
        try:
            with wp.Tape() as tape:
                self._evaluate_forces(*self.step_arguments)
            optimizers = [o for o in (particle, body) if o is not None]
            final_gradients = self._vector_gradients()
            if not all(bool(torch.isfinite(g).all()) for g in final_gradients):
                self.failed_updates += 1
                return
            # The pre-update norm is a fixed local reference; the target is zero
            # residual. Worlds and particle/body objectives receive equal weight.
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
            seeds = {}
            if particle is not None:
                seeds[self.particle_forces] = wp.from_torch(-vectors[0].contiguous(), dtype=wp.vec3)
            if body is not None:
                vector = vectors[-1].reshape(-1, 2, 3)
                seeds[self.body_forces] = wp.from_torch(-vector[:, 0].contiguous(), dtype=wp.vec3)
                seeds[self.body_torques] = wp.from_torch(
                    (-vector[:, 1] / body.length[:, None]).contiguous(), dtype=wp.vec3
                )
            tape.backward(grads=seeds)
            outputs, gradients = [], []
            with torch.enable_grad():
                if particle is not None:
                    candidate = (particle.before_displacement + particle.proposal) * wp.to_torch(self.truncation_ts)[
                        :, None
                    ]
                    if self.particle_enable_self_contact:
                        bound = self._self_contact_query_radius * self.dat_conservative_bound_relaxation * 0.5
                        candidate = candidate * (bound / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp(
                            max=1
                        )
                    outputs.append(candidate)
                    derivative = wp.to_torch(state_in.particle_q.grad).clone().detach()
                    gradients.append(derivative.masked_fill(~particle.active | ~particle.graph.free[:, None], 0))
                if body is not None:
                    q = quaternion_product(rotation_quaternion(body.proposal[:, 3:]), body.rotation)
                    # Warp differentiates body-origin poses; the learned translation
                    # moves the COM. Pull back through that offset and the rotation.
                    position = body.proposal[:, :3] - quat_rotate(q, wp.to_torch(self.model.body_com))
                    outputs.append(torch.cat((position, q), -1))
                    derivative = wp.to_torch(state_in.body_q.grad).clone().detach()
                    # These rows have a constant zero proposal. Their exact
                    # weight derivative is zero, even if an axis-angle adjoint
                    # overflows at an effectively zero inertial rotation.
                    inactive = ~body.active[::2] | ~body.graph.free[::2, None]
                    gradients.append(derivative.masked_fill(inactive, 0))
                valid = all(bool(torch.isfinite(g).all()) for g in gradients)
                valid &= all(bool(torch.isfinite(x).all()) for x in outputs)
                if valid:
                    torch.autograd.backward(outputs, [g / self.iterations for g in gradients])
                else:
                    self.failed_updates += 1
        finally:
            # The following sweep/timestep must not backpropagate through mutable
            # simulator buffers or earlier history. Only this proposal is learned.
            for array in arrays:
                array.requires_grad = False
