# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""One shared vector network and a first-order training bridge to Newton forces."""

import numpy as np
import torch
import warp as wp
from experiments.neio import SolverNeIO
from experiments.neio.deformable import NeIODeformable
from experiments.neio.network import NeIONetwork, history_features
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

    def __init__(self, network, kind, learn=False, history="carry"):
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
        return self.network(
            features,
            graph.source,
            graph.target,
            graph.inv_degree,
            graph.group,
            graph.inv_group_size,
            self.log_scale.clone(),
        )

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
            self.proposal = result
            # One equal-weight contribution per world; preserve units of the objective.
            graph = self.graph
            self.energy_scale = self._gradient_rms(self._gradient).clamp_min(1e-3) * self.network.initial_scale
            self.energy_scale *= graph.free_count[graph.group, None].clamp_min(1) * graph.group_count
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

    def __init__(self, model, network, *, learn=False, history="carry", **kwargs):
        if kwargs.get("rigid_soft_enable_dat"):
            raise ValueError("The training bridge does not support rigid-soft DAT")
        self.learn = learn
        self.failed_updates = 0
        self.residuals = []
        super().__init__(
            model,
            rigid_optimizer=SharedOptimizer(network, "rigid", learn, history),
            particle_optimizer=SharedOptimizer(network, "particle", learn, history),
            **kwargs,
        )

    def _solve_rigid_body_iteration(self, state_in, state_out, control, contacts, dt):
        self.step_arguments = state_in, state_out, control, contacts, dt
        super()._solve_rigid_body_iteration(*self.step_arguments)

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        if not self.learn:
            return
        self._evaluate_forces(*self.step_arguments)
        outputs, gradients = [], []
        with torch.enable_grad():
            particle = self.particle_optimizer
            if particle is not None:
                # Differentiate radial clipping exactly, holding the collision-derived
                # truncation coefficients fixed (the declared first-order surrogate).
                candidate = (particle.before_displacement + particle.proposal) * wp.to_torch(self.truncation_ts)[
                    :, None
                ]
                if self.particle_enable_self_contact:
                    bound = self._self_contact_query_radius * self.dat_conservative_bound_relaxation * 0.5
                    candidate = candidate * (bound / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp(max=1)
                outputs.append(candidate)
                gradients.append(-wp.to_torch(self.particle_forces).clone() / particle.energy_scale)
            body = self.rigid_optimizer
            if body is not None:
                q = quaternion_product(rotation_quaternion(body.proposal[:, 3:]), body.rotation)
                scale = body.energy_scale.reshape(-1, 2, 1)[:, 0]
                outputs.extend((body.proposal[:, :3], q))
                gradients.extend(
                    (
                        -wp.to_torch(self.body_forces).clone() / scale,
                        quaternion_energy_gradient(q.detach(), -wp.to_torch(self.body_torques).clone()) / scale,
                    )
                )
            valid = all(bool(torch.isfinite(g).all()) for g in gradients)
            valid &= all(bool(torch.isfinite(x).all()) for x in outputs)
            if valid:
                torch.autograd.backward(
                    outputs,
                    [g.detach() / (self.iterations * ((particle is not None) + (body is not None))) for g in gradients],
                )
            else:
                self.failed_updates += 1
        self.residuals.append(float(torch.stack([g.detach().square().mean() for g in gradients]).sum().sqrt()))
