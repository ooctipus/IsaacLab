# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-wrench and generalized-effort diagnostics for kinematic trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import quat_apply

from .newton_kinematics import NewtonKinematics


@dataclass(frozen=True, slots=True)
class ContactFeasibilityResult:
    """GPU-resident contact-wrench feasibility diagnostics.

    No field is an acceptance decision. Contact transitions are reported because
    continuous inverse dynamics is not an impact model; callers must calibrate
    these diagnostics against downstream tracking before defining a gate.

    Attributes:
        contact_wrench_world: External support wrenches at body centers of mass
            [N, N·m], shape ``[frame_count, support_count, 6]``.
        generalized_effort: Required generalized forces [N or N·m, depending
            on joint type], shape ``[frame_count, dof_count]``.
        generalized_effort_margin: Signed distance to the nearest effort bound
            [N or N·m, depending on joint type], shape
            ``[frame_count, dof_count]``. Floating-base entries are ``NaN``.
        effort_margin_ratio: Minimum actuated effort margin divided by the
            corresponding half range, shape ``[frame_count]``.
        balance_force_residual_n: Floating-base force residual [N], shape
            ``[frame_count]``.
        balance_torque_residual_nm: Floating-base torque residual [N·m], shape
            ``[frame_count]``.
        normal_force_n: Support normal forces [N], shape
            ``[frame_count, support_count]``. Inactive entries are ``NaN``.
        friction_margin_n: Coulomb margins ``mu*f_n - ||f_t||`` [N], shape
            ``[frame_count, support_count]``. Inactive entries are ``NaN``.
        contact_transition: Whether source contact activity changed from the
            preceding frame in the same segment, shape ``[frame_count]``.
        segment_balance_force_residual_n: Maximum non-transition force residual
            [N] per segment.
        segment_balance_torque_residual_nm: Maximum non-transition torque residual
            [N·m] per segment.
        segment_effort_margin_ratio: Minimum non-transition actuated effort margin
            ratio per segment.
        segment_normal_force_min_n: Minimum active non-transition normal force [N]
            per segment.
        segment_friction_margin_n: Minimum active non-transition Coulomb margin [N]
            per segment.
        segment_contact_transition_count: Contact changes per segment.
    """

    contact_wrench_world: torch.Tensor
    generalized_effort: torch.Tensor
    generalized_effort_margin: torch.Tensor
    effort_margin_ratio: torch.Tensor
    balance_force_residual_n: torch.Tensor
    balance_torque_residual_nm: torch.Tensor
    normal_force_n: torch.Tensor
    friction_margin_n: torch.Tensor
    contact_transition: torch.Tensor
    segment_balance_force_residual_n: torch.Tensor
    segment_balance_torque_residual_nm: torch.Tensor
    segment_effort_margin_ratio: torch.Tensor
    segment_normal_force_min_n: torch.Tensor
    segment_friction_margin_n: torch.Tensor
    segment_contact_transition_count: torch.Tensor


class ContactFeasibilityWorkspace:
    """Reusable capacity-sized workspace for batched contact diagnostics.

    The workspace fixes one Newton model, maximum frame count, and ordered
    support-body mapping. :meth:`evaluate` accepts any active prefix up to that
    capacity, pads only the private tail, and reuses the inverse-dynamics,
    body-state, generalized-force, wrench, and contact-map allocations across
    complete-segment batches.

    Args:
        kinematics: Fixed Newton mechanics and canonical topology.
        frame_capacity: Maximum frame count evaluated per batch.
        support_body_indices: Body receiving each ordered support force.
    """

    def __init__(
        self,
        kinematics: NewtonKinematics,
        frame_capacity: int,
        support_body_indices: tuple[int, ...],
    ) -> None:
        model = kinematics.model
        if frame_capacity < 1 or not support_body_indices:
            raise ValueError("Contact-feasibility workspace requires positive frame and support capacities.")
        if any(body < 0 or body >= model.body_count for body in support_body_indices):
            raise ValueError("Contact-feasibility support body index is out of range.")
        topology = kinematics.topology
        free_types = (int(newton.JointType.FREE), int(newton.JointType.DISTANCE))
        descendant_free = np.isin(topology.joint_type, free_types) & (topology.joint_parent >= 0)
        if np.any(descendant_free):
            raise NotImplementedError("Contact feasibility does not support descendant FREE or DISTANCE joints.")
        root_joints = np.flatnonzero(topology.joint_parent < 0)
        if root_joints.shape != (1,):
            raise ValueError("Contact feasibility requires one rooted articulation.")
        root_joint = int(root_joints[0])
        root_begin = int(topology.joint_qd_start[root_joint])
        root_end = int(topology.joint_qd_start[root_joint + 1])
        root_type = int(topology.joint_type[root_joint])
        if root_type in free_types:
            if (
                root_begin != 0
                or root_end != 6
                or tuple(int(value) for value in topology.joint_dof_dim[root_joint]) != (3, 3)
            ):
                raise ValueError("A floating root must own the leading three linear and three angular DOFs.")
            root_dof_count = 6
        else:
            if root_begin != 0 or root_end != 0:
                raise NotImplementedError("A non-floating root must be fixed for contact-feasibility evaluation.")
            root_dof_count = 0
        effort_lower = topology.joint_effort_lower[root_dof_count:]
        effort_upper = topology.joint_effort_upper[root_dof_count:]
        if (
            np.any(~np.isfinite(effort_lower))
            or np.any(~np.isfinite(effort_upper))
            or np.any(effort_lower >= effort_upper)
        ):
            raise NotImplementedError("Every non-root degree of freedom must declare a finite nonempty actuator range.")
        self.kinematics = kinematics
        self.frame_capacity = frame_capacity
        self.support_body_indices = support_body_indices
        self.support_count = len(support_body_indices)
        self.root_dof_count = root_dof_count
        device = wp.to_torch(model.joint_q).device
        self.support_body_index = torch.tensor(support_body_indices, dtype=torch.int64, device=device)
        self.effort_lower = torch.tensor(effort_lower, dtype=torch.float32, device=device)
        self.effort_upper = torch.tensor(effort_upper, dtype=torch.float32, device=device)
        self.effort_scale = (0.5 * (self.effort_upper - self.effort_lower)).clamp_min_(1.0e-6)
        self.body_mass = wp.to_torch(model.body_mass)
        self.body_com_local = wp.to_torch(model.body_com).view(1, model.body_count, 3)
        self.total_mass = self.body_mass.sum()
        self.gravity = torch.tensor(topology.gravity, dtype=torch.float32, device=device)
        self.weight_n = (self.total_mass * torch.linalg.vector_norm(self.gravity)).clamp_min_(1.0)
        self.joint_q = torch.empty((frame_capacity, model.joint_coord_count), dtype=torch.float32, device=device)
        self.joint_qd = torch.empty((frame_capacity, model.joint_dof_count), dtype=torch.float32, device=device)
        self.joint_qdd = torch.empty_like(self.joint_qd)
        self.body_q = torch.empty((frame_capacity, model.body_count, 7), dtype=torch.float32, device=device)
        self.body_qd = torch.empty((frame_capacity, model.body_count, 6), dtype=torch.float32, device=device)
        self.body_com_world = torch.empty((frame_capacity, model.body_count, 3), dtype=torch.float32, device=device)
        self.support_position_world = torch.empty(
            (frame_capacity, self.support_count, 3), dtype=torch.float32, device=device
        )
        self.generalized_free = torch.empty_like(self.joint_qd)
        self.generalized_trial = torch.empty_like(self.joint_qd)
        self.body_f = torch.zeros((frame_capacity, model.body_count, 6), dtype=torch.float32, device=device)
        self.contact_map = torch.empty(
            (frame_capacity, model.joint_dof_count, self.support_count, 3),
            dtype=torch.float32,
            device=device,
        )
        self.support_active = torch.empty((frame_capacity, self.support_count), dtype=torch.bool, device=device)
        self.support_normal_world = torch.empty(
            (frame_capacity, self.support_count, 3), dtype=torch.float32, device=device
        )
        self.friction_coefficient = torch.empty(
            (frame_capacity, self.support_count), dtype=torch.float32, device=device
        )
        self.axes = torch.eye(3, dtype=torch.float32, device=device)
        self.inverse = newton.dynamics.DynamicsInverse(model, frame_capacity)

    @staticmethod
    def estimate_memory(kinematics: NewtonKinematics, frame_capacity: int, support_count: int) -> int:
        """Estimate a conservative peak additional workspace [byte].

        Args:
            kinematics: Fixed Newton mechanics.
            frame_capacity: Candidate frame capacity.
            support_count: Ordered contact-point count.

        Returns:
            Persistent buffers, returned diagnostics, and peak vectorized
            projected-gradient temporaries [byte].
        """
        if frame_capacity < 1 or support_count < 1:
            raise ValueError("Contact-feasibility memory dimensions must be positive.")
        model = kinematics.model
        persistent_scalars_per_frame = (
            model.joint_coord_count
            + 4 * model.joint_dof_count
            + 22 * model.body_count
            + 3 * model.joint_dof_count * support_count
            + 7 * support_count
        )
        temporary_scalars_per_frame = 8 * model.joint_dof_count + 12 * model.body_count + 32 * support_count + 64
        inverse_bytes = newton.dynamics.DynamicsInverse.estimate_memory(model, frame_capacity)
        bool_bytes = frame_capacity * support_count
        fixed_bytes = 9 * 4 + 8 * support_count + 12 * model.joint_dof_count + 16 * model.body_count + 32
        return (
            frame_capacity * (persistent_scalars_per_frame + temporary_scalars_per_frame) * 4
            + inverse_bytes
            + bool_bytes
            + fixed_bytes
        )

    def evaluate(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        joint_qdd: torch.Tensor,
        *,
        support_point_body_m: torch.Tensor,
        support_active: torch.Tensor,
        support_normal_world: torch.Tensor,
        friction_coefficient: torch.Tensor,
        segment_offsets: tuple[int, ...],
        iterations: int,
        effort_weight: float,
        force_regularization: float,
    ) -> ContactFeasibilityResult:
        """Evaluate one complete-segment batch in the reusable workspace."""
        return _contact_feasibility_evaluate(
            self,
            joint_q,
            joint_qd,
            joint_qdd,
            support_point_body_m=support_point_body_m,
            support_active=support_active,
            support_normal_world=support_normal_world,
            friction_coefficient=friction_coefficient,
            segment_offsets=segment_offsets,
            iterations=iterations,
            effort_weight=effort_weight,
            force_regularization=force_regularization,
        )


def _contact_force_project(
    force_world_n: torch.Tensor,
    normal_world: torch.Tensor,
    friction_coefficient: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Project point forces onto unilateral circular Coulomb cones."""
    normal_force = (force_world_n * normal_world).sum(dim=-1)
    tangent = force_world_n - normal_force[..., None] * normal_world
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    inside = (normal_force >= 0.0) & (tangent_norm <= friction_coefficient * normal_force)
    polar = normal_force <= -friction_coefficient * tangent_norm
    boundary_normal = (normal_force + friction_coefficient * tangent_norm).div(1.0 + friction_coefficient.square())
    boundary_tangent_norm = friction_coefficient * boundary_normal
    boundary = boundary_normal[..., None] * normal_world + boundary_tangent_norm[..., None] * tangent.div(
        tangent_norm.clamp_min(1.0e-12)[..., None]
    )
    projected = torch.where(inside[..., None], force_world_n, torch.where(polar[..., None], 0.0, boundary))
    return torch.where(active[..., None], projected, 0.0)


def contact_feasibility_evaluate(
    kinematics: NewtonKinematics,
    joint_q: torch.Tensor,
    joint_qd: torch.Tensor,
    joint_qdd: torch.Tensor,
    *,
    support_body_indices: tuple[int, ...],
    support_point_body_m: torch.Tensor,
    support_active: torch.Tensor,
    support_normal_world: torch.Tensor,
    friction_coefficient: torch.Tensor,
    segment_offsets: tuple[int, ...],
    iterations: int,
    effort_weight: float,
    force_regularization: float,
) -> ContactFeasibilityResult:
    """Evaluate one exact-size batch with a temporary reusable workspace.

    Call :class:`ContactFeasibilityWorkspace` directly when processing
    multiple memory-planned complete-segment batches.

    Args:
        kinematics: Fixed Newton mechanics and canonical topology.
        joint_q: Generalized positions [m or rad, depending on joint type].
        joint_qd: Generalized velocities [m/s or rad/s, depending on joint type].
        joint_qdd: Generalized accelerations [m/s² or rad/s², depending on joint type].
        support_body_indices: Body receiving each ordered support force.
        support_point_body_m: Body-local support points [m].
        support_active: Source-inferred support activity.
        support_normal_world: Unit world support normals.
        friction_coefficient: Coulomb coefficient per frame and support.
        segment_offsets: Strictly increasing prefix frame offsets.
        iterations: Projected-gradient iteration count.
        effort_weight: Relative normalized actuator-bound violation weight.
        force_regularization: Dimensionless contact-force regularization.

    Returns:
        Frame- and segment-level physical diagnostics without an acceptance gate.
    """
    workspace = ContactFeasibilityWorkspace(kinematics, joint_q.shape[0], support_body_indices)
    return workspace.evaluate(
        joint_q,
        joint_qd,
        joint_qdd,
        support_point_body_m=support_point_body_m,
        support_active=support_active,
        support_normal_world=support_normal_world,
        friction_coefficient=friction_coefficient,
        segment_offsets=segment_offsets,
        iterations=iterations,
        effort_weight=effort_weight,
        force_regularization=force_regularization,
    )


def _contact_feasibility_evaluate(
    workspace: ContactFeasibilityWorkspace,
    joint_q: torch.Tensor,
    joint_qd: torch.Tensor,
    joint_qdd: torch.Tensor,
    *,
    support_point_body_m: torch.Tensor,
    support_active: torch.Tensor,
    support_normal_world: torch.Tensor,
    friction_coefficient: torch.Tensor,
    segment_offsets: tuple[int, ...],
    iterations: int,
    effort_weight: float,
    force_regularization: float,
) -> ContactFeasibilityResult:
    """Evaluate point-contact wrench and effort feasibility for a fixed trajectory.

    The trajectory, contact schedule, and contact points remain fixed. The only
    optimized variables are point-contact forces, projected after every step
    onto each declared Coulomb cone. Newton evaluates
    ``M(q) qdd + h(q, qd) - J(q)^T f`` through its public batched inverse-
    dynamics API; no private Jacobian or simulator state is used.

    The controlled-domain actuation contract is explicit: a root ``FREE`` or
    ``DISTANCE`` joint owns the six unactuated floating-base coordinates, and
    every non-root degree of freedom is actuated with a finite nonempty effort
    range in :attr:`NewtonKinematics.topology`. Descendant free joints and
    passive non-root mechanisms are rejected rather than guessed.

    Args:
        workspace: Fixed mechanics, support mapping, and reusable storage.
        joint_q: Generalized positions [m or rad, depending on joint type],
            shape ``[frame_count, coordinate_count]``.
        joint_qd: Generalized velocities [m/s or rad/s, depending on joint type],
            shape ``[frame_count, dof_count]``.
        joint_qdd: Generalized accelerations [m/s² or rad/s², depending on
            joint type], shape ``[frame_count, dof_count]``.
        support_point_body_m: Robot contact points [m] in each support body's
            local frame, shape ``[support_count, 3]``.
        support_active: Source-inferred support activity, shape
            ``[frame_count, support_count]``.
        support_normal_world: Unit support normals in world coordinates, shape
            ``[frame_count, support_count, 3]``.
        friction_coefficient: Coulomb coefficient per frame and support, shape
            ``[frame_count, support_count]``.
        segment_offsets: Strictly increasing prefix frame offsets.
        iterations: Projected-gradient iteration count.
        effort_weight: Relative weight of normalized actuator-bound violations.
        force_regularization: Dimensionless contact-force regularization.

    Returns:
        Frame- and segment-level physical diagnostics without an acceptance gate.
    """
    kinematics = workspace.kinematics
    model = kinematics.model
    topology = kinematics.topology
    active_frame_count = joint_q.shape[0] if joint_q.ndim else -1
    frame_count = workspace.frame_capacity
    support_body_indices = workspace.support_body_indices
    support_count = workspace.support_count
    expected = (
        (joint_q, torch.float32, (active_frame_count, model.joint_coord_count), "joint_q"),
        (joint_qd, torch.float32, (active_frame_count, model.joint_dof_count), "joint_qd"),
        (joint_qdd, torch.float32, (active_frame_count, model.joint_dof_count), "joint_qdd"),
        (
            support_point_body_m,
            torch.float32,
            (support_count, 3),
            "support_point_body_m",
        ),
        (support_active, torch.bool, (active_frame_count, support_count), "support_active"),
        (
            support_normal_world,
            torch.float32,
            (active_frame_count, support_count, 3),
            "support_normal_world",
        ),
        (
            friction_coefficient,
            torch.float32,
            (active_frame_count, support_count),
            "friction_coefficient",
        ),
    )
    device = str(wp.device_from_torch(joint_q.device))
    for tensor, dtype, shape, name in expected:
        if tensor.dtype is not dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape}.")
        if tensor.device != joint_q.device:
            raise ValueError("Contact-feasibility tensors must share one device.")
    if device != str(model.device):
        raise ValueError(f"Contact-feasibility tensors must reside on {model.device}.")
    if (
        active_frame_count < 1
        or active_frame_count > frame_count
        or len(segment_offsets) < 2
        or segment_offsets[0] != 0
        or segment_offsets[-1] != active_frame_count
        or any(stop <= start for start, stop in zip(segment_offsets[:-1], segment_offsets[1:], strict=True))
    ):
        raise ValueError("Contact feasibility requires a valid nonempty complete-segment prefix.")
    if iterations < 1 or effort_weight < 0.0 or force_regularization < 0.0:
        raise ValueError("Solver iterations must be positive and objective weights nonnegative.")
    normal_norm = torch.linalg.vector_norm(support_normal_world, dim=-1)
    torch._assert_async(torch.all((normal_norm - 1.0).abs() <= 1.0e-4), "Support normals must be unit vectors.")
    torch._assert_async(
        torch.all(torch.isfinite(friction_coefficient) & (friction_coefficient >= 0.0)),
        "Friction coefficients must be finite and nonnegative.",
    )

    workspace.joint_q[:active_frame_count].copy_(joint_q)
    workspace.joint_qd[:active_frame_count].copy_(joint_qd)
    workspace.joint_qdd[:active_frame_count].copy_(joint_qdd)
    workspace.support_active[:active_frame_count].copy_(support_active)
    workspace.support_normal_world[:active_frame_count].copy_(support_normal_world)
    workspace.friction_coefficient[:active_frame_count].copy_(friction_coefficient)
    if active_frame_count < frame_count:
        workspace.joint_q[active_frame_count:].copy_(joint_q[-1])
        workspace.joint_qd[active_frame_count:].zero_()
        workspace.joint_qdd[active_frame_count:].zero_()
        workspace.support_active[active_frame_count:].zero_()
        workspace.support_normal_world[active_frame_count:].zero_()
        workspace.support_normal_world[active_frame_count:, :, 2] = 1.0
        workspace.friction_coefficient[active_frame_count:].zero_()
    joint_q = workspace.joint_q
    joint_qd = workspace.joint_qd
    joint_qdd = workspace.joint_qdd
    support_active = workspace.support_active
    support_normal_world = workspace.support_normal_world
    friction_coefficient = workspace.friction_coefficient

    root_dof_count = workspace.root_dof_count

    body_q = workspace.body_q
    body_qd = workspace.body_qd
    joint_q_wp = wp.from_torch(joint_q, dtype=wp.float32)
    joint_qd_wp = wp.from_torch(joint_qd, dtype=wp.float32)
    joint_qdd_wp = wp.from_torch(joint_qdd, dtype=wp.float32)
    newton.eval_fk_batched(
        model,
        joint_q_wp,
        joint_qd_wp,
        wp.from_torch(body_q, dtype=wp.transformf),
        wp.from_torch(body_qd, dtype=wp.spatial_vectorf),
    )
    body_com_local = workspace.body_com_local.expand(frame_count, -1, -1)
    body_com_world = workspace.body_com_world
    body_com_world.copy_(
        body_q[..., :3]
        + quat_apply(body_q[..., 3:7].reshape(-1, 4), body_com_local.reshape(-1, 3)).view(
            frame_count, model.body_count, 3
        )
    )

    support_pose = body_q.index_select(1, workspace.support_body_index)
    support_point_body = support_point_body_m[None].expand(frame_count, -1, -1)
    support_position_world_m = workspace.support_position_world
    support_position_world_m.copy_(
        support_pose[..., :3]
        + quat_apply(support_pose[..., 3:7].reshape(-1, 4), support_point_body.reshape(-1, 3)).view(
            frame_count, support_count, 3
        )
    )

    inverse = workspace.inverse
    generalized_free = workspace.generalized_free
    generalized_trial = workspace.generalized_trial
    generalized_free_wp = wp.from_torch(generalized_free, dtype=wp.float32)
    generalized_trial_wp = wp.from_torch(generalized_trial, dtype=wp.float32)
    inverse.compute(joint_q_wp, joint_qd_wp, joint_qdd_wp, generalized_free_wp)

    body_f = workspace.body_f
    body_f_wp = wp.from_torch(body_f, dtype=wp.spatial_vectorf)
    contact_map = workspace.contact_map
    axes = workspace.axes
    for support_slot, body_index in enumerate(support_body_indices):
        lever = support_position_world_m[:, support_slot] - body_com_world[:, body_index]
        for axis_index in range(3):
            body_f.zero_()
            body_f[:, body_index, axis_index] = 1.0
            body_f[:, body_index, 3:].copy_(torch.cross(lever, axes[axis_index].expand_as(lever), dim=-1))
            inverse.compute(joint_q_wp, joint_qd_wp, joint_qdd_wp, generalized_trial_wp, body_f_wp)
            contact_map[:, :, support_slot, axis_index].copy_(generalized_free - generalized_trial)
    contact_map.mul_(support_active[:, None, :, None])
    frame_count = active_frame_count
    joint_q = joint_q[:frame_count]
    body_com_world = body_com_world[:frame_count]
    support_position_world_m = support_position_world_m[:frame_count]
    generalized_free = generalized_free[:frame_count]
    contact_map = contact_map[:frame_count]
    support_active = support_active[:frame_count]
    support_normal_world = support_normal_world[:frame_count]
    friction_coefficient = friction_coefficient[:frame_count]

    body_mass = workspace.body_mass
    total_mass = workspace.total_mass
    gravity = workspace.gravity
    weight_n = workspace.weight_n
    system_com = (body_com_world * body_mass[None, :, None]).sum(dim=1) / total_mass.clamp_min(1.0e-8)
    characteristic_length_m = torch.sqrt(
        (body_mass[None, :] * (body_com_world - system_com[:, None]).square().sum(dim=-1)).sum(dim=1)
        / total_mass.clamp_min(1.0e-8)
    ).clamp_min(0.1)

    active_count = support_active.sum(dim=1, keepdim=True).clamp_min(1)
    gravity_compensation = (-total_mass * gravity).view(1, 1, 3)
    force_world_n = (
        support_normal_world
        * ((support_normal_world * gravity_compensation).sum(dim=-1).clamp_min(0.0) / active_count)[..., None]
    )
    force_world_n = _contact_force_project(force_world_n, support_normal_world, friction_coefficient, support_active)

    if root_dof_count:
        root_scale = torch.cat(
            (
                weight_n.expand(frame_count, 3),
                (weight_n * characteristic_length_m)[:, None].expand(-1, 3),
            ),
            dim=1,
        )
        root_map = contact_map[:, :root_dof_count]
        lipschitz = (root_map / root_scale[:, :, None, None]).square().sum(dim=(1, 2, 3))
    else:
        root_scale = joint_q.new_empty((frame_count, 0))
        root_map = contact_map[:, :0]
        lipschitz = joint_q.new_zeros(frame_count)
    if root_dof_count < topology.dof_count:
        effort_lower = workspace.effort_lower
        effort_upper = workspace.effort_upper
        effort_scale = workspace.effort_scale
        actuated_map = contact_map[:, root_dof_count:]
        lipschitz.add_(effort_weight * (actuated_map / effort_scale[None, :, None, None]).square().sum(dim=(1, 2, 3)))
    else:
        effort_lower = joint_q.new_empty(0)
        effort_upper = joint_q.new_empty(0)
        effort_scale = joint_q.new_empty(0)
        actuated_map = contact_map[:, :0]
    regularization_scale = force_regularization / weight_n.square()
    step_size = 0.95 / (lipschitz + regularization_scale).clamp_min(1.0e-12)

    for _ in range(iterations):
        required = generalized_free - torch.einsum("fjsd,fsd->fj", contact_map, force_world_n)
        gradient = regularization_scale * force_world_n
        if root_dof_count:
            root_gradient = required[:, :root_dof_count] / root_scale.square()
            gradient.sub_(torch.einsum("fjsd,fj->fsd", root_map, root_gradient))
        if root_dof_count < topology.dof_count and effort_weight:
            effort = required[:, root_dof_count:]
            effort_gradient = (effort - effort_lower).clamp_max(0.0)
            effort_gradient.add_((effort - effort_upper).clamp_min(0.0)).div_(effort_scale.square())
            gradient.sub_(effort_weight * torch.einsum("fjsd,fj->fsd", actuated_map, effort_gradient))
        force_world_n.sub_(step_size[:, None, None] * gradient)
        force_world_n = _contact_force_project(
            force_world_n, support_normal_world, friction_coefficient, support_active
        )

    generalized_effort = generalized_free - torch.einsum("fjsd,fsd->fj", contact_map, force_world_n)
    support_com = body_com_world.index_select(1, workspace.support_body_index)
    contact_torque_world_nm = torch.cross(support_position_world_m - support_com, force_world_n, dim=-1)
    contact_wrench_world = torch.cat((force_world_n, contact_torque_world_nm), dim=-1)

    if root_dof_count:
        root_effort = generalized_effort[:, :root_dof_count]
        balance_force_residual_n = torch.linalg.vector_norm(root_effort[:, :3], dim=-1)
        balance_torque_residual_nm = torch.linalg.vector_norm(root_effort[:, 3:], dim=-1)
    else:
        balance_force_residual_n = joint_q.new_zeros(frame_count)
        balance_torque_residual_nm = joint_q.new_zeros(frame_count)
    generalized_effort_margin = torch.full_like(generalized_effort, torch.nan)
    if root_dof_count < topology.dof_count:
        effort = generalized_effort[:, root_dof_count:]
        effort_margin = torch.minimum(effort - effort_lower, effort_upper - effort)
        generalized_effort_margin[:, root_dof_count:] = effort_margin
        effort_margin_ratio = (effort_margin / effort_scale).amin(dim=1)
    else:
        effort_margin_ratio = joint_q.new_full((frame_count,), torch.nan)

    normal_force_n = (force_world_n * support_normal_world).sum(dim=-1)
    tangent_force = force_world_n - normal_force_n[..., None] * support_normal_world
    friction_margin_n = friction_coefficient * normal_force_n - torch.linalg.vector_norm(tangent_force, dim=-1)
    normal_force_n = torch.where(support_active, normal_force_n, torch.nan)
    friction_margin_n = torch.where(support_active, friction_margin_n, torch.nan)

    contact_transition = torch.zeros(frame_count, dtype=torch.bool, device=joint_q.device)
    segment_transition_count: list[torch.Tensor] = []
    segment_force_residual: list[torch.Tensor] = []
    segment_torque_residual: list[torch.Tensor] = []
    segment_effort_margin: list[torch.Tensor] = []
    segment_normal_margin: list[torch.Tensor] = []
    segment_friction_margin: list[torch.Tensor] = []
    for start, stop in zip(segment_offsets[:-1], segment_offsets[1:], strict=True):
        if stop - start > 1:
            contact_transition[start + 1 : stop] = (
                support_active[start + 1 : stop] != support_active[start : stop - 1]
            ).any(dim=1)
        valid = ~contact_transition[start:stop]
        segment_transition_count.append(contact_transition[start:stop].sum())
        segment_force_residual.append(torch.where(valid, balance_force_residual_n[start:stop], -torch.inf).amax())
        segment_torque_residual.append(torch.where(valid, balance_torque_residual_nm[start:stop], -torch.inf).amax())
        segment_effort_margin.append(torch.where(valid, effort_margin_ratio[start:stop], torch.inf).amin())
        valid_support = valid[:, None] & support_active[start:stop]
        segment_normal_margin.append(torch.where(valid_support, normal_force_n[start:stop], torch.inf).amin())
        segment_friction_margin.append(torch.where(valid_support, friction_margin_n[start:stop], torch.inf).amin())

    segment_effort_margin_ratio = torch.stack(segment_effort_margin)
    segment_normal_force_min_n = torch.stack(segment_normal_margin)
    segment_friction_margin_n = torch.stack(segment_friction_margin)
    segment_effort_margin_ratio = torch.where(
        torch.isinf(segment_effort_margin_ratio), torch.nan, segment_effort_margin_ratio
    )
    segment_normal_force_min_n = torch.where(
        torch.isinf(segment_normal_force_min_n), torch.nan, segment_normal_force_min_n
    )
    segment_friction_margin_n = torch.where(
        torch.isinf(segment_friction_margin_n), torch.nan, segment_friction_margin_n
    )
    return ContactFeasibilityResult(
        contact_wrench_world=contact_wrench_world,
        generalized_effort=generalized_effort,
        generalized_effort_margin=generalized_effort_margin,
        effort_margin_ratio=effort_margin_ratio,
        balance_force_residual_n=balance_force_residual_n,
        balance_torque_residual_nm=balance_torque_residual_nm,
        normal_force_n=normal_force_n,
        friction_margin_n=friction_margin_n,
        contact_transition=contact_transition,
        segment_balance_force_residual_n=torch.stack(segment_force_residual),
        segment_balance_torque_residual_nm=torch.stack(segment_torque_residual),
        segment_effort_margin_ratio=segment_effort_margin_ratio,
        segment_normal_force_min_n=segment_normal_force_min_n,
        segment_friction_margin_n=segment_friction_margin_n,
        segment_contact_transition_count=torch.stack(segment_transition_count),
    )
