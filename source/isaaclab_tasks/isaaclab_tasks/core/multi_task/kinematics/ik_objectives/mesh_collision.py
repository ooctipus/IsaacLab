# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective and probe construction for robot-to-mesh collision avoidance."""

from __future__ import annotations

import math
from collections import defaultdict

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp

from ..collider_geometry import points_transform_xyzw
from ._kernels import jac_fill_row


@wp.kernel
def _basis_column_write(
    total_residuals: int,
    column: int,
    value: float,
    output: wp.array1d(dtype=wp.float32),
):
    problem_index = wp.tid()
    output[problem_index * total_residuals + column] = value


@wp.kernel
def _mesh_collision_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    probe_bodies: wp.array1d(dtype=wp.int32),
    probe_offsets: wp.array1d(dtype=wp.vec3),
    probe_contact_slots: wp.array1d(dtype=wp.int32),
    contact_mask: wp.array2d(dtype=wp.uint8),
    mesh_id: wp.uint64,
    obstacle_q: wp.array1d(dtype=wp.transformf),
    weight: float,
    margin: float,
    max_distance: float,
    use_up_axis: wp.uint8,
    up_axis: wp.vec3,
    start_index: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem_index, probe_index = wp.tid()
    contact_slot = probe_contact_slots[probe_index]
    if contact_slot >= 0 and contact_mask[problem_index, contact_slot] != wp.uint8(0):
        return

    world_point = wp.transform_point(body_q[problem_index, probe_bodies[probe_index]], probe_offsets[probe_index])
    obstacle_pose = obstacle_q[problem_index]
    local_point = wp.transform_point(wp.transform_inverse(obstacle_pose), world_point)
    query = wp.mesh_query_point(mesh_id, local_point, max_distance)
    if not query.result:
        return

    local_surface = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    signed_depth = -query.sign * wp.length(local_point - local_surface)
    depth = signed_depth
    if use_up_axis != wp.uint8(0):
        world_surface = wp.transform_point(obstacle_pose, local_surface)
        depth = wp.max(signed_depth, wp.dot(world_surface - world_point, up_axis))
    x = depth / margin
    softplus = float(0.0)
    if x >= 0.0:
        softplus = x + wp.log(1.0 + wp.exp(-x))
    else:
        softplus = wp.log(1.0 + wp.exp(x))
    residuals[problem_index, start_index + probe_index] = weight * softplus * margin


@wp.kernel
def _mesh_collision_jacobian(
    mesh_id: wp.uint64,
    probe_bodies: wp.array1d(dtype=wp.int32),
    probe_offsets: wp.array1d(dtype=wp.vec3),
    probe_contact_slots: wp.array1d(dtype=wp.int32),
    contact_mask: wp.array2d(dtype=wp.uint8),
    obstacle_q: wp.array1d(dtype=wp.transformf),
    weight: float,
    margin: float,
    max_distance: float,
    use_up_axis: wp.uint8,
    up_axis: wp.vec3,
    affects_dof: wp.array2d(dtype=wp.uint8),
    body_q: wp.array2d(dtype=wp.transform),
    joint_screw: wp.array2d(dtype=wp.spatial_vector),
    start_index: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_index, probe_index = wp.tid()
    contact_slot = probe_contact_slots[probe_index]
    if contact_slot >= 0 and contact_mask[problem_index, contact_slot] != wp.uint8(0):
        return

    body_index = probe_bodies[probe_index]
    world_point = wp.transform_point(body_q[problem_index, body_index], probe_offsets[probe_index])
    obstacle_pose = obstacle_q[problem_index]
    local_point = wp.transform_point(wp.transform_inverse(obstacle_pose), world_point)
    query = wp.mesh_query_point(mesh_id, local_point, max_distance)
    if not query.result:
        return

    local_surface = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    delta = local_point - local_surface
    distance = wp.length(delta)
    signed_depth = -query.sign * distance
    depth = signed_depth
    signed_branch = wp.uint8(1)
    if use_up_axis != wp.uint8(0):
        world_surface = wp.transform_point(obstacle_pose, local_surface)
        upward_depth = wp.dot(world_surface - world_point, up_axis)
        if upward_depth > signed_depth:
            depth = upward_depth
            signed_branch = wp.uint8(0)

    gradient_world = wp.vec3(0.0, 0.0, 0.0)
    if signed_branch != wp.uint8(0):
        if distance <= 1.0e-8:
            return
        scale = -query.sign / distance
        gradient_local = wp.vec3(scale * delta[0], scale * delta[1], scale * delta[2])
        gradient_world = wp.quat_rotate(wp.transform_get_rotation(obstacle_pose), gradient_local)
    else:
        gradient_world = -up_axis

    x = depth / margin
    sigmoid = float(0.0)
    if x >= 0.0:
        sigmoid = 1.0 / (1.0 + wp.exp(-x))
    else:
        exponential = wp.exp(x)
        sigmoid = exponential / (1.0 + exponential)

    dof_count = affects_dof.shape[1]
    for dof_index in range(dof_count):
        if affects_dof[probe_index, dof_index] == wp.uint8(0):
            continue
        screw = joint_screw[problem_index, dof_index]
        linear = wp.vec3(screw[0], screw[1], screw[2])
        angular = wp.vec3(screw[3], screw[4], screw[5])
        point_velocity = linear + wp.cross(angular, world_point)
        jacobian[problem_index, start_index + probe_index, dof_index] = (
            weight * sigmoid * wp.dot(gradient_world, point_velocity)
        )


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Return approximately uniform unit-sphere directions."""
    if count <= 1:
        return np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    angle = np.pi * (3.0 - np.sqrt(5.0))
    points = np.empty((count, 3), dtype=np.float32)
    for index in range(count):
        y = 1.0 - 2.0 * index / float(count - 1)
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        azimuth = angle * index
        points[index] = (np.cos(azimuth) * radius, y, np.sin(azimuth) * radius)
    return points


def _primitive_surface_probes(shape_type: int, shape_scale, shape_transform, count: int) -> np.ndarray:
    """Return shape-local surface probes for one supported primitive."""
    from newton import GeoType

    del shape_transform
    if shape_type == int(GeoType.SPHERE):
        return float(shape_scale[0]) * _fibonacci_sphere(count)
    if shape_type == int(GeoType.BOX):
        half_extents = np.asarray(shape_scale[:3], dtype=np.float32)
        signs = np.array(
            [
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return signs * half_extents
    if shape_type == int(GeoType.CAPSULE):
        radius = float(shape_scale[0])
        half_length = float(shape_scale[1])
        cap_count = max(1, count // 2)
        top = radius * _fibonacci_sphere(cap_count)
        bottom = radius * _fibonacci_sphere(cap_count)
        top[:, 2] += half_length
        bottom[:, 2] -= half_length
        return np.concatenate((top, bottom))
    if shape_type == int(GeoType.CYLINDER):
        radius = float(shape_scale[0])
        half_length = float(shape_scale[1])
        ring_count = max(4, count // 2)
        azimuth = np.arange(ring_count, dtype=np.float32) * (2.0 * np.pi / ring_count)
        ring = np.stack((radius * np.cos(azimuth), radius * np.sin(azimuth)), axis=-1)
        lower = np.column_stack((ring, np.full(ring_count, -half_length, dtype=np.float32)))
        upper = np.column_stack((ring, np.full(ring_count, half_length, dtype=np.float32)))
        return np.concatenate((lower, upper))
    return np.empty((0, 3), dtype=np.float32)


def collision_probes_sample(
    builder: newton.ModelBuilder,
    contact_body_ids: tuple[int, ...] | list[int],
    n_samples: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample collision probes from every body shape.

    Args:
        builder: Finalized robot model builder.
        contact_body_ids: Bodies whose probes map to contact-mask columns.
        n_samples: Maximum retained probes per body.

    Returns:
        Probe body indices, body-local offsets [m], and contact-mask slots.
    """
    contact_slots = {body: slot for slot, body in enumerate(contact_body_ids)}
    body_shapes: dict[int, list[int]] = defaultdict(list)
    for shape_index, body_index in enumerate(builder.shape_body):
        body_shapes[int(body_index)].append(shape_index)

    probe_bodies: list[int] = []
    probe_offsets: list[np.ndarray] = []
    probe_slots: list[int] = []
    for body_index in sorted(body_shapes):
        candidates: list[np.ndarray] = []
        for shape_index in body_shapes[body_index]:
            source = builder.shape_source[shape_index]
            if source is not None and hasattr(source, "vertices") and len(source.vertices) > 0:
                shape_points = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
            else:
                shape_points = _primitive_surface_probes(
                    int(builder.shape_type[shape_index]),
                    builder.shape_scale[shape_index],
                    builder.shape_transform[shape_index],
                    n_samples,
                )
            transform = np.asarray(builder.shape_transform[shape_index], dtype=np.float32)
            candidates.append(points_transform_xyzw(shape_points, transform[:3], transform[3:7]))
        candidates = [values for values in candidates if values.shape[0]]
        if not candidates:
            continue
        points = np.concatenate(candidates)
        count = min(n_samples, points.shape[0])
        selected = [0]
        minimum_distance = np.full(points.shape[0], np.inf)
        for _ in range(count - 1):
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(points - points[selected[-1]], axis=1),
            )
            selected.append(int(np.argmax(minimum_distance)))
        probe_bodies.extend([body_index] * count)
        probe_offsets.extend(points[selected])
        probe_slots.extend([contact_slots.get(body_index, -1)] * count)
    return (
        np.asarray(probe_bodies, dtype=np.int32),
        np.asarray(probe_offsets, dtype=np.float32).reshape(-1, 3),
        np.asarray(probe_slots, dtype=np.int32),
    )


class IKObjectiveMeshCollision(ik.IKObjective):
    """Penalize robot-body probes penetrating a posed obstacle mesh.

    Args:
        probe_offsets: Body-local probe offsets [m], shape [probe_count, 3].
        probe_bodies: Probe body indices, shape [probe_count].
        mesh: Obstacle Warp mesh or mesh identifier.
        probe_affects_dof: Canonical probe-body ancestry mask, shape [probe_count, dof_count].
        obstacle_pose: Per-problem obstacle poses [m, quaternion xyzw], shape
            [problem_count, 7].
        weight: Residual weight.
        margin: Softplus smoothing scale [m].
        max_distance: Mesh-query radius [m].
        probe_contact_slots: Optional contact-mask column per probe; -1 is always active.
        contact_mask: Optional mutable uint8 mask, shape [problem_count, contact_count].
        one_sided_up_axis: Optional world unit vector. When present, also penalize
            a probe below the closest surface along this axis.
    """

    def __init__(
        self,
        probe_offsets: np.ndarray,
        probe_bodies: np.ndarray,
        probe_affects_dof: np.ndarray,
        mesh: wp.Mesh | int,
        obstacle_pose: torch.Tensor,
        weight: float,
        margin: float,
        max_distance: float = 0.25,
        probe_contact_slots: np.ndarray | None = None,
        contact_mask: torch.Tensor | None = None,
        one_sided_up_axis: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self._probe_offsets_np = np.asarray(probe_offsets, dtype=np.float32)
        self._probe_bodies_np = np.asarray(probe_bodies, dtype=np.int32)
        self._probe_affects_dof_np = np.asarray(probe_affects_dof, dtype=np.uint8)
        if self._probe_offsets_np.ndim != 2 or self._probe_offsets_np.shape[1] != 3:
            raise ValueError("Mesh-collision probe offsets must have shape [probe_count, 3].")
        if self._probe_bodies_np.shape != (self._probe_offsets_np.shape[0],):
            raise ValueError("Mesh-collision probe body indices must have shape [probe_count].")
        if self._probe_affects_dof_np.ndim != 2 or self._probe_affects_dof_np.shape[0] != len(self._probe_bodies_np):
            raise ValueError("Mesh-collision ancestry must have shape [probe_count, dof_count].")
        if obstacle_pose.ndim != 2 or obstacle_pose.shape[1] != 7:
            raise ValueError("Mesh-collision obstacle poses must have shape [problem_count, 7].")
        self._probe_contact_slots_np = (
            np.full(self._probe_bodies_np.shape, -1, dtype=np.int32)
            if probe_contact_slots is None
            else np.asarray(probe_contact_slots, dtype=np.int32)
        )
        if self._probe_contact_slots_np.shape != self._probe_bodies_np.shape:
            raise ValueError("Mesh-collision contact slots must have shape [probe_count].")
        if np.any(self._probe_contact_slots_np >= 0) and contact_mask is None:
            raise ValueError("Contact-gated mesh probes require a contact mask.")
        if contact_mask is not None and (contact_mask.ndim != 2 or contact_mask.shape[0] != obstacle_pose.shape[0]):
            raise ValueError("Mesh-collision contact mask must have shape [problem_count, contact_count].")
        self._contact_mask_t = contact_mask
        self._obstacle_pose_t = obstacle_pose.contiguous()
        self.mesh_id = wp.uint64(mesh.id if hasattr(mesh, "id") else int(mesh))
        self.weight = float(weight)
        self.margin = float(margin)
        self.max_distance = float(max_distance)
        if not math.isfinite(self.margin) or self.margin <= 0.0:
            raise ValueError("Mesh-collision margin must be finite and positive.")
        if not math.isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise ValueError("Mesh-collision max_distance must be finite and positive.")
        axis = (0.0, 0.0, 0.0) if one_sided_up_axis is None else one_sided_up_axis
        axis_array = np.asarray(axis, dtype=np.float32)
        if axis_array.shape != (3,):
            raise ValueError("Mesh-collision one-sided up axis must have three components.")
        norm = float(np.linalg.norm(axis_array))
        if one_sided_up_axis is not None and norm <= 0.0:
            raise ValueError("Mesh-collision one-sided up axis must be nonzero.")
        self._up_axis = wp.vec3(*(axis_array / norm if norm > 0.0 else axis_array))
        self._use_up_axis = wp.uint8(one_sided_up_axis is not None)

    def supports_analytic(self) -> bool:
        """Return whether this objective supplies an analytic Jacobian."""
        return True

    def residual_dim(self) -> int:
        """Return the number of collision probes."""
        return self._probe_offsets_np.shape[0]

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        """Bind immutable probes and mutable per-problem evidence to the solver."""
        self._require_batch_layout()
        if self._obstacle_pose_t.shape[0] != self.n_batch:
            raise ValueError("Mesh-collision obstacle pose rows must equal the IK batch size.")
        self._probe_offsets = wp.from_numpy(self._probe_offsets_np, dtype=wp.vec3, device=self.device)
        self._probe_bodies = wp.from_numpy(self._probe_bodies_np, dtype=wp.int32, device=self.device)
        self._probe_contact_slots = wp.from_numpy(
            self._probe_contact_slots_np,
            dtype=wp.int32,
            device=self.device,
        )
        if self._contact_mask_t is None:
            self._contact_mask_t = torch.zeros(
                (self.n_batch, 1), dtype=torch.uint8, device=self._obstacle_pose_t.device
            )
        self._contact_mask = wp.from_torch(self._contact_mask_t, dtype=wp.uint8)
        self._obstacle_pose = wp.from_torch(self._obstacle_pose_t, dtype=wp.transformf)
        if jacobian_mode in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            self._affects_dof = wp.array(self._probe_affects_dof_np, dtype=wp.uint8, device=self.device)
        if jacobian_mode in (ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED):
            self._autodiff_basis = wp.zeros(
                self.n_batch * self.total_residuals,
                dtype=wp.float32,
                device=self.device,
            )

    def estimate_memory(
        self,
        model: newton.Model,
        jacobian_mode: ik.IKJacobianType,
        n_problems: int,
        n_batch: int,
        total_residuals: int,
    ) -> int:
        """Estimate immutable probes and Jacobian workspaces [byte]."""
        del model
        fixed_bytes = self._probe_offsets_np.nbytes + self._probe_bodies_np.nbytes + self._probe_contact_slots_np.nbytes
        if jacobian_mode in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            fixed_bytes += self._probe_affects_dof_np.nbytes
        workspace_bytes = 0 if self._contact_mask_t is not None else n_problems * wp.types.type_size_in_bytes(wp.uint8)
        if jacobian_mode in (ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED):
            workspace_bytes += n_batch * total_residuals * wp.types.type_size_in_bytes(wp.float32)
        return int(fixed_bytes + workspace_bytes)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        """Write one smoothed penetration residual per probe."""
        del joint_q, model, problem_idx
        wp.launch(
            _mesh_collision_residuals,
            dim=(body_q.shape[0], self.residual_dim()),
            inputs=[
                body_q,
                self._probe_bodies,
                self._probe_offsets,
                self._probe_contact_slots,
                self._contact_mask,
                self.mesh_id,
                self._obstacle_pose,
                self.weight,
                self.margin,
                self.max_distance,
                self._use_up_axis,
                self._up_axis,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        """Materialize one residual basis column at a time for Warp autodiff."""
        del model
        dof_count = dq_dof.shape[1]
        for residual_index in range(self.residual_dim()):
            column = self.residual_offset + residual_index
            wp.launch(
                _basis_column_write,
                dim=self.n_batch,
                inputs=[self.total_residuals, column, 1.0],
                outputs=[self._autodiff_basis],
                device=self.device,
            )
            tape.backward(grads={tape.outputs[0]: self._autodiff_basis})
            wp.launch(
                jac_fill_row,
                dim=self.n_batch,
                inputs=[tape.gradients[dq_dof], dof_count, start_idx + residual_index],
                outputs=[jacobian],
                device=self.device,
            )
            wp.launch(
                _basis_column_write,
                dim=self.n_batch,
                inputs=[self.total_residuals, column, 0.0],
                outputs=[self._autodiff_basis],
                device=self.device,
            )
            tape.zero()

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """Write the analytic point-to-mesh Jacobian."""
        del joint_q, model
        wp.launch(
            _mesh_collision_jacobian,
            dim=(body_q.shape[0], self.residual_dim()),
            inputs=[
                self.mesh_id,
                self._probe_bodies,
                self._probe_offsets,
                self._probe_contact_slots,
                self._contact_mask,
                self._obstacle_pose,
                self.weight,
                self.margin,
                self.max_distance,
                self._use_up_axis,
                self._up_axis,
                self._affects_dof,
                body_q,
                joint_S_s,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


def build_mesh_collision_objective(cfg, context):
    """Build a collision objective from explicit probes and obstacle poses."""
    from .context import IKMeshCollisionObjectiveBuildContext, IKObjectiveBuild

    if not isinstance(context, IKMeshCollisionObjectiveBuildContext):
        raise TypeError("Mesh-collision objectives require IKMeshCollisionObjectiveBuildContext.")
    objective = IKObjectiveMeshCollision(
        probe_offsets=context.probe_offsets,
        probe_bodies=context.probe_bodies,
        mesh=context.collision_mesh,
        probe_affects_dof=context.kinematics.topology.body_dof_ancestry[context.probe_bodies],
        obstacle_pose=context.obstacle_pose,
        weight=cfg.weight,
        margin=cfg.margin,
        max_distance=cfg.max_distance,
        probe_contact_slots=context.probe_contact_slots,
        contact_mask=context.contact_mask,
        one_sided_up_axis=cfg.one_sided_up_axis,
    )
    return IKObjectiveBuild(objectives=(objective,))
