# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Support-frame gap, upright, and planted-point IK features."""

from __future__ import annotations

import math

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp


@wp.kernel
def _support_patch_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    support_pose: wp.array2d(dtype=wp.float32),
    points_body: wp.array(dtype=wp.vec3),
    target_points_world: wp.array2d(dtype=wp.vec3),
    body: int,
    normal_body: wp.vec3,
    inverse_gap_tolerance: float,
    inverse_upright_tolerance: float,
    inverse_point_tolerance: float,
    start: int,
    problem_indices: wp.array(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    """Write normalized support features for one patch."""
    row = wp.tid()
    target_row = problem_indices[row]
    body_pose = body_q[row, body]
    body_rotation = wp.transform_get_rotation(body_pose)
    support_origin = wp.vec3(support_pose[target_row, 0], support_pose[target_row, 1], support_pose[target_row, 2])
    support_rotation = wp.quat(
        support_pose[target_row, 3],
        support_pose[target_row, 4],
        support_pose[target_row, 5],
        support_pose[target_row, 6],
    )
    support_inverse = wp.quat_inverse(support_rotation)
    tangent_x = wp.quat_rotate(support_rotation, wp.vec3(1.0, 0.0, 0.0))
    tangent_y = wp.quat_rotate(support_rotation, wp.vec3(0.0, 1.0, 0.0))
    support_normal = wp.quat_rotate(support_rotation, wp.vec3(0.0, 0.0, 1.0))
    patch_normal = wp.quat_rotate(body_rotation, normal_body)
    gap = float(0.0)
    point_count = points_body.shape[0]
    for point in range(point_count):
        world_point = wp.transform_point(body_pose, points_body[point])
        target_point = target_points_world[point, target_row]
        gap += wp.dot(world_point - support_origin, support_normal)
        support_error = wp.quat_rotate(support_inverse, world_point - target_point)
        point_start = start + 4 + 3 * point
        residuals[row, point_start] = inverse_point_tolerance * support_error[0]
        residuals[row, point_start + 1] = inverse_point_tolerance * support_error[1]
        residuals[row, point_start + 2] = inverse_point_tolerance * support_error[2]
    residuals[row, start] = inverse_gap_tolerance * gap / float(point_count)
    residuals[row, start + 1] = inverse_upright_tolerance * wp.dot(tangent_x, patch_normal)
    residuals[row, start + 2] = inverse_upright_tolerance * wp.dot(tangent_y, patch_normal)
    residuals[row, start + 3] = inverse_upright_tolerance * (wp.dot(support_normal, patch_normal) - 1.0)


@wp.kernel
def _support_patch_jacobian(
    body: int,
    points_body: wp.array(dtype=wp.vec3),
    normal_body: wp.vec3,
    support_pose: wp.array2d(dtype=wp.float32),
    affects_dof: wp.array(dtype=wp.uint8),
    body_q: wp.array2d(dtype=wp.transform),
    joint_screw: wp.array2d(dtype=wp.spatial_vector),
    start: int,
    inverse_gap_tolerance: float,
    inverse_upright_tolerance: float,
    inverse_point_tolerance: float,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Write analytic support-feature Jacobians."""
    row, dof = wp.tid()
    if affects_dof[dof] == wp.uint8(0):
        return
    body_pose = body_q[row, body]
    body_rotation = wp.transform_get_rotation(body_pose)
    support_rotation = wp.quat(support_pose[row, 3], support_pose[row, 4], support_pose[row, 5], support_pose[row, 6])
    support_inverse = wp.quat_inverse(support_rotation)
    tangent_x = wp.quat_rotate(support_rotation, wp.vec3(1.0, 0.0, 0.0))
    tangent_y = wp.quat_rotate(support_rotation, wp.vec3(0.0, 1.0, 0.0))
    support_normal = wp.quat_rotate(support_rotation, wp.vec3(0.0, 0.0, 1.0))
    patch_normal = wp.quat_rotate(body_rotation, normal_body)
    screw = joint_screw[row, dof]
    linear = wp.vec3(screw[0], screw[1], screw[2])
    angular = wp.vec3(screw[3], screw[4], screw[5])
    point_count = points_body.shape[0]
    gap_derivative = float(0.0)
    for point in range(point_count):
        world_point = wp.transform_point(body_pose, points_body[point])
        point_velocity = linear + wp.cross(angular, world_point)
        gap_derivative += wp.dot(point_velocity, support_normal)
        support_velocity = wp.quat_rotate(support_inverse, point_velocity)
        point_start = start + 4 + 3 * point
        jacobian[row, point_start, dof] = inverse_point_tolerance * support_velocity[0]
        jacobian[row, point_start + 1, dof] = inverse_point_tolerance * support_velocity[1]
        jacobian[row, point_start + 2, dof] = inverse_point_tolerance * support_velocity[2]
    normal_derivative = wp.cross(angular, patch_normal)
    jacobian[row, start, dof] = inverse_gap_tolerance * gap_derivative / float(point_count)
    jacobian[row, start + 1, dof] = inverse_upright_tolerance * wp.dot(tangent_x, normal_derivative)
    jacobian[row, start + 2, dof] = inverse_upright_tolerance * wp.dot(tangent_y, normal_derivative)
    jacobian[row, start + 3, dof] = inverse_upright_tolerance * wp.dot(support_normal, normal_derivative)


class IKObjectiveSupportPatch(ik.IKObjective):
    """Expose normalized support gap, yaw-free upright, and planted-point errors.

    Rows zero through three yield mean support gap and the signed yaw-free
    patch-normal difference. The remaining rows are support-frame errors from
    actual patch points to caller-owned world targets. All point rows use one
    patch-cardinality-invariant RMS tolerance.

    Args:
        body: Body index that owns the patch.
        points_body: Patch points [m], shape ``[point_count, 3]``.
        target_points_world: World point targets [m], shape ``[point_count, batch_size, 3]``.
        normal_body: Unit patch normal in the body frame.
        support_pose: World support poses [m and quaternion], shape ``[batch_size, 7]``.
        affects_dof: Canonical body ancestry, shape ``[dof_count]``, uint8.
        gap_tolerance_m: Support-gap scale [m].
        tilt_tolerance_rad: Upright-angle scale [rad].
        point_tolerance_m: RMS point-error tolerance [m] across the patch points.
    """

    def __init__(
        self,
        *,
        body: int,
        points_body: torch.Tensor,
        target_points_world: torch.Tensor,
        normal_body: torch.Tensor,
        support_pose: torch.Tensor,
        affects_dof: np.ndarray,
        gap_tolerance_m: float,
        tilt_tolerance_rad: float,
        point_tolerance_m: float,
    ) -> None:
        super().__init__()
        if (
            points_body.ndim != 2
            or points_body.shape[0] < 1
            or points_body.shape[1] != 3
            or normal_body.shape != (3,)
            or support_pose.ndim != 2
            or support_pose.shape[1] != 7
            or target_points_world.shape != (points_body.shape[0], support_pose.shape[0], 3)
            or points_body.dtype is not torch.float32
            or target_points_world.dtype is not torch.float32
            or normal_body.dtype is not torch.float32
            or support_pose.dtype is not torch.float32
            or points_body.device != support_pose.device
            or target_points_world.device != support_pose.device
            or normal_body.device != support_pose.device
            or not points_body.is_contiguous()
            or not target_points_world.is_contiguous()
            or not support_pose.is_contiguous()
        ):
            raise ValueError(
                "Support-patch geometry, targets, and poses must be contiguous same-device float32 tensors."
            )
        normal_length = torch.linalg.vector_norm(normal_body)
        if not torch.isfinite(normal_length) or not torch.isclose(normal_length, normal_length.new_tensor(1.0)):
            raise ValueError("The support-patch normal must be a finite unit vector.")
        if (
            not math.isfinite(gap_tolerance_m)
            or not math.isfinite(tilt_tolerance_rad)
            or not math.isfinite(point_tolerance_m)
            or gap_tolerance_m <= 0.0
            or tilt_tolerance_rad <= 0.0
            or tilt_tolerance_rad >= 0.5 * math.pi
            or point_tolerance_m <= 0.0
        ):
            raise ValueError("Support-patch objective tolerances must be finite and positive.")
        self.body = body
        self._points_body_t = points_body
        self.normal_body = wp.vec3(*(float(value) for value in normal_body.tolist()))
        self._target_points_world_t = target_points_world
        self._support_pose_t = support_pose
        self._affects_dof_np = np.asarray(affects_dof, dtype=np.uint8)
        self.inverse_gap_tolerance = 1.0 / gap_tolerance_m
        self.inverse_upright_tolerance = 1.0 / (2.0 * math.sin(0.5 * tilt_tolerance_rad))
        self.inverse_point_tolerance = 1.0 / (point_tolerance_m * math.sqrt(points_body.shape[0]))

    def supports_analytic(self) -> bool:
        """Return ``True`` because the objective supplies an analytic Jacobian."""
        return True

    def residual_dim(self) -> int:
        """Return four support-pose rows and three target-relative rows per patch point."""
        return 4 + 3 * self._points_body_t.shape[0]

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        """Bind caller-owned support poses, point targets, and static patch geometry."""
        self._require_batch_layout()
        if (
            self._support_pose_t.shape[0] != self.n_batch
            or self._target_points_world_t.shape[1] != self.n_batch
            or self._affects_dof_np.shape != (model.joint_dof_count,)
        ):
            raise ValueError("Support-patch pose/target rows or ancestry differ from the IK batch.")
        if jacobian_mode not in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            raise ValueError("Support-patch objectives require an analytic or mixed Jacobian.")
        self._points_body = wp.from_torch(self._points_body_t, dtype=wp.vec3)
        self._support_pose = wp.from_torch(self._support_pose_t)
        self._target_points_world = wp.from_torch(self._target_points_world_t, dtype=wp.vec3)
        self._affects_dof = wp.from_numpy(self._affects_dof_np, dtype=wp.uint8, device=self.device)

    def estimate_memory(self, model, jacobian_mode, n_problems, n_batch, total_residuals) -> int:
        """Estimate the objective-owned ancestry mask [byte]."""
        del n_problems, n_batch, total_residuals
        if jacobian_mode not in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            raise ValueError("Support-patch objectives require an analytic or mixed Jacobian.")
        return model.joint_dof_count * wp.types.type_size_in_bytes(wp.uint8)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        """Write support gap, upright, and support-frame target-relative point errors."""
        del joint_q, model
        wp.launch(
            _support_patch_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self._support_pose,
                self._points_body,
                self._target_points_world,
                self.body,
                self.normal_body,
                self.inverse_gap_tolerance,
                self.inverse_upright_tolerance,
                self.inverse_point_tolerance,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_screw, start_idx) -> None:
        """Write analytic support-feature Jacobian blocks."""
        del joint_q
        wp.launch(
            _support_patch_jacobian,
            dim=(body_q.shape[0], model.joint_dof_count),
            inputs=[
                self.body,
                self._points_body,
                self.normal_body,
                self._support_pose,
                self._affects_dof,
                body_q,
                joint_screw,
                start_idx,
                self.inverse_gap_tolerance,
                self.inverse_upright_tolerance,
                self.inverse_point_tolerance,
            ],
            outputs=[jacobian],
            device=self.device,
        )
