# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: joint default regularization."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import warp as wp


@wp.kernel
def _default_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    target_q: wp.array1d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    residuals[problem, start_idx + dof_idx] = weight * (joint_q[problem, coord_idx] - target_q[dof_idx])


@wp.kernel
def _default_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight


class IKObjectiveJointDefault(ik.IKObjective):
    """Penalize deviation from a target joint configuration.

    Each DOF contributes one residual: ``weight * (q - q_target)``.
    The analytic Jacobian is a constant diagonal of ``weight``.

    Args:
        target_joint_q: Target joint coordinates ``[joint_dof_count]``.
        weight: Scalar multiplier for the regularization residual.
        skip_root: Exclude the free-root joint (joint 0) DOFs.
    """

    def __init__(self, target_joint_q: wp.array, weight: float = 0.1, *, skip_root: bool = True) -> None:
        super().__init__()
        self.target_joint_q = target_joint_q
        self.weight = weight
        self.n_dofs = len(target_joint_q)
        self._skip_root = skip_root
        self.dof_to_coord: wp.array | None = None

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        start_joint = 1 if self._skip_root else 0
        for j in range(start_joint, model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                dof_to_coord_np[dof0 + k] = coord0 + k
        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return self.n_dofs

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _default_residuals,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[joint_q, self.target_joint_q, self.dof_to_coord, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        wp.launch(
            _default_jac_analytic,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[self.dof_to_coord, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
