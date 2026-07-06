# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective for per-problem joint-coordinate targets."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp


@wp.kernel
def _joint_pin_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array1d(dtype=wp.int32),
    targets: wp.array2d(dtype=wp.float32),
    weight: float,
    start_index: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem_index, target_index = wp.tid()
    coordinate_index = coordinate_indices[target_index]
    residuals[problem_index, start_index + target_index] = weight * (
        joint_q[problem_index, coordinate_index] - targets[problem_index, target_index]
    )


@wp.kernel
def _joint_pin_jacobian(
    dof_indices: wp.array1d(dtype=wp.int32),
    weight: float,
    start_index: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_index, target_index = wp.tid()
    jacobian[problem_index, start_index + target_index, dof_indices[target_index]] = weight


class IKObjectiveJointPin(ik.IKObjective):
    """Pin selected joint coordinates to per-problem targets.

    Args:
        coordinate_indices: Joint-coordinate indices, shape [target_count].
        dof_indices: Matching joint-DOF indices, shape [target_count].
        targets: Per-problem targets [m or rad, depending on joint type], shape
            [problem_count, target_count].
        weight: Residual weight.
    """

    def __init__(
        self,
        coordinate_indices: np.ndarray,
        dof_indices: np.ndarray,
        targets: torch.Tensor,
        weight: float,
    ) -> None:
        super().__init__()
        self._coordinate_indices_np = np.asarray(coordinate_indices, dtype=np.int32)
        self._dof_indices_np = np.asarray(dof_indices, dtype=np.int32)
        if self._coordinate_indices_np.ndim != 1 or self._dof_indices_np.shape != self._coordinate_indices_np.shape:
            raise ValueError("Joint-pin coordinate and DOF indices must be equal-shaped one-dimensional arrays.")
        if targets.ndim != 2 or targets.shape[1] != self._coordinate_indices_np.shape[0]:
            raise ValueError("Joint-pin targets must have shape [problem_count, target_count].")
        self._targets_t = targets.contiguous()
        self.weight = float(weight)

    def supports_analytic(self) -> bool:
        """Return whether this objective supplies an analytic Jacobian."""
        return True

    def residual_dim(self) -> int:
        """Return the number of pinned coordinates."""
        return self._coordinate_indices_np.shape[0]

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        """Bind fixed indices and per-problem targets to the solver device."""
        del model, jacobian_mode
        self._require_batch_layout()
        self._coordinate_indices = wp.from_numpy(
            self._coordinate_indices_np,
            dtype=wp.int32,
            device=self.device,
        )
        self._dof_indices = wp.from_numpy(self._dof_indices_np, dtype=wp.int32, device=self.device)
        self._targets = wp.from_torch(self._targets_t)

    def estimate_memory(
        self,
        model: newton.Model,
        jacobian_mode: ik.IKJacobianType,
        n_problems: int,
        n_batch: int,
        total_residuals: int,
    ) -> int:
        """Estimate immutable indices and per-problem pin targets [byte]."""
        del model, jacobian_mode, n_batch, total_residuals
        count = self._coordinate_indices_np.shape[0]
        int_bytes = wp.types.type_size_in_bytes(wp.int32)
        float_bytes = wp.types.type_size_in_bytes(wp.float32)
        return 2 * count * int_bytes + n_problems * count * float_bytes

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        """Write weighted coordinate errors for every problem."""
        del model, problem_idx
        wp.launch(
            _joint_pin_residuals,
            dim=(body_q.shape[0], self.residual_dim()),
            inputs=[joint_q, self._coordinate_indices, self._targets, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """Write the constant sparse coordinate Jacobian."""
        del joint_q, model, joint_S_s
        wp.launch(
            _joint_pin_jacobian,
            dim=(body_q.shape[0], self.residual_dim()),
            inputs=[self._dof_indices, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )


def build_joint_pin_objective(cfg, context):
    """Build one per-problem joint-coordinate target objective."""
    from .context import IKJointPinObjectiveBuildContext, IKObjectiveBuild

    if not isinstance(context, IKJointPinObjectiveBuildContext):
        raise TypeError("Joint-pin objectives require IKJointPinObjectiveBuildContext.")
    objective = IKObjectiveJointPin(
        coordinate_indices=context.coordinate_indices,
        dof_indices=context.dof_indices,
        targets=context.targets,
        weight=cfg.weight,
    )
    return IKObjectiveBuild(objectives=(objective,))
