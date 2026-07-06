# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: regularize a subset of joint DOFs toward per-DOF target angles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .cfg import IKObjectiveJointRegularizeCfg
    from .context import IKObjectiveBuildContext


@wp.kernel
def _subset_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    coord_indices: wp.array1d(dtype=wp.int32),
    targets: wp.array1d(dtype=wp.float32),
    weight: wp.float32,
    start_idx: wp.int32,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, i = wp.tid()
    coord_idx = coord_indices[i]
    residuals[problem, start_idx + i] = weight * (joint_q[problem, coord_idx] - targets[i])


@wp.kernel
def _subset_jac_analytic(
    dof_indices: wp.array1d(dtype=wp.int32),
    weight: wp.float32,
    start_idx: wp.int32,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, i = wp.tid()
    dof_idx = dof_indices[i]
    jacobian[problem, start_idx + i, dof_idx] = weight


class IKObjectiveJointRegularize(ik.IKObjective):
    """Penalize a subset of joint DOFs toward per-DOF target angles.

    Residual ``i`` is ``weight * (q[coord[i]] - targets[i])`` for each DOF
    in ``joint_dof_indices``. The analytic Jacobian is a constant diagonal
    of ``weight`` restricted to those rows/columns. DOFs not listed are
    untouched -- unlike :class:`IKObjectiveJointDefault`, which regularizes
    every DOF.

    Primary use case: selectively regularize joint subsets (e.g. pull HAA
    toward 0, HFE toward a crouch angle) during retarget IK. Each matched
    DOF gets its own target, so one objective can mix targets across joint
    groups.

    Args:
        cfg: :class:`~.cfg.IKObjectiveJointRegularizeCfg` with
            ``joint_targets`` (regex → target-angle map) and ``weight``.
        context: Explicit kinematics used for joint-name resolution.
    """

    def __init__(
        self,
        cfg: IKObjectiveJointRegularizeCfg,
        context: IKObjectiveBuildContext,
    ) -> None:
        super().__init__()
        targets_map = cfg.joint_targets
        if not targets_map:
            raise ValueError("IKObjectiveJointRegularize requires at least one entry in cfg.joint_targets.")
        coordinate_targets: dict[int, tuple[int, float]] = {}
        for pattern, target in targets_map.items():
            coordinates, velocities, _ = context.kinematics.find_joint_scalar_coordinates(pattern)
            for coordinate, velocity in zip(coordinates, velocities, strict=True):
                coordinate_targets[coordinate] = (velocity, float(target))
        if not coordinate_targets:
            raise ValueError(
                f"IKObjectiveJointRegularize: none of the patterns {list(targets_map)} matched a scalar joint."
            )
        self._coordinate_indices = sorted(coordinate_targets)
        self._velocity_indices = [coordinate_targets[index][0] for index in self._coordinate_indices]
        self._dof_targets = [coordinate_targets[index][1] for index in self._coordinate_indices]
        self.weight = float(cfg.weight)
        self.coord_indices: wp.array | None = None
        self.dof_indices: wp.array | None = None
        self.targets: wp.array | None = None

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        coord_np = np.asarray(self._coordinate_indices, dtype=np.int32)
        dof_np = np.asarray(self._velocity_indices, dtype=np.int32)
        targets_np = np.asarray(self._dof_targets, dtype=np.float32)
        self.coord_indices = wp.array(coord_np, dtype=wp.int32, device=self.device)
        self.dof_indices = wp.array(dof_np, dtype=wp.int32, device=self.device)
        self.targets = wp.array(targets_np, dtype=wp.float32, device=self.device)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return len(self._coordinate_indices)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _subset_residuals,
            dim=[joint_q.shape[0], self.residual_dim()],
            inputs=[joint_q, self.coord_indices, self.targets, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        wp.launch(
            _subset_jac_analytic,
            dim=[joint_q.shape[0], self.residual_dim()],
            inputs=[self.dof_indices, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
