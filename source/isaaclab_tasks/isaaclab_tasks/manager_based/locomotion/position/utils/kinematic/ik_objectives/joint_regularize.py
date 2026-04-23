# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: regularize a subset of joint DOFs toward per-DOF target angles."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import warp as wp


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
        joint_dof_indices: Revolute-joint DOF indices as returned by
            :meth:`NewtonKinematics.find_joint_dof_indices` -- i.e.
            indices into ``joint_q[7:]`` (after the 7 free-root
            coordinates).
        joint_dof_targets: Per-DOF target angles [rad]. Must match the
            length of ``joint_dof_indices``. Pass a scalar to use the
            same target for every listed DOF.
        weight: Uniform residual weight [unitless].
    """

    def __init__(
        self,
        joint_dof_indices: list[int],
        joint_dof_targets: float | list[float] = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not joint_dof_indices:
            raise ValueError("IKObjectiveJointRegularize requires at least one DOF index.")
        self._dof_rev_indices = list(joint_dof_indices)
        n = len(self._dof_rev_indices)
        if isinstance(joint_dof_targets, (int, float)):
            targets = [float(joint_dof_targets)] * n
        else:
            if len(joint_dof_targets) != n:
                raise ValueError(
                    f"IKObjectiveJointRegularize: joint_dof_targets has length {len(joint_dof_targets)}"
                    f" but joint_dof_indices has length {n}."
                )
            targets = [float(t) for t in joint_dof_targets]
        self._dof_targets = targets
        self.weight = float(weight)
        self.coord_indices: wp.array | None = None
        self.dof_indices: wp.array | None = None
        self.targets: wp.array | None = None

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        # ``joint_q`` has the free-root 7 coords first; ``joint_qd`` has 6 DOFs
        # first. Revolute indices supplied by ``find_joint_dof_indices`` are
        # relative to ``joint_q[7:]``, so coord = i + 7 and DOF = i + 6.
        coord_np = np.asarray([i + 7 for i in self._dof_rev_indices], dtype=np.int32)
        dof_np = np.asarray([i + 6 for i in self._dof_rev_indices], dtype=np.int32)
        targets_np = np.asarray(self._dof_targets, dtype=np.float32)
        self.coord_indices = wp.array(coord_np, dtype=wp.int32, device=self.device)
        self.dof_indices = wp.array(dof_np, dtype=wp.int32, device=self.device)
        self.targets = wp.array(targets_np, dtype=wp.float32, device=self.device)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return len(self._dof_rev_indices)

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
