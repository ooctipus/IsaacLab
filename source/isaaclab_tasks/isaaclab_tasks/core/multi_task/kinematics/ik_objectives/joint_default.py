# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: joint default regularization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .cfg import IKObjectiveJointDefaultCfg
    from .context import IKObjectiveBuildContext


@wp.kernel
def _default_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array1d(dtype=wp.int32),
    targets: wp.array1d(dtype=wp.float32),
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, index = wp.tid()
    coordinate = coordinate_indices[index]
    residuals[problem, start_idx + index] = weight * (joint_q[problem, coordinate] - targets[index])


@wp.kernel
def _default_jac_analytic(
    velocity_indices: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, index = wp.tid()
    jacobian[problem, start_idx + index, velocity_indices[index]] = weight


class IKObjectiveJointDefault(ik.IKObjective):
    """Penalize deviation from the robot's default joint configuration.

    Each (non-root) DOF contributes one residual
    ``weight * (q - q_default)`` where ``q_default`` is taken from
    :attr:`~isaaclab_tasks.core.multi_task.kinematics.NewtonKinematics.default_joint_q`.
    The analytic Jacobian is a constant diagonal of ``weight``.

    Args:
        cfg: :class:`~.cfg.IKObjectiveJointDefaultCfg` with ``weight`` and
            ``skip_root``.
        context: Explicit shared kinematics.
    """

    def __init__(
        self,
        cfg: IKObjectiveJointDefaultCfg,
        context: IKObjectiveBuildContext,
    ) -> None:
        super().__init__()
        self.weight = float(cfg.weight)
        default_joint_q = np.asarray(context.kinematics.default_joint_q, dtype=np.float32)
        topology = context.kinematics.topology
        q_start = topology.joint_q_start
        qd_start = topology.joint_qd_start
        coordinate_indices: list[int] = []
        velocity_indices: list[int] = []
        first_joint = 1 if cfg.skip_root and context.kinematics.n_root_coords else 0
        for joint_index in range(first_joint, topology.joint_count):
            q_begin, q_end = int(q_start[joint_index]), int(q_start[joint_index + 1])
            qd_begin, qd_end = int(qd_start[joint_index]), int(qd_start[joint_index + 1])
            q_width = q_end - q_begin
            qd_width = qd_end - qd_begin
            if q_width == qd_width:
                coordinate_indices.extend(range(q_begin, q_end))
                velocity_indices.extend(range(qd_begin, qd_end))
                continue
            joint_name = context.kinematics.joint_names[joint_index]
            raise ValueError(
                f"IKObjectiveJointDefault cannot regularize joint {joint_name!r}: "
                f"its {q_width} position coordinates do not map one-to-one to {qd_width} velocity coordinates."
            )
        self._coordinate_indices_np = np.asarray(coordinate_indices, dtype=np.int32)
        self._velocity_indices_np = np.asarray(velocity_indices, dtype=np.int32)
        self._targets_np = default_joint_q[self._coordinate_indices_np]
        self.coordinate_indices: wp.array | None = None
        self.velocity_indices: wp.array | None = None
        self.targets: wp.array | None = None

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        self.coordinate_indices = wp.array(self._coordinate_indices_np, dtype=wp.int32, device=self.device)
        self.velocity_indices = wp.array(self._velocity_indices_np, dtype=wp.int32, device=self.device)
        self.targets = wp.array(self._targets_np, dtype=wp.float32, device=self.device)

    def estimate_memory(
        self,
        model: newton.Model,
        jacobian_mode: ik.IKJacobianType,
        n_problems: int,
        n_batch: int,
        total_residuals: int,
    ) -> int:
        """Estimate immutable coordinate, velocity, and default-target arrays [byte]."""
        del model, jacobian_mode, n_problems, n_batch, total_residuals
        count = len(self._coordinate_indices_np)
        int_bytes = wp.types.type_size_in_bytes(wp.int32)
        float_bytes = wp.types.type_size_in_bytes(wp.float32)
        return count * (2 * int_bytes + float_bytes)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return len(self._coordinate_indices_np)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _default_residuals,
            dim=[joint_q.shape[0], self.residual_dim()],
            inputs=[joint_q, self.coordinate_indices, self.targets, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        wp.launch(
            _default_jac_analytic,
            dim=[joint_q.shape[0], self.residual_dim()],
            inputs=[self.velocity_indices, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
