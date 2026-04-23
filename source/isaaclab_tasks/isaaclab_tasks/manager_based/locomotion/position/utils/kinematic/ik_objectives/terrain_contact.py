# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: terrain surface contact."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row


@wp.kernel
def _terrain_contact_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    mesh_id: wp.uint64,
    foot_body_indices: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row, foot_idx = wp.tid()
    tf = body_q[row, foot_body_indices[foot_idx]]
    foot_pos = wp.transform_get_translation(tf)
    query = wp.mesh_query_point(mesh_id, foot_pos, 2.0)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        residuals[row, start_idx + foot_idx] = weight * wp.length(foot_pos - closest)


class IKObjectiveTerrainContact(ik.IKObjective):
    """Penalize feet not touching the terrain surface.

    Args:
        mesh_id: Warp mesh identifier.
        foot_body_indices: Newton body indices for the feet.
        weight: Residual weight.
    """

    def __init__(self, mesh_id: int, foot_body_indices: list[int], weight: float = 5.0) -> None:
        super().__init__()
        self.mesh_id = mesh_id
        self._foot_body_indices_np = np.array(foot_body_indices, dtype=np.int32)
        self.n_feet = len(foot_body_indices)
        self.weight = weight

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return self.n_feet

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)
        self._e_arrays = []
        for r in range(self.n_feet):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for b in range(self.n_batch):
                e[b, self.residual_offset + r] = 1.0
            self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=d))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _terrain_contact_residuals,
            dim=[body_q.shape[0], self.n_feet],
            inputs=[body_q, self.mesh_id, self._foot_body_indices, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_feet):
            tape.backward(grads={tape.outputs[0]: self._e_arrays[r]})
            wp.launch(
                jac_fill_row,
                dim=self.n_batch,
                inputs=[tape.gradients[dq_dof], n_dofs, start_idx + r],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()
