# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: stability margin (CoM over support centroid)."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row


@wp.kernel
def _stability_margin_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    n_bodies: int,
    active_mask: wp.array2d(dtype=wp.int32),
    foot_body_indices: wp.array1d(dtype=wp.int32),
    n_feet: int,
    total_mass_inv: float,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    com_x, com_y = float(0.0), float(0.0)
    for b in range(n_bodies):
        pos = wp.transform_get_translation(body_q[row, b])
        com_x = com_x + body_mass[b] * pos[0]
        com_y = com_y + body_mass[b] * pos[1]
    com_x, com_y = com_x * total_mass_inv, com_y * total_mass_inv

    cx, cy, n_active = float(0.0), float(0.0), float(0.0)
    for f in range(n_feet):
        if active_mask[row, f] > 0:
            fp = wp.transform_get_translation(body_q[row, foot_body_indices[f]])
            cx, cy = cx + fp[0], cy + fp[1]
            n_active = n_active + 1.0
    if n_active > 0.0:
        cx, cy = cx / n_active, cy / n_active

    dx, dy = com_x - cx, com_y - cy
    residuals[row, start_idx] = weight * wp.sqrt(dx * dx + dy * dy + 1.0e-8)


class IKObjectiveStabilityMargin(ik.IKObjective):
    """Center the CoM over the support polygon for static stability.

    Args:
        model: Newton model (for body masses).
        foot_body_indices: Newton body indices for the feet.
        active_mask: Per-candidate active foot mask ``[n_problems, nc]``.
        weight: Residual weight.
    """

    def __init__(self, model: newton.Model, foot_body_indices: list[int],
                 active_mask: wp.array, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self._foot_body_indices_np = np.array(foot_body_indices, dtype=np.int32)
        self.n_feet = len(foot_body_indices)
        self.n_bodies = model.body_count
        bm = model.body_mass.numpy()
        self._total_mass_inv = float(1.0 / (bm.sum() + 1e-10))
        self._body_mass_np = bm.astype(np.float32)
        self.active_mask = active_mask

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return 1

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)
        e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
        for b in range(self.n_batch):
            e[b, self.residual_offset] = 1.0
        self._e_array = wp.array(e.flatten(), dtype=wp.float32, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(_stability_margin_residuals, dim=body_q.shape[0],
                  inputs=[body_q, self._body_mass_dev, self.n_bodies, self.active_mask,
                          self._foot_body_indices, self.n_feet, self._total_mass_inv,
                          self.weight, start_idx],
                  outputs=[residuals], device=self.device)

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self._e_array})
        wp.launch(jac_fill_row, dim=self.n_batch,
                  inputs=[tape.gradients[dq_dof], dq_dof.shape[1], start_idx],
                  outputs=[jacobian], device=self.device)
        tape.zero()
