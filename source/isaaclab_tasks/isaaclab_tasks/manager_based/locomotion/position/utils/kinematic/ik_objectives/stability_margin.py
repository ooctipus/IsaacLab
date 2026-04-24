# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: stability margin (CoM inside support polygon).

Penalizes configurations where the CoM projects outside the convex hull
of the foot positions (static instability under gravity). No gradient
inside the polygon — any interior position is stable — so the objective
does not bias the solve toward the polygon centroid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row

if TYPE_CHECKING:
    from ...mdp.retarget.pipeline import RetargetPipeline
    from .cfg import IKObjectiveStabilityMarginCfg


@wp.kernel
def _stability_margin_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    n_bodies: int,
    foot_body_indices: wp.array1d(dtype=wp.int32),
    is_contact: wp.array2d(dtype=wp.uint8),
    scratch_xy: wp.array2d(dtype=wp.vec2),
    n_feet: int,
    total_mass_inv: float,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Residual = ``max(0, -margin)`` where margin is the signed distance
    from the CoM (XY projection) to the nearest edge of the *active*
    support polygon. Only feet with ``is_contact[row, i] != 0`` form
    polygon vertices; lifted feet are skipped so tripod/biped placements
    compute the CoM constraint against their actual contact set. The
    active contacts are sorted CCW per-problem by their angle around
    the active centroid so any contact subset yields a valid polygon
    traversal.

    Returns zero residual when fewer than 3 feet are in contact (support
    collapses to a point or segment -- measure-zero stability, left to
    the :class:`SupportPolygonStability` criterion to gate rigorously).

    Args:
        foot_body_indices: Foot body ids in any order; ``is_contact`` is
            indexed matching this order (slot order, NOT CCW).
        is_contact: Per-problem per-slot contact flag (uint8).
        scratch_xy: Per-problem workspace ``[N, n_feet]`` of ``vec2``.
            Used to materialize active-contact xy, then sorted in-place.
    """
    row = wp.tid()

    # Gather active feet xy into scratch (packed into slots 0..n_active).
    n_active = int(0)
    for i in range(n_feet):
        if is_contact[row, i] != wp.uint8(0):
            pos = wp.transform_get_translation(body_q[row, foot_body_indices[i]])
            scratch_xy[row, n_active] = wp.vec2(pos[0], pos[1])
            n_active = n_active + 1
    if n_active < 3:
        residuals[row, start_idx] = 0.0
        return

    # Mass-weighted CoM in XY (over the whole body, not just active feet
    # -- physical CoM doesn't care about contact state).
    com_x, com_y = float(0.0), float(0.0)
    for b in range(n_bodies):
        pos = wp.transform_get_translation(body_q[row, b])
        com_x = com_x + body_mass[b] * pos[0]
        com_y = com_y + body_mass[b] * pos[1]
    com_x = com_x * total_mass_inv
    com_y = com_y * total_mass_inv

    # Sort active feet CCW by angle around their own centroid.
    cx = float(0.0)
    cy = float(0.0)
    for k in range(n_active):
        v = scratch_xy[row, k]
        cx = cx + v[0]
        cy = cy + v[1]
    inv_n = 1.0 / float(n_active)
    cx = cx * inv_n
    cy = cy * inv_n
    for i in range(n_active):
        for j in range(i + 1, n_active):
            vi = scratch_xy[row, i]
            vj = scratch_xy[row, j]
            ai = wp.atan2(vi[1] - cy, vi[0] - cx)
            aj = wp.atan2(vj[1] - cy, vj[0] - cx)
            if aj < ai:
                scratch_xy[row, i] = vj
                scratch_xy[row, j] = vi

    # Minimum signed distance to any edge. CCW ordering → positive inside.
    min_signed = float(1.0e9)
    for i in range(n_active):
        j = (i + 1) % n_active
        vi = scratch_xy[row, i]
        vj = scratch_xy[row, j]
        ex = vj[0] - vi[0]
        ey = vj[1] - vi[1]
        edge_len = wp.sqrt(ex * ex + ey * ey + 1.0e-12)
        px = com_x - vi[0]
        py = com_y - vi[1]
        signed = (ex * py - ey * px) / edge_len
        if signed < min_signed:
            min_signed = signed

    violation = wp.max(0.0, -min_signed)
    residuals[row, start_idx] = weight * violation


class IKObjectiveStabilityMargin(ik.IKObjective):
    """Hinge penalty on CoM projecting outside the support polygon.

    Residual is zero whenever the mass-weighted CoM's XY projection lies
    inside the convex hull of the feet, and grows linearly with distance
    when outside. This matches the physical static-balance condition
    (CoM must lie over the support polygon) without biasing the IK
    toward the polygon centroid.

    Args:
        cfg: :class:`~.cfg.IKObjectiveStabilityMarginCfg` with ``weight``.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``kin.model`` (body masses), ``foot_body_ids``, and
            ``sampler._foot_ccw_order`` (CCW re-ordering).
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(
        self,
        cfg: IKObjectiveStabilityMarginCfg,
        pipeline: RetargetPipeline,
        wp_mesh: object = None,
    ) -> None:
        super().__init__()
        self.weight = cfg.weight
        # Slot-ordered foot body ids (no assumption about CCW ordering --
        # the kernel sorts active contacts by angle per-problem).
        self._foot_body_indices_np = np.asarray(pipeline.foot_body_ids, dtype=np.int32)
        self.n_feet = int(self._foot_body_indices_np.shape[0])
        model = pipeline.kin.model
        self.n_bodies = model.body_count
        bm = model.body_mass.numpy()
        self._total_mass_inv = float(1.0 / (bm.sum() + 1e-10))
        self._body_mass_np = bm.astype(np.float32)
        # Stash pipeline for ``is_contact`` read-out in init_buffers
        # (called after the sampler has populated the buffer).
        self._pipeline = pipeline

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

        # Snapshot ``is_contact`` per problem from the populated buffer.
        # The sampler ran before the IK objectives were built, so buffer
        # contents are stable. Slot order matches ``foot_body_indices``
        # (no CCW reindex -- kernel sorts active contacts per-problem).
        import torch  # local import to avoid top-level torch dep

        buf = self._pipeline.buffer
        n = self.n_batch
        is_c_u8 = buf.is_contact_t[: n * self.n_feet].view(n, self.n_feet).to(torch.uint8).contiguous()
        self._is_contact_t = is_c_u8  # keep torch reference alive for Warp view
        self._is_contact = wp.from_torch(is_c_u8, dtype=wp.uint8)

        # Per-problem workspace for sorting active contact xy CCW in-kernel.
        self._scratch_xy = wp.zeros(shape=(n, self.n_feet), dtype=wp.vec2, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _stability_margin_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self._body_mass_dev,
                self.n_bodies,
                self._foot_body_indices,
                self._is_contact,
                self._scratch_xy,
                self.n_feet,
                self._total_mass_inv,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self._e_array})
        wp.launch(
            jac_fill_row,
            dim=self.n_batch,
            inputs=[tape.gradients[dq_dof], dq_dof.shape[1], start_idx],
            outputs=[jacobian],
            device=self.device,
        )
        tape.zero()
