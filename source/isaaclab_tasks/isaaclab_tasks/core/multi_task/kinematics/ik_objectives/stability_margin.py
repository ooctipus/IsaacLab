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
    from isaaclab_tasks.core.multi_task.terrain.retarget.pipeline import RetargetPipeline

    from .cfg import IKObjectiveStabilityMarginCfg


def _build_stability_jac_relations(model: newton.Model, foot_body_ids: list[int]) -> dict:
    """Precompute tables used by the analytic Jacobian kernel.

    For each dof, we need the joint that owns it (so we can index into the
    per-joint subtree COM cache). For each joint, we need its subtree
    bodies (for the per-iteration COM compute) and total mass. For each
    foot, we need a (foot, dof) bitmap saying whether motion of that dof
    moves that foot.
    """
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    bm = model.body_mass.numpy()
    qd_start = model.joint_qd_start.numpy()
    n_joints = model.joint_count
    n_bodies = model.body_count
    n_dofs = model.joint_dof_count

    dof_to_joint = np.zeros(n_dofs, dtype=np.int32)
    for jg in range(n_joints):
        end = qd_start[jg + 1] if jg + 1 < len(qd_start) else n_dofs
        for d in range(int(qd_start[jg]), int(end)):
            dof_to_joint[d] = jg

    children: dict[int, list[int]] = {b: [] for b in range(-1, n_bodies)}
    for jg in range(n_joints):
        children[int(jp[jg])].append(int(jc[jg]))

    def _subtree(child_body: int) -> list[int]:
        if child_body < 0:
            return []
        out, queue = [child_body], [child_body]
        while queue:
            x = queue.pop()
            for ch in children[x]:
                out.append(ch)
                queue.append(ch)
        return out

    flat, offsets, masses = [], [0], []
    subtree_set: list[set[int]] = []
    for jg in range(n_joints):
        bodies = _subtree(int(jc[jg]))
        flat.extend(bodies)
        offsets.append(len(flat))
        masses.append(float(sum(bm[b] for b in bodies)))
        subtree_set.append(set(bodies))

    foot_in_subtree = np.zeros((len(foot_body_ids), n_dofs), dtype=np.uint8)
    for fi, f_body in enumerate(foot_body_ids):
        for d in range(n_dofs):
            if int(f_body) in subtree_set[int(dof_to_joint[d])]:
                foot_in_subtree[fi, d] = 1

    return {
        "dof_to_joint": dof_to_joint,
        "joint_subtree_bodies": np.array(flat, dtype=np.int32),
        "joint_subtree_offsets": np.array(offsets, dtype=np.int32),
        "joint_subtree_mass": np.array(masses, dtype=np.float32),
        "foot_in_subtree": foot_in_subtree,
    }


@wp.kernel
def _compute_joint_subtree_origin_coms(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    subtree_bodies: wp.array1d(dtype=wp.int32),
    subtree_offsets: wp.array1d(dtype=wp.int32),
    subtree_inv_mass: wp.array1d(dtype=wp.float32),
    out_com: wp.array2d(dtype=wp.vec3),
):
    """Mass-weighted average of body **origin** positions per joint subtree.

    Matches the residual kernel's CoM aggregation, which weights body
    origin positions (not body-COM offsets) by mass. Using origin keeps
    the residual and Jacobian consistent so analytic = derivative of what
    the residual actually computes.
    """
    p, j = wp.tid()
    s = subtree_offsets[j]
    e = subtree_offsets[j + 1]
    com = wp.vec3(0.0, 0.0, 0.0)
    for i in range(s, e):
        b = subtree_bodies[i]
        com = com + body_mass[b] * wp.transform_get_translation(body_q[p, b])
    out_com[p, j] = com * subtree_inv_mass[j]


@wp.kernel
def _stability_margin_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    n_bodies: int,
    foot_body_indices: wp.array1d(dtype=wp.int32),
    is_contact: wp.array2d(dtype=wp.uint8),
    scratch_xy: wp.array2d(dtype=wp.vec2),
    scratch_slot: wp.array2d(dtype=wp.int32),
    n_feet: int,
    total_mass_inv: float,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
    active_a_slot: wp.array1d(dtype=wp.int32),
    active_b_slot: wp.array1d(dtype=wp.int32),
    active_e_xy: wp.array1d(dtype=wp.vec2),
    active_p_xy: wp.array1d(dtype=wp.vec2),
    active_edge_len: wp.array1d(dtype=wp.float32),
):
    """Residual = ``max(0, -margin)`` where margin is the signed distance
    from the CoM (XY projection) to the nearest edge of the *active*
    support polygon. Only feet with ``is_contact[row, i] != 0`` form
    polygon vertices; lifted feet are skipped. Active contacts are sorted
    CCW per-problem by their angle around the active centroid.

    Returns zero residual when fewer than 3 feet are in contact (support
    collapses to a point or segment -- measure-zero stability, left to
    the :class:`SupportPolygonStability` criterion to gate rigorously).

    Also writes the active edge cache (``active_*``) used by the analytic
    Jacobian kernel: foot slot indices forming the violating edge, and
    the edge / com-vector / edge-length values that the chain rule needs.
    Sets ``active_a_slot = -1`` when residual = 0 (hinge, no gradient).
    """
    row = wp.tid()

    # Gather active feet xy + their original slot indices into scratch.
    n_active = int(0)
    for i in range(n_feet):
        if is_contact[row, i] != wp.uint8(0):
            pos = wp.transform_get_translation(body_q[row, foot_body_indices[i]])
            scratch_xy[row, n_active] = wp.vec2(pos[0], pos[1])
            scratch_slot[row, n_active] = i
            n_active = n_active + 1
    if n_active < 3:
        residuals[row, start_idx] = 0.0
        active_a_slot[row] = -1
        return

    # Mass-weighted CoM in XY (over the whole body, not just active feet --
    # physical CoM doesn't care about contact state).
    com_x = float(0.0)
    com_y = float(0.0)
    for b in range(n_bodies):
        pos_b = wp.transform_get_translation(body_q[row, b])
        com_x = com_x + body_mass[b] * pos_b[0]
        com_y = com_y + body_mass[b] * pos_b[1]
    com_x = com_x * total_mass_inv
    com_y = com_y * total_mass_inv

    # Sort active feet CCW by angle around their own centroid (carrying slot ids).
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
                si = scratch_slot[row, i]
                sj = scratch_slot[row, j]
                scratch_slot[row, i] = sj
                scratch_slot[row, j] = si

    # Find the minimum signed distance and remember which edge produced it.
    min_signed = float(1.0e9)
    active_i = int(0)
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
            active_i = i

    violation = wp.max(0.0, -min_signed)
    residuals[row, start_idx] = weight * violation

    if violation > 0.0:
        i_a = active_i
        i_b = (active_i + 1) % n_active
        vi = scratch_xy[row, i_a]
        vj = scratch_xy[row, i_b]
        ex = vj[0] - vi[0]
        ey = vj[1] - vi[1]
        L = wp.sqrt(ex * ex + ey * ey + 1.0e-12)
        active_a_slot[row] = scratch_slot[row, i_a]
        active_b_slot[row] = scratch_slot[row, i_b]
        active_e_xy[row] = wp.vec2(ex, ey)
        active_p_xy[row] = wp.vec2(com_x - vi[0], com_y - vi[1])
        active_edge_len[row] = L
    else:
        active_a_slot[row] = -1


@wp.kernel
def _stability_margin_jac_analytic(
    body_q: wp.array2d(dtype=wp.transform),
    foot_body_indices: wp.array1d(dtype=wp.int32),
    foot_in_subtree: wp.array2d(dtype=wp.uint8),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    dof_to_joint: wp.array1d(dtype=wp.int32),
    joint_subtree_mass: wp.array1d(dtype=wp.float32),
    joint_subtree_com: wp.array2d(dtype=wp.vec3),
    total_mass_inv: float,
    active_a_slot: wp.array1d(dtype=wp.int32),
    active_b_slot: wp.array1d(dtype=wp.int32),
    active_e_xy: wp.array1d(dtype=wp.vec2),
    active_p_xy: wp.array1d(dtype=wp.vec2),
    active_edge_len: wp.array1d(dtype=wp.float32),
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """One thread per ``(problem, dof)``. Hinge gradient is 0 inside the
    polygon (encoded by ``active_a_slot < 0``). When outside, only the
    active edge contributes; the chain rule combines:

    * Foot velocity at each endpoint via ``v_d + ω_d × pos_foot`` gated
      by the precomputed ``foot_in_subtree`` membership table.
    * CoM velocity via ``(M_s / M_total) * (v_d + ω_d × C_s)`` where
      ``M_s`` and ``C_s`` are the total mass and COM of the subtree of
      the dof's joint.

    With ``e = pos_b - pos_a`` and ``p = com - pos_a`` (xy only),

        s = e_x p_y - e_y p_x,    L = |e|,    signed = s / L
        residual = -weight * signed   (when active)

    so

        d(residual)/dq = -weight * (ds/dq * L - s * dL/dq) / L^2.

    Assumes ``jacobian`` is zeroed upstream. Returns without writing for
    inactive (problem, dof) pairs.
    """
    p, d = wp.tid()

    a_slot = active_a_slot[p]
    if a_slot < 0:
        return

    b_slot = active_b_slot[p]
    e_xy = active_e_xy[p]
    p_xy = active_p_xy[p]
    L = active_edge_len[p]

    a_body = foot_body_indices[a_slot]
    b_body = foot_body_indices[b_slot]
    pos_a = wp.transform_get_translation(body_q[p, a_body])
    pos_b = wp.transform_get_translation(body_q[p, b_body])

    S = joint_S_s[p, d]
    v = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])

    d_pos_a = wp.vec3(0.0, 0.0, 0.0)
    if foot_in_subtree[a_slot, d] != wp.uint8(0):
        d_pos_a = v + wp.cross(omega, pos_a)
    d_pos_b = wp.vec3(0.0, 0.0, 0.0)
    if foot_in_subtree[b_slot, d] != wp.uint8(0):
        d_pos_b = v + wp.cross(omega, pos_b)

    j_d = dof_to_joint[d]
    M_s = joint_subtree_mass[j_d]
    C_s = joint_subtree_com[p, j_d]
    d_com = (M_s * total_mass_inv) * (v + wp.cross(omega, C_s))

    d_ex = d_pos_b[0] - d_pos_a[0]
    d_ey = d_pos_b[1] - d_pos_a[1]
    d_px = d_com[0] - d_pos_a[0]
    d_py = d_com[1] - d_pos_a[1]

    ex = e_xy[0]
    ey = e_xy[1]
    px = p_xy[0]
    py = p_xy[1]

    d_s = d_ex * py + ex * d_py - d_ey * px - ey * d_px
    d_L = (ex * d_ex + ey * d_ey) / L
    s = ex * py - ey * px
    d_signed = (d_s * L - s * d_L) / (L * L)

    jacobian[p, start_idx, d] = -weight * d_signed


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
            ``kin.model`` (body masses, kinematic tree), ``foot_body_ids``
            (slot order), and ``buffer.is_contact_t`` (per-problem active
            contacts snapshotted at IK build time).
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
        self._foot_body_indices_np = np.asarray(pipeline.foot_body_ids, dtype=np.int32)
        self.n_feet = int(self._foot_body_indices_np.shape[0])
        model = pipeline.kin.model
        self.n_bodies = model.body_count
        self.n_joints = model.joint_count
        bm = model.body_mass.numpy()
        self._total_mass_inv = float(1.0 / (bm.sum() + 1e-10))
        self._body_mass_np = bm.astype(np.float32)
        rel = _build_stability_jac_relations(model, list(pipeline.foot_body_ids))
        self._dof_to_joint_np = rel["dof_to_joint"]
        self._joint_subtree_bodies_np = rel["joint_subtree_bodies"]
        self._joint_subtree_offsets_np = rel["joint_subtree_offsets"]
        self._joint_subtree_mass_np = rel["joint_subtree_mass"]
        self._joint_subtree_inv_mass_np = (1.0 / (self._joint_subtree_mass_np + 1e-10)).astype(np.float32)
        self._foot_in_subtree_np = rel["foot_in_subtree"]
        self._pipeline = pipeline

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return 1

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        n = self.n_batch

        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)

        # Snapshot ``is_contact`` per problem from the populated buffer.
        # The sampler ran before IK objectives were built, so buffer
        # contents are stable. Slot order matches ``foot_body_indices``.
        import torch  # local import to avoid top-level torch dep

        buf = self._pipeline.buffer
        is_c_u8 = buf.is_contact_t[: n * self.n_feet].view(n, self.n_feet).to(torch.uint8).contiguous()
        self._is_contact_t = is_c_u8  # keep torch reference alive for Warp view
        self._is_contact = wp.from_torch(is_c_u8, dtype=wp.uint8)

        self._scratch_xy = wp.zeros(shape=(n, self.n_feet), dtype=wp.vec2, device=d)
        self._scratch_slot = wp.zeros(shape=(n, self.n_feet), dtype=wp.int32, device=d)

        # Active-edge cache populated by the residual kernel and read by
        # the analytic Jacobian kernel on the same iteration.
        self._active_a_slot = wp.zeros(shape=(n,), dtype=wp.int32, device=d)
        self._active_b_slot = wp.zeros(shape=(n,), dtype=wp.int32, device=d)
        self._active_e_xy = wp.zeros(shape=(n,), dtype=wp.vec2, device=d)
        self._active_p_xy = wp.zeros(shape=(n,), dtype=wp.vec2, device=d)
        self._active_edge_len = wp.zeros(shape=(n,), dtype=wp.float32, device=d)

        # Per-joint subtree COM (refreshed every iteration in compute_residuals).
        self._dof_to_joint = wp.array(self._dof_to_joint_np, dtype=wp.int32, device=d)
        self._joint_subtree_bodies = wp.array(self._joint_subtree_bodies_np, dtype=wp.int32, device=d)
        self._joint_subtree_offsets = wp.array(self._joint_subtree_offsets_np, dtype=wp.int32, device=d)
        self._joint_subtree_mass = wp.array(self._joint_subtree_mass_np, dtype=wp.float32, device=d)
        self._joint_subtree_inv_mass = wp.array(self._joint_subtree_inv_mass_np, dtype=wp.float32, device=d)
        self._joint_subtree_com_buf = wp.zeros(shape=(n, self.n_joints), dtype=wp.vec3, device=d)
        self._foot_in_subtree = wp.array(self._foot_in_subtree_np, dtype=wp.uint8, device=d)

        # Autodiff scratch only when the solver may take that path.
        self._e_array: wp.array | None = None
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            e = np.zeros((n, self.total_residuals), dtype=np.float32)
            for b in range(n):
                e[b, self.residual_offset] = 1.0
            self._e_array = wp.array(e.flatten(), dtype=wp.float32, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n = body_q.shape[0]
        wp.launch(
            _compute_joint_subtree_origin_coms,
            dim=[n, self.n_joints],
            inputs=[
                body_q,
                self._body_mass_dev,
                self._joint_subtree_bodies,
                self._joint_subtree_offsets,
                self._joint_subtree_inv_mass,
            ],
            outputs=[self._joint_subtree_com_buf],
            device=self.device,
        )
        wp.launch(
            _stability_margin_residuals,
            dim=n,
            inputs=[
                body_q,
                self._body_mass_dev,
                self.n_bodies,
                self._foot_body_indices,
                self._is_contact,
                self._scratch_xy,
                self._scratch_slot,
                self.n_feet,
                self._total_mass_inv,
                self.weight,
                start_idx,
            ],
            outputs=[
                residuals,
                self._active_a_slot,
                self._active_b_slot,
                self._active_e_xy,
                self._active_p_xy,
                self._active_edge_len,
            ],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """Reads the active-edge cache populated by the most recent
        :meth:`compute_residuals` call. Newton's IK solver always evaluates
        residuals before Jacobians per iteration, so the cache is fresh.
        """
        self._require_batch_layout()
        n_dofs = model.joint_dof_count
        wp.launch(
            _stability_margin_jac_analytic,
            dim=[self.n_batch, n_dofs],
            inputs=[
                body_q,
                self._foot_body_indices,
                self._foot_in_subtree,
                joint_S_s,
                self._dof_to_joint,
                self._joint_subtree_mass,
                self._joint_subtree_com_buf,
                self._total_mass_inv,
                self._active_a_slot,
                self._active_b_slot,
                self._active_e_xy,
                self._active_p_xy,
                self._active_edge_len,
                self.weight,
                start_idx,
            ],
            outputs=[jacobian],
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
