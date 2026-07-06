# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: stability margin (CoM inside support polygon).

Penalizes configurations where the CoM projects outside the convex hull
of the support body positions (static instability under gravity). No gradient
inside the polygon — any interior position is stable — so the objective
does not bias the solve toward the polygon centroid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp

from ._kernels import jac_fill_row

if TYPE_CHECKING:
    from ..newton_kinematics import NewtonKinematics
    from .cfg import IKObjectiveStabilityMarginCfg
    from .context import IKContactObjectiveBuildContext


@wp.kernel
def _compute_joint_subtree_coms(
    body_q: wp.array2d(dtype=wp.transform),
    body_com: wp.array1d(dtype=wp.vec3),
    body_mass: wp.array1d(dtype=wp.float32),
    subtree_bodies: wp.array1d(dtype=wp.int32),
    subtree_offsets: wp.array1d(dtype=wp.int32),
    subtree_inv_mass: wp.array1d(dtype=wp.float32),
    out_com: wp.array2d(dtype=wp.vec3),
):
    """Mass-weighted body COM positions per joint subtree [m]."""
    p, j = wp.tid()
    s = subtree_offsets[j]
    e = subtree_offsets[j + 1]
    com = wp.vec3(0.0, 0.0, 0.0)
    for i in range(s, e):
        b = subtree_bodies[i]
        com = com + body_mass[b] * wp.transform_point(body_q[p, b], body_com[b])
    out_com[p, j] = com * subtree_inv_mass[j]


@wp.kernel
def _stability_margin_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_com: wp.array1d(dtype=wp.vec3),
    body_mass: wp.array1d(dtype=wp.float32),
    n_bodies: int,
    support_body_indices: wp.array1d(dtype=wp.int32),
    is_contact: wp.array2d(dtype=wp.uint8),
    scratch_xy: wp.array2d(dtype=wp.vec2),
    scratch_slot: wp.array2d(dtype=wp.int32),
    n_supports: int,
    total_mass_inv: float,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
    signed_margin: wp.array1d(dtype=wp.float32),
    active_a_slot: wp.array1d(dtype=wp.int32),
    active_b_slot: wp.array1d(dtype=wp.int32),
    active_e_xy: wp.array1d(dtype=wp.vec2),
    active_p_xy: wp.array1d(dtype=wp.vec2),
    active_edge_len: wp.array1d(dtype=wp.float32),
):
    """Residual = ``max(0, -margin)`` where margin is the signed distance
    from the CoM (XY projection) to the nearest edge of the *active*
    support polygon. Only support bodies with ``is_contact[row, i] != 0`` form
    polygon vertices; lifted support bodies are skipped. Active contacts are sorted
    CCW per-problem by their angle around the active centroid.

    Returns zero residual and zero signed margin when fewer than three support
    bodies are in contact. The separate active-contact count disambiguates that
    sentinel when acceptance applies its minimum-contact bound.

    Also writes the active edge cache (``active_*``) used by the analytic
    Jacobian kernel: support body slot indices forming the violating edge, and
    the edge / com-vector / edge-length values that the chain rule needs.
    Sets ``active_a_slot = -1`` when residual = 0 (hinge, no gradient).
    """
    row = wp.tid()

    # Gather active support bodies xy + their original slot indices into scratch.
    n_active = int(0)
    for i in range(n_supports):
        if is_contact[row, i] != wp.uint8(0):
            pos = wp.transform_get_translation(body_q[row, support_body_indices[i]])
            scratch_xy[row, n_active] = wp.vec2(pos[0], pos[1])
            scratch_slot[row, n_active] = i
            n_active = n_active + 1
    if n_active < 3:
        residuals[row, start_idx] = 0.0
        signed_margin[row] = 0.0
        active_a_slot[row] = -1
        return

    # Mass-weighted CoM in XY (over the whole body, not just active support bodies --
    # physical CoM doesn't care about contact state).
    com_x = float(0.0)
    com_y = float(0.0)
    for b in range(n_bodies):
        pos_b = wp.transform_point(body_q[row, b], body_com[b])
        com_x = com_x + body_mass[b] * pos_b[0]
        com_y = com_y + body_mass[b] * pos_b[1]
    com_x = com_x * total_mass_inv
    com_y = com_y * total_mass_inv

    # Sort active support bodies CCW by angle around their own centroid (carrying slot ids).
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

    signed_margin[row] = min_signed
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
    support_body_indices: wp.array1d(dtype=wp.int32),
    support_in_subtree: wp.array2d(dtype=wp.uint8),
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

    * Foot velocity at each endpoint via ``v_d + ω_d × pos_support`` gated
      by the precomputed ``support_in_subtree`` membership table.
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

    a_body = support_body_indices[a_slot]
    b_body = support_body_indices[b_slot]
    pos_a = wp.transform_get_translation(body_q[p, a_body])
    pos_b = wp.transform_get_translation(body_q[p, b_body])

    S = joint_S_s[p, d]
    v = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])

    d_pos_a = wp.vec3(0.0, 0.0, 0.0)
    if support_in_subtree[a_slot, d] != wp.uint8(0):
        d_pos_a = v + wp.cross(omega, pos_a)
    d_pos_b = wp.vec3(0.0, 0.0, 0.0)
    if support_in_subtree[b_slot, d] != wp.uint8(0):
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


def stability_margin_measure(
    kinematics: NewtonKinematics,
    body_q: torch.Tensor,
    support_body_indices: tuple[int, ...],
    is_contact: torch.Tensor,
    batch_capacity: int | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the objective signed stability margin with bounded scratch [m].

    Args:
        kinematics: Owner of canonical body mass and body-local COM offsets.
        body_q: Final body poses [m, quaternion xyzw], shape [row_count, body_count, 7].
        support_body_indices: Support-body indices matching contact-mask columns.
        is_contact: Active support mask, shape [row_count, support_count].
        batch_capacity: Maximum rows held by temporary measure workspaces.
        output: Optional preallocated signed-margin output, shape [row_count].

    Returns:
        Signed nearest-edge margin [m], shape [row_count].
    """
    topology = kinematics.topology
    if (
        body_q.ndim != 3
        or body_q.shape[1:] != (topology.body_count, 7)
        or body_q.dtype is not torch.float32
        or not body_q.is_contiguous()
    ):
        raise ValueError("Stability body poses must be contiguous float32 [row_count, body_count, 7].")
    count = body_q.shape[0]
    support_count = len(support_body_indices)
    if is_contact.shape != (count, support_count) or is_contact.device != body_q.device:
        raise ValueError("Stability contact mask must match body-pose rows, support columns, and device.")
    if count == 0:
        return torch.empty(0, dtype=torch.float32, device=body_q.device)
    capacity = count if batch_capacity is None else min(count, batch_capacity)
    if capacity < 1:
        raise ValueError("Stability batch_capacity must be positive.")

    device = str(body_q.device)
    body_q_wp = wp.from_torch(body_q, dtype=wp.transformf)
    contact_u8 = torch.empty((capacity, support_count), dtype=torch.uint8, device=body_q.device)
    contact_wp = wp.from_torch(contact_u8, dtype=wp.uint8)
    support_wp = wp.array(support_body_indices, dtype=wp.int32, device=device)
    scratch_xy = wp.zeros((capacity, support_count), dtype=wp.vec2, device=device)
    scratch_slot = wp.zeros((capacity, support_count), dtype=wp.int32, device=device)
    residuals = wp.zeros((capacity, 1), dtype=wp.float32, device=device)
    if output is not None and (
        output.shape != (count,) or output.dtype is not torch.float32 or output.device != body_q.device
    ):
        raise ValueError("Stability output must be float32 [row_count] on the body-pose device.")
    margin = torch.empty(count, dtype=torch.float32, device=body_q.device) if output is None else output
    margin_wp = wp.from_torch(margin)
    active_a = wp.empty(capacity, dtype=wp.int32, device=device)
    active_b = wp.empty(capacity, dtype=wp.int32, device=device)
    active_e = wp.empty(capacity, dtype=wp.vec2, device=device)
    active_p = wp.empty(capacity, dtype=wp.vec2, device=device)
    active_length = wp.empty(capacity, dtype=wp.float32, device=device)
    body_com = wp.array(topology.body_com, dtype=wp.vec3, device=device)
    body_mass = wp.array(topology.body_mass, dtype=wp.float32, device=device)
    for start in range(0, count, capacity):
        stop = min(start + capacity, count)
        active_count = stop - start
        contact_u8[:active_count].copy_(is_contact[start:stop])
        wp.launch(
            _stability_margin_residuals,
            dim=active_count,
            inputs=[
                body_q_wp[start:stop],
                body_com,
                body_mass,
                topology.body_count,
                support_wp,
                contact_wp[:active_count],
                scratch_xy[:active_count],
                scratch_slot[:active_count],
                support_count,
                float(1.0 / topology.body_mass.sum()),
                0.0,
                0,
            ],
            outputs=[
                residuals[:active_count],
                margin_wp[start:stop],
                active_a[:active_count],
                active_b[:active_count],
                active_e[:active_count],
                active_p[:active_count],
                active_length[:active_count],
            ],
            device=device,
        )
    return margin


class IKObjectiveStabilityMargin(ik.IKObjective):
    """Hinge penalty on CoM projecting outside the support polygon.

    Residual is zero whenever the mass-weighted CoM's XY projection lies
    inside the convex hull of the support bodies, and grows linearly with distance
    when outside. This matches the physical static-balance condition
    (CoM must lie over the support polygon) without biasing the IK
    toward the polygon centroid.

    Args:
        cfg: :class:`~.cfg.IKObjectiveStabilityMarginCfg` with ``weight``.
        context: Explicit mechanics, contact-body identities, and active-contact scratch.
    """

    def __init__(
        self,
        cfg: IKObjectiveStabilityMarginCfg,
        context: IKContactObjectiveBuildContext,
    ) -> None:
        super().__init__()
        self.weight = cfg.weight
        self._support_body_indices_np = np.asarray(context.contact_body_ids, dtype=np.int32)
        self.n_supports = int(self._support_body_indices_np.shape[0])
        topology = context.kinematics.topology
        self.n_bodies = topology.body_count
        self.n_joints = topology.joint_count
        self._total_mass_inv = float(1.0 / topology.body_mass.sum())
        self._body_mass_np = topology.body_mass
        self._body_com_np = topology.body_com
        self._dof_to_joint_np = topology.dof_joint
        self._joint_subtree_bodies_np = topology.joint_subtree_bodies
        self._joint_subtree_offsets_np = topology.joint_subtree_offsets
        self._joint_subtree_mass_np = topology.joint_subtree_mass
        self._joint_subtree_inv_mass_np = topology.joint_subtree_inverse_mass
        self._support_in_subtree_np = topology.body_dof_ancestry[self._support_body_indices_np]
        self._contact_mask = context.contact_mask

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return 1

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        n = self.n_batch

        self._support_body_indices = wp.array(self._support_body_indices_np, dtype=wp.int32, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)
        self._body_com_dev = wp.array(self._body_com_np, dtype=wp.vec3, device=d)

        # The solver owner refreshes this fixed-size scratch before each
        # chunk. Warp retains a view, so chunk reuse sees the active rows.
        self._is_contact_t = self._contact_mask
        self._is_contact = wp.from_torch(self._is_contact_t, dtype=wp.uint8)

        self._scratch_xy = wp.zeros(shape=(n, self.n_supports), dtype=wp.vec2, device=d)
        self._scratch_slot = wp.zeros(shape=(n, self.n_supports), dtype=wp.int32, device=d)

        # Active-edge cache populated by the residual kernel and read by
        # the analytic Jacobian kernel on the same iteration.
        self._signed_margin = wp.zeros(shape=(n,), dtype=wp.float32, device=d)
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
        self._support_in_subtree = wp.array(self._support_in_subtree_np, dtype=wp.uint8, device=d)

        # Autodiff scratch only when the solver may take that path.
        self._e_array: wp.array | None = None
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            e = np.zeros((n, self.total_residuals), dtype=np.float32)
            for b in range(n):
                e[b, self.residual_offset] = 1.0
            self._e_array = wp.array(e.flatten(), dtype=wp.float32, device=d)

    def estimate_memory(
        self,
        model: newton.Model,
        jacobian_mode: ik.IKJacobianType,
        n_problems: int,
        n_batch: int,
        total_residuals: int,
    ) -> int:
        """Estimate immutable support facts and active-edge workspaces [byte]."""
        del model, n_problems
        fixed_bytes = sum(
            values.nbytes
            for values in (
                self._support_body_indices_np,
                self._body_mass_np,
                self._body_com_np,
                self._dof_to_joint_np,
                self._joint_subtree_bodies_np,
                self._joint_subtree_offsets_np,
                self._joint_subtree_mass_np,
                self._joint_subtree_inv_mass_np,
                self._support_in_subtree_np,
            )
        )
        vec2_bytes = wp.types.type_size_in_bytes(wp.vec2)
        int_bytes = wp.types.type_size_in_bytes(wp.int32)
        float_bytes = wp.types.type_size_in_bytes(wp.float32)
        workspace_bytes = n_batch * (
            self.n_supports * (vec2_bytes + int_bytes)
            + self.n_joints * wp.types.type_size_in_bytes(wp.vec3)
            + 3 * float_bytes
            + 2 * int_bytes
            + 2 * vec2_bytes
        )
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            workspace_bytes += n_batch * total_residuals * float_bytes
        return int(fixed_bytes + workspace_bytes)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n = body_q.shape[0]
        wp.launch(
            _compute_joint_subtree_coms,
            dim=[n, self.n_joints],
            inputs=[
                body_q,
                self._body_com_dev,
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
                self._body_com_dev,
                self._body_mass_dev,
                self.n_bodies,
                self._support_body_indices,
                self._is_contact,
                self._scratch_xy,
                self._scratch_slot,
                self.n_supports,
                self._total_mass_inv,
                self.weight,
                start_idx,
            ],
            outputs=[
                residuals,
                self._signed_margin,
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
                self._support_body_indices,
                self._support_in_subtree,
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
