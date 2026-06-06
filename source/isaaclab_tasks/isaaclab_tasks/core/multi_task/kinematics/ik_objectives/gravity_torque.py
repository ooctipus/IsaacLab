# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: gravity torque minimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row

if TYPE_CHECKING:
    from isaaclab_tasks.core.multi_task.terrain.retarget.pipeline import RetargetPipeline

    from .cfg import IKObjectiveGravityTorqueCfg


def _build_subtree_info(model: newton.Model) -> dict:
    """Extract revolute-joint subtree structure from a Newton Model."""
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    ja = model.joint_axis.numpy()
    bm = model.body_mass.numpy()
    n_joints = model.joint_count
    n_bodies = model.body_count

    children: dict[int, list[int]] = {i: [] for i in range(-1, n_bodies)}
    for j in range(n_joints):
        children[int(jp[j])].append(int(jc[j]))

    def _get_subtree(root_body: int) -> list[int]:
        result, queue = [root_body], [root_body]
        while queue:
            b = queue.pop(0)
            for ch in children[b]:
                result.append(ch)
                queue.append(ch)
        return result

    rev_parent_bodies, rev_axes, all_downstream = [], [], []
    for j in range(n_joints):
        if int(jt[j]) != 1:
            continue
        rev_parent_bodies.append(int(jp[j]))
        rev_axes.append(ja[j].tolist() if j < len(ja) else [1, 0, 0])
        all_downstream.append(_get_subtree(int(jc[j])))

    flat, offsets, masses = [], [0], []
    for ds in all_downstream:
        flat.extend(ds)
        offsets.append(len(flat))
        masses.append(float(sum(bm[b] for b in ds)))

    return {
        "n_rev": len(rev_parent_bodies),
        "parent_bodies": np.array(rev_parent_bodies, dtype=np.int32),
        "axes_local": np.array(rev_axes, dtype=np.float32),
        "downstream_bodies": np.array(flat, dtype=np.int32),
        "downstream_offsets": np.array(offsets, dtype=np.int32),
        "subtree_mass": np.array(masses, dtype=np.float32),
    }


@wp.kernel
def _compute_subtree_com(
    body_q: wp.array2d(dtype=wp.transform),
    body_com: wp.array1d(dtype=wp.vec3),
    body_mass: wp.array1d(dtype=wp.float32),
    downstream_bodies: wp.array1d(dtype=wp.int32),
    downstream_offsets: wp.array1d(dtype=wp.int32),
    subtree_inv_mass: wp.array1d(dtype=wp.float32),
    subtree_com_out: wp.array2d(dtype=wp.vec3),
):
    row, jidx = wp.tid()
    start = downstream_offsets[jidx]
    end_val = downstream_offsets[jidx + 1]
    com = wp.vec3(0.0, 0.0, 0.0)
    for i in range(start, end_val):
        bid = downstream_bodies[i]
        com = com + body_mass[bid] * wp.transform_point(body_q[row, bid], body_com[bid])
    subtree_com_out[row, jidx] = com * subtree_inv_mass[jidx]


def _build_jac_relations(model: newton.Model, parent_bodies_obj: np.ndarray) -> dict:
    """Precompute per ``(objective_joint j_obj, global dof d)`` relation tables
    for the analytic Jacobian.

    Each entry classifies how dof ``d`` affects revolute joint ``j_obj``'s
    residual:

    * ``code = 0``: unrelated — Jacobian entry is zero.
    * ``code = 1``: ``d`` is upstream of ``j_obj`` — entire subtree of ``j_obj``
      moves rigidly under unit motion of ``d``. ``dr/dq_d = ω_d × r``,
      ``da/dq_d = ω_d × a``.
    * ``code = 2``: ``d``'s joint sits inside subtree of ``j_obj`` (or
      ``d`` is ``j_obj``'s own dof). Only bodies downstream of ``d`` move,
      so ``dp_j/dq_d = 0``, ``da/dq_d = 0``, and
      ``dc_j/dq_d = ratio * (v_d + ω_d × c_d_sub)`` where
      ``ratio = m_subtree(d) / m_subtree(j_obj)`` and ``c_d_sub`` is
      ``d``'s subtree COM (looked up via ``c_idx``).

    Tables are returned as numpy arrays of shape ``(n_rev, n_dofs)`` so the
    kernel can index them directly.
    """
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    bm = model.body_mass.numpy()
    qd_start = model.joint_qd_start.numpy()
    n_joints = model.joint_count
    n_bodies = model.body_count
    n_dofs = model.joint_dof_count
    n_rev = len(parent_bodies_obj)

    # Map dof -> global joint.
    dof_to_joint = np.zeros(n_dofs, dtype=np.int32)
    for jg in range(n_joints):
        dof_end = qd_start[jg + 1] if jg + 1 < len(qd_start) else n_dofs
        for dq in range(int(qd_start[jg]), int(dof_end)):
            dof_to_joint[dq] = jg

    # Map global revolute joint -> objective joint index (preserving the
    # iteration order ``_build_subtree_info`` used to enumerate them).
    obj_of_global: dict[int, int] = {}
    obj_idx = 0
    for jg in range(n_joints):
        if int(jt[jg]) == 1:
            obj_of_global[jg] = obj_idx
            obj_idx += 1

    # Per-global-joint subtree (set of bodies downstream of joint's child)
    # and total mass — needed for both the body-membership check and the
    # case-2 ratio.
    children: dict[int, list[int]] = {b: [] for b in range(-1, n_bodies)}
    for jg in range(n_joints):
        children[int(jp[jg])].append(int(jc[jg]))

    def _subtree_bodies(child_body: int) -> set[int]:
        if child_body < 0:
            return set()
        out, q = {child_body}, [child_body]
        while q:
            x = q.pop()
            for ch in children[x]:
                out.add(ch)
                q.append(ch)
        return out

    subtree_set: dict[int, set[int]] = {}
    subtree_mass: dict[int, float] = {}
    for jg in range(n_joints):
        sset = _subtree_bodies(int(jc[jg]))
        subtree_set[jg] = sset
        subtree_mass[jg] = float(sum(bm[b] for b in sset))

    # Identify each j_obj's global joint index.
    g_of_obj: list[int] = []
    for jg in range(n_joints):
        if int(jt[jg]) == 1:
            g_of_obj.append(jg)
    assert len(g_of_obj) == n_rev, "_build_jac_relations: revolute count mismatch"

    code = np.zeros((n_rev, n_dofs), dtype=np.uint8)
    ratio = np.zeros((n_rev, n_dofs), dtype=np.float32)
    c_idx = np.zeros((n_rev, n_dofs), dtype=np.int32)

    for j_obj in range(n_rev):
        g_j = g_of_obj[j_obj]
        j_subtree = subtree_set[g_j]
        m_j = subtree_mass[g_j]
        j_parent_body = int(parent_bodies_obj[j_obj])
        for d in range(n_dofs):
            d_joint = int(dof_to_joint[d])
            d_child = int(jc[d_joint])
            if d_child in j_subtree:
                # Case 2: d's joint is in j_obj's subtree (or == j_obj itself).
                if d_joint not in obj_of_global:
                    # d's joint is non-revolute (fixed/free) inside subtree —
                    # rare in practice; treat as 0 to be safe.
                    continue
                code[j_obj, d] = 2
                c_idx[j_obj, d] = obj_of_global[d_joint]
                ratio[j_obj, d] = subtree_mass[d_joint] / m_j if m_j > 0 else 0.0
            elif j_parent_body in subtree_set.get(d_joint, set()):
                # Case 1: d is upstream of j_obj (j_obj's parent body is
                # downstream of d's joint).
                code[j_obj, d] = 1
                # c_idx and ratio unused for case 1; leave as 0/0.
    return {"code": code, "ratio": ratio, "c_idx": c_idx}


@wp.kernel
def _gravity_torque_jac_analytic(
    body_q: wp.array2d(dtype=wp.transform),
    subtree_com: wp.array2d(dtype=wp.vec3),
    subtree_mass: wp.array1d(dtype=wp.float32),
    joint_body: wp.array1d(dtype=wp.int32),
    joint_axis_local: wp.array1d(dtype=wp.vec3),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    code: wp.array2d(dtype=wp.uint8),
    ratio_arr: wp.array2d(dtype=wp.float32),
    c_idx_arr: wp.array2d(dtype=wp.int32),
    gravity: wp.vec3,
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """One thread per ``(problem, j_obj, dof)``. Skips unrelated entries.

    Differentiates the Cauchy-Schwarz slack
    ``f = R*G - r_p · g_p`` where ``R = |r_p|, G = |g_p|``,
    ``r_p = r - (r·a)a``, ``g_p = g - (g·a)a``, ``r = c_j - p_j``, ``a = â_j``.
    Under unit dof motion the inputs change as:

    * Case 1 (``d`` upstream of ``j``): ``dr = ω×r``, ``da = ω×a``.
    * Case 2 (``d`` inside subtree of ``j``): ``dr = ratio*(v + ω×c_d_sub)``,
      ``da = 0``.

    Then by chain rule:

    * ``d r_p = dr - (dr·a)a - (r·da)a - (r·a)da``
    * ``d g_p = -(g·da)a - (g·a)da``
    * ``dR = (r_p·d r_p)/R``,  ``dG = (g_p·d g_p)/G``
    * ``df = dR*G + R*dG - d r_p · g_p - r_p · d g_p``

    Final entry: ``weight * sqrt(m_j) * df``.

    Assumes ``jacobian`` is zeroed upstream (matches the Newton IK contract
    used by other analytic objectives). The unrelated branch returns
    without writing.
    """
    p, j, d = wp.tid()

    rel = code[j, d]
    if rel == wp.uint8(0):
        return

    parent_tf = body_q[p, joint_body[j]]
    axis = wp.transform_vector(parent_tf, joint_axis_local[j])
    p_j = wp.transform_get_translation(parent_tf)
    c_j = subtree_com[p, j]
    r = c_j - p_j
    r_perp = r - wp.dot(r, axis) * axis
    g_perp = gravity - wp.dot(gravity, axis) * axis

    R = wp.sqrt(wp.dot(r_perp, r_perp) + 1.0e-12)
    G = wp.sqrt(wp.dot(g_perp, g_perp) + 1.0e-12)

    S = joint_S_s[p, d]
    v_orig = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])

    dr = wp.vec3(0.0, 0.0, 0.0)
    da = wp.vec3(0.0, 0.0, 0.0)
    if rel == wp.uint8(1):
        dr = wp.cross(omega, r)
        da = wp.cross(omega, axis)
    else:
        c_eff = subtree_com[p, c_idx_arr[j, d]]
        rt = ratio_arr[j, d]
        dr = rt * (v_orig + wp.cross(omega, c_eff))
        # da remains zero in case 2.

    r_a = wp.dot(r, axis)
    dr_perp = dr - wp.dot(dr, axis) * axis - wp.dot(r, da) * axis - r_a * da
    g_a = wp.dot(gravity, axis)
    dg_perp = -wp.dot(gravity, da) * axis - g_a * da

    dR_dq = wp.dot(r_perp, dr_perp) / R
    dG_dq = wp.dot(g_perp, dg_perp) / G
    df = dR_dq * G + R * dG_dq - wp.dot(dr_perp, g_perp) - wp.dot(r_perp, dg_perp)

    jacobian[p, start_idx + j, d] = weight * wp.sqrt(subtree_mass[j]) * df


@wp.kernel
def _gravity_torque_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    subtree_com: wp.array2d(dtype=wp.vec3),
    subtree_mass: wp.array1d(dtype=wp.float32),
    joint_body: wp.array1d(dtype=wp.int32),
    joint_axis_local: wp.array1d(dtype=wp.vec3),
    gravity: wp.vec3,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Per-joint "excess PE above hang-down" residual.

    Project the joint-to-subtree-COM vector ``r`` and the gravity vector
    onto the plane perpendicular to the joint axis, giving ``r_perp``
    and ``g_perp``. The stable hang configuration has ``r_perp`` aligned
    with ``g_perp`` (COM on the gravity side of the joint axis). Using
    the Cauchy-Schwarz slack

        residual = sqrt(m) * (|r_perp|*|g_perp| - r_perp.g_perp)

    gives a strictly non-negative residual that is zero iff ``r_perp``
    is parallel to ``g_perp`` with the same sign -- a unique minimum.
    The naive signed-torque residual ``axis.(r x m g)`` has the same
    magnitude at the pointing-up equilibrium and, when squared for
    least-squares, creates a spurious second minimum that can attract
    unconstrained limbs away from the hang pose.

    Residual is also automatically zero when the joint axis is parallel
    to gravity (``g_perp = 0``, e.g. yaw-like joints), which correctly
    reports no hang-direction preference for such joints.
    """
    row, jidx = wp.tid()
    parent_tf = body_q[row, joint_body[jidx]]
    axis = wp.transform_vector(parent_tf, joint_axis_local[jidx])
    joint_pos = wp.transform_get_translation(parent_tf)
    r = subtree_com[row, jidx] - joint_pos

    r_perp = r - wp.dot(r, axis) * axis
    g_perp = gravity - wp.dot(gravity, axis) * axis

    # Smoothed magnitudes: guard the jacobian at |r_perp|=0 / |g_perp|=0.
    r_perp_mag = wp.sqrt(wp.dot(r_perp, r_perp) + 1.0e-12)
    g_perp_mag = wp.sqrt(wp.dot(g_perp, g_perp) + 1.0e-12)

    excess_pe = r_perp_mag * g_perp_mag - wp.dot(r_perp, g_perp)
    mass_sqrt = wp.sqrt(subtree_mass[jidx])
    residuals[row, start_idx + jidx] = weight * mass_sqrt * excess_pe


class IKObjectiveGravityTorque(ik.IKObjective):
    """Minimize subtree gravitational excess PE for natural hanging poses.

    For each revolute joint, penalises a Cauchy-Schwarz slack that is
    zero iff the joint-to-subtree-COM vector, projected onto the plane
    perpendicular to the joint axis, is parallel to the in-plane gravity
    component (the hang-down direction). See
    :func:`_gravity_torque_residuals` for the residual form.

    Unlike a naive signed-torque residual, the squared cost has a unique
    global minimum (no pointing-up spurious equilibrium) and is smooth
    everywhere. This makes the objective safe to enable alongside
    foot-contact and joint-regularize objectives without risking
    runaway toward the unstable inverted pose under finite weight.

    Args:
        cfg: :class:`~.cfg.IKObjectiveGravityTorqueCfg` with ``weight``.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``kin.model`` (kinematic tree, body masses, gravity).
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(self, cfg: IKObjectiveGravityTorqueCfg, pipeline: RetargetPipeline, wp_mesh: object = None) -> None:
        super().__init__()
        self.weight = cfg.weight
        model = pipeline.kin.model
        info = _build_subtree_info(model)
        self.n_rev = info["n_rev"]
        self._parent_bodies_np = info["parent_bodies"]
        self._axes_local_np = info["axes_local"]
        self._downstream_bodies_np = info["downstream_bodies"]
        self._downstream_offsets_np = info["downstream_offsets"]
        self._subtree_mass_np = info["subtree_mass"]
        self._subtree_inv_mass_np = (1.0 / (self._subtree_mass_np + 1e-10)).astype(np.float32)
        self._gravity_np = model.gravity.numpy()[0].astype(np.float32)
        self._body_com_np = model.body_com.numpy()
        self._body_mass_np = model.body_mass.numpy()
        # Per-(j_obj, dof) relation tables for the analytic Jacobian.
        rel = _build_jac_relations(model, self._parent_bodies_np)
        self._jac_code_np = rel["code"]
        self._jac_ratio_np = rel["ratio"]
        self._jac_c_idx_np = rel["c_idx"]

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return self.n_rev

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._joint_body = wp.array(self._parent_bodies_np, dtype=wp.int32, device=d)
        self._joint_axis_local = wp.from_numpy(self._axes_local_np, dtype=wp.vec3, device=d)
        self._downstream_bodies = wp.array(self._downstream_bodies_np, dtype=wp.int32, device=d)
        self._downstream_offsets = wp.array(self._downstream_offsets_np, dtype=wp.int32, device=d)
        self._subtree_mass = wp.array(self._subtree_mass_np, dtype=wp.float32, device=d)
        self._subtree_inv_mass = wp.array(self._subtree_inv_mass_np, dtype=wp.float32, device=d)
        self._body_com = wp.from_numpy(self._body_com_np, dtype=wp.vec3, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)
        self._gravity_vec = wp.vec3(*self._gravity_np.tolist())
        self._subtree_com_buf = wp.zeros((self.n_batch, self.n_rev), dtype=wp.vec3, device=d)
        # Analytic-Jacobian relation tables on device.
        self._jac_code = wp.array(self._jac_code_np, dtype=wp.uint8, device=d)
        self._jac_ratio = wp.array(self._jac_ratio_np, dtype=wp.float32, device=d)
        self._jac_c_idx = wp.array(self._jac_c_idx_np, dtype=wp.int32, device=d)
        # Autodiff scratch: only allocate when the solver may take the
        # autodiff path. ``MIXED`` calls ``compute_jacobian_analytic`` for
        # this objective (see :meth:`supports_analytic`) so it skips the
        # allocation too.
        self._e_arrays: list[wp.array] = []
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            for r in range(self.n_rev):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for b in range(self.n_batch):
                    e[b, self.residual_offset + r] = 1.0
                self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=d))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n = body_q.shape[0]
        wp.launch(
            _compute_subtree_com,
            dim=[n, self.n_rev],
            inputs=[
                body_q,
                self._body_com,
                self._body_mass_dev,
                self._downstream_bodies,
                self._downstream_offsets,
                self._subtree_inv_mass,
            ],
            outputs=[self._subtree_com_buf],
            device=self.device,
        )
        wp.launch(
            _gravity_torque_residuals,
            dim=[n, self.n_rev],
            inputs=[
                body_q,
                self._subtree_com_buf,
                self._subtree_mass,
                self._joint_body,
                self._joint_axis_local,
                self._gravity_vec,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_rev):
            tape.backward(grads={tape.outputs[0]: self._e_arrays[r]})
            wp.launch(
                jac_fill_row,
                dim=self.n_batch,
                inputs=[tape.gradients[dq_dof], n_dofs, start_idx + r],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """One fused kernel writes every ``[problem, j_obj, dof]`` Jacobian
        entry. Reuses ``self._subtree_com_buf`` populated by the most-recent
        :meth:`compute_residuals` call (Newton's IK solver always runs
        residuals before Jacobians per iteration), so the per-joint subtree
        COMs are fresh.
        """
        self._require_batch_layout()
        n_dofs = model.joint_dof_count
        wp.launch(
            _gravity_torque_jac_analytic,
            dim=[self.n_batch, self.n_rev, n_dofs],
            inputs=[
                body_q,
                self._subtree_com_buf,
                self._subtree_mass,
                self._joint_body,
                self._joint_axis_local,
                joint_S_s,
                self._jac_code,
                self._jac_ratio,
                self._jac_c_idx,
                self._gravity_vec,
                self.weight,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )
