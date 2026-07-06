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
    from ..newton_kinematics import NewtonKinematics
    from .cfg import IKObjectiveGravityTorqueCfg
    from .context import IKObjectiveBuildContext


def _build_subtree_info(topology: NewtonKinematics.Topology) -> dict:
    """Derive revolute-joint objective rows from canonical subtree facts."""
    revolute_joints = np.flatnonzero(topology.joint_type == int(newton.JointType.REVOLUTE)).astype(np.int32)
    parent_bodies: list[int] = []
    axes_local: list[np.ndarray] = []
    downstream_bodies: list[int] = []
    downstream_offsets = [0]
    for joint in revolute_joints:
        parent = int(topology.joint_parent[joint])
        begin = int(topology.joint_qd_start[joint])
        end = int(topology.joint_qd_start[joint + 1])
        if parent < 0 or end - begin != 1:
            raise ValueError("Gravity torque requires non-root one-DOF revolute joints.")
        subtree_begin = int(topology.joint_subtree_offsets[joint])
        subtree_end = int(topology.joint_subtree_offsets[joint + 1])
        parent_bodies.append(parent)
        axes_local.append(topology.joint_axis[begin])
        downstream_bodies.extend(int(body) for body in topology.joint_subtree_bodies[subtree_begin:subtree_end])
        downstream_offsets.append(len(downstream_bodies))
    return {
        "joint_indices": revolute_joints,
        "n_rev": len(revolute_joints),
        "parent_bodies": np.asarray(parent_bodies, dtype=np.int32),
        "axes_local": np.asarray(axes_local, dtype=np.float32).reshape(-1, 3),
        "downstream_bodies": np.asarray(downstream_bodies, dtype=np.int32),
        "downstream_offsets": np.asarray(downstream_offsets, dtype=np.int32),
        "subtree_mass": topology.joint_subtree_mass[revolute_joints],
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


def _build_jac_relations(
    topology: NewtonKinematics.Topology,
    revolute_joints: np.ndarray,
) -> dict:
    """Derive gravity Jacobian relation tables from canonical topology."""
    subtree_sets = []
    for joint in range(topology.joint_count):
        begin = int(topology.joint_subtree_offsets[joint])
        end = int(topology.joint_subtree_offsets[joint + 1])
        subtree_sets.append(set(int(body) for body in topology.joint_subtree_bodies[begin:end]))

    objective_index = np.full(topology.joint_count, -1, dtype=np.int32)
    objective_index[revolute_joints] = np.arange(len(revolute_joints), dtype=np.int32)
    code = np.zeros((len(revolute_joints), topology.dof_count), dtype=np.uint8)
    ratio = np.zeros((len(revolute_joints), topology.dof_count), dtype=np.float32)
    com_index = np.zeros((len(revolute_joints), topology.dof_count), dtype=np.int32)

    for row, joint in enumerate(revolute_joints):
        subtree = subtree_sets[int(joint)]
        subtree_mass = float(topology.joint_subtree_mass[joint])
        parent_body = int(topology.joint_parent[joint])
        for dof, dof_joint_value in enumerate(topology.dof_joint):
            dof_joint = int(dof_joint_value)
            if int(topology.joint_child[dof_joint]) in subtree:
                downstream_row = int(objective_index[dof_joint])
                if downstream_row < 0:
                    continue
                code[row, dof] = 2
                com_index[row, dof] = downstream_row
                if subtree_mass > 0.0:
                    ratio[row, dof] = topology.joint_subtree_mass[dof_joint] / subtree_mass
            elif parent_body in subtree_sets[dof_joint]:
                code[row, dof] = 1
    return {"code": code, "ratio": ratio, "c_idx": com_index}


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
        context: Explicit shared kinematics.
    """

    def __init__(self, cfg: IKObjectiveGravityTorqueCfg, context: IKObjectiveBuildContext) -> None:
        super().__init__()
        self.weight = cfg.weight
        topology = context.kinematics.topology
        info = _build_subtree_info(topology)
        self.n_rev = info["n_rev"]
        self._parent_bodies_np = info["parent_bodies"]
        self._axes_local_np = info["axes_local"]
        self._downstream_bodies_np = info["downstream_bodies"]
        self._downstream_offsets_np = info["downstream_offsets"]
        self._subtree_mass_np = info["subtree_mass"]
        self._subtree_inv_mass_np = topology.joint_subtree_inverse_mass[info["joint_indices"]]
        self._gravity_np = topology.gravity
        self._body_com_np = topology.body_com
        self._body_mass_np = topology.body_mass
        # Per-(j_obj, dof) relation tables for the analytic Jacobian.
        rel = _build_jac_relations(topology, info["joint_indices"])
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

    def estimate_memory(
        self,
        model: newton.Model,
        jacobian_mode: ik.IKJacobianType,
        n_problems: int,
        n_batch: int,
        total_residuals: int,
    ) -> int:
        """Estimate immutable mechanics and per-batch subtree workspaces [byte]."""
        del model, n_problems
        fixed_bytes = sum(
            values.nbytes
            for values in (
                self._parent_bodies_np,
                self._axes_local_np,
                self._downstream_bodies_np,
                self._downstream_offsets_np,
                self._subtree_mass_np,
                self._subtree_inv_mass_np,
                self._body_com_np,
                self._body_mass_np,
                self._jac_code_np,
                self._jac_ratio_np,
                self._jac_c_idx_np,
            )
        )
        workspace_bytes = n_batch * self.n_rev * wp.types.type_size_in_bytes(wp.vec3)
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            workspace_bytes += self.n_rev * n_batch * total_residuals * wp.types.type_size_in_bytes(wp.float32)
        return int(fixed_bytes + workspace_bytes)

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
