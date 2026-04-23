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
    from ...mdp.retarget.pipeline import RetargetPipeline
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

    def supports_analytic(self) -> bool:
        return False

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
        self._e_arrays = []
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
