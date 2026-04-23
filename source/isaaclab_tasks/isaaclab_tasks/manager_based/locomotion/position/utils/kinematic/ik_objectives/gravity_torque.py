# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: gravity torque minimization."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row


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
    row, jidx = wp.tid()
    parent_tf = body_q[row, joint_body[jidx]]
    axis_world = wp.transform_vector(parent_tf, joint_axis_local[jidx])
    joint_pos = wp.transform_get_translation(parent_tf)
    r = subtree_com[row, jidx] - joint_pos
    f = subtree_mass[jidx] * gravity
    residuals[row, start_idx + jidx] = weight * wp.dot(axis_world, wp.cross(r, f))


class IKObjectiveGravityTorque(ik.IKObjective):
    """Minimize static gravity compensation torques for natural poses.

    Args:
        model: Newton model (used to extract kinematic tree).
        weight: Scalar multiplier for the torque residual.
    """

    def __init__(self, model: newton.Model, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = weight
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
