# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom IK objectives for the offline factory pipeline.

* :class:`FactoryCollisionObjective` -- differentiable probe-vs-mesh avoidance:
  a closest-point signed-distance + softplus penalty with an ANALYTIC Jacobian,
  adapted from the terrain ``IKObjectiveTerrainCollision``. General over
  (probe body, obstacle) pairs: probes may live on ANY robot link (each probe's
  gradient flows only through its own body's ancestor dofs), and the obstacle
  mesh carries a per-problem pose -- identity for static world meshes, the nut
  placement for the held asset. One instance per obstacle.
* :class:`JointPinObjective` -- pin selected joint coordinates to per-problem
  targets. Used to fix both finger coords to half the grasp-pair separation,
  which enforces the gripper mimic constraint (``q_l == q_r``) structurally and
  makes the aperture exact data instead of a solved quantity (it also removes a
  null direction that degrades the LM solve).

The warp kernels operate on warp transforms directly, so this fork's
``(x, y, z, w)`` quaternion layout needs no conversion.
"""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp


@wp.kernel
def _collision_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    mesh_id: wp.uint64,
    obstacle_q: wp.array1d(dtype=wp.transformf),  # per-problem obstacle pose
    weight: float,
    margin: float,
    max_dist: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row, p = wp.tid()
    world_p = wp.transform_point(body_q[row, probe_body[p]], probe_offset[p])
    local = wp.transform_point(wp.transform_inverse(obstacle_q[row]), world_p)
    q = wp.mesh_query_point(mesh_id, local, max_dist)
    if q.result:
        surf = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        depth = -q.sign * wp.length(local - surf)  # >0 inside the obstacle
        residuals[row, start_idx + p] = weight * wp.log(1.0 + wp.exp(depth / margin)) * margin


@wp.kernel
def _collision_jac(
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    obstacle_q: wp.array1d(dtype=wp.transformf),
    weight: float,
    margin: float,
    max_dist: float,
    n_dofs: int,
    body_affects_dof: wp.array1d(dtype=wp.uint8),  # [n_bodies * n_dofs] ancestry mask
    body_q: wp.array2d(dtype=wp.transform),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    # one thread per (problem, probe): the mesh query depends only on the probe's
    # world position, so it runs ONCE and the dof loop reuses its gradient --
    # a per-(prob, probe, dof) launch would repeat the query n_dofs times.
    prob, p = wp.tid()
    world_p = wp.transform_point(body_q[prob, probe_body[p]], probe_offset[p])
    obs_q = obstacle_q[prob]
    local = wp.transform_point(wp.transform_inverse(obs_q), world_p)
    q = wp.mesh_query_point(mesh_id, local, max_dist)
    if not q.result:
        return
    surf = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    delta = local - surf
    dist = wp.length(delta)
    if dist <= 1.0e-8:
        return
    depth = -q.sign * dist
    s = -q.sign / dist
    grad_local = wp.vec3(s * delta[0], s * delta[1], s * delta[2])
    grad_world = wp.quat_rotate(wp.transform_get_rotation(obs_q), grad_local)
    x = depth / margin
    sig = float(0.0)
    if x >= 0.0:
        sig = 1.0 / (1.0 + wp.exp(-x))
    else:
        e = wp.exp(x)
        sig = e / (1.0 + e)
    base = probe_body[p] * n_dofs
    for dof in range(n_dofs):
        # gradient flows only through the probe body's OWN ancestor dofs -- a dof
        # downstream of (or off-chain from) the probe body does not move the probe.
        if body_affects_dof[base + dof] != wp.uint8(0):
            sv = joint_S_s[prob, dof]
            v_probe = wp.vec3(sv[0], sv[1], sv[2]) + wp.cross(wp.vec3(sv[3], sv[4], sv[5]), world_p)
            jacobian[prob, start_idx + p, dof] = weight * sig * wp.dot(grad_world, v_probe)


def ancestor_dof_mask(model: newton.Model, body: int) -> np.ndarray:
    """``uint8[n_dofs]`` mask: DOFs whose joint is an ancestor of ``body``."""
    qd_start = model.joint_qd_start.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    dof_to_joint = np.empty(qd_start[-1], dtype=np.int32)
    for j in range(len(qd_start) - 1):
        dof_to_joint[qd_start[j] : qd_start[j + 1]] = j
    body_to_joint = np.full(model.body_count, -1, np.int32)
    for j in range(model.joint_count):
        if jc[j] != -1:
            body_to_joint[jc[j]] = j
    ancestors = np.zeros(model.joint_count, dtype=bool)
    b = body
    while b != -1:
        j = int(body_to_joint[b])
        if j == -1:
            break
        ancestors[j] = True
        b = int(jp[j])
    return ancestors[dof_to_joint].astype(np.uint8)


class FactoryCollisionObjective(ik.IKObjective):
    """Penalize surface probes on robot links penetrating an obstacle mesh.

    Probes may live on any robot body; each probe's gradient flows only through
    that body's ancestor dofs. The obstacle mesh is queried in its own frame via a
    per-problem pose, so one class covers static world obstacles (identity poses)
    and the per-problem-posed held asset alike.

    Args:
        probe_offsets: Probe offsets in their probe-body frames [m], shape ``[P, 3]``.
        probe_bodies: Body index each probe is attached to, shape ``[P]``.
        mesh_id: Obstacle :class:`warp.Mesh` id (in the obstacle's own frame).
        obstacle_pose: Per-problem obstacle pose [n_problems, 7] (pos [m] + quat
            xyzw); identity rows for a static world-frame mesh.
        weight: Objective weight.
        margin: Softplus smoothing scale [m].
    """

    def __init__(
        self,
        probe_offsets: np.ndarray,
        probe_bodies: np.ndarray,
        mesh_id: int,
        obstacle_pose: torch.Tensor,
        weight: float,
        margin: float,
        max_dist: float = 0.25,
    ):
        super().__init__()
        self.mesh_id = wp.uint64(mesh_id)
        self.weight = weight
        self.margin = margin
        self.max_dist = max_dist
        self._probe_np = probe_offsets.astype(np.float32)
        self._probe_body_np = probe_bodies.astype(np.int32)
        self._obstacle_pose_t = obstacle_pose.contiguous()
        self.n_samples = len(probe_offsets)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return self.n_samples

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        self._probe = wp.from_numpy(self._probe_np, dtype=wp.vec3, device=self.device)
        self._probe_body = wp.from_numpy(self._probe_body_np, dtype=wp.int32, device=self.device)
        self._obstacle_q = wp.from_torch(self._obstacle_pose_t, dtype=wp.transformf)
        self._n_dofs = model.joint_dof_count
        if jacobian_mode in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            mask = np.stack([ancestor_dof_mask(model, b) for b in range(model.body_count)])
            self._affects = wp.array(mask.flatten(), dtype=wp.uint8, device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _collision_residuals,
            dim=[body_q.shape[0], self.n_samples],
            inputs=[
                body_q,
                self._probe_body,
                self._probe,
                self.mesh_id,
                self._obstacle_q,
                self.weight,
                self.margin,
                self.max_dist,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        wp.launch(
            _collision_jac,
            dim=[body_q.shape[0], self.n_samples],
            inputs=[
                self.mesh_id,
                self._probe_body,
                self._probe,
                self._obstacle_q,
                self.weight,
                self.margin,
                self.max_dist,
                self._n_dofs,
                self._affects,
                body_q,
                joint_S_s,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _joint_pin_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    coords: wp.array1d(dtype=wp.int32),
    targets: wp.array2d(dtype=wp.float32),
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row, k = wp.tid()
    residuals[row, start_idx + k] = weight * (joint_q[row, coords[k]] - targets[row, k])


@wp.kernel
def _joint_pin_jacobian(
    dofs: wp.array1d(dtype=wp.int32),
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    prob, k = wp.tid()
    jacobian[prob, start_idx + k, dofs[k]] = weight


class JointPinObjective(ik.IKObjective):
    """Pin selected joint coordinates to per-problem targets.

    Args:
        coords: Joint coordinate indices to pin, shape ``[K]``.
        dofs: Matching DOF indices (== coords for revolute/prismatic chains), shape ``[K]``.
        targets: Per-problem pin values [m or rad, depending on joint type],
            shape ``[n_problems, K]``.
        weight: Objective weight.
    """

    def __init__(self, coords: np.ndarray, dofs: np.ndarray, targets: torch.Tensor, weight: float):
        super().__init__()
        self._coords_np = coords.astype(np.int32)
        self._dofs_np = dofs.astype(np.int32)
        self._targets_t = targets.contiguous()
        self.weight = weight

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return len(self._coords_np)

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        self._coords = wp.from_numpy(self._coords_np, dtype=wp.int32, device=self.device)
        self._dofs = wp.from_numpy(self._dofs_np, dtype=wp.int32, device=self.device)
        self._targets = wp.from_torch(self._targets_t)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _joint_pin_residuals,
            dim=[body_q.shape[0], len(self._coords_np)],
            inputs=[joint_q, self._coords, self._targets, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        wp.launch(
            _joint_pin_jacobian,
            dim=[body_q.shape[0], len(self._coords_np)],
            inputs=[self._dofs, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
