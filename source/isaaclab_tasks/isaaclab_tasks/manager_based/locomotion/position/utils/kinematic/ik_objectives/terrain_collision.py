# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: terrain collision avoidance."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row

if TYPE_CHECKING:
    from ...mdp.retarget.pipeline import RetargetPipeline
    from .cfg import IKObjectiveTerrainCollisionCfg


@wp.kernel
def _write_basis_column(
    total_residuals: int,
    col: int,
    value: float,
    out: wp.array1d(dtype=wp.float32),
):
    """Write ``value`` at column ``col`` of every batch row in a flat ``(n_batch * total_residuals)`` buffer."""
    b = wp.tid()
    out[b * total_residuals + col] = value


@wp.kernel
def _terrain_collision_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    weight: float,
    margin: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row, probe_idx = wp.tid()
    tf = body_q[row, probe_body[probe_idx]]
    probe_pos = wp.transform_point(tf, probe_offset[probe_idx])
    query = wp.mesh_query_point(mesh_id, probe_pos, 2.0)
    if query.result:
        surface_pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(probe_pos - surface_pt)
        sign_pen = -query.sign * dist
        z_pen = surface_pt[2] - probe_pos[2]
        depth = wp.max(sign_pen, z_pen)
        pen = wp.log(1.0 + wp.exp(depth / margin)) * margin
        residuals[row, start_idx + probe_idx] = weight * pen


@wp.kernel
def _terrain_collision_jac_analytic(
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    weight: float,
    margin: float,
    affects_dof: wp.array2d(dtype=wp.uint8),  # (n_probes, n_dofs)
    body_q: wp.array2d(dtype=wp.transform),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_batch, n_dofs)
    start_idx: int,
    # output
    jacobian: wp.array3d(dtype=wp.float32),
):
    """One thread per (problem, probe, dof). Writes a single Jacobian entry.

    Assumes ``jacobian`` has been zeroed upstream. Threads that find the
    probe outside the mesh search radius, on the softplus tail with
    negligible gradient, or on a DoF that cannot move the probe body
    simply return without writing.
    """
    problem_idx, probe_idx, dof_idx = wp.tid()

    if affects_dof[probe_idx, dof_idx] == 0:
        return

    body_idx = probe_body[probe_idx]
    tf = body_q[problem_idx, body_idx]
    probe_pos = wp.transform_point(tf, probe_offset[probe_idx])
    query = wp.mesh_query_point(mesh_id, probe_pos, 2.0)
    if not query.result:
        return

    surface_pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    delta = probe_pos - surface_pt
    dist = wp.length(delta)
    sign_pen = -query.sign * dist
    z_pen = surface_pt[2] - probe_pos[2]
    depth = wp.max(sign_pen, z_pen)

    # d(depth)/d(probe_pos), matching the wp.max subgradient (takes the
    # sign_pen branch at equality to mirror autodiff's >= behaviour).
    grad_depth = wp.vec3(0.0, 0.0, 0.0)
    if sign_pen >= z_pen:
        if dist > 1.0e-8:
            inv_dist = 1.0 / dist
            s = -query.sign * inv_dist
            grad_depth = wp.vec3(s * delta[0], s * delta[1], s * delta[2])
    else:
        grad_depth = wp.vec3(0.0, 0.0, -1.0)

    # Numerically stable softplus derivative: sigmoid(depth / margin).
    x = depth / margin
    sig = float(0.0)
    if x >= 0.0:
        sig = 1.0 / (1.0 + wp.exp(-x))
    else:
        e = wp.exp(x)
        sig = e / (1.0 + e)

    # Spatial velocity of the probe point for a unit rate along this DoF.
    S = joint_S_s[problem_idx, dof_idx]
    v_orig = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])
    v_probe = v_orig + wp.cross(omega, probe_pos)

    jacobian[problem_idx, start_idx + probe_idx, dof_idx] = (
        weight * sig * (grad_depth[0] * v_probe[0] + grad_depth[1] * v_probe[1] + grad_depth[2] * v_probe[2])
    )


def _build_collision_probes(
    builder: newton.ModelBuilder,
    exclude_bodies: list[int],
    n_samples: int = 16,
) -> tuple[list[int], list[tuple[float, float, float]]]:
    """Sample probe points on each body's actual mesh surface."""
    body_verts: dict[int, list[np.ndarray]] = defaultdict(list)
    for si in range(len(builder.shape_body)):
        bid = int(builder.shape_body[si])
        if bid in exclude_bodies:
            continue
        src = builder.shape_source[si]
        if src is not None and hasattr(src, "vertices") and len(src.vertices) > 0:
            verts = np.array(src.vertices, dtype=np.float32)
            if len(verts.shape) == 1:
                verts = verts.reshape(-1, 3)
            body_verts[bid].append(verts)

    probe_bodies: list[int] = []
    probe_offsets: list[tuple[float, float, float]] = []
    for bid in sorted(body_verts.keys()):
        all_v = np.concatenate(body_verts[bid], axis=0)
        if len(all_v) == 0:
            continue
        n = min(n_samples, len(all_v))
        if n <= 0:
            continue
        selected = [0]
        min_dists = np.full(len(all_v), np.inf)
        for _ in range(n - 1):
            d = np.linalg.norm(all_v - all_v[selected[-1]], axis=1)
            min_dists = np.minimum(min_dists, d)
            selected.append(int(np.argmax(min_dists)))
        for idx in selected:
            probe_bodies.append(bid)
            probe_offsets.append(tuple(float(x) for x in all_v[idx]))
    return probe_bodies, probe_offsets


def _build_affects_dof_per_body(model: newton.Model) -> np.ndarray:
    """Return a ``(n_bodies, n_dofs)`` mask of whether a DoF can move a body.

    Walks the Newton joint tree once to mark every ancestor joint of each
    body, then expands joint indices to DoF indices via ``joint_qd_start``.
    """
    n_bodies = model.body_count
    n_dofs = model.joint_dof_count

    joint_qd_start = model.joint_qd_start.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()

    dof_to_joint = np.empty(n_dofs, dtype=np.int32)
    for j in range(len(joint_qd_start) - 1):
        dof_to_joint[joint_qd_start[j] : joint_qd_start[j + 1]] = j

    body_to_joint = np.full(n_bodies, -1, dtype=np.int32)
    for j in range(model.joint_count):
        c = int(joint_child[j])
        if c != -1:
            body_to_joint[c] = j

    mask = np.zeros((n_bodies, n_dofs), dtype=np.uint8)
    for b in range(n_bodies):
        ancestors = np.zeros(model.joint_count, dtype=bool)
        body = b
        while body != -1:
            j = int(body_to_joint[body])
            if j == -1:
                break
            ancestors[j] = True
            body = int(joint_parent[j])
        mask[b] = ancestors[dof_to_joint].astype(np.uint8)
    return mask


class IKObjectiveTerrainCollision(ik.IKObjective):
    """Penalize robot body surface points penetrating the terrain mesh.

    Args:
        cfg: :class:`~.cfg.IKObjectiveTerrainCollisionCfg` with
            ``weight``, ``margin``, ``n_samples``.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``kin.builder`` (probe generation) and ``foot_body_ids``
            (excluded from probing).
        wp_mesh: Terrain Warp mesh; ``mesh_id`` is stored for GPU queries.
    """

    def __init__(
        self,
        cfg: IKObjectiveTerrainCollisionCfg,
        pipeline: RetargetPipeline,
        wp_mesh: object,
    ) -> None:
        super().__init__()
        self.mesh_id = wp_mesh.id
        self.weight = cfg.weight
        self.margin = cfg.margin
        bodies, offsets = _build_collision_probes(pipeline.kin.builder, pipeline.foot_body_ids, cfg.n_samples)
        self.n_probes = len(bodies)
        self._probe_body_np = np.array(bodies, dtype=np.int32)
        self._probe_offset_np = np.array(offsets, dtype=np.float32)

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return self.n_probes

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._probe_body = wp.array(self._probe_body_np, dtype=wp.int32, device=d)
        self._probe_offset = wp.from_numpy(self._probe_offset_np, dtype=wp.vec3, device=d)

        if jacobian_mode in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            # Ancestor mask per probe: (n_probes, n_dofs) uint8.
            body_mask = _build_affects_dof_per_body(model)
            probe_mask = body_mask[self._probe_body_np]
            self._affects_dof = wp.array(probe_mask, dtype=wp.uint8, device=d)

        if jacobian_mode in (ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED):
            # Shared scratch basis vector reused for every probe backward pass.
            # Each probe's basis is ``1`` at ``residual_offset + r`` and ``0``
            # elsewhere in every batch row — materialising all ``n_probes``
            # copies would cost ``n_probes * n_batch * total_residuals * 4`` B
            # (>1 GB at high batch counts); we instead fill column ``r`` before
            # each backward and clear it after.
            self._e_scratch = wp.zeros(self.n_batch * self.total_residuals, dtype=wp.float32, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(
            _terrain_collision_residuals,
            dim=[body_q.shape[0], self.n_probes],
            inputs=[body_q, self.mesh_id, self._probe_body, self._probe_offset, self.weight, self.margin, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_probes):
            col = self.residual_offset + r
            wp.launch(
                _write_basis_column,
                dim=self.n_batch,
                inputs=[self.total_residuals, col, 1.0],
                outputs=[self._e_scratch],
                device=self.device,
            )
            tape.backward(grads={tape.outputs[0]: self._e_scratch})
            wp.launch(
                jac_fill_row,
                dim=self.n_batch,
                inputs=[tape.gradients[dq_dof], n_dofs, start_idx + r],
                outputs=[jacobian],
                device=self.device,
            )
            wp.launch(
                _write_basis_column,
                dim=self.n_batch,
                inputs=[self.total_residuals, col, 0.0],
                outputs=[self._e_scratch],
                device=self.device,
            )
            tape.zero()

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """Fuse all ``n_probes * n_dofs`` Jacobian entries into a single kernel launch.

        The softplus-smoothed collision residual depends on ``probe_pos``
        through a ``max``-selected depth measure; we differentiate that
        analytically and compose with Newton's spatial motion subspace
        (:paramref:`joint_S_s`) to get the velocity of each probe point
        per DoF.
        """
        n_dofs = model.joint_dof_count
        wp.launch(
            _terrain_collision_jac_analytic,
            dim=[body_q.shape[0], self.n_probes, n_dofs],
            inputs=[
                self.mesh_id,
                self._probe_body,
                self._probe_offset,
                self.weight,
                self.margin,
                self._affects_dof,
                body_q,
                joint_S_s,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )
