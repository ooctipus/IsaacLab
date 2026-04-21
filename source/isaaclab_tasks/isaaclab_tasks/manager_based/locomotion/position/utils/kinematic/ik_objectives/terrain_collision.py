# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: terrain collision avoidance."""

from __future__ import annotations

from collections import defaultdict

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row


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


class IKObjectiveTerrainCollision(ik.IKObjective):
    """Penalize robot body surface points penetrating the terrain mesh.

    Args:
        mesh_id: Warp mesh identifier.
        builder: Newton model builder (for mesh vertex access).
        exclude_bodies: Body indices to skip (e.g. foot bodies).
        weight: Residual weight.
        margin: Softplus temperature [m].
        n_samples: Surface sample points per body.
    """

    def __init__(self, mesh_id: int, builder: newton.ModelBuilder, exclude_bodies: list[int],
                 weight: float = 3.0, margin: float = 0.05, n_samples: int = 16) -> None:
        super().__init__()
        self.mesh_id = mesh_id
        self.weight = weight
        self.margin = margin
        bodies, offsets = _build_collision_probes(builder, exclude_bodies, n_samples)
        self.n_probes = len(bodies)
        self._probe_body_np = np.array(bodies, dtype=np.int32)
        self._probe_offset_np = np.array(offsets, dtype=np.float32)

    def supports_analytic(self) -> bool:
        return False

    def residual_dim(self) -> int:
        return self.n_probes

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        self._probe_body = wp.array(self._probe_body_np, dtype=wp.int32, device=d)
        self._probe_offset = wp.from_numpy(self._probe_offset_np, dtype=wp.vec3, device=d)
        self._e_arrays = []
        for r in range(self.n_probes):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for b in range(self.n_batch):
                e[b, self.residual_offset + r] = 1.0
            self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=d))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        wp.launch(_terrain_collision_residuals, dim=[body_q.shape[0], self.n_probes],
                  inputs=[body_q, self.mesh_id, self._probe_body, self._probe_offset,
                          self.weight, self.margin, start_idx],
                  outputs=[residuals], device=self.device)

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        n_dofs = dq_dof.shape[1]
        for r in range(self.n_probes):
            tape.backward(grads={tape.outputs[0]: self._e_arrays[r]})
            wp.launch(jac_fill_row, dim=self.n_batch,
                      inputs=[tape.gradients[dq_dof], n_dofs, start_idx + r],
                      outputs=[jacobian], device=self.device)
            tape.zero()
