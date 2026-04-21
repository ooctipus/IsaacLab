# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test mesh_query_point gradient continuity across triangle edges.

Sweeps a query point across a stair edge and records the closest surface
point + gradient of distance w.r.t. query position.

Usage::

    ./isaaclab.sh -p scripts/tools/test_mesh_query_gradient.py
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import numpy as np
import warp as wp


@wp.kernel
def _query_closest(
    mesh_id: wp.uint64,
    query_points: wp.array1d(dtype=wp.vec3),
    closest_points: wp.array1d(dtype=wp.vec3),
    distances: wp.array1d(dtype=wp.float32),
):
    i = wp.tid()
    q = wp.mesh_query_point(mesh_id, query_points[i], 10.0)
    if q.result:
        closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        closest_points[i] = closest
        distances[i] = wp.length(query_points[i] - closest)


@wp.kernel
def _compute_dist_single(
    mesh_id: wp.uint64,
    query_point: wp.array1d(dtype=wp.vec3),
    loss: wp.array1d(dtype=wp.float32),
):
    q = wp.mesh_query_point(mesh_id, query_point[0], 10.0)
    if q.result:
        closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        d = wp.length(query_point[0] - closest)
        loss[0] = d


def main():
    wp.init()

    # Build a simple stair: two steps
    # Step 1: z=0, from x=-1 to x=0
    # Step 2: z=0.2, from x=0 to x=1
    verts = np.array([
        # Step 1 (tread at z=0)
        [-1.0, -0.5, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.5, 0.0],
        [-1.0, 0.5, 0.0],
        # Step 2 (tread at z=0.2)
        [0.0, -0.5, 0.2],
        [1.0, -0.5, 0.2],
        [1.0, 0.5, 0.2],
        [0.0, 0.5, 0.2],
        # Riser (vertical face at x=0, from z=0 to z=0.2)
        [0.0, -0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, -0.5, 0.2],
        [0.0, 0.5, 0.2],
    ], dtype=np.float32)

    faces = np.array([
        # Step 1
        [0, 1, 2], [0, 2, 3],
        # Step 2
        [4, 5, 6], [4, 6, 7],
        # Riser
        [8, 10, 11], [8, 11, 9],
    ], dtype=np.int32)

    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3),
        indices=wp.array(faces.flatten(), dtype=wp.int32),
    )

    # === Closest point sweep ===
    N = 100
    x_vals = np.linspace(-0.5, 0.5, N)
    query_pts = np.zeros((N, 3), dtype=np.float32)
    query_pts[:, 0] = x_vals
    query_pts[:, 2] = 0.3

    query_wp = wp.array(query_pts, dtype=wp.vec3)
    closest_wp = wp.zeros(N, dtype=wp.vec3)
    dist_wp = wp.zeros(N, dtype=wp.float32)

    wp.launch(_query_closest, dim=N, inputs=[mesh.id, query_wp, closest_wp, dist_wp])

    closest_np = closest_wp.numpy()
    dist_np = dist_wp.numpy()

    print("=== Closest point sweep across stair edge ===")
    print(f"Query: x in [{x_vals[0]:.2f}, {x_vals[-1]:.2f}], y=0, z=0.3")
    print(f"Step 1: z=0 (x<0), Step 2: z=0.2 (x>0), Riser at x=0")
    print()
    print(f"{'x':>6s}  {'closest_x':>9s}  {'closest_z':>9s}  {'dist':>6s}")
    for i in range(0, N, 5):
        cx, cy, cz = closest_np[i]
        print(f"{x_vals[i]:6.3f}  {cx:9.4f}  {cz:9.4f}  {dist_np[i]:6.4f}")

    z_vals = closest_np[:, 2]
    max_jump = np.max(np.abs(np.diff(z_vals)))
    print(f"\nMax z-jump between adjacent samples: {max_jump:.6f}")
    if max_jump > 0.05:
        print("WARNING: significant discontinuity in closest point")
    else:
        print("OK: closest point appears continuous")

    # === Gradient test ===
    print("\n=== Gradient at z=0.3 (above both steps) ===")
    print(f"{'x':>6s}  {'dist':>6s}  {'grad_x':>8s}  {'grad_y':>8s}  {'grad_z':>8s}")
    for x_test in [-0.3, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.3]:
        pt = wp.array([wp.vec3(x_test, 0.0, 0.3)], dtype=wp.vec3, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_compute_dist_single, dim=1, inputs=[mesh.id, pt, loss])

        tape.backward(loss)
        grad = pt.grad.numpy()[0] if pt.grad is not None else [0, 0, 0]
        loss_val = loss.numpy()[0]
        print(f"{x_test:+6.3f}  {loss_val:6.4f}  {grad[0]:+8.4f}  {grad[1]:+8.4f}  {grad[2]:+8.4f}")
        tape.zero()

    # === Gradient at z=0.1 (between step heights) ===
    print("\n=== Gradient at z=0.1 (between step heights, near riser) ===")
    print(f"{'x':>6s}  {'dist':>6s}  {'grad_x':>8s}  {'grad_z':>8s}")
    for x_test in [-0.15, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.15]:
        pt = wp.array([wp.vec3(x_test, 0.0, 0.1)], dtype=wp.vec3, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_compute_dist_single, dim=1, inputs=[mesh.id, pt, loss])

        tape.backward(loss)
        grad = pt.grad.numpy()[0] if pt.grad is not None else [0, 0, 0]
        loss_val = loss.numpy()[0]
        print(f"{x_test:+6.3f}  {loss_val:6.4f}  {grad[0]:+8.4f}  {grad[2]:+8.4f}")
        tape.zero()


if __name__ == "__main__":
    main()
