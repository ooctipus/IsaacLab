# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone verification: fused (shared-pattern) raycast vs the upstream two-kernel path.

This test does not require Isaac Sim, the env, or any task config. It builds a tiny
synthetic mesh + a batch of random sensor transforms in Warp directly, then runs both
the upstream legacy path (:func:`isaaclab.sensors.ray_caster.kernels.update_ray_caster_kernel`
followed by :func:`isaaclab.utils.warp.kernels.raycast_dynamic_meshes_kernel`) and the
self-contained fused path
(:func:`isaaclab_tasks.core.multi_task.sensors.kernels.raycast_dynamic_meshes_with_world_transform_shared_pattern_kernel`)
on the same inputs. If the two paths agree on every ray-hit position across all three
alignment modes (world/yaw/base), the fused kernel is correct.

Run with::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/scripts/verify_fused_raycast.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
import warp as wp

from isaaclab.sensors.ray_caster.kernels import update_ray_caster_kernel
from isaaclab.utils.warp.kernels import raycast_dynamic_meshes_kernel

from isaaclab_tasks.core.multi_task.sensors.kernels import fast_terrain_raycast_kernel


def _make_box_mesh(device: str, half_extent: float = 50.0) -> wp.Mesh:
    """Build a single axis-aligned box mesh in Warp.

    The box is large enough to catch every randomly oriented downward-pointing ray that
    the test generates, so we get a non-trivial hit on every ray.
    """
    h = half_extent
    # Eight corners of a box centered at origin with side ``2*h``.
    corners = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float32,
    )
    # 12 triangles, two per face (winding doesn't matter for mesh_query_ray hit position).
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # -z
            [4, 6, 5],
            [4, 7, 6],  # +z
            [0, 5, 1],
            [0, 4, 5],  # -y
            [2, 6, 7],
            [2, 7, 3],  # +y
            [1, 5, 6],
            [1, 6, 2],  # +x
            [0, 3, 7],
            [0, 7, 4],  # -x
        ],
        dtype=np.int32,
    ).reshape(-1)
    points = wp.array(corners, dtype=wp.vec3f, device=device)
    indices = wp.array(faces, dtype=int, device=device)
    return wp.Mesh(points=points, indices=indices)


def _make_pattern(num_rays: int, device: str) -> tuple[wp.array, wp.array]:
    """A small grid of downward-pointing rays at z=+1, replicated per env."""
    side = int(math.isqrt(num_rays))
    assert side * side == num_rays, "num_rays must be a perfect square"
    xs = torch.linspace(-1.0, 1.0, side)
    ys = torch.linspace(-1.0, 1.0, side)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    starts = torch.stack([gx, gy, torch.full_like(gx, 1.0)], dim=-1).reshape(-1, 3)
    dirs = torch.zeros_like(starts)
    dirs[:, 2] = -1.0
    return starts.to(device), dirs.to(device)


def _random_transforms(num_envs: int, device: str, seed: int = 0):
    """Random per-env sensor poses + drifts + a single shared cfg offset.

    Sensor poses are per-env and far from origin (to stress fp32). The cfg-style offset
    is a *single* shared offset across envs, matching production usage where ``cfg.offset``
    is a scalar that gets folded into the (shared 1D) ray pattern at sensor init. Drifts
    are per-env (resampled per episode in the real sensor).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    transforms = torch.zeros(num_envs, 7, device=device)
    # Big translations (e.g. far envs in a 120 m × 120 m grid) so any precision regression shows up.
    transforms[:, :3] = (torch.rand((num_envs, 3), generator=g, device=device) - 0.5) * 4000.0
    rand_quat = torch.rand((num_envs, 4), generator=g, device=device) * 2.0 - 1.0
    rand_quat = rand_quat / rand_quat.norm(dim=-1, keepdim=True)
    transforms[:, 3:] = rand_quat
    # Single shared cfg offset.
    offset_pos = (torch.rand((3,), generator=g, device=device) - 0.5) * 0.5
    offset_q = torch.rand((4,), generator=g, device=device) * 2.0 - 1.0
    offset_q = offset_q / offset_q.norm()
    drift = (torch.rand((num_envs, 3), generator=g, device=device) - 0.5) * 0.05
    rcd = (torch.rand((num_envs, 3), generator=g, device=device) - 0.5) * 0.05
    return transforms, offset_pos, offset_q, drift, rcd


def _run_legacy(
    transforms_wp,
    env_mask,
    offset_pos_wp,
    offset_quat_wp,
    drift_wp,
    rcd_wp,
    ray_starts_local,
    ray_directions_local,
    alignment_mode: int,
    mesh_ids,
    mesh_positions,
    mesh_rotations,
    num_envs: int,
    num_rays: int,
    device: str,
):
    """Legacy two-kernel path: write world-frame buffers, then raycast."""
    pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
    quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
    ray_starts_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)
    ray_directions_w = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)
    ray_hits = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)
    ray_distance = wp.full(shape=(num_envs, num_rays), value=float("inf"), dtype=wp.float32, device=device)
    ray_normal = wp.empty((1, 1), dtype=wp.vec3, device=device)
    ray_face_id = wp.empty((1, 1), dtype=wp.int32, device=device)
    ray_mesh_id = wp.empty((1, 1), dtype=wp.int16, device=device)

    wp.launch(
        update_ray_caster_kernel,
        dim=(num_envs, num_rays),
        inputs=[
            transforms_wp,
            env_mask,
            offset_pos_wp,
            offset_quat_wp,
            drift_wp,
            rcd_wp,
            ray_starts_local,
            ray_directions_local,
            alignment_mode,
        ],
        outputs=[pos_w, quat_w, ray_starts_w, ray_directions_w],
        device=device,
    )
    wp.launch(
        raycast_dynamic_meshes_kernel,
        dim=(mesh_ids.shape[1], num_envs, num_rays),
        inputs=[
            env_mask,
            mesh_ids,
            ray_starts_w,
            ray_directions_w,
            ray_hits,
            ray_distance,
            ray_normal,
            ray_face_id,
            ray_mesh_id,
            mesh_positions,
            mesh_rotations,
            1e6,
            int(False),
            int(False),
            int(False),
        ],
        device=device,
    )
    return ray_hits


def _run_fused(
    transforms_wp,
    env_mask,
    drift_wp,
    rcd_wp,
    ray_starts_local,
    ray_directions_local,
    alignment_mode: int,
    mesh_ids,
    mesh_poses,
    num_envs: int,
    num_rays: int,
    device: str,
):
    """Fused path: combined world transform + raycast against a 1D shared pattern.

    The fused kernel also writes the packed sensor pose (``transformf``) from the
    (mesh=0, ray=0) thread of each env. We allocate that as scratch here since this
    verification only compares ray hits.
    """
    ray_hits = wp.zeros((num_envs, num_rays), dtype=wp.vec3f, device=device)
    ray_distance = wp.full(shape=(num_envs, num_rays), value=float("inf"), dtype=wp.float32, device=device)
    sensor_pose_scratch = wp.zeros(num_envs, dtype=wp.transformf, device=device)

    wp.launch(
        fast_terrain_raycast_kernel,
        dim=(mesh_ids.shape[1], num_envs, num_rays),
        inputs=[
            transforms_wp,
            env_mask,
            drift_wp,
            rcd_wp,
            ray_starts_local,
            ray_directions_local,
            alignment_mode,
            mesh_ids,
            mesh_poses,
            ray_hits,
            ray_distance,
            sensor_pose_scratch,
            1e6,
        ],
        device=device,
    )
    return ray_hits


def _compare(label: str, ray_hits_legacy: wp.array, ray_hits_fused: wp.array) -> tuple[float, bool]:
    legacy_t = wp.to_torch(ray_hits_legacy)
    fused_t = wp.to_torch(ray_hits_fused)
    diff = (legacy_t - fused_t).abs()
    max_abs = float(diff.max().item())
    bit_identical = bool(torch.equal(legacy_t, fused_t))
    print(
        f"  {label:<28} max |Δ| = {max_abs:.3e}   bit-identical: {bit_identical}"
        f"   (legacy mean |hit| = {legacy_t.abs().mean().item():.3f})"
    )
    return max_abs, bit_identical


def main() -> int:
    wp.init()
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available.")
        return 0

    device = "cuda:0"
    num_envs = 64
    num_rays = 16 * 16  # 256, perfect square
    num_meshes = 1

    print(f"Verifying fused raycast: num_envs={num_envs}, num_rays={num_rays}, num_meshes={num_meshes}")
    # Box has to be larger than the worst-case sensor translation below, otherwise rays miss
    # entirely and we'd be "verifying" all-zero outputs (which trivially match).
    box = _make_box_mesh(device, half_extent=5000.0)

    # Single shared mesh per env; pose at origin with identity rotation.
    mesh_ids = wp.array2d(np.full((num_envs, num_meshes), box.id, dtype=np.uint64), dtype=wp.uint64, device=device)
    mesh_positions = wp.zeros((num_envs, num_meshes), dtype=wp.vec3f, device=device)
    mesh_rotations = wp.zeros((num_envs, num_meshes), dtype=wp.quatf, device=device)
    # quaternion (0,0,0,1) is identity in Warp's (x,y,z,w) convention.
    rot_torch = wp.to_torch(mesh_rotations)
    rot_torch[..., 3] = 1.0
    # Packed mesh poses (transformf) for the fused path: same data laid out as a single
    # ``(N, n_meshes, 7)`` float32 tensor viewed as ``transformf``.
    mesh_poses_t = torch.zeros(num_envs, num_meshes, 7, device=device)
    mesh_poses_t[..., 6] = 1.0  # quaternion w = 1 (identity rotation)
    mesh_poses_wp = wp.from_torch(mesh_poses_t.contiguous()).view(wp.transformf)

    # Local-frame ray pattern. The cfg-style offset is folded into the pattern up-front,
    # exactly the way both upstream :class:`RayCaster._initialize_rays_impl` and our
    # :class:`FastTerrainScanner._initialize_rays` do at init. After this fold, both the
    # legacy and the fused kernels see *zero* sensor offset and an offset-baked pattern;
    # this mirrors production exactly.
    starts_1d, dirs_1d = _make_pattern(num_rays, device)

    transforms_t, off_pos, off_q, drift_t, rcd_t = _random_transforms(num_envs, device, seed=42)
    transforms_wp = wp.from_torch(transforms_t.contiguous()).view(wp.transformf)
    drift_wp = wp.from_torch(drift_t.contiguous(), dtype=wp.vec3f)
    rcd_wp = wp.from_torch(rcd_t.contiguous(), dtype=wp.vec3f)
    env_mask = wp.array(np.ones(num_envs, dtype=bool), dtype=wp.bool, device=device)

    # Fold cfg offset into the local pattern (init-time constant).
    import isaaclab.utils.math as math_utils  # noqa: PLC0415

    starts_folded = starts_1d + off_pos
    dirs_folded = math_utils.quat_apply(off_q.unsqueeze(0).expand(num_rays, 4), dirs_1d)

    # Legacy path consumes the per-env (2D) replicated pattern; fused path consumes the
    # 1D shared pattern directly. Same logical geometry on both sides.
    starts_2d = starts_folded.unsqueeze(0).expand(num_envs, -1, -1).contiguous()
    dirs_2d = dirs_folded.unsqueeze(0).expand(num_envs, -1, -1).contiguous()
    ray_starts_local = wp.from_torch(starts_2d, dtype=wp.vec3f)
    ray_directions_local = wp.from_torch(dirs_2d, dtype=wp.vec3f)
    ray_starts_local_shared = wp.from_torch(starts_folded.contiguous(), dtype=wp.vec3f)
    ray_directions_local_shared = wp.from_torch(dirs_folded.contiguous(), dtype=wp.vec3f)

    # Kernel-level sensor offset is identity in production for *both* paths (the cfg offset
    # was just folded into the pattern). Build matching identity per-env arrays for the
    # legacy kernel signature.
    offset_pos_wp = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
    identity_q = torch.zeros(num_envs, 4, device=device)
    identity_q[:, 3] = 1.0
    offset_quat_wp = wp.from_torch(identity_q.contiguous(), dtype=wp.quatf)

    overall_ok = True
    for label, mode in (("alignment=WORLD (0)", 0), ("alignment=YAW (1)", 1), ("alignment=BASE (2)", 2)):
        legacy = _run_legacy(
            transforms_wp,
            env_mask,
            offset_pos_wp,
            offset_quat_wp,
            drift_wp,
            rcd_wp,
            ray_starts_local,
            ray_directions_local,
            mode,
            mesh_ids,
            mesh_positions,
            mesh_rotations,
            num_envs,
            num_rays,
            device,
        )
        fused_shared = _run_fused(
            transforms_wp,
            env_mask,
            drift_wp,
            rcd_wp,
            ray_starts_local_shared,
            ray_directions_local_shared,
            mode,
            mesh_ids,
            mesh_poses_wp,
            num_envs,
            num_rays,
            device,
        )
        max_abs, _ = _compare(f"{label} fused-shared vs legacy", legacy, fused_shared)
        # Tolerance: world coords up to ~4000 m × 1e-6 fp32 epsilon ≈ 4 mm. The math in
        # ``compute_world_frame_ray`` was copied verbatim from upstream's
        # ``update_ray_caster_kernel``, so we expect bit-identical results, but allow 1e-3 m
        # as a loose ceiling for any compiler-level FMA reordering between the two kernels.
        if max_abs > 1e-3:
            overall_ok = False
            print(f"    [FAIL] {label}: max diff {max_abs:.3e} m > 1e-3 m")

    if overall_ok:
        print("\nfused == legacy across all alignment modes — fused kernel is verified.")
        return 0
    print("\n[FAIL] fused vs legacy mismatch in at least one alignment mode.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
