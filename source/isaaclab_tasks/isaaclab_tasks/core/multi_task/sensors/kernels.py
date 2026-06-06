# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the fast terrain scanner.

Two kernels:

* :func:`fill_inf_kernel` — initializes the per-step output / scratch buffers to inf for
  active envs. The main raycast kernel uses ``atomic_min`` on the distance buffer for
  closest-hit resolution across meshes, so the buffer must start at inf each step.
* :func:`fast_terrain_raycast_kernel` — single-launch fused kernel that, per (mesh, env, ray)
  thread, computes the sensor world pose, transforms the local ray to world frame, then
  to mesh frame, runs ``mesh_query_ray``, and writes the closest hit. The sensor pose
  output (``sensor_pose_w``) is written by the (mesh_id=0, ray_id=0) thread for each env,
  eliminating a separate dim=num_envs pose-only kernel launch.

Buffers grouped by what they describe (so the kernel signature reads as semantic units):

* **Sensor pose** — single ``transformf`` per env (translation + quaternion).
* **Mesh pose** — single ``transformf`` per (env, mesh).
* **Drift** — two per-env ``vec3f`` arrays kept separate (sensor-position drift vs.
  ray-cast-pattern drift) since they're applied at different points in the math.
* **Ray pattern** — 1D ``vec3f`` arrays (start + direction), shared across envs.

The math is bit-identical to upstream :func:`isaaclab.sensors.ray_caster.kernels.update_ray_caster_kernel`
+ :func:`isaaclab.utils.warp.kernels.raycast_dynamic_meshes_kernel`. The local pattern is
indexed as 1D — one pattern entry per ray, shared across envs — so no per-env replication.
The cfg-defined sensor offset is folded into the local ray pattern at sensor init time
(``_initialize_rays``), matching upstream's design and removing two no-op identity buffers
from the kernel signature.
"""

from __future__ import annotations

import warp as wp

# Alignment-mode constants.
ALIGNMENT_WORLD = wp.constant(0)
ALIGNMENT_YAW = wp.constant(1)
ALIGNMENT_BASE = wp.constant(2)

# ``+inf`` constant — Warp codegen rejects ``float("inf")`` inside kernels (no overload of the
# ``float`` builtin for string args), so we materialize it at module load time.
INF_F = wp.constant(float("inf"))


@wp.func
def quat_yaw_only(q: wp.quatf) -> wp.quatf:
    """Yaw-only quaternion: project a general orientation onto the rotation about ``z``."""
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    half_yaw = yaw * 0.5
    return wp.quatf(0.0, 0.0, wp.sin(half_yaw), wp.cos(half_yaw))


@wp.func
def compute_world_frame_ray(
    sensor_pose: wp.transformf,
    drift: wp.vec3f,
    ray_cast_drift: wp.vec3f,
    local_start: wp.vec3f,
    local_dir: wp.vec3f,
    alignment_mode: int,
):
    """Transform a local-frame ray into world frame using the sensor pose.

    The cfg sensor offset is already baked into ``local_start`` / ``local_dir`` at sensor
    init, so this function only applies the runtime sensor pose + per-episode drifts.

    Returns ``(combined_pos, combined_quat, ray_start_w, ray_direction_w)``. Math mirrors
    upstream ``update_ray_caster_kernel`` with the (always-identity) view-to-sensor offset
    elided.
    """
    combined_pos = wp.transform_get_translation(sensor_pose) + drift
    combined_quat = wp.transform_get_rotation(sensor_pose)

    if alignment_mode == ALIGNMENT_WORLD:
        pos_drifted = wp.vec3f(
            combined_pos[0] + ray_cast_drift[0],
            combined_pos[1] + ray_cast_drift[1],
            combined_pos[2],
        )
        ray_start_w = local_start + pos_drifted
        ray_direction_w = local_dir
    elif alignment_mode == ALIGNMENT_YAW:
        yaw_q = quat_yaw_only(combined_quat)
        rot_drift = wp.quat_rotate(yaw_q, ray_cast_drift)
        pos_drifted = wp.vec3f(
            combined_pos[0] + rot_drift[0],
            combined_pos[1] + rot_drift[1],
            combined_pos[2],
        )
        ray_start_w = wp.quat_rotate(yaw_q, local_start) + pos_drifted
        ray_direction_w = local_dir
    else:
        rot_drift = wp.quat_rotate(combined_quat, ray_cast_drift)
        pos_drifted = wp.vec3f(
            combined_pos[0] + rot_drift[0],
            combined_pos[1] + rot_drift[1],
            combined_pos[2],
        )
        ray_start_w = wp.quat_rotate(combined_quat, local_start) + pos_drifted
        ray_direction_w = wp.quat_rotate(combined_quat, local_dir)

    return combined_pos, combined_quat, ray_start_w, ray_direction_w


@wp.kernel(enable_backward=False)
def fill_inf_kernel(
    env_mask: wp.array(dtype=wp.bool),
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
):
    """Initialize ``ray_hits`` (vec3 of inf) and ``ray_distance`` (float inf) for active envs.

    Launch with ``dim=(num_envs, num_rays)``.
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    ray_hits[env, ray] = wp.vec3f(INF_F, INF_F, INF_F)
    ray_distance[env, ray] = INF_F


@wp.kernel(enable_backward=False)
def fast_terrain_raycast_kernel(
    # per-env sensor pose (frame-view world pose, packed translation + quaternion)
    sensor_poses_in: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    # per-env drifts — kept separate because they're applied at different math points
    drift: wp.array(dtype=wp.vec3f),
    ray_cast_drift: wp.array(dtype=wp.vec3f),
    # 1D shared local-frame ray pattern (sensor offset already folded in at init)
    ray_starts_local: wp.array(dtype=wp.vec3f),
    ray_directions_local: wp.array(dtype=wp.vec3f),
    alignment_mode: int,
    # per-(env, mesh) mesh handle + world pose (packed translation + quaternion)
    mesh: wp.array2d(dtype=wp.uint64),
    mesh_poses: wp.array2d(dtype=wp.transformf),
    # per-(env, ray) raycast outputs / scratch
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
    # per-env sensor pose output (written by the (mesh=0, ray=0) thread per env)
    sensor_poses_out: wp.array(dtype=wp.transformf),
    # constants
    max_dist: float,
):
    """Single fused kernel: sensor pose + world-frame ray transform + multi-mesh closest-hit.

    Avoids the upstream four-launch sequence (pose update / fill_inf_hits /
    fill_inf_distance / raycast). The pose is written by the (mesh_id=0, ray_id=0) thread
    of each env; ``ray_hits`` / ``ray_distance`` must be inf-initialized by
    :func:`fill_inf_kernel` *before* this kernel since ``atomic_min`` on the distance
    buffer needs a fresh starting value each step.

    Launch with ``dim=(n_meshes, num_envs, num_rays)``.
    """
    tid_mesh_id, tid_env, tid_ray = wp.tid()
    if not env_mask[tid_env]:
        return

    combined_pos, combined_quat, ray_start_w, ray_direction_w = compute_world_frame_ray(
        sensor_poses_in[tid_env],
        drift[tid_env],
        ray_cast_drift[tid_env],
        ray_starts_local[tid_ray],
        ray_directions_local[tid_ray],
        alignment_mode,
    )

    # Sensor pose: written once per env by the first-mesh-first-ray thread.
    if tid_mesh_id == 0 and tid_ray == 0:
        sensor_poses_out[tid_env] = wp.transform(combined_pos, combined_quat)

    # Mesh-local ray, mesh_query_ray, atomic-min closest-hit. Same logic as upstream
    # ``raycast_dynamic_meshes_kernel`` from this point on.
    mesh_pose = mesh_poses[tid_env, tid_mesh_id]
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    direction = wp.transform_vector(mesh_pose_inv, ray_direction_w)
    start_pos = wp.transform_point(mesh_pose_inv, ray_start_w)

    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)
    if mesh_query_ray_t.result:
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # TODO(warp#1058): atomic_min return value is wrong on GPU; the equality check below
        # is racy when two meshes tie on distance. Hit *position* is still correct because
        # all tying threads compute the same world-space point.
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            hit_pos = start_pos + mesh_query_ray_t.t * direction
            ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)
