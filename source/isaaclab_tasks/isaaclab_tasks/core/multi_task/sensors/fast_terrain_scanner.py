# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fast height-scanner sensor specialized for static terrain.

This is a *fresh* implementation — it does not inherit from
:class:`isaaclab.sensors.ray_caster.MultiMeshRayCaster`. The only base class is
:class:`isaaclab.sensors.SensorBase`, which provides the env sensor lifecycle
hooks (``_initialize_callback`` / ``_invalidate_initialize_callback``,
``_update_outdated_buffers``, etc.). Everything else is local to this module so
a future rebase onto a new IsaacLab version requires no merge work in shared
sensor code — delete this folder or re-point the cfg back to upstream.

Buffer grouping (semantic packing for kernel signature clarity + memory locality):

* **Sensor pose** — single ``transformf`` per env (translation + quaternion combined).
  ``data.pos_w.torch`` and ``data.quat_w.torch`` are zero-copy slices of the same
  underlying ``[N, 7]`` float32 tensor (slots ``[..., :3]`` and ``[..., 3:]``).
* **Mesh pose** — single ``transformf`` per (env, mesh), built once at init from each
  terrain prim's resolved world pose.
* **Drift** — kept as two per-env ``vec3`` arrays (``drift`` + ``ray_cast_drift``);
  they enter the world-frame transform at different points in the math, so packing
  them would just obscure the kernel reads.
* **Sensor cfg offset** — folded into the local ray pattern at sensor init
  (``_initialize_rays``) instead of carried as identity per-env arrays into the kernel.

What's specialized vs upstream :class:`MultiMeshRayCaster`:

* **Static-terrain assumption.** No ``_update_mesh_transforms`` per step; per-env
  mesh world poses are resolved once and held constant.
* **Single fused kernel for the per-step work.** Sensor pose write + world-frame ray
  transform + multi-mesh closest-hit raycast in one launch. The pose is written by
  the (mesh_id=0, ray_id=0) thread of each env, eliminating the upstream pose-only
  kernel launch.
* **Shared 1D local-frame ray pattern.** Pattern stored once (``num_rays``) instead of
  replicated per env (``num_envs * num_rays``).
* **No persistent world-frame ray buffers.** Computed in registers per-thread inside
  the raycast kernel — saves ``num_envs * num_rays * 24 B`` of GPU memory.
* **Light-weight data exposure.** ``data.pos_w`` / ``data.quat_w`` /
  ``data.ray_hits_w`` are tiny ``_TorchView`` wrappers with a single ``.torch``
  attribute. Same public surface as upstream's :class:`~isaaclab.utils.warp.ProxyArray`
  but no lazy-cache machinery.
* **Pruned cfg.** No ``RaycastTargetCfg`` / ``track_mesh_transforms`` /
  ``merge_prim_meshes`` / ``is_shared`` / ``reference_meshes`` /
  ``update_mesh_ids`` knobs — those configure features we don't use.

Math: bit-identical to upstream legacy raycast for the height-scan pattern.
``compute_world_frame_ray`` in :mod:`.kernels` mirrors upstream
``update_ray_caster_kernel`` (with the always-identity sensor offset elided); the
multi-mesh closest-hit loop is copied from upstream ``raycast_dynamic_meshes_kernel``
with no logic changes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.mesh import create_trimesh_from_geom_mesh
from isaaclab.utils.warp import convert_to_warp_mesh

from .kernels import fast_terrain_raycast_kernel, fill_inf_kernel

if TYPE_CHECKING:
    from .fast_terrain_scanner_cfg import FastTerrainScannerCfg


class _TorchView:
    """Minimal stand-in for :class:`~isaaclab.utils.warp.ProxyArray`'s ``.torch`` interface.

    The obs term reads ``sensor.data.ray_hits_w.torch[..., 2]`` (and similar). Upstream
    returns a ProxyArray with lazy torch caching; we cache eagerly at init since the
    tensors live for the sensor lifetime, eliminating per-access dispatch overhead.
    """

    __slots__ = ("torch",)

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class FastTerrainScannerData:
    """Light-weight data container — ``.pos_w`` / ``.quat_w`` / ``.ray_hits_w`` only."""

    __slots__ = ("pos_w", "quat_w", "ray_hits_w")

    def __init__(self, pos_w_torch: torch.Tensor, quat_w_torch: torch.Tensor, ray_hits_w_torch: torch.Tensor):
        self.pos_w = _TorchView(pos_w_torch)
        self.quat_w = _TorchView(quat_w_torch)
        self.ray_hits_w = _TorchView(ray_hits_w_torch)


class FastTerrainScanner(SensorBase):
    """Height-scanner sensor specialized for static terrain. See module docstring."""

    cfg: FastTerrainScannerCfg
    """The configuration parameters."""

    # Process-global cache: parse each unique ``(prim_path, device)`` mesh once. Mirrors the
    # caching in upstream ``RayCaster.meshes`` so re-instancing the sensor doesn't re-parse.
    meshes: ClassVar[dict[tuple[str, str], wp.Mesh]] = {}

    def __init__(self, cfg: FastTerrainScannerCfg):
        super().__init__(cfg)
        # ``body_prim_path`` is the physics-body path the scanner is attached to. The pose is
        # read straight from that body's articulation tensors (see ``bind_articulation`` /
        # ``_get_view_transforms_wp``), so — like the upstream ray casters — no sensor prim is
        # spawned and ``SensorBase._initialize_impl`` resolves ``num_envs`` from ``cfg.prim_path``.
        self.body_prim_path = cfg.prim_path.rstrip("/")
        # data is created at the end of ``_initialize_impl`` once we know num_rays.
        self._data: FastTerrainScannerData | None = None  # type: ignore[assignment]
        self._articulation = None
        self._body_idx: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_articulation(self, articulation, body_name: str) -> None:
        """Bind the sensor pose to a specific articulation body.

        The scanner reads the body pose directly from the :class:`Articulation`'s
        GPU-backed ``body_pos_w`` / ``body_quat_w`` tensors, which are always
        up-to-date.

        Args:
            articulation: The :class:`Articulation` asset (e.g. ``env.scene["robot"]``).
            body_name: Name of the body the sensor is attached to (e.g. ``"base"``).
        """
        body_ids, _ = articulation.find_bodies(body_name)
        if not body_ids:
            raise ValueError(f"Body '{body_name}' not found in articulation.")
        self._body_idx = body_ids[0]
        self._articulation = articulation

    @property
    def num_instances(self) -> int:
        return self._view_count

    @property
    def data(self) -> FastTerrainScannerData:
        # Lazy-update: matches upstream's behavior of refreshing data on access.
        self._update_outdated_buffers()
        return self._data  # type: ignore[return-value]

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset per-env drift values and timing state."""
        super().reset(env_ids, env_mask)
        if env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
            num_envs_ids = len(env_ids)
        elif env_ids is not None:
            num_envs_ids = len(env_ids)
        else:
            env_ids = slice(None)
            num_envs_ids = self._view_count

        # Resample sensor-position drift.
        self._drift_torch[env_ids] = torch.empty(num_envs_ids, 3, device=self._device).uniform_(*self.cfg.drift_range)
        # Resample per-axis ray-cast drift.
        ranges = torch.tensor(
            [self.cfg.ray_cast_drift_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=self._device
        )
        self._ray_cast_drift_torch[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self._device
        )

    # ------------------------------------------------------------------
    # SensorBase hooks
    # ------------------------------------------------------------------

    def _initialize_impl(self):
        super()._initialize_impl()
        self._view_count = self._num_envs

        # Resolve alignment mode once at init.
        self._alignment_mode = {"world": 0, "yaw": 1, "base": 2}[self.cfg.ray_alignment]

        # Build warp meshes from cfg.mesh_prim_paths and produce the per-env mesh-id table
        # plus the per-(env, mesh) ``transformf`` pose array.
        self._initialize_warp_meshes()

        # Pattern + drift buffers + raycast output buffer.
        self._initialize_rays()

        # Lightweight data view — ``pos_w`` / ``quat_w`` are zero-copy slices into the
        # single ``_sensor_pose_w`` ``transformf`` array (laid out as ``[N, 7]`` float32:
        # the first 3 components are translation, the last 4 are the quaternion).
        sensor_pose_torch = wp.to_torch(self._sensor_pose_w)
        self._data = FastTerrainScannerData(
            pos_w_torch=sensor_pose_torch[..., :3],
            quat_w_torch=sensor_pose_torch[..., 3:],
            ray_hits_w_torch=wp.to_torch(self._ray_hits_w),
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Single-step buffer fill: inf-init + fused raycast."""
        # Inf-init scratch for atomic-min closest-hit (one combined kernel for hits + distance).
        wp.launch(
            fill_inf_kernel,
            dim=(self._view_count, self.num_rays),
            inputs=[env_mask, self._ray_hits_w, self._ray_distance_w],
            device=self._device,
        )

        # Fused: pose write + world-frame ray transform + multi-mesh closest-hit raycast.
        sensor_poses_in = self._get_view_transforms_wp()
        wp.launch(
            fast_terrain_raycast_kernel,
            dim=(self._mesh_ids_wp.shape[1], self._view_count, self.num_rays),
            inputs=[
                sensor_poses_in,
                env_mask,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                self._alignment_mode,
                self._mesh_ids_wp,
                self._mesh_poses_w,
                self._ray_hits_w,
                self._ray_distance_w,
                self._sensor_pose_w,
                float(self.cfg.max_distance),
            ],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/toggle the ray-hit point-cloud markers. Mirrors :class:`RayCaster`."""
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw the current world-frame ray hits, skipping unset (inf) entries."""
        # Buffers are created in ``_initialize_impl``; the callback can fire before then.
        if self._data is None or getattr(self, "_ray_hits_w", None) is None:
            return
        viz_points = wp.to_torch(self._ray_hits_w).reshape(-1, 3)
        # Drop rays that did not hit any mesh (left at inf by ``fill_inf_kernel``).
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points.shape[0] == 0:
            return
        self.ray_visualizer.visualize(viz_points)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _initialize_warp_meshes(self):
        """Upload source Warp meshes and expand them over the clone-plan env count."""
        per_env_mesh_ids: list[list[int]] = [[] for _ in range(self._view_count)]
        # Each pose row is laid out as ``(tx, ty, tz, qx, qy, qz, qw)`` for ``wp.transformf``.
        per_env_poses: list[list[tuple[float, ...]]] = [[] for _ in range(self._view_count)]

        for prim_expr in self.cfg.mesh_prim_paths:
            prim_expr = prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*")
            target_prims = sim_utils.find_matching_prims(prim_expr)
            if not target_prims:
                raise RuntimeError(f"FastTerrainScanner: no prims matched '{prim_expr}'.")

            # ``len(target_prims) == 1`` ⇒ a source/global prim shared by all backend clones.
            # Otherwise we expect one authored mesh per clone-plan environment.
            is_global = len(target_prims) == 1
            if not is_global and len(target_prims) != self._view_count:
                raise RuntimeError(
                    f"FastTerrainScanner: '{prim_expr}' matched {len(target_prims)} prims; expected"
                    f" 1 (source/global) or {self._view_count} (per-env)."
                )

            for env_idx, target_prim in enumerate(target_prims):
                wp_mesh = self._build_or_lookup_mesh(target_prim)
                pos, quat = sim_utils.resolve_prim_pose(target_prim)
                pose_row = (*pos, *quat)
                if is_global:
                    for env_id in range(self._view_count):
                        per_env_mesh_ids[env_id].append(wp_mesh.id)
                        per_env_poses[env_id].append(pose_row)
                else:
                    per_env_mesh_ids[env_idx].append(wp_mesh.id)
                    per_env_poses[env_idx].append(pose_row)

        n_meshes_per_env = len(per_env_mesh_ids[0])
        if any(len(ids) != n_meshes_per_env for ids in per_env_mesh_ids):
            raise RuntimeError("FastTerrainScanner: per-env mesh count mismatch across cfg.mesh_prim_paths.")

        # ``_mesh_ids_wp`` is bound directly to the kernel as ``mesh: wp.array2d(uint64)``.
        ids_np = np.asarray(per_env_mesh_ids, dtype=np.uint64)
        self._mesh_ids_wp = wp.array2d(ids_np, dtype=wp.uint64, device=self._device)

        # Per-(env, mesh) world pose, resolved once at init then held constant. The kernel
        # reads ``mesh_poses[env, mesh]`` directly as a ``transformf``.
        poses_t = torch.tensor(per_env_poses, device=self._device, dtype=torch.float32).contiguous()
        self._mesh_poses_w = wp.from_torch(poses_t).view(wp.transformf)
        # Hold strong reference so the underlying tensor storage isn't freed.
        self._mesh_poses_w_storage = poses_t

    def _build_or_lookup_mesh(self, target_prim) -> wp.Mesh:
        """Parse a single prim's first child ``Mesh`` into a Warp mesh, with global cache.

        Per-env terrain prims (``/World/envs/env_0/ground``, ``/World/envs/env_1/ground``, …)
        carry identical *local* mesh data — only their world-frame transform differs. Caching
        on the full path would upload one Warp mesh + BVH per env (e.g. ~14 MB × 4096 ≈ 57 GB),
        blowing GPU memory; instead we normalize ``env_N`` → ``env_0`` in the cache key so all
        envs share a single upload. Per-env world poses are still resolved separately by the
        caller via :func:`resolve_prim_pose`. Mirrors upstream
        :class:`MultiMeshRayCaster._build_warp_meshes`'s ``is_shared=True`` path.
        """
        prim_path = str(target_prim.GetPath())
        canonical_path = re.sub(r"/env_\d+/", "/env_0/", prim_path)
        cache_key = (canonical_path, self._device)
        if cache_key in FastTerrainScanner.meshes:
            return FastTerrainScanner.meshes[cache_key]

        mesh_prim = sim_utils.get_first_matching_child_prim(target_prim.GetPath(), lambda p: p.GetTypeName() == "Mesh")
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"FastTerrainScanner: no child Mesh prim under '{prim_path}'.")

        trimesh_mesh = create_trimesh_from_geom_mesh(mesh_prim)
        wp_mesh = convert_to_warp_mesh(trimesh_mesh.vertices, trimesh_mesh.faces, device=self._device)
        FastTerrainScanner.meshes[cache_key] = wp_mesh
        return wp_mesh

    def _initialize_rays(self):
        """Allocate the 1D ray pattern, drift buffers, and per-step output buffers.

        The cfg sensor offset (``cfg.offset.pos`` / ``cfg.offset.rot``) is folded into the
        local ray pattern here, so the kernel never sees it as a separate input. The kernel
        only applies the runtime sensor pose plus per-episode drifts.
        """
        # Pattern (1D — shared across envs, no per-env replication).
        ray_starts_torch, ray_directions_torch = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(ray_directions_torch)

        # Apply sensor offset to the pattern up-front (init-time constant).
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        ray_directions_torch = math_utils.quat_apply(
            offset_quat.repeat(len(ray_directions_torch), 1), ray_directions_torch
        )
        ray_starts_torch = ray_starts_torch + offset_pos

        self._ray_starts_local = wp.from_torch(ray_starts_torch.contiguous(), dtype=wp.vec3f)
        self._ray_directions_local = wp.from_torch(ray_directions_torch.contiguous(), dtype=wp.vec3f)

        # Drift buffers — Warp arrays with zero-copy torch views for ``reset()`` indexing.
        self._drift = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        self._ray_cast_drift = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        self._drift_torch = wp.to_torch(self._drift)
        self._ray_cast_drift_torch = wp.to_torch(self._ray_cast_drift)

        # Per-env sensor pose output (translation + quaternion combined into a single
        # ``transformf``). The data class slices this into ``pos_w`` / ``quat_w`` torch views.
        self._sensor_pose_w = wp.zeros(self._view_count, dtype=wp.transformf, device=self._device)
        self._ray_hits_w = wp.zeros((self._view_count, self.num_rays), dtype=wp.vec3f, device=self._device)
        # ``atomic_min`` closest-hit scratch — re-init'd each step by ``fill_inf_kernel``.
        self._ray_distance_w = wp.zeros((self._view_count, self.num_rays), dtype=wp.float32, device=self._device)

    def _get_view_transforms_wp(self) -> wp.array:
        """Pack sensor poses into a Warp ``transformf`` array (tx, ty, tz, qx, qy, qz, qw).

        Reads body poses directly from the articulation's GPU-backed tensors
        (always fresh from the simulation backend).
        """
        if self._body_idx is None or self._articulation is None:
            raise RuntimeError("FastTerrainScanner requires bind_articulation() before sensor update.")
        pos_torch = self._articulation.data.body_pos_w.torch[:, self._body_idx]
        quat_torch = self._articulation.data.body_quat_w.torch[:, self._body_idx]
        if pos_torch.shape[0] != self._view_count or quat_torch.shape[0] != self._view_count:
            raise RuntimeError(
                "FastTerrainScanner articulation body tensors do not match the clone-plan environment count: "
                f"{pos_torch.shape[0]} poses for {self._view_count} envs."
            )
        poses = torch.cat([pos_torch, quat_torch], dim=-1).contiguous()
        return wp.from_torch(poses).view(wp.transformf)
