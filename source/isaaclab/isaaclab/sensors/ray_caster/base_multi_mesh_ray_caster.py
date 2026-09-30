# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab import cloner
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.warp import ProxyArray, convert_to_warp_mesh
from isaaclab.utils.warp import kernels as warp_kernels

from .base_ray_caster import BaseRayCaster, _inverse_transform_vertices, _transform_vertices
from .kernels import copy_mesh_poses_to_table_kernel, fill_ray_hits_distance_inf_kernel
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData

if TYPE_CHECKING:
    from isaaclab.cloner.clone_plan import ClonePlan, GeometryLayout

    from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

logger = logging.getLogger(__name__)


class BaseMultiMeshRayCaster(BaseRayCaster):
    """A multi-mesh ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    Mesh geometry declared by the clone plan is converted to Warp meshes and stored in the
    :attr:`meshes` dictionary. The ray-caster casts the configured ray pattern against those meshes.

    Compared to the default RayCaster, the MultiMeshRayCaster provides additional functionality and flexibility as
    an extension of the default RayCaster with the following enhancements:

    - Raycasting against multiple target types : Supports primitive shapes (spheres, cubes, etc.) as well as arbitrary
      meshes.
    - Dynamic mesh tracking : Keeps track of specified meshes, enabling raycasting against moving parts
      (e.g., robot links, articulated bodies, or dynamic obstacles).
    - Memory-efficient caching : Avoids redundant memory usage by reusing mesh data across environments.

    .. warning::
        **Known limitation (multi-mesh closest-hit resolution):** When two meshes produce a
        hit at the exact same distance for a given ray, the ``atomic_min`` + equality-check
        pattern in the raycasting kernel is not fully thread-safe. The hit *position* is always
        correct, but auxiliary outputs (normals, face IDs, mesh IDs) may originate from
        different meshes for the affected ray. This requires an exact floating-point tie and is
        rare in practice. See `warp#1058 <https://github.com/NVIDIA/warp/issues/1058>`_ for
        upstream progress on a thread-safe ``atomic_min`` return value.

    Example usage to raycast against the visual meshes of a robot (e.g. ANYmal):

    .. code-block:: python

        ray_caster_cfg = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            mesh_prim_paths=[
                "/World/Ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LF_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RF_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LH_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RH_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/base/visuals"),
            ],
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(2.5, 2.5), direction=(0, 0, -1)),
        )

    """

    cfg: MultiMeshRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: MultiMeshRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._num_meshes_per_env: dict[str, int] = {}

        self._raycast_targets_cfg: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = []
        for target in self.cfg.mesh_prim_paths:
            if isinstance(target, str):
                target_cfg = cfg.RaycastTargetCfg(prim_expr=target, track_mesh_transforms=False)
            else:
                target_cfg = target
            target_cfg.prim_expr = cloner.expand_env_regex_ns(target_cfg.prim_expr)
            self._raycast_targets_cfg.append(target_cfg)

        self._data = MultiMeshRayCasterData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())} \n"
            f"\tnumber of sensors    : {self._view_count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view_count}"
        )

    """
    Properties
    """

    @property
    def data(self) -> MultiMeshRayCasterData:
        self._update_outdated_buffers()
        return self._data

    """
    Implementation.
    """

    def _initialize_warp_meshes(self):
        """Initialize mesh buffers exclusively from the completed clone plan."""
        plan = SimulationContext.instance().get_clone_plan()
        if plan is None or not plan.is_complete or plan.env_ids is None:
            raise RuntimeError(f"RayCaster at {self.cfg.prim_path!r} requires a completed clone plan.")
        env_to_index = {int(env_id): index for index, env_id in enumerate(plan.env_ids.tolist())}
        target_records = []
        dummy_mesh_id: int | None = None
        self._mesh_views = []

        for target_cfg in self._raycast_targets_cfg:
            records_per_env, dummy_mesh_id, tracked_body_paths = self._build_mesh_records(
                target_cfg, plan, env_to_index, dummy_mesh_id
            )
            widths = {len(records) for records in records_per_env}
            if target_cfg.track_mesh_transforms and len(widths) != 1:
                raise ValueError(
                    f"Tracked target {target_cfg.prim_expr!r} has different mesh counts across environments."
                )
            width = max(widths)
            self._num_meshes_per_env[target_cfg.prim_expr] = width
            target_records.append(records_per_env)
            self._mesh_views.append(
                self._create_tracked_target_view(tracked_body_paths) if target_cfg.track_mesh_transforms else None
            )

        if dummy_mesh_id is None:
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

        total_meshes_per_env = sum(
            self._num_meshes_per_env[target_cfg.prim_expr] for target_cfg in self._raycast_targets_cfg
        )
        mesh_ids = np.full((self._num_envs, total_meshes_per_env), dummy_mesh_id, dtype=np.uint64)
        mesh_positions = np.full((self._num_envs, total_meshes_per_env, 3), 1.0e9, dtype=np.float32)
        mesh_orientations = np.zeros((self._num_envs, total_meshes_per_env, 4), dtype=np.float32)
        mesh_orientations[..., 3] = 1.0

        mesh_offset = 0
        for target_cfg, records_per_env in zip(self._raycast_targets_cfg, target_records):
            target_width = self._num_meshes_per_env[target_cfg.prim_expr]
            for env_id, records in enumerate(records_per_env):
                if not records:
                    continue
                count = len(records)
                record_mesh_ids, record_positions, record_orientations = zip(*records)
                target_slice = slice(mesh_offset, mesh_offset + count)
                mesh_ids[env_id, target_slice] = np.asarray(record_mesh_ids, dtype=np.uint64)
                mesh_positions[env_id, target_slice] = np.asarray(record_positions, dtype=np.float32)
                mesh_orientations[env_id, target_slice] = np.asarray(record_orientations, dtype=np.float32)
            mesh_offset += target_width

        self._mesh_ids_wp = wp.array2d(mesh_ids, dtype=wp.uint64, device=self.device)
        self._mesh_positions_w = wp.array2d(mesh_positions, dtype=wp.vec3f, device=self.device)
        self._mesh_orientations_w = wp.array2d(mesh_orientations, dtype=wp.quatf, device=self.device)

    def _build_mesh_records(
        self,
        target_cfg: MultiMeshRayCasterCfg.RaycastTargetCfg,
        plan: ClonePlan,
        env_to_index: dict[int, int],
        dummy_mesh_id: int | None,
    ):
        """Build per-environment mesh records from exact planned geometry."""
        records_per_env = [[] for _ in range(self._num_envs)]
        tracked_body_paths = []
        for target, geometries in plan.match_geometry_targets(target_cfg.prim_expr):
            env_indices = range(self._num_envs) if target.env_id is None else (env_to_index[target.env_id],)
            if target_cfg.track_mesh_transforms:
                by_body: dict[str, list[GeometryLayout]] = {}
                for geometry in geometries:
                    body_path = geometry.frame.body_path
                    if body_path is None:
                        raise ValueError(
                            f"Tracked ray-cast geometry {geometry.path!r} has no planned rigid-body binding."
                        )
                    by_body.setdefault(body_path, []).append(geometry)
                for body_path, body_geometries in by_body.items():
                    body_frames = [geometry.frame for geometry in body_geometries]
                    mesh_id = self._load_planned_mesh(
                        body_geometries,
                        [
                            _transform_vertices(geometry.vertices, frame.pose)
                            for geometry, frame in zip(body_geometries, body_frames)
                        ],
                        body_frames[0].body_view_path,
                        target_cfg,
                    )
                    for env_index in env_indices:
                        records_per_env[env_index].append((mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0)))
                    tracked_body_paths.append(body_path)
                    dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                continue

            has_bound_geometry = any(geometry.frame.body_path is not None for geometry in geometries)
            if target.body_path is not None or has_bound_geometry:
                raise ValueError(f"Static ray-cast target {target.path!r} contains rigid-body-bound geometry.")
            mesh_id = self._load_planned_mesh(
                geometries,
                [
                    _inverse_transform_vertices(
                        _transform_vertices(geometry.vertices, geometry.frame.pose), target.pose
                    )
                    for geometry in geometries
                ],
                target.source_path,
                target_cfg,
            )
            record = (mesh_id, target.pose[:3], target.pose[3:])
            for env_index in env_indices:
                records_per_env[env_index].append(record)
            dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id

        if target_cfg.track_mesh_transforms and len(tracked_body_paths) != len(set(tracked_body_paths)):
            raise ValueError(f"Tracked target {target_cfg.prim_expr!r} resolves multiple meshes on one rigid body.")
        return records_per_env, dummy_mesh_id, tracked_body_paths

    def _load_planned_mesh(
        self,
        geometries: tuple[GeometryLayout, ...] | list[GeometryLayout],
        vertices: list[np.ndarray],
        reference: str | None,
        target_cfg: MultiMeshRayCasterCfg.RaycastTargetCfg,
    ) -> int:
        """Create or reuse one Warp mesh from planned geometry."""
        prim_key = (f"{'|'.join(geometry.source_path for geometry in geometries)}@{reference}", self._device)
        if prim_key in BaseMultiMeshRayCaster.meshes:
            return BaseMultiMeshRayCaster.meshes[prim_key].id
        faces, offset = [], 0
        for geometry in geometries:
            faces.append(geometry.faces + offset)
            offset += len(geometry.vertices)
        points = np.concatenate(vertices)
        triangles = np.concatenate(faces)
        wp_mesh = convert_to_warp_mesh(points, triangles, device=self._device)
        BaseMultiMeshRayCaster.meshes[prim_key] = wp_mesh
        logger.info(
            f"Loaded {len(geometries)} planned geometries with {len(points)} vertices below"
            f" ray-cast target {target_cfg.prim_expr!r}."
        )
        return wp_mesh.id

    def _create_tracked_target_view(self, target_prim_paths: str | list[str]):
        raise NotImplementedError("Tracked multi-mesh targets must be implemented by the active physics backend.")

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # Persistent buffer for tracking closest-hit distance across meshes (for atomic_min)
        self._ray_distance_wp = wp.empty((self._view_count, self.num_rays), dtype=wp.float32, device=self._device)
        if self.cfg.update_mesh_ids:
            self._data.ray_mesh_ids = ProxyArray(
                wp.zeros((self._view_count, self.num_rays), dtype=wp.int16, device=self._device)
            )
        else:
            # Dummy 1×1 buffer so the kernel launch always has a valid array to bind
            self._ray_mesh_id_wp = wp.empty((1, 1), dtype=wp.int16, device=self._device)
        # Persistent dummy buffers for unused kernel outputs; allocated once to avoid per-step allocations.
        self._dummy_normal_wp = wp.empty((1, 1), dtype=wp.vec3, device=self._device)
        self._dummy_face_id_wp = wp.empty((1, 1), dtype=wp.int32, device=self._device)

    def _update_mesh_transforms(self) -> None:
        """Update world-frame mesh positions and orientations for dynamically tracked targets.

        Iterates over all tracked views and writes the current world poses into
        the rectangular mesh pose buffers. Static (non-tracked) targets are
        skipped; their initial poses were set during :meth:`_initialize_warp_meshes`.
        """
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            pos_w, ori_w = view.get_world_poses(None)
            view_count = getattr(view, "count", pos_w.warp.shape[0])
            meshes_per_env = view_count
            if view_count != 1:
                # Backend views return a flat list across envs; the mesh table is indexed per env.
                meshes_per_env = view_count // self._num_envs

            wp.launch(
                copy_mesh_poses_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    pos_w.warp,
                    ori_w.warp,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(view_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()

        # Fill output and distance buffers with inf for masked environments
        wp.launch(
            fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, False],
            outputs=[self._data._ray_hits_w, self._ray_distance_wp, self._dummy_normal_wp],
            device=self._device,
        )

        n_meshes = self._mesh_ids_wp.shape[1]
        return_normal = False
        return_face_id = False
        write_mesh_ids = self.cfg.update_mesh_ids

        # Ray-cast against all meshes; closest hit wins via atomic_min on ray_distance.
        wp.launch(
            warp_kernels.raycast_dynamic_meshes_kernel,
            dim=(n_meshes, self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._data._ray_hits_w,
                self._ray_distance_wp,
                self._dummy_normal_wp,
                self._dummy_face_id_wp,
                self._data.ray_mesh_ids.warp if self.cfg.update_mesh_ids else self._ray_mesh_id_wp,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(self.cfg.max_distance),
                int(return_normal),
                int(return_face_id),
                int(write_mesh_ids),
            ],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        # clear mesh views so they are re-created on the next initialization
        self._mesh_views = []
