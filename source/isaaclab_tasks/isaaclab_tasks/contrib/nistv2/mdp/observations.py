# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slot-scalable observations for NIST board assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils import resolve_matching_prims_from_source

from .assembly_state import _assembly_state
from .reset import board_reset

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


@wp.kernel(enable_backward=False)
def _scene_point_cloud_in_root_frame(
    body_q: wp.array(dtype=wp.transformf),
    fixed_body_ids: wp.array2d(dtype=wp.int32),
    held_body_ids: wp.array2d(dtype=wp.int32),
    fixed_kind_by_slot: wp.array2d(dtype=wp.int32),
    fixture_index_by_variant: wp.array(dtype=wp.int32),
    variant_ids: wp.array2d(dtype=wp.uint8),
    fixed_points: wp.array2d(dtype=wp.vec3f),
    held_points: wp.array2d(dtype=wp.vec3f),
    robot_points: wp.array(dtype=wp.vec3f),
    robot_point_body_ids: wp.array(dtype=wp.int32),
    robot_poses: wp.array2d(dtype=wp.transformf),
    robot_root_poses: wp.array(dtype=wp.transformf),
    point_counts: wp.vec3i,
    num_slots: int,
    num_fixed_slots: int,
    output: wp.array2d(dtype=wp.vec3f),
):
    world, point = wp.tid()
    fixed_end = num_slots * point_counts[0]
    held_end = fixed_end + num_slots * point_counts[1]

    if point < fixed_end:
        slot = point // point_counts[0]
        fixed_point = point - slot * point_counts[0]
        variant = int(variant_ids[world, slot])
        fixture = fixture_index_by_variant[variant]
        fixed_slot = int(0)
        for candidate in range(num_fixed_slots):
            if fixed_kind_by_slot[world, candidate] == fixture:
                fixed_slot = candidate
        point_w = wp.transform_point(body_q[fixed_body_ids[world, fixed_slot]], fixed_points[variant, fixed_point])
    elif point < held_end:
        held_point = point - fixed_end
        slot = held_point // point_counts[1]
        held_local_point = held_point - slot * point_counts[1]
        variant = int(variant_ids[world, slot])
        point_w = wp.transform_point(body_q[held_body_ids[world, slot]], held_points[variant, held_local_point])
    else:
        robot_point = point - held_end
        robot_local_point = robot_points[robot_point]
        point_w = wp.transform_point(robot_poses[world, robot_point_body_ids[robot_point]], robot_local_point)

    root_pose = robot_root_poses[world]
    root_position = wp.transform_get_translation(root_pose)
    root_rotation = wp.transform_get_rotation(root_pose)
    output[world, point] = wp.quat_rotate_inv(root_rotation, point_w - root_position)


def _sample_source_points(
    env: ManagerBasedEnv,
    prim_path: str,
    source_paths: tuple[str, ...],
    num_points: int,
) -> torch.Tensor:
    from isaaclab_tasks.contrib.nist.utils import mesh_ops
    from isaaclab_tasks.contrib.nist.utils.rigid_object_hasher import RigidObjectHasher

    hasher = RigidObjectHasher(
        env.num_envs,
        prim_path,
        device=env.device,
        rigid_body_root=True,
        compact_sources=True,
        source_paths=source_paths,
    )
    if hasher.num_root != len(source_paths):
        raise ValueError(
            f"Expected {len(source_paths)} collision sources below {prim_path!r}, found {hasher.num_root}."
        )
    points = mesh_ops.sample_object_point_cloud(
        len(source_paths), num_points, prim_path, env.device, rigid_object_hasher=hasher
    )
    if points is None:
        raise ValueError(f"No collision geometry found below {prim_path!r}.")
    return points.contiguous()


def _sample_variant_points(
    env: ManagerBasedEnv, asset: RigidObject, num_variants: int, num_points: int
) -> torch.Tensor:
    source_paths = asset.cfg.spawn.spawn_paths
    if source_paths is None or len(source_paths) != num_variants or any(path is None for path in source_paths):
        raise ValueError(f"{asset.cfg.prim_path!r} must expose {num_variants} collision sources in catalog order.")
    return _sample_source_points(env, asset.cfg.prim_path, tuple(source_paths), num_points)


def _sample_fixed_points(
    env: ManagerBasedEnv, reset: board_reset, fixed_assets: tuple[RigidObject, ...], num_points: int
) -> torch.Tensor:
    layout = reset.layout
    if layout.fixed_assets_are_variant_banks:
        asset = fixed_assets[0]
        bank_sources = asset.cfg.spawn.spawn_paths
        if (
            bank_sources is None
            or len(bank_sources) != layout.num_variants
            or any(path is None for path in bank_sources)
        ):
            raise ValueError(f"{asset.cfg.prim_path!r} must expose {layout.num_variants} collision sources.")
        fixture_sources = tuple(bank_sources[index] for index in layout.fixture_variant_indices)
        source_paths = tuple(fixture_sources[index] for index in layout.fixture_index_by_variant)
        return _sample_source_points(env, asset.cfg.prim_path, source_paths, num_points)

    fixture_sources = []
    for asset in fixed_assets:
        source, _ = resolve_matching_prims_from_source(asset.cfg.prim_path, expected_num_matches=1)[0]
        fixture_sources.append(source.GetPath().pathString)
    source_paths = tuple(fixture_sources[index] for index in layout.fixture_index_by_variant)
    return _sample_source_points(env, fixed_assets[0].cfg.prim_path, source_paths, num_points)


def _sample_articulation_points(
    env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg, num_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from pxr import UsdPhysics

    from isaaclab_tasks.contrib.nist.utils import mesh_ops
    from isaaclab_tasks.contrib.nist.utils.rigid_object_hasher import RigidObjectHasher

    body_ids = list(range(asset.num_bodies)) if isinstance(asset_cfg.body_ids, slice) else asset_cfg.body_ids
    body_names = [asset.body_names[body_id] for body_id in body_ids]
    points_per_body, remainder = divmod(num_points, len(body_ids))
    body_counts = [points_per_body + (index < remainder) for index in range(len(body_ids))]

    point_clouds = []
    point_body_ids = []
    for body_id, body_name, body_count in zip(body_ids, body_names, body_counts, strict=True):
        if body_count == 0:
            continue
        _, prim_path = resolve_matching_prims_from_source(
            asset.cfg.prim_path,
            predicate=lambda candidate, name=body_name: (
                candidate.GetName() == name and candidate.HasAPI(UsdPhysics.RigidBodyAPI)
            ),
            expected_num_matches=1,
        )[0]
        hasher = RigidObjectHasher(env.num_envs, prim_path, device=env.device, compact_sources=True)
        if hasher.num_root != 1:
            raise ValueError(f"Robot body {prim_path!r} must have one clone source.")
        points = mesh_ops.sample_object_point_cloud(
            1, body_count, prim_path, env.device, rigid_object_hasher=hasher, seed=42 + body_id
        )
        if points is None:
            raise ValueError(f"No collision geometry found below {prim_path!r}.")
        point_clouds.append(points[0])
        point_body_ids.extend([body_id] * body_count)

    if not point_clouds:
        raise ValueError(f"No collision geometry found below {asset.cfg.prim_path!r}.")
    return (
        torch.cat(point_clouds).contiguous(),
        torch.tensor(point_body_ids, dtype=torch.int32, device=env.device),
    )


class scene_point_cloud_b(ManagerTermBase):
    """Slot-ordered fixed, held, and robot surface points in the robot root frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        reset = env.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("scene_point_cloud_b requires the resolved board reset term.")
        state = _assembly_state(env)
        layout = reset.layout
        robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        fixed_assets: tuple[RigidObject, ...] = tuple(env.scene[name] for name in layout.fixed_asset_names)
        held_asset: RigidObject = env.scene[layout.held_asset_names[0]]
        robot_asset: Articulation = env.scene[robot_asset_cfg.name]

        point_counts = (
            cfg.params.get("fixed_num_points", 256),
            cfg.params.get("held_num_points", 256),
            cfg.params.get("robot_num_points", 256),
        )
        if any(count <= 0 for count in point_counts):
            raise ValueError(f"Point counts must be positive, got {point_counts}.")

        fixed_points = _sample_fixed_points(env, reset, fixed_assets, point_counts[0])
        held_points = _sample_variant_points(env, held_asset, layout.num_variants, point_counts[1])
        robot_points, robot_body_ids = _sample_articulation_points(env, robot_asset, robot_asset_cfg, point_counts[2])
        self._kernel_inputs = (
            NewtonManager.get_state_0().body_q,
            state._fixed_body_ids,
            state._held_body_ids,
            wp.from_torch(reset.fixed_kind_by_slot, dtype=wp.int32),
            wp.array(layout.fixture_index_by_variant, dtype=wp.int32, device=env.device),
            state._variant_ids,
            wp.from_torch(fixed_points, dtype=wp.vec3f),
            wp.from_torch(held_points, dtype=wp.vec3f),
            wp.from_torch(robot_points, dtype=wp.vec3f),
            wp.from_torch(robot_body_ids, dtype=wp.int32),
            robot_asset.data.body_link_pose_w.warp,
            robot_asset.data.root_link_pose_w.warp,
            wp.vec3i(*point_counts),
            layout.num_slots,
            layout.num_fixed_slots,
        )

        num_points = layout.num_slots * (point_counts[0] + point_counts[1]) + point_counts[2]
        output = wp.empty((env.num_envs, num_points), dtype=wp.vec3f, device=env.device)
        self._kernel_outputs = (output,)
        self._num_points = num_points
        output_torch = wp.to_torch(output)
        self._output_torch = output_torch.flatten(1) if cfg.params.get("flatten", True) else output_torch

    def __call__(
        self,
        env: ManagerBasedEnv,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        fixed_num_points: int = 256,
        held_num_points: int = 256,
        robot_num_points: int = 256,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Return surface points [m], ordered by fixed slots, held slots, then robot."""
        wp.launch(
            _scene_point_cloud_in_root_frame,
            dim=(self.num_envs, self._num_points),
            inputs=self._kernel_inputs,
            outputs=self._kernel_outputs,
            device=self.device,
        )
        return self._output_torch
