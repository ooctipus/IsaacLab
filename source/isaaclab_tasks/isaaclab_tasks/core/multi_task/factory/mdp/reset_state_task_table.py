# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the Factory reset-state task table (the analog of the locomotion
``RelativeStateTaskTable``).

The offline Newton-IK pipeline fills candidate rows, then rows are serialized
directly into the same layout as ``get_reset_state``. No live-env batching or
sim stepping is involved. Survivors are paired spawn x target WITHIN each board
configuration. The result is a flat, index-based
:class:`FactoryResetStateTaskTable` the command term consumes; the command
itself only owns per-env lifecycle tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.utils.grid_downsample import extract_features, grid_bucket_downsample

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg


@dataclass
class FactoryResetStateTaskTable:
    """Flat, index-based reset-state table for the factory :class:`~...mdp.commands.StateCommand`.

    Attributes:
        state_data: Reset-state rows [num_states, row_dim] in env-local
            serialization (the exact :func:`get_reset_state` layout).
        state_tag_indices: Placement-strategy tag per row [num_states].
        state_board_indices: Board-configuration index per row [num_states].
        state_tag_names: Human-readable tag names (index = tag id).
        state_coords: FPS features per row [num_states, feat] for sampler layout.
        spawn_index: Spawn row index per task slot [num_slots].
        target_index: Goal row index per task slot [num_slots], paired within a board.
        slot_indices: ``arange(num_slots)``.
        task_tag_indices: Spawn tag per slot [num_slots].
        num_states: Stored row count.
        built_size: Rows produced before the final geometric filters.
        target_size: Density target ``rows_per_board x num_boards``.
    """

    state_data: torch.Tensor
    state_tag_indices: torch.Tensor
    state_board_indices: torch.Tensor
    state_tag_names: list[str]
    state_coords: torch.Tensor
    spawn_index: torch.Tensor
    target_index: torch.Tensor
    slot_indices: torch.Tensor
    task_tag_indices: torch.Tensor
    num_states: int
    built_size: int
    target_size: int

    # optional success-grid geometry (numpy), stashed only when
    # ``FactoryResetStateTableCfg.stash_viz_geometry`` is set; see
    # :mod:`~..viz.geometry` and :mod:`~..viz.sampler_images`.
    viz_link_polys: object | None = None
    viz_nut_polys: object | None = None
    viz_board_polys: object | None = None
    viz_bolt_polys: object | None = None
    viz_cell_of_state: object | None = None
    viz_n_boards: int | None = None

    @property
    def num_tasks(self) -> int:
        """Number of task slots (spawn x target pairs)."""
        return int(self.spawn_index.shape[0])

    def gather(self, task_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(spawn_states, target_states)`` reset-state rows for the slots.

        Args:
            task_rows: Task-slot indices ``[n]``.

        Returns:
            Spawn and target reset-state rows, each ``[n, row_dim]``.
        """
        spawn_states = self.state_data[self.spawn_index[task_rows]]
        target_states = self.state_data[self.target_index[task_rows]]
        return spawn_states, target_states


def build_factory_reset_state_task_table(cfg: StateCommandCfg, env: ManagerBasedRLEnv) -> FactoryResetStateTaskTable:
    """Fill the table from the offline Newton-IK pipeline, then pair.

    Args:
        cfg: The command cfg (``task_table`` carries the pipeline + density knobs;
            ``payload.reset_assets`` defines the row layout).
        env: The live env whose scene resolves asset identity and row layout.

    Returns:
        The finalized :class:`FactoryResetStateTaskTable`.
    """
    table_cfg = cfg.task_table
    # the reset-asset set is derivable from the cfg, so the table is built before
    # the payload (one construction order shared with the locomotion command)
    reset_assets = sorted(
        (set(env.scene._articulations) | set(env.scene._rigid_objects)) & set(cfg.payload.reset_assets)
    )
    state_data, state_tag_indices, state_board_indices, state_tag_names, built, target, viz_geom = (
        _precollect_from_pipeline(env, table_cfg, reset_assets)
    )
    coords, spawn_index, target_index, slot_indices, task_tag_indices = _pair_within_boards(
        table_cfg, state_data, state_tag_indices, state_board_indices
    )
    return FactoryResetStateTaskTable(
        state_data=state_data,
        state_tag_indices=state_tag_indices,
        state_board_indices=state_board_indices,
        state_tag_names=state_tag_names,
        state_coords=coords,
        spawn_index=spawn_index,
        target_index=target_index,
        slot_indices=slot_indices,
        task_tag_indices=task_tag_indices,
        num_states=int(state_data.shape[0]),
        built_size=built,
        target_size=target,
        **(viz_geom or {}),
    )


def _precollect_from_pipeline(env, table_cfg, reset_assets):
    """Run the offline pipeline and serialize geometric survivor rows."""
    from ..retarget import FactoryIKPipeline

    pcfg = table_cfg.pipeline_cfg
    robot = env.scene[pcfg.robot.asset_cfg.name]
    # the pipeline holds no asset paths or robot identity of its own: assets
    # resolve from THIS env's scene cfg, the robot USD and stance from the live
    # articulation (the terrain ``kin.usd_path`` patching pattern)
    pcfg.device = str(env.device)
    pcfg.scene = env.cfg.scene
    if getattr(robot.cfg.spawn, "usd_path", None):
        pcfg.robot.usd_path = robot.cfg.spawn.usd_path
    default_q = wp.to_torch(robot.data.default_joint_pos)[0]
    pcfg.robot.default_joint_q = {name: float(q) for name, q in zip(robot.joint_names, default_q)}
    pipeline = FactoryIKPipeline(pcfg)
    # the table size is DERIVED: rows-per-configuration density x library size
    target_size = int(table_cfg.rows_per_board) * int(pcfg.board.num_boards)
    result = pipeline.build_balanced_table(target_size)
    tag_names = list(result.tag_names)
    m = pipeline.model

    # pipeline poses are in the robot base frame; stored rows are env-local, so
    # the robot base must sit at its env origin with identity rotation.
    root_rel = wp.to_torch(robot.data.root_pos_w)[0] - env.scene.env_origins[0]
    if float(root_rel.norm()) > 1e-3:
        raise RuntimeError(f"robot base is {root_rel.tolist()} m off its env origin; pipeline frames assume 0")

    # newton coord -> robot joint index, mapped by joint name (mimic'd follower
    # fingers may be absent from the articulation -- the present ones suffice)
    names = list(robot.joint_names)
    arm_pairs = [(c, names.index(n)) for c, n in zip(m.arm_coords, m.arm_joint_names)]
    finger_pairs = [(c, names.index(n)) for c, n in zip(m.finger_coords, m.finger_joint_names) if n in names]
    if not finger_pairs:
        raise RuntimeError(f"none of the finger joints {m.finger_joint_names} exist on the robot: {names}")

    n_base = len(pipeline.placement_sampler.tag_names)
    grasped = result.tag < n_base
    # closing direction is mode-dependent: pinch grasps squeeze by closing,
    # expansion grasps (inside a bore) by opening
    squeeze = torch.where(result.family % 2 == 1, table_cfg.finger_squeeze, -table_cfg.finger_squeeze)
    squeeze = torch.where(grasped, squeeze, torch.zeros_like(squeeze))

    total = result.joint_q.shape[0]
    valid = _nut_bounds_mask(table_cfg, result.nut_pose[:, :3])
    state_data = _serialize_pipeline_rows(
        env,
        table_cfg,
        reset_assets,
        robot,
        result,
        arm_pairs,
        finger_pairs,
        squeeze,
        valid,
    )
    state_tag_indices = result.tag[valid].contiguous()
    state_board_indices = result.board_index[valid].contiguous()
    survived = int(valid.sum())
    per_tag = _tag_counts(state_tag_indices, tag_names)
    print(f"[reset_state] geometric pipeline table: {total} rows -> {survived} stored {per_tag}")
    viz_geom = _build_viz_geometry(table_cfg, pipeline, result, valid)
    return state_data, state_tag_indices, state_board_indices, tag_names, total, target_size, viz_geom


def _nut_bounds_mask(table_cfg, nut_xyz: torch.Tensor) -> torch.Tensor:
    """Return the configured geometric survivor mask for held-asset root positions."""
    if table_cfg.nut_bounds is None:
        return torch.ones(nut_xyz.shape[0], dtype=torch.bool, device=nut_xyz.device)
    axes = ("x", "y", "z")
    lo = torch.tensor([table_cfg.nut_bounds.get(axis, (-1e9, 1e9))[0] for axis in axes], device=nut_xyz.device)
    hi = torch.tensor([table_cfg.nut_bounds.get(axis, (-1e9, 1e9))[1] for axis in axes], device=nut_xyz.device)
    return ((nut_xyz >= lo) & (nut_xyz <= hi)).all(dim=1)


def _serialize_pipeline_rows(env, table_cfg, reset_assets, robot, result, arm_pairs, finger_pairs, squeeze, valid):
    """Serialize geometric pipeline rows in the exact ``get_reset_state(..., relative=True)`` layout."""
    pcfg = table_cfg.pipeline_cfg
    reset_asset_set = set(reset_assets)
    pose_by_asset = {
        pcfg.placement.held_asset_cfg.name: result.nut_pose[valid],
        pcfg.board.board_asset_cfg.name: result.board_pose[valid],
        pcfg.board.fixed_asset_cfg.name: result.bolt_pose[valid],
    }
    row_count = int(valid.sum())
    states: list[torch.Tensor] = []
    for name, articulation in env.scene._articulations.items():
        if name not in reset_asset_set:
            continue
        if name == pcfg.robot.asset_cfg.name:
            root_state = _repeat_root_state_relative(env, articulation, row_count)
            joint_pos = _robot_joint_pos_from_pipeline(
                robot, result.joint_q[valid], arm_pairs, finger_pairs, squeeze[valid]
            )
            joint_vel = torch.zeros_like(joint_pos)
            states.extend((root_state, joint_pos, joint_vel))
        else:
            root_state = _repeat_root_state_relative(env, articulation, row_count)
            joint_pos = wp.to_torch(articulation.data.default_joint_pos)[0].expand(row_count, -1).clone()
            joint_vel = torch.zeros_like(joint_pos)
            states.extend((root_state, joint_pos, joint_vel))

    for name, rigid_object in env.scene._rigid_objects.items():
        if name not in reset_asset_set:
            continue
        if name in pose_by_asset:
            states.append(_root_state_from_pose(pose_by_asset[name]))
        else:
            states.append(_repeat_root_state_relative(env, rigid_object, row_count))
    if not states:
        raise RuntimeError(f"no reset assets from {reset_assets} exist in the factory scene")
    return torch.cat(states, dim=-1).contiguous()


def _robot_joint_pos_from_pipeline(robot, joint_q, arm_pairs, finger_pairs, squeeze):
    """Map Newton joint coordinates into the live robot articulation joint order."""
    joint_pos = wp.to_torch(robot.data.default_joint_pos)[0].expand(joint_q.shape[0], -1).clone()
    joint_pos[:, [i for _, i in arm_pairs]] = joint_q[:, [c for c, _ in arm_pairs]]
    joint_pos[:, [i for _, i in finger_pairs]] = joint_q[:, [c for c, _ in finger_pairs]] + squeeze.unsqueeze(-1)
    return joint_pos


def _root_state_from_pose(pose: torch.Tensor) -> torch.Tensor:
    """Create root-state rows from env-local pose rows and zero root velocity."""
    root_state = torch.zeros(pose.shape[0], 13, device=pose.device, dtype=pose.dtype)
    root_state[:, :7] = pose
    return root_state


def _repeat_root_state_relative(env, asset, row_count: int) -> torch.Tensor:
    """Repeat the first env's current root state in env-local coordinates."""
    root_state = wp.to_torch(asset.data.root_state_w)[0].clone()
    root_state[:3] -= env.scene.env_origins[0]
    return root_state.unsqueeze(0).expand(row_count, -1).clone()


def _build_viz_geometry(table_cfg, pipeline, result, valid):
    """Build optional success-grid geometry for the kept geometric rows."""
    if not bool(getattr(table_cfg, "stash_viz_geometry", False)) or not bool(valid.any()):
        return None
    from ..viz.geometry import build_success_grid_geometry

    torch.cuda.empty_cache()  # the table build runs at peak sim memory
    return build_success_grid_geometry(
        pipeline.model,
        result.joint_q[valid],
        result.nut_pose[valid],
        result.bolt_pose[valid],
        result.board_pose[valid],
        result.board_index[valid],
    )


def _tag_counts(state_tag_indices: torch.Tensor, tag_names: list[str]) -> dict[str, int]:
    """Return non-empty per-tag counts for table-build logging."""
    return {name: int((state_tag_indices == tag).sum()) for tag, name in enumerate(tag_names)}


def _pair_within_boards(table_cfg, state_data, state_tag_indices, state_board_indices):
    """Pair spawn x target WITHIN each board configuration (per-cell pairing)."""
    coords = extract_features(state_data, table_cfg.state_table_fps_features).contiguous()
    num_states = int(state_data.shape[0])
    state_ids = torch.arange(num_states, device=state_data.device, dtype=torch.long)
    # a goal solved against a different board pose would point at the wrong bolt.
    # Goals are a spatially-spread SUBSET of the board's own rows
    # (targets_per_board <= rows_per_board by contract); geometric filters can
    # leave a board with fewer rows, in which case all of them serve as goals.
    targets_per_board = int(table_cfg.targets_per_board)
    if targets_per_board <= 0 or targets_per_board > int(table_cfg.rows_per_board):
        raise ValueError(
            f"targets_per_board={targets_per_board} must be in [1, rows_per_board="
            f"{int(table_cfg.rows_per_board)}]: targets are picked FROM each board's stored rows."
        )
    spawn_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for board in torch.unique(state_board_indices):
        board_ids = state_ids[state_board_indices == board]
        k = min(targets_per_board, int(board_ids.shape[0]))
        local_target = grid_bucket_downsample(coords[board_ids], k).sort().values
        board_targets = board_ids[local_target]
        spawn_chunks.append(board_ids.repeat_interleave(int(board_targets.shape[0])))
        target_chunks.append(board_targets.repeat(int(board_ids.shape[0])))
    spawn_index = torch.cat(spawn_chunks)
    target_index = torch.cat(target_chunks)
    slot_indices = torch.arange(spawn_index.shape[0], device=state_data.device, dtype=torch.long)
    task_tag_indices = state_tag_indices[spawn_index]
    return coords, spawn_index, target_index, slot_indices, task_tag_indices
