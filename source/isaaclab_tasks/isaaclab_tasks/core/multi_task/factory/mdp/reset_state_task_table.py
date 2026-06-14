# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the Factory reset-state task table (the analog of the locomotion
``RelativeStateTaskTable``).

The offline Newton-IK pipeline fills candidate rows; each batch is written into
the live envs, settled for a few physics steps (drifting rows rejected -- the
simulation-validated acceptance label), and harvested back through
``get_reset_state`` so the stored rows use the exact env serialization. The
survivors are then paired spawn x target WITHIN each board configuration. The
result is a flat, index-based :class:`FactoryResetStateTaskTable` the command
term consumes; the command itself only owns per-env lifecycle tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.curriculum import get_reset_state
from isaaclab_tasks.core.multi_task.utils.grid_downsample import extract_features, grid_bucket_downsample

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg


@dataclass
class FactoryResetStateTaskTable:
    """Flat, index-based reset-state table for the factory :class:`~...mdp.commands.StateCommand`.

    Attributes:
        state_data: Settled reset-state rows [num_states, row_dim] in env-local
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
        built_size: Rows produced before the settle gate (survival denominator).
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
    """Fill the table from the offline Newton-IK pipeline + a settle gate, then pair.

    Args:
        cfg: The command cfg (``task_table`` carries the pipeline + density knobs;
            ``payload.reset_assets`` defines the row layout).
        env: The live env (assets resolve from its scene; the settle gate steps it).

    Returns:
        The finalized :class:`FactoryResetStateTaskTable`.
    """
    table_cfg = cfg.task_table
    # the reset-asset set is derivable from the cfg, so the table is built before
    # the payload (one construction order shared with the locomotion command)
    reset_assets = sorted(
        (set(env.scene._articulations) | set(env.scene._rigid_objects)) & set(cfg.payload.reset_assets)
    )
    state_data, state_tag_indices, state_board_indices, state_tag_names, built, target = _precollect_from_pipeline(
        env, table_cfg, reset_assets
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
    )


def _precollect_from_pipeline(env, table_cfg, reset_assets):
    """Run the offline pipeline, settle each batch, harvest the survivors."""
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

    # pipeline poses are in the robot base frame; the env writes are env-local,
    # so the robot base must sit at its env origin with identity rotation
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

    assets = [
        (env.scene["held_asset"], result.nut_pose),
        (env.scene["nistboard"], result.board_pose),
        (env.scene["fixed_asset"], result.bolt_pose),
    ]
    total = result.joint_q.shape[0]
    rows, row_tags, row_boards, survived = [], [], [], 0
    for start in range(0, total, env.num_envs):
        stop = min(start + env.num_envs, total)
        b = slice(start, stop)
        ids = torch.arange(stop - start, device=env.device)
        jp = wp.to_torch(robot.data.default_joint_pos)[ids].clone()
        jq = result.joint_q[b]
        jp[:, [i for _, i in arm_pairs]] = jq[:, [c for c, _ in arm_pairs]]
        jp[:, [i for _, i in finger_pairs]] = jq[:, [c for c, _ in finger_pairs]] + squeeze[b].unsqueeze(-1)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        robot.set_joint_position_target(jp, env_ids=ids)
        zeros6 = torch.zeros(ids.numel(), 6, device=env.device)
        for asset, pose in assets:
            root = torch.cat([pose[b, :3] + env.scene.env_origins[ids], pose[b, 3:7]], dim=-1)
            asset.write_root_pose_to_sim(root, env_ids=ids)
            asset.write_root_com_velocity_to_sim(zeros6, env_ids=ids)
        valid = torch.ones(ids.numel(), dtype=torch.bool, device=env.device)
        if table_cfg.settle_steps > 0:
            env.scene.write_data_to_sim()
            for _ in range(table_cfg.settle_steps):
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            held = env.scene["held_asset"]
            drift = (wp.to_torch(held.data.root_pos_w)[ids] - env.scene.env_origins[ids] - result.nut_pose[b, :3]).norm(
                dim=-1
            )
            valid = drift < table_cfg.settle_max_drift
        states = get_reset_state(env, ids, reset_assets, is_relative=True)
        rows.append(states[valid])
        row_tags.append(result.tag[b][valid])
        row_boards.append(result.board_index[b][valid])
        survived += int(valid.sum())
    state_data = torch.cat(rows).contiguous()
    state_tag_indices = torch.cat(row_tags).contiguous()
    state_board_indices = torch.cat(row_boards).contiguous()
    per_tag = {
        name: int((state_tag_indices == t).sum()) for t, name in enumerate(tag_names) if bool((result.tag == t).any())
    }
    print(f"[reset_state] pipeline table: {total} rows -> {survived} survived the settle gate {per_tag}")
    # ``built`` is the PRE-settle row count (the survival denominator); the stored
    # ``state_data`` is the survivors
    return state_data, state_tag_indices, state_board_indices, tag_names, total, target_size


def _pair_within_boards(table_cfg, state_data, state_tag_indices, state_board_indices):
    """Pair spawn x target WITHIN each board configuration (per-cell pairing)."""
    coords = extract_features(state_data, table_cfg.state_table_fps_features).contiguous()
    num_states = int(state_data.shape[0])
    state_ids = torch.arange(num_states, device=state_data.device, dtype=torch.long)
    # a goal solved against a different board pose would point at the wrong bolt.
    # Goals are a spatially-spread SUBSET of the board's own rows
    # (targets_per_board <= rows_per_board by contract); the settle gate can leave
    # a board with fewer rows, in which case all of them serve as goals.
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
