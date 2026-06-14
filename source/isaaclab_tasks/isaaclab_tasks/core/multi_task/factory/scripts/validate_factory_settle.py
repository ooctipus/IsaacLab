# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-D validation: the pipeline-filled reset-state table + settle gate, live.

Boots the live Factory env with ``FactoryResetStateTableCfg.pipeline_cfg`` wired
(the offline Newton-IK fill replacing the sim-in-the-loop reset strategies). The
command-term init itself builds the table, settles every row for
``settle_steps`` physics steps, and rejects rows whose held asset drifts -- this
script then reports the per-tag survival, exercises table consumption by
resampling every env a few times, and verifies the held asset lands where the
stored rows say it should.

Run:
  SCRIPT=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/validate_factory_settle.py
  ./isaaclab.sh -p $SCRIPT --headless --num_envs 64 presets=franka,nut_thread_m16
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

parser = argparse.ArgumentParser(description="Validate the pipeline-filled factory reset-state table.")
parser.add_argument("--task", type=str, default="Isaac-Factory-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rows_per_board", type=int, default=2)
parser.add_argument("--settle_steps", type=int, default=24)
parser.add_argument("--resample_rounds", type=int, default=3)
add_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        # mutate the COMPOSED pipeline cfg (its preset fields are already resolved
        # by the env preset system); smaller build rounds because the pipeline
        # shares the device with the running sim
        st = env_cfg.commands.reset_state.task_table
        st.pipeline_cfg.placement.placements_per_board = 1
        st.rows_per_board = args_cli.rows_per_board
        st.targets_per_board = min(int(st.targets_per_board), args_cli.rows_per_board)
        st.settle_steps = args_cli.settle_steps
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped
        try:
            _validate(env)
        finally:
            env.close()


def _validate(env) -> None:
    import warp as wp

    cmd = env.command_manager.get_term("reset_state")
    table = cmd.table
    n_rows = int(table.state_data.shape[0])
    tag_counts = {
        name: int((table.state_tag_indices == t).sum())
        for t, name in enumerate(table.state_tag_names or [])
        if int((table.state_tag_indices == t).sum()) > 0
    }
    print(f"\n[settle] table: {n_rows} rows after the settle gate; per-tag {tag_counts}")
    built = int(table.built_size)
    survival = n_rows / built
    print(
        f"[settle] survival: {survival:.1%} of the {built} built rows"
        f" (density target {int(table.target_size)} = rows_per_board x num_boards)"
    )

    # exercise consumption: resample every env, step once, and check the held
    # asset sits where the stored row says (a deep consistency check of the
    # write -> settle -> harvest -> store -> resample loop).
    held = env.scene["held_asset"]
    offset = cmd.payload._held_asset_root_offset
    all_ids = torch.arange(env.num_envs, device=env.device)
    worst = 0.0
    for _ in range(args_cli.resample_rounds):
        cmd._resample_command(all_ids)
        rows = table.state_data[table.spawn_index[cmd.cmd_indices[all_ids]]]
        expect = rows[:, offset : offset + 3] + env.scene.env_origins
        got = wp.to_torch(held.data.root_pos_w)[all_ids]
        worst = max(worst, float((got - expect).norm(dim=-1).max()))
    print(f"[settle] resample consistency: worst held-asset placement error {worst * 1e3:.2f} mm")

    # board-configuration pairing: every (spawn, target) slot must share its board
    # configuration -- a goal solved against a different board points at the wrong
    # bolt. Checked over the whole slot table, not just the sampled slots.
    spawn_boards = table.state_board_indices[table.spawn_index]
    target_boards = table.state_board_indices[table.target_index]
    n_mismatched = int((spawn_boards != target_boards).sum())
    board_off = cmd.payload._root_state_offset("nistboard")
    spawn_board_pos = table.state_data[table.spawn_index, board_off : board_off + 3]
    target_board_pos = table.state_data[table.target_index, board_off : board_off + 3]
    worst_board = float((spawn_board_pos - target_board_pos).norm(dim=-1).max())
    print(
        f"[settle] board pairing: {n_mismatched} mismatched slots of {int(table.spawn_index.shape[0])};"
        f" worst spawn-target board position gap {worst_board * 1e3:.2f} mm"
    )

    # survival varies ~37-58% run-to-run with the random table contents and the
    # per-board tag mix (the physically-unsustainable tags are MEANT to be
    # culled; low densities weight fragile tags more); the hard checks are the
    # consistency + tag coverage, survival only guards against collapse
    ok = n_rows > 0 and len(tag_counts) >= 2 and survival > 0.3 and worst < 1e-3 and n_mismatched == 0
    print(
        "\n[settle] "
        + (
            "PASS: pipeline-filled table survives the settle gate and resamples consistently."
            if ok
            else "WARN: low survival, missing tags, or inconsistent resampling -- inspect above."
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
