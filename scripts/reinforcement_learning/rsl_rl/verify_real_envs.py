# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end asset verification of Sawyer-Real envs.

For each ``Isaac-Metaworld-<Task>-Sawyer-Real-v0`` env, this script:

1. Boots the env (Sawyer + asset + tcp_frame + keypoint_frame + commands + rewards).
2. Reads the per-env handle marker world position at ``joint=closed``.
3. Sets the asset joint to its expected ``open`` value via direct write.
4. Reads the handle marker world position again — should match the goal.
5. Computes the success indicator (handle within threshold of goal).

Returns PASS if the marker positions before/after the joint write match
the env's command (init/goal) within the success threshold. This proves
the env's asset, scene, command, and reward wiring are all consistent.
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--joint", required=True, help="Joint name to write")
parser.add_argument("--joint_value", type=float, required=True, help="Joint value at task goal state")
parser.add_argument("--threshold", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = 4
env = gym.make(args.task, cfg=env_cfg)
inner = env.unwrapped

obs, _ = env.reset()

cab = inner.scene["cabinet"]
ft = inner.scene["keypoint_frame"]
origins = inner.scene.env_origins.detach().cpu().numpy()
cmd_term = inner.command_manager.get_term("ee_pose")

# Read goal command (env-local target).
goal_e = cmd_term.command.detach().cpu().numpy()  # (N, 3)
init_e = cmd_term.obj_init_pos_e.detach().cpu().numpy()  # (N, 3)

# Closed-state handle position (right after reset).
handle_w_closed = ft.data.target_pos_w.torch.detach().cpu().numpy()[:, 0]
handle_e_closed = handle_w_closed - origins
init_err = float(np.linalg.norm(handle_e_closed[0] - init_e[0]))

# Set joint to goal value.
joint_idx = cab.find_joints(args.joint)[0][0]
target = torch.full((4, 1), args.joint_value, device=inner.device)
cab.write_joint_position_to_sim(target, joint_ids=[joint_idx])
for _ in range(8):
    env.step(torch.zeros((4, 4), device=inner.device))

handle_w_open = ft.data.target_pos_w.torch.detach().cpu().numpy()[:, 0]
handle_e_open = handle_w_open - origins
goal_err = float(np.linalg.norm(handle_e_open[0] - goal_e[0]))

print(f"\n=== {args.task} ===")
print(f"  joint: {args.joint}={args.joint_value:+.4f}")
print(f"  init handle:     {handle_e_closed[0].round(3).tolist()}")
print(f"  init command:    {init_e[0].round(3).tolist()}      err = {init_err*1000:.1f} mm  {'PASS' if init_err < args.threshold else 'FAIL'}")
print(f"  goal handle:     {handle_e_open[0].round(3).tolist()}")
print(f"  goal command:    {goal_e[0].round(3).tolist()}      err = {goal_err*1000:.1f} mm  {'PASS' if goal_err < args.threshold else 'FAIL'}")

env.close()
sim_app.close()
