# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic-action probe for the push / pick-place sim dynamics.

Even with reward parity confirmed (mean |Δr| = 0.0001 vs MW after the
dt-scaling fix), the trained MT3 policy hits 0 % success on push and
pick-place. The remaining suspects are PhysX cube-friction or the
gripper-pad contact behavior.

This script bypasses RL entirely: it scripts a 4-phase action sequence
known to push a cube to a goal in MW (open gripper → reach above cube →
descend → drive forward), runs it in the IsaacLab sim, and logs:

* Per-step env-local TCP and cube positions
* Per-step cube-to-target distance
* Per-step gripper opening
* Whether the push success criterion ever fires

Run::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/probe_push_dynamics.py

If the cube doesn't move at all under the scripted "drive forward"
phase, the gripper-cube contact is the culprit (PhysX cube ejecting
or sliding past the pads). If the cube moves but stops short of the
goal, friction tuning is the issue. If it lands within 5 cm, the
issue is back to PPO exploration / the policy not finding this
trajectory.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Metaworld-Push-Sawyer-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

launcher = AppLauncher(args)
sim_app = launcher.app

import importlib  # noqa: E402

import gymnasium  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402


def _to_torch(x):
    return getattr(x, "torch", x)


def _resolve_cfg(task_id: str):
    spec = gymnasium.spec(task_id)
    mod, _, cls = spec.kwargs["env_cfg_entry_point"].partition(":")
    return getattr(importlib.import_module(mod), cls)()


def _scripted_actions(cube_e: np.ndarray, goal_e: np.ndarray, n_steps: int) -> np.ndarray:
    """Closed-loop scripted policy ported from MW's
    ``sawyer_push_v3_policy.SawyerPushV3Policy.get_action``:

    * If TCP-z > cube-z + 0.05: descend (action z = -1, gripper open)
    * Elif cube-x distance to TCP > 0.04: align xy (action x,y toward cube)
    * Else: push toward goal (action xy toward goal, gripper closed)

    We don't have access to the per-step closed-loop state here (this is a
    static schedule), so approximate as a 3-phase sequence:

      0..30:  descend in z (-1) with gripper open (-1)
      30..50: align over cube — action towards cube xy delta
      50..n:  push along the cube→goal direction with gripper closed.
    """
    a = np.zeros((n_steps, 4), dtype=np.float32)

    # Phase 1: descend.
    a[0:30, 2] = -1.0
    a[0:30, 3] = -1.0  # MW: -1 = open

    # Phase 2: align xy over cube (one-shot vector).
    align_steps = slice(30, 50)
    align_xy = (cube_e[:2] - np.array([0.0, 0.6])) * 10.0  # crude proportional
    a[align_steps, 0] = float(np.clip(align_xy[0], -1.0, 1.0))
    a[align_steps, 1] = float(np.clip(align_xy[1], -1.0, 1.0))
    a[align_steps, 3] = -1.0

    # Phase 3: push toward goal.
    push_steps = slice(50, n_steps)
    push_dir = goal_e[:2] - cube_e[:2]
    push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
    a[push_steps, 0] = float(np.clip(push_dir[0], -1.0, 1.0))
    a[push_steps, 1] = float(np.clip(push_dir[1], -1.0, 1.0))
    a[push_steps, 3] = +1.0  # MW: +1 = close

    return a


def main() -> None:
    cfg = _resolve_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=cfg)
    inner = env.unwrapped

    env.reset()
    # Settle.
    zero = torch.zeros(inner.num_envs, 4, device=inner.device)
    for _ in range(3):
        env.step(zero)

    # Capture the canonical cube + goal for env 0.
    origins = inner.scene.env_origins
    cube = inner.scene["cube"]
    cube_e0 = (_to_torch(cube.data.root_pos_w)[0] - origins[0]).detach().cpu().numpy()
    cmd = inner.command_manager.get_term("ee_pose")
    goal_e0 = _to_torch(cmd.command)[0].detach().cpu().numpy()

    print(f"\n[probe] task={args.task}")
    print(f"  cube init env-local: {cube_e0.round(4)}")
    print(f"  goal      env-local: {goal_e0.round(4)}")
    print(f"  initial cube→goal:    {float(np.linalg.norm(cube_e0[:2] - goal_e0[:2])):.3f} m")

    actions = _scripted_actions(cube_e0, goal_e0, args.steps)
    actions_t = torch.from_numpy(actions).to(inner.device)

    print(
        f"\n{'step':>4s} {'tcp_x':>7s} {'tcp_y':>7s} {'tcp_z':>7s}   "
        f"{'cube_x':>7s} {'cube_y':>7s} {'cube_z':>7s}   "
        f"{'cube→goal':>10s} {'tcp→cube':>9s} {'grip':>5s}   {'reward':>8s}"
    )
    print("-" * 110)

    ft = inner.scene["tcp_frame"]
    success_step = -1
    for t in range(args.steps):
        a = actions_t[t : t + 1].expand(inner.num_envs, -1).contiguous()
        obs, rew, _, _, _ = env.step(a)
        cube_w = _to_torch(cube.data.root_pos_w)
        cube_e = (cube_w[0] - origins[0]).detach().cpu().numpy()
        pad_e = _to_torch(ft.data.target_pos_source)[0]
        tcp_e = (0.5 * (pad_e[0] + pad_e[1])).detach().cpu().numpy()
        pad_gap = float(torch.linalg.norm(pad_e[0] - pad_e[1]).item())
        gripper_open = float(np.clip(pad_gap / 0.1, 0.0, 1.0))
        cube_to_goal = float(np.linalg.norm(cube_e[:2] - goal_e0[:2]))
        tcp_to_cube = float(np.linalg.norm(cube_e - tcp_e))
        r = float(_to_torch(rew)[0].item())

        if cube_to_goal < 0.05 and success_step < 0:
            success_step = t

        if t % 10 == 0 or t == args.steps - 1:
            print(
                f"{t:>4d} {tcp_e[0]:>+7.3f} {tcp_e[1]:>+7.3f} {tcp_e[2]:>+7.3f}   "
                f"{cube_e[0]:>+7.3f} {cube_e[1]:>+7.3f} {cube_e[2]:>+7.3f}   "
                f"{cube_to_goal:>10.4f} {tcp_to_cube:>9.4f} {gripper_open:>5.2f}   {r:>8.3f}"
            )

    print("-" * 110)
    if success_step >= 0:
        print(f"[probe] PUSH SUCCESS reached at step {success_step}")
    else:
        cube_w_final = _to_torch(cube.data.root_pos_w)
        cube_e_final = (cube_w_final[0] - origins[0]).detach().cpu().numpy()
        cube_moved = float(np.linalg.norm(cube_e_final - cube_e0))
        print(f"[probe] no success — cube moved {cube_moved * 1000:.1f} mm total over {args.steps} steps")
        if cube_moved < 0.005:
            print("[probe] DIAGNOSIS — cube barely moved. Gripper-cube contact failure")
            print("                    (PhysX likely ejecting cube or pads slip past it).")
        elif cube_moved < 0.05:
            print("[probe] DIAGNOSIS — cube moved a bit but didn't reach goal. Friction or trajectory issue.")
        else:
            print("[probe] DIAGNOSIS — cube traveled far but missed goal. Aim issue, not contact.")

    env.close()


if __name__ == "__main__":
    main()
    sim_app.close()
