# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a Meta-World heterogeneous multitask checkpoint.

Works for MT3 (single ``keypoint_frame`` on the cube) **and** MT5 /
MT10 / MT15 / MT25 (per-task ``*_keypoint`` frame transformers — drawer,
button, window, faucet, door, peg, etc.). The script auto-detects which
mode by looking at which scene entities are present.

Loads a trained policy and reports per-task success rate by running one
500-step episode (Meta-World's ``max_path_length``) and counting envs
whose success indicator fires on **any** step (MW's per-episode
success criterion).
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episode_length_s", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
args.headless = True
# Clear our consumed args from sys.argv before launching the simulator —
# AppLauncher / Hydra inspect sys.argv themselves and will error on the
# script-specific flags they don't recognise.
sys.argv = [sys.argv[0]] + remaining
launcher = AppLauncher(args)
sim_app = launcher.app

import importlib.metadata as md  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402

# Task-id → success-fn lookup. Tasks 3..9 use a generic
# obj-to-target distance threshold since the MT10-stub rewards are
# placeholders without proper articulated assets.
from isaaclab_contrib.tasks.manipulation.metaworld.mdp import (  # noqa: E402
    keypoint_at_target,
    obj_to_target_dist,
    reach_success,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def push_success(env, *, object_cfg, goal_command_name: str):
    """Cube within :data:`PUSH_TARGET_RADIUS` of goal — same definition MW uses."""
    return keypoint_at_target(env, keypoint_frame_cfg=object_cfg, goal_command_name=goal_command_name, threshold=0.05)


def pick_place_success(env, *, object_cfg, goal_command_name: str):
    """Cube within :data:`PICK_PLACE_SUCCESS_RADIUS` of goal."""
    return keypoint_at_target(env, keypoint_frame_cfg=object_cfg, goal_command_name=goal_command_name, threshold=0.07)


def _generic_obj_success(env, *, threshold: float, object_cfg, goal_command_name: str):
    import torch as _t

    return (
        obj_to_target_dist(env, keypoint_frame_cfg=object_cfg, goal_command_name=goal_command_name) <= threshold
    ).to(_t.float32)


# Per-task success-indicator table.  Two orderings are supported:
#
# * MT3 (and any cube-only multi-task) — task names {reach, push, pick_place}
#   at indices 0..2; bespoke per-task indicators.
# * MT5 / MT10 / MT15 (and the heterogeneous articulated env) — TASK_NAMES
#   from ``multi_task_env_cfg.py`` at indices 0..14; all use the keypoint
#   distance ≤ 0.05 m success rule.
#
# The script picks one of the two by detecting the task list at runtime
# from ``env.unwrapped.command_manager.get_term('ee_pose').cfg.tasks``.
_MT3_FNS = {
    0: ("reach", reach_success, {"frame_transformer_cfg": None, "goal_command_name": "ee_pose"}),
    1: ("push", push_success, {"object_cfg": None, "goal_command_name": "ee_pose"}),
    2: ("pick_place", pick_place_success, {"object_cfg": None, "goal_command_name": "ee_pose"}),
}

_MT15_FNS = {
    i: (n, _generic_obj_success, {"threshold": 0.05, "object_cfg": None, "goal_command_name": "ee_pose"})
    for i, n in enumerate(
        [
            "drawer_open",
            "drawer_close",
            "button_press_topdown",
            "coffee_button",
            "window_open",
            "window_close",
            "faucet_open",
            "faucet_close",
            "dial_turn",
            "lever_pull",
            "door_open",
            "door_close",
            "door_lock",
            "door_unlock",
            "peg_insert_side",
        ]
    )
}

env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device if args.device else "cuda:0"
env_cfg.episode_length_s = args.episode_length_s

env = gym.make(args.task, cfg=env_cfg)
wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(env.unwrapped.device))
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=str(env.unwrapped.device))

# Wire SceneEntityCfg references now that scene is resolved.
from isaaclab.managers import SceneEntityCfg  # noqa: E402

_TCP = SceneEntityCfg("tcp_frame")
_TCP.resolve(env.unwrapped.scene)

# Choose MT3 vs MT15 success table by inspecting the active task list.
_active_task_names = {t.name for t in env.unwrapped.command_manager.get_term("ee_pose").cfg.tasks}
_scene_keys = set(env.unwrapped.scene.keys())

if {"reach", "push", "pick_place"}.issubset(_active_task_names) and "keypoint_frame" in _scene_keys:
    # MT3 — single ``keypoint_frame`` on the cube.
    _SUCCESS_FNS = _MT3_FNS
    _OBJ = SceneEntityCfg("keypoint_frame")
    _OBJ.resolve(env.unwrapped.scene)
    _SUCCESS_FNS[0][2]["frame_transformer_cfg"] = _TCP
    for _k in range(1, len(_SUCCESS_FNS)):
        _SUCCESS_FNS[_k][2]["object_cfg"] = _OBJ
else:
    # MT5 / MT10 / MT15 / MT25 / MT50 — per-task keypoint frames. Map each
    # task's name to its keypoint-frame scene entity. MT50 cube-tail tasks
    # share ``cube_keypoint``.
    _ARTICULATED_KEYPOINT = {
        # MT15 articulated.
        "drawer_open": "drawer_keypoint",
        "drawer_close": "drawer_keypoint",
        "button_press_topdown": "button_keypoint",
        "coffee_button": "button_keypoint",
        "window_open": "window_keypoint",
        "window_close": "window_keypoint",
        "faucet_open": "faucet_keypoint",
        "faucet_close": "faucet_keypoint",
        "dial_turn": "faucet_keypoint",
        "lever_pull": "faucet_keypoint",
        "door_open": "door_keypoint",
        "door_close": "door_keypoint",
        "door_lock": "door_keypoint",
        "door_unlock": "door_keypoint",
        "peg_insert_side": "peg_keypoint",
        # MT25 additions.
        "handle_press": "handle_press_keypoint",
        "handle_pull": "handle_press_keypoint",
        "handle_press_side": "handle_press_side_keypoint",
        "handle_pull_side": "handle_press_side_keypoint",
        "peg_unplug_side": "peg_unplug_keypoint",
        "box_close": "box_close_keypoint",
        "plate_slide": "plate_keypoint",
        "plate_slide_back": "plate_keypoint",
        "button_press_topdown_wall": "button_keypoint",
        "hammer": "nail_keypoint",
        # MT50 cube-tail tasks — read ``cube_keypoint``.
        "reach": "cube_keypoint",
        "push": "cube_keypoint",
        "pick_place": "cube_keypoint",
        "push_back": "cube_keypoint",
        "push_wall": "cube_keypoint",
        "reach_wall": "cube_keypoint",
        "pick_place_wall": "cube_keypoint",
        "basketball": "cube_keypoint",
        "shelf_place": "cube_keypoint",
        "soccer": "cube_keypoint",
        "sweep": "cube_keypoint",
        "sweep_into": "cube_keypoint",
        "coffee_push": "cube_keypoint",
        "coffee_pull": "cube_keypoint",
        "stick_push": "cube_keypoint",
        "stick_pull": "cube_keypoint",
        "bin_picking": "cube_keypoint",
        "hand_insert": "cube_keypoint",
        "pick_out_of_hole": "cube_keypoint",
        "assembly": "cube_keypoint",
        "disassemble": "cube_keypoint",
        # MT50 articulated additions.
        "plate_slide_side": "plate_side_keypoint",
        "plate_slide_back_side": "plate_side_keypoint",
        "button_press": "button_front_keypoint",
        "button_press_wall": "button_front_keypoint",
    }
    _SUCCESS_FNS = {}
    for tidx, tname in enumerate([t.name for t in env.unwrapped.command_manager.get_term("ee_pose").cfg.tasks]):
        kp_name = _ARTICULATED_KEYPOINT.get(tname)
        if kp_name is None or kp_name not in _scene_keys:
            print(f"[eval] skipping task {tname!r} — no keypoint frame in scene")
            continue
        kp_cfg = SceneEntityCfg(kp_name)
        kp_cfg.resolve(env.unwrapped.scene)
        _SUCCESS_FNS[tidx] = (
            tname,
            _generic_obj_success,
            {"threshold": 0.05, "object_cfg": kp_cfg, "goal_command_name": "ee_pose"},
        )

cmd = env.unwrapped.command_manager.get_term("ee_pose")
task_id = cmd.task_id  # (num_envs,)
n_tasks = cmd.num_tasks
print(f"Evaluating {args.task}, {args.num_envs} envs, {n_tasks} tasks")
print(f"Task assignment count: {[(task_id == k).sum().item() for k in range(n_tasks)]}")

# Reset and evaluate one episode (resampling_time_range is 1e6 in the cfg
# so commands stay fixed within an episode).
obs_ret = wrapped.get_observations()
obs = obs_ret[0] if isinstance(obs_ret, tuple) else obs_ret
ep_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
print(f"Episode steps: {ep_steps}")

# Track per-env, per-task "achieved success at any step in episode".
ever_success = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.bool)

with torch.no_grad():
    for step in range(ep_steps):
        action = policy(obs)
        obs, _, _, _ = wrapped.step(action)
        # Compute per-task success indicator on each env. We compute all 3
        # and select by task_id.
        succ_per_env = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.float32)
        for tidx, (_, fn, kwargs) in _SUCCESS_FNS.items():
            if tidx >= n_tasks:
                continue
            succ = fn(env.unwrapped, **kwargs)  # (N,) float
            mask = task_id == tidx
            succ_per_env[mask] = succ[mask]
        ever_success |= succ_per_env > 0.5

print()
print("Per-task success rate (any step in episode):")
print(f"{'task':<14} {'success':>9} {'n_envs':>8}")
for tidx, (name, _, _) in _SUCCESS_FNS.items():
    if tidx >= n_tasks:
        continue
    mask = task_id == tidx
    n = int(mask.sum().item())
    if n == 0:
        continue
    sr = float(ever_success[mask].float().mean().item())
    print(f"{name:<14} {sr * 100:>8.2f}% {n:>8}")

env.close()
sim_app.close()
