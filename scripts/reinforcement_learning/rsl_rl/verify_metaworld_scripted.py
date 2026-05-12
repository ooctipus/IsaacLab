# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a Meta-World *scripted* expert policy in our IsaacLab port and
report per-task success rate.

Approach:
* Reset the IsaacLab env (Sawyer + asset).
* Each step, read the true scene state (TCP, object positions, joint
  state) and assemble a Meta-World-format observation buffer.
* Feed that to the corresponding ``Sawyer<Task>V3Policy`` from MW's
  ``metaworld.policies`` module — these are deterministic scripted
  controllers that achieve ~100 % success in MuJoCo.
* Apply the resulting 4-d action through our env's standard step.
* Count an episode as a success if the task's success criterion is ever
  satisfied during the episode.

If our IsaacLab port reproduces MW's scene + physics + action interface
faithfully, the same scripted policy should also succeed in IsaacLab.
This is *much* faster than RL training: 500 steps × 1 env ≈ 5 s, no
seed variance, and immediately exposes port bugs (geometry, action
mapping, asset orientation) as 0 % success.
"""
from __future__ import annotations

import argparse
import sys
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help="Gym task id, e.g. Isaac-Metaworld-Drawer-Open-Sawyer-Real-v0")
parser.add_argument("--mw_env", required=True, help="MW env class import path, e.g. metaworld.envs.sawyer_drawer_open_v3:SawyerDrawerOpenEnvV3")
parser.add_argument("--mw_policy", required=True, help="MW policy class import path, e.g. metaworld.policies.sawyer_drawer_open_v3_policy:SawyerDrawerOpenV3Policy")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episode_steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + remaining
launcher = AppLauncher(args)
sim_app = launcher.app

# --- Imports that need the simulator to be running --------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

# Add MW to import path so we can load its expert policies (they're pure
# numpy and don't require the MW venv to be active).
import sys as _sys
_sys.path.insert(0, "/home/zhengyuz/Projects/Metaworld")


def _import_class(spec: str):
    """Resolve ``module.path:ClassName`` → class object."""
    mod, _, cls = spec.partition(":")
    if not cls:
        mod, _, cls = spec.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


# ---------------------------------------------------------------------------
# Per-task observation builders
# ---------------------------------------------------------------------------
#
# Each task's MW policy parses obs[:7] (hand_pos, gripper, obj1_pos) and may
# also read obj1_quat, obj2_pos, obj2_quat, goal. We synthesize the obs from
# our IsaacLab scene state. Padding with zeros is fine — MW policies don't
# read obs[7:] beyond the goal (which lives at obs[36:39] in the 39-d MW obs).


def _build_obs_drawer_open(env, task_state) -> np.ndarray:
    """obs[:3] = TCP, obs[3] = gripper (unused by policy), obs[4:7] = handle pos.
    No goal is read by the drawer-open policy; pad to 39 with zeros."""
    obs = np.zeros(39, dtype=np.float64)
    obs[:3] = task_state["tcp_e"]
    obs[3] = task_state["gripper_open"]
    obs[4:7] = task_state["handle_e"]
    return obs


def _build_obs_push(env, task_state) -> np.ndarray:
    """MW push obs: [tcp(3), gripper(1), puck(3), unused(...), goal(3)]."""
    obs = np.zeros(39, dtype=np.float64)
    obs[:3] = task_state["tcp_e"]
    obs[3] = task_state["gripper_open"]
    obs[4:7] = task_state["cube_e"]
    obs[-3:] = task_state["goal_e"]
    return obs


_BUILDERS = {
    "Isaac-Metaworld-Drawer-Open-Sawyer-Real-v0": _build_obs_drawer_open,
    "Isaac-Metaworld-Drawer-Close-Sawyer-Real-v0": _build_obs_drawer_open,
    "Isaac-Metaworld-Push-Sawyer-v0": _build_obs_push,
    "Isaac-Metaworld-Reach-Sawyer-v0": _build_obs_push,  # same parser shape
    "Isaac-Metaworld-Pick-Place-Sawyer-v0": _build_obs_push,
}


# ---------------------------------------------------------------------------
# Per-task success indicators (on env-local positions)
# ---------------------------------------------------------------------------


def _success_drawer_open(task_state, target_e) -> bool:
    """MW drawer-open: success when handle is within 3 cm of target."""
    handle = task_state["handle_e"]
    return float(np.linalg.norm(handle - target_e)) <= 0.03


def _success_drawer_close(task_state, target_e) -> bool:
    """MW drawer-close: success when handle is within 5.5 cm of target (closed)."""
    handle = task_state["handle_e"]
    return float(np.linalg.norm(handle - target_e)) <= 0.055


def _success_reach(task_state, target_e) -> bool:
    """MW reach: TCP within 5 cm of target."""
    return float(np.linalg.norm(task_state["tcp_e"] - target_e)) <= 0.05


def _success_push(task_state, target_e) -> bool:
    """MW push: cube within 5 cm of target."""
    return float(np.linalg.norm(task_state["cube_e"] - target_e)) <= 0.05


def _success_pick_place(task_state, target_e) -> bool:
    """MW pick-place: cube within 7 cm of target (lift)."""
    return float(np.linalg.norm(task_state["cube_e"] - target_e)) <= 0.07


_SUCCESS_FNS = {
    "Isaac-Metaworld-Drawer-Open-Sawyer-Real-v0": _success_drawer_open,
    "Isaac-Metaworld-Drawer-Close-Sawyer-Real-v0": _success_drawer_close,
    "Isaac-Metaworld-Reach-Sawyer-v0": _success_reach,
    "Isaac-Metaworld-Push-Sawyer-v0": _success_push,
    "Isaac-Metaworld-Pick-Place-Sawyer-v0": _success_pick_place,
}


# ---------------------------------------------------------------------------
# Read scene state from our IsaacLab env
# ---------------------------------------------------------------------------


def _read_state(env) -> dict[str, np.ndarray]:
    """Read scene state and return env-local positions.

    Returns a dict with shape ``(num_envs, 3)`` arrays for each spatial
    quantity. Positions are in the env-local frame (world − env_origin).
    """
    inner = env.unwrapped
    origins = inner.scene.env_origins.detach().cpu().numpy()  # (N, 3)
    tcp_ft = inner.scene["tcp_frame"]
    pad_w = tcp_ft.data.target_pos_w.torch.detach().cpu().numpy()  # (N, 2, 3)
    tcp_w = 0.5 * (pad_w[:, 0] + pad_w[:, 1])
    pad_gap = np.linalg.norm(pad_w[:, 0] - pad_w[:, 1], axis=-1)  # (N,)
    state: dict[str, np.ndarray] = {
        "tcp_e": tcp_w - origins,
        "gripper_open": np.clip(pad_gap / 0.1, 0.0, 1.0),
        "origins": origins,
    }
    # Optional: cabinet handle (drawer tasks).
    if "handle_frame" in inner.scene.keys():
        handle_ft = inner.scene["handle_frame"]
        handle_w = handle_ft.data.target_pos_w.torch.detach().cpu().numpy()[:, 0]
        state["handle_e"] = handle_w - origins
    # Optional: cube (push/reach/pick-place).
    if "cube" in inner.scene.keys():
        cube = inner.scene["cube"]
        cube_w = cube.data.root_pos_w.torch.detach().cpu().numpy()
        state["cube_e"] = cube_w - origins
    # Optional: goal command (paired-command tasks).
    try:
        cmd = inner.command_manager.get_command("ee_pose").detach().cpu().numpy()  # (N, 3)
        state["goal_e"] = cmd
    except Exception:
        pass
    return state


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    if args.task not in _BUILDERS:
        raise SystemExit(f"No obs builder registered for task '{args.task}'. Add one to _BUILDERS.")

    builder = _BUILDERS[args.task]
    success_fn = _SUCCESS_FNS[args.task]

    env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg)

    PolicyCls = _import_class(args.mw_policy)
    policy = PolicyCls()

    # Read goal from the env's command (env-local frame).
    inner = env.unwrapped
    cmd_term = inner.command_manager.get_term("ee_pose")

    obs, _ = env.reset()
    n = args.num_envs
    ever_success = np.zeros(n, dtype=bool)
    # cmd_term.command is a torch tensor (CommandTerm property).
    targets = cmd_term.command.detach().cpu().numpy()  # (N, 3) env-local goal

    for step in range(args.episode_steps):
        state = _read_state(env)
        # Vector dispatch: build (N, 4) action by calling the policy per env.
        actions = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            per_env_state = {k: v[i] for k, v in state.items() if k != "origins"}
            obs_i = builder(env, per_env_state)
            actions[i] = policy.get_action(obs_i)
        # Clip to the [-1, 1] action range MW expects.
        actions = np.clip(actions, -1.0, 1.0)
        act_tensor = torch.as_tensor(actions, device=inner.device)
        obs, _, _, _, _ = env.step(act_tensor)

        # Debug print every 100 steps for env 0.
        if step % 100 == 0:
            s0 = {k: v[0] for k, v in state.items() if k != "origins"}
            tcp = s0["tcp_e"]
            obj = s0.get("handle_e", s0.get("cube_e", np.zeros(3)))
            obj_label = "handle" if "handle_e" in s0 else "cube"
            print(
                f"step {step}: tcp={tcp.round(3).tolist()}, "
                f"{obj_label}={obj.round(3).tolist()}, "
                f"goal={targets[0].round(3).tolist()}, "
                f"|{obj_label}-goal|={float(np.linalg.norm(obj - targets[0])):.3f}, "
                f"action[env0]={actions[0].round(2).tolist()}"
            )

        # Check success per env.
        state = _read_state(env)
        for i in range(n):
            per_env_state = {k: v[i] for k, v in state.items() if k != "origins"}
            if success_fn(per_env_state, targets[i]):
                ever_success[i] = True

    rate = ever_success.mean()
    print(f"\n=== {args.task} ===")
    print(f"Scripted-expert success: {ever_success.sum()}/{n} = {rate*100:.1f}%")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
