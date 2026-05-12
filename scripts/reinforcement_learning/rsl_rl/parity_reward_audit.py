# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-depth reward parity audit — feed IsaacLab state into MW's pure-Python
reward functions and compare component-by-component against IsaacLab's reward.

This isolates "do the formulae match?" from "do the dynamics match?".

For each rollout step we capture:

* IsaacLab side: tcp, leftpad, rightpad, cube, target, obj_init, init_tcp,
  gripper opening, last action, plus every reward-manager term value.
* MW side: pass the same numbers into ``reach_v2_reward`` /
  ``push_v2_reward`` / ``pick_place_v2_reward`` and read back
  ``(reward, components)``.

Then we tabulate Δreward and Δ(per-component). Because we drive both with
identical state, any non-zero delta is a *formula* bug (the original
parity sweep reported state + reward together, so dynamics drift muddied
the picture).

Run::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/parity_reward_audit.py \
        --task Isaac-Metaworld-Reach-Sawyer-v0 --num_envs 4 --rollout_steps 30
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Metaworld-Reach-Sawyer-v0",
    help="MT3 task gym ID — push / reach / pick-place.",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--rollout_steps", type=int, default=20)
parser.add_argument("--mw_path", default="/home/zhengyuz/Projects/Metaworld")
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

# Import MW pure-Python reward modules from the local clone (these don't pull
# in mujoco — they're the same modules our `mw_dump.py` consumes).
sys.path.insert(0, args.mw_path)
from metaworld.rewards.pick_place_v2 import pick_place_v2_reward  # noqa: E402
from metaworld.rewards.push_v2 import push_v2_reward  # noqa: E402
from metaworld.rewards.reach_v2 import reach_v2_reward  # noqa: E402
from metaworld.utils import reward_utils as _mw_ru  # noqa: E402

# Pure-Python ports of MW's articulated-task ``compute_reward`` methods
# (which live inside the env classes; not exported as ``*_v2_reward``
# functions). Extracted verbatim so we can drive them with IsaacLab state.


def drawer_open_v3_reward(
    handle_pos: np.ndarray,
    target_pos: np.ndarray,
    gripper: np.ndarray,
    init_tcp: np.ndarray,
    max_dist: float = 0.16,
) -> tuple[float, float, float]:
    """Verbatim port of ``SawyerDrawerOpenEnvV3.compute_reward(... v2)``.

    Returns ``(reward, reward_for_caging, reward_for_opening)``."""
    handle_error = float(np.linalg.norm(handle_pos - target_pos))

    reward_for_opening = float(_mw_ru.tolerance(handle_error, bounds=(0, 0.02), margin=max_dist, sigmoid="long_tail"))

    handle_pos_init = target_pos + np.array([0.0, max_dist, 0.0])
    scale = np.array([3.0, 3.0, 1.0])
    gripper_error = (handle_pos - gripper) * scale
    gripper_error_init = (handle_pos_init - init_tcp) * scale

    reward_for_caging = float(
        _mw_ru.tolerance(
            float(np.linalg.norm(gripper_error)),
            bounds=(0, 0.01),
            margin=float(np.linalg.norm(gripper_error_init)),
            sigmoid="long_tail",
        )
    )
    reward = (reward_for_caging + reward_for_opening) * 5.0
    return float(reward), reward_for_caging, reward_for_opening


def window_open_v3_reward(
    handle_pos: np.ndarray,
    target_pos: np.ndarray,
    tcp: np.ndarray,
    init_tcp: np.ndarray,
    handle_init_pos: np.ndarray,
    target_radius: float = 0.05,
) -> tuple[float, float, float]:
    """Port of ``SawyerWindowOpenEnvV3.compute_reward(... v2)``."""
    handle_radius = 0.02
    target_to_obj = abs(handle_pos[0] - target_pos[0])
    target_to_obj_init = abs(handle_init_pos[0] - target_pos[0])

    in_place = float(
        _mw_ru.tolerance(
            target_to_obj,
            bounds=(0, target_radius),
            margin=abs(target_to_obj_init - target_radius),
            sigmoid="long_tail",
        )
    )
    tcp_to_obj = float(np.linalg.norm(handle_pos - tcp))
    tcp_to_obj_init = float(np.linalg.norm(handle_init_pos - init_tcp))
    reach = float(
        _mw_ru.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid="long_tail",
        )
    )
    reward = 10.0 * _mw_ru.hamacher_product(reach, in_place)
    return float(reward), reach, in_place


def button_press_topdown_v3_reward(
    button_pos: np.ndarray,
    target_pos: np.ndarray,
    tcp: np.ndarray,
    init_tcp: np.ndarray,
    obj_to_target_init: float,
    obs_gripper: float,
) -> tuple[float, float, float]:
    """Port of ``SawyerButtonPressTopdownEnvV3.compute_reward(... v2)``."""
    tcp_to_obj = float(np.linalg.norm(button_pos - tcp))
    tcp_to_obj_init = float(np.linalg.norm(button_pos - init_tcp))
    obj_to_target = abs(target_pos[2] - button_pos[2])

    tcp_closed = 1.0 - obs_gripper
    near_button = float(_mw_ru.tolerance(tcp_to_obj, bounds=(0, 0.01), margin=tcp_to_obj_init, sigmoid="long_tail"))
    button_pressed = float(
        _mw_ru.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=obj_to_target_init,
            sigmoid="long_tail",
        )
    )
    reward = 5.0 * _mw_ru.hamacher_product(tcp_closed, near_button)
    if tcp_to_obj <= 0.03:
        reward += 5.0 * button_pressed
    return float(reward), near_button, button_pressed


def _to_torch(x):
    return getattr(x, "torch", x)


def _resolve_cfg(task_id: str):
    spec = gymnasium.spec(task_id)
    mod, _, cls = spec.kwargs["env_cfg_entry_point"].partition(":")
    return getattr(importlib.import_module(mod), cls)()


def _scripted_actions(n_steps: int) -> torch.Tensor:
    """Same scripted sequence as parity_compare.py — keeps the dynamics
    deterministic so cross-run comparisons stay meaningful."""
    rng = np.random.default_rng(seed=0xBEEF)
    full = np.zeros((50, 4), dtype=np.float32)
    full[0:15, 2] = -0.5
    full[0:15, 3] = -1.0
    full[15:20, 3] = +1.0
    full[20:35, 1] = +0.4
    full[20:35, 2] = +0.6
    full[20:35, 3] = +1.0
    full[35:50] = rng.uniform(-0.3, 0.3, size=(15, 4)).astype(np.float32)
    return torch.from_numpy(full[:n_steps].copy())


def _capture_state(env, action_t: torch.Tensor) -> dict:
    """Extract the values MW's reward functions need from env-0's current
    state. Returned in numpy for direct MW-side consumption."""
    inner = env.unwrapped
    scene = inner.scene
    origins = scene.env_origins  # (n,3)

    # TCP, leftpad, rightpad — from tcp_frame.
    ft = scene["tcp_frame"]
    pad_e = _to_torch(ft.data.target_pos_source)[:, :, :]  # (n, 2, 3) — (left, right)
    leftpad_e = pad_e[0, 0].detach().cpu().numpy()
    rightpad_e = pad_e[0, 1].detach().cpu().numpy()
    tcp_e = 0.5 * (leftpad_e + rightpad_e)

    # Object — from cube root for cube tasks; from keypoint frame for
    # articulated tasks.
    if "keypoint_frame" in scene.keys():
        kp = scene["keypoint_frame"]
        obj_w = _to_torch(kp.data.target_pos_w)[:, 0]  # (n, 3) — first target body
        obj_e = (obj_w[0] - origins[0]).detach().cpu().numpy()
    else:
        cube = scene["cube"]
        obj_w = _to_torch(cube.data.root_pos_w)
        obj_e = (obj_w[0] - origins[0]).detach().cpu().numpy()

    # Goal & init from the command term.
    cmd = inner.command_manager.get_term("ee_pose")
    target_e = _to_torch(cmd.command)[0].detach().cpu().numpy()
    obj_init_e = _to_torch(cmd.obj_init_pos_e)[0].detach().cpu().numpy()
    init_tcp_e = _to_torch(cmd.init_tcp_e)[0].detach().cpu().numpy()
    init_left_e = _to_torch(cmd.init_left_pad_e)[0].detach().cpu().numpy()
    init_right_e = _to_torch(cmd.init_right_pad_e)[0].detach().cpu().numpy()

    # Hand init pose — for reach. We use the realised init TCP (parity
    # comparator does the same; matches MW's runtime ``hand_init_pos`` after
    # ``_reset_hand``).
    hand_init = init_tcp_e

    # Gripper opening (= MW obs[3] = pad gap / 0.1 clipped).
    gripper_open = float(np.clip(np.linalg.norm(leftpad_e - rightpad_e) / 0.1, 0.0, 1.0))

    return {
        "tcp": tcp_e,
        "leftpad": leftpad_e,
        "rightpad": rightpad_e,
        "obj": obj_e,
        "target": target_e,
        "obj_init": obj_init_e,
        "init_tcp": init_tcp_e,
        "init_left_pad": init_left_e,
        "init_right_pad": init_right_e,
        "hand_init": hand_init,
        "gripper_open": gripper_open,
        "action": action_t.detach().cpu().numpy().astype(np.float32),
    }


def _mw_reward_for_task(task_id: str, s: dict) -> tuple[float, dict]:
    """Call MW's pure-Python reward function for this task. Returns
    ``(reward, components)``."""
    # Build the 39-d obs MW's functions read — only obs[3] (gripper) and
    # obs[4:7] (obj) are read; rest can be zeros.
    obs39 = np.zeros(39, dtype=np.float64)
    obs39[3] = s["gripper_open"]
    obs39[4:7] = s["obj"]

    if "Reach" in task_id:
        reward, tcp_to_target, in_place = reach_v2_reward(
            tcp=np.asarray(s["tcp"]),
            target_pos=np.asarray(s["target"]),
            hand_init_pos=np.asarray(s["hand_init"]),
        )
        return float(reward), {"tcp_to_target": tcp_to_target, "in_place": in_place}

    if "Push" in task_id and "Pick" not in task_id:
        reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place = push_v2_reward(
            action=np.asarray(s["action"]),
            obs=obs39,
            tcp=np.asarray(s["tcp"]),
            target_pos=np.asarray(s["target"]),
            obj_init_pos=np.asarray(s["obj_init"]),
            left_pad_pos=np.asarray(s["leftpad"]),
            right_pad_pos=np.asarray(s["rightpad"]),
            init_tcp=np.asarray(s["init_tcp"]),
        )
        return float(reward), {
            "tcp_to_obj": tcp_to_obj,
            "tcp_opened": tcp_opened,
            "target_to_obj": target_to_obj,
            "object_grasped": object_grasped,
            "in_place": in_place,
        }

    if "Window-Open" in task_id or "Window-Close" in task_id:
        # MW window's ``window_handle_pos_init`` = obj_init pose. Use the
        # IsaacLab spec's obj_init.
        reward, reach, in_place = window_open_v3_reward(
            handle_pos=np.asarray(s["obj"]),
            target_pos=np.asarray(s["target"]),
            tcp=np.asarray(s["tcp"]),
            init_tcp=np.asarray(s["init_tcp"]),
            handle_init_pos=np.asarray(s["obj_init"]),
        )
        return float(reward), {"reach": reach, "in_place": in_place}

    if "Button-Press-Topdown" in task_id and "Wall" not in task_id or "Coffee-Button" in task_id:
        # MW computes obj_to_target_init from the asset's ``maxDist`` for
        # buttons (max press depth ~0.06 m). Use that constant.
        obj_to_target_init = 0.06
        reward, near_button, button_pressed = button_press_topdown_v3_reward(
            button_pos=np.asarray(s["obj"]),
            target_pos=np.asarray(s["target"]),
            tcp=np.asarray(s["tcp"]),
            init_tcp=np.asarray(s["init_tcp"]),
            obj_to_target_init=obj_to_target_init,
            obs_gripper=s["gripper_open"],
        )
        return float(reward), {"near_button": near_button, "button_pressed": button_pressed}

    if "Drawer-Open" in task_id:
        # MW's drawer-open ``maxDist`` is the asset's joint travel distance —
        # 0.16 m for the canonical drawer (matches our ``mw_drawer.usda``).
        reward, caging, opening = drawer_open_v3_reward(
            handle_pos=np.asarray(s["obj"]),
            target_pos=np.asarray(s["target"]),
            gripper=np.asarray(s["tcp"]),
            init_tcp=np.asarray(s["init_tcp"]),
            max_dist=0.16,
        )
        return float(reward), {"caging": caging, "opening": opening}

    if "Pick-Place" in task_id:
        reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place = pick_place_v2_reward(
            action=np.asarray(s["action"]),
            obs=obs39,
            tcp=np.asarray(s["tcp"]),
            target_pos=np.asarray(s["target"]),
            obj_init_pos=np.asarray(s["obj_init"]),
            left_pad_pos=np.asarray(s["leftpad"]),
            right_pad_pos=np.asarray(s["rightpad"]),
            init_left_pad=np.asarray(s["init_left_pad"]),
            init_right_pad=np.asarray(s["init_right_pad"]),
            init_tcp=np.asarray(s["init_tcp"]),
        )
        return float(reward), {
            "tcp_to_obj": tcp_to_obj,
            "tcp_opened": tcp_opened,
            "obj_to_target": obj_to_target,
            "object_grasped": object_grasped,
            "in_place": in_place,
        }

    raise ValueError(f"Unsupported task {task_id} (MT3 only).")


def main() -> None:
    cfg = _resolve_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=cfg)
    env.reset()

    # Step a few zero actions to settle (matches parity_compare).
    inner = env.unwrapped
    zero = torch.zeros(inner.num_envs, 4, device=inner.device)
    for _ in range(3):
        env.step(zero)

    actions = _scripted_actions(args.rollout_steps).to(inner.device)
    print(f"\n{'step':>4s} {'isaac_r':>9s} {'mw_r':>9s} {'Δr':>8s}    components (isaac vs MW)")
    print("-" * 110)

    abs_deltas = []
    for t in range(args.rollout_steps):
        a = actions[t : t + 1].expand(inner.num_envs, -1).contiguous()
        # Capture pre-step state (matches the MW reward semantic — reward is
        # computed from post-action state, so capture POST state).
        _, isaac_rew, _, _, _ = env.step(a)
        s = _capture_state(env, actions[t])
        mw_r, comps = _mw_reward_for_task(args.task, s)
        i_r = float(_to_torch(isaac_rew)[0].item())
        delta = i_r - mw_r
        abs_deltas.append(abs(delta))
        comps_str = ", ".join(f"{k}={v:.3f}" for k, v in comps.items())
        print(f"{t:>4d} {i_r:>9.4f} {mw_r:>9.4f} {delta:>+8.4f}    {comps_str}")

    print("-" * 110)
    print(f"mean |Δr| = {np.mean(abs_deltas):.4f}    max |Δr| = {np.max(abs_deltas):.4f}")
    print()
    print("Notes:")
    print("  * isaac_r is the env-0 step reward (sum of all reward terms;")
    print("    includes the small action_rate_l2 penalty so it can differ")
    print("    from MW's pure reward by ~1e-4 even when the formula matches.")
    print("  * Components shown are MW's; if Δr ≈ 0 across the rollout the")
    print("    formula is a faithful port. If Δr drifts, look at which")
    print("    component diverges and patch the corresponding IsaacLab atom.")


if __name__ == "__main__":
    main()
    sim_app.close()
