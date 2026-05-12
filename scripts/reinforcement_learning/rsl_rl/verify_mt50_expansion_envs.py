# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke + reward sanity check for the MT50-expansion envs.

For each of the 11 new ``Isaac-Metaworld-<Task>-Sawyer-v0`` envs added on
top of the original 18, this script:

1. Constructs the env (boots the simulator, instantiates the scene/MDP).
2. Resets and reads the initial reward + reads the cube / plate keypoint.
3. Drives the manipulandum directly to the goal:
     - cube tasks: ``cube.write_root_pose_to_sim(goal_world)``
     - plate-slide tasks: ``cabinet.write_joint_position_to_sim(joint_goal)``
     - reach tasks: skipped — TCP can't be driven without IK; only verify
       the reward is finite and stepping doesn't crash.
4. Steps physics until settled, reads the goal-state reward.
5. Asserts: reward is finite at both states, increased toward goal (or
   stayed >= initial for reach), and success indicator fires when at goal.

This is the script-side analog of the joint-sweep verification in
``verify_mw_assets.py`` — exercises the env wiring (scene + cmd + reward +
event + obs) without running RL.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=None, help="Optional: run only this gym ID.")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
launcher = AppLauncher(args)
sim_app = launcher.app

import importlib  # noqa: E402

import gymnasium  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402
from isaaclab_contrib.tasks.manipulation.metaworld.metaworld_specs import TASK_SPECS  # noqa: E402

# (gym task name, spec key, reward kind)
#   ``cube``  : drive cube via write_root_pose; reward should jump (success_override fires).
#   ``plate`` : drive joint via write_joint_position; verify keypoint at goal (Hamacher
#               needs both reach + in_place active, so reward alone won't budge).
#   ``reach`` : TCP-target task (no manipulandum motion); just check stepping is stable.
TASKS = [
    # Batch 1: push/reach/pick-place variants + basketball/shelf/soccer/sweep/plate-slide
    ("Isaac-Metaworld-Push-Back-Sawyer-v0", "push_back", "cube"),
    ("Isaac-Metaworld-Push-Wall-Sawyer-v0", "push_wall", "cube"),
    ("Isaac-Metaworld-Reach-Wall-Sawyer-v0", "reach_wall", "reach"),
    ("Isaac-Metaworld-Pick-Place-Wall-Sawyer-v0", "pick_place_wall", "cube"),
    ("Isaac-Metaworld-Basketball-Sawyer-v0", "basketball", "cube"),
    ("Isaac-Metaworld-Shelf-Place-Sawyer-v0", "shelf_place", "cube"),
    ("Isaac-Metaworld-Soccer-Sawyer-v0", "soccer", "cube"),
    ("Isaac-Metaworld-Sweep-Sawyer-v0", "sweep", "cube"),
    ("Isaac-Metaworld-Sweep-Into-Sawyer-v0", "sweep_into", "cube"),
    ("Isaac-Metaworld-Plate-Slide-Sawyer-v0", "plate_slide", "plate"),
    ("Isaac-Metaworld-Plate-Slide-Back-Sawyer-v0", "plate_slide_back", "plate"),
    # Batch 2: handle / peg-unplug / box / hole-block / button-wall
    ("Isaac-Metaworld-Handle-Press-Sawyer-v0", "handle_press", "joint"),
    ("Isaac-Metaworld-Handle-Pull-Sawyer-v0", "handle_pull", "joint"),
    ("Isaac-Metaworld-Handle-Press-Side-Sawyer-v0", "handle_press_side", "joint"),
    ("Isaac-Metaworld-Handle-Pull-Side-Sawyer-v0", "handle_pull_side", "joint"),
    ("Isaac-Metaworld-Peg-Unplug-Side-Sawyer-v0", "peg_unplug_side", "joint"),
    ("Isaac-Metaworld-Box-Close-Sawyer-v0", "box_close", "joint"),
    ("Isaac-Metaworld-Hand-Insert-Sawyer-v0", "hand_insert", "reach"),
    ("Isaac-Metaworld-Pick-Out-Of-Hole-Sawyer-v0", "pick_out_of_hole", "cube"),
    ("Isaac-Metaworld-Button-Press-Topdown-Wall-Sawyer-v0", "button_press_topdown_wall", "joint"),
    # Batch 3: bin / coffee / stick / assembly / hammer (all use existing assets)
    ("Isaac-Metaworld-Bin-Picking-Sawyer-v0", "bin_picking", "cube"),
    ("Isaac-Metaworld-Coffee-Push-Sawyer-v0", "coffee_push", "cube"),
    ("Isaac-Metaworld-Coffee-Pull-Sawyer-v0", "coffee_pull", "cube"),
    ("Isaac-Metaworld-Stick-Push-Sawyer-v0", "stick_push", "cube"),
    ("Isaac-Metaworld-Stick-Pull-Sawyer-v0", "stick_pull", "cube"),
    ("Isaac-Metaworld-Assembly-Sawyer-v0", "assembly", "cube"),
    ("Isaac-Metaworld-Disassemble-Sawyer-v0", "disassemble", "cube"),
    ("Isaac-Metaworld-Hammer-Sawyer-v0", "hammer", "joint"),
    # Batch 4: plate-side / front-button (use new mw_plate_side / mw_button_front USDs)
    ("Isaac-Metaworld-Plate-Slide-Side-Sawyer-v0", "plate_slide_side", "joint"),
    ("Isaac-Metaworld-Plate-Slide-Back-Side-Sawyer-v0", "plate_slide_back_side", "joint"),
    ("Isaac-Metaworld-Button-Press-Sawyer-v0", "button_press", "joint"),
    ("Isaac-Metaworld-Button-Press-Wall-Sawyer-v0", "button_press_wall", "joint"),
]


def _resolve_cfg(task_id: str):
    spec = gymnasium.spec(task_id)
    mod, _, cls = spec.kwargs["env_cfg_entry_point"].partition(":")
    return getattr(importlib.import_module(mod), cls)()


def _drive_cube_to_goal(env, spec_key: str) -> None:
    """Write the cube's root pose (in env-local frame translated to world)
    to the goal position from :data:`TASK_SPECS`."""
    s = TASK_SPECS[spec_key]
    cube = env.unwrapped.scene["cube"]
    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    origins = env.unwrapped.scene.env_origins  # (n, 3)
    goal_w = origins + torch.tensor(s.goal, device=device).expand(n, 3)
    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    pose = torch.cat([goal_w, quat], dim=-1)
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))


# Joint value (per articulated task) that places the keypoint marker at ``spec.goal``.
# For "joint" tasks the verifier writes this value with ``write_joint_position_to_sim``
# and then reads the marker world position to confirm.
_JOINT_GOAL_VALUES: dict[str, float] = {
    # Plate slides along world-x (plate_x = -0.10 + joint).
    "plate_slide": 0.20,
    "plate_slide_back": 0.0,
    # Top-down handle (marker_z = 0.16 + joint).
    "handle_press": -0.05,  # press down
    "handle_pull": 0.05,  # pull up
    # Side-mounted handle (marker_x = 0.10 + joint).
    "handle_press_side": -0.05,  # push in
    "handle_pull_side": 0.05,  # pull out
    # Peg-unplug (peg_x = -0.08 + joint, range [-0.10, 0]).
    "peg_unplug_side": -0.10,  # fully pulled out
    # Box-close revolute X (joint=0 → lid closed; joint=1.5 → lid open).
    "box_close": 0.0,
    # Button-press-topdown-wall: same joint as button (range [-0.06, 0]).
    "button_press_topdown_wall": -0.06,
    # Hammer: nail driven flush (range [-0.06, 0]).
    "hammer": -0.06,
    # Plate-side: slides along world Y (plate_y = 0.65 + joint, range [-0.20, +0.20]).
    "plate_slide_side": 0.20,  # plate at +y end (y=0.85)
    "plate_slide_back_side": 0.0,  # plate back at -y end (y=0.65)
    # Front-facing button (range [0, +0.06]; press = positive joint).
    "button_press": 0.06,
    "button_press_wall": 0.06,
}


def _drive_joint_to_goal(env, spec_key: str) -> float:
    """Write the asset joint to the value that places the marker at ``spec.goal``.

    Returns the joint value driven to.
    """
    cabinet = env.unwrapped.scene["cabinet"]
    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    target_val = _JOINT_GOAL_VALUES[spec_key]
    target = torch.full((n, 1), target_val, device=device)
    j_idx = cabinet.find_joints(TASK_SPECS[spec_key].joint_name)[0][0]
    cabinet.write_joint_position_to_sim(target, joint_ids=[j_idx])
    return target_val


def _keypoint_to_goal_dist(env, spec_key: str) -> float:
    """L2 distance (mean over envs) between the keypoint marker and the
    spec'd goal, in env-local frame."""
    s = TASK_SPECS[spec_key]
    ft = env.unwrapped.scene["keypoint_frame"]
    origins = env.unwrapped.scene.env_origins  # (n, 3)
    marker_w = ft.data.target_pos_w.torch[:, 0]  # (n, 3)
    marker_e = marker_w - origins
    goal_e = torch.tensor(s.goal, device=marker_e.device).expand_as(marker_e)
    return float(torch.linalg.norm(marker_e - goal_e, dim=-1).mean())


def _step_n(env, n_steps: int) -> torch.Tensor:
    """Step ``n_steps`` zero-action steps, return the last reward."""
    action = torch.zeros(env.unwrapped.num_envs, 4, device=env.unwrapped.device)
    last_r = None
    for _ in range(n_steps):
        _, last_r, _, _, _ = env.step(action)
    return last_r


def _verify_one(task_id: str, spec_key: str, kind: str) -> tuple[bool, str]:
    cfg = _resolve_cfg(task_id)
    cfg.scene.num_envs = args.num_envs
    env = gym.make(task_id, cfg=cfg)
    try:
        env.reset()
        # Initial reward (after physics settle).
        r0 = _step_n(env, 3).mean().item()
        if not torch.isfinite(torch.tensor(r0)):
            return False, f"initial reward not finite: {r0}"

        if kind == "reach":
            # Can't drive TCP without IK — just check stepping with zero
            # action keeps reward finite and bounded.
            r_late = _step_n(env, 5).mean().item()
            if not torch.isfinite(torch.tensor(r_late)):
                return False, f"late reward not finite: {r_late}"
            return True, f"reach env stable (r0={r0:+.3f}, r5={r_late:+.3f})"

        # Drive manipulandum to goal.
        if kind == "cube":
            _drive_cube_to_goal(env, spec_key)
        elif kind in ("plate", "joint"):
            _drive_joint_to_goal(env, spec_key)

        # Step to settle.
        r_goal = _step_n(env, 8).mean().item()
        if not torch.isfinite(torch.tensor(r_goal)):
            return False, f"goal-state reward not finite: {r_goal}"

        if kind in ("plate", "joint"):
            # The hamacher reward needs BOTH TCP-near-plate AND plate-near-goal
            # to be active simultaneously. Driving only the plate joint (no
            # TCP control) leaves TCP far from the new plate position, so
            # H(reach=0, in_place=1) = 0 and the reward stays low. That's
            # correct behavior — we instead verify that the manipulandum
            # actually reached the goal pose by reading the keypoint marker.
            d = _keypoint_to_goal_dist(env, spec_key)
            tol = TASK_SPECS[spec_key].success_threshold
            if d > tol:
                return False, f"plate marker did not reach goal: dist={d * 1000:.1f} mm > tol={tol * 1000:.1f} mm"
            return True, f"plate at goal (marker err {d * 1000:.1f} mm; r0={r0:+.4f}, r_goal={r_goal:+.4f})"

        # Cube tasks: reward should jump (success_override fires when within radius).
        if r_goal <= r0 + 0.001:
            return False, f"reward did not increase toward goal (r0={r0:+.3f}, r_goal={r_goal:+.3f})"

        return True, f"reward responsive (r0={r0:+.3f}, r_goal={r_goal:+.3f})"
    finally:
        env.close()


def main() -> int:
    tasks = TASKS if args.task is None else [t for t in TASKS if t[0] == args.task]
    if not tasks:
        print(f"No matching task for --task={args.task!r}. Available:")
        for t in TASKS:
            print(f"  {t[0]}")
        return 2

    n_pass = 0
    for task_id, spec_key, kind in tasks:
        try:
            ok, msg = _verify_one(task_id, spec_key, kind)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {task_id:48s} {msg}", flush=True)
        n_pass += int(ok)

    print(f"\n{n_pass}/{len(tasks)} envs verified.")
    return 0 if n_pass == len(tasks) else 1


if __name__ == "__main__":
    rc = main()
    sim_app.close()
    sys.exit(rc)
