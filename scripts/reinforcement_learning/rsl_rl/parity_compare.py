# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity-compare an IsaacLab Meta-World env against the MW reference dump.

For one task at a time, this script:

1. Loads ``utils/parity/data/<task>-v3.json`` (from ``mw_dump.py``).
2. Builds the corresponding ``Isaac-Metaworld-<Task>-Sawyer-v0`` env.
3. Resets and captures IsaacLab-side state for each parity layer:

   * L1 placement: ee/cube/cabinet/goal-marker world poses (env-local).
   * L2 joint state: cabinet joint qpos (when present).
   * L3 sampling: 50-seed (obj_init, target) pairs from the command term.
   * L5 obs: 39-d policy obs at reset + along the rollout.
   * L6 reward: per-step reward + components for the same scripted action
     sequence used in the MW dump.

L4 (action target) is captured by reading the DiffIK ``processed_action``
each step and comparing against MW's ``mocap_pos_pre`` snapshot. L7
(dynamics) is intentionally out of scope (PhysX vs MuJoCo divergence).

A per-task report is written to ``utils/parity/reports/<task>.json``;
``--summary-only`` skips the per-step diff and just prints the headline
delta in mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=None, help="MW v3 task name (e.g. drawer-open-v3). Default: all 50.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--rollout_steps", type=int, default=20, help="Steps to run for L4/L5/L6 comparison.")
parser.add_argument("--summary_only", action="store_true")
parser.add_argument(
    "--data_dir",
    default="source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/metaworld/utils/parity/data",
)
parser.add_argument(
    "--report_dir",
    default="source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/metaworld/utils/parity/reports",
)
parser.add_argument("--threshold_pos_mm", type=float, default=20.0, help="L1 pose tol [mm]")
parser.add_argument("--threshold_joint", type=float, default=0.05, help="L2 joint qpos tol [m or rad]")
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

# Force headless + small num_envs to keep boot cheap.
launcher = AppLauncher(args)
sim_app = launcher.app

import importlib  # noqa: E402

import gymnasium  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_contrib.tasks  # noqa: F401, E402
from isaaclab_contrib.tasks.manipulation.metaworld.metaworld_specs import TASK_SPECS  # noqa: E402
from isaaclab_contrib.tasks.manipulation.metaworld.utils.parity.task_mapping import MW_TO_ISAAC  # noqa: E402

DATA_DIR = Path(args.data_dir)
REPORT_DIR = Path(args.report_dir)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_cfg(task_id: str):
    spec = gymnasium.spec(task_id)
    mod, _, cls = spec.kwargs["env_cfg_entry_point"].partition(":")
    return getattr(importlib.import_module(mod), cls)()


# Same scripted action sequence as MW dump (mw_dump._rollout_actions, but
# truncated to ``--rollout_steps``).
def _rollout_actions(n_steps: int) -> torch.Tensor:
    """Same scripted sequence as ``mw_dump._rollout_actions`` (truncated)."""
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


def _to_torch(x):
    """Accept torch.Tensor or warp ProxyArray; return torch.Tensor."""
    return getattr(x, "torch", x)


def _env_local_pose(world, origins) -> np.ndarray:
    """Return env-0 local position [3] in numpy."""
    return (_to_torch(world)[0] - _to_torch(origins)[0]).detach().cpu().numpy()


def _layer1_placement(env, payload: dict) -> dict:
    """L1: compare env-local positions of the ee, cube/cabinet, goal marker."""
    inner = env.unwrapped
    scene = inner.scene
    origins = scene.env_origins  # (n,3)

    out: dict = {}

    # ee (TCP) — read tcp_frame target_pos_w[:,0] (single-target frame transformer
    # hangs the ee marker on body 'hand').
    if "tcp_frame" in scene.keys():
        tcp_w = _to_torch(scene["tcp_frame"].data.target_pos_w)[:, 0]
        out["tcp_e"] = _env_local_pose(tcp_w, origins).tolist()
    # MW's reset.body_xpos has 'hand' for TCP (env-local since MW envs are at origin).
    mw_tcp = next((b["xpos"] for b in payload["reset"]["body_xpos"] if b["name"] == "hand"), None)
    out["mw_tcp"] = mw_tcp

    # Cube / cabinet root (manipulandum anchor).
    if "cube" in scene.keys():
        cube_w = _to_torch(scene["cube"].data.root_pos_w)
        out["cube_e"] = _env_local_pose(cube_w, origins).tolist()
    if "cabinet" in scene.keys():
        cab_w = _to_torch(scene["cabinet"].data.root_pos_w)
        out["cabinet_e"] = _env_local_pose(cab_w, origins).tolist()

    # Keypoint frame — the actual manipulandum reference body the reward
    # reads (drawer handle / button cap / faucet tip / etc.). This is what
    # should match MW's per-task ``_get_pos_objects()`` output.
    if "keypoint_frame" in scene.keys():
        kp_w = _to_torch(scene["keypoint_frame"].data.target_pos_w)[:, 0]
        out["keypoint_e"] = _env_local_pose(kp_w, origins).tolist()

    # Goal pose — read from command term (goal_marker is a static FabricFrameView
    # without .data; the authoritative goal is the command's per-env target).
    try:
        cmd_term = inner.command_manager.get_term("ee_pose")
        goal_e = _to_torch(cmd_term.command).detach().cpu().numpy()[0]
        out["goal_marker_e"] = goal_e.tolist()
    except (KeyError, AttributeError):
        pass

    # Spec-side init/goal (env-local) — checked against the MW obj_init / target.
    out["mw_obj_init"] = payload["reset"]["obj_init_pos"]
    out["mw_target"] = payload["reset"]["target_pos"]
    out["mw_hand_init"] = payload["reset"]["hand_init_pos"]

    # Headline deltas (mm). For articulated tasks we compare the keypoint
    # frame to MW's obj_init (which IS the manipulandum starting position).
    deltas = {}
    if "tcp_e" in out and out["mw_tcp"] is not None:
        deltas["tcp_mm"] = float(np.linalg.norm(np.array(out["tcp_e"]) - np.array(out["mw_tcp"])) * 1000.0)
    if "goal_marker_e" in out:
        deltas["goal_marker_mm"] = float(
            np.linalg.norm(np.array(out["goal_marker_e"]) - np.array(out["mw_target"])) * 1000.0
        )
    if "keypoint_e" in out:
        deltas["keypoint_to_mw_obj_mm"] = float(
            np.linalg.norm(np.array(out["keypoint_e"]) - np.array(out["mw_obj_init"])) * 1000.0
        )
    elif "cube_e" in out and spec_key_is_mt3(payload):
        # MT3 — cube IS the manipulandum.
        deltas["cube_to_mw_obj_mm"] = float(
            np.linalg.norm(np.array(out["cube_e"]) - np.array(out["mw_obj_init"])) * 1000.0
        )
    out["deltas_mm"] = deltas
    return out


def spec_key_is_mt3(payload: dict) -> bool:
    return payload["meta"]["task"] in ("reach-v3", "push-v3", "pick-place-v3")


def _layer2_joint_state(env, payload: dict) -> dict:
    """L2: compare cabinet joint qpos vs MW joint qpos (matched by name)."""
    inner = env.unwrapped
    scene = inner.scene
    if "cabinet" not in scene.keys():
        return {"skipped": "no cabinet"}

    cab = scene["cabinet"]
    isaac_joint_names = list(cab.data.joint_names)
    isaac_qpos = _to_torch(cab.data.joint_pos).detach().cpu().numpy()[0]  # env 0
    isaac = {n: float(isaac_qpos[i]) for i, n in enumerate(isaac_joint_names)}

    # MW joints — keep only ones that match by name (welded asset's joints).
    mw_joints = {j["name"]: j["qpos"] for j in payload["reset"]["joint_qpos"]}

    matched = {}
    for n in isaac_joint_names:
        if n in mw_joints:
            mw_v = mw_joints[n][0] if isinstance(mw_joints[n], list) else float(mw_joints[n])
            matched[n] = {"isaac": isaac[n], "mw": mw_v, "delta": isaac[n] - mw_v}

    return {
        "isaac_joint_names": isaac_joint_names,
        "mw_joint_names": list(mw_joints.keys()),
        "matched": matched,
        "max_abs_delta": max((abs(d["delta"]) for d in matched.values()), default=0.0),
    }


def _layer3_sampling(env, payload: dict, n_samples: int = 50) -> dict:
    """L3: re-reset N times and collect (obj_init, target) per env-0; compare moments."""
    inner = env.unwrapped
    scene = inner.scene
    cmd_term = inner.command_manager.get_term("ee_pose")

    obj_inits, targets = [], []
    for k in range(n_samples):
        # Re-seed Isaac sampler is not exposed here, but reset re-samples the
        # paired (obj_init, target) per env. We sample env-0 each reset.
        env.reset()
        obj_init_e = _to_torch(cmd_term.obj_init_pos_e).detach().cpu().numpy()[0]
        target_e = _to_torch(cmd_term.command).detach().cpu().numpy()[0]
        obj_inits.append(obj_init_e.tolist())
        targets.append(target_e.tolist())

    isaac_obj = np.array(obj_inits)  # (n,3)
    isaac_tgt = np.array(targets)
    mw_obj = np.array([s["obj_init_pos"] for s in payload["samples"]])
    mw_tgt = np.array([s["target_pos"] for s in payload["samples"]])

    def moments(a: np.ndarray) -> dict:
        return {
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "min": a.min(axis=0).tolist(),
            "max": a.max(axis=0).tolist(),
        }

    return {
        "n_isaac": int(isaac_obj.shape[0]),
        "n_mw": int(mw_obj.shape[0]),
        "isaac_obj_init": moments(isaac_obj),
        "mw_obj_init": moments(mw_obj),
        "isaac_target": moments(isaac_tgt),
        "mw_target": moments(mw_tgt),
        "mean_delta_obj_mm": float(np.linalg.norm(isaac_obj.mean(axis=0) - mw_obj.mean(axis=0)) * 1000.0),
        "mean_delta_target_mm": float(np.linalg.norm(isaac_tgt.mean(axis=0) - mw_tgt.mean(axis=0)) * 1000.0),
        "std_delta_obj_mm": float(np.linalg.norm(isaac_obj.std(axis=0) - mw_obj.std(axis=0)) * 1000.0),
        "std_delta_target_mm": float(np.linalg.norm(isaac_tgt.std(axis=0) - mw_tgt.std(axis=0)) * 1000.0),
    }


def _layer56_rollout(env, payload: dict, n_steps: int) -> dict:
    """L5 (obs) + L6 (reward) — drive the same scripted actions and compare."""
    inner = env.unwrapped
    actions_t = _rollout_actions(n_steps).to(inner.device)

    # Reset to canonical seeded init (matches MW dump's canonical task).
    env.reset()
    obs_init_t = inner.observation_manager.compute()["policy"]
    obs_init = _to_torch(obs_init_t)[0].detach().cpu().numpy()

    isaac_obs = []
    isaac_rew = []
    isaac_processed_actions = []  # L4 — DiffIK xyz target post-clip
    for t in range(n_steps):
        a = actions_t[t : t + 1].expand(inner.num_envs, -1).contiguous()
        obs, rew, _, _, _ = env.step(a)
        # ManagerBasedRLEnv returns obs dict; pull the policy group.
        if isinstance(obs, dict):
            policy_obs = _to_torch(obs["policy"])[0].detach().cpu().numpy()
        else:
            policy_obs = _to_torch(obs)[0].detach().cpu().numpy()
        isaac_obs.append(policy_obs.tolist())
        isaac_rew.append(float(_to_torch(rew)[0].item()))

        # Read DiffIK term's processed action (env-local target xyz).
        # MetaworldArmAction stores the *world-frame* target after clipping.
        try:
            arm_term = inner.action_manager.get_term("arm_xyz_delta")
            proc = getattr(arm_term, "processed_actions", None)
            if proc is not None:
                isaac_processed_actions.append(_to_torch(proc)[0].detach().cpu().numpy().tolist())
        except Exception:  # noqa: BLE001
            pass

    mw_steps = payload["rollout"][:n_steps]
    mw_obs = np.array([s["obs"] for s in mw_steps])
    mw_rew = np.array([s["reward"] for s in mw_steps])
    isaac_obs_a = np.array(isaac_obs)
    isaac_rew_a = np.array(isaac_rew)

    # Per-component obs delta (mean abs) — useful to localize which obs
    # component drifts (TCP, gripper, obj1, goal, prev-obs slice, etc.).
    if isaac_obs_a.shape == mw_obs.shape:
        per_component = np.abs(isaac_obs_a - mw_obs).mean(axis=0).tolist()
    else:
        per_component = None

    return {
        "n_steps": n_steps,
        "obs_init_isaac": obs_init.tolist(),
        "obs_init_mw": payload["reset"]["obs"],
        "obs_init_delta_mean": float(np.abs(obs_init - np.array(payload["reset"]["obs"])).mean()),
        "obs_per_component_mean_abs": per_component,
        "reward_isaac": isaac_rew_a.tolist(),
        "reward_mw": mw_rew.tolist(),
        "reward_mean_delta": float(np.abs(isaac_rew_a - mw_rew).mean()) if mw_rew.size == isaac_rew_a.size else None,
        "isaac_processed_actions": isaac_processed_actions,
        "mw_mocap_pre": [s.get("mocap_pos_pre") for s in mw_steps],
    }


def _verify_one(mw_task: str) -> dict:
    isaac_id, spec_key = MW_TO_ISAAC[mw_task]
    payload_path = DATA_DIR / f"{mw_task}.json"
    if not payload_path.exists():
        return {"task": mw_task, "error": f"missing dump: {payload_path}"}
    payload = json.loads(payload_path.read_text())

    cfg = _resolve_cfg(isaac_id)
    cfg.scene.num_envs = args.num_envs
    env = gym.make(isaac_id, cfg=cfg)
    try:
        env.reset()
        # Step a few zero actions to settle.
        zero = torch.zeros(env.unwrapped.num_envs, 4, device=env.unwrapped.device)
        for _ in range(3):
            env.step(zero)

        report = {
            "task": mw_task,
            "isaac_id": isaac_id,
            "spec_key": spec_key,
            "spec": (
                {
                    "obj_init": list(TASK_SPECS[spec_key].obj_init),
                    "goal": list(TASK_SPECS[spec_key].goal),
                    "joint_name": TASK_SPECS[spec_key].joint_name,
                    "joint_reset_value": TASK_SPECS[spec_key].joint_reset_value,
                }
                if spec_key in TASK_SPECS
                else None
            ),
            "L1_placement": _layer1_placement(env, payload),
            "L2_joint_state": _layer2_joint_state(env, payload),
            "L3_sampling": _layer3_sampling(env, payload, n_samples=20),
            "L56_rollout": _layer56_rollout(env, payload, n_steps=args.rollout_steps),
        }
    finally:
        env.close()

    return report


def _summary_line(rep: dict) -> str:
    if "error" in rep:
        return f"[FAIL] {rep['task']}: {rep['error']}"
    L1 = rep["L1_placement"]["deltas_mm"]
    tcp_mm = L1.get("tcp_mm", float("nan"))
    goal_mm = L1.get("goal_marker_mm", float("nan"))
    kp_mm = L1.get("keypoint_to_mw_obj_mm") or L1.get("cube_to_mw_obj_mm") or float("nan")
    L2_max = rep["L2_joint_state"].get("max_abs_delta", 0.0)
    L3_obj_mm = rep["L3_sampling"]["mean_delta_obj_mm"]
    L3_tgt_mm = rep["L3_sampling"]["mean_delta_target_mm"]
    L3_isaac_std = sum(rep["L3_sampling"]["isaac_obj_init"]["std"])
    L6_dr = rep["L56_rollout"].get("reward_mean_delta")
    L6_str = f"{L6_dr:.3f}" if L6_dr is not None else "n/a"
    return (
        f"[OK]   {rep['task']:>32s}  "
        f"L1: tcp={tcp_mm:6.1f} goal={goal_mm:6.1f} kp={kp_mm:6.1f}  "
        f"L2: jnt={L2_max:.3f}  "
        f"L3: dobj={L3_obj_mm:5.1f} dtgt={L3_tgt_mm:5.1f} std={L3_isaac_std:.3f}  "
        f"L6: |Δr|={L6_str}"
    )


def main() -> None:
    if args.task:
        tasks = [args.task]
    else:
        tasks = list(MW_TO_ISAAC.keys())

    summary_lines = []
    for mw_task in tasks:
        try:
            rep = _verify_one(mw_task)
        except Exception as e:  # noqa: BLE001
            import traceback

            rep = {"task": mw_task, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
        out_path = REPORT_DIR / f"{mw_task}.json"
        out_path.write_text(json.dumps(rep, indent=2))
        line = _summary_line(rep)
        summary_lines.append(line)
        print(line)

    # Aggregate summary at end.
    print()
    print("=" * 110)
    print(f"summary  ({len(summary_lines)} tasks)")
    print("=" * 110)
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
    sim_app.close()
