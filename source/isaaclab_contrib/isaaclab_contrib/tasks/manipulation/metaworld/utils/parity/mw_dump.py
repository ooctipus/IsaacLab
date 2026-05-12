# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump Meta-World v3 reference fixtures for parity comparison.

Run with the Meta-World venv (NOT the IsaacLab venv) — this script imports
``mujoco`` and ``metaworld`` and must not pull in IsaacLab::

    /home/zhengyuz/Projects/Metaworld/.venv/bin/python \
        source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/metaworld/utils/parity/mw_dump.py

For each MT50 task, writes ``data/<task>.json`` with:

* ``meta``: dt, max_path_length, action/obs space bounds.
* ``reset``: full body xpos table, joint qpos table, hand_init_pos,
  obj_init_pos, target_pos, obs vector — at the canonical ``train_tasks[0]``.
* ``samples``: 50 (obj_init_pos, target_pos) pairs by iterating over
  ``train_tasks[:50]`` — captures the MW per-task sampling distribution.
* ``rollout``: 50-step trajectory of (action, obs, reward, info) under a
  scripted action sequence (deterministic — see ``_rollout_actions``).

The IsaacLab side then loads each task's JSON and compares L1-L7 layers.

Goals (cf. parity-verification plan):

* L1 placement: compare ``reset.body_xpos`` rows for asset-anchor + goal
  marker bodies between MW and IsaacLab (both with ``set_task(train_tasks[0])``).
* L2 joint state: compare ``reset.joint_qpos`` (welded asset's articulated
  DoF — drawer slide, button slider, faucet rotation, etc.).
* L3 sampling: compare ``samples`` distribution moments (mean, std, range)
  with the IsaacLab MetaworldPairedCommand sampler.
* L4 action target: compare DiffIK target pose at each step in ``rollout``
  against MW's mocap pose post-clip.
* L5 obs: compare obs[:39] component-wise between MW rollout and IsaacLab
  rollout (with same scripted actions).
* L6 reward: compare ``rollout.reward`` and ``rollout.info`` per step.
* L7 dynamics: out-of-scope (PhysX vs MuJoCo intentionally diverge).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import metaworld
import numpy as np

_HERE = Path(__file__).parent
_DATA = _HERE / "data"

# MT50 task names (V3) — frozen from MT50.train_classes.keys() at v3.0.0.
MT50_TASKS = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
]


def _rollout_actions(rng: np.random.Generator, n_steps: int = 50) -> np.ndarray:
    """Deterministic scripted action sequence used for rollout parity.

    Reach-down then close-then-lift pattern. Same actions on MW and IsaacLab
    so we can compare obs/reward step-by-step.
    """
    actions = np.zeros((n_steps, 4), dtype=np.float32)
    # Phase 1 — reach down (-Z) for 15 steps, gripper open.
    actions[0:15, 2] = -0.5
    actions[0:15, 3] = -1.0
    # Phase 2 — close gripper for 5 steps (no XYZ motion).
    actions[15:20, 3] = +1.0
    # Phase 3 — lift up (+Z) and translate +Y for 15 steps, gripper closed.
    actions[20:35, 1] = +0.4
    actions[20:35, 2] = +0.6
    actions[20:35, 3] = +1.0
    # Phase 4 — random small actions (deterministic via rng) for 15 steps.
    actions[35:50] = rng.uniform(-0.3, 0.3, size=(15, 4)).astype(np.float32)
    return actions


def _dump_one_task(task_name: str, n_distribution: int = 50) -> dict:
    """Run MW for one task and return the fixture payload."""
    mt = metaworld.MT1(task_name, seed=0)
    env_cls = mt.train_classes[task_name]
    env = env_cls()

    # Anchor task = train_tasks[0]: reproducible.
    canonical_task = mt.train_tasks[0]
    env.set_task(canonical_task)
    obs, info = env.reset(seed=0)

    data = env.unwrapped.data
    model = env.unwrapped.model

    # Body xpos table (named bodies only — ignore the empty '' body name).
    bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if not name:
            continue
        bodies.append(
            {
                "name": name,
                "xpos": data.body(i).xpos.tolist(),
                "xquat": data.body(i).xquat.tolist(),  # MuJoCo quat is (w,x,y,z)
            }
        )

    # Joint qpos table.
    joints = []
    for i in range(model.njnt):
        j = model.jnt(i)
        # qpos addr is shape (1,) for hinge/slide, (7,) for free joints.
        addr0 = int(j.qposadr[0])
        # determine qpos width from joint type
        # mujoco.mjtJoint: 0=free(7), 1=ball(4), 2=slide(1), 3=hinge(1)
        jtype = int(j.type[0])
        width = {0: 7, 1: 4, 2: 1, 3: 1}.get(jtype, 1)
        qpos_slice = data.qpos[addr0 : addr0 + width].tolist()
        joints.append({"name": j.name, "type": jtype, "qpos": qpos_slice})

    reset_payload = {
        "body_xpos": bodies,
        "joint_qpos": joints,
        "hand_init_pos": np.asarray(env.hand_init_pos).tolist(),
        "init_tcp": np.asarray(env.init_tcp).tolist() if hasattr(env, "init_tcp") else None,
        "obj_init_pos": np.asarray(env.obj_init_pos).tolist(),
        "target_pos": np.asarray(env._target_pos).tolist(),
        "obs": obs.tolist(),
    }

    # Sampling distribution — iterate train_tasks[:n_distribution].
    samples = []
    for k in range(min(n_distribution, len(mt.train_tasks))):
        env.set_task(mt.train_tasks[k])
        env.reset(seed=k)
        samples.append(
            {
                "obj_init_pos": np.asarray(env.obj_init_pos).tolist(),
                "target_pos": np.asarray(env._target_pos).tolist(),
                "hand_init_pos": np.asarray(env.hand_init_pos).tolist(),
            }
        )

    # Scripted-action rollout — back to canonical task.
    env.set_task(canonical_task)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(seed=0xBEEF)
    actions = _rollout_actions(rng, n_steps=50)
    rollout_steps = []
    for t in range(actions.shape[0]):
        a = actions[t].copy()
        # Snapshot pre-step mocap pose (DiffIK comparison anchor).
        try:
            mocap_pos = data.mocap_pos[0].tolist() if data.mocap_pos.shape[0] > 0 else None
            mocap_quat = data.mocap_quat[0].tolist() if data.mocap_quat.shape[0] > 0 else None
        except Exception:  # noqa: BLE001
            mocap_pos = None
            mocap_quat = None
        obs2, rew, term, trunc, info = env.step(a)
        rollout_steps.append(
            {
                "t": t,
                "action": a.tolist(),
                "mocap_pos_pre": mocap_pos,
                "mocap_quat_pre": mocap_quat,
                "obs": obs2.tolist(),
                "reward": float(rew),
                "terminated": bool(term),
                "truncated": bool(trunc),
                "info": {k: float(v) if np.isscalar(v) else v for k, v in info.items()},
            }
        )

    return {
        "meta": {
            "task": task_name,
            "dt": float(env.unwrapped.dt) if hasattr(env.unwrapped, "dt") else 0.0125,
            "max_path_length": int(env.max_path_length),
            "obs_space": {
                "low": np.asarray(env.observation_space.low).tolist(),
                "high": np.asarray(env.observation_space.high).tolist(),
                "shape": list(env.observation_space.shape),
            },
            "act_space": {
                "low": np.asarray(env.action_space.low).tolist(),
                "high": np.asarray(env.action_space.high).tolist(),
                "shape": list(env.action_space.shape),
            },
        },
        "reset": reset_payload,
        "samples": samples,
        "rollout": rollout_steps,
    }


def main(tasks: list[str] | None = None) -> None:
    _DATA.mkdir(parents=True, exist_ok=True)
    tasks = tasks or MT50_TASKS

    ok, fail = 0, 0
    failures: list[tuple[str, str]] = []
    for task in tasks:
        out = _DATA / f"{task}.json"
        try:
            payload = _dump_one_task(task)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {task}: {type(e).__name__}: {e}")
            failures.append((task, traceback.format_exc()))
            fail += 1
            continue
        with open(out, "w") as f:
            json.dump(payload, f)
        n_bytes = out.stat().st_size
        print(f"[OK]   {task} -> {out.name} ({n_bytes // 1024} KB)")
        ok += 1

    print(f"\n[summary] ok={ok} fail={fail} (out={_DATA})")
    if failures:
        log = _DATA / "_failures.log"
        with open(log, "w") as f:
            for task, tb in failures:
                f.write(f"=== {task} ===\n{tb}\n\n")
        print(f"[failures] written to {log}")
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else None)
