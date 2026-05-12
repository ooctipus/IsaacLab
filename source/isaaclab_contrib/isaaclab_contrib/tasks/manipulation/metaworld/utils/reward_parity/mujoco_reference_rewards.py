# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate reference V2 reward outputs for reach / push / pick-place from
Meta-World's pure-Python NumPy implementations. Output JSON is consumed by
:mod:`verify_rewards`.

Run with **Meta-World's** dedicated venv::

    /home/zhengyuz/Projects/Metaworld/.venv/bin/python \
        source/.../metaworld/assets/reward_parity/mujoco_reference_rewards.py

The MW venv has the ``metaworld.rewards.*`` modules importable without
loading the full ``metaworld`` package (which would pull in MuJoCo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# We import the MW reward modules directly by path — going through
# `import metaworld` would drag in mujoco env classes we don't need.
sys.path.insert(0, "/home/zhengyuz/Projects/Metaworld")

from metaworld.rewards.pick_place_v2 import pick_place_v2_reward  # noqa: E402
from metaworld.rewards.push_v2 import push_v2_reward  # noqa: E402
from metaworld.rewards.reach_v2 import reach_v2_reward  # noqa: E402

_HERE = Path(__file__).parent
_OUT = _HERE / "reward_reference.json"


def _state_for_reach(rng: np.random.Generator) -> dict:
    """Synthetic state for reach: tcp + target + hand_init."""
    return {
        "tcp": rng.uniform([-0.2, 0.5, 0.06], [0.2, 0.7, 0.6]).tolist(),
        "target": rng.uniform([-0.1, 0.8, 0.05], [0.1, 0.9, 0.30]).tolist(),
        "hand_init": [0.0, 0.6, 0.2],
    }


def _state_for_push(rng: np.random.Generator) -> dict:
    """Synthetic state for push: full caging + in_place inputs."""
    obj_init = rng.uniform([-0.1, 0.6, 0.02], [0.1, 0.7, 0.02])
    target = rng.uniform([-0.1, 0.8, 0.01], [0.1, 0.9, 0.02])
    while np.linalg.norm(obj_init[:2] - target[:2]) < 0.15:
        target = rng.uniform([-0.1, 0.8, 0.01], [0.1, 0.9, 0.02])
    return {
        "obj_init": obj_init.tolist(),
        "target": target.tolist(),
        "obj": rng.uniform([-0.2, 0.5, 0.02], [0.2, 0.9, 0.10]).tolist(),
        "tcp": rng.uniform([-0.2, 0.5, 0.06], [0.2, 0.7, 0.6]).tolist(),
        "init_tcp": rng.uniform([-0.2, 0.5, 0.06], [0.2, 0.7, 0.6]).tolist(),
        "left_pad": rng.uniform([-0.2, 0.5, 0.05], [0.2, 0.8, 0.6]).tolist(),
        "right_pad": rng.uniform([-0.2, 0.5, 0.05], [0.2, 0.8, 0.6]).tolist(),
        "action": rng.uniform(-1.0, 1.0, size=4).tolist(),
    }


def _state_for_pick_place(rng: np.random.Generator) -> dict:
    """Pick-place needs per-pad init positions in addition to push fields."""
    base = _state_for_push(rng)
    # Re-sample target in the pick-place range.
    obj_init = np.array(base["obj_init"])
    target = rng.uniform([-0.1, 0.8, 0.05], [0.1, 0.9, 0.30])
    while np.linalg.norm(obj_init[:2] - target[:2]) < 0.15:
        target = rng.uniform([-0.1, 0.8, 0.05], [0.1, 0.9, 0.30])
    base["target"] = target.tolist()
    base["init_left_pad"] = rng.uniform([-0.2, 0.5, 0.05], [0.2, 0.8, 0.6]).tolist()
    base["init_right_pad"] = rng.uniform([-0.2, 0.5, 0.05], [0.2, 0.8, 0.6]).tolist()
    return base


def main() -> None:
    rng = np.random.default_rng(seed=0xBEEF)
    payload: dict[str, list[dict]] = {"reach": [], "push": [], "pick_place": []}

    # Reach — 20 states.
    for _ in range(20):
        s = _state_for_reach(rng)
        reward, tcp_to_target, in_place = reach_v2_reward(
            tcp=np.array(s["tcp"]),
            target_pos=np.array(s["target"]),
            hand_init_pos=np.array(s["hand_init"]),
        )
        payload["reach"].append(
            {"state": s, "reward": float(reward), "tcp_to_target": float(tcp_to_target), "in_place": float(in_place)}
        )

    # Push — 20 states. push_v2_reward expects (action, obs, **state) where
    # obs[3]=tcp_opened and obs[4:7]=obj position.
    for _ in range(20):
        s = _state_for_push(rng)
        action = np.array(s["action"], dtype=np.float32)
        # build obs slice — only [3] and [4:7] are read
        obs = np.zeros(39, dtype=np.float64)
        # tcp_opened in obs[3] is the *gripper opening* (normalised pad gap).
        # The MW reward uses obs[3] as the trigger; we feed action[-1]'s
        # post-clip equivalent so it's directly comparable.
        obs[3] = float(np.clip(action[-1], 0.0, 1.0))
        obs[4:7] = s["obj"]
        reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place = push_v2_reward(
            action=action,
            obs=obs,
            tcp=np.array(s["tcp"]),
            target_pos=np.array(s["target"]),
            obj_init_pos=np.array(s["obj_init"]),
            left_pad_pos=np.array(s["left_pad"]),
            right_pad_pos=np.array(s["right_pad"]),
            init_tcp=np.array(s["init_tcp"]),
        )
        payload["push"].append(
            {
                "state": s,
                "reward": float(reward),
                "tcp_to_obj": float(tcp_to_obj),
                "tcp_opened": float(tcp_opened),
                "target_to_obj": float(target_to_obj),
                "object_grasped": float(object_grasped),
                "in_place": float(in_place),
            }
        )

    # Pick-place — 20 states.
    for _ in range(20):
        s = _state_for_pick_place(rng)
        action = np.array(s["action"], dtype=np.float32)
        obs = np.zeros(39, dtype=np.float64)
        obs[3] = float(np.clip(action[-1], 0.0, 1.0))
        obs[4:7] = s["obj"]
        reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place = pick_place_v2_reward(
            action=action,
            obs=obs,
            tcp=np.array(s["tcp"]),
            target_pos=np.array(s["target"]),
            obj_init_pos=np.array(s["obj_init"]),
            left_pad_pos=np.array(s["left_pad"]),
            right_pad_pos=np.array(s["right_pad"]),
            init_left_pad=np.array(s["init_left_pad"]),
            init_right_pad=np.array(s["init_right_pad"]),
            init_tcp=np.array(s["init_tcp"]),
        )
        payload["pick_place"].append(
            {
                "state": s,
                "reward": float(reward),
                "tcp_to_obj": float(tcp_to_obj),
                "tcp_opened": float(tcp_opened),
                "obj_to_target": float(obj_to_target),
                "object_grasped": float(object_grasped),
                "in_place": float(in_place),
            }
        )

    with open(_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ref] wrote {_OUT}")
    for task in payload:
        rs = [r["reward"] for r in payload[task]]
        print(f"[ref] {task}: n={len(rs)} reward range [{min(rs):.4f}, {max(rs):.4f}] mean={np.mean(rs):.4f}")


if __name__ == "__main__":
    main()
