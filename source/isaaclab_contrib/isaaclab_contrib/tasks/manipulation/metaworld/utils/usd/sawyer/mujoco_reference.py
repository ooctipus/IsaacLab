# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate gold-standard gripper pad world positions from Meta-World's MuJoCo
sim, for several arm joint configurations. Output is JSON, consumed by
:mod:`verify_gripper`.

Run with **Meta-World's** dedicated venv, not IsaacLab's::

    /home/zhengyuz/Projects/Metaworld/.venv/bin/python \
        source/.../metaworld/assets/sawyer/mujoco_reference.py

The output file lands at ``mujoco_reference.json`` next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).parent
_OUT = _HERE / "mujoco_reference.json"
# We use the Meta-World reach task to host the Sawyer rig with the gripper.
_MJCF = "/home/zhengyuz/Projects/Metaworld/metaworld/assets/sawyer_xyz/sawyer_reach_v3.xml"

# Joint configurations to test. Each entry: name → (joint name → angle [rad]).
# right_j0..6 are the 7 arm joints; r_close/l_close start at 0 (rest = open).
_CONFIGS: dict[str, dict[str, float]] = {
    "default_pose": {
        "right_j0": 0.0,
        "right_j1": -0.785,
        "right_j2": 0.0,
        "right_j3": 1.05,
        "right_j4": 0.0,
        "right_j5": 1.3,
        "right_j6": 0.0,
        "r_close": 0.0,
        "l_close": 0.0,
    },
    "fully_closed": {
        "right_j0": 0.0,
        "right_j1": -0.785,
        "right_j2": 0.0,
        "right_j3": 1.05,
        "right_j4": 0.0,
        "right_j5": 1.3,
        "right_j6": 0.0,
        "r_close": 0.04,
        "l_close": -0.03,
    },
    "rotated_j0": {
        "right_j0": 1.0,
        "right_j1": -0.785,
        "right_j2": 0.0,
        "right_j3": 1.05,
        "right_j4": 0.0,
        "right_j5": 1.3,
        "right_j6": 0.0,
        "r_close": 0.0,
        "l_close": 0.0,
    },
    "wrist_twist": {
        "right_j0": 0.0,
        "right_j1": -0.785,
        "right_j2": 0.0,
        "right_j3": 1.05,
        "right_j4": 0.0,
        "right_j5": 1.3,
        "right_j6": 1.5,
        "r_close": 0.0,
        "l_close": 0.0,
    },
}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_MJCF)
    data = mujoco.MjData(model)
    print(f"[ref] loaded {_MJCF}, nq={model.nq} nv={model.nv}")
    print(f"[ref] joint names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}")

    payload: dict[str, dict[str, list[float]]] = {}
    for name, joints in _CONFIGS.items():
        # Reset and write joint values.
        mujoco.mj_resetData(model, data)
        for jname, value in joints.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, f"joint {jname!r} not found in MJCF"
            qpos_addr = model.jnt_qposadr[jid]
            data.qpos[qpos_addr] = value
        mujoco.mj_forward(model, data)

        leftpad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leftpad")
        rightpad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rightpad")
        assert leftpad_id >= 0 and rightpad_id >= 0

        leftpad_w = data.xpos[leftpad_id].tolist()
        rightpad_w = data.xpos[rightpad_id].tolist()
        tcp_w = (np.array(leftpad_w) + np.array(rightpad_w)) / 2.0
        gap = float(np.linalg.norm(np.array(leftpad_w) - np.array(rightpad_w)))

        payload[name] = {
            "joints": joints,
            "leftpad_w": leftpad_w,
            "rightpad_w": rightpad_w,
            "tcp_w": tcp_w.tolist(),
            "gap": gap,
        }
        print(f"[ref] {name}: leftpad={leftpad_w} rightpad={rightpad_w} gap={gap:.4f}")

    with open(_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ref] wrote {_OUT}")


if __name__ == "__main__":
    main()
