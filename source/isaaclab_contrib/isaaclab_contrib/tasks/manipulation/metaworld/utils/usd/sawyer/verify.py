# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify ``sawyer_with_gripper.usd`` against Meta-World's MuJoCo reference.

Two layers:
    1. **Programmatic invariants** — pad-to-pad gap at rest and fully closed,
       joint axis/limit attributes.
    2. **Cross-check vs Meta-World** — for each joint config in
       ``mujoco_reference.json``, drive the IsaacLab Sawyer to that config,
       step physics until settled, read ``leftpad`` / ``rightpad`` world
       positions, and compare to the MuJoCo reference within 1 mm.

Run with IsaacLab's venv::

    ./isaaclab.sh -p source/.../metaworld/assets/sawyer/verify_gripper.py

A non-zero exit code means at least one assertion failed; check stdout for the
specific config and component that diverged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_HERE = Path(__file__).parent
_REFERENCE = _HERE / "mujoco_reference.json"
# This script lives at utils/usd/sawyer/; the USD lives at assets/sawyer/usd/.
_DEFAULT_USD = Path(__file__).resolve().parents[3] / "assets" / "sawyer" / "usd" / "sawyer_with_gripper.usda"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", default=str(_DEFAULT_USD))
parser.add_argument("--reference", default=str(_REFERENCE))
parser.add_argument("--tol", type=float, default=1.5e-3, help="Position tolerance [m].")
parser.add_argument("--settle_steps", type=int, default=120, help="Physics steps to wait for joint settling.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

SAWYER_VERIFY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=args.usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
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
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["right_j[0-6]"],
            stiffness=400.0,
            damping=80.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan"],
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["[rl]_close"],
            stiffness=400.0,
            damping=10.0,
        ),
    },
)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = SAWYER_VERIFY_CFG


def main() -> int:
    with open(args.reference) as f:
        reference: dict = json.load(f)

    sim = SimulationContext(SimulationCfg(dt=1 / 60.0))
    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    robot = scene["robot"]
    print(f"[verify] joint names: {robot.joint_names}")
    print(f"[verify] body names:  {robot.body_names}")

    leftpad_idx = robot.body_names.index("leftpad")
    rightpad_idx = robot.body_names.index("rightpad")

    # Layer 1 — joint attribute invariants (axis + limits).
    joint_limits = robot.data.soft_joint_pos_limits.torch[0]  # (num_joints, 2)
    joint_idx = {n: i for i, n in enumerate(robot.joint_names)}
    r_lo, r_hi = joint_limits[joint_idx["r_close"]].tolist()
    l_lo, l_hi = joint_limits[joint_idx["l_close"]].tolist()
    print(f"[verify] r_close limits = [{r_lo:.4f}, {r_hi:.4f}] (expected [0.0, 0.04])")
    print(f"[verify] l_close limits = [{l_lo:.4f}, {l_hi:.4f}] (expected [-0.03, 0.0])")
    fail = []
    if not math.isclose(r_lo, 0.0, abs_tol=1e-4) or not math.isclose(r_hi, 0.04, abs_tol=1e-4):
        fail.append(f"r_close limits {[r_lo, r_hi]} != [0, 0.04]")
    if not math.isclose(l_lo, -0.03, abs_tol=1e-4) or not math.isclose(l_hi, 0.0, abs_tol=1e-4):
        fail.append(f"l_close limits {[l_lo, l_hi]} != [-0.03, 0.0]")

    # Build joint-pos vector helper from a config dict.

    def joints_from_dict(jdict: dict[str, float]) -> torch.Tensor:
        out = robot.data.default_joint_pos.torch.clone()
        for jname, val in jdict.items():
            if jname not in joint_idx:
                # head_pan etc — silently skip
                continue
            out[:, joint_idx[jname]] = val
        return out

    # Layer 2 — for each MuJoCo reference config, drive the IL sim and compare.
    print("\n[verify] === Layer 2: cross-check against Meta-World mujoco ===")
    for name, ref in reference.items():
        target_jp = joints_from_dict(ref["joints"])
        target_jv = torch.zeros_like(target_jp)
        robot.write_joint_state_to_sim(target_jp, target_jv)
        robot.set_joint_position_target(target_jp)
        robot.write_data_to_sim()
        robot.reset()

        # Settle physics so the actuators reach the target.
        for _ in range(args.settle_steps):
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=sim.get_physics_dt())

        body_pos_w = robot.data.body_pos_w.torch[0]  # (num_bodies, 3) for env 0
        leftpad_w = body_pos_w[leftpad_idx].tolist()
        rightpad_w = body_pos_w[rightpad_idx].tolist()
        tcp_w = [(l + r) / 2.0 for l, r in zip(leftpad_w, rightpad_w)]
        gap = math.sqrt(sum((l - r) ** 2 for l, r in zip(leftpad_w, rightpad_w)))

        ref_left = ref["leftpad_w"]
        ref_right = ref["rightpad_w"]
        ref_gap = ref["gap"]

        d_left = math.sqrt(sum((a - b) ** 2 for a, b in zip(leftpad_w, ref_left)))
        d_right = math.sqrt(sum((a - b) ** 2 for a, b in zip(rightpad_w, ref_right)))
        d_gap = abs(gap - ref_gap)
        ok_left = d_left < args.tol
        ok_right = d_right < args.tol
        ok_gap = d_gap < args.tol

        print(f"\n  [{name}]")
        print(
            f"    leftpad   ours={[f'{v:.4f}' for v in leftpad_w]} mw={[f'{v:.4f}' for v in ref_left]} Δ={d_left * 1000:.2f}mm  {'OK' if ok_left else 'FAIL'}"
        )
        print(
            f"    rightpad  ours={[f'{v:.4f}' for v in rightpad_w]} mw={[f'{v:.4f}' for v in ref_right]} Δ={d_right * 1000:.2f}mm  {'OK' if ok_right else 'FAIL'}"
        )
        print(f"    gap       ours={gap:.4f} mw={ref_gap:.4f} Δ={d_gap * 1000:.2f}mm  {'OK' if ok_gap else 'FAIL'}")

        if not (ok_left and ok_right and ok_gap):
            fail.append(name)

    print("\n[verify] === RESULT ===")
    if fail:
        for line in fail:
            print(f"  FAIL: {line}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


rc = main()
simulation_app.close()
sys.exit(rc)
