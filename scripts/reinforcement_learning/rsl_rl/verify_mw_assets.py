# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-only verification of hand-built MW-style task assets.

For each asset we:

1. Load the USD into a minimal IsaacLab scene (just the asset + ground + light).
2. Sweep the actuated joint over a few values from its limit range.
3. Read the marker body (handle / button_top / hole / etc.) world position.
4. Compare against the analytically-expected position (computed from the
   joint kinematics) and report pass/fail per sweep value.

This is the verification approach for "implement and verify analytically
without RL training": a deterministic check that the USD's joint axis,
range, and welded-marker frame all match the MW MJCF specification.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", default=None, help="Run only the named asset (drawer/button/window/faucet/door)")
AppLauncher.add_app_launcher_args(parser)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
launcher = AppLauncher(args)
sim_app = launcher.app
import os, sys as _sys
_sys.stdout.flush()
# Force unbuffered stdout for the asset-test prints.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import math  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_contrib.tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import FrameTransformerCfg  # noqa: E402
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# Asset cfgs.
from isaaclab_contrib.tasks.manipulation.metaworld.assets.button import MW_BUTTON_CFG  # noqa: E402
from isaaclab_contrib.tasks.manipulation.metaworld.assets.door import MW_DOOR_CFG  # noqa: E402
from isaaclab_contrib.tasks.manipulation.metaworld.assets.drawer import MW_DRAWER_CFG  # noqa: E402
from isaaclab_contrib.tasks.manipulation.metaworld.assets.faucet import MW_FAUCET_CFG  # noqa: E402
from isaaclab_contrib.tasks.manipulation.metaworld.assets.peg import MW_PEG_BLOCK_CFG  # noqa: E402
from isaaclab_contrib.tasks.manipulation.metaworld.assets.window import MW_WINDOW_CFG  # noqa: E402


def _expected_drawer(jp: float) -> tuple[float, float, float]:
    """drawer handle slides along world Y from y=0.74 (closed) to y=0.58 (open)."""
    return (0.0, 0.74 + jp, 0.09)


def _expected_button(jp: float) -> tuple[float, float, float]:
    """Button slides along world Z from z=0.13 (extended) to z=0.07 (pressed)."""
    return (0.0, 0.85, 0.13 + jp)


def _expected_window(jp: float) -> tuple[float, float, float]:
    """Window slider handle moves along world X from x=-0.04 (closed) to x=+0.16 (open)."""
    return (-0.04 + jp, 0.785 - 0.02, 0.10)


def _expected_faucet(jp_rad: float) -> tuple[float, float, float]:
    """Faucet handle tip rotates around Z axis at (0, 0.85, 0.174). Handle has
    length 0.12 in -y at angle 0; rotated by ``jp_rad`` around Z."""
    base = np.array([0.0, 0.85, 0.174])
    # Tip at angle 0: (0, 0.85 - 0.12, 0.174). Rotated:
    dx = 0.0
    dy = -0.12
    rotated = np.array([
        dx * math.cos(jp_rad) - dy * math.sin(jp_rad),
        dx * math.sin(jp_rad) + dy * math.cos(jp_rad),
        0.0,
    ])
    return tuple((base + rotated).tolist())


def _expected_door(jp_rad: float) -> tuple[float, float, float]:
    """Door handle is on a panel that rotates around vertical Z axis at left
    side (x=0). Handle is at offset (0.32, -0.08, 0) from the hinge in
    door-link local frame; world position is base + rotated offset.

    Base = door_link_pos = (0, 0.95, 0.15). Rotation in -Z (open toward -y)?
    Joint axis = Z, range [-90, 0] deg. Rotation by ``jp_rad`` (negative when
    open) around Z sends (+x, 0) → (cos*x, sin*x, 0)... but y=0 in offset
    means we just rotate (0.32, -0.08).
    """
    base = np.array([0.0, 0.95, 0.15])
    dx, dy = 0.32, -0.08
    rotated = np.array([
        dx * math.cos(jp_rad) - dy * math.sin(jp_rad),
        dx * math.sin(jp_rad) + dy * math.cos(jp_rad),
        0.0,
    ])
    return tuple((base + rotated).tolist())


# ---------------------------------------------------------------------------
# Verification spec table
# ---------------------------------------------------------------------------

ASSETS = [
    {
        "name": "drawer",
        "cfg": MW_DRAWER_CFG,
        "joint": "goal_slidey",
        "source_body": "drawercase",
        "marker_body": "drawer_handle",
        "joint_values": [0.0, -0.04, -0.08, -0.12, -0.16],
        "expected_fn": _expected_drawer,
        "tolerance_m": 0.005,
    },
    {
        "name": "button",
        "cfg": MW_BUTTON_CFG,
        "joint": "btnbox_joint",
        "source_body": "button_box",
        "marker_body": "button_top",
        "joint_values": [0.0, -0.02, -0.04, -0.06],
        "expected_fn": _expected_button,
        "tolerance_m": 0.005,
    },
    {
        "name": "window",
        "cfg": MW_WINDOW_CFG,
        "joint": "window_slide",
        "source_body": "window_frame",
        "marker_body": "window_handle",
        "joint_values": [0.0, 0.05, 0.10, 0.20],
        "expected_fn": _expected_window,
        "tolerance_m": 0.005,
    },
    {
        "name": "faucet",
        "cfg": MW_FAUCET_CFG,
        "joint": "knob_Joint_1",
        "source_body": "faucet_base",
        "marker_body": "handle_tip",
        "joint_values": [0.0, math.pi / 6, math.pi / 3, math.pi / 2],
        "expected_fn": _expected_faucet,
        "tolerance_m": 0.01,
    },
    {
        "name": "door",
        "cfg": MW_DOOR_CFG,
        "joint": "door_hinge",
        "source_body": "door_frame",
        "marker_body": "door_handle",
        "joint_values": [0.0, -math.pi / 6, -math.pi / 3, -math.pi / 2],
        "expected_fn": _expected_door,
        "tolerance_m": 0.01,
    },
]


def main() -> None:
    import sys as _s
    def _p(msg):
        print(msg, flush=True)
        _s.stderr.write(msg + "\n")
        _s.stderr.flush()
    _p(f"\n{'='*80}\nMeta-World task asset verification\n{'='*80}")
    summary = []

    asset_subset = ASSETS
    if args.asset is not None:
        asset_subset = [s for s in ASSETS if s["name"] == args.asset]
        if not asset_subset:
            raise SystemExit(f"Unknown asset name: {args.asset}")

    for spec in asset_subset:
        _p(f"\n— {spec['name']} —")
        # Build a minimal scene with this asset + a frame transformer for the marker.

        @configclass
        class TestSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
            cabinet = spec["cfg"].replace(prim_path="{ENV_REGEX_NS}/Cabinet")
            handle_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/" + spec["source_body"],
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Cabinet/" + spec["marker_body"],
                        name="marker",
                        offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                    ),
                ],
            )

        sim_cfg = SimulationCfg(dt=1.0 / 200.0)
        sim = SimulationContext(sim_cfg)
        scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        cab = scene["cabinet"]
        ft = scene["handle_frame"]
        joint_idx = cab.find_joints(spec["joint"])[0][0]

        all_pass = True
        for jv in spec["joint_values"]:
            target = torch.full((1, 1), jv, device=cab.device)
            cab.write_joint_position_to_sim(target, joint_ids=[joint_idx])
            for _ in range(5):
                sim.step()
                scene.update(sim_cfg.dt)
            origins = scene.env_origins.detach().cpu().numpy()
            marker_w = ft.data.target_pos_w.torch.detach().cpu().numpy()[:, 0]
            marker_e = (marker_w - origins)[0]
            expected = np.array(spec["expected_fn"](jv))
            err = float(np.linalg.norm(marker_e - expected))
            pass_str = "PASS" if err <= spec["tolerance_m"] else "FAIL"
            if err > spec["tolerance_m"]:
                all_pass = False
            _p(f"  joint={jv:+.4f}  marker={marker_e.round(3).tolist()!s:<25} expected={expected.round(3).tolist()!s:<25} err={err*1000:.1f}mm  {pass_str}")

        summary.append((spec["name"], all_pass))
        sim.clear_instance()

    # Final summary
    _p(f"\n{'='*80}\nResults\n{'='*80}")
    for name, ok in summary:
        _p(f"  {name:<10} {'PASS' if ok else 'FAIL'}")
    sim_app.close()


if __name__ == "__main__":
    main()
