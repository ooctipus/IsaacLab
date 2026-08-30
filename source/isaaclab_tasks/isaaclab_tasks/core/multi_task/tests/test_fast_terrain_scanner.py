# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Behavioral contract tests for :class:`FastTerrainScanner`.

Asserts that when the sensor is attached to an articulation body, its world
pose (``pos_w``, ``quat_w``) and ray hits (``ray_hits_w``) follow the body
across both physics integration and explicit state writes.

Regression target: PhysX does not write articulation body transforms back to
Fabric, so a naive :class:`FrameView`-backed sensor reads stale spawn-time
poses and the agent ends up training on a constant heightmap. The fix routes
the sensor pose through ``Articulation.data.body_{pos,quat}_w``, which is
GPU-fresh from PhysX every step. These tests assert the contract regardless of
*how* the sensor gets bound — today via :meth:`FastTerrainScanner.bind_articulation`,
tomorrow via auto-resolution from the prim path.
"""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("isaaclab.app") is None:
    pytest.skip("IsaacLab app launcher not available.", allow_module_level=True)

from isaaclab.app import AppLauncher

if not AppLauncher.is_available():
    pytest.skip("Full Isaac Sim runtime not available.", allow_module_level=True)

simulation_app = AppLauncher(headless=True).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.cloner import ClonePlan
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh

from isaaclab_tasks.core.multi_task.sensors import FastTerrainScanner, FastTerrainScannerCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

pytestmark = pytest.mark.isaacsim_ci

_GROUND_PATH = "/World/Ground"
_ROBOT_PATH = "/World/Cartpole"
_BODY_NAME = "pole"
_DT = 0.005


def _make_sim() -> sim_utils.SimulationContext:
    """Fresh stage with a 200 m flat ground plane at z=0."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=_DT))
    sim.set_clone_plan(
        ClonePlan(
            sources=("/World",),
            destinations=("/World",),
            clone_mask=torch.ones((1, 1), dtype=torch.bool, device=sim.device),
        )
    )
    mesh = make_plane(size=(200, 200), height=0.0, center_zero=True)
    create_prim_from_mesh(_GROUND_PATH, mesh)
    sim_utils.update_stage()
    return sim


def _scanner_cfg() -> FastTerrainScannerCfg:
    """Sensor parented to the cartpole pole body, scanning straight down."""
    return FastTerrainScannerCfg(
        prim_path=f"{_ROBOT_PATH}/{_BODY_NAME}",
        mesh_prim_paths=[_GROUND_PATH],
        update_period=0.0,
        offset=FastTerrainScannerCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=(2.0, 2.0)),
        ray_alignment="world",
    )


@pytest.fixture
def env():
    """Build {sim, articulation, scanner} with the scanner bound to the pole body.

    The full path is exercised: stage → ground → cartpole spawn → scanner spawn
    under the body prim → ``sim.reset()`` → ``bind_articulation``.
    """
    sim = _make_sim()
    articulation = Articulation(CARTPOLE_CFG.replace(prim_path=_ROBOT_PATH))
    sim_utils.update_stage()
    scanner = FastTerrainScanner(_scanner_cfg())
    sim.reset()
    scanner.bind_articulation(articulation, _BODY_NAME)
    body_idx = articulation.find_bodies(_BODY_NAME)[0][0]
    yield sim, articulation, scanner, body_idx
    sim.stop()
    sim.clear_instance()


def _refresh(scanner: FastTerrainScanner, articulation: Articulation) -> None:
    """Pump articulation buffers and force the sensor to recompute."""
    articulation.update(_DT)
    scanner.update(_DT, force_recompute=True)


def _mean_ray_hit_x(scanner: FastTerrainScanner) -> float:
    """Mean x-coordinate of finite ray hits in env 0.

    With ``ray_alignment="world"`` and a symmetric grid pattern, this collapses
    to the sensor's world x-coordinate — a direct readout that catches stale
    sensor poses *and* miswired ray transforms.
    """
    hits_x = scanner.data.ray_hits_w.torch[0, :, 0]
    finite = hits_x[torch.isfinite(hits_x)]
    return finite.mean().item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sensor_pose_matches_body_at_init(env):
    """Right after ``sim.reset`` + bind, sensor must mirror body pose exactly."""
    _, articulation, scanner, body_idx = env
    _refresh(scanner, articulation)

    expected_pos = articulation.data.body_pos_w.torch[:, body_idx]
    expected_quat = articulation.data.body_quat_w.torch[:, body_idx]
    assert torch.allclose(scanner.data.pos_w.torch, expected_pos, atol=1e-5), (
        f"Sensor pos_w {scanner.data.pos_w.torch.tolist()} does not match body_pos_w {expected_pos.tolist()} at init."
    )
    assert torch.allclose(scanner.data.quat_w.torch, expected_quat, atol=1e-5), (
        f"Sensor quat_w {scanner.data.quat_w.torch.tolist()} does not match"
        f" body_quat_w {expected_quat.tolist()} at init."
    )


def test_sensor_tracks_body_through_physics_steps(env):
    """Physics integration must propagate to the sensor.

    Apply non-zero joint velocities (slider + pole revolute) and let the
    integrator run. The pole body translates *and* rotates. Sensor pose, sensor
    quat, and ray hits must all reflect the new body pose.
    """
    sim, articulation, scanner, body_idx = env
    _refresh(scanner, articulation)
    pos_before = scanner.data.pos_w.torch.clone()
    quat_before = scanner.data.quat_w.torch.clone()
    mean_x_before = _mean_ray_hit_x(scanner)

    articulation.write_joint_position_to_sim_index(position=torch.zeros(1, 2, device=articulation.device))
    articulation.write_joint_velocity_to_sim_index(velocity=torch.tensor([[10.0, 5.0]], device=articulation.device))
    for _ in range(80):
        sim.step(render=False)
        articulation.update(_DT)
    scanner.update(_DT, force_recompute=True)

    # Strong invariant: sensor pose tracks body tensors exactly.
    expected_pos = articulation.data.body_pos_w.torch[:, body_idx]
    expected_quat = articulation.data.body_quat_w.torch[:, body_idx]
    assert torch.allclose(scanner.data.pos_w.torch, expected_pos, atol=1e-5), (
        "Sensor pos_w drifted from body_pos_w after physics steps."
    )
    assert torch.allclose(scanner.data.quat_w.torch, expected_quat, atol=1e-5), (
        "Sensor quat_w drifted from body_quat_w after physics steps."
    )

    # Behavior: pose actually changed (not just trivially-equal stale values).
    pos_delta = (scanner.data.pos_w.torch - pos_before).norm().item()
    quat_delta = (scanner.data.quat_w.torch - quat_before).norm().item()
    assert pos_delta > 0.05, (
        f"Body translated under joint velocity but sensor pos_w barely moved"
        f" ({pos_delta:.4f}m). Sensor is reading stale data."
    )
    assert quat_delta > 0.05, (
        f"Pole rotated under joint velocity but sensor quat_w barely moved"
        f" (Δ={quat_delta:.4f}). quat tracking is broken or wired to the wrong body."
    )

    # Ray hits must reflect the pose change. Mean ray-hit x ≈ sensor world x.
    mean_x_after = _mean_ray_hit_x(scanner)
    sensor_x_delta = (scanner.data.pos_w.torch[0, 0] - pos_before[0, 0]).item()
    assert abs((mean_x_after - mean_x_before) - sensor_x_delta) < 0.1, (
        f"Ray-hit x shift ({mean_x_after - mean_x_before:.3f}m) does not match sensor x shift ({sensor_x_delta:.3f}m)."
    )


def test_sensor_tracks_body_through_state_reset(env):
    """Instantaneous joint-state writes must propagate to the sensor.

    Mirrors what an env reset does: jam new joint positions, step once to flush
    PhysX → articulation buffers, and verify the sensor sees the new pose.
    """
    sim, articulation, scanner, body_idx = env
    _refresh(scanner, articulation)
    pos_before = scanner.data.pos_w.torch.clone()
    quat_before = scanner.data.quat_w.torch.clone()
    mean_x_before = _mean_ray_hit_x(scanner)

    new_joint_pos = torch.tensor([[3.0, 0.6]], device=articulation.device)
    articulation.write_joint_position_to_sim_index(position=new_joint_pos)
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 2, device=articulation.device))
    sim.step(render=False)
    articulation.update(_DT)
    scanner.update(_DT, force_recompute=True)

    expected_pos = articulation.data.body_pos_w.torch[:, body_idx]
    expected_quat = articulation.data.body_quat_w.torch[:, body_idx]
    assert torch.allclose(scanner.data.pos_w.torch, expected_pos, atol=1e-5), (
        "Sensor pos_w does not match body_pos_w after joint-state reset."
    )
    assert torch.allclose(scanner.data.quat_w.torch, expected_quat, atol=1e-5), (
        "Sensor quat_w does not match body_quat_w after joint-state reset."
    )

    pos_delta = (scanner.data.pos_w.torch - pos_before).norm().item()
    quat_delta = (scanner.data.quat_w.torch - quat_before).norm().item()
    assert pos_delta > 0.5, (
        f"Joint-state reset moved the slider by 3m but sensor pos_w shift was"
        f" {pos_delta:.4f}m. Sensor is stuck on stale pose."
    )
    assert quat_delta > 0.05, (
        f"Joint-state reset rotated the pole by 0.6 rad but sensor quat_w barely changed (Δ={quat_delta:.4f})."
    )

    mean_x_after = _mean_ray_hit_x(scanner)
    sensor_x_delta = (scanner.data.pos_w.torch[0, 0] - pos_before[0, 0]).item()
    assert abs((mean_x_after - mean_x_before) - sensor_x_delta) < 0.1, (
        f"Ray-hit x shift ({mean_x_after - mean_x_before:.3f}m) does not match"
        f" sensor x shift ({sensor_x_delta:.3f}m) after reset."
    )


def test_bind_articulation_unknown_body_raises(env):
    """Binding to a nonexistent body name must fail loudly, not silently."""
    _, articulation, scanner, _ = env
    # ``find_bodies`` raises ValueError on no match before ``bind_articulation``'s
    # own guard kicks in; either path is fine, so just assert ValueError.
    with pytest.raises(ValueError):
        scanner.bind_articulation(articulation, "definitely_not_a_body")
