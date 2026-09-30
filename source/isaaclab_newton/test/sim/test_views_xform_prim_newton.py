# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backend tests for FrameView.

Imports the shared contract tests and provides the Newton-specific
``view_factory`` fixture.  Also includes Newton-only guard tests and
the world-attached prim edge case.
"""

import sys
from pathlib import Path

from isaaclab.test.utils import test_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

import pytest
import torch
import warp as wp
from frame_view_contract_utils import *  # noqa: F401, F403 — import all contract tests
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle, _wp_vec3f, _wp_vec4f
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.views import NewtonSiteFrameView as FrameView

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

NEWTON_SIM_CFG = SimulationCfg(physics=MJWarpSolverCfg())
WORLD_MARKER_POS = (5.0, 3.0, 1.0)
PARENT_OFFSET = (0.2, 0.3, 0.4)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    camera_mount: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube/CameraMount",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CHILD_OFFSET),
    )
    nested_parent: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube/NestedParent",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PARENT_OFFSET),
    )
    nested_child: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube/NestedParent/Child",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CHILD_OFFSET),
    )
    static_marker: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/StaticMarker",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=WORLD_MARKER_POS),
    )


def _sim_context(device, num_envs=4):
    NEWTON_SIM_CFG.device = device
    return build_simulation_context(device=device, sim_cfg=NEWTON_SIM_CFG)


def _frame_view(sim, prim_path: str, device: str) -> FrameView:
    view = FrameView(prim_path, simulation_context=sim, device=device)
    view.initialize(sim.get_clone_plan(), sim.get_scene_data_provider())
    return view


def _get_body_positions(num_envs, device="cpu"):
    manager = sim_utils.SimulationContext.instance()._physics_manager
    model = manager.get_model()
    body_labels = list(model.body_label)
    body_q_t = wp.to_torch(manager.get_state_0().body_q)
    return torch.stack([body_q_t[body_labels.index(f"/World/envs/env_{i}/Cube"), :3] for i in range(num_envs)])


def _set_body_positions(positions, num_envs):
    manager = sim_utils.SimulationContext.instance()._physics_manager
    model = manager.get_model()
    body_labels = list(model.body_label)
    body_q_t = wp.to_torch(manager.get_state_0().body_q)
    for i in range(num_envs):
        body_q_t[body_labels.index(f"/World/envs/env_{i}/Cube"), :3] = positions[i]


# ------------------------------------------------------------------
# Contract fixture
# ------------------------------------------------------------------


@pytest.fixture
def view_factory():
    """Newton factory: CameraMount child Xform at CHILD_OFFSET under each Cube body."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        ctx = _sim_context(device, num_envs=num_envs)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        scene_cfg = _SceneCfg(num_envs=num_envs, env_spacing=2.0)
        InteractiveScene(scene_cfg)
        view = _frame_view(sim, "/World/envs/env_[^/]+/Cube/CameraMount", device)
        sim.reset()

        return ViewBundle(
            view=view,
            get_parent_pos=_get_body_positions,
            set_parent_pos=_set_body_positions,
            teardown=lambda: ctx.__exit__(None, None, None),
        )

    return factory


# ==================================================================
# Newton-only: guard tests
# ==================================================================


@pytest.mark.parametrize("device", test_devices())
def test_reject_body_path(device):
    """FrameView rejects prim paths that resolve to a Newton physics body."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene_cfg = _SceneCfg(num_envs=2, env_spacing=2.0)
    InteractiveScene(scene_cfg)
    sim.reset()

    with pytest.raises(ValueError, match="physics body"):
        _frame_view(sim, "/World/envs/env_[^/]+/Cube", device)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_clone_plan_view_uses_source_child_without_destination_usd(device):
    """FrameView expands a registered body-local site through the ClonePlan."""
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene_cfg = _SceneCfg(num_envs=num_envs, env_spacing=2.0)
    InteractiveScene(scene_cfg)

    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Cube").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Cube").IsValid()
    view = _frame_view(sim, "/World/envs/env_[^/]+/Cube/CameraMount", device)
    sim.reset()

    assert view.count == num_envs
    assert not stage.GetPrimAtPath("/World/envs/env_1/Cube/CameraMount").IsValid()
    pos = view.get_world_poses()[0].torch
    expected = _get_body_positions(num_envs, device) + torch.tensor(CHILD_OFFSET, device=device)
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_view_can_resolve_from_body_labels_after_reset(device):
    """FrameView can resolve a body-local frame directly from Newton body labels."""
    num_envs = 3
    ctx = _sim_context(device, num_envs=num_envs)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene_cfg = _SceneCfg(num_envs=num_envs, env_spacing=2.0)
    InteractiveScene(scene_cfg)
    sim.reset()
    view = _frame_view(sim, "/World/envs/env_[^/]+/Cube/CameraMount", device)

    pos = view.get_world_poses()[0].torch
    expected = _get_body_positions(num_envs, device) + torch.tensor(CHILD_OFFSET, device=device)
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", test_devices())
def test_local_pose_is_relative_to_non_physics_parent(device):
    """Local transforms are relative to the immediate planned parent, not its rigid body."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=2, env_spacing=2.0))
    sim.reset()

    view = _frame_view(sim, "/World/envs/env_[^/]+/Cube/NestedParent/Child", device)
    local_pos = view.get_local_poses()[0].torch
    torch.testing.assert_close(local_pos, torch.tensor(CHILD_OFFSET, device=device).expand(2, -1), atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


# ==================================================================
# Newton edge case: world-attached prim (body=-1)
# ==================================================================


@pytest.mark.parametrize("device", test_devices())
def test_world_attached_returns_initial_pose(device):
    """A world-rooted frame returns its configured position."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene_cfg = _SceneCfg(num_envs=2, env_spacing=2.0)
    InteractiveScene(scene_cfg)

    sim.reset()
    view = _frame_view(sim, "/World/StaticMarker", device)

    pos = view.get_world_poses()[0].torch
    expected = torch.tensor([list(WORLD_MARKER_POS)], device=device)
    torch.testing.assert_close(pos, expected, atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)


@pytest.mark.parametrize("device", test_devices())
def test_world_attached_set_world_roundtrip(device):
    """A world-attached prim can be repositioned via set_world_poses."""
    ctx = _sim_context(device, num_envs=2)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    scene_cfg = _SceneCfg(num_envs=2, env_spacing=2.0)
    InteractiveScene(scene_cfg)

    sim.reset()
    view = _frame_view(sim, "/World/StaticMarker", device)

    new_pos = _wp_vec3f([[10.0, 20.0, 30.0]], device=device)
    new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]], device=device)
    with view.xform_world_space_writer() as w:
        w.set_poses(new_pos, new_quat)

    ret_pos, ret_quat = view.get_world_poses()
    torch.testing.assert_close(ret_pos.torch, wp.to_torch(new_pos), atol=1e-5, rtol=0)
    torch.testing.assert_close(ret_quat.torch, wp.to_torch(new_quat), atol=1e-5, rtol=0)
    ctx.__exit__(None, None, None)
