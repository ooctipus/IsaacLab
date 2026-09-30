# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for the OVPhysX FrameView.

Run via ``./scripts/run_ovphysx.sh -m pytest`` (kitless, no ``AppLauncher``).
"""

from __future__ import annotations

import pytest

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene_data import SceneDataFormat  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.sim.views import FrameView  # noqa: E402

OVPHYSX_SIM_CFG = SimulationCfg(physics=OvPhysxCfg())

pytestmark = pytest.mark.device_split


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_factory_dispatches_to_ovphysx_frame_view(device):
    """``FrameView(...)`` under an OVPhysX ``SimulationContext`` returns an ``OvPhysxFrameView``."""
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG) as sim:
        InteractiveScene(_OvPhysxFrameViewSceneCfg(num_envs=1, env_spacing=2.0))

        from isaaclab_ov.sim.views import OvPhysxFrameView

        view = _frame_view(sim, "/World/StaticMarker", device)
        assert isinstance(view, OvPhysxFrameView), f"Expected OvPhysxFrameView, got {type(view).__name__}"


def test_world_attached_source_prim_expands_from_clone_plan():
    """A source-only world frame expands across cloned environments without USD replication."""
    device = "cpu"
    OVPHYSX_SIM_CFG.device = device
    with build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_OvPhysxFrameViewSceneCfg(num_envs=4, env_spacing=2.0))
        sim.reset()

        stage = sim.stage
        view = _frame_view(sim, "/World/envs/env_[^/]+/WorldCamera", device)

        assert not stage.GetPrimAtPath("/World/envs/env_1/WorldCamera").IsValid()
        assert view.count == scene.num_envs
        assert view.prims == []
        assert view.prim_paths == [f"/World/envs/env_{i}/WorldCamera" for i in range(scene.num_envs)]
        positions, _ = view.get_world_poses()
    expected_positions = scene.env_origins + torch.tensor([0.25, -0.5, 1.0], device=device)
    torch.testing.assert_close(positions.torch, expected_positions)


# ==================================================================
# Shared FrameView contract suite
# ==================================================================

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

import torch  # noqa: E402
import warp as wp  # noqa: E402
from frame_view_contract_utils import *  # noqa: F401, F403, E402 -- import all contract tests
from frame_view_contract_utils import CHILD_OFFSET, ViewBundle  # noqa: E402

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402


@configclass
class _OvPhysxFrameViewSceneCfg(InteractiveSceneCfg):
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
    world_camera: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorldCamera",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, -0.5, 1.0)),
    )
    static_marker: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/StaticMarker",
        spawn=sim_utils.PinholeCameraCfg(),
    )


def _frame_view(sim, prim_path: str, device: str) -> FrameView:
    view = FrameView(prim_path, simulation_context=sim, device=device)
    view.initialize(sim.get_clone_plan(), sim.get_scene_data_provider())
    return view


@pytest.fixture
def view_factory():
    """OVPhysX factory: CameraMount child Xform at CHILD_OFFSET under each Cube body."""
    contexts: list = []

    def _build(num_envs: int, device: str) -> ViewBundle:
        OVPHYSX_SIM_CFG.device = device
        ctx = build_simulation_context(device=device, sim_cfg=OVPHYSX_SIM_CFG)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        contexts.append(ctx)

        InteractiveScene(_OvPhysxFrameViewSceneCfg(num_envs=num_envs, env_spacing=2.0))

        sim.reset()
        view = _frame_view(sim, "/World/envs/env_[^/]+/Cube/CameraMount", device)

        plan = sim.get_clone_plan()
        path_to_row = {path: index for index, path in enumerate(plan.iter_rigid_body_paths())}
        cube_rows = [path_to_row[f"/World/envs/env_{i}/Cube"] for i in range(num_envs)]
        body_q = sim.get_scene_data_provider().request_transforms(SceneDataFormat.Transform).transforms
        pose_buf_torch = wp.to_torch(body_q)

        def _get_parent_pos(n: int, dev: str) -> torch.Tensor:
            return pose_buf_torch[cube_rows, :3].to(dev).clone()

        def _set_parent_pos(positions: torch.Tensor, n: int) -> None:
            pose_buf_torch[cube_rows, :3] = positions.to(pose_buf_torch.device, pose_buf_torch.dtype)

        return ViewBundle(
            view=view,
            get_parent_pos=_get_parent_pos,
            set_parent_pos=_set_parent_pos,
            teardown=lambda: None,
        )

    yield _build

    for cm in contexts:
        cm.__exit__(None, None, None)
