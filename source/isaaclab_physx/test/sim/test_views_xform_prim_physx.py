# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan-bound PhysX FrameView contract over one declarative clone lifecycle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sim"))

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device

simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

import pytest  # noqa: E402
import torch  # noqa: E402
from frame_view_contract_utils import *  # noqa: F401, F403, E402
from frame_view_contract_utils import ATOL, CHILD_OFFSET, ViewBundle  # noqa: E402
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.views import PhysxFrameView as FrameView  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci
WORLD_FRAME_POS = (0.25, -0.5, 1.0)


@configclass
class _PhysicsFrameSceneCfg(InteractiveSceneCfg):
    body: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    child: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Body/Child",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CHILD_OFFSET),
    )
    world_frame: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorldFrame",
        spawn=sim_utils.PinholeCameraCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=WORLD_FRAME_POS),
    )


def _skip_if_unavailable(device: str) -> None:
    if not device.startswith("cuda"):
        return
    index = int(device.split(":")[1]) if ":" in device else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        pytest.skip(f"{device} is unavailable")


@pytest.fixture
def view_factory(request):
    """Build one body-attached plan view from one ReplicateSession lifecycle."""

    def factory(num_envs: int, device: str) -> ViewBundle:
        _skip_if_unavailable(device)
        context = build_simulation_context(
            device=device,
            sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, device=device),
        )
        sim = context.__enter__()
        sim._app_control_on_stop_handle = None
        scene_cfg = _PhysicsFrameSceneCfg(num_envs=num_envs, env_spacing=2.0)
        child = FrameView("/World/envs/env_[^/]+/Body/Child", simulation_context=sim, device=device)
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        child.initialize(sim.get_clone_plan(), sim.get_scene_data_provider())
        closed = False

        def teardown() -> None:
            nonlocal closed
            if not closed:
                closed = True
                context.__exit__(None, None, None)

        request.addfinalizer(teardown)

        def get_parent_pos(_num_envs: int, _device: str) -> torch.Tensor:
            return scene["body"].data.root_link_pos_w.torch.clone()

        def set_parent_pos(positions: torch.Tensor, _num_envs: int) -> None:
            root_pose = torch.zeros((num_envs, 7), device=device)
            root_pose[:, :3] = positions
            root_pose[:, 6] = 1.0
            scene["body"].write_root_pose_to_sim_index(root_pose=root_pose)

        return ViewBundle(child, get_parent_pos, set_parent_pos, teardown)

    return factory


def test_frame_read_pulls_current_physx_pose_without_renderer():
    """A frame read consumes the current native SDP generation without a renderer."""
    device = resolve_test_sim_device()
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, device=device)
    ) as sim:
        sim._app_control_on_stop_handle = None
        view = FrameView("/World/envs/env_[^/]+/Body/Child", simulation_context=sim, device=device)
        scene_cfg = _PhysicsFrameSceneCfg(num_envs=2, env_spacing=2.0)
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        view.initialize(sim.get_clone_plan(), sim.get_scene_data_provider())
        before = view.get_world_poses()[0].torch.clone()
        root_pose = torch.zeros((2, 7), device=device)
        root_pose[:, 0] = torch.tensor([5.0, 8.0], device=device)
        root_pose[:, 2] = 1.0
        root_pose[:, 6] = 1.0
        scene["body"].write_root_pose_to_sim_index(root_pose=root_pose)

        after = view.get_world_poses()[0].torch
        expected = root_pose[:, :3] + torch.tensor(CHILD_OFFSET, device=device)
        assert not torch.equal(before, after)
        torch.testing.assert_close(after, expected, atol=ATOL, rtol=0)
        assert not sim._renderer_entries and not sim.visualizers


def test_world_attached_frame_read_does_not_request_physics(monkeypatch):
    """A fixed planned frame needs neither physics transforms nor renderer storage."""
    device = resolve_test_sim_device()
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, device=device)
    ) as sim:
        sim._app_control_on_stop_handle = None
        view = FrameView("/World/envs/env_[^/]+/WorldFrame", simulation_context=sim, device=device)
        scene_cfg = _PhysicsFrameSceneCfg(num_envs=2, env_spacing=2.0)
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        provider = sim.get_scene_data_provider()
        view.initialize(sim.get_clone_plan(), provider)
        monkeypatch.setattr(
            provider,
            "request_transforms",
            lambda *_args, **_kwargs: pytest.fail("world-attached frame requested physics transforms"),
        )

        positions = view.get_world_poses()[0].torch
        expected = scene.env_origins + torch.tensor(WORLD_FRAME_POS, device=device)
        torch.testing.assert_close(positions, expected, atol=ATOL, rtol=0)
        assert not sim._renderer_entries and not sim.visualizers
