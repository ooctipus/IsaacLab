# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Everything else follows."""

import pytest
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Asset, AssetBase, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString

pytestmark = pytest.mark.integration


@configclass
class _GroundSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())


def test_asset_base_cfg_names_asset_implementation():
    cfg = AssetBaseCfg(prim_path="/World/Asset")

    assert isinstance(cfg.class_type, ResolvableString)
    assert cfg.class_type.__name__ == Asset.__name__
    assert issubclass(AssetBase, Asset)


def test_asset_construction_uses_custom_plan_template_without_mutating_cfg():
    cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Ground", spawn=sim_utils.GroundPlaneCfg())
    spawn_calls: list[str] = []
    spawn_ground_plane = cfg.spawn.func

    def count_spawn(prim_path, spawn_cfg, **kwargs):
        spawn_calls.append(prim_path)
        return spawn_ground_plane(prim_path, spawn_cfg, **kwargs)

    cfg.spawn.func = count_spawn

    with build_simulation_context(sim_cfg=sim_utils.SimulationCfg(physics=PhysxCfg()), device="cpu") as sim:
        sim._app_control_on_stop_handle = None
        with cloner.ReplicateSession((cfg,), num_clones=2, env_spacing=1.0, env_template="/World/scenes/scene_{}"):
            asset = cfg.class_type(cfg)

        assert type(asset) is Asset
        assert not hasattr(asset, "data")
        assert cfg.prim_path == "{ENV_REGEX_NS}/Ground"
        assert asset.cfg.prim_path == "/World/scenes/scene_[^/]+/Ground"
        assert spawn_calls == ["/World/scenes/scene_0/Ground"]
        assert asset.prim.IsValid()
        assert asset.prim.GetPath().pathString == "/World/scenes/scene_0/Ground"


def test_interactive_scene_exposes_asset_in_extras():
    with build_simulation_context(sim_cfg=sim_utils.SimulationCfg(physics=PhysxCfg()), device="cpu") as sim:
        sim._app_control_on_stop_handle = None
        cfg = _GroundSceneCfg(num_envs=2, env_spacing=1.0, filter_collisions=False)
        scene = cfg.class_type(cfg)

        assert type(scene.extras["ground"]) is Asset
        assert scene["ground"] is scene.extras["ground"]
