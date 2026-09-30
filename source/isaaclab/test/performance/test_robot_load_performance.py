# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import pytest
from isaaclab_physx.physics import PhysxCfg

from isaaclab import cloner
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer

from isaaclab_assets import ANYMAL_D_CFG, CARTPOLE_CFG

pytestmark = pytest.mark.integration

NUM_ENVS = 4096
SPACING = 2.0


@configclass
class DirectCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/Robots_[^/]*/Robot")
    num_envs: int = NUM_ENVS
    env_spacing: float = SPACING


@pytest.mark.parametrize(
    ("name", "robot_cfg", "expected_load_time", "device"),
    [
        # TODO: regression - this used to be 10
        ("Cartpole", CARTPOLE_CFG, 15.0, "cuda:0"),
        ("Cartpole", CARTPOLE_CFG, 15.0, "cpu"),
        # TODO: regression - this used to be 40
        ("Anymal_D", ANYMAL_D_CFG, 60.0, "cuda:0"),
        ("Anymal_D", ANYMAL_D_CFG, 60.0, "cpu"),
    ],
)
def test_robot_load_performance(name, robot_cfg, expected_load_time, device):
    """Test robot load time."""
    cfg = DirectCfg(robot=robot_cfg.replace(prim_path="/World/Robots_[^/]*/Robot"))
    with build_simulation_context(sim_cfg=cfg.sim, device=device) as sim:
        sim._app_control_on_stop_handle = None
        with Timer(f"{name} load time for device {device}") as timer:
            with cloner.ReplicateSession((cfg.robot,), cfg.num_envs, cfg.env_spacing, env_template="/World/Robots_{}"):
                cfg.robot.class_type(cfg.robot)
            sim.reset()
            elapsed_time = timer.time_elapsed
        assert elapsed_time <= expected_load_time
