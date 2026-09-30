# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendSceneCfg, MultiBackendSimulationCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class CartpolePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonSolverCfg = MJWarpSolverCfg(
        njmax=5,
        nconmax=3,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonSolverCfg = KaminoPADMMSolverCfg(
        sparse_jacobian=True,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_vbd: NewtonSolverCfg = VBDSolverCfg(debug_mode=False, use_cuda_graph=True)
    default = newton_mjwarp


@configclass
class CartpoleSceneCfg(MultiBackendSceneCfg):
    """Cartpole assets constructed and cloned as one scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
        ),
    )


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=1 / 120, render_interval=decimation, physics=CartpolePhysicsCfg()
    )

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_cart_position_range = (-1.0, 1.0)  # [m]
    initial_cart_velocity_range = (-0.5, 0.5)  # [m/s]
    initial_pole_angle_range = (-0.25 * math.pi, 0.25 * math.pi)  # [rad]
    initial_pole_velocity_range = (-0.25 * math.pi, 0.25 * math.pi)  # [rad/s]
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
