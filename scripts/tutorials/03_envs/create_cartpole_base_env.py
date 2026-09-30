# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    uv run python scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append launcher arguments and parse Hydra-style preset overrides
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import math

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleSceneCfg

from isaaclab_newton.physics import MJWarpSolverCfg  # isort:skip
from isaaclab_ov.physics import OvPhysxCfg  # isort:skip
from isaaclab_physx.physics import PhysxCfg  # isort:skip


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.005,
        render_interval=decimation,
        device=args_cli.device,
        physics=preset(
            default=PhysxCfg(),
            isaacsim_physx=PhysxCfg(),
            ovphysx=OvPhysxCfg(),
            newton_mjwarp=MJWarpSolverCfg(),
        ),
    )


def main():
    """Main function."""
    env_cfg = resolve_config(CartpoleEnvCfg(), config_overrides)
    env_cfg.scene.num_envs = args_cli.num_envs
    with launch_simulation(env_cfg, args_cli):
        env = ManagerBasedEnv(cfg=env_cfg)

        count = 0
        while env.sim.is_headless_or_exist_active_visualizer():
            with torch.inference_mode():
                if count % 300 == 0:
                    count = 0
                    env.reset()
                    print("-" * 80)
                    print("[INFO]: Resetting environment...")
                joint_efforts = torch.randn_like(env.action_manager.action)
                obs, _ = env.step(joint_efforts)
                print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
                count += 1

        env.close()


if __name__ == "__main__":
    main()
