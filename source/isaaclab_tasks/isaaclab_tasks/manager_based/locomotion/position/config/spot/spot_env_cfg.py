# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_assets.robots.spot import SPOT_JOINT_POSITION, SPOT_CFG

from ... import mdp
from ... import position_env_cfg


@configclass
class SpotActionsCfg:
    actions = SPOT_JOINT_POSITION


@configclass
class SportRewardsCfg(position_env_cfg.RewardsCfg):
    move_forward = RewTerm(
        func=mdp.forward_velocity, weight=0.4, params={"std": 1},
    )

@configclass
class G2CurriculumCfg(position_env_cfg.CurriculumCfg):

    remove_forward_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "move_forward"})


@configclass
class SpotEnvMixin:
    actions: SpotActionsCfg = SpotActionsCfg()
    rewards: SportRewardsCfg = SportRewardsCfg()
    curriculum: G2CurriculumCfg = G2CurriculumCfg()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as spot's body names for sensors
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"

        # overwrite as spot's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "body"
        self.viewer.body_name = "body"

        self.sim.dt = 0.002
        self.decimation = 10


@configclass
class SpotLocomotionPositionCommandEnvCfg(SpotEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
