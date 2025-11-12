# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip

from ... import mdp
from ... import position_env_cfg


@configclass
class H1RewardsCfg(position_env_cfg.RewardsCfg):
    move_forward = RewTerm(
        # h1 need a maller forward velocity reward for success reward being not over dominating
        func=mdp.forward_velocity, weight=0.1, params={"std": 1},
    )


@configclass
class H1CurriculumCfg(position_env_cfg.CurriculumCfg):
    remove_forward_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "move_forward"})


@configclass
class H1EnvMixin:
    rewards: H1RewardsCfg = H1RewardsCfg()
    curriculum: H1CurriculumCfg = H1CurriculumCfg()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as H1's body names for sensors
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # overwrite as H1's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        self.viewer.body_name = "torso_link"


@configclass
class H1LocomotionPositionCommandEnvCfg(H1EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
