# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
import isaaclab_assets.robots.anymal as anymal
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from ... import position_env_cfg
from ... import mdp


@configclass
class AnymalCActionsCfg:
    actions: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class AnymalCRewardCfg(position_env_cfg.RewardsCfg):
    explore = RewTerm(func=mdp.exploration_reward, weight=1.0)


@configclass
class AnymalCCurriculumCfg(position_env_cfg.CurriculumCfg):
    remove_gait_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "explore"})


@configclass
class AnymalCEnvMixin:
    actions: AnymalCActionsCfg = AnymalCActionsCfg()
    rewards: AnymalCRewardCfg = AnymalCRewardCfg()
    curriculum: AnymalCCurriculumCfg = AnymalCCurriculumCfg()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore


@configclass
class AnymalCSpotLocomotionPositionCommandEnvCfg(AnymalCEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
