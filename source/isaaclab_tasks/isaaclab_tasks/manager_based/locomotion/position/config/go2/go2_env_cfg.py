# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from ... import position_env_cfg
from ... import mdp


@configclass
class Go2ActionsCfg:
    actions: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
    )


@configclass
class Go2RewardsCfg(position_env_cfg.RewardsCfg):
    move_forward = RewTerm(func=mdp.forward_velocity, weight=0.4, params={"std": 1})

    gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.8,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class G2Curriculum(position_env_cfg.CurriculumCfg):
    remove_gait_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "gait"})

    remove_forward_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "move_forward"})


@configclass
class Go2EnvMixin:
    actions: Go2ActionsCfg = Go2ActionsCfg()
    rewards: Go2RewardsCfg = Go2RewardsCfg()
    curriculum: G2Curriculum = G2Curriculum()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path="https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/Go2/go2.usd"

@configclass
class Go2LocomotionPositionCommandEnvCfg(Go2EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
