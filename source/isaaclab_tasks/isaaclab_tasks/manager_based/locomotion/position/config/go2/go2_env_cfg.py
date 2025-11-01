# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from ... import position_env_cfg
from ... import mdp


@configclass
class Go2ActionsCfg:
    actions: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class Go2RewardsCfg(position_env_cfg.RewardsCfg):

    move_forward = RewTerm(
        func=mdp.forward_velocity,
        weight=0.3,
        params={
            "std": 1,
            "max_iter": 200,
        },
    )

    air_time = RewTerm(
        func=mdp.air_time_reward,
        weight=1.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    gait = RewTerm(
        func=mdp.GaitReward,
        weight=2.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "max_iterations": 400.0,
        },
    )

    # -- penalties
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )


@configclass
class Go2EnvMixin:
    actions: Go2ActionsCfg = Go2ActionsCfg()
    rewards: Go2RewardsCfg = Go2RewardsCfg()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path="https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/Go2/go2.usd"
        self.rewards.undesired_contact.params["sensor_cfg"].body_names = [".*thigh"]
        self.rewards.feet_lin_acc_l2.params["robot_cfg"].body_names = ".*_foot"
        self.rewards.feet_rot_acc_l2.params["robot_cfg"].body_names = ".*_foot"
        self.rewards.stalling.weight *= 10
        self.rewards.feet_lin_acc_l2.weight *= 2
        self.rewards.joint_torque_l2.weight *= 2
        self.rewards.action_rate_l2.weight *= 2
        self.sim.dt = 0.0025
        self.decimation = 8


@configclass
class Go2LocomotionPositionCommandEnvCfg(Go2EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
