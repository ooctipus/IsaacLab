# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots import SPOT_JOINT_POSITION, SPOT_CFG

from ... import mdp
from ... import position_env_cfg


@configclass
class SpotActionsCfg:
    actions = SPOT_JOINT_POSITION


@configclass
class SportRewardsCfg(position_env_cfg.RewardsCfg):
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
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
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
class SpotEnvMixin:
    actions: SpotActionsCfg = SpotActionsCfg()
    rewards: SportRewardsCfg = SportRewardsCfg()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as spot's body names for sensors
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.pattern_cfg.size = (3.5, 1.5)

        # overwrite as spot's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "body"

        self.rewards.undesired_contact.params["sensor_cfg"].body_names = ["body", ".*leg"]
        self.rewards.feet_lin_acc_l2.params["robot_cfg"].body_names = ".*_foot"
        self.rewards.feet_rot_acc_l2.params["robot_cfg"].body_names = ".*_foot"
        self.rewards.illegal_contact_penalty.params["sensor_cfg"].body_names = "body"

        self.terminations.base_contact.params["sensor_cfg"].body_names = "body"
        self.viewer.body_name = "body"

        self.sim.dt = 0.002
        self.decimation = 10


@configclass
class SpotLocomotionPositionCommandEnvCfg(SpotEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
