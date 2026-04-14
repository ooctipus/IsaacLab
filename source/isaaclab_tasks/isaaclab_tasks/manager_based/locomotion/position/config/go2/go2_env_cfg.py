# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.utils import configclass

from ... import position_env_cfg


@configclass
class Go2EnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path="https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/Go2/go2.usd"
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # original joint position felt a more squatted stand, whereas this stands higher and seemed more elegant.
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.35,
            "R[L,R]_thigh_joint": 0.35,
            ".*_calf_joint": -0.873,
        }


@configclass
class Go2LocomotionPositionCommandEnvCfg(Go2EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
