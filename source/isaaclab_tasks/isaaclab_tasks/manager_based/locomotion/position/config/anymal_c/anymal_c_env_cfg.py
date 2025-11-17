# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
import isaaclab_assets.robots.anymal as anymal
from isaaclab.utils import configclass

from ... import position_env_cfg


@configclass
class AnymalCEnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path = "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "^(?!.*(?:FOOT|THIGH)).*$"
        if hasattr(self.terminations, "log_gait"):
            self.terminations.log_gait.params["async_pairs"] = ("LF_FOOT", "RF_FOOT"), ("RH_FOOT", "LH_FOOT"), ("LF_FOOT", "LH_FOOT"), ("RF_FOOT", "RH_FOOT")
            self.terminations.log_gait.params["sync_pairs"] = (("LF_FOOT", "RH_FOOT"), ("RF_FOOT", "LH_FOOT"))


@configclass
class AnymalCSpotLocomotionPositionCommandEnvCfg(AnymalCEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
