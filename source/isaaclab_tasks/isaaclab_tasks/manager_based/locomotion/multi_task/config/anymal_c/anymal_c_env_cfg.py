# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab.utils import configclass

import isaaclab_assets.robots.anymal as anymal

from ... import multi_task_env_cfg


@configclass
class AnymalCEnvMixin:
    def __post_init__(self: multi_task_env_cfg.MultiTaskCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path = (
            "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        )
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class AnymalCMultiTaskCommandEnvCfg(AnymalCEnvMixin, multi_task_env_cfg.MultiTaskCommandEnvCfg):
    pass
