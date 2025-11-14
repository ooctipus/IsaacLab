# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import H1_CFG  # isort: skip

from ... import position_env_cfg


@configclass
class H1EnvMixin:

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as H1's body names for sensors
        self.scene.robot = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/H1/h1.usd"
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # overwrite as H1's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "^(?!.*ankle_link).*$"
        self.viewer.body_name = "torso_link"


@configclass
class H1LocomotionPositionCommandEnvCfg(H1EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
