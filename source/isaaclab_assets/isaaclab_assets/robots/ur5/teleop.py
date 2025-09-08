# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from octilab.devices import KeyboardCfg, TeleopCfg


@configclass
class Ur5TeleopCfg:
    keyboard: TeleopCfg = TeleopCfg(
        teleop_devices={
            "device1": TeleopCfg.TeleopDevicesCfg(
                attach_body=SceneEntityCfg("robot", body_names="robotiq_base_link"),
                attach_scope="self",
                pose_reference_body=SceneEntityCfg("robot", body_names="base_link_inertia"),
                reference_axis_remap=("-x", "-y", "z"),
                command_type="pose",
                debug_vis=True,
                teleop_interface_cfg=KeyboardCfg(
                    pos_sensitivity=0.01,
                    rot_sensitivity=0.04,
                    enable_gripper_command=True,
                ),
            ),
        }
    )
