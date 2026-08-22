# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset configuration for the full NIST board."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.factory_presets import (
    EndEffectorBodyCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    IKJointNamesCfg,
)
from isaaclab_tasks.contrib.nist.utils import BetaSamplingStrategyCfg, SamplerCfg
from isaaclab_tasks.utils import SuccessMonitorCfg

from . import mdp
from .board_layout import FIXED_ASSET_NAME_BY_VARIANT, HELD_ASSET_NAMES
from .newton_selection import NewtonBodySelectorCfg


@configclass
class FactoryBoardEventCfg:
    """Generate reset states and own the shared Newton board state."""

    reset_board = EventTerm(
        func=mdp.board_reset,
        mode="reset",
        params={
            "robot_ik_cfg": SceneEntityCfg("robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()),
            "robot_gripper_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg()),
            "gripper_grasp_offset": GripperGraspOffsetCfg(),
            "state_table_size": 65536,
            "fallen_state_table_size": 8192,
            "settle_steps": 20,
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=50),
            "sampling": SamplerCfg(
                strategies=[
                    BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates")
                ],
                eps=1.0e-4,
            ),
            "report": True,
        },
    )

    assembly_state = EventTerm(
        func=mdp.AssemblyState,
        mode="startup",
        params={
            "held_bodies": NewtonBodySelectorCfg(path=tuple(rf".*/{name}(?:/.*)?" for name in HELD_ASSET_NAMES)),
            "fixed_bodies": NewtonBodySelectorCfg(
                path=tuple(rf".*/{name}(?:/.*)?" for name in FIXED_ASSET_NAME_BY_VARIANT)
            ),
            "robot_root_body": NewtonBodySelectorCfg(path=r".*/Robot(?:/.*)?/panda_link0"),
            "workspace": {"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )
