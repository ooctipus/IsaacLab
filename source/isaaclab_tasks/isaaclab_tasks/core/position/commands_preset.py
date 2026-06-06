# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command presets selectable via ``env.commands.goal_point.commands=<name>``."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .mdp.commands import RelativeStateCommandCfg


@configclass
class CommandsPresetCfg(PresetCfg):
    """Named command configurations for the position locomotion task."""

    all_commands = {
        "lin_vel_cmd": RelativeStateCommandCfg.VelocityCommands(
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-2.0, 2.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-0.2, 0.2),
            duration=(0.05, 4.0),
        ),
        "ang_vel_cmd": RelativeStateCommandCfg.VelocityCommands(
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-2.0, 2.0),
            duration=(0.05, 4.0),
        ),
        "terrain_position_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "terrain_pose_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=None,
            pos_y=None,
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 4.0),
        ),
        "position_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "pose_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
    }
    terrain = {
        "terrain_pose_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
        "terrain_position_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=None, pos_y=None, pos_z=None, roll=(-0.0, 0.0), pitch=(-0.0, 0.0), yaw=None, duration=(0.05, 4.0)
        ),
        "ang_vel_cmd": RelativeStateCommandCfg.VelocityCommands(
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-2.0, 2.0),
            duration=(0.05, 4.0),
        ),
    }
    terrain_pos = {
        "terrain_position_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
    }
    terrain_pose = {
        "terrain_pose_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
    }
    terrain_stand_up = {
        "terrain_stand_up_cmd": RelativeStateCommandCfg.TerrainCommands(
            pos_x=None, pos_y=None, pos_z=None, roll=(-0.0, 0.0), pitch=(-0.0, 0.0), yaw=None, duration=(0.05, 4.0)
        ),
    }
    pose = {
        "pose_cmd": RelativeStateCommandCfg.PoseCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
    }
    pos = {
        "position_cmd": RelativeStateCommandCfg.PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
    }
    vel = (
        {
            "lin_vel_cmd": RelativeStateCommandCfg.VelocityCommands(
                lin_vel_x=(-2.0, 2.0),
                lin_vel_y=(-2.0, 2.0),
                lin_vel_z=None,
                ang_vel_x=None,
                ang_vel_y=None,
                ang_vel_z=(-0.2, 0.2),
                duration=(0.05, 2.0),
            ),
        },
    )
    default = terrain
