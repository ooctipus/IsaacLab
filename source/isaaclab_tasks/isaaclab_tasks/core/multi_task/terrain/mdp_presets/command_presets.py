# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command presets selectable via ``env.commands.goal_point.commands=<name>``."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .. import mdp


@configclass
class CommandsPresetCfg(PresetCfg):
    """Named command configurations for the position locomotion task."""

    all_commands = {
        "lin_vel_cmd": mdp.VelocityCommands(
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
        "ang_vel_cmd": mdp.VelocityCommands(
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
        "terrain_position_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 2.0),
        ),
        "terrain_pose_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": mdp.TerrainCommands(
            match_base_pos=False,
            match_base_rot=True,
            duration=(0.05, 4.0),
        ),
        "position_cmd": mdp.PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "pose_cmd": mdp.PoseCommands(
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
        "terrain_pose_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 2.0),
        ),
        "terrain_position_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 2.0),
        ),
    }
    terrain_pos = {
        "terrain_position_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 1.0),
        ),
    }
    terrain_pose = {
        "terrain_pose_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 1.0),
        ),
    }
    pose = {
        "pose_cmd": mdp.PoseCommands(
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
        "position_cmd": mdp.PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
    }
    vel = {
        "lin_vel_cmd": mdp.VelocityCommands(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-2.0, 2.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-0.2, 0.2),
            duration=(0.05, 2.0),
        ),
    }
    default = terrain


@configclass
class CommandPayloadPresetCfg(PresetCfg):
    """Named payload configurations for the position locomotion command."""

    base = mdp.BaseStatePayloadCfg(
        pos_std=0.4,
        rot_std=0.5,
        lin_vel_std=0.2,
        ang_vel_std=0.2,
        success_effort_multiplier=0.8,
        success_min_foot_weight_fraction=0.80,
        success_body_lin_speed_thresh=0.30,
        success_body_ang_speed_thresh=0.30,
    )
    base_foot = mdp.BaseFootStatePayloadCfg(
        pos_std=0.4,
        rot_std=0.5,
        lin_vel_std=0.2,
        ang_vel_std=0.2,
        foot_pos_std=0.25,
        success_effort_multiplier=0.8,
        success_min_foot_weight_fraction=0.80,
        success_body_lin_speed_thresh=0.30,
        success_body_ang_speed_thresh=0.30,
    )
    default = base
