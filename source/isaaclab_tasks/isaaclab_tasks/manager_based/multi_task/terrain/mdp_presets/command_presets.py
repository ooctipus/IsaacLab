# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command presets selectable via ``env.commands.goal_point.commands=<name>``."""

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from ...mdp.util.kinematics import NewtonKinematicsCfg
from ...mdp.util.kinematics.ik_objectives.cfg import (
    IKObjectiveGravityTorqueCfg,
    IKObjectiveStabilityMarginCfg,
    IKObjectiveTerrainCollisionCfg,
)
from .. import mdp
from ..mdp.retarget import RetargetPipelineCfg
from ..mdp.retarget.cfg import PatchSamplingCfg, SamplerCfg
from ..utils.criteria_cfg import (
    CollisionCheckCfg,
    FootPositionErrorCfg,
    LateralHipLimitCfg,
    SolverCostOutlierCfg,
    SupportPolygonStabilityCfg,
)
from .robots.robot_presets import (
    FootBodyNamesCfg,
    RetargetJointRegularizeTargetsCfg,
    RetargetLateralHipJointPatternCfg,
)


@configclass
class CommandsPresetCfg(PresetCfg):
    """Named command configurations for the position locomotion task."""

    all_commands = {
        "lin_vel_cmd": mdp.RelativeStateCommandCfg.VelocityCommands(
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
        "ang_vel_cmd": mdp.RelativeStateCommandCfg.VelocityCommands(
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
        "terrain_position_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            match_feet=True,
            duration=(0.05, 2.0),
        ),
        "terrain_pose_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            match_feet=True,
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=False,
            match_base_rot=True,
            match_feet=True,
            duration=(0.05, 4.0),
        ),
        "position_cmd": mdp.RelativeStateCommandCfg.PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "pose_cmd": mdp.RelativeStateCommandCfg.PoseCommands(
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
        "terrain_pose_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            match_feet=True,
            duration=(0.05, 2.0),
        ),
        "terrain_position_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            match_feet=True,
            duration=(0.05, 2.0),
        ),
    }
    terrain_pos = {
        "terrain_position_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            match_feet=True,
            duration=(0.05, 0.05),
        ),
    }
    terrain_pose = {
        "terrain_pose_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            match_feet=True,
            duration=(0.05, 0.05),
        ),
    }
    pose = {
        "pose_cmd": mdp.RelativeStateCommandCfg.PoseCommands(
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
        "position_cmd": mdp.RelativeStateCommandCfg.PositionCommands(
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
            "lin_vel_cmd": mdp.RelativeStateCommandCfg.VelocityCommands(
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
    advanced_skills = pos


@configclass
class CommandsCfg:
    "Command specifications for the MDP."

    goal_point = mdp.RelativeStateCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        pos_std=0.7,
        rot_std=0.7,
        lin_vel_std=0.3,
        ang_vel_std=0.3,
        foot_pos_std=0.5,
        debug_vis=True,
        pool_spacing=0.2,
        pool_sampling_size=(10.0, 10.0),
        commands=CommandsPresetCfg(),  # type: ignore
        pipeline_cfg=RetargetPipelineCfg(
            kin=NewtonKinematicsCfg(usd_path=""),
            sampler=SamplerCfg(
                patch=PatchSamplingCfg(
                    contact_radius=0.04,
                    max_height_diff=0.03,
                    horizontal_scale=0.03,
                    oversample_ratio=5.0,
                ),
                min_contacts=3,
                terrain_snap_distance=0.2,
                outward_snap_penalty=1.0,
            ),
            foot_body_names=FootBodyNamesCfg(),  # type: ignore[arg-type]
            lateral_hip_joint_pattern=RetargetLateralHipJointPatternCfg(),  # type: ignore[arg-type]
            base_pos_weight=0.05,
            base_rot_weight=0.5,
            joint_regularize_targets=RetargetJointRegularizeTargetsCfg(),  # type: ignore[arg-type]
            extra_objectives=[
                IKObjectiveTerrainCollisionCfg(weight=2.0, margin=0.05, n_samples=4),
                IKObjectiveStabilityMarginCfg(weight=1.0),
                IKObjectiveGravityTorqueCfg(weight=0.02),
            ],
            criteria=[
                CollisionCheckCfg(n_samples=16, max_pen=0.02),
                LateralHipLimitCfg(max_angle=1.05),
                SupportPolygonStabilityCfg(),
                FootPositionErrorCfg(max_err=0.4, aggregate="sum"),
                SolverCostOutlierCfg(threshold_multiplier=3.0),
            ],
            ik_iterations=200,
            ik_convergence_threshold=0.01,
        ),
    )
