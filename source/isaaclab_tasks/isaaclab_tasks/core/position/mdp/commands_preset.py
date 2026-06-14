# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
    IKObjectiveGravityTorqueCfg,
    IKObjectiveJointDefaultCfg,
    IKObjectiveStabilityMarginCfg,
    IKObjectiveTerrainCollisionCfg,
)
from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command_cfg import StateCommandCfg
from isaaclab_tasks.core.multi_task.terrain.mdp.commands.commands_cfg import (
    BaseFootStatePayloadCfg,
    BaseStatePayloadCfg,
    PoseCommands,
    PositionCommands,
    TaskTableCfg,
    TerrainCommands,
    VelocityCommands,
)
from isaaclab_tasks.core.multi_task.terrain.retarget import RetargetPipelineCfg
from isaaclab_tasks.core.multi_task.terrain.retarget.cfg import (
    PatchSamplingCfg,
    SamplerSizingCfg,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.cfg import (
    SamplerCfg as RetargetSamplerCfg,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg import (
    CollisionCheckCfg,
    FootPositionErrorCfg,
    JointWithinLimitCfg,
    LateralHipLimitCfg,
    SolverCostOutlierCfg,
    SupportPolygonStabilityCfg,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.feature_extractors import XYZYawFeatures
from isaaclab_tasks.utils import PresetCfg

from ..commands_preset import CommandsPresetCfg as FlatPatchCommandsPresetCfg
from .commands import RelativeStateCommandCfg as FlatPatchRelativeStateCommandCfg


@configclass
class CommandsPresetCfg(PresetCfg):
    """Named command configurations for the position locomotion task."""

    all_commands = {
        "lin_vel_cmd": VelocityCommands(
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
        "ang_vel_cmd": VelocityCommands(
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
        "terrain_position_cmd": TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 2.0),
        ),
        "terrain_pose_cmd": TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": TerrainCommands(
            match_base_pos=False,
            match_base_rot=True,
            duration=(0.05, 4.0),
        ),
        "position_cmd": PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "pose_cmd": PoseCommands(
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
        "terrain_pose_cmd": TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 2.0),
        ),
        "terrain_position_cmd": TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 2.0),
        ),
    }
    terrain_pos = {
        "terrain_position_cmd": TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 1.0),
        ),
    }
    terrain_pose = {
        "terrain_pose_cmd": TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 1.0),
        ),
    }
    pose = {
        "pose_cmd": PoseCommands(
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
        "position_cmd": PositionCommands(
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
        "lin_vel_cmd": VelocityCommands(
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

    base = BaseStatePayloadCfg(
        pos_std=0.4,
        rot_std=0.5,
        lin_vel_std=0.2,
        ang_vel_std=0.2,
    )
    base_foot = BaseFootStatePayloadCfg(
        pos_std=0.4,
        rot_std=0.5,
        lin_vel_std=0.2,
        ang_vel_std=0.2,
        foot_pos_std=0.25,
    )
    default = base


@configclass
class FootSampledCommandsCfg:
    """Command specifications for the MDP."""

    goal_point = StateCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        randomize_command_indices=False,
        states_relative=True,
        commands=CommandsPresetCfg(),  # type: ignore
        payload=CommandPayloadPresetCfg(),  # type: ignore
        task_table=TaskTableCfg(
            pool_spacing=0.5,
            max_spawns_per_cell=20,
            num_targets_per_cell=20,
            pipeline_cfg=RetargetPipelineCfg(
                asset_cfg=SceneEntityCfg("robot"),
                kin=NewtonKinematicsCfg(usd_path=""),
                sampler=RetargetSamplerCfg(
                    patch=PatchSamplingCfg(
                        contact_radius=0.04,
                        max_height_diff=0.03,
                        horizontal_scale=0.01,
                        oversample_ratio=5.0,
                    ),
                    sizing=SamplerSizingCfg(fps_features=XYZYawFeatures(yaw_scale=0.1), criteria_yield=0.10),
                    min_contacts=3,
                    terrain_snap_distance=0.2,
                    outward_snap_penalty=1.0,
                ),
                foot_body_names=".*FOOT.*",
                lateral_hip_joint_pattern=".*HAA",
                base_pos_weight=0.05,
                base_rot_weight=0.5,
                joint_regularize_targets={".*HAA": 0.0, ".*F_KFE": -0.8, ".*H_KFE": 0.8},
                extra_objectives=[
                    IKObjectiveTerrainCollisionCfg(weight=2.0, margin=0.05, n_samples=4),
                    IKObjectiveStabilityMarginCfg(weight=1.0),
                    IKObjectiveGravityTorqueCfg(weight=0.02),
                    IKObjectiveJointDefaultCfg(weight=0.5),
                ],
                criteria=[
                    CollisionCheckCfg(n_samples=16, max_pen=0.02),
                    JointWithinLimitCfg(limit_ratio=0.9),
                    LateralHipLimitCfg(max_angle=1.05),
                    SupportPolygonStabilityCfg(),
                    FootPositionErrorCfg(max_err=0.4, aggregate="sum"),
                    SolverCostOutlierCfg(threshold_multiplier=3.0),
                ],
                ik_iterations=200,
                ik_convergence_threshold=0.01,
            ),
        ),
    )


@configclass
class FlatPatchCommandsCfg:
    """Old flat-patch command stack for the position locomotion task."""

    goal_point = FlatPatchRelativeStateCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        pos_std=0.5,
        rot_std=0.5,
        lin_vel_std=0.3,
        ang_vel_std=0.3,
        foot_body_names=[".*FOOT.*"],
        debug_vis=True,
        commands=FlatPatchCommandsPresetCfg(),  # type: ignore
    )


@configclass
class CommandsCfg(PresetCfg):
    """Command stack presets for the position locomotion task."""

    foot_sampled_commands: FootSampledCommandsCfg = FootSampledCommandsCfg()
    flat_patch_commands: FlatPatchCommandsCfg = FlatPatchCommandsCfg()
    default = foot_sampled_commands
