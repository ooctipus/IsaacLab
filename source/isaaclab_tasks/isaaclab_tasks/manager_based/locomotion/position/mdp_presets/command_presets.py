# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command presets selectable via ``env.commands.goal_point.commands=<name>``."""

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from .. import mdp
from ..mdp.retarget import RetargetPipelineCfg
from ..mdp.retarget.cfg import PatchSamplingCfg, SamplerCfg, SamplerSizingCfg
from ..utils.criteria_cfg import (
    CollisionCheckCfg,
    FootPositionErrorCfg,
    HaaLimitCfg,
    SolverCostOutlierCfg,
    SupportPolygonStabilityCfg,
)
from ..utils.kinematic import NewtonKinematicsCfg
from ..utils.kinematic.ik_objectives.cfg import (
    IKObjectiveGravityTorqueCfg,
    IKObjectiveJointRegularizeCfg,
    IKObjectiveStabilityMarginCfg,
    IKObjectiveTerrainCollisionCfg,
)
from .robots.robot_presets import (
    RetargetFootBodyNamesCfg,
    RetargetHaaJointPatternCfg,
    RetargetJointRegularizeTargetsCfg,
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
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "terrain_pose_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            pos_x=None,
            pos_y=None,
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 4.0),
        ),
        "position_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "pose_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
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
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
        "terrain_position_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            pos_x=None, pos_y=None, pos_z=None, roll=(-0.0, 0.0), pitch=(-0.0, 0.0), yaw=None, duration=(0.05, 4.0)
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
    }
    terrain_pos = {
        "terrain_position_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
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
        "terrain_pose_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
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
        "terrain_stand_up_cmd": mdp.RelativeStateCommandCfg.TerrainCommands(
            pos_x=None, pos_y=None, pos_z=None, roll=(-0.0, 0.0), pitch=(-0.0, 0.0), yaw=None, duration=(0.05, 4.0)
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
        pos_std=0.5,
        rot_std=0.5,
        lin_vel_std=0.3,
        ang_vel_std=0.3,
        debug_vis=True,
        commands=CommandsPresetCfg(),  # type: ignore
        # Retarget pipeline: terrain-conforming IK spawn pool feeding the
        # per-cell task table. The kinematics section below is populated
        # from ``env.scene.robot`` at command-term init (the ArticulationCfg
        # is the single source of truth for USD / base height / default
        # stance); everything else is set explicitly here so the pipeline
        # configuration is visible at the call site rather than hidden
        # behind an opaque preset object.
        pipeline_cfg=RetargetPipelineCfg(
            kin=NewtonKinematicsCfg(),
            sampler=SamplerCfg(
                patch=PatchSamplingCfg(
                    contact_radius=0.04,
                    max_height_diff=0.03,
                    horizontal_scale=0.03,
                    oversample_ratio=5.0,
                    min_center_dist=0.05,
                ),
                sizing=SamplerSizingCfg(
                    final_fps_oversample=3.0,
                    criteria_yield=0.25,
                    polygon_fps_oversample=5.0,
                    polygon_assembly_yield=0.8,
                    morph_patch_oversample=4.0,
                ),
            ),
            foot_body_names=RetargetFootBodyNamesCfg(),  # type: ignore[arg-type]
            haa_joint_pattern=RetargetHaaJointPatternCfg(),  # type: ignore[arg-type]
            # Per-robot joint-regularize targets (HAA -> 0, knees -> init pose,
            # HFE left free). Resolved from the active robot preset and
            # consumed by :class:`IKObjectiveJointRegularizeCfg` below.
            joint_regularize_targets=RetargetJointRegularizeTargetsCfg(),  # type: ignore[arg-type]
            # Extra IK objectives (appended to the standard foot-contact /
            # base-pose / joint-limit set).
            extra_objectives=[
                IKObjectiveTerrainCollisionCfg(weight=3.0, margin=0.05, n_samples=4),
                IKObjectiveStabilityMarginCfg(weight=1.0),
                # Small gravity-torque penalty pulls unconstrained DOFs toward
                # natural hanging postures (e.g. raised legs on nc<4 stances).
                IKObjectiveGravityTorqueCfg(weight=0.02),
                # Joint-regularize targets come from the pipeline cfg's
                # ``joint_regularize_targets`` (robot preset); leaving
                # ``joint_targets`` empty here falls back to that.
                IKObjectiveJointRegularizeCfg(weight=3.0),
            ],
            # Acceptance criteria applied in order: hard physical constraints
            # first so their rejection buckets report true physical invalidity,
            # residual IK-quality checks last on the physically valid subset.
            # HaaLimitCfg reads its regex from ``haa_joint_pattern`` above.
            criteria=[
                CollisionCheckCfg(n_samples=16, max_pen=0.02),
                HaaLimitCfg(max_angle=1.05),
                SupportPolygonStabilityCfg(),
                FootPositionErrorCfg(max_err=0.1, aggregate="sum"),
                SolverCostOutlierCfg(threshold_multiplier=3.0),
            ],
            ik_iterations=200,
            ik_convergence_threshold=0.01,
        ),
    )
