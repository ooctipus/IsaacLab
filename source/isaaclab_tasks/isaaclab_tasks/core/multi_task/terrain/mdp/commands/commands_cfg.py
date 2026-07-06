# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion building blocks for the shared :class:`~...mdp.commands.StateCommand`.

The env wires up a :class:`~...mdp.commands.StateCommandCfg` directly; this module
only supplies the locomotion ``task_table`` cfg, the base/foot payload cfgs (which
own their debug visualizers), and the command-variant cfgs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import BLUE_ARROW_X_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from ....kinematics.ik_objectives.cfg import IKObjectiveBaseCfg
from ....kinematics.newton_kinematics_cfg import NewtonKinematicsBuildCfg
from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...retarget.cfg import SamplerBaseCfg
from ...retarget.criteria_cfg import CriterionBaseCfg
from .state_command_payloads import CommandPayloadBaseFootState, CommandPayloadBaseState
from .task_table_builder import build_relative_state_task_table

# Default debug-vis marker cfgs shared by the base/foot payloads (configclass
# deep-copies these per instance).
_GOAL_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_state",
    markers={
        "vel_arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "cuboid": sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "pose_arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
        ),
    },
)

_CURRENT_VEL_VISUALIZER_CFG = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
_CURRENT_VEL_VISUALIZER_CFG.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class PositionTerrainStanceGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Generate terrain contacts, base targets, and IK seeds."""

    class_type: Callable = "{DIR}.task_table_builder:generate_position_terrain_stance"
    sampler: SamplerBaseCfg = MISSING  # type: ignore[assignment]
    foot_body_names: list[str] | str = MISSING  # type: ignore[assignment]


@configclass
class PositionIKSolveCfg(StateCommandCfg.TaskTableCfg.SolveCfg):
    """Solve one flat tuple of declared Newton IK objectives."""

    class_type: Callable = "{DIR}.task_table_builder:solve_position_terrain_stance"
    objectives: tuple[IKObjectiveBaseCfg, ...] = ()


@configclass
class PositionFpsSelectionCfg(StateCommandCfg.TaskTableCfg.SelectionCfg):
    """Thin accepted terrain states in one declared feature space."""

    class_type: Callable = "{DIR}.task_table_builder:select_position_terrain_stance"
    features: Callable | None = None


@configclass
class PositionTerrainStanceFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """One terrain-stance construction family."""

    name: str = "terrain_stance"
    generate: tuple[PositionTerrainStanceGenerateCfg, ...] = ()
    solve: PositionIKSolveCfg | None = None
    criteria: tuple[CriterionBaseCfg, ...] = ()
    selection: PositionFpsSelectionCfg = PositionFpsSelectionCfg()


@configclass
class PositionSameCellPairingCfg:
    """Pair selected Position states only within one terrain cell."""

    exclude_self: bool = True
    max_spawns_per_cell: int = 0
    num_targets_per_cell: int = 0


@configclass
class TaskTableCfg(StateCommandCfg.TaskTableCfg):
    """Visible Position mechanics, density, family, and pairing policy."""

    class_type: Callable = build_relative_state_task_table
    """Callable that builds the task table."""

    kinematics: NewtonKinematicsBuildCfg = NewtonKinematicsBuildCfg()
    """Newton robot mechanics shared by every Position family."""

    pool_spacing: float = 1.0
    """Target spacing between final IK-solved terrain states [m].

    The table builder derives pool size from sampling area, so sample density
    scales with the full terrain grid or :attr:`pool_sampling_size`.
    """

    pool_spacing_area_divisor: float = 3.0
    """Area divisor used to derive spacing-mode pool size."""

    pool_sampling_size: tuple[float, float] | None = None
    """Optional centered terrain-state sampling window size ``(x, y)`` [m].

    ``None`` samples over the full terrain grid. When set, the table builder
    still bins states against the full terrain grid, but the IK retarget
    sampler is clipped to this centered window before pool sizing and sampling.
    """

    pairing: PositionSameCellPairingCfg = PositionSameCellPairingCfg()
    """Table-global spawn/target pairing after family selection."""


@configclass
class BaseStatePayloadCfg(StateCommandCfg.PayloadCfg):
    """Payload that commands base state only."""

    class_type: type = CommandPayloadBaseState
    """Payload worker class."""

    pos_std: float = 0.5
    """Default base-position success threshold [m]. Per-task overridable via command rows."""

    rot_std: float = 0.5
    """Default base-orientation success threshold [rad]. Per-task overridable via command rows."""

    lin_vel_std: float = 0.5
    """Default linear-velocity success threshold [m/s]. Per-task overridable via command rows."""

    ang_vel_std: float = 0.5
    """Default angular-velocity success threshold [rad/s]. Per-task overridable via command rows."""

    normalize_command_obs: bool = False
    """Whether to divide command channels by the per-task success threshold."""

    success_effort_multiplier: float = 0.8
    """Specific-effort threshold multiplier for the successful-hold naturalness gate."""

    joint_wrench_sensor_name: str = "joint_wrench"
    """Scene name of the joint-wrench sensor used by the success gate."""

    contact_sensor_name: str = "contact_forces"
    """Scene name of the contact sensor used by the feet-bear-weight gate."""

    success_min_foot_weight_fraction: float = 0.80
    """Minimum body-weight fraction that feet must support while accumulating successful hold time."""

    success_body_lin_speed_thresh: float = 0.30
    """Per-body linear-speed ceiling [m/s] while accumulating successful hold time."""

    success_body_ang_speed_thresh: float = 0.30
    """Per-body angular-speed ceiling [rad/s] while accumulating successful hold time."""

    goal_visualizer_cfg: VisualizationMarkersCfg = _GOAL_VISUALIZER_CFG
    """Debug marker for the goal state (pos/pose/vel)."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_VEL_VISUALIZER_CFG
    """Debug marker for the current base velocity."""


@configclass
class BaseFootStatePayloadCfg(StateCommandCfg.PayloadCfg):
    """Payload that commands base state and terrain target foot positions."""

    class_type: type = CommandPayloadBaseFootState
    """Payload worker class."""

    pos_std: float = 0.5
    """Default base-position success threshold [m]. Per-task overridable via command rows."""

    rot_std: float = 0.5
    """Default base-orientation success threshold [rad]. Per-task overridable via command rows."""

    lin_vel_std: float = 0.5
    """Default linear-velocity success threshold [m/s]. Per-task overridable via command rows."""

    ang_vel_std: float = 0.5
    """Default angular-velocity success threshold [rad/s]. Per-task overridable via command rows."""

    foot_pos_std: float = 0.1
    """Default per-foot target-error threshold [m]."""

    normalize_command_obs: bool = False
    """Whether to divide command channels by the per-task success threshold."""

    success_effort_multiplier: float = 0.8
    """Specific-effort threshold multiplier for the successful-hold naturalness gate."""

    joint_wrench_sensor_name: str = "joint_wrench"
    """Scene name of the joint-wrench sensor used by the success gate."""

    contact_sensor_name: str = "contact_forces"
    """Scene name of the contact sensor used by the feet-bear-weight gate."""

    success_min_foot_weight_fraction: float = 0.80
    """Minimum body-weight fraction that feet must support while accumulating successful hold time."""

    success_body_lin_speed_thresh: float = 0.30
    """Per-body linear-speed ceiling [m/s] while accumulating successful hold time."""

    success_body_ang_speed_thresh: float = 0.30
    """Per-body angular-speed ceiling [rad/s] while accumulating successful hold time."""

    goal_visualizer_cfg: VisualizationMarkersCfg = _GOAL_VISUALIZER_CFG
    """Debug marker for the goal state (pos/pose/vel/foot)."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_VEL_VISUALIZER_CFG
    """Debug marker for the current base velocity."""


@configclass
class Commands:
    pos_x: tuple[float, float] | None = None
    """Range for the x position (in m)."""

    pos_y: tuple[float, float] | None = None
    """Range for the y position (in m)."""

    pos_z: tuple[float, float] | None = None
    """Range for the y position (in m)."""

    roll: tuple[float, float] | None = None
    """Range for the base roll orientation (in radian)."""

    pitch: tuple[float, float] | None = None
    """Range for the base pitch orientation (in radian)."""

    yaw: tuple[float, float] | None = None
    """Range for the base yaw orientation (in radian)."""

    lin_vel_x: tuple[float, float] | None = None
    """Range for the linear-x velocity command (in m/s)."""

    lin_vel_y: tuple[float, float] | None = None
    """Range for the linear-y velocity command (in m/s)."""

    lin_vel_z: tuple[float, float] | None = None
    """Range for the linear-z velocity command (in m/s)."""

    ang_vel_x: tuple[float, float] | None = None
    """Range for the angular-x velocity command (in rad/s)."""

    ang_vel_y: tuple[float, float] | None = None
    """Range for the angular-y velocity command (in rad/s)."""

    ang_vel_z: tuple[float, float] | None = None
    """Range for the angular-z velocity command (in rad/s)."""

    duration: tuple[float, float] = (1.0, 1.0)
    """time required to be considered as success."""

    pos_std: float | None = None
    """Per-task override for the base-position success threshold [m]. ``None``
    falls back to the active payload's default."""

    rot_std: float | None = None
    """Per-task override for the base-orientation success threshold [rad]."""

    lin_vel_std: float | None = None
    """Per-task override for the linear-velocity success threshold [m/s]."""

    ang_vel_std: float | None = None
    """Per-task override for the angular-velocity success threshold [rad/s]."""


@configclass
class PositionCommands(Commands):
    """Uniform distribution ranges for the position commands."""

    pos_x: tuple[float, float] | None = MISSING
    """Range for the x position (in m)."""

    pos_y: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""

    pos_z: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""


@configclass
class PoseCommands(Commands):
    pos_x: tuple[float, float] | None = MISSING
    """Range for the x position (in m)."""

    pos_y: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""

    pos_z: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""

    roll: tuple[float, float] | None = MISSING
    """Range for the base roll orientation (in radian)."""

    pitch: tuple[float, float] | None = MISSING
    """Range for the base pitch orientation (in radian)."""

    yaw: tuple[float, float] | None = MISSING
    """Range for the base yaw orientation (in radian)."""


@configclass
class VelocityCommands(Commands):
    lin_vel_x: tuple[float, float] | None = MISSING
    """Range for the linear-x velocity command (in m/s)."""

    lin_vel_y: tuple[float, float] | None = MISSING
    """Range for the linear-y velocity command (in m/s)."""

    lin_vel_z: tuple[float, float] | None = MISSING
    """Range for the linear-z velocity command (in m/s)."""

    ang_vel_x: tuple[float, float] | None = MISSING
    """Range for the angular-x velocity command (in rad/s)."""

    ang_vel_y: tuple[float, float] | None = MISSING
    """Range for the angular-y velocity command (in rad/s)."""

    ang_vel_z: tuple[float, float] | None = MISSING
    """Range for the angular-z velocity command (in rad/s)."""


@configclass
class TerrainCommands:
    """Terrain-state command backed by sampled spawn/target states."""

    target_key: str = "target"
    """State-buffer key used for command target states."""

    match_base_pos: bool = True
    """Whether to command the target state's base position [m]."""

    match_base_rot: bool = False
    """Whether to command the target state's base orientation [rad]."""

    duration: tuple[float, float] = (1.0, 1.0)
    """Time required to be considered successful [s]."""

    pos_std: float | None = None
    """Per-task override for the base-position success threshold [m]. ``None``
    falls back to the active payload's default."""

    rot_std: float | None = None
    """Per-task override for the base-orientation success threshold [rad]."""

    lin_vel_std: float | None = None
    """Per-task override for the linear-velocity success threshold [m/s]."""

    ang_vel_std: float | None = None
    """Per-task override for the angular-velocity success threshold [rad/s]."""
