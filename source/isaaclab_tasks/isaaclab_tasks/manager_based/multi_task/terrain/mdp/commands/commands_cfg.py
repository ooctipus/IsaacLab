# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import BLUE_ARROW_X_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .state_command import RelativeStateCommand

if TYPE_CHECKING:
    from ...mdp.retarget.cfg import RetargetPipelineCfg


@configclass
class RelativeStateCommandCfg(CommandTermCfg):
    """Configuration for the relative state command generator."""

    class_type: type = RelativeStateCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    pipeline_cfg: RetargetPipelineCfg = MISSING  # type: ignore[assignment]
    """Retarget pipeline configuration for generating IK-solved spawn states."""

    pool_spacing: float = 1.0
    """Target spacing between final IK-solved terrain states [m].

    The command derives pool size from sampling area, so sample density scales
    with the full terrain grid or :attr:`pool_sampling_size`.
    """

    pool_spacing_area_divisor: float = 3.0
    """Area divisor used to derive spacing-mode pool size [unitless]."""

    pool_sampling_size: tuple[float, float] | None = None
    """Optional centered terrain-state sampling window size ``(x, y)`` [m].

    ``None`` samples over the full terrain grid. When set, the command still
    bins states against the full terrain grid, but the IK retarget sampler is
    clipped to this centered window before pool sizing and sampling.
    """

    exclude_self_pairs: bool = True
    """Whether to drop ``(spawn == target)`` pairs from the per-cell Cartesian product.

    When ``True``, the per-cell ``n × n`` pair grid is reduced to ``n × (n - 1)``
    by removing the diagonal — i.e. trivial zero-distance tasks where the
    spawn state is also the target. Cells with fewer than two valid states
    then contribute no pairs.
    """

    pos_std: float = 0.5
    """Default base-position success threshold [m]. Per-task overrideable via
    :attr:`Commands.pos_std` / :attr:`TerrainCommands.pos_std`."""

    rot_std: float = 0.5
    """Default base-orientation success threshold [rad]. Per-task overrideable."""

    lin_vel_std: float = 0.5
    """Default linear-velocity success threshold [m/s]. Per-task overrideable."""

    ang_vel_std: float = 0.5
    """Default angular-velocity success threshold [rad/s]. Per-task overrideable."""

    foot_pos_std: float = 0.1
    """Default per-foot target-error threshold [m] for terrain-state success.
    Per-task overrideable via :attr:`TerrainCommands.foot_pos_std`."""

    normalize_command_obs: bool = False
    """Whether to divide :attr:`RelativeStateCommand.command` channels by the
    per-task success threshold so the policy sees a unit-scaled "distance to
    success" signal that is comparable across tasks with different stds.
    """

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
        falls back to :attr:`RelativeStateCommandCfg.pos_std`."""

        rot_std: float | None = None
        """Per-task override for the base-orientation success threshold [rad]."""

        lin_vel_std: float | None = None
        """Per-task override for the linear-velocity success threshold [m/s]."""

        ang_vel_std: float | None = None
        """Per-task override for the angular-velocity success threshold [rad/s]."""

        foot_pos_std: float | None = None
        """Per-task override for the per-foot target-error threshold [m]."""

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

        spawn_key: str = "spawn"
        """State-buffer key used for reset spawn states."""

        target_key: str = "target"
        """State-buffer key used for command target states."""

        match_base_pos: bool = True
        """Whether to command the target state's base position [m]."""

        match_base_rot: bool = False
        """Whether to command the target state's base orientation [rad]."""

        match_feet: bool = True
        """Whether success requires matching target foot positions [m]."""

        duration: tuple[float, float] = (1.0, 1.0)
        """Time required to be considered successful [s]."""

        pos_std: float | None = None
        """Per-task override for the base-position success threshold [m]. ``None``
        falls back to :attr:`RelativeStateCommandCfg.pos_std`."""

        rot_std: float | None = None
        """Per-task override for the base-orientation success threshold [rad]."""

        lin_vel_std: float | None = None
        """Per-task override for the linear-velocity success threshold [m/s]."""

        ang_vel_std: float | None = None
        """Per-task override for the angular-velocity success threshold [rad/s]."""

        foot_pos_std: float | None = None
        """Per-task override for the per-foot target-error threshold [m]."""

    commands: dict[str, Commands | TerrainCommands] = {}
    """Distribution ranges for the position commands."""

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
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

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
