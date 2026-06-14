# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import BLUE_ARROW_X_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from .state_command import RelativeStateCommand


@configclass
class RelativeStateCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = RelativeStateCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    pos_std: float = 0.5

    rot_std: float = 0.5

    lin_vel_std: float = 0.5

    ang_vel_std: float = 0.5

    foot_body_names: list[str] = MISSING
    """Regex(es) selecting support-foot bodies. Used for ``N_support_feet`` and
    the characteristic limb length ``L_ref`` (base-to-mean-foot Z)."""

    success_effort_multiplier: float = 0.8
    """Specific-effort threshold scaled by ``1 / N_support_feet``.

    Success gate: ``max_j |τ_react,axis_j| / (m·g·L_ref) < multiplier / N_support_feet``.
    For a quadruped, ``0.6`` → per-foot threshold ``0.15``.
    """

    joint_wrench_sensor_name: str = "joint_wrench"
    """Scene name of the :class:`~isaaclab.sensors.JointWrenchSensor` (wrench is
    read instead of ``applied_torque`` so joint-stop reactions are counted)."""

    contact_sensor_name: str = "contact_forces"
    """Scene name of the :class:`~isaaclab.sensors.ContactSensor` used for the
    feet-bear-weight gate."""

    success_min_foot_weight_fraction: float = 0.80
    """Minimum fraction of ``m·g`` borne by the feet.

    Success gate: ``sum_f max(0, F_z[f]) / (m·g) >= multiplier``. Rejects poses
    where thighs/shanks rest on the terrain — those reactions go through hip
    constraints off the joint axis and are invisible to the effort gate.
    """

    success_body_lin_speed_thresh: float = 0.30
    """Per-body linear-speed ceiling [m/s] for the "bodies settled" gate."""

    success_body_ang_speed_thresh: float = 0.30
    """Per-body angular-speed ceiling [rad/s] for the "bodies settled" gate."""

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
    class TerrainCommands(Commands):
        """Uniform distribution ranges for the position commands."""

        target_key: str = "target"

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

    commands: dict[str, Commands] = {}
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
        },
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
