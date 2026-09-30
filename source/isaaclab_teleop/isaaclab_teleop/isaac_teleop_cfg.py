# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration class for IsaacTeleop-based teleoperation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from .control_events import TELEOP_CONTROL_CHANNEL_UUID

_CLOUDXR_ENV_DIR = Path(__file__).resolve().parent

CLOUDXR_AVP_ENV: str = str(_CLOUDXR_ENV_DIR / "avp-cloudxr.env")
"""Absolute path to the Apple Vision Pro CloudXR ``.env`` profile (``auto-native``)."""

CLOUDXR_JS_ENV: str = str(_CLOUDXR_ENV_DIR / "cloudxrjs-cloudxr.env")
"""Absolute path to the CloudXR JS (Quest/Pico) ``.env`` profile (``auto-webrtc``)."""

CLOUDXR_STANDALONE_ENV: str = str(_CLOUDXR_ENV_DIR / "cloudxr-standalone.env")
"""Absolute path to the standalone (headless, no XR client) CloudXR ``.env`` profile.

Default profile for teleop scripts run without ``--xr``, where IsaacTeleop is a
pure input/output transport and creates its own OpenXR session. It forces a
``quest3`` device profile so the CloudXR runtime advertises an OpenXR system with
no client connected, working around ``XR_ERROR_FORM_FACTOR_UNAVAILABLE`` (``-35``).
"""

HAND_JOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HandJointMarkers",
    markers={
        "joint": sim_utils.SphereCfg(
            radius=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)
CONTROLLER_AIM_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ControllerAimMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        )
    },
)

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner
    from isaacteleop.teleop_session_manager import PluginConfig, RetargetingExecutionConfig


@configclass
class XrCameraFeedCfg:
    """Configuration for one camera image panel shown in XR."""

    camera_name: str = MISSING
    """Name of the :class:`~isaaclab.sensors.Camera` in the interactive scene."""

    enabled: bool = True
    """Whether to create and update this feed."""

    panel_width_m: float = 0.48
    """Physical panel width [m]."""

    distance_m: float = 0.8
    """Distance in front of the viewer anchor [m].

    This value is unused when :attr:`XrCameraFeedLayoutCfg.placement` is
    ``"world"``.
    """

    offset_m: tuple[float, float] = (0.0, 0.0)
    """Horizontal and vertical panel offset in the selected placement frame [m]."""

    max_update_hz: float = 30.0
    """Maximum provider upload rate [Hz]. Set to zero to update after every rendered frame."""

    label: str | None = None
    """Optional short label shown above the image."""


@configclass
class XrCameraFeedLayoutCfg:
    """Declarative placement and packing for enabled XR camera feeds."""

    mode: Literal["manual", "horizontal", "vertical", "grid"] = "manual"
    """Layout mode. Manual preserves each feed's offset and distance."""

    placement: Literal["viewer_start", "head_locked", "world"] = "viewer_start"
    """Reference frame used to place the panels.

    ``"viewer_start"`` captures the first valid viewer eye position and yaw,
    then leaves the panels fixed in the world. ``"head_locked"`` follows the
    viewer with full pose. ``"world"`` uses :attr:`world_position_m` and
    :attr:`world_orientation_xyzw` as a fixed pose in the Isaac Lab USD stage
    world.
    """

    center_offset_m: tuple[float, float] = (0.0, 0.0)
    """Horizontal and vertical center of an automatic layout [m]."""

    distance_m: float = 0.8
    """Distance of every automatically placed panel from the viewer anchor [m].

    This value is unused when :attr:`placement` is ``"world"``.
    """

    panel_gap_m: float = 0.04
    """Edge-to-edge gap between automatically placed panels [m]."""

    max_columns: int = 2
    """Maximum number of columns in grid mode."""

    world_position_m: tuple[float, float, float] | None = None
    """Layout-plane center in the Isaac Lab USD stage world [m].

    Isaac Lab stages are Z-up. This value is required when :attr:`placement`
    is ``"world"``.
    """

    world_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Panel-local-to-world orientation as a quaternion in ``xyzw`` order.

    Panel local +X is image right, +Y is image up, and +Z points from the
    panel's readable side toward the viewer. Feed and layout offsets are
    applied in the resulting local XY plane.
    """


@configclass
class TeleopPipelineCfg:
    """Configuration selecting one IsaacTeleop retargeting pipeline implementation."""

    class_type: Callable[[TeleopPipelineCfg], OutputCombiner] = MISSING
    """Pipeline implementation constructed as ``class_type(cfg)``."""


@configclass
class IsaacTeleopCfg:
    """Configuration for IsaacTeleop-based teleoperation.

    This configuration class defines the parameters needed to create a IsaacTeleop
    teleoperation session integrated with Isaac Lab environments.

    The selected pipeline implementation returns an ``OutputCombiner`` with one
    ``"action"`` output containing the flattened action tensor.

    Example:
        .. code-block:: python

            def build_pipeline(cfg):
                controllers = ControllersSource(name="controllers")
                se3 = Se3AbsRetargeter(cfg, name="ee_pose")
                # ... connect and flatten with TensorReorderer ...
                return OutputCombiner({"action": reorderer.output("output")})


            teleop_cfg = IsaacTeleopCfg(pipeline_cfg=TeleopPipelineCfg(class_type=build_pipeline))
    """

    xr_camera_feeds: list[XrCameraFeedCfg] = field(default_factory=list)
    """Existing task camera outputs to show as XR image panels.

    The default empty list disables PiP.
    """

    xr_camera_feed_layout: XrCameraFeedLayoutCfg = field(default_factory=XrCameraFeedLayoutCfg)
    """Placement and packing applied to the ordered enabled camera feeds."""

    pipeline_cfg: TeleopPipelineCfg = MISSING
    """Retargeting pipeline implementation configuration.

    Its ``class_type`` must return an ``OutputCombiner`` with an ``"action"`` output
    containing the flattened action tensor matching the Isaac Lab action space.
    Use TensorReorderer to flatten multiple retargeter outputs into a single array.
    """

    plugins: list[PluginConfig] = field(default_factory=list)
    """List of IsaacTeleop plugin configurations.

    Plugins can provide additional functionality like synthetic hand tracking
    from controller inputs.
    """

    sim_device: str = "cuda:0"
    """Torch device string for placing output action tensors."""

    hand_joint_visualizer_cfg: VisualizationMarkersCfg = HAND_JOINT_MARKER_CFG
    """Marker configuration for tracked hand joints."""

    controller_aim_visualizer_cfg: VisualizationMarkersCfg = CONTROLLER_AIM_MARKER_CFG
    """Marker configuration for controller aim poses."""

    retargeting_execution: RetargetingExecutionConfig | None = None
    """IsaacTeleop retargeting execution settings.

    Left as ``None`` by default so that importing and constructing this config
    never requires the optional ``isaacteleop`` package (e.g. on platforms where
    it is not installed). When ``None``, Isaac Lab resolves it at session start to
    IsaacTeleop's pipelined, deadline-paced default
    (``RetargetingExecutionConfig(mode="pipelined", pacing=DeadlinePacingConfig(safety_margin_s=0.025))``),
    where ``isaacteleop`` is guaranteed to be available. Set this explicitly to
    ``RetargetingExecutionConfig(mode="sync")`` for exact current-frame
    retargeting while debugging or comparing behavior.
    """

    teleoperation_active_default: bool = False
    """Whether teleoperation should be active by default when the session starts.

    When ``False`` (the default), the teleop session remains inactive until a
    ``"START"`` command is received from xr_core via the message bus.
    """

    control_channel_uuid: bytes | None = TELEOP_CONTROL_CHANNEL_UUID
    """16-byte UUID for the teleop control message channel.

    Defaults to :data:`~isaaclab_teleop.TELEOP_CONTROL_CHANNEL_UUID`
    (``uuid5(NAMESPACE_DNS, "teleop_command")``), which is the well-known
    channel both the Isaac Lab server and CloudXR JS client use to
    exchange start/stop/reset commands.

    When set, a ``teleop_control_pipeline`` is created automatically
    using :class:`~isaaclab_teleop.teleop_message_processor.TeleopMessageProcessor`
    and :class:`~isaacteleop.teleop_session_manager.DefaultTeleopStateManager`.
    The remote client sends UTF-8 control commands over the OpenXR opaque
    data channel identified by this UUID, and the results are exposed via
    :func:`~isaaclab_teleop.poll_control_events`.

    Set to ``None`` to disable the control channel entirely.
    """

    target_frame_prim_path: str | None = None
    """Optional USD prim path whose world frame becomes the target coordinate
    frame for all output poses.

    When set, the device automatically reads this prim's world transform each
    frame and uses its inverse as the ``target_T_world`` rebase matrix in
    :meth:`~isaaclab_teleop.IsaacTeleopDevice.advance`.  An explicit
    ``target_T_world`` argument to :meth:`~isaaclab_teleop.IsaacTeleopDevice.advance`
    takes precedence over this config.

    Typical usage: set to the robot base link prim path so that an IK
    controller receives end-effector poses in the robot's base frame.

    Example::

        IsaacTeleopCfg(
            target_frame_prim_path="/World/envs/env_0/Robot/base_link",
            ...
        )
    """

    app_name: str = "IsaacLabTeleop"
    """Application name for the IsaacTeleop session."""
