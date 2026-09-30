# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Literal

from isaaclab.renderers import RendererCfg
from isaaclab.sim import FisheyeCameraCfg, PinholeCameraCfg
from isaaclab.utils.configclass import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .camera import Camera


@configclass
class CameraCfg(SensorBaseCfg):
    """Configuration for a camera sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera)
          convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    class_type: type[Camera] | str = "{DIR}.camera:Camera"

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.

    .. note::
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """

    spawn: PinholeCameraCfg | FisheyeCameraCfg | None = MISSING
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """

    data_types: list[str] = ["rgb"]
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"].

    Please refer to the :class:`Camera` class for a list of available data types.
    """

    width: int = MISSING
    """Width of the image in pixels."""

    height: int = MISSING
    """Height of the image in pixels."""

    update_latest_camera_pose: bool = False
    """Whether to update the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is updated in the camera's data which will slow down performance
    due to the use of :class:`FrameView`.
    If False, the pose of the camera during initialization is returned.
    """

    background_color: tuple[float, float, float] | None = None
    """Background color for the camera as normalized RGB floats ``(red, green, blue)`` in ``[0, 1]``.

    When set, pixels that miss all geometry are filled with this solid color.
    When ``None`` (the default), each backend uses its own default background.
    """

    renderer_cfg: RendererCfg = MISSING
    """Concrete renderer configuration for the camera sensor."""

    isp_cfg: Any | None = None
    """Concrete post-render ISP configuration, or ``None`` to disable ISP.

    The cfg applies once per Camera sensor batch. The PPISP Warp kernel takes
    scalar coefficients, so every cloned view in a tiled batch shares the same
    ISP configuration — there is no per-view ISP today.

    :mod:`isaaclab.sensors.camera` does not depend on any ISP implementation; the
    annotation is intentionally loose (``Any``) so the sensor layer can carry the
    cfg through to a renderer that knows what to do with it.
    """
