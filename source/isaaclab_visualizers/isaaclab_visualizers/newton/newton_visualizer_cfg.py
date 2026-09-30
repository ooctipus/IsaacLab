# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Newton GL and RTX visualizer backends."""

from __future__ import annotations

import typing

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if typing.TYPE_CHECKING:
    from isaaclab.visualizers import BaseVisualizer

    from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Shared configuration base for Newton visualizer backends."""

    class_type: type[BaseVisualizer] | str | None = None
    """Visualizer implementation class. Concrete configs must set this field."""

    visualizer_type: str | None = None
    """Visualizer type identifier. Concrete configs must set this field."""

    window_width: int = 1920
    """Window width in pixels."""

    window_height: int = 1080
    """Window height in pixels."""

    headless: bool = False
    """Run the Newton viewer without requiring a display server."""

    update_frequency: int = 1
    """Visualizer update frequency (renders every N simulation frames)."""

    world_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Visual spacing between simulation worlds along each axis [m].

    Non-zero axes arrange visible worlds in a compact grid without changing their simulated poses.
    """

    show_joints: bool = False
    """Show joint visualization."""

    show_contacts: bool = False
    """Show contact visualization."""

    show_collision: bool = False
    """Show collision visualization."""

    show_springs: bool = False
    """Show spring visualization."""

    show_inertia_boxes: bool = False
    """Show inertia box visualization."""

    show_com: bool = False
    """Show center of mass visualization."""

    show_particles: bool = False
    """Show particle visualization."""

    particle_color: tuple[float, float, float] | None = None
    """Optional particle color RGB [0, 1]. Uses Newton viewer defaults when ``None``."""

    enable_picking: bool = True
    """Enable right-click dragging with Newton rigid-body solvers.

    Supported coupled solvers may expose dragging through a rigid-body entry.
    Disabled automatically for headless viewers, standalone MPM, and non-Newton
    physics. MPM particles are not pickable.
    """

    enable_shadows: bool = True
    """Enable shadow rendering."""

    enable_sky: bool = True
    """Enable sky rendering."""

    enable_wireframe: bool = False
    """Enable wireframe rendering."""

    sky_upper_color: tuple[float, float, float] = (0.2, 0.4, 0.6)
    """Sky upper color RGB [0, 1]."""

    sky_lower_color: tuple[float, float, float] = (0.5, 0.6, 0.7)
    """Sky lower color RGB [0, 1]."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color RGB [0, 1]."""


@configclass
class NewtonGLVisualizerCfg(NewtonVisualizerCfg):
    """Configuration for the Newton OpenGL rasterizer visualizer.

    Selects Newton's OpenGL backend — fast local window with the full Isaac Lab
    feature set: streaming camera panel, particle color override, and live scalar/array plots.

    The streaming camera panel is opt-in through ``streaming_view=True``.
    """

    class_type: type[NewtonGLVisualizer] | str = "{DIR}.newton_visualizer:NewtonGLVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "newton_gl"
    """Visualizer type identifier. Do not change."""


@configclass
class NewtonRTXVisualizerCfg(VisualizerCfg):
    """Present an explicitly planned camera stream in the Newton GL window.

    The camera's ``renderer_cfg`` selects the renderer. This visualizer only displays and captures
    its :class:`~isaaclab.sensors.camera.CameraData`, so it does not create an OVRTX runtime or USD
    stage. Set :attr:`streaming_camera` to a camera declared by the clone plan.
    """

    class_type: type[NewtonRTXVisualizer] | str = "{DIR}.newton_visualizer:NewtonRTXVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "newton_rtx"
    """Visualizer type identifier. Do not change."""

    window_width: int = 1920
    """Image-sink window width in pixels."""

    window_height: int = 1080
    """Image-sink window height in pixels."""

    headless: bool = False
    """Create the image sink without a display server."""

    update_frequency: int = 1
    """Camera presentation frequency in simulation frames."""

    streaming_view: bool = True
    """Always enabled; this visualizer requires a planned camera stream."""
