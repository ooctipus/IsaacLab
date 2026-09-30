# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kit-based visualizer."""

from __future__ import annotations

import typing

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if typing.TYPE_CHECKING:
    from .kit_visualizer import KitVisualizer


@configclass
class KitVisualizerCfg(VisualizerCfg):
    """Configuration for Kit visualizer using Isaac Sim viewport.

    .. note::
        The streaming camera panel (``streaming_view=True``) requires the
        ``--enable_cameras`` CLI flag. Initialization fails without camera rendering.
        Set ``dock_position="RIGHT"`` so
        the panel appears side-by-side with the Viewport instead of as a hidden tab.
    """

    class_type: type[KitVisualizer] | str = "{DIR}.kit_visualizer:KitVisualizer"
    """Visualizer implementation class."""

    prim_path: str = "/World/KitViewerCamera"
    """Global clone-plan path of the camera controlled by the Kit viewport."""

    visualizer_type: str = "kit"
    """Type identifier for Kit visualizer."""

    viewport_name: str | None = None
    """Name for a new viewport window when :attr:`create_viewport` is ``True``.

    If ``None``, a default name (``"Visualizer Viewport"``) is used.
    """

    create_viewport: bool = False
    """If ``True``, create a new viewport window; if ``False``, use the active viewport window."""

    headless: bool = False
    """Run without creating viewport windows when supported by the app."""

    dock_position: str = "SAME"
    """Dock position for the streaming image panel and any new viewport window.

    Options: ``'LEFT'``, ``'RIGHT'``, ``'BOTTOM'``, ``'SAME'``.

    .. note::
        ``'SAME'`` (the default) places the streaming panel as a hidden tab in the
        same dock group as the main Viewport — you must click the panel's tab to see it.
        Use ``'RIGHT'`` to keep both the Viewport and the streaming panel visible
        side-by-side.
    """

    window_width: int = 1280
    """Viewport width in pixels (when :attr:`create_viewport` is ``True``)."""

    window_height: int = 720
    """Viewport height in pixels (when :attr:`create_viewport` is ``True``)."""

    origin_type: str = "world"
    """Frame in which :attr:`~isaaclab.visualizers.VisualizerCfg.eye` and
    :attr:`~isaaclab.visualizers.VisualizerCfg.lookat` are interpreted.

    Options:

    * ``"world"``: global origin.
    * ``"env"``: origin of the environment at :attr:`origin_env_index`.
    * ``"asset"``: a scene asset (or body) specified by :attr:`origin_track_path`.
    """

    origin_env_index: int = 0
    """Index of the environment used as the viewport camera origin.

    Only meaningful when :attr:`origin_type` is ``"env"`` or ``"asset"``.
    """

    origin_track_path: str | None = None
    """Planned rigid-body prim path tracked by the viewport camera.

    The path must be covered by the clone plan. It may name an exact published rigid body or an
    asset root, in which case its first published rigid body is tracked. It may contain
    ``{ENV_REGEX_NS}``; :attr:`origin_env_index` selects the exact clone. Required when
    :attr:`origin_type` is ``"asset"``.

    Examples::

        origin_track_path = "{ENV_REGEX_NS}/Robot/panda_hand"
    """
