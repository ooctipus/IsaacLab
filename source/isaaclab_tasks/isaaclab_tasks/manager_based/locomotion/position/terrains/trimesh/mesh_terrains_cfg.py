# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab import terrains
from isaaclab.utils import configclass

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshObjTerrainCfg(terrains.SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = "{DIR}.mesh_terrains:obj_terrain"

    obj_path: str = MISSING

    spawn_origin_path: str = MISSING


@configclass
class TerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = "{DIR}.mesh_terrains:terrain_gen"

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    terrain_styles: list = MISSING

    yaml_path: str = (MISSING,)

    spawn_origin_path: str = MISSING

    python_script: str = MISSING


@configclass
class MeshStonesEverywhereTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with stones everywhere."""

    function = "{DIR}.mesh_terrains:stones_everywhere_terrain"

    w_gap: tuple[float, float] = MISSING
    """Stone gap width."""

    w_stone: tuple[float, float] = MISSING
    """Grid square stone size (width)."""

    s_max: tuple[float, float] = MISSING
    """Maximum shift, both x and y shift is uniformly sampled from [-s_max, s_max]."""

    h_max: tuple[float, float] = MISSING
    """Maximum height, uniformly sampled from [-h_max, h_max], default height is 1.0 m."""

    holes_depth: float = MISSING
    """Holes depth [m]."""

    platform_width: float = MISSING
    """Platform width [m]."""


@configclass
class MeshBalanceBeamsTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with balance-beams."""

    function = "{DIR}.mesh_terrains:balance_beams_terrain"

    platform_width: float = MISSING
    """Platform width [m]."""

    h_offset: tuple[float, float] = MISSING
    """Height offset [m]."""

    w_stone: tuple[float, float] = MISSING
    """Stone width [m]."""

    mid_gap: tuple[float, float] = MISSING
    """Gap between two beams [m]."""


@configclass
class MeshSteppingBeamsTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with stepping-beams."""

    function = "{DIR}.mesh_terrains:stepping_beams_terrain"

    platform_width: float = MISSING
    """Platform width [m]."""

    h_offset: tuple[float, float] = MISSING
    """Height offset [m]."""

    w_stone: tuple[float, float] = MISSING
    """Stone width [m]."""

    l_stone: tuple[float, float] = MISSING
    """Length of the stepping beams [m]."""

    gap: tuple[float, float] = MISSING
    """Gap between two beams [m]."""

    yaw: tuple[float, float] = MISSING
    """Yaw angle of the stepping beams [rad]."""


@configclass
class MeshDiversityBoxTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with boxes for anymal parkour."""

    function = "{DIR}.mesh_terrains:box_terrain"

    box_width_range: tuple[float, float] = MISSING
    """Box width range [m]."""

    box_length_range: tuple[float, float] = MISSING
    """Box length range [m]."""

    box_height_range: tuple[float, float] = MISSING
    """Box height range [m]."""

    box_gap_range: tuple[float, float] = None  # type: ignore
    """Gap between two boxes [m]."""

    up_or_down: str = None  # type: ignore
    """Flag for climbing up (box at origin) or down (box near origin)."""


@configclass
class MeshPassageTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with passage."""

    function = "{DIR}.mesh_terrains:passage_terrain"

    passage_width: float | tuple[float, float] = MISSING
    """Passage width (y direction) [m]."""

    passage_height: float | tuple[float, float] = MISSING
    """Passage height [m]."""

    passage_length: float | tuple[float, float] = MISSING
    """Passage length (x direction) [m]."""


@configclass
class MeshStructuredTerrainCfg(terrains.SubTerrainBaseCfg):
    """Configuration for a structured terrain."""

    function = "{DIR}.mesh_terrains:structured_terrain"
    terrain_type: Literal["stairs", "inverted_stairs", "obstacles", "walls"] = MISSING


@configclass
class MeshRadiatingBeamTerrainCfg(terrains.MeshStarTerrainCfg):
    """A terrain with beams radiating from a central platform to an outer border.

    The terrain consists of three layers:

    1. **Central platform** — a cylinder at the terrain center whose diameter
       is :attr:`platform_width` (inherited) and whose top surface can be
       raised above the border by :attr:`platform_height`.
    2. **Outer border** — a rectangular ring between :attr:`border_size` and
       :attr:`~isaaclab.terrains.SubTerrainBaseCfg.size`.  Its top surface
       is always at the base level (z = 0).
    3. **Beams** — bridges connecting the platform edge to the inner rim of
       the border.  The number and angular placement are controlled by
       :attr:`num_bars` and :attr:`beam_distribution`.

    Beams come in two styles selected via :attr:`beam_style`:

    - :class:`FlatBeamCfg` — each beam is a single continuous box that
      tilts when ``platform_height != 0``.
    - :class:`BoxBeamCfg` — each beam is tiled with small box segments
      whose size, gap, position jitter, and yaw jitter are individually
      configurable.  When the beam width exceeds twice the segment size,
      multiple columns are placed across the width in a staggered
      (brick-wall) pattern.

    Inherited fields from :class:`~isaaclab.terrains.MeshStarTerrainCfg`:

    - :attr:`platform_width` — diameter of the central cylinder [m].
    - :attr:`bar_width_range` — ``(easy, hard)`` beam width [m],
      interpolated with difficulty.
    - :attr:`bar_height_range` — ``(easy, hard)`` beam / border thickness
      [m], interpolated with difficulty.
    """

    # ------------------------------------------------------------------
    # Nested beam-style configs
    # ------------------------------------------------------------------

    @configclass
    class FlatBeamCfg:
        """Flat beam style — one continuous tilted box per beam."""

    @configclass
    class BoxBeamCfg:
        """Box beam style — segmented boxes tiled along (and across) each beam.

        When the beam is wide enough (``bar_width >= 2 * box_length``),
        boxes are tiled across the width too, with odd columns staggered
        by half a stride for a brick-wall pattern.
        """

        box_length: float | tuple[float, float] = 0.3
        """Side length of each square box segment [m].

        A tuple ``(easy, hard)`` is interpolated linearly with difficulty.
        """

        box_gap: float | tuple[float, float] = 0.0
        """Gap between consecutive segments along the beam axis [m].

        A tuple ``(easy, hard)`` is interpolated linearly with difficulty.
        0 means segments are placed flush.
        """

        box_position_variation: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Per-segment random position jitter ``(dx, dy, dz)`` in beam-local
        frame [m].

        Each component is sampled from ``uniform(-val, +val)``.

        - ``dx`` — along the beam axis.
        - ``dy`` — across the beam width.
        - ``dz`` — vertical.
        """

        box_yaw_variation: float = 0.0
        """Per-segment random yaw rotation [rad].

        Sampled from ``uniform(-val, +val)`` and applied about the
        segment's own center.
        """

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    function = "{DIR}.mesh_terrains:beam_terrain"

    border_size: tuple[float, float] = MISSING
    """Inner dimensions ``(x, y)`` of the rectangular area enclosed by the
    border [m].

    Beams extend from the platform edge to this boundary.  The border
    ring occupies the space between ``border_size`` and
    :attr:`~isaaclab.terrains.SubTerrainBaseCfg.size`.
    """

    border_enabled: bool = True
    """Whether to generate the outer border ring. Defaults to ``True``.

    When ``False``, the border mesh is omitted but ``border_size`` is
    still used to determine where beams end.
    """

    num_bars: tuple[int, int] = MISSING
    """Number of beams as ``(easy, hard)``.

    Interpolated with difficulty and rounded to an integer.
    """

    beam_distribution: Literal["uniform", "random"] = MISSING
    """Angular placement of beams.

    - ``"uniform"`` — beams are evenly spaced at ``360° / num_bars``.
    - ``"random"``  — beams are placed at random non-overlapping angles.
    """

    platform_height: float | tuple[float, float] | None = None
    """Elevation of the central platform above the border [m].

    A positive value raises the platform; beams ramp down from the
    elevated platform to the border.  A negative value lowers the
    platform; beams ramp upward.

    A tuple ``(easy, hard)`` is interpolated linearly with difficulty.
    ``None`` (default) means flush with the border (elevation = 0).
    """

    platform_height_noise: float = 0.0
    """Random noise added to :attr:`platform_height` [m].

    The final elevation is ``platform_height + uniform(-noise, +noise)``.
    """

    beam_style: FlatBeamCfg | BoxBeamCfg = FlatBeamCfg()
    """Beam style configuration.

    Assign a :class:`FlatBeamCfg` (default) for continuous beams or a
    :class:`BoxBeamCfg` for segmented box beams.
    """
