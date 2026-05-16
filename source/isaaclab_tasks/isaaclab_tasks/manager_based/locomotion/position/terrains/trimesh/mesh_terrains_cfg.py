# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab import terrains
from isaaclab.utils.configclass import configclass

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


# ======================================================================
# MeshMazeTerrainCfg
# ======================================================================


@configclass
class MeshMazeTerrainCfg(terrains.SubTerrainBaseCfg):
    """A maze terrain generated on a 2D grid.

    A recursive backtracker carves passages through a full grid of walls.
    Each remaining wall segment is extruded into an axis-aligned box.
    Cell size is derived automatically from ``size / grid_dims``.
    """

    function = "{DIR}.mesh_terrains:maze_terrain"

    grid_cols: int | tuple[int, int] = 5
    """Number of maze columns. A tuple ``(easy, hard)`` is interpolated.

    More columns = narrower corridors = harder.
    """

    grid_rows: int | tuple[int, int] = 5
    """Number of maze rows. A tuple ``(easy, hard)`` is interpolated."""

    wall_thickness: float = 0.15
    """Thickness of each wall segment [m]."""

    wall_height: float | tuple[float, float] = 3.0
    """Height of each wall segment [m]."""

    open_ratio: float = 0.0
    """Fraction of additional walls to remove after maze generation.

    0 = pure maze (single solution path). Values toward 1 remove more
    walls, creating a more open terrain with multiple passages.
    """

    boundary_walls: bool = True
    """Whether to add walls around the outer perimeter. Defaults to ``True``."""

    grid_noise: float = 0.0
    """Random offset applied to each interior grid node as a fraction of
    cell size [0..1).

    0 = perfectly regular grid. 0.3 = each node jitters up to 30% of the
    cell size, producing irregular corridors of varying width.
    """


# ======================================================================
# MeshContourTerrainCfg
# ======================================================================


@configclass
class MeshContourTerrainCfg(terrains.SubTerrainBaseCfg):
    """Stepped contour terrain generated from a Perlin noise heightfield.

    A 2D noise field is sampled, then sliced at evenly-spaced height
    thresholds to produce contour rings.  Each ring is extruded into a
    mesh, stacking from bottom to top to form stepped topographic hills
    and valleys.
    """

    function = "{DIR}.mesh_terrains:contour_terrain"

    num_levels: int | tuple[int, int] = 6
    """Number of contour levels. A tuple ``(easy, hard)`` is interpolated.

    More levels = thinner steps = harder footing.
    """

    level_height: float | tuple[float, float] = 0.3
    """Height of each contour step [m]."""

    noise_scale: float = 3.0
    """Spatial frequency of the Perlin noise.

    Lower = broader hills, higher = more detailed features.
    """

    noise_octaves: int = 3
    """Number of Perlin noise octaves for detail layering."""

    noise_seed: int | None = None
    """Random seed for the noise field. ``None`` uses a random seed."""

    smoothing: float = 0.5
    """Contour polygon simplification tolerance [m].

    Higher values produce smoother, less jagged contour outlines.
    """

    @configclass
    class StoneCfg:
        """Configuration for scattering stones on contour terraces."""

        num_stones: int | tuple[int, int] = 15
        """Number of stones to scatter. A tuple ``(easy, hard)`` is interpolated."""

        size_range: tuple[float, float] = (0.2, 0.6)
        """Min and max stone radius [m]."""

        height_scale: float = 0.7
        """Vertical scale relative to radius (< 1 = flattened, > 1 = tall)."""

        roughness: float = 0.3
        """Amount of vertex displacement for irregular shape [0..1]."""

    stones: StoneCfg | None = None
    """Optional stone scattering on terraces. ``None`` disables stones."""


# ======================================================================
# Shared passway / beam style configs (used by multiple terrain types)
# ======================================================================


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


# ======================================================================
# MeshRadiatingBeamTerrainCfg
# ======================================================================


@configclass
class MeshRadiatingBeamTerrainCfg(terrains.MeshStarTerrainCfg):
    """A terrain with beams radiating from a central platform to an outer boundary.

    The terrain consists of three layers:

    1. **Central platform** — a cylinder at the terrain center whose diameter
       is :attr:`platform_width` (inherited) and whose top surface can be
       raised above the border by :attr:`platform_height`.
    2. **Border** — configurable via :attr:`border`:

       - :class:`SquareBorderCfg` — rectangular ring (default).
       - :class:`PlatformBorderCfg` — cylinders at each beam endpoint.
       - ``None`` — no border geometry.

    3. **Beams** — bridges connecting the platform edge to the boundary
       defined by the border config.  The number and angular placement are
       controlled by :attr:`num_bars` and :attr:`beam_distribution`.

    Beams come in two styles selected via :attr:`beam_style`:

    - :class:`FlatBeamCfg` — each beam is a single continuous box that
      tilts when ``platform_height != 0``.
    - :class:`BoxBeamCfg` — each beam is tiled with small box segments
      whose size, gap, position jitter, and yaw jitter are individually
      configurable.

    Inherited fields from :class:`~isaaclab.terrains.MeshStarTerrainCfg`:

    - :attr:`platform_width` — diameter of the central cylinder [m].
    - :attr:`bar_width_range` — ``(easy, hard)`` beam width [m],
      interpolated with difficulty.
    - :attr:`bar_height_range` — ``(easy, hard)`` beam / border thickness
      [m], interpolated with difficulty.
    """

    # backward-compatible aliases
    FlatBeamCfg = FlatBeamCfg
    BoxBeamCfg = BoxBeamCfg

    @configclass
    class SquareBorderCfg:
        """Rectangular border ring surrounding the terrain."""

        inner_size: tuple[float, float] = MISSING
        """Inner dimensions ``(x, y)`` of the open area inside the border [m]."""

    @configclass
    class PlatformBorderCfg:
        """Cylindrical platforms placed at the end of each beam."""

        inner_size: tuple[float, float] = MISSING
        """Defines how far from center beams extend ``(x, y)`` [m]."""

        radius: float = MISSING
        """Radius of each endpoint platform cylinder [m]."""

    function = "{DIR}.mesh_terrains:beam_terrain"

    border: SquareBorderCfg | PlatformBorderCfg | None = MISSING
    """Outer border configuration."""

    num_bars: tuple[int, int] = MISSING
    """Number of beams as ``(easy, hard)``."""

    beam_distribution: Literal["uniform", "random"] = MISSING
    """Angular placement of beams: ``"uniform"`` or ``"random"``."""

    platform_height: float | tuple[float, float] | None = None
    """Elevation of the central platform above the border [m].

    A tuple ``(easy, hard)`` is interpolated linearly with difficulty.
    ``None`` (default) means flush with the border.
    """

    platform_height_noise: float = 0.0
    """Random noise added to :attr:`platform_height` [m]."""

    beam_style: FlatBeamCfg | BoxBeamCfg = FlatBeamCfg()
    """Beam style: :class:`FlatBeamCfg` (default) or :class:`BoxBeamCfg`."""

    ground_plane: bool = True
    """Whether to generate a ground plane below the terrain. Defaults to ``True``.

    Set to ``False`` for a floating beam terrain with no ground,
    forcing the robot to stay on beams.
    """


# ======================================================================
# MeshFloatingIslandTerrainCfg
# ======================================================================


@configclass
class MeshFloatingIslandTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain of collision-free floating islands connected by passways.

    The generation pipeline:

    1. **Islands** are placed at random positions via rejection sampling
       (no overlaps).  Shape is configurable via :attr:`island_style`.
    2. A **graph** is built over the islands (Delaunay, MST, or KNN) and
       edges that pass through other islands are pruned.
    3. **Passways** are generated along each graph edge, reusing
       :class:`FlatBeamCfg` / :class:`BoxBeamCfg`.
    """

    # backward-compatible aliases
    FlatBeamCfg = FlatBeamCfg
    BoxBeamCfg = BoxBeamCfg

    @configclass
    class CylinderIslandCfg:
        """Cylindrical island platforms."""

        radius: float | tuple[float, float] = 0.8
        """Radius of each island [m].

        A tuple ``(easy, hard)`` is interpolated with difficulty.
        """

    @configclass
    class BoxIslandCfg:
        """Rectangular box island platforms."""

        length: float | tuple[float, float] = 1.0
        """Length of each island along a random orientation [m]."""

        width: float | tuple[float, float] = 1.0
        """Width of each island [m]."""

    @configclass
    class DelaunayGraphCfg:
        """Connect islands via Delaunay triangulation."""

    @configclass
    class MSTGraphCfg:
        """Connect islands via minimum spanning tree (every island reachable)."""

    @configclass
    class KNNGraphCfg:
        """Connect each island to its *k* nearest neighbors."""

        k: int = 3
        """Number of nearest neighbors per island."""

    function = "{DIR}.mesh_terrains:floating_island_terrain"

    num_islands: int | tuple[int, int] = 8
    """Number of islands. A tuple ``(easy, hard)`` is interpolated."""

    island_style: CylinderIslandCfg | BoxIslandCfg = CylinderIslandCfg()
    """Shape of the island platforms."""

    island_height: float = 1.5
    """Thickness of each island [m]."""

    island_height_variation: float = 0.0
    """Per-island random vertical offset [m], sampled from ``uniform(-val, +val)``."""

    island_margin: float = 0.5
    """Minimum gap between island bounding circles during placement [m]."""

    graph: DelaunayGraphCfg | MSTGraphCfg | KNNGraphCfg = DelaunayGraphCfg()
    """Graph algorithm used to connect islands."""

    passway_style: FlatBeamCfg | BoxBeamCfg = FlatBeamCfg()
    """Passway style between connected islands."""

    passway_width: float | tuple[float, float] = 0.5
    """Width of passways [m]. A tuple ``(easy, hard)`` is interpolated."""

    passway_height: float | None = None
    """Thickness of passways [m]. ``None`` (default) uses :attr:`island_height`."""

    passway_curvature: float = 0.0
    """Maximum lateral offset of the passway midpoint as a fraction of the
    passway length.

    A random control point is placed perpendicular to the straight line
    between two islands at ``distance = curvature * length``, producing
    a quadratic bezier arc.  0 (default) means straight passways.
    """
