# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the retargeting pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field

from isaaclab.utils.configclass import configclass


@configclass
class SamplerBaseCfg:
    """Base configuration for a pipeline sampler.

    Subclass this to define a concrete sampling strategy and set
    :attr:`class_type` to the corresponding :class:`SamplerBase` subclass.
    """

    class_type: type = None  # type: ignore[assignment]
    """Implementation class.  Must be a :class:`SamplerBase` subclass."""


@configclass
class PatchSamplingCfg:
    """Terrain patch-detection parameters for :class:`Sampler`.

    Drives the morphological flatness filter that finds candidate foot
    contact patches on the terrain heightmap. All values are terrain
    properties -- robot geometry is derived from :class:`NewtonKinematics`.
    """

    contact_radius: float = 0.04
    """Contact patch radius for morphological flatness check [m]."""

    max_height_diff: float = 0.03
    """Maximum height variation within a contact patch [m]."""

    horizontal_scale: float = 0.03
    """Heightmap rasterization grid spacing [m]."""

    oversample_ratio: float = 5.0
    """Oversample factor for farthest-point refinement of morphological patches.

    Values above ``1.0`` instruct the morphological filter to first extract
    ``oversample_ratio * num_patches`` candidates and then thin them to
    ``num_patches`` via farthest-point sampling, giving spatially uniform
    coverage instead of density-proportional sampling.

    FPS matters on heterogeneous tiles: e.g. ``HfPyramidStairsTerrainCfg``
    with ``border_width=1.0`` puts ~40% of the rasterized heightmap in a
    single flat z-band, so uniform-random cell sampling
    (``oversample_ratio=1.0``) over-represents the flat border at the
    expense of stairs. The default ``5.0`` roughly halves the top-flat
    bias on such tiles while staying cheap on GPU.
    """

    min_center_dist: float = 0.05
    """Minimum distance from center to candidate (avoid overlapping) [m]."""

    x_range: tuple[float, float] | None = None
    """Optional X sampling bounds [m], relative to the sub-terrain origin.

    When ``None`` (default), the morphological sampler uses the full mesh
    XY extent -- which on grid-style arenas includes the flat border padding
    added by :class:`isaaclab.terrains.TerrainGeneratorCfg`. Set this to the
    inner non-border extent (e.g. ``(-num_rows * size_x / 2, +num_rows *
    size_x / 2)``) to keep patches on the actual terrain tiles.
    """

    y_range: tuple[float, float] | None = None
    """Optional Y sampling bounds [m], relative to the sub-terrain origin.

    See :attr:`x_range` for semantics.
    """


@configclass
class SamplerSizingCfg:
    """Yield-rate cascade that back-derives sampler stage sizes from ``n_desired``.

    Every knob is either an **oversample** multiplier at a downsampling
    stage (pool gets bigger so the downstream thinning has room to
    select a spatially diverse subset) or a **yield** fraction at a
    filter stage (what fraction survives). The cascade runs from the
    final output backwards -- see :func:`compute_sampler_sizing` for
    the full walk.
    """

    final_fps_oversample: float = 3.0
    """Headroom multiplier into the final FPS (robots post-criteria / robots desired).

    ``1.5`` keeps the final FPS pool 50% larger than the desired robot
    count so spatial spread is achievable rather than degenerate.
    """

    fps_spacing: float | None = None
    """Optional spacing-driven target count for the final FPS thinning.

    When set, the post-IK FPS step computes its target placement count
    *from the actual feature-space bounding box of surviving candidates*
    rather than using the upstream ``n_desired``. Specifically:

    .. code-block:: text

        bbox       = features.amax(0) - features.amin(0)
        xy_area    = bbox[0] * bbox[1]                # treat xy as 2-D manifold
        extra_vol  = bbox[3:].prod() if D > 3 else 1
        D_eff      = 2 + max(0, D - 3)                # ignore z; count xyz extras
        k_target   = max(1, int(xy_area * extra_vol / spacing**D_eff))
        n_select   = min(k_target, n_valid)

    Lets the same ``spacing`` mean different placement counts depending on
    the chosen :paramref:`fps_features` — adding yaw or rotation
    naturally scales the count up because the metric volume to fill is
    larger. ``None`` (default) keeps the production behaviour where the
    final FPS thins to ``n_desired`` directly.
    """

    fps_features: Callable | None = None
    """Custom feature extractor for the final FPS spatial-thinning step.

    Maps ``(states: [N, joint_coord_count]) -> [N, D]`` where ``D`` is
    the metric-space dimensionality the FPS thins in. ``None`` (default)
    falls back to root xyz (3-D Cartesian) — same behaviour as before.

    The grid bucketer partitions whatever ``D`` you return into cells of
    side ``(volume / k)^{1/D}``, so adding orientation/joint dimensions
    *automatically* refines the sampling at the same ``pool_spacing``:
    the volume to fill grows, more buckets exist, and the same target
    count covers more pose diversity.

    See :mod:`.feature_extractors` for canned options:

    * :func:`~.feature_extractors.xyz_features` — current default.
    * :class:`~.feature_extractors.XYZYawFeatures` (``yaw_scale``).
    * :class:`~.feature_extractors.XYZAxisAngleFeatures` (``rot_scale``).
    * :class:`~.feature_extractors.XYZJointsFeatures` (``joint_scale``).

    Or pass any function with the matching signature for full control —
    e.g. ``lambda s: torch.cat([s[:, :3], s[:, 7:13] * 0.4], dim=-1)``.
    """

    criteria_yield: float = 0.25
    """Expected fraction of IK solves surviving acceptance criteria.

    Empirically ~0.45-0.55 on rough terrains with collision,
    lateral-hip, and stability filters; raise toward ``0.75`` on flat
    terrain or when most criteria are disabled.
    """

    polygon_fps_oversample: float = 5.0
    """Headroom multiplier into the polygon grid-bucket FPS stage.

    Polygon assembly is cheap (centers are reused across yaws via
    per-foot reachability sampling), so a generous pool gives bucket-FPS
    meaningful spatial diversity to thin from. IK cost is unaffected
    since the post-FPS target-n is what feeds the solver.
    """

    polygon_assembly_yield: float = 0.8
    """Expected fraction of neighborhoods whose assembled polygon passes base-height.

    Each neighborhood emits exactly one polygon (foot assignment is
    pinned at sample time via per-foot reachability envelopes), so this
    is a per-polygon yield. Typically ~0.8 on well-formed sub-terrains --
    lower on heavily cluttered meshes where many sectors lack valid
    contact patches.
    """

    morph_patch_oversample: float = 4.0
    """Morph-patches per foot slot per final robot.

    Decoupled from K since centers are reused across yaws. Only needs
    to be large enough that each foot-sector query within a
    neighborhood's reachability ball returns at least one patch. Scales
    with ``n_final`` (a proxy for terrain area), not K.
    """


@configclass
class SamplerCfg(SamplerBaseCfg):
    """Configuration for template-projection support-polygon sampling.

    Splits into two sub-configs for clarity:

    * :attr:`patch` -- terrain patch detection (flatness filter, sampling
      extent).
    * :attr:`sizing` -- yield-rate cascade that back-derives stage sizes
      from the caller's ``n_desired``.

    Geometry-dependent thresholds (nominal foot angle, standing height)
    are derived automatically from the robot's random-joint FK
    distribution in :class:`~isaaclab_tasks.core.multi_task.terrain.retarget.contact_sampling.Sampler`.
    """

    class_type: type | str = "isaaclab_tasks.core.multi_task.terrain.retarget.contact_sampling:Sampler"
    """Sampler implementation class."""

    patch: PatchSamplingCfg = PatchSamplingCfg()
    """Terrain patch-detection parameters (flatness filter, sampling extent)."""

    sizing: SamplerSizingCfg = SamplerSizingCfg()
    """Yield-rate cascade that back-derives stage sizes from ``n_desired``."""

    min_contacts: int = -1
    """Minimum contact slots a candidate must fill to be accepted.

    ``-1`` (the default) requires every slot to find a morph patch
    within :attr:`terrain_snap_distance`: every accepted candidate has
    ``is_contact = True`` for all slots.

    A positive integer ``m`` (``1 <= m <= nc``) enables soft polygons:
    a candidate is accepted when at least ``m`` slots snap to a patch;
    remaining slots become *air* (``is_contact = False``) with a
    template-projected target. Downstream criteria (stability,
    foot-position error) consume ``is_contact`` to ignore air slots.
    Use ``m = 2`` to let mixed-geometry terrain (stepping-stone-with-gap,
    narrow ledges) emit nc=2/3/4 stances.
    """

    fk_num_samples: int = 100000
    """Number of random-joint FK samples used to build the canonical-shape
    library and estimate per-foot nominal angles / standing height."""

    fk_joint_range: float = 1.57
    """Default clamp [rad] for random-joint FK sampling around the URDF default.

    Each revolute joint is sampled uniformly in ``default ± fk_joint_range``,
    intersected with the URDF joint limits and with any more-specific
    :attr:`fk_joint_range_overrides` entry. Needed because quadruped
    URDFs often specify HFE/KFE as ``±9.42`` rad (placeholder) that the
    USD loader promotes to ``±1e10``; uniform sampling over that range
    yields wrap-around jq whose foot positions look fine as a convex
    quad but whose leg paths route across the chassis (visually crossed
    legs).

    ``1.57`` rad (π/2) covers a quarter rotation per joint, which is
    usually enough for broad FK coverage. Override
    :attr:`fk_joint_range_overrides` only for joints whose mechanical
    range is known to be tighter than the URDF claims.
    """

    fk_joint_range_overrides: dict[str, float] = field(default_factory=dict)
    """Per-joint-name-regex clamps that override :attr:`fk_joint_range`.

    Keys are regex patterns matched against full joint names with
    :func:`re.fullmatch`; values are the clamp [rad] applied to those
    joints. Use this to express mechanical limits that the URDF ↔ USD
    pipeline stripped, for example::

        fk_joint_range_overrides={"<joint_regex>": 0.7}

    Without this, affected joints get the default ``1.57`` clamp and FK
    samples can produce visually crossed-leg IK poses even though the
    foot targets are valid.
    """

    fk_num_retained: int = 5000
    """Number of canonical-shape templates kept via FPS thinning.

    ``fk_num_samples`` is used for robust quantile estimation of the
    nominal angle / standing height; of the hull-valid remainder a
    farthest-point subset of this size becomes the template library
    from which each candidate draws at query time. FPS (via
    ``grid_bucket_downsample`` on the flattened canonical shape)
    guarantees each retained template represents a geometrically
    distinct stance, so random ``tpl_idx`` draws cover the FK
    manifold evenly. Bump if you want acrobatic stance diversity.
    """

    outward_snap_penalty: float = 0.0
    """LSA cost multiplier for radially-outward foot snaps [unitless].

    Added to each contact-foot's LSA cost as
    ``penalty × max(0, r_patch − r_template)`` where ``r`` is distance
    to the stance centroid. Motivation: nearest-patch snapping is
    biased outward in practice — each foot's nearest patch is on
    average slightly farther from centroid than the template
    predicted, and a 4-foot stance that each inflates by a few cm
    balloons past the leg's reach envelope and IK fails (foot_err
    rejects dominate). Set ``1.0``–``2.0`` on rough / sloped terrain
    where the outward-snap failure mode dominates criteria yield.
    Default ``0`` keeps plain nearest-patch snapping for sparse /
    small-mesh cases where any reachable patch beats air.
    """

    terrain_snap_distance: float = 0.15
    """Snap distance [m] for per-foot contact decision.

    After the constrained bipartite assignment picks a morph patch for
    each foot (minimising total template-projected-to-patch distance
    with one-patch-per-foot, convex-stance, and winding-preservation
    constraints), each foot is classified:

    * distance to assigned patch ``<= terrain_snap_distance`` → **contact**,
      target = patch xyz + ``foot_ground_offset``.
    * distance ``> terrain_snap_distance`` → **air**, target = template-
      predicted world xyz (z clamped above local terrain surface).

    Tuned to morph-patch density: morph sampling yields ~15 cm grid on
    typical ``n_desired``, so 15 cm catches the nearest patch reliably
    without requiring the foot to land exactly on a grid point. Bump
    to 0.25-0.30 for sparse / rough terrain (CONTOUR, EXTREME_STAIR at
    high difficulty).
    """
