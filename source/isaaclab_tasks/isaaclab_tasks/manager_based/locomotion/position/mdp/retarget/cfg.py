# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the retargeting pipeline."""

from __future__ import annotations

from dataclasses import MISSING, field

from isaaclab.utils import configclass

from ...utils.criteria_cfg import CriterionBaseCfg
from ...utils.kinematic.ik_objectives.cfg import IKObjectiveBaseCfg
from ...utils.kinematic.newton_kinematics import NewtonKinematicsCfg


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
    """Terrain patch-detection parameters for :class:`TerrainFirstSampler`.

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

    criteria_yield: float = 0.25
    """Expected fraction of IK solves surviving acceptance criteria.

    Empirically ~0.45-0.55 on rough terrains with collision + HAA +
    stability filters; raise toward ``0.75`` on flat terrain or when
    most criteria are disabled.
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
class TerrainFirstSamplerCfg(SamplerBaseCfg):
    """Configuration for terrain-first support-polygon sampling.

    Splits into two sub-configs for clarity:

    * :attr:`patch` -- terrain patch detection (flatness filter, sampling
      extent).
    * :attr:`sizing` -- yield-rate cascade that back-derives stage sizes
      from the caller's ``n_desired``.

    Geometry-dependent thresholds (reachability envelopes, base-height
    bounds) are derived automatically from the robot's default stance
    in :class:`~isaaclab_tasks.manager_based.locomotion.position.utils.sampling.TerrainFirstSampler`.
    """

    class_type: type | str = "isaaclab_tasks.manager_based.locomotion.position.utils.sampling:TerrainFirstSampler"
    """Sampler implementation class."""

    patch: PatchSamplingCfg = PatchSamplingCfg()
    """Terrain patch-detection parameters (flatness filter, sampling extent)."""

    sizing: SamplerSizingCfg = SamplerSizingCfg()
    """Yield-rate cascade that back-derives stage sizes from ``n_desired``."""

    min_contacts: int = -1
    """Minimum contact slots a candidate must fill to be accepted.

    ``-1`` (the default) preserves the original hard-polygon behavior:
    every slot must find an in-envelope terrain patch, otherwise the
    candidate is rejected and every accepted candidate has
    ``is_contact = True`` for all slots.

    A positive integer ``m`` (``1 <= m <= nc``) enables the soft polygon
    builder: a candidate is accepted when at least ``m`` slots found a
    patch; the remaining slots are classified as *air* (``is_contact =
    False``) with a template-projected target position. Downstream
    criteria (stability, foot-position error) consume ``is_contact`` to
    ignore air slots. Use ``m = 2`` to let mixed-geometry terrain
    (stepping-stone-with-gap, narrow ledges) emit nc=2/3/4 stances.
    """

    fk_num_samples: int = 100000
    """Number of random-joint FK samples used to estimate the per-foot
    reach envelope and the canonical-shape NN library."""

    fk_num_retained: int = 25000
    """Number of FK samples kept as the canonical-shape NN library.

    ``fk_num_samples`` is used for robust quantile estimation of the
    reach envelope; a uniform subset of size ``fk_num_retained`` is
    retained as the NN library consulted at query time. Larger values
    give tighter NN coverage at linear memory + match-time cost.
    """

    fk_shape_tol: float = 0.08
    """NN acceptance radius in canonical shape space [m].

    A polygon passes the shape gate iff every foot has some FK sample
    within this distance (per-foot L2, worst foot). Tighter values
    reject more polygons; looser values accept more including near-
    degenerate shapes. ``0.08`` matches empirical inter-foot spread
    of ~3-6 cm around default stance on typical quadrupeds.

    Unused when :attr:`use_template_projection` is ``True``.
    """

    use_template_projection: bool = False
    """Project FK templates onto terrain instead of matching.

    When ``True``, the query-time path switches from
    "sample-polygon-then-NN-match" to "pick-template-then-project":

    1. Sample a random FK template per candidate (no library FPS, no
       symmetry augmentation needed -- all FK samples are realizable
       polygons by construction).
    2. Un-canonicalize the template at the candidate's ``(center, yaw)``
       to get world-frame per-foot positions.
    3. For each foot, ``torch.cdist`` against morphological patches; if
       the nearest patch is within :attr:`foot_contact_radius`, the
       foot contacts that patch. Otherwise the foot is air.
    4. Templates whose canonical ``|z|`` exceeds
       :attr:`foot_contact_radius` for some slot mark that slot as
       air-in-FK-pose regardless of terrain.

    Contact count emerges from template+terrain geometry: no
    reachability sectors, no shape match, no ``matched_perm``. This
    makes 2-contact/3-contact stances first-class rather than
    fallbacks. Dropped when ``False`` (legacy behavior).
    """

    on_plane_tol: float = 0.03
    """Canonical |z| [m] within which a foot is on the stance plane.

    Build-time classifier: a template slot is on-plane iff its
    canonical ``|z|`` is below this threshold. Physical — a foot
    sole within ~3 cm of the stance plane is touching / near-touching.
    Unused when :attr:`use_template_projection` is ``False``.
    """

    terrain_presence_radius: float = 0.15
    """xy radius [m] for "is there a morph patch near this foot's projected xy?".

    Query-time presence check: ``torch.cdist`` between projected foot
    xy and morph-patch xy; foot is contact iff nearest patch is within
    this radius AND template slot is on-plane. Tuned to morph-patch
    density: morph sampling yields ~15 cm grid on typical n_desired,
    so 15 cm catches the nearest patch reliably without requiring the
    foot to land exactly on a grid point. Unused when
    :attr:`use_template_projection` is ``False``.
    """

    template_min_on_plane: int = -1
    """Optional stance-quality filter for the template pool.

    FK uniform joint sampling spans the full reachable foot workspace,
    including extreme joint configurations (fully folded / fully
    extended) that would be unusual as static stances. Setting this to
    ``k > 0`` drops FK samples where fewer than ``k`` feet lie close
    to the template's plane-fit stance plane (canonical ``|z|`` <
    :attr:`on_plane_tol`) -- a cheap proxy for "this FK pose looks
    like a plausible quadruped stance". ``-1`` (default) disables the
    filter.

    This is **not** a contact-decision input. In the template-
    projection path, contact is decided purely by terrain
    (``patch_near``): a foot at a different world z (e.g. one step
    higher on stairs) is still a valid contact. The filter only
    shapes which *xy layouts* the pool samples from.
    """


@configclass
class TemplateMatchedSamplerCfg(TerrainFirstSamplerCfg):
    """Configuration for hybrid template-matched support-polygon sampling.

    Keeps :class:`~isaaclab_tasks.manager_based.locomotion.position.utils.sampling.TerrainFirstSampler`'s
    polygon-assembly machinery (per-foot reach-envelope sampling,
    morphological-patch pool, plane-fit IK seeding) but replaces the
    pass/fail NN against the full FK shape distribution with a NN match
    against an FPS-thinned template library (~500-1000 templates) that
    returns the matched template id. The id carries per-placement slot
    assignment, which lets the sampler generalise beyond homogeneous
    quadrupeds.

    Build-time symmetry augmentation (:attr:`symmetry_permutations`)
    provides permutation coverage for symmetric robots: for each base
    template, each non-identity permutation is applied to the FK world
    foot positions and re-canonicalised, producing additional templates
    with their permutation stored as the slot-assignment gather index.

    Inherits :attr:`patch`, :attr:`sizing`, :attr:`min_contacts`,
    :attr:`fk_num_samples`, :attr:`fk_num_retained`, and :attr:`fk_shape_tol`
    from :class:`TerrainFirstSamplerCfg`.
    """

    class_type: type | str = "isaaclab_tasks.manager_based.locomotion.position.utils.sampling:TemplateMatchedSampler"
    """Sampler implementation class."""

    n_templates: int = 1000
    """Target size of the FPS-thinned template library (pre-symmetry-aug).

    Smaller than :attr:`fk_num_retained` — FPS thinning picks diverse
    canonical-shape representatives so NN coverage stays broad.
    Empirically, 1000 × 4 symmetry elements (homogeneous quadruped)
    gives ~70-80% per-foot-independent NN coverage on flat anymal_c at
    :attr:`template_shape_tol` = 0.10m; bump to 2000-4000 to close the
    coverage gap with the FK-sample library.
    """

    template_shape_tol: float = 0.10
    """Accept-match tolerance in canonical shape space [m].

    Per-foot-independent worst-foot L2 distance threshold. Slightly
    looser than :attr:`fk_shape_tol` to compensate for the sparser
    :attr:`n_templates` pool.
    """

    symmetry_permutations: list[list[int]] = field(default_factory=list)
    """Body-plan symmetry permutations for build-time template augmentation.

    Each inner list is a permutation of ``[0, 1, ..., nc-1]`` describing
    how slot indices map under one symmetry element. The identity
    permutation is always included -- this list adds *non-identity*
    elements only. Empty list (the default) means no augmentation
    (``|G|=1``), appropriate for robots whose FK sample distribution
    already covers symmetric variants densely or for fully asymmetric
    body plans.

    For homogeneous quadrupeds, pass the three non-identity cyclic
    rotations to opt into 4-fold symmetry (``|G|=4``): e.g.
    ``[[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]`` in nominal CCW order.
    """


@configclass
class RetargetPipelineCfg:
    """Full retarget pipeline configuration.

    Nests the kinematics, sampler, and foot specification so the
    pipeline can be constructed with ``cfg.class_type(cfg)``.
    """

    class_type: type | str = "{DIR}.pipeline:RetargetPipeline"
    """Pipeline implementation class."""

    kin: NewtonKinematicsCfg = MISSING  # type: ignore[assignment]
    """Kinematics model configuration."""

    sampler: SamplerBaseCfg = MISSING  # type: ignore[assignment]
    """Sampler configuration (with ``class_type`` set)."""

    foot_body_names: list[str] = MISSING  # type: ignore[assignment]
    """Body names of the feet (exact match against Newton body names)."""

    haa_joint_pattern: str | None = None
    """Optional regex matching hip-abduction/adduction joint names.

    Consumed by the default criteria factory to build a
    :class:`~isaaclab_tasks.manager_based.locomotion.position.utils.criteria.HaaLimit`
    criterion. ``None`` disables the HAA check (appropriate for robots that
    have no abduction joints or where over-splay is not a concern).
    """

    joint_regularize_targets: dict[str, float] = field(default_factory=dict)
    """Optional joint-name regex -> target-angle mapping for IK regularization.

    Consumed by :class:`IKObjectiveJointRegularizeCfg` when its own
    :attr:`joint_targets` is empty -- a robot preset can set this once
    at the pipeline level and every regularize objective inherits it.
    Empty dict disables the regularizer.
    """

    ik_iterations: int = 200
    """Maximum number of IK solver iterations."""

    ik_convergence_threshold: float = 0.01
    """Stop IK early when mean cost change falls below this threshold."""

    base_pos_weight: float = 0.05
    """Weight of the base-position IK objective [unitless].

    Keeps the IK near the sampler's plane-fit base position. Small by
    default so the foot-contact targets (weight 1.0) dominate -- the
    base is a soft anchor, not a hard target.
    """

    base_rot_weight: float = 0.5
    """Weight of the base-orientation IK objective [unitless].

    Pulls the base quaternion toward the sampler's plane-fit target.
    For nc<4 stances (raised legs) the default 0.5 can be overpowered
    by the stability-margin objective at weight 1.0, producing poses
    that tilt the base to project the COM onto a 2-foot segment.
    Raise to 2.0-5.0 when running sub-4-contact IK to hold the base
    upright; leave at 0.5 for full-contact quadruped stances where the
    plane-fit already agrees with stability.
    """

    extra_objectives: list[IKObjectiveBaseCfg] = field(default_factory=list)
    """IK objectives appended to the standard pipeline set.

    Each entry declares the objective class and its static parameters;
    runtime state (``kin``, ``foot_body_ids``, ``wp_mesh.id``,
    ``sampler``) is injected by :meth:`IKObjectiveBaseCfg.build`.
    Empty list runs the pipeline with only the standard objectives
    (foot-position contact, base pose, joint limits).
    """

    criteria: list[CriterionBaseCfg] = field(default_factory=list)
    """Acceptance criteria applied in list order to post-IK candidates.

    Each entry declares the criterion class, its :attr:`CriterionBaseCfg.name`
    (which keys the rejection summary), and its static parameters;
    runtime state (``kin``, ``foot_body_ids``, ``_solver_costs``) is
    injected by :meth:`CriterionBaseCfg.build`. Empty list keeps every
    geometry-valid IK solve.
    """
