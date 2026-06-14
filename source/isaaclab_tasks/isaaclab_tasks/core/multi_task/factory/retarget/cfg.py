# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the offline factory Newton-IK reset-state pipeline.

Mirrors the terrain ``RetargetPipelineCfg`` shape: the usage site reads like the
pipeline itself -- a placement sampler, a grasp-pair sampler, the IK solve, ONE
avoidance objective family (``None`` disables steering), a typed CRITERIA list
whose membership controls which acceptance gates run, and the reach-row pass::

    FactoryIKPipelineCfg(
        placement=PlacementSamplingCfg(...),
        solve=IKSolveCfg(iterations=250, pos_tol=0.004, avoidance=CollisionAvoidanceCfg(weight=20.0)),
        criteria=[
            CollisionCheckCfg(max_pen=0.0005, self_max_pen=0.002),
        ],
        reach=ReachRowsCfg(standoff_range=(0.03, 0.15)),
    )

This replaces the sim-in-the-loop ``FactoryResetStateCommand`` table fill
(``CollisionAnalyzer`` + ``RigidObjectHasher`` + in-sim DLS IK) with an offline
batched Newton-IK solve over fingertip contact-pair targets.
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ..factory_presets import (
    EndEffectorBodyCfg,
    FactoryAssemblyProfileCfg,
    FingerBodyNamesCfg,
    FixedAssetMapCfg,
    GripperBodyNamesCfg,
    HeldAssetAlignOffsetCfg,
)


@configclass
class BoardLibraryCfg:
    """The board+bolt CONFIGURATION LIBRARY -- the analog of the terrain grid.

    Describes the WORLD, independent of the nut and the robot: a fixed set of
    board poses (the bolt rides each board at its keypoint offset). The library
    is sampled once per table build and never changes; every nut placement binds
    to one configuration, rows record their ``board_index`` (= terrain
    ``tile_index``), and spawn x target pairing happens WITHIN a configuration
    (a goal solved against a different board pose would point at the wrong bolt).
    """

    board_asset_cfg: SceneEntityCfg = SceneEntityCfg("nistboard")
    """The board scene entity carrying the fixed asset (the posed assembly group)."""

    fixed_asset_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset")
    """The fixed (assembly socket) scene entity, riding the board keypoint."""

    fixed_asset_map: dict = FixedAssetMapCfg()  # type: ignore[assignment]
    """Scene-entity -> board-keypoint mapping (variant preset): where on the board
    the fixed asset mounts. The same source of truth ``reset_fixed_assets`` used live."""

    num_boards: int = 128
    """Library size: how many distinct board+bolt configurations exist."""

    library_oversample: float = 2.0
    """Candidate configurations sampled per kept one. Feasibility is proven by the
    single build round itself -- a candidate qualifies when it supplies at least
    ``rows_per_board`` accepted rows -- and ``num_boards`` qualified candidates
    are kept by pose-space FPS (spread, not ease)."""

    pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.2),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.8, 0.8),
        }
    )
    """Board pose DELTA [m, rad] around its scene init pose. Poses are
    oversampled, collision-rejected against the table and robot base, and
    FPS-downsampled to :attr:`num_boards`; arm reachability is left to the IK
    funnel."""

    oversample: float = 4.0
    """Poses sampled per library slot before rejection + FPS (single shot, no
    resample loop)."""

    clear_tol: float = 0.0005
    """Reject a board whose surface probes penetrate the table or the robot base
    deeper than this [m]."""


@configclass
class GraspSamplingCfg:
    """Antipodal grasp-pair sampling on the held-asset mesh.

    Mesh-general: pairs are surface points with opposed normals (within the
    friction cone about the pair axis) whose separation fits the gripper aperture
    range. Hex-flat pinches, axial pinches, wall (rim) pinches, and bore-expansion
    grasps all fall out of the same condition -- no annotated grasp keypoint, no
    asset-specific parameterization. An FK library over ALL gripper orientations
    seeds each IK problem from the nearest template by pair geometry.
    """

    grasps_per_placement: int = 8
    """Pairs sampled (with replacement) per nut placement (the ``G`` in ``W x G``)."""

    ik_seeds_per_grasp: int = 1
    """IK problems solved per grasp candidate, each seeded from a different nearby
    FK template. ``> 1`` explores multiple arm branches (elbow/approach homotopy
    classes) per grasp -- a gradient solve cannot re-route around the board from a
    single seed -- at linear solver cost. All surviving branches are kept as rows."""

    n_surface_samples: int = 2048
    """Area-weighted surface samples on the held collider mesh for pair generation."""

    friction_mu: float = 0.5
    """Friction coefficient; contact normals must lie within ``atan(mu)`` of the
    pair axis for the pair to qualify as antipodal (force closure under friction)."""

    aperture_range: tuple[float, float] = (0.002, 0.08)
    """Pair-separation limits [m]. The upper bound is the gripper's full opening
    (Franka: 2 x 0.04 finger travel)."""

    n_pairs_retained: int = 512
    """Pair budget after grid-bucket FPS thinning in (midpoint, axis) feature space."""

    seed_axis_scale: float = 0.3
    """[m] per unit of pair-axis direction in the seed-match / FPS feature space."""

    fk_num_samples: int = 8000
    """Random-config FK samples used to build the seed library."""

    fk_num_retained: int = 1500
    """Seed-library templates kept via grid-bucket FPS thinning."""

    fk_joint_range: float = 1.5
    """Arm-joint sampling clamp [rad] around the franka default for the FK library."""


@configclass
class PlacementSamplingCfg:
    """Nut placement sampling WITHIN the board configurations.

    Each placement puts the nut concentric on its configuration's bolt at a
    sampled assembly fraction (``on_bolt``), resting on its board (``on_table``),
    or floating freely (``in_air``). Always *nut-first*: the nut pose is sampler
    data (the assets are not in the kinematic chain), grasp pairs are sampled on
    it, and the arm is IK-solved to put the finger pads on them.
    """

    held_asset_cfg: SceneEntityCfg = SceneEntityCfg("held_asset")
    """The held (grasped) scene entity -- the object whose states are sampled."""

    assembly_profile: object = FactoryAssemblyProfileCfg()  # type: ignore[assignment]
    """Assembly-path profile (variant preset) the on-bolt bands sample along."""

    align_offset: object = HeldAssetAlignOffsetCfg()  # type: ignore[assignment]
    """Held-asset alignment keypoint offset (variant preset)."""

    placements_per_board: int = 4
    """Nut placements sampled PER board configuration in a build round (the total
    scales with the library automatically: ``placements_per_board x len(library)``
    candidates -- growing or oversampling the library never thins each board's
    feasibility evidence)."""

    grasp: GraspSamplingCfg = GraspSamplingCfg()
    """How each placement becomes states: antipodal grasp pairs cast onto the
    placed nut (``grasps_per_placement`` x ``ik_seeds_per_grasp`` IK problems per
    placement). Nested here because grasps are per-placement work, not a separate
    stage: cells -> placements -> grasps -> seeds."""

    placement_weights: dict[str, float] = field(
        default_factory=lambda: {"on_bolt": 0.5, "on_table": 0.2, "in_air": 0.3}
    )
    """Relative sampling weight per placement type. Keys must be a subset of
    ``{"on_bolt", "on_table", "in_air"}``; placements are split proportionally."""

    assembly_bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "near_seated": (0.0, 0.33),
            "mid_insertion": (0.33, 0.85),
            "above_tip": (0.85, 1.6),
        }
    )
    """``on_bolt`` assembly-fraction bands (fraction along the insertion axis).
    Each band becomes its own tag so the curriculum can monitor seating depth."""

    in_air_pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.15, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.015, 0.2),
            "roll": (-1.57, 1.57),
            "pitch": (-1.57, 1.57),
            "yaw": (-3.14, 3.14),
        }
    )
    """Free-space nut pose range [m, rad], relative to the franka base. Matches the
    original ``GRIPPER_GRASP_ASSET_IN_AIR.reset_asset_in_air`` range."""

    on_table_pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.25, 0.6),
            "y": (-0.25, 0.25),
            "yaw": (-3.14, 3.14),
        }
    )
    """Board-resting nut xy/yaw range [m, rad], expressed in WORLD at the board's
    canonical scene pose; the sampled pose is re-expressed in the board frame and
    rides the per-sub-world board (so a tilted board carries its resting nut).
    Height is pinned to :attr:`table_height`; roll/pitch are zero (nut lies flat
    on the board)."""

    table_height: float = 0.04
    """Nut-root height [m] when resting on the board AT ITS CANONICAL scene pose
    (board top + nut half-height); rides the sampled board like the xy/yaw range."""


@configclass
class CollisionAvoidanceCfg:
    """Differentiable avoidance objectives steering the refine phase.

    One instance per obstacle is created automatically: every robot link (base
    excluded) vs each static obstacle and the posed board + bolt, plus the hand
    vs the per-problem-posed nut (pad probes exempt -- they grasp it). Objectives
    STEER; the criteria below remain the correctness guarantee. Set
    :attr:`FactoryIKPipelineCfg.avoidance` to ``None`` to disable steering."""

    weight: float = 20.0
    """Penalty weight."""

    margin: float = 0.001
    """Softplus smoothing scale [m]. Keep small (~1 mm) so the penalty reacts to
    near-contact/penetration only, not to centimetre-scale clearance."""

    n_samples: int = 48
    """Probe budget PER OBSTACLE (a strided subset of the criteria probe sets).
    Steering needs far fewer probes than gating; LM cost scales with this."""

    max_dist: float = 0.25
    """``wp.mesh_query_point`` search radius [m] -- steering only needs the near field."""


@configclass
class JointLimitObjectiveCfg:
    """Soft joint-limit objective in the LM solve."""

    weight: float = 10.0
    """Objective weight."""


@configclass
class JointDefaultObjectiveCfg:
    """Pulls every ARM coord toward the robot's stance (locomotion's
    ``IKObjectiveJointDefault``, arm-only: finger coords are owned by
    :class:`FingerPinObjectiveCfg`). Small weight -- it only takes up slack the
    fingertip targets leave."""

    weight: float = 0.002
    """Uniform residual weight applied to every arm coord. Factory needs
    millimetre fingertip accuracy, so this must stay ~25x below locomotion's
    0.05 (measured: 0.05 -> 99% unreachable; 0.002 IMPROVES reach and halves
    joint-limit rejections by centering the arm)."""


@configclass
class FingerPinObjectiveCfg:
    """Pins the finger coords to half the pair separation -- enforces the gripper
    mimic constraint structurally and conditions the LM solve."""

    weight: float = 10.0
    """Objective weight."""


@configclass
class IKSolveCfg:
    """The batched LM solve over per-fingertip position targets.

    Two phases: a reach-only solve for every candidate, then -- when
    :attr:`FactoryIKPipelineCfg.avoidance` is set -- a warm-seeded avoidance
    refine for the reachable survivors only (collision steering never improves
    reach, so the unreachable are culled before paying for it).
    """

    iterations: int = 250
    """Maximum LM iterations for the reach phase."""

    convergence_threshold: float = 1e-7
    """Stop early when the mean-cost change between iteration batches drops below this."""

    refine_iterations: int = 40
    """Max LM iterations for the avoidance-refine phase. Starting from the
    converged reach solution, it only trades millimetres of clearance, not finds
    the basin."""

    pos_tol: float = 0.004
    """Max per-fingertip position error [m] for a candidate to count as reachable
    (the phase-1 cull and the final reachability gate)."""

    objectives: list = field(
        default_factory=lambda: [
            JointLimitObjectiveCfg(),
            FingerPinObjectiveCfg(),
            CollisionAvoidanceCfg(),
        ]
    )
    """Soft constraint terms in the LM solve (the fingertip position targets are
    the solve itself, not a member). Membership enables a term -- the mirror of
    :attr:`FactoryRobotCfg.criteria` for hard gates. Omit
    :class:`CollisionAvoidanceCfg` to solve without obstacle steering."""


@configclass
class CollisionCheckCfg:
    """No unintended penetration between ANY participating bodies.

    One gate, four tests under the hood: robot links vs the obstacles and the
    posed board + bolt (point signed distance PLUS exact edge-vs-mesh crossing
    tests -- points alone miss the ~4 mm board slicing between them), gripper vs
    the held asset (SYMMETRIC, post aperture-relief), held asset vs the
    obstacles, and robot link-vs-link. Which contacts are INTENDED is pipeline
    policy, not configuration: grasp pads on the held surface, the nut on the
    bolt within its assembly band, the nut resting on the board for ``on_table``,
    and design-mounted link clusters within :attr:`adjacency_hops`.
    """

    n_samples: int = 240
    """Surface probe points (FPS) per checked body set."""

    max_pen: float = 0.0005
    """Reject below ``-max_pen`` [m] min signed distance (unintended contacts)."""

    self_max_pen: float = 0.002
    """Link-vs-link allowance [m]; a few mm absorbs design-close collider overlap."""

    adjacency_hops: int = 2
    """Kinematic-tree distance (in joints) within which link pairs are NOT
    checked (e.g. ``panda_link7 -> link8 -> hand`` colliders overlap by design)."""

    query_radius: float = 0.05
    """``wp.mesh_query_point`` search radius [m]. Must exceed the obstacles'
    deepest interior; crossings at any depth are the edge test's job."""


@configclass
class JointWithinLimitCfg:
    """Criterion: every arm coord inside its effective interval, shrunk by
    ``limit_ratio`` (locomotion semantics: Newton joint limits intersected with
    ``stance +- fk_joint_range``, then shrunk around the center). Rejects
    solutions parked against a joint stop -- reachable on paper, fragile in sim."""

    limit_ratio: float = 0.9
    """Allowed fraction of the effective joint interval."""


@configclass
class ReachRowsCfg:
    """Approach poses derived from accepted grasps: pad targets backed off along
    the achieved approach axis and re-solved (warm-seeded). in_air parents are
    excluded -- a floating nut with a non-grasping gripper is not a physical
    state. ``None`` on the pipeline cfg disables reach rows."""

    per_grasp: int = 1
    """Approach poses generated per accepted grasp."""

    standoff_range: tuple[float, float] = (0.03, 0.15)
    """Standoff distance [m] sampled per reach row along the approach axis."""

    clearance: float = 0.005
    """Min gripper<->nut clearance [m] (no contact intended)."""


@configclass
class FactoryRobotCfg:
    """The ROBOT side of the pipeline: who the robot is and how it is placed on
    each sampled candidate -- identity, the batched IK solve (with its avoidance
    objectives), the acceptance criteria, and the derived reach rows. The
    counterpart of the board/placement SAMPLER side, mirroring locomotion's
    kin + IK weights + objectives + criteria group."""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """The robot scene entity (USD resolved from the pipeline ``scene`` /
    :attr:`usd_path`)."""

    usd_path: str = ""
    """Robot USD path override. Empty resolves from the scene's robot entry; the
    production wiring patches it from ``env.scene["robot"].cfg.spawn.usd_path``
    (the terrain ``kin.usd_path`` pattern)."""

    default_joint_q: dict[str, float] | None = None
    """Joint-name -> stance value [rad or m] for the FK-library center and IK seed
    fallback. ``None`` resolves from the scene robot's ``init_state.joint_pos``
    (name patterns supported); the production wiring patches the live robot's
    default joint state."""

    ee_body_name: str = EndEffectorBodyCfg()  # type: ignore[assignment]
    """End-effector body (robot preset): the jaw-axis reference frame for pad
    derivation and the approach-axis readout."""

    finger_body_names: list[str] = FingerBodyNamesCfg()  # type: ignore[assignment]
    """Finger bodies carrying the pad contact points (robot preset; IK target links)."""

    gripper_body_names: list[str] = GripperBodyNamesCfg()  # type: ignore[assignment]
    """Robot bodies probed for the gripper-vs-held-asset checks (robot preset)."""

    solve: IKSolveCfg = IKSolveCfg()
    """The two-phase batched LM solve (avoidance objectives nested)."""

    criteria: list = field(
        default_factory=lambda: [
            CollisionCheckCfg(),
        ]
    )
    """Acceptance gates, applied as a waterfall after the solve. Membership
    controls which gates run; each carries its own probe budget and tolerance."""

    reach: ReachRowsCfg | None = ReachRowsCfg()
    """Reach/standoff row generation from accepted grasps (``None`` disables)."""


@configclass
class FactoryIKPipelineCfg:
    """Full offline factory Newton-IK reset-state pipeline configuration."""

    class_type: type | str = "{DIR}.pipeline:FactoryIKPipeline"
    """Pipeline implementation class."""

    scene: object | None = None
    """The RESOLVED scene cfg the assets come from: the production wiring assigns
    ``env.cfg.scene``; standalone tools assign a ``FactorySceneCfg`` variant via
    :func:`resolve_standalone`. Asset USDs, spawn scales, and init poses all
    resolve from here -- the pipeline holds no asset paths of its own. A ``None``
    placeholder (the terrain ``kin.usd_path=""`` pattern) survives env-cfg
    validation; the model fails loud if it is still unset at build time."""

    device: str = "cuda:0"
    """Warp/torch device for the model and batched solve (the production wiring
    patches ``env.device``)."""

    seed: int = 42
    """Torch RNG seed for the sampling stages (pair sampling, FK library, placements)."""

    obstacle_asset_names: list[str] = field(default_factory=lambda: ["table"])
    """Scene assets treated as STATIC collision obstacles, resolved from the
    variant's ``FactorySceneCfg`` entry (USD collider + init pose). Only rigid
    assets with a USD spawn qualify -- not every scene object is collidable
    (ground plane, lights, sensors, the robot itself). The nistboard and the fixed
    asset are NOT static: they form the per-sub-world posed assembly group
    (:attr:`PlacementSamplingCfg.board_pose_range`)."""

    board: BoardLibraryCfg = BoardLibraryCfg()
    """The fixed board+bolt configuration library (the world; terrain-grid analog)."""

    placement: PlacementSamplingCfg = PlacementSamplingCfg()
    """Nut placement sampling within the board configurations."""

    robot: FactoryRobotCfg = FactoryRobotCfg()
    """The robot and how it is placed on each candidate (identity, two-phase
    solve, acceptance criteria, reach rows)."""


def find_criterion(criteria: list, cls: type):
    """Return the first criterion cfg of type ``cls`` in ``criteria``, or ``None``.

    Membership in the criteria list is what enables a gate, so consumers resolve
    their cfg through this and skip the gate (and its probe buffers) when absent.
    """
    for c in criteria:
        if isinstance(c, cls):
            return c
    return None


def resolve_from_task(
    task: str = "Isaac-Factory-v0", agent: str = "rsl_rl_cfg_entry_point"
) -> FactoryResetStateTableCfg:  # noqa: F821
    """Resolve the PRODUCTION reset-state table cfg for standalone (no-env) tools.

    Mirrors ``terrain/scripts/validate_spawn_points.py``: registers the factory
    task, resolves the env cfg exactly the way the train script does (the
    ``presets=...`` tokens in ``sys.argv``), and reuses the env's scene and robot
    identity -- so validators and visualizers reflect training behaviour instead
    of hand-rolled constants. Safe to call before Kit launches.

    Returns the env's ``commands.reset_state.task_table`` cfg with
    ``pipeline_cfg.scene`` and ``pipeline_cfg.robot.usd_path`` populated (the robot
    USD is downloaded from Nucleus when needed). The robot stance resolves from
    ``scene.robot.init_state`` inside the model, as in the live wiring.
    """
    import importlib

    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    from isaaclab_tasks.utils.hydra import resolve_task_config

    importlib.import_module("isaaclab_tasks.core.multi_task.factory.config")
    env_cfg, _ = resolve_task_config(task, agent)
    table_cfg = env_cfg.commands.reset_state.task_table
    pcfg = table_cfg.pipeline_cfg
    pcfg.scene = env_cfg.scene
    usd_path = env_cfg.scene.robot.spawn.usd_path
    status = check_file_path(usd_path)
    if status == 0:
        raise FileNotFoundError(f"robot USD not found: {usd_path}")
    if status == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)
    pcfg.robot.usd_path = usd_path
    return table_cfg
