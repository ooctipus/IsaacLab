# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the simulator-free Factory reset-state builder.

The composition root declares independent board geometry and explicit task
families. Each family owns its generate, solve, criteria, and selection stages;
there is no hidden global placement mixture or empirical yield conversion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import Literal

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ...kinematics.ik_objectives.cfg import IKObjectiveBaseCfg
from ...mdp.commands.state_command import StateCommandCfg
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

    library_oversample: float = 4.0
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
    oversampled, collision-rejected against the table and complete default robot, and
    FPS-downsampled to :attr:`num_boards`; arm reachability is left to the IK
    funnel."""

    oversample: float = 10.0
    """Poses sampled per library slot before rejection + FPS (single shot, no
    resample loop)."""

    clear_tol: float = 0.0005
    """Reject a board group intersecting the table or any default-pose robot collider
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
class FactoryAssemblyPoseGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Generate held-asset poses along the declared assembly path."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_generate_assembly_pose"
    assembly_profile: object = FactoryAssemblyProfileCfg()  # type: ignore[assignment]
    align_offset: object = HeldAssetAlignOffsetCfg()  # type: ignore[assignment]
    assembly_bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "near_seated": (0.0, 0.33),
            "mid_insertion": (0.33, 0.85),
            "above_tip": (0.85, 1.6),
        }
    )


@configclass
class FactorySupportPoseGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Generate held-asset poses supported by the board."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_generate_support_pose"
    tag: str = "on_table"
    pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"x": (0.25, 0.6), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)}
    )
    table_height: float = 0.04


@configclass
class FactoryFreePoseGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Generate free-space held-asset poses."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_generate_free_pose"
    tag: str = "in_air"
    pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.15, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.015, 0.2),
            "roll": (-1.57, 1.57),
            "pitch": (-1.57, 1.57),
            "yaw": (-3.14, 3.14),
        }
    )


@configclass
class FactoryGraspTargetGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Generate antipodal held-asset contact targets."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_generate_grasp_targets"
    sampling: GraspSamplingCfg = GraspSamplingCfg()
    grasps_per_placement: int = 8


@configclass
class FactoryRobotSeedGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Choose robot IK seeds for generated grasp targets."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_generate_robot_seeds"
    ik_seeds_per_grasp: int = 4


@configclass
class FactoryIKSolveCfg(StateCommandCfg.TaskTableCfg.SolveCfg):
    """One batched LM solve over per-fingertip targets.

    Independent criteria certify target accuracy, limits, and collision geometry.
    """

    class_type: Callable | str = "{DIR}.task_table_builder:factory_solve_ik"
    max_iterations: int = 250
    """Maximum continuous LM iterations for the target solve."""

    objectives: tuple[IKObjectiveBaseCfg, ...] = MISSING
    """Complete flat objective tuple for the numerical solve."""


@configclass
class FactoryApproachTargetGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Offset grasp targets along the seed end-effector approach axis."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_generate_approach_targets"
    standoff_range: tuple[float, float] = (0.03, 0.15)
    """Approach standoff range [m]."""
    clearance: float = 0.005
    """Required gripper-to-held clearance [m]."""


@configclass
class CollisionCheckCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """No unintended penetration between ANY participating bodies.

    One gate, four tests under the hood: robot links vs the obstacles and the
    posed board + bolt (point signed distance PLUS exact edge-vs-mesh crossing
    tests -- points alone miss the ~4 mm board slicing between them), gripper vs
    the held asset (symmetric), held asset vs the obstacles, and robot
    link-vs-link. Generated family facts explicitly identify pad contact,
    fixed-asset contact, and board support; design-mounted link clusters are
    excluded by :attr:`adjacency_hops`.
    """

    class_type: Callable | str = "{DIR}.task_table_builder:factory_collision_criterion"
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
class JointWithinLimitCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Criterion: every arm coord inside its effective interval, shrunk by
    ``limit_ratio`` (locomotion semantics: Newton joint limits intersected with
    ``stance +- fk_joint_range``, then shrunk around the center). Rejects
    solutions parked against a joint stop, which are numerically valid but fragile in simulation."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_joint_limit_criterion"
    limit_ratio: float = 0.9
    """Allowed fraction of the effective joint interval."""


@configclass
class FactoryTargetErrorCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Accept candidates whose solved fingertip target error meets the solve tolerance."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_target_error_criterion"
    max_error_m: float = MISSING
    """Maximum accepted per-fingertip target error [m]."""


@configclass
class FactoryHeldPoseBoundsCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Accept generated held-object positions inside declared world-local bounds."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_held_pose_bounds_criterion"
    bounds: dict[str, tuple[float, float]] = MISSING


@configclass
class FactoryFpsSelectionCfg(StateCommandCfg.TaskTableCfg.SelectionCfg):
    """Select an exact quota per board in one family-specific feature space."""

    class_type: Callable | str = "{DIR}.task_table_builder:factory_fps_selection"
    position_frame: Literal["fixed_asset", "world"] = "world"
    position_axes: tuple[int, ...] = (0, 1, 2)
    position_weight: float = 1.0
    approach_weight: float = 0.15
    tag_weight: float = 0.2


@configclass
class FactoryFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """One explicit Factory placement, solve, acceptance, and selection route."""

    fraction: float = 0.0
    candidate_oversample: float = 80.0


@configclass
class FactoryRobotCfg:
    """Robot identity needed by every Factory task family."""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """Scene articulation providing the USD and default joint state directly."""

    ee_body_name: str = EndEffectorBodyCfg()  # type: ignore[assignment]
    """End-effector body (robot preset): the jaw-axis reference frame for pad
    derivation and the approach-axis readout."""

    finger_body_names: list[str] = FingerBodyNamesCfg()  # type: ignore[assignment]
    """Finger bodies carrying the pad contact points (robot preset; IK target links)."""

    gripper_body_names: list[str] = GripperBodyNamesCfg()  # type: ignore[assignment]
    """Robot bodies probed for the gripper-vs-held-asset checks (robot preset)."""


@configclass
class FactoryGeometryCfg:
    """Shared Factory mechanics and geometry used by declared task families."""

    obstacle_asset_names: list[str] = field(default_factory=lambda: ["table"])
    """Scene assets treated as STATIC collision obstacles, resolved from the
    variant's ``FactorySceneCfg`` entry (USD collider + init pose). Only rigid
    assets with a USD spawn qualify -- not every scene object is collidable
    (ground plane, lights, sensors, the robot itself). The nistboard and the fixed
    asset are not static: they form the per-board posed assembly group."""

    board: BoardLibraryCfg = BoardLibraryCfg()
    """The fixed board+bolt configuration library (the world; terrain-grid analog)."""

    held_asset_cfg: SceneEntityCfg = SceneEntityCfg("held_asset")
    """Held rigid object whose pose and contact geometry are generated."""

    robot: FactoryRobotCfg = FactoryRobotCfg()
    """Robot identity shared by all families."""
