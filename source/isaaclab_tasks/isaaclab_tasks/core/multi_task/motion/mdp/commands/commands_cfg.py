# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable configuration schemas for the motion :class:`StateCommand` payload."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import string_to_callable

from ....kinematics.ik_objectives.cfg import (
    IKObjectiveBaseCfg,
    IKObjectiveJointDefaultCfg,
    IKObjectiveJointPinCfg,
    IKObjectiveMeshCollisionCfg,
    IKObjectiveMeshNonpenetrationCfg,
)
from ....kinematics.newton_kinematics_cfg import NewtonKinematicsBuildCfg
from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...data.frames import (
    MotionSourceProjection,
    MotionSourceProjectionAnalytic,
    MotionSourceProjectionExact,
    MotionSourceProjectionTrajectory,
)
from ...data.skeleton import MotionSkeleton
from ...data.source import MotionClipSource, MotionSourceCfg
from ...identity import validate_sha256
from ...robots.target import MotionFrameTarget

if TYPE_CHECKING:
    from ....kinematics import NewtonKinematics
    from ....mdp.commands.state_command.task_table_view import TaskTableView


_BUILDER_MODULE = "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder"
_TRAJECTORY_MODULE = "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory"


@configclass
class MotionExactCoordinatesGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Decode exact source coordinates and materialize robot frames."""

    class_type: Callable | str = f"{_BUILDER_MODULE}:motion_generate_exact_coordinates"
    source_projection_type: type[MotionSourceProjection] = MotionSourceProjectionExact


@configclass
class MotionAnalyticCoordinatesGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Materialize a direct analytic source-to-robot coordinate map."""

    class_type: Callable | str = f"{_BUILDER_MODULE}:motion_generate_analytic_coordinates"
    source_projection_type: type[MotionSourceProjection] = MotionSourceProjectionAnalytic


@configclass
class MotionSourceEvidenceGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Decode calibrated source evidence and a limit-valid robot initializer."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_generate_source_evidence"
    source_projection_type: type[MotionSourceProjection] = MotionSourceProjectionTrajectory


@configclass
class MotionSourceGlobalPositionObjectiveCfg(StateCommandCfg.TaskTableCfg.ObjectiveCfg):
    """Length-normalized world landmark-position objective."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_objective_source_global_position"
    weight: float = 1.0
    root_weight: float = 10.0


@configclass
class MotionSourceRotationObjectiveCfg(StateCommandCfg.TaskTableCfg.ObjectiveCfg):
    """Target-owned source-landmark orientation objectives."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_objective_source_rotation"


@configclass
class MotionSourceDirectionPointObjectiveCfg(StateCommandCfg.TaskTableCfg.ObjectiveCfg):
    """Auxiliary distal-point geometry at quarter precision relative to primary body origins."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_objective_source_direction_point"
    weight: float = 0.25


@configclass
class MotionContactObjectiveCfg(StateCommandCfg.TaskTableCfg.ObjectiveCfg):
    """Normalized support gap, yaw-free upright, and planted support-point target errors."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_objective_contact"


@configclass
class MotionTrajectorySolveCfg(StateCommandCfg.TaskTableCfg.SolveCfg):
    """One memory-bounded, contact-aware whole-clip trajectory solve."""

    @configclass
    class ContactCfg:
        """Source-foot soft hysteresis and strict planted-state certification policy."""

        enter_height_m: float = 0.03
        exit_height_m: float = 0.06
        enter_speed_mps: float = 0.15
        exit_speed_mps: float = 0.30
        persistence_seconds: float = 0.08
        confidence_window_seconds: float = 5.0 / 60.0
        """Centered confidence window [s]; nearest odd sample count with ties resolved upward."""
        point_tolerance_m: float = 0.01
        """Patch-RMS planted-point target tolerance [m]."""

    @configclass
    class DynamicsCfg:
        """Post-selection inverse-dynamics diagnostic policy.

        ``friction_coefficient`` is the nominal ground coefficient used only by
        the corpus diagnostic. Runtime friction randomization may differ. The
        resulting values are evidence columns, never an acceptance gate.
        """

        friction_coefficient: float = 0.7
        iterations: int = 96
        effort_weight: float = 1.0
        force_regularization: float = 1.0e-6

        def __post_init__(self) -> None:
            """Require one finite projected inverse-dynamics policy."""
            scalars = (self.friction_coefficient, self.effort_weight, self.force_regularization)
            if self.iterations < 1 or any(not math.isfinite(value) or value < 0.0 for value in scalars):
                raise ValueError("Trajectory dynamics values must be finite and nonnegative with positive iterations.")

    @configclass
    class AcceptanceCfg:
        """IsaacLab-owned hard source-before-contact publication policy."""

        @configclass
        class SourceCfg:
            """Target-owned source-fidelity maxima required for phase commit and publication."""

            required_position_upper_m: float = 0.020
            required_distal_position_upper_m: float = 0.030
            required_distal_direction_upper_rad: float = 0.100
            root_rotation_upper_rad: float = 0.100

            def __post_init__(self) -> None:
                """Require finite positive source-fidelity bounds."""
                values = (
                    self.required_position_upper_m,
                    self.required_distal_position_upper_m,
                    self.required_distal_direction_upper_rad,
                    self.root_rotation_upper_rad,
                )
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0 for value in values
                ):
                    raise ValueError("Source-fidelity bounds must be finite and positive.")
                angular = (self.required_distal_direction_upper_rad, self.root_rotation_upper_rad)
                if any(value >= math.pi for value in angular):
                    raise ValueError("Source-fidelity angular bounds must be below pi [rad].")

        @configclass
        class ContactCfg:
            """Applicable-contact maxima and explicit corpus contact requirement."""

            gap_upper_m: float = 0.010
            tilt_upper_rad: float = 0.100
            slip_speed_upper_mps: float = 0.050
            cumulative_drift_upper_m: float = 0.020
            require_any_stable_contact: bool = True

            def __post_init__(self) -> None:
                """Require finite positive contact bounds and an explicit corpus policy."""
                values = (
                    self.gap_upper_m,
                    self.tilt_upper_rad,
                    self.slip_speed_upper_mps,
                    self.cumulative_drift_upper_m,
                )
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0 for value in values
                ):
                    raise ValueError("Contact publication bounds must be finite and positive.")
                if self.tilt_upper_rad >= 0.5 * math.pi:
                    raise ValueError("Contact tilt publication bounds must be below a right angle [rad].")
                if type(self.require_any_stable_contact) is not bool:
                    raise TypeError("require_any_stable_contact must be a bool.")

        source: SourceCfg = SourceCfg()
        contact: ContactCfg = ContactCfg()

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_solve_trajectory"
    convergence_tolerance: float = 1.0e-6
    """Maximum predicted descent per active objective scalar at a convergence check."""
    convergence_check_interval: int = 1
    """Outer iterations between model-stationarity checks and host completion polls."""
    max_iterations: int = 200
    """Maximum nonlinear iterations for each source and contact-geometry phase."""
    objectives: tuple[StateCommandCfg.TaskTableCfg.ObjectiveCfg | IKObjectiveBaseCfg, ...] = (
        MotionSourceGlobalPositionObjectiveCfg(),
        MotionSourceRotationObjectiveCfg(),
        MotionSourceDirectionPointObjectiveCfg(),
        MotionContactObjectiveCfg(),
        IKObjectiveJointDefaultCfg(weight=1.0),
        IKObjectiveJointPinCfg(weight=1.0),
        IKObjectiveMeshCollisionCfg(weight=5.0, margin=0.03, n_samples=4),
        IKObjectiveMeshNonpenetrationCfg(tolerance_m=0.002, maximum_penetration_m=0.0, n_samples=4),
    )
    contact: ContactCfg = ContactCfg()
    dynamics: DynamicsCfg = DynamicsCfg()
    acceptance: AcceptanceCfg = AcceptanceCfg()
    source_position_velocity_weight: float = 1.0e-4
    source_position_acceleration_weight: float = 1.0e-8
    source_rotation_velocity_weight: float = 1.0e-4
    source_rotation_acceleration_weight: float = 1.0e-8
    joint_default_position_weight: float = 2.5e-3
    """Position precision on the raw default-joint coordinate error [rad⁻²]."""
    joint_temporal_velocity_weight: float = 1.0e-4
    """Velocity precision on actual joint coordinates and contact reference deviations [s²]."""
    joint_temporal_acceleration_weight: float = 1.0e-8
    """Acceleration precision on actual joint coordinates and contact reference deviations [s⁴]."""
    joint_temporal_jerk_weight: float = 1.0e-8
    """Jerk precision on actual joint coordinates and contact reference deviations [s⁶]."""
    damping: float = 1.0e-4
    krylov_max_iterations: int = 128
    """Maximum PCG/MINRES iterations for each nonlinear trajectory step."""
    krylov_relative_tolerance: float = 1.0e-4
    """Scale-free preconditioned PCG/MINRES residual tolerance."""
    kkt_relative_tolerance: float = 1.0e-4
    """Maximum relative affine KKT-root correction and physical complementarity."""

    def __post_init__(self) -> None:
        """Require one finite positive whole-trajectory policy."""
        scalars = (
            self.contact.enter_height_m,
            self.contact.exit_height_m,
            self.contact.enter_speed_mps,
            self.contact.exit_speed_mps,
            self.contact.persistence_seconds,
            self.contact.confidence_window_seconds,
            self.contact.point_tolerance_m,
            self.source_position_velocity_weight,
            self.source_position_acceleration_weight,
            self.source_rotation_velocity_weight,
            self.source_rotation_acceleration_weight,
            self.joint_default_position_weight,
            self.joint_temporal_velocity_weight,
            self.joint_temporal_acceleration_weight,
            self.joint_temporal_jerk_weight,
            self.damping,
            self.krylov_relative_tolerance,
            self.kkt_relative_tolerance,
        )
        coordinate_objectives = tuple(
            objective
            for objective in self.objectives
            if type(objective) in (IKObjectiveJointDefaultCfg, IKObjectiveJointPinCfg)
        )
        if {type(objective) for objective in coordinate_objectives} != {
            IKObjectiveJointDefaultCfg,
            IKObjectiveJointPinCfg,
        } or any(objective.weight != 1.0 for objective in coordinate_objectives):
            raise ValueError(
                "Motion joint-default and joint-reference residual channels must each be declared once at unit weight."
            )
        if self.convergence_tolerance is None or (
            not math.isfinite(self.convergence_tolerance) or self.convergence_tolerance < 0.0
        ):
            raise ValueError("Trajectory convergence tolerance must be finite and nonnegative.")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("Trajectory max_iterations must be a positive integer.")
        if type(self.convergence_check_interval) is not int or self.convergence_check_interval < 1:
            raise ValueError("Trajectory convergence check interval must be a positive integer.")
        if type(self.krylov_max_iterations) is not int or self.krylov_max_iterations < 1:
            raise ValueError("Trajectory Krylov max iterations must be a positive integer.")
        if any(not math.isfinite(value) or value < 0.0 for value in scalars):
            raise ValueError("Trajectory contact, temporal, and damping values must be finite and nonnegative.")
        if not (
            self.contact.enter_height_m <= self.contact.exit_height_m
            and self.contact.enter_speed_mps <= self.contact.exit_speed_mps
            and self.contact.persistence_seconds > 0.0
            and self.contact.point_tolerance_m > 0.0
            and self.damping > 0.0
            and 0.0 < self.krylov_relative_tolerance < 1.0
            and 0.0 < self.kkt_relative_tolerance < 1.0
        ):
            raise ValueError("Trajectory contact policies, scales, damping, or linear controls are invalid.")


@configclass
class MotionTargetCoordinateCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require finite coordinates, a normalized root quaternion, and finite target FK."""

    class_type: Callable | str = f"{_BUILDER_MODULE}:motion_criterion_target_coordinates"


@configclass
class MotionTargetCoordinateLimitsCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require target joint positions and stored velocities within declared robot limits."""

    class_type: Callable | str = f"{_BUILDER_MODULE}:motion_criterion_target_coordinate_limits"


@configclass
class MotionConstraintGeometryFeasibleCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require every constrained trajectory iteration to admit a geometrically feasible step."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_criterion_constraint_geometry_feasible"


@configclass
class MotionInnerSolveConvergedCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require every trajectory iteration's inner solve to converge."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_criterion_inner_solve_converged"


@configclass
class MotionRequiredRefinementConvergedCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require every nonlinear phase to converge when the aligned seed needed refinement."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_criterion_required_refinement_converged"


@configclass
class MotionSourceFidelityCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Bound every solver-independent source-fidelity maximum."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_criterion_source_fidelity"


@configclass
class MotionContactCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Bound applicable source-stable contact and preserve explicit N/A rows."""

    class_type: Callable | str = f"{_TRAJECTORY_MODULE}:motion_criterion_contact"


@configclass
class MotionGroundPenetrationCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Bound route-independent target collision penetration below flat ground."""

    class_type: Callable | str = f"{_BUILDER_MODULE}:motion_criterion_ground_penetration"
    upper_m: float = 0.002

    def __post_init__(self) -> None:
        """Require one finite positive penetration bound [m]."""
        if not isinstance(self.upper_m, (int, float)) or not math.isfinite(self.upper_m) or self.upper_m <= 0.0:
            raise ValueError("Ground-penetration bounds must be finite and positive [m].")


@configclass
class MotionExactFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """Preserve exact reference coordinates after intrinsic target validation."""

    name: str = "exact"
    generate: tuple[StateCommandCfg.TaskTableCfg.GenerateTermCfg, ...] = (MotionExactCoordinatesGenerateCfg(),)
    solve: StateCommandCfg.TaskTableCfg.SolveCfg | None = None
    criteria: tuple[StateCommandCfg.TaskTableCfg.CriterionCfg, ...] = (MotionTargetCoordinateCriterionCfg(),)
    selection: StateCommandCfg.TaskTableCfg.SelectionCfg | None = None


@configclass
class MotionAnalyticFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """Preserve reference-backed analytic coordinates after intrinsic target validation."""

    name: str = "analytic"
    generate: tuple[StateCommandCfg.TaskTableCfg.GenerateTermCfg, ...] = (MotionAnalyticCoordinatesGenerateCfg(),)
    solve: StateCommandCfg.TaskTableCfg.SolveCfg | None = None
    criteria: tuple[StateCommandCfg.TaskTableCfg.CriterionCfg, ...] = (MotionTargetCoordinateCriterionCfg(),)
    selection: StateCommandCfg.TaskTableCfg.SelectionCfg | None = None


@configclass
class MotionTrajectoryFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """Generate calibrated source evidence and solve accepted whole trajectories."""

    name: str = "trajectory"
    generate: tuple[StateCommandCfg.TaskTableCfg.GenerateTermCfg, ...] = (MotionSourceEvidenceGenerateCfg(),)
    solve: StateCommandCfg.TaskTableCfg.SolveCfg | None = MotionTrajectorySolveCfg()
    criteria: tuple[StateCommandCfg.TaskTableCfg.CriterionCfg, ...] = (
        MotionConstraintGeometryFeasibleCriterionCfg(),
        MotionRequiredRefinementConvergedCriterionCfg(),
        MotionSourceFidelityCriterionCfg(),
        MotionContactCriterionCfg(),
        MotionTargetCoordinateCriterionCfg(),
        MotionTargetCoordinateLimitsCriterionCfg(),
        MotionGroundPenetrationCriterionCfg(),
    )
    selection: StateCommandCfg.TaskTableCfg.SelectionCfg | None = None


@configclass
class MotionTaskTableCfg(StateCommandCfg.TaskTableCfg):
    """Typed inputs used to build one source-to-robot motion table."""

    @configclass
    class ContactChannelCfg:
        """One canonical source contact channel formed from semantic probes."""

        name: str = MISSING
        source_probe_roles: tuple[str, ...] = MISSING

        def __post_init__(self) -> None:
            """Require one named channel with at least two unique source probes."""
            if (
                not self.name
                or len(self.source_probe_roles) < 2
                or len(set(self.source_probe_roles)) != len(self.source_probe_roles)
                or any(not role for role in self.source_probe_roles)
            ):
                raise ValueError("Contact channels require a name and at least two unique source probe roles.")

    @configclass
    class TargetKinematicsCfg:
        """Target semantics, source projection, and scene-owned kinematics policy."""

        @configclass
        class ContactPatchCfg:
            """One robot collider patch driven by a canonical source contact channel."""

            channel: str = MISSING
            body_name: str = MISSING
            points_per_body: int = 3
            height_band_m: float = 0.005

            def __post_init__(self) -> None:
                """Require one named three-point robot contact patch."""
                if (
                    not self.channel
                    or not self.body_name
                    or self.points_per_body != 3
                    or not math.isfinite(self.height_band_m)
                    or self.height_band_m < 0.0
                ):
                    raise ValueError("Contact patches require channel/body names and exactly three low points.")

        @configclass
        class CalibrationCfg:
            """One target-owned calibration artifact."""

            artifact: str = MISSING
            """Target-root-relative calibration file."""

            artifact_sha256: str = MISSING
            """Expected lowercase SHA-256 digest of :attr:`artifact`."""

            def __post_init__(self) -> None:
                """Require one relative path and lowercase SHA-256 digest."""
                artifact = Path(self.artifact)
                if not self.artifact or artifact.is_absolute() or ".." in artifact.parts:
                    raise ValueError("Target calibration artifacts must be target-root-relative.")
                validate_sha256("target calibration artifact_sha256", self.artifact_sha256)

        class Factory(Protocol):
            """Construct target frame semantics from scene-owned robot mechanics."""

            def __call__(
                self,
                reference: NewtonKinematics,
                contact_patches: tuple[MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg, ...],
                *,
                calibration_artifact_root: str,
                calibration: MotionTaskTableCfg.TargetKinematicsCfg.CalibrationCfg | None,
            ) -> MotionFrameTarget:
                """Build one target frame map and optional calibrated conversion."""

        target_factory: Factory = MISSING
        source_projection_factory: Callable[
            [
                MotionSkeleton,
                MotionFrameTarget,
                MotionClipSource,
                tuple[MotionTaskTableCfg.ContactChannelCfg, ...],
                torch.Tensor,
            ],
            MotionSourceProjection,
        ] = MISSING
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        kinematics: NewtonKinematicsBuildCfg = NewtonKinematicsBuildCfg()
        contact_patches: tuple[ContactPatchCfg, ...] = MISSING
        physics_types: tuple[type, ...] = MISSING
        supports_physical_evidence: bool = False
        supports_randomization: bool = False
        calibration: CalibrationCfg | None = None
        """Optional robot-owned calibration used by direct source projections."""

        def __post_init__(self) -> None:
            """Require unique patches and an explicit live-physics contract."""
            channels = tuple(patch.channel for patch in self.contact_patches)
            bodies = tuple(patch.body_name for patch in self.contact_patches)
            if not channels or len(set(channels)) != len(channels) or len(set(bodies)) != len(bodies):
                raise ValueError("Target contact patches require unique channels and robot bodies.")
            if not self.physics_types or any(not isinstance(physics_type, type) for physics_type in self.physics_types):
                raise ValueError("Target kinematics require at least one concrete physics type.")

    class_type: Callable | str = "{DIR}.motion_task_table:build_motion_task_table"
    """Callable that streams and binds the selected source table."""

    source: MotionSourceCfg = MISSING  # type: ignore[assignment]
    """Native motion source and its frozen train/evaluation artifacts."""

    contact_channels: tuple[ContactChannelCfg, ...] = MISSING
    """Canonical source contact channels shared by every source skeleton."""

    target_kinematics: TargetKinematicsCfg = MISSING  # type: ignore[assignment]
    """Robot-axis target construction selected as one coherent value."""

    families: tuple[StateCommandCfg.TaskTableCfg.FamilyCfg, ...] = (
        MotionExactFamilyCfg(),
        MotionAnalyticFamilyCfg(),
        MotionTrajectoryFamilyCfg(),
    )
    """Visible exact, analytic, and trajectory family programs."""

    task_row_mode: Literal["source_frames", "clip_time_ranges"] = MISSING
    """Task-table row layout."""

    source_artifact_root: str = ""
    """Deployment root containing selected source-motion artifacts."""

    target_artifact_root: str = ""
    """Deployment root containing selected robot-calibration artifacts."""

    motion_split: Literal["train", "evaluation"] = "train"
    """Source split materialized into this table."""

    def build_inspection_view(
        self,
        command_cfg: StateCommandCfg,
        scene_cfg: object,
        device: str,
        *,
        sequence_limit: int,
    ) -> TaskTableView:
        """Build a simulator-free view that retains accepted and rejected candidate evidence."""
        return string_to_callable(f"{_BUILDER_MODULE}:build_motion_task_table_inspection")(
            command_cfg, scene_cfg, device, sequence_limit=sequence_limit
        )

    def __post_init__(self) -> None:
        """Validate deployment roots and the selected split."""
        if not isinstance(self.source_artifact_root, str):
            raise TypeError("Motion source artifact root must be a string.")
        if not isinstance(self.target_artifact_root, str):
            raise TypeError("Motion target artifact root must be a string.")
        if self.motion_split not in ("train", "evaluation"):
            raise ValueError("motion_split must select train or evaluation.")
        names = tuple(family.name for family in self.families)
        channel_names = tuple(channel.name for channel in self.contact_channels)
        if not channel_names or len(set(channel_names)) != len(channel_names):
            raise ValueError("Source contact channels require unique names.")
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Motion family names must be nonempty and unique.")
        projection_types = tuple(
            getattr(family.generate[0], "source_projection_type", None) if family.generate else None
            for family in self.families
        )
        route_types = (
            MotionSourceProjectionExact,
            MotionSourceProjectionAnalytic,
            MotionSourceProjectionTrajectory,
        )
        if len(projection_types) != len(route_types) or set(projection_types) != set(route_types):
            raise ValueError("Motion families must declare exactly one generate term for each source projection type.")


@configclass
class MotionStatePayloadCfg(StateCommandCfg.PayloadCfg):
    """Motion descriptor and simulator-reset configuration."""

    class_type: type | str = "{DIR}.motion_state_payload:MotionStatePayload"
    """Payload worker class."""

    robot_asset_name: str = MISSING  # type: ignore[assignment]
    """Scene articulation that receives decoded reset state."""

    reset_transform_factory: Callable[..., object] = MISSING
    """Factory for robot-specific reset-state transformations."""

    reset_transform_binds: dict[str, str] = {}
    """Constructor keyword expressions resolved against ``env`` and ``payload``."""

    reset_transform_params: dict[str, object] = {}
    """Arguments passed to :attr:`reset_transform_factory`."""

    root_velocity_frame: Literal["link", "center_of_mass"] = MISSING
    """Root frame receiving reset linear and angular velocity."""

    reset_sources: tuple[tuple[str, float], ...] = MISSING
    """Named reset sources and their sampling probabilities."""
