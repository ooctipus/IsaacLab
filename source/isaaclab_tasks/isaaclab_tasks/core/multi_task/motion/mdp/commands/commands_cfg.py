# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable configuration schemas for the motion :class:`StateCommand` payload."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.utils.configclass import configclass

from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...data.frames import MotionFrameBuilder
from ...data.skeleton import MotionSkeleton
from ...data.source import MotionSourceCfg

if TYPE_CHECKING:
    from ....kinematics import NewtonKinematics


_STAGE_MODULE = "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table"


@configclass
class MotionCoordinateRouteCfg:
    """Family names selected by exact coordinate compatibility."""

    exact_family: str = "exact_coordinates"
    semantic_family: str = "semantic_sequence"

    def __post_init__(self) -> None:
        """Require two distinct named routes."""
        if not self.exact_family or not self.semantic_family or self.exact_family == self.semantic_family:
            raise ValueError("Motion exact and semantic family names must be distinct and nonempty.")


@configclass
class MotionExactCoordinatesGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Decode exact source coordinates and materialize robot frames."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_generate_exact_coordinates"


@configclass
class MotionSemanticTargetsGenerateCfg(StateCommandCfg.TaskTableCfg.GenerateTermCfg):
    """Decode source semantics and generate concrete robot landmark targets."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_generate_semantic_targets"


@configclass
class MotionLandmarkPositionObjectiveCfg(StateCommandCfg.TaskTableCfg.ObjectiveCfg):
    """Length-normalized world landmark-position objective."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_objective_landmark_position"
    weight: float = 1.0
    root_weight: float = 10.0


@configclass
class MotionLandmarkRotationObjectiveCfg(StateCommandCfg.TaskTableCfg.ObjectiveCfg):
    """World landmark-orientation objective."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_objective_landmark_rotation"
    weight: float = 1.0
    root_weight: float = 10.0
    canonicalize_error: bool = True


@configclass
class MotionSemanticSolveCfg(StateCommandCfg.TaskTableCfg.SolveCfg):
    """One GPU-batched semantic IK solve over all frames in a clip."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_solve_semantic_sequence"
    objectives: tuple[StateCommandCfg.TaskTableCfg.ObjectiveCfg, ...] = (
        MotionLandmarkPositionObjectiveCfg(),
        MotionLandmarkRotationObjectiveCfg(),
    )
    max_iterations: int = 30
    convergence_check_interval: int = 5
    projection_interval: int = 5
    lambda_initial: float = 0.1
    lambda_factor: float = 2.0
    lambda_min: float = 1.0e-5
    lambda_max: float = 1.0e10
    rho_min: float = 1.0e-3
    history_length: int = 10
    h0_scale: float = 1.0
    wolfe_c1: float = 1.0e-4
    wolfe_c2: float = 0.9
    support_landmark_atol_m: float = 2.0e-6

    def __post_init__(self) -> None:
        """Require one coherent projected Levenberg-Marquardt policy."""
        if self.max_iterations < 1 or self.projection_interval < 1 or self.convergence_check_interval < 1:
            raise ValueError("Semantic solve iteration counts must be positive.")
        if self.convergence_check_interval % self.projection_interval:
            raise ValueError("Motion convergence checks must follow a completed coordinate projection.")
        if not 0.0 < self.lambda_min <= self.lambda_initial <= self.lambda_max:
            raise ValueError("LM damping must satisfy 0 < lambda_min <= lambda_initial <= lambda_max.")
        if self.lambda_factor <= 1.0 or self.rho_min <= 0.0 or self.history_length < 1 or self.h0_scale <= 0.0:
            raise ValueError("LM factor, rho, history length, and initial inverse-Hessian scale must be positive.")
        if not 0.0 < self.wolfe_c1 < self.wolfe_c2 < 1.0 or self.support_landmark_atol_m <= 0.0:
            raise ValueError("Wolfe constants and support-landmark tolerance are invalid.")


@configclass
class MotionFrameFiniteCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require every materialized robot-frame value to be finite."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_criterion_frame_finite"


@configclass
class MotionObjectiveMeasureCriterionCfg(StateCommandCfg.TaskTableCfg.CriterionCfg):
    """Require one cached semantic objective measure to be finite and bounded."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_criterion_objective_measure"
    objective: Literal["landmark_position", "landmark_rotation"] = MISSING
    upper: float = MISSING

    def __post_init__(self) -> None:
        """Require an explicit finite positive residual bound."""
        if self.objective not in ("landmark_position", "landmark_rotation"):
            raise ValueError("Semantic objective criteria require a known objective.")
        if not isinstance(self.upper, (int, float)) or not math.isfinite(self.upper) or self.upper <= 0.0:
            raise ValueError("Semantic objective criteria require a finite positive upper bound.")


@configclass
class MotionClipSelectionCfg(StateCommandCfg.TaskTableCfg.SelectionCfg):
    """Retain accepted clips in deterministic source order."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_select_source_order"
    max_clips: int | None = None

    def __post_init__(self) -> None:
        """Require a positive optional coverage cap."""
        if self.max_clips is not None and self.max_clips < 1:
            raise ValueError("Motion max_clips must be positive when configured.")


@configclass
class MotionSemanticSegmentSelectionCfg(StateCommandCfg.TaskTableCfg.SelectionCfg):
    """Retain maximal position-valid semantic runs and cut discontinuous edges."""

    class_type: Callable | str = f"{_STAGE_MODULE}:motion_select_semantic_segments"
    max_branch_jump_rad: float = math.pi

    def __post_init__(self) -> None:
        """Require one finite positive branch-discontinuity threshold [rad]."""
        if not math.isfinite(self.max_branch_jump_rad) or self.max_branch_jump_rad <= 0.0:
            raise ValueError("Motion branch-jump threshold must be finite and positive [rad].")


@configclass
class MotionExactFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """Exact-coordinate generation, finite-output checks, and stable source ordering."""

    name: str = "exact_coordinates"
    generate: tuple[StateCommandCfg.TaskTableCfg.GenerateTermCfg, ...] = (MotionExactCoordinatesGenerateCfg(),)
    solve: StateCommandCfg.TaskTableCfg.SolveCfg | None = None
    criteria: tuple[StateCommandCfg.TaskTableCfg.CriterionCfg, ...] = (MotionFrameFiniteCriterionCfg(),)
    selection: StateCommandCfg.TaskTableCfg.SelectionCfg = MotionClipSelectionCfg()


@configclass
class MotionSemanticFamilyCfg(StateCommandCfg.TaskTableCfg.FamilyCfg):
    """Semantic target generation, one batched solve, checks, and ordering."""

    name: str = "semantic_sequence"
    generate: tuple[StateCommandCfg.TaskTableCfg.GenerateTermCfg, ...] = (MotionSemanticTargetsGenerateCfg(),)
    solve: StateCommandCfg.TaskTableCfg.SolveCfg | None = MotionSemanticSolveCfg()
    criteria: tuple[StateCommandCfg.TaskTableCfg.CriterionCfg, ...] = (
        MotionObjectiveMeasureCriterionCfg(objective="landmark_position", upper=0.15),
        MotionFrameFiniteCriterionCfg(),
    )
    selection: StateCommandCfg.TaskTableCfg.SelectionCfg = MotionSemanticSegmentSelectionCfg()


@configclass
class MotionTaskTableCfg(StateCommandCfg.TaskTableCfg):
    """Typed inputs used to build one source-to-robot motion table."""

    @configclass
    class TargetKinematicsCfg:
        """Coupled exact target builder and reference-kinematics factories."""

        frame_builder_factory: Callable[[MotionSkeleton, NewtonKinematics], MotionFrameBuilder] = MISSING
        reference_kinematics_factory: Callable[[str, str | torch.device], NewtonKinematics] = MISSING

    class_type: Callable | str = "{DIR}.motion_task_table:build_motion_task_table"
    """Callable that streams and binds the selected source table."""

    source: MotionSourceCfg = MISSING  # type: ignore[assignment]
    """Native motion source and its frozen train/evaluation artifacts."""

    target_kinematics: TargetKinematicsCfg = MISSING  # type: ignore[assignment]
    """Robot-axis target construction selected as one coherent value."""

    route: MotionCoordinateRouteCfg = MotionCoordinateRouteCfg()
    """Resolve exactly one family from source/robot coordinate compatibility."""

    families: tuple[StateCommandCfg.TaskTableCfg.FamilyCfg, ...] = (
        MotionExactFamilyCfg(),
        MotionSemanticFamilyCfg(),
    )
    """Visible exact and semantic family programs."""

    task_row_mode: Literal["source_frames", "clip_time_ranges"] = MISSING
    """Task-table row layout."""

    source_artifact_root: str = ""
    """Deployment root containing selected source-motion artifacts."""

    reference_artifact_root: str = ""
    """Deployment root containing narrow external reference-only artifacts."""

    motion_split: Literal["train", "evaluation"] = "train"
    """Source split materialized into this table."""

    def __post_init__(self) -> None:
        """Validate deployment roots and the selected split."""
        if not isinstance(self.source_artifact_root, str) or not isinstance(self.reference_artifact_root, str):
            raise TypeError("Motion deployment roots must be strings.")
        if self.motion_split not in ("train", "evaluation"):
            raise ValueError("motion_split must select train or evaluation.")
        names = tuple(family.name for family in self.families)
        if len(set(names)) != len(names) or set(names) != {self.route.exact_family, self.route.semantic_family}:
            raise ValueError("Motion families must define exactly the configured exact and semantic routes.")


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
