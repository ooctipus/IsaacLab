# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration blocks for the motion :class:`StateCommand` payload."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.utils.configclass import configclass

from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...data import MotionSampleGrid
from .motion_state_payload import MotionStatePayload
from .motion_task_table import TaskSamplingLaw, _task_sampling_law, build_motion_task_table

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ....kinematics import NewtonKinematics
    from ...config.sources import MotionSourceCfg
    from ..runtime import MotionRuntime
    from .motion_task_table import MotionFrameBuilder


@configclass
class MotionTaskTableCfg(StateCommandCfg.TaskTableCfg):
    """Direct source-to-table construction and descriptor-row policy."""

    class_type: Callable = build_motion_task_table
    """Callable that streams and binds the selected source table."""

    source: MotionSourceCfg = MISSING  # type: ignore[assignment]
    """Selected native source and its frozen train/evaluation artifacts."""

    source_artifact_root: str = ""
    """Deployment root containing selected source-motion artifacts."""

    reference_artifact_root: str = ""
    """Deployment root containing narrow external reference-only artifacts."""

    motion_split: Literal["train", "evaluation"] = "train"
    """Source split materialized into this table."""

    frame_builder_factory: Callable[[ManagerBasedRLEnv], MotionFrameBuilder] = MISSING
    """Build source-to-trajectory conversion from the live articulation."""

    reference_kinematics_factory: Callable[[ManagerBasedRLEnv], NewtonKinematics] = MISSING
    """Build the exact reference model used by this table's frame builder."""

    expert_sample_grid: MotionSampleGrid = MISSING
    """Sample clock used when the learner consumes this table as expert data."""

    task_row_mode: Literal["source_frames", "clip_time_ranges"] = MISSING
    """Controlled rule that derives selectable rows from valid source clips."""

    @property
    def task_sampling_law(self) -> TaskSamplingLaw:
        """Sampling law implied by :attr:`task_row_mode`."""
        return _task_sampling_law(self.task_row_mode)

    reset_sources: tuple[tuple[str, float], ...] = MISSING
    """Ordered reset-source names and sampling probabilities."""

    def __post_init__(self) -> None:
        """Reject incomplete direct table-construction inputs."""
        if not isinstance(self.source_artifact_root, str) or not isinstance(self.reference_artifact_root, str):
            raise TypeError("Motion deployment roots must be strings.")
        if self.motion_split not in ("train", "evaluation"):
            raise ValueError("motion_split must select train or evaluation.")
        if not callable(self.frame_builder_factory):
            raise TypeError("Motion frame-builder factory must be callable.")
        if not callable(self.reference_kinematics_factory):
            raise TypeError("Motion reference-kinematics factory must be callable.")
        if not isinstance(self.expert_sample_grid, MotionSampleGrid):
            raise TypeError("expert_sample_grid must be MotionSampleGrid.")
        if self.task_row_mode not in ("source_frames", "clip_time_ranges"):
            raise ValueError("task_row_mode must be 'source_frames' or 'clip_time_ranges'.")
        if not isinstance(self.reset_sources, tuple) or not self.reset_sources:
            raise ValueError("reset_sources must be a nonempty tuple of name/probability pairs.")
        names: list[str] = []
        probabilities: list[float] = []
        for source in self.reset_sources:
            if not isinstance(source, tuple) or len(source) != 2:
                raise ValueError("Each reset source must be one (name, probability) tuple.")
            name, probability = source
            if not isinstance(name, str) or not name:
                raise ValueError("Reset-source names must be nonempty strings.")
            if isinstance(probability, bool) or not isinstance(probability, int | float):
                raise ValueError("Reset-source probabilities must be real scalars.")
            names.append(name)
            probabilities.append(float(probability))
        if len(set(names)) != len(names) or any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("Reset sources require unique names and finite nonnegative probabilities.")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("Reset-source probabilities must sum to one.")


@configclass
class MotionStatePayloadCfg(StateCommandCfg.PayloadCfg):
    """Runtime motion-state, reset, reference, and evidence configuration."""

    @configclass
    class RawEvidenceCfg:
        """One named raw transition fact written through :meth:`record_step`."""

        name: str = MISSING  # type: ignore[assignment]
        """Stable evidence-channel name."""

        width: int = 1
        """Number of scalar values per environment."""

        unit: str = ""
        """SI unit label, or an empty string for a dimensionless fact."""

        anchor: str = MISSING  # type: ignore[assignment]
        """Declared transition anchor, such as ``transition_reached_physics``."""

    class_type: type = MotionStatePayload
    """Payload worker class."""

    robot_asset_name: str = MISSING  # type: ignore[assignment]
    """Scene articulation that receives decoded reset state."""

    transition_state_factory: Callable[[ManagerBasedRLEnv, MotionStatePayload], MotionRuntime] | str = MISSING
    """Build per-transition runtime state after the command payload exists."""

    reset_transform_factory: (
        Callable[
            [ManagerBasedRLEnv],
            Callable[[MotionStatePayload.ResetState, torch.Tensor, torch.Generator], MotionStatePayload.ResetState],
        ]
        | None
    ) = None
    """Optional live-environment factory for one runtime transform of decoded reset rows.

    The second argument contains sampled reset-source indices and the third owns all reset draws.
    """
    root_velocity_frame: Literal["link", "center_of_mass"] = "link"
    """Root frame whose world velocity receives decoded reset values."""

    step_fields: tuple[str, ...] = ()
    """Trajectory fields refreshed after each applied transition.

    Reset fields are inferred from the concrete table columns and refreshed only
    while binding an episode. Native forward/backward actors normally leave
    this empty because they consume live observations, not tracking targets.
    """

    command_fields: tuple[str, ...] = ()
    """Reference fields flattened into the generic policy command tensor.

    Every command field must also appear in :attr:`step_fields`.
    """

    episode_length_steps: int = MISSING  # type: ignore[assignment]
    """Episode horizon used to normalize the episode-relative phase."""

    history_fields: tuple[tuple[str, int], ...] = ()
    """Ordered names and widths retained as applied-transition history."""

    history_length: int = 0
    """Number of newest-first values retained per history field.

    Zero disables history and requires :attr:`history_fields` to be empty.
    """

    raw_evidence: tuple[RawEvidenceCfg, ...] = ()
    """Fixed named raw evidence layout filled by the pre-final step hook."""

    auxiliary_evidence: tuple[str, ...] = ()
    """Ordered raw evidence channels exported to the learner."""

    def __post_init__(self) -> None:
        """Reject a transition runtime that cannot be resolved at manager load."""
        if not callable(self.transition_state_factory) and not isinstance(self.transition_state_factory, str):
            raise TypeError("transition_state_factory must be callable or lazily resolvable.")
