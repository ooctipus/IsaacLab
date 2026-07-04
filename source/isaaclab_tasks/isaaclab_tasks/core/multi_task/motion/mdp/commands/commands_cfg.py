# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable configuration schemas for the motion :class:`StateCommand` payload."""

from __future__ import annotations

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
    from isaaclab.assets import Articulation

    from ....kinematics import NewtonKinematics


@configclass
class MotionTaskTableCfg(StateCommandCfg.TaskTableCfg):
    """Typed inputs used to build one source-to-robot motion table."""

    class_type: Callable | str = "{DIR}.motion_task_table:build_motion_task_table"
    """Callable that streams and binds the selected source table."""

    source: MotionSourceCfg = MISSING  # type: ignore[assignment]
    """Native motion source and its frozen train/evaluation artifacts."""

    frame_builder_factory: Callable[[MotionSkeleton, NewtonKinematics, Articulation], MotionFrameBuilder] = MISSING
    """Factory that converts source clips into the selected robot coordinates."""

    reference_kinematics_factory: Callable[[str, str | torch.device], NewtonKinematics] = MISSING
    """Factory for exact reference kinematics used by the selected frame builder."""

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
