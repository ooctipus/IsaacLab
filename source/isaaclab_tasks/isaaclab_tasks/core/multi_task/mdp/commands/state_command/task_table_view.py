# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free task-table evidence and kinematic views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from .reset_state_bank import ResetStateBank

if TYPE_CHECKING:
    import newton


def _check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Check one fixed-shape immutable tensor field."""
    if value.dtype is not dtype or tuple(value.shape) != shape or not value.is_contiguous() or value.requires_grad:
        raise ValueError(f"{name} must be contiguous detached {dtype} with shape {shape}.")


@dataclass(frozen=True, slots=True)
class TaskTableSequenceIndex:
    """Compact physical-state addressing for static tuples or timed sequences."""

    offsets: torch.Tensor
    """Prefix offsets into flattened frames, shape [sequence_count + 1], int64."""

    state_indices: torch.Tensor | None = None
    """Optional reset-state row per flattened frame, shape [frame_count], int64.

    ``None`` means flattened frames address reset-state rows directly, so a
    contiguous motion corpus needs no materialized identity tensor.
    """

    frame_dt: torch.Tensor | None = None
    """Optional sample period [s] per sequence, shape [sequence_count], float32.

    ``None`` marks a logical tuple, such as a Position or Factory spawn/target
    pair, rather than inventing temporal meaning for its two frames.
    """

    def __post_init__(self) -> None:
        """Check the flattened sequence layout."""
        if self.offsets.ndim != 1 or self.offsets.dtype is not torch.int64 or not self.offsets.is_contiguous():
            raise ValueError("Task-table sequence offsets must be contiguous int64 [sequence_count + 1].")
        if (
            self.offsets.numel() < 2
            or int(self.offsets[0]) != 0
            or not bool(torch.all(self.offsets[1:] > self.offsets[:-1]))
        ):
            raise ValueError("Task-table sequence offsets must start at zero and increase strictly.")
        if self.state_indices is not None:
            _check_tensor("Task-table state indices", self.state_indices, (self.frame_count,), torch.int64)
            if self.state_indices.device != self.offsets.device:
                raise ValueError("Task-table state indices and sequence offsets must share one device.")
        if self.frame_dt is not None:
            _check_tensor("Task-table frame periods", self.frame_dt, (self.sequence_count,), torch.float32)
            if self.frame_dt.device != self.offsets.device or not bool(torch.all(self.frame_dt > 0.0)):
                raise ValueError("Task-table frame periods must be positive and share the sequence-index device.")

    @property
    def sequence_count(self) -> int:
        """Number of indexed task sequences."""
        return self.offsets.numel() - 1

    @property
    def frame_count(self) -> int:
        """Number of frames on the flattened sequence axis."""
        return int(self.offsets[-1])

    @property
    def is_timed(self) -> bool:
        """Whether frames have declared physical sample periods."""
        return self.frame_dt is not None

    def state_rows(self, sequence_indices: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        """Map matching sequence and local-frame indices to reset-state rows."""
        if (
            sequence_indices.shape != frame_indices.shape
            or sequence_indices.ndim != 1
            or sequence_indices.dtype is not torch.int64
            or frame_indices.dtype is not torch.int64
            or sequence_indices.device != self.offsets.device
            or frame_indices.device != self.offsets.device
        ):
            raise ValueError("Sequence and local-frame indices must be matching int64 vectors on the index device.")
        if bool(torch.any((sequence_indices < 0) | (sequence_indices >= self.sequence_count))):
            raise IndexError("Task-table sequence index is out of range.")
        frame_counts = self.offsets[sequence_indices + 1] - self.offsets[sequence_indices]
        if bool(torch.any((frame_indices < 0) | (frame_indices >= frame_counts))):
            raise IndexError("Task-table local-frame index is out of range for its sequence.")
        flat_indices = self.offsets[sequence_indices] + frame_indices
        return flat_indices if self.state_indices is None else self.state_indices[flat_indices]


@dataclass(frozen=True, slots=True)
class TaskTableKinematicView:
    """Shared and repeatable Newton geometry with a canonical state-to-coordinate map."""

    model_builder_state: newton.ModelBuilder
    """Resolved one-state Newton builder repeated for every displayed state."""

    joint_q_default: torch.Tensor
    """Default Newton generalized coordinates, shape [coordinate_count], float32."""

    root_entity_names: tuple[str, ...]
    """Canonical reset-state entities represented by free-root coordinates."""

    root_state_indices: torch.Tensor
    """Entity-axis indices into :attr:`ResetStateBank.root_pose`, shape [root_count], int64."""

    root_q_indices: torch.Tensor
    """Newton xyzw free-root coordinate indices, shape [root_count, 7], int64."""

    joint_coordinate_names: tuple[tuple[str, str], ...]
    """Canonical ``(entity_name, joint_name)`` identities represented in Newton."""

    joint_state_indices: torch.Tensor
    """Indices into :attr:`ResetStateBank.joint_position`, shape [joint_count], int64."""

    joint_q_indices: torch.Tensor
    """Newton coordinate indices for canonical joint columns, shape [joint_count], int64."""

    model_builder_shared: newton.ModelBuilder | None = None
    """Optional global Newton geometry added once outside every repeated world."""

    world_spacing: tuple[float, float, float] = (3.0, 3.0, 0.0)
    """Default Viser spacing [m] between repeated state worlds."""

    def __post_init__(self) -> None:
        """Check one direct state-to-coordinate map."""
        if len(self.world_spacing) != 3 or not all(math.isfinite(value) for value in self.world_spacing):
            raise ValueError("Task-table world spacing must contain three finite distances [m].")
        if self.joint_q_default.ndim != 1 or self.joint_q_default.dtype is not torch.float32:
            raise ValueError("Default Newton generalized coordinates must be float32 [coordinate_count].")
        _check_tensor("Root state indices", self.root_state_indices, (len(self.root_entity_names),), torch.int64)
        _check_tensor("Root q indices", self.root_q_indices, (len(self.root_entity_names), 7), torch.int64)
        joint_count = len(self.joint_coordinate_names)
        _check_tensor("Joint state indices", self.joint_state_indices, (joint_count,), torch.int64)
        _check_tensor("Joint q indices", self.joint_q_indices, (joint_count,), torch.int64)
        mappings = (self.root_state_indices, self.root_q_indices, self.joint_state_indices, self.joint_q_indices)
        if not self.joint_q_default.is_contiguous() or any(
            value.device != self.joint_q_default.device for value in mappings
        ):
            raise ValueError("Task-table q defaults and mappings must be contiguous on one device.")
        q_indices = torch.cat((self.root_q_indices.reshape(-1), self.joint_q_indices))
        if q_indices.numel() and (
            bool(torch.any((q_indices < 0) | (q_indices >= self.joint_q_default.numel())))
            or torch.unique(q_indices).numel() != q_indices.numel()
        ):
            raise ValueError("Task-table q mappings must be unique indices into joint_q_default.")

    def joint_q_into(self, state_bank: ResetStateBank, state_rows: torch.Tensor, out: torch.Tensor) -> None:
        """Gather selected reset states into caller-owned Newton coordinates."""
        expected_shape = (state_rows.numel(), self.joint_q_default.numel())
        if (
            state_rows.ndim != 1
            or state_rows.dtype is not torch.int64
            or state_rows.device != self.joint_q_default.device
        ):
            raise ValueError("Task-table state rows must be a one-dimensional int64 tensor on the mapping device.")
        if (
            tuple(out.shape) != expected_shape
            or out.dtype is not self.joint_q_default.dtype
            or out.device != self.joint_q_default.device
            or not out.is_contiguous()
        ):
            raise ValueError(f"Task-table joint-q output must be contiguous float32 with shape {expected_shape}.")
        out.copy_(self.joint_q_default)
        if self.root_state_indices.numel():
            root_pose = state_bank.root_pose[state_rows][:, self.root_state_indices].reshape(state_rows.shape[0], -1)
            out[:, self.root_q_indices.reshape(-1)] = root_pose
        if self.joint_state_indices.numel():
            out[:, self.joint_q_indices] = state_bank.joint_position[state_rows][:, self.joint_state_indices]


@dataclass(frozen=True, slots=True)
class TaskTablePointEvidence:
    """One named point batch with explicit global or reset-state scope."""

    name: str
    points: torch.Tensor
    """Environment-local positions [m], shape [scope_count, point_count, 3], float32."""
    scope: Literal["global", "state"] = "state"
    """Whether the first axis contains one global row or every reset-state row."""
    valid: torch.Tensor | None = None
    """Optional active-point mask, shape [scope_count, point_count], bool."""
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Normalized RGB color shared by the batch."""
    radius: float = 0.01
    """Displayed point radius [m]."""

    def __post_init__(self) -> None:
        """Check point geometry and its optional mask."""
        if self.scope not in ("global", "state"):
            raise ValueError("Point evidence scope must be 'global' or 'state'.")
        if not self.name or self.points.ndim != 3 or self.points.shape[1] < 1 or self.points.shape[2] != 3:
            raise ValueError("Point evidence requires a name and float32 [scope_count, point_count, 3] points.")
        _check_tensor("Point evidence", self.points, tuple(self.points.shape), torch.float32)
        if self.valid is not None:
            _check_tensor("Point evidence validity", self.valid, tuple(self.points.shape[:2]), torch.bool)
            if self.valid.device != self.points.device:
                raise ValueError("Point evidence and its validity mask must share one device.")


@dataclass(frozen=True, slots=True)
class TaskTableLineEvidence:
    """One named line-segment batch with explicit global or reset-state scope."""

    name: str
    endpoints: torch.Tensor
    """Environment-local endpoints [m], shape [scope_count, line_count, 2, 3], float32."""
    scope: Literal["global", "state"] = "state"
    """Whether the first axis contains one global row or every reset-state row."""
    valid: torch.Tensor | None = None
    """Optional active-line mask, shape [scope_count, line_count], bool."""
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Normalized RGB color shared by the batch."""
    width: float = 0.01
    """Displayed line width [m]."""

    def __post_init__(self) -> None:
        """Check line geometry and its optional mask."""
        if self.scope not in ("global", "state"):
            raise ValueError("Line evidence scope must be 'global' or 'state'.")
        if (
            not self.name
            or self.endpoints.ndim != 4
            or self.endpoints.shape[1] < 1
            or self.endpoints.shape[2:] != (2, 3)
        ):
            raise ValueError("Line evidence requires a name and float32 [scope_count, line_count, 2, 3] endpoints.")
        _check_tensor("Line evidence", self.endpoints, tuple(self.endpoints.shape), torch.float32)
        if self.valid is not None:
            _check_tensor("Line evidence validity", self.valid, tuple(self.endpoints.shape[:2]), torch.bool)
            if self.valid.device != self.endpoints.device:
                raise ValueError("Line evidence and its validity mask must share one device.")


@dataclass(frozen=True, slots=True)
class TaskTableQuality:
    """Named quality scalars with explicit global, sequence, or reset-state scope."""

    names: tuple[str, ...]
    values: torch.Tensor
    """Quality scalar values, shape [scope_count, scalar_count], float32."""
    scope: Literal["global", "sequence", "state"] = "state"
    """Whether rows describe the table, its sequences, or its reset states."""

    def __post_init__(self) -> None:
        """Check scalar identities and columns."""
        if self.scope not in ("global", "sequence", "state"):
            raise ValueError("Task-table quality scope must be 'global', 'sequence', or 'state'.")
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Task-table quality names must be nonempty and unique.")
        _check_tensor("Task-table quality", self.values, (self.values.shape[0], len(self.names)), torch.float32)


@dataclass(frozen=True, slots=True)
class TaskTableView:
    """Shared simulator-free state, mechanics, sequence, and evidence boundary."""

    sequences: TaskTableSequenceIndex
    state_bank: ResetStateBank
    """Canonical reset-state storage retained without copying."""
    kinematic_view: TaskTableKinematicView
    points: tuple[TaskTablePointEvidence, ...] = ()
    lines: tuple[TaskTableLineEvidence, ...] = ()
    quality: TaskTableQuality | None = None

    def __post_init__(self) -> None:
        """Check shared row, device, name, and mechanics identity."""
        device = self.state_bank.root_pose.device
        if self.sequences.offsets.device != device or self.kinematic_view.joint_q_default.device != device:
            raise ValueError("Task-table sequences, states, and kinematic mappings must share one device.")
        if self.sequences.state_indices is None:
            if self.sequences.frame_count > self.state_bank.row_count:
                raise ValueError("Contiguous task-table frames exceed the reset-state bank.")
        elif not bool(
            torch.all((self.sequences.state_indices >= 0) & (self.sequences.state_indices < self.state_bank.row_count))
        ):
            raise ValueError("Task-table sequence indices must address reset-state rows.")

        roots = self.kinematic_view.root_state_indices
        if roots.numel() and bool(torch.any((roots < 0) | (roots >= self.state_bank.layout.entity_count))):
            raise ValueError("Kinematic root mappings must address reset-state entities.")
        if tuple(self.state_bank.layout.names[int(index)] for index in roots) != self.kinematic_view.root_entity_names:
            raise ValueError("Kinematic root names must match their reset-state entity indices.")

        joint_names = tuple(
            (entity, joint)
            for entity, joints in zip(self.state_bank.layout.names, self.state_bank.layout.joint_names, strict=True)
            for joint in joints
        )
        joint_indices = self.kinematic_view.joint_state_indices
        if joint_indices.numel() and bool(torch.any((joint_indices < 0) | (joint_indices >= len(joint_names)))):
            raise ValueError("Kinematic joint mappings must address reset-state joint columns.")
        if tuple(joint_names[int(index)] for index in joint_indices) != self.kinematic_view.joint_coordinate_names:
            raise ValueError("Kinematic joint names must match their reset-state joint indices.")

        evidence = (*self.points, *self.lines)
        if len({item.name for item in evidence}) != len(evidence):
            raise ValueError("Task-table point and line evidence names must be unique.")
        for item in evidence:
            values = item.points if isinstance(item, TaskTablePointEvidence) else item.endpoints
            expected_rows = 1 if item.scope == "global" else self.state_bank.row_count
            if values.shape[0] != expected_rows or values.device != device:
                raise ValueError("Task-table evidence rows must match their declared scope on the bank device.")
        if self.quality is not None:
            expected_rows = {
                "global": 1,
                "sequence": self.sequences.sequence_count,
                "state": self.state_bank.row_count,
            }[self.quality.scope]
            if self.quality.values.shape[0] != expected_rows or self.quality.values.device != device:
                raise ValueError("Task-table quality rows must match their declared scope on the bank device.")
