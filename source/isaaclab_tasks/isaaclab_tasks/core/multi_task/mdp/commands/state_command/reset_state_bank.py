# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical simulator-free reset-state table storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class ResetStateLayout:
    """Immutable entity and joint layout of a reset-state bank."""

    names: tuple[str, ...]
    """Entity names in canonical reset-asset order."""

    kinds: tuple[Literal["articulation", "rigid_object"], ...]
    """Entity kinds in the order of :attr:`names`."""

    joint_names: tuple[tuple[str, ...], ...]
    """Joint names for every entity in canonical simulator order."""

    joint_offsets: tuple[int, ...]
    """Prefix offsets into concatenated joint columns, shape [entity_count + 1]."""

    def __post_init__(self) -> None:
        """Validate ordered names, entity kinds, and exact joint offsets."""
        if not isinstance(self.names, tuple) or not self.names:
            raise ValueError("Reset-state entity names must be a nonempty tuple.")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("Reset-state entity names must be nonempty strings.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Reset-state entity names must be unique.")
        if not isinstance(self.kinds, tuple) or len(self.kinds) != len(self.names):
            raise ValueError("Reset-state kinds must contain one entry per entity.")
        if any(kind not in ("articulation", "rigid_object") for kind in self.kinds):
            raise ValueError("Reset-state entity kinds must be 'articulation' or 'rigid_object'.")
        if not isinstance(self.joint_names, tuple) or len(self.joint_names) != len(self.names):
            raise ValueError("Reset-state joint names must contain one tuple per entity.")

        expected_offsets = [0]
        for name, kind, names in zip(self.names, self.kinds, self.joint_names, strict=True):
            if not isinstance(names, tuple):
                raise ValueError(f"Reset-state joints for entity {name!r} must be a tuple.")
            if any(not isinstance(joint_name, str) or not joint_name for joint_name in names):
                raise ValueError(f"Reset-state joint names for entity {name!r} must be nonempty strings.")
            if len(set(names)) != len(names):
                raise ValueError(f"Reset-state joint names for entity {name!r} must be unique.")
            if kind == "rigid_object" and names:
                raise ValueError(f"Rigid object {name!r} cannot declare joints.")
            expected_offsets.append(expected_offsets[-1] + len(names))

        if (
            not isinstance(self.joint_offsets, tuple)
            or any(type(offset) is not int for offset in self.joint_offsets)
            or self.joint_offsets != tuple(expected_offsets)
        ):
            message = f"Reset-state joint offsets must equal the exact prefix offsets {tuple(expected_offsets)}."
            raise ValueError(message)

    @property
    def entity_count(self) -> int:
        """Number of entities represented by the layout."""
        return len(self.names)

    def entity_index(self, name: str) -> int:
        """Return the exact canonical index of an entity.

        Args:
            name: Exact entity name.

        Returns:
            Canonical entity index.

        Raises:
            KeyError: If :paramref:`name` is not present.
        """
        try:
            return self.names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown reset-state entity: {name!r}.") from error

    def joint_slice(self, name: str) -> slice:
        """Return the concatenated joint-column slice of an entity.

        Args:
            name: Exact entity name.

        Returns:
            Slice into the joint position and velocity tensors.

        Raises:
            KeyError: If :paramref:`name` is not present.
        """
        index = self.entity_index(name)
        return slice(self.joint_offsets[index], self.joint_offsets[index + 1])


@dataclass(frozen=True, slots=True)
class ResetStateBank:
    """Immutable-layout structure of arrays for simulator reset states.

    Tensor storage is retained without copies and is read-only after construction
    by contract. Every tensor is contiguous, detached float32 on one device.
    """

    layout: ResetStateLayout
    """Canonical entity and joint layout shared by every row."""

    root_pose: torch.Tensor
    """Entity root position and xyzw orientation [m, unitless], shape [row_count, entity_count, 7]."""

    root_velocity: torch.Tensor
    """Entity root linear and angular velocity [m/s, rad/s], shape [row_count, entity_count, 6]."""

    joint_position: torch.Tensor
    """Concatenated joint positions [m or rad, depending on joint type], shape [row_count, joint_count]."""

    joint_velocity: torch.Tensor
    """Concatenated joint velocities [m/s or rad/s], shape [row_count, joint_count]."""

    def __post_init__(self) -> None:
        """Validate one exact row axis, entity axis, joint axis, and tensor contract."""
        if not isinstance(self.layout, ResetStateLayout):
            raise TypeError("Reset-state bank layout must be ResetStateLayout.")
        tensors = {
            "root_pose": self.root_pose,
            "root_velocity": self.root_velocity,
            "joint_position": self.joint_position,
            "joint_velocity": self.joint_velocity,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("Reset-state bank columns must be torch.Tensor instances.")

        row_count = self.root_pose.shape[0] if self.root_pose.ndim > 0 else 0
        joint_count = self.layout.joint_offsets[-1]
        expected_shapes = {
            "root_pose": (row_count, self.layout.entity_count, 7),
            "root_velocity": (row_count, self.layout.entity_count, 6),
            "joint_position": (row_count, joint_count),
            "joint_velocity": (row_count, joint_count),
        }
        if row_count < 1:
            raise ValueError("Reset-state bank must contain at least one row.")
        for name, value in tensors.items():
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(f"Reset-state column {name!r} must have shape {expected_shapes[name]}.")
            if (
                value.dtype is not torch.float32
                or value.device != self.root_pose.device
                or not value.is_contiguous()
                or value.requires_grad
            ):
                raise ValueError(
                    f"Reset-state column {name!r} must be contiguous detached float32 on {self.root_pose.device}."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"Reset-state column {name!r} must contain only finite values.")

        rotation_norm = torch.linalg.vector_norm(self.root_pose[..., 3:7], dim=-1)
        if not torch.allclose(rotation_norm, torch.ones_like(rotation_norm), rtol=1.0e-5, atol=1.0e-5):
            raise ValueError("Reset-state root poses must contain unit xyzw quaternions.")

    @property
    def row_count(self) -> int:
        """Number of reset states stored in the bank."""
        return self.root_pose.shape[0]
