# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit inputs and outputs for building Newton IK objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import newton.ik as ik
    import numpy as np
    import warp as wp

    from ..newton_kinematics import NewtonKinematics


@dataclass(frozen=True, slots=True)
class IKObjectiveBuildContext:
    """Shared mechanics and batch size for one numerical solve."""

    kinematics: NewtonKinematics
    asset_name: str
    batch_size: int


@dataclass(frozen=True, slots=True)
class IKContactObjectiveBuildContext(IKObjectiveBuildContext):
    """Solve context extended with generic contact identities and activity."""

    contact_body_ids: tuple[int, ...]
    contact_mask: torch.Tensor
    """Mutable uint8 scratch, shape [solve_batch, contact_count]."""


@dataclass(frozen=True, slots=True)
class IKJointPinObjectiveBuildContext(IKObjectiveBuildContext):
    """Solve context with explicit joint indices and generated targets."""

    coordinate_indices: np.ndarray
    dof_indices: np.ndarray
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class IKPositionObjectiveBuildContext(IKObjectiveBuildContext):
    """Position-objective context with explicit point offsets."""

    body_offsets: np.ndarray


@dataclass(frozen=True, slots=True)
class IKObjectiveMeshCollisionBuildContext(IKObjectiveBuildContext):
    """Solve context with explicit probes, obstacle poses, and optional continuous contact confidence."""

    collision_mesh: wp.Mesh
    obstacle_pose: torch.Tensor
    probe_offsets: np.ndarray
    probe_bodies: np.ndarray
    probe_contact_slots: np.ndarray | None = None
    contact_confidence: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class IKConstraintMeshClearanceBuildContext(IKObjectiveBuildContext):
    """Solve context with ungated probes and obstacle poses for hard clearance."""

    collision_mesh: wp.Mesh
    obstacle_pose: torch.Tensor
    probe_offsets: np.ndarray
    probe_bodies: np.ndarray


@dataclass(frozen=True, slots=True)
class IKConstraintBuild:
    """Linearizable constraint features and their ordered upper bounds."""

    features: tuple[ik.IKObjective, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class IKObjectiveBuild:
    """Built objectives plus the generated target field they consume."""

    objectives: tuple[ik.IKObjective, ...]
    target_bind: str | None = None
    """Named generated tensor populated before every solver chunk."""
