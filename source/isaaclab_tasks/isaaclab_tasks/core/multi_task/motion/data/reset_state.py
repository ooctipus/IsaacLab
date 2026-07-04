# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-ready motion reset-state tensors shared with robot transforms."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MotionResetState:
    """Simulator-ready root and joint state in fixed world-frame and xyzw semantics."""

    root_position: torch.Tensor
    """Root-link position [m], shape [batch, 3], float."""

    root_rotation_xyzw: torch.Tensor
    """Root-link xyzw orientation, shape [batch, 4], float."""

    root_linear_velocity_world: torch.Tensor
    """Root-link linear velocity [m/s], shape [batch, 3], float."""

    root_angular_velocity_world: torch.Tensor
    """Root-link angular velocity [rad/s], shape [batch, 3], float."""

    joint_position: torch.Tensor
    """Simulator-ordered joint positions [rad], shape [batch, joint_count], float."""

    joint_velocity: torch.Tensor
    """Simulator-ordered joint velocities [rad/s], shape [batch, joint_count], float."""
