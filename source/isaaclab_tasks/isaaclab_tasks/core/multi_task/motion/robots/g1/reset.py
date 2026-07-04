# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 reference and lie-down reset transform."""

from __future__ import annotations

import math

import torch

from ...data import MotionResetState


class G1ReferenceAndLieDownReset:
    """Select reference and BFM-Zero lie-down states in fixed storage."""

    reset_source_names = ("reference", "lie_down")

    def __init__(
        self,
        *,
        capacity: int,
        device: str | torch.device,
        lie_down_root_height_m: float,
        lie_down_roll_magnitude_rad: float,
        lie_down_negative_roll_probability: float,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("G1 reset capacity must be a positive integer.")
        if not math.isfinite(lie_down_root_height_m) or lie_down_root_height_m <= 0.0:
            raise ValueError("G1 lie-down root height must be finite and positive [m].")
        if not math.isfinite(lie_down_roll_magnitude_rad) or lie_down_roll_magnitude_rad <= 0.0:
            raise ValueError("G1 lie-down roll magnitude must be finite and positive [rad].")
        if (
            not math.isfinite(lie_down_negative_roll_probability)
            or not 0.0 <= lie_down_negative_roll_probability <= 1.0
        ):
            raise ValueError("G1 negative-roll probability must be finite and lie in [0, 1].")
        half_angle = 0.5 * lie_down_roll_magnitude_rad
        half_sine = math.sin(half_angle)
        device = torch.device(device)
        self._capacity = capacity
        self._lie_down_root_height = torch.tensor(lie_down_root_height_m, device=device)
        self._root_position = torch.empty(capacity, 3, device=device)
        self._root_rotation = torch.empty(capacity, 4, device=device)
        self._negative_roll_probability = lie_down_negative_roll_probability
        self._rotated_root = torch.empty(capacity, 4, device=device)
        self._rotation_scratch = torch.empty(capacity, device=device)
        self._lie_down = torch.empty(capacity, dtype=torch.bool, device=device)
        self._random = torch.empty((), device=device)
        self._random_is_low = torch.empty((), dtype=torch.bool, device=device)
        self._delta_x = torch.empty((), device=device)
        self._delta_x_negative = torch.tensor(-half_sine, device=device)
        self._delta_x_positive = torch.tensor(half_sine, device=device)
        self._delta_w = math.cos(half_angle)

    def __call__(
        self,
        reference: MotionResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionResetState:
        """Select source zero as reference and source one as native G1 lie-down."""
        count = reset_source_indices.shape[0]
        if count > self._capacity:
            raise ValueError(f"G1 reset batch {count} exceeds capacity {self._capacity}.")
        root_position = self._root_position[:count]
        root_rotation = self._root_rotation[:count]
        rotated = self._rotated_root[:count]
        scratch = self._rotation_scratch[:count]
        lie_down = self._lie_down[:count]

        torch.eq(reset_source_indices, 1, out=lie_down)
        root_position.copy_(reference.root_position)
        torch.where(lie_down, self._lie_down_root_height, root_position[:, 2], out=root_position[:, 2])

        torch.rand((), device=root_rotation.device, generator=generator, out=self._random)
        torch.lt(self._random, self._negative_roll_probability, out=self._random_is_low)
        torch.where(
            self._random_is_low,
            self._delta_x_negative,
            self._delta_x_positive,
            out=self._delta_x,
        )

        reference_rotation = reference.root_rotation_xyzw
        torch.mul(reference_rotation[:, 0], self._delta_w, out=rotated[:, 0])
        torch.mul(reference_rotation[:, 3], self._delta_x, out=scratch)
        rotated[:, 0].add_(scratch)
        torch.mul(reference_rotation[:, 1], self._delta_w, out=rotated[:, 1])
        torch.mul(reference_rotation[:, 2], self._delta_x, out=scratch)
        rotated[:, 1].sub_(scratch)
        torch.mul(reference_rotation[:, 2], self._delta_w, out=rotated[:, 2])
        torch.mul(reference_rotation[:, 1], self._delta_x, out=scratch)
        rotated[:, 2].add_(scratch)
        torch.mul(reference_rotation[:, 3], self._delta_w, out=rotated[:, 3])
        torch.mul(reference_rotation[:, 0], self._delta_x, out=scratch)
        rotated[:, 3].sub_(scratch)
        torch.where(lie_down.unsqueeze(-1), rotated, reference_rotation, out=root_rotation)

        return MotionResetState(
            root_position=root_position,
            root_rotation_xyzw=root_rotation,
            root_linear_velocity_world=reference.root_linear_velocity_world,
            root_angular_velocity_world=reference.root_angular_velocity_world,
            joint_position=reference.joint_position,
            joint_velocity=reference.joint_velocity,
        )
