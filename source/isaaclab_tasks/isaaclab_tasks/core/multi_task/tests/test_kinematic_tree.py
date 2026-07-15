# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared kinematic-tree rotation operators."""

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_from_rotation_vector, quat_mul

from isaaclab_tasks.core.multi_task.kinematics import (
    fit_ordered_hinge_coordinates,
    ordered_hinge_coordinate_velocity,
    ordered_hinge_rotation,
    time_gaussian_filter,
    time_gaussian_filter_segmented,
    time_gradient,
    time_gradient_segmented,
    time_quaternion_angular_velocity,
    time_quaternion_angular_velocity_segmented,
    time_unwrap_angles,
    time_unwrap_angles_segmented,
)


def test_quaternion_angular_velocity_ignores_alternating_representation_signs() -> None:
    """Equivalent per-frame quaternion signs must produce the same angular velocity."""
    step_seconds = 0.1
    angle = torch.arange(6, dtype=torch.float64) * 0.2
    rotation_vector = torch.zeros(1, 6, 3, dtype=torch.float64)
    rotation_vector[..., 2] = angle
    rotation = quat_from_rotation_vector(rotation_vector)
    alternating_sign = torch.tensor((1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=torch.float64).view(1, 6, 1)

    angular_velocity = time_quaternion_angular_velocity(rotation * alternating_sign, step_seconds)

    expected = torch.zeros_like(rotation_vector)
    expected[:, :-1, 2] = 2.0
    torch.testing.assert_close(angular_velocity, expected, rtol=1.0e-12, atol=1.0e-12)


def test_quaternion_angular_velocity_tracks_a_full_turn_across_the_sign_boundary() -> None:
    """A 0-to-2pi sweep must retain its analytical rate through a quaternion sign boundary."""
    step_seconds = 0.01
    angle = torch.linspace(0.0, 2.0 * torch.pi, 65, dtype=torch.float64)
    rotation_vector = torch.zeros(1, 65, 3, dtype=torch.float64)
    rotation_vector[..., 1] = angle
    rotation = quat_from_rotation_vector(rotation_vector)
    rotation = torch.where(rotation[..., 3:] < 0.0, -rotation, rotation)

    angular_velocity = time_quaternion_angular_velocity(rotation, step_seconds)

    expected = torch.zeros_like(rotation_vector)
    expected[:, :-1, 1] = (angle[1] - angle[0]) / step_seconds
    torch.testing.assert_close(angular_velocity, expected, rtol=1.0e-12, atol=1.0e-12)


def test_segmented_gradient_does_not_cross_a_discontinuous_boundary() -> None:
    """Flat differentiation must equal independent differentiation of each segment."""
    values = torch.tensor((0.0, 1.0, 3.0, 100.0, 104.0, 110.0), dtype=torch.float64)[:, None]
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)
    steps = torch.tensor((0.5, 2.0), dtype=torch.float32)

    actual = time_gradient_segmented(values, offsets, steps)
    expected = torch.cat(
        tuple(time_gradient(segment[None], float(step)).squeeze(0) for segment, step in zip(values.split(3), steps))
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_segmented_quaternion_velocity_does_not_cross_a_discontinuous_boundary() -> None:
    """The first rotation of a new segment must not form an edge with the preceding segment."""
    angles = torch.tensor((0.0, 0.1, 0.3, 2.0, 2.4, 3.0), dtype=torch.float64)
    rotation_vector = torch.zeros(6, 3, dtype=torch.float64)
    rotation_vector[:, 2] = angles
    rotation = quat_from_rotation_vector(rotation_vector)
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)
    steps = torch.tensor((0.1, 0.2), dtype=torch.float32)

    actual = time_quaternion_angular_velocity_segmented(rotation, offsets, steps)
    expected = torch.cat(
        tuple(
            time_quaternion_angular_velocity(segment[None], float(step)).squeeze(0)
            for segment, step in zip(rotation.split(3), steps)
        )
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(actual[(2, 5), :], torch.zeros(2, 3, dtype=torch.float64))


def test_segmented_derivatives_write_caller_owned_strided_outputs() -> None:
    """Motion qd assembly reuses one output tensor without a full-width return allocation."""
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)
    steps = torch.tensor((0.1, 0.2), dtype=torch.float32)
    values = torch.arange(18, dtype=torch.float32).view(6, 3)
    rotation_vector = torch.zeros(6, 3)
    rotation_vector[:, 2] = torch.tensor((0.0, 0.1, 0.3, 2.0, 2.4, 3.0))
    rotation = quat_from_rotation_vector(rotation_vector)
    expected_gradient = time_gradient_segmented(values, offsets, steps)
    expected_angular = time_quaternion_angular_velocity_segmented(rotation, offsets, steps)
    output = torch.empty(6, 6)
    gradient_output = output[:, :3]
    angular_output = output[:, 3:]

    gradient_result = time_gradient_segmented(values, offsets, steps, gradient_output)
    angular_result = time_quaternion_angular_velocity_segmented(rotation, offsets, steps, angular_output)

    assert gradient_result is gradient_output
    assert angular_result is angular_output
    torch.testing.assert_close(gradient_output, expected_gradient)
    torch.testing.assert_close(angular_output, expected_angular)


def test_segmented_gaussian_filter_does_not_blend_adjacent_segments() -> None:
    """Nearest padding must clamp at each segment boundary rather than only the flat tensor boundary."""
    values = torch.tensor((0.0, 1.0, 2.0, 100.0, 101.0, 102.0), dtype=torch.float32)[:, None]
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)

    actual = time_gaussian_filter_segmented(values, offsets)
    expected = torch.cat(tuple(time_gaussian_filter(segment[None]).squeeze(0) for segment in values.split(3)))

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_segmented_angle_unwrap_restarts_at_each_clip() -> None:
    """Angle representatives are continuous within clips without coupling adjacent clips."""
    values = torch.tensor(((3.0,), (-3.0,), (-2.8,), (-3.0,), (3.0,), (2.8,)))
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)

    actual = time_unwrap_angles_segmented(values, offsets)
    expected = torch.cat(tuple(time_unwrap_angles(segment) for segment in values.split(3)))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[3], values[3])


def test_ordered_hinge_coordinate_velocity_inverts_the_non_small_angle_jacobian() -> None:
    """D6 coordinate rates must invert rotated instantaneous axes, not fixed-axis projections."""
    coordinates = torch.tensor(((0.7, -0.5, 1.1), (-0.9, 0.4, -0.6)), dtype=torch.float32)
    expected = torch.tensor(((1.2, -0.8, 0.4), (-0.3, 0.9, 1.4)), dtype=torch.float32)
    axes = torch.eye(3, dtype=torch.float32)
    axis_0 = axes[0].expand(2, 3)
    axis_1_local = axes[1].expand(2, 3)
    axis_2_local = axes[2].expand(2, 3)
    first = quat_from_rotation_vector(coordinates[:, :1] * axis_0)
    second = quat_mul(first, quat_from_rotation_vector(coordinates[:, 1:2] * axis_1_local))
    jacobian = torch.stack((axis_0, quat_apply(first, axis_1_local), quat_apply(second, axis_2_local)), dim=-1)
    angular_velocity = (jacobian @ expected.unsqueeze(-1)).squeeze(-1)

    actual = ordered_hinge_coordinate_velocity(coordinates, axes, angular_velocity)
    fixed_axis_projection = angular_velocity @ axes.T

    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    assert torch.max(torch.abs(fixed_axis_projection - expected)) > 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise training-time TF32 matmul.")
def test_three_hinge_fit_remains_finite_with_training_tf32() -> None:
    """A valid source rotation must remain finite with the training entry point's TF32 setting."""
    rotation = torch.tensor(
        [[-0.22445420920848846, -0.004686277359724045, 0.02124209515750408, 0.9742419719696045]],
        dtype=torch.float32,
        device="cuda",
    )
    axes = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        coordinates, residual = fit_ordered_hinge_coordinates(rotation, axes)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    assert torch.all(torch.isfinite(coordinates))
    assert torch.all(torch.isfinite(residual))
    reconstructed = ordered_hinge_rotation(coordinates, axes)
    expected = torch.nn.functional.normalize(rotation, dim=-1)
    torch.testing.assert_close(reconstructed, expected, atol=2.0e-6, rtol=2.0e-6)


def test_segmented_time_operators_reject_nonfloating_values() -> None:
    """Every segmented temporal operator requires floating-point physical values."""
    values = torch.arange(6, dtype=torch.int64)[:, None]
    rotations = torch.zeros(6, 4, dtype=torch.int64)
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)
    steps = torch.tensor((0.1, 0.2), dtype=torch.float32)

    with pytest.raises(ValueError, match="floating-point"):
        time_gradient_segmented(values, offsets, steps)
    with pytest.raises(ValueError, match="floating-point"):
        time_quaternion_angular_velocity_segmented(rotations, offsets, steps)
    with pytest.raises(ValueError, match="floating-point"):
        time_gaussian_filter_segmented(values, offsets)


def test_segmented_time_operators_are_bitwise_equal_for_duplicate_segments() -> None:
    """Duplicate segments must produce exactly duplicate derivatives and filtered values."""
    segment = torch.tensor(((0.0, 1.0), (1.0, 3.0), (3.0, 6.0)), dtype=torch.float32)
    values = segment.repeat(2, 1)
    rotation_vector = torch.zeros(3, 3)
    rotation_vector[:, 1] = torch.tensor((0.0, 0.2, 0.5))
    rotation = quat_from_rotation_vector(rotation_vector).repeat(2, 1)
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)
    steps = torch.tensor((0.1, 0.1), dtype=torch.float32)

    results = (
        time_gradient_segmented(values, offsets, steps),
        time_quaternion_angular_velocity_segmented(rotation, offsets, steps),
        time_gaussian_filter_segmented(values, offsets),
    )

    for result in results:
        assert torch.equal(result[:3], result[3:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for segmented temporal parity.")
def test_segmented_time_operators_match_cpu_on_cuda() -> None:
    """CUDA segmented operators must retain the Torch CPU laws within float32 tolerance."""
    values = torch.tensor((0.0, 1.0, 3.0, 100.0, 104.0, 110.0), dtype=torch.float32)[:, None]
    rotation_vector = torch.zeros(6, 3)
    rotation_vector[:, 2] = torch.tensor((0.0, 0.1, 0.3, 2.0, 2.4, 3.0))
    rotation = quat_from_rotation_vector(rotation_vector)
    offsets = torch.tensor((0, 3, 6), dtype=torch.int64)
    steps = torch.tensor((0.1, 0.2), dtype=torch.float32)

    expected = (
        time_gradient_segmented(values, offsets, steps),
        time_quaternion_angular_velocity_segmented(rotation, offsets, steps),
        time_gaussian_filter_segmented(values, offsets),
    )
    actual = (
        time_gradient_segmented(values.cuda(), offsets.cuda(), steps.cuda()),
        time_quaternion_angular_velocity_segmented(rotation.cuda(), offsets.cuda(), steps.cuda()),
        time_gaussian_filter_segmented(values.cuda(), offsets.cuda()),
    )

    for cpu, cuda in zip(expected, actual, strict=True):
        torch.testing.assert_close(cuda.cpu(), cpu, rtol=2.0e-6, atol=2.0e-6)
