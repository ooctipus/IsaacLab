# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for generalized effort limits exposed by shared Newton kinematics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics


def _array(values, dtype: torch.dtype) -> torch.Tensor:
    """Create one CPU tensor exposing Newton's public NumPy boundary."""
    return torch.tensor(values, dtype=dtype)


def test_joint_effort_bounds_aggregate_single_dof_actuators() -> None:
    """Joint-transmission force ranges compose in generalized-force space."""
    model = SimpleNamespace(
        joint_effort_limit=_array([100.0, 100.0], torch.float32),
        mujoco=SimpleNamespace(
            actuator_trnid=_array([[0, 0], [0, 0], [1, 0], [1, 0]], torch.int32),
            actuator_trntype=_array([0, 0, 2, 0], torch.int32),
            actuator_gear=_array(
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                torch.float32,
            ),
            actuator_has_forcerange=_array([True, True, True, False], torch.bool),
            actuator_forcerange=_array(
                [[-3.0, 4.0], [-2.0, 5.0], [-100.0, 100.0], [-7.0, 7.0]],
                torch.float32,
            ),
        ),
    )

    lower, upper = NewtonKinematics._joint_effort_bounds(model)

    np.testing.assert_allclose(lower, [-11.0, -100.0])
    np.testing.assert_allclose(upper, [10.0, 100.0])


def test_joint_effort_bounds_intersect_joint_limits() -> None:
    """Actuator bounds cannot widen explicit generalized-effort limits."""
    model = SimpleNamespace(
        joint_effort_limit=_array([9.0], torch.float32),
        mujoco=SimpleNamespace(
            actuator_trnid=_array([[0, 0]], torch.int32),
            actuator_trntype=_array([0], torch.int32),
            actuator_gear=_array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], torch.float32),
            actuator_has_forcerange=_array([True], torch.bool),
            actuator_forcerange=_array([[-20.0, 20.0]], torch.float32),
        ),
    )

    lower, upper = NewtonKinematics._joint_effort_bounds(model)

    np.testing.assert_allclose(lower, [-9.0])
    np.testing.assert_allclose(upper, [9.0])


def test_joint_effort_bounds_reject_invalid_generalized_velocity() -> None:
    """A ranged joint actuator must target an existing generalized velocity."""
    model = SimpleNamespace(
        joint_effort_limit=_array([9.0], torch.float32),
        mujoco=SimpleNamespace(
            actuator_trnid=_array([[1, 0]], torch.int32),
            actuator_trntype=_array([0], torch.int32),
            actuator_gear=_array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], torch.float32),
            actuator_has_forcerange=_array([True], torch.bool),
            actuator_forcerange=_array([[-1.0, 1.0]], torch.float32),
        ),
    )

    with pytest.raises(ValueError, match="invalid generalized velocity"):
        NewtonKinematics._joint_effort_bounds(model)
