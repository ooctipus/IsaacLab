# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for binding partial Newton articulation views to model-global actuators."""

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
from newton import Model


def _control(view) -> NewtonActuatorControl:
    control = NewtonActuatorControl.__new__(NewtonActuatorControl)
    control._articulation = SimpleNamespace(_root_view=view, device="cpu")
    return control


def test_irregular_frequency_layout_resolves_absolute_dof_indices() -> None:
    """Resolve each view row through explicit per-world articulation offsets."""
    layout = SimpleNamespace(
        base_offsets=wp.array([[2], [9]], dtype=wp.int32, device="cpu"),
        local_indices=wp.array([1, 3], dtype=wp.int32, device="cpu"),
    )
    view = SimpleNamespace(
        frequency_layouts={Model.AttributeFrequency.JOINT_DOF: layout},
        world_count=2,
        count_per_world=1,
        world_ids=[2, 5],
    )

    assert _control(view)._joint_dof_index_map().tolist() == [[3, 5], [10, 12]]


def test_partial_view_reset_translates_rows_to_model_worlds() -> None:
    """Translate local rows before resetting model-global actuator history."""
    view = SimpleNamespace(world_ids=[2, 5], world_count=2)
    control = _control(view)

    assert torch.equal(control._model_world_ids(slice(None)), torch.tensor([2, 5]))
    assert torch.equal(control._model_world_ids(torch.tensor([1])), torch.tensor([5]))
