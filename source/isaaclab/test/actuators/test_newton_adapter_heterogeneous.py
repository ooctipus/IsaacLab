# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for Newton actuator state over heterogeneous flat DOF layouts."""

import numpy as np
import torch
import warp as wp

from isaaclab.actuators.newton import NewtonActuatorAdapter
from isaaclab.actuators.newton.kernels import gather_computed_effort


class _ResetRecorder:
    def __init__(self):
        self.masks: list[list[bool]] = []

    def reset(self, mask: wp.array | None) -> None:
        self.masks.append([True] if mask is None else mask.numpy().tolist())


class _State:
    def __init__(self):
        self.delay_state = _ResetRecorder()
        self.controller_state = None


class _Actuator:
    def __init__(self, indices: list[int]):
        self.indices = wp.array(indices, dtype=wp.uint32, device="cpu")
        self.control_computed_output_attr = None

    def state(self) -> _State:
        return _State()

    def is_graphable(self) -> bool:
        return True

    def is_stateful(self) -> bool:
        return True


def test_reset_projects_model_worlds_through_irregular_flat_dofs() -> None:
    """Reset the actuator slots owned by a selected model world, independent of stride."""
    actuator = _Actuator([0, 3, 4])
    adapter = NewtonActuatorAdapter._from_flat(
        [actuator],
        world_count=3,
        device="cpu",
        dof_count=5,
        dof_world_id=wp.array([0, 0, 1, 1, 2], dtype=wp.int32, device="cpu"),
    )

    adapter.reset([1])

    assert adapter._states_a[0].delay_state.masks == [[False, True, False]]
    assert adapter._states_b[0].delay_state.masks == [[False, True, False]]


def test_computed_effort_gathers_nonuniform_articulation_rows() -> None:
    """Gather telemetry when one articulation occupies nonuniform locations in the model."""
    adapter = NewtonActuatorAdapter._from_flat(
        [],
        world_count=3,
        device="cpu",
        dof_count=9,
        dof_world_id=wp.array([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=wp.int32, device="cpu"),
    )
    adapter._computed_effort.assign(np.arange(9, dtype=np.float32))
    binding = adapter._bind_articulation_flat(
        implicit_joint_indices=[],
        dof_index_map=torch.tensor([[0, 2], [4, 6], [7, 8]], dtype=torch.long),
    )

    assert binding.computed_effort_src is adapter._computed_effort
    assert binding.computed_effort_gather_map is not None
    wp.launch(
        gather_computed_effort,
        dim=binding.computed_effort_view.shape,
        inputs=[binding.computed_effort_src, binding.computed_effort_gather_map],
        outputs=[binding.computed_effort_view],
        device="cpu",
    )
    np.testing.assert_array_equal(binding.computed_effort_view.numpy(), [[0.0, 2.0], [4.0, 6.0], [7.0, 8.0]])


def test_flat_adapter_rejects_duplicate_actuator_ownership() -> None:
    """Reject two actuators writing the same global DOF."""
    world_map = wp.array([0, 1], dtype=wp.int32, device="cpu")
    with np.testing.assert_raises_regex(ValueError, "more than one actuator"):
        NewtonActuatorAdapter._from_flat(
            [_Actuator([0]), _Actuator([0])],
            world_count=2,
            device="cpu",
            dof_count=2,
            dof_world_id=world_map,
        )
