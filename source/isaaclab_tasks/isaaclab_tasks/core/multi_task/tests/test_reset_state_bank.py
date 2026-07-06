# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for canonical simulator-free reset-state table storage."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from isaaclab_tasks.core.multi_task.mdp.commands.state_command import ResetStateBank, ResetStateLayout


def _bank(layout: ResetStateLayout, row_count: int = 4) -> ResetStateBank:
    root_pose = torch.zeros(row_count, layout.entity_count, 7)
    root_pose[..., 6] = 1.0
    joint_count = layout.joint_offsets[-1]
    return ResetStateBank(
        layout=layout,
        root_pose=root_pose,
        root_velocity=torch.zeros(row_count, layout.entity_count, 6),
        joint_position=torch.zeros(row_count, joint_count),
        joint_velocity=torch.zeros(row_count, joint_count),
    )


@pytest.mark.parametrize(
    ("layout", "expected_slices"),
    (
        (
            ResetStateLayout(
                names=("robot",),
                kinds=("articulation",),
                joint_names=(("left_joint", "right_joint"),),
                joint_offsets=(0, 2),
            ),
            {"robot": slice(0, 2)},
        ),
        (
            ResetStateLayout(
                names=("robot", "board", "nut"),
                kinds=("articulation", "rigid_object", "rigid_object"),
                joint_names=(("left_joint", "right_joint"), (), ()),
                joint_offsets=(0, 2, 2, 2),
            ),
            {"robot": slice(0, 2), "board": slice(2, 2), "nut": slice(2, 2)},
        ),
        (
            ResetStateLayout(
                names=("robot", "gripper", "board", "bolt", "nut"),
                kinds=("articulation", "articulation", "rigid_object", "rigid_object", "rigid_object"),
                joint_names=(("shoulder", "elbow"), ("finger",), (), (), ()),
                joint_offsets=(0, 2, 3, 3, 3, 3),
            ),
            {
                "robot": slice(0, 2),
                "gripper": slice(2, 3),
                "board": slice(3, 3),
                "bolt": slice(3, 3),
                "nut": slice(3, 3),
            },
        ),
    ),
)
def test_reset_state_bank_represents_one_three_and_n_entities(
    layout: ResetStateLayout, expected_slices: dict[str, slice]
) -> None:
    """One canonical entity axis represents Position, Factory, and general N-entity tables."""
    bank = _bank(layout)

    assert bank.row_count == 4
    assert layout.entity_count == len(expected_slices)
    assert tuple(layout.entity_index(name) for name in layout.names) == tuple(range(layout.entity_count))
    assert {name: layout.joint_slice(name) for name in layout.names} == expected_slices
    assert bank.root_pose.shape == (4, layout.entity_count, 7)
    assert bank.joint_position.shape == (4, layout.joint_offsets[-1])


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "names": (),
            "kinds": (),
            "joint_names": (),
            "joint_offsets": (0,),
        },
        {
            "names": ("robot", "robot"),
            "kinds": ("articulation", "articulation"),
            "joint_names": ((), ()),
            "joint_offsets": (0, 0, 0),
        },
        {
            "names": ("object",),
            "kinds": ("rigid_object",),
            "joint_names": (("impossible_joint",),),
            "joint_offsets": (0, 1),
        },
        {
            "names": ("robot",),
            "kinds": ("articulation",),
            "joint_names": (("joint",),),
            "joint_offsets": (0, 2),
        },
        {
            "names": ("robot",),
            "kinds": ("unsupported",),
            "joint_names": ((),),
            "joint_offsets": (0, 0),
        },
    ),
)
def test_reset_state_layout_rejects_invalid_metadata(kwargs: dict[str, object]) -> None:
    """Malformed identity, kind, and offset metadata fails at construction."""
    with pytest.raises(ValueError):
        ResetStateLayout(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("root_pose", torch.zeros(2, 1, 6)),
        ("root_velocity", torch.zeros(2, 2, 6)),
        ("joint_position", torch.zeros(2, 2)),
        ("joint_velocity", torch.zeros(2, 0)),
    ),
)
def test_reset_state_bank_rejects_invalid_shapes(field: str, value: torch.Tensor) -> None:
    """Every physical column must match the layout's row, entity, and joint axes."""
    layout = ResetStateLayout(
        names=("robot",),
        kinds=("articulation",),
        joint_names=(("joint",),),
        joint_offsets=(0, 1),
    )
    values = {
        "layout": layout,
        "root_pose": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]] * 2),
        "root_velocity": torch.zeros(2, 1, 6),
        "joint_position": torch.zeros(2, 1),
        "joint_velocity": torch.zeros(2, 1),
    }
    values[field] = value

    with pytest.raises(ValueError):
        ResetStateBank(**values)  # type: ignore[arg-type]


def test_reset_state_bank_rejects_invalid_tensor_values_and_storage() -> None:
    """Nonfinite, nonunit, non-float32, and differentiable columns are rejected."""
    layout = ResetStateLayout(names=("box",), kinds=("rigid_object",), joint_names=((),), joint_offsets=(0, 0))
    valid = _bank(layout, row_count=2)

    nonfinite = valid.root_velocity.clone()
    nonfinite[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        ResetStateBank(layout, valid.root_pose, nonfinite, valid.joint_position, valid.joint_velocity)

    nonunit = valid.root_pose.clone()
    nonunit[..., 6] = 2.0
    with pytest.raises(ValueError, match="unit xyzw"):
        ResetStateBank(layout, nonunit, valid.root_velocity, valid.joint_position, valid.joint_velocity)

    with pytest.raises(ValueError, match="float32"):
        ResetStateBank(
            layout,
            valid.root_pose.double(),
            valid.root_velocity.double(),
            valid.joint_position.double(),
            valid.joint_velocity.double(),
        )

    differentiable = valid.root_velocity.clone().requires_grad_()
    with pytest.raises(ValueError, match="detached"):
        ResetStateBank(layout, valid.root_pose, differentiable, valid.joint_position, valid.joint_velocity)


def test_reset_state_layout_lookup_is_exact() -> None:
    """Entity lookup does not normalize, sort, or infer role names."""
    layout = ResetStateLayout(names=("Robot",), kinds=("articulation",), joint_names=((),), joint_offsets=(0, 0))

    assert layout.entity_index("Robot") == 0
    with pytest.raises(KeyError, match="robot"):
        layout.entity_index("robot")


def test_reset_state_bank_metadata_is_immutable_and_tensor_storage_is_zero_copy() -> None:
    """Frozen slotted owners retain caller tensors without an alternate representation."""
    layout = ResetStateLayout(names=("box",), kinds=("rigid_object",), joint_names=((),), joint_offsets=(0, 0))
    root_pose = torch.zeros(2, 1, 7)
    root_pose[..., 6] = 1.0
    bank = ResetStateBank(layout, root_pose, torch.zeros(2, 1, 6), torch.zeros(2, 0), torch.zeros(2, 0))

    assert bank.root_pose is root_pose
    assert not hasattr(layout, "__dict__")
    assert not hasattr(bank, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(layout, "names", ("other",))
    with pytest.raises(FrozenInstanceError):
        setattr(bank, "root_pose", root_pose.clone())
    assert not hasattr(bank, "flat")
    assert not hasattr(bank, "states")
