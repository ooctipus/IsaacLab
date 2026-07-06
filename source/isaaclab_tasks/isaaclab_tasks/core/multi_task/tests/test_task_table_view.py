# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared simulator-free task-table view contract."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from isaaclab_tasks.core.multi_task.mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableKinematicView,
    TaskTableLineEvidence,
    TaskTablePointEvidence,
    TaskTableQuality,
    TaskTableSequenceIndex,
    TaskTableView,
)


def _state_bank(row_count: int = 5) -> ResetStateBank:
    layout = ResetStateLayout(
        names=("robot", "object"),
        kinds=("articulation", "rigid_object"),
        joint_names=(("hip", "knee"), ()),
        joint_offsets=(0, 2, 2),
    )
    root_pose = torch.zeros(row_count, 2, 7)
    root_pose[..., 6] = 1.0
    root_pose[:, 0, 0] = torch.arange(row_count)
    root_pose[:, 1, 1] = torch.arange(row_count) + 10.0
    joint_position = torch.arange(row_count * 2, dtype=torch.float32).reshape(row_count, 2)
    return ResetStateBank(
        layout=layout,
        root_pose=root_pose,
        root_velocity=torch.zeros(row_count, 2, 6),
        joint_position=joint_position,
        joint_velocity=torch.zeros_like(joint_position),
    )


def _kinematic_view() -> TaskTableKinematicView:
    return TaskTableKinematicView(
        model_builder_state=object(),
        joint_q_default=torch.tensor([0.0] * 16 + [7.0, 8.0]),
        root_entity_names=("robot", "object"),
        root_state_indices=torch.tensor([0, 1]),
        root_q_indices=torch.arange(14).reshape(2, 7),
        joint_coordinate_names=(("robot", "hip"), ("robot", "knee")),
        joint_state_indices=torch.tensor([0, 1]),
        joint_q_indices=torch.tensor([14, 15]),
    )


def test_sequence_index_addresses_static_pairs_and_timed_clips() -> None:
    """Static pairs use explicit rows while contiguous clips need no identity tensor."""
    static = TaskTableSequenceIndex(offsets=torch.tensor([0, 2, 4]), state_indices=torch.tensor([3, 4, 0, 2]))
    assert not static.is_timed
    assert torch.equal(static.state_rows(torch.tensor([0, 1]), torch.tensor([1, 0])), torch.tensor([4, 0]))

    timed = TaskTableSequenceIndex(offsets=torch.tensor([0, 3, 5]), frame_dt=torch.tensor([1.0 / 30.0, 1.0 / 60.0]))
    assert timed.is_timed
    assert timed.state_indices is None
    assert torch.equal(timed.state_rows(torch.tensor([0, 1]), torch.tensor([2, 1])), torch.tensor([2, 4]))

    with pytest.raises(IndexError, match="local-frame"):
        timed.state_rows(torch.tensor([0]), torch.tensor([3]))
    with pytest.raises(IndexError, match="local-frame"):
        timed.state_rows(torch.tensor([1]), torch.tensor([-1]))


def test_kinematic_view_defaults_to_repeated_tiled_state_geometry() -> None:
    """Tables without global geometry retain the common tiled visualization layout."""
    kinematics = _kinematic_view()

    assert kinematics.model_builder_shared is None
    assert kinematics.world_spacing == (3.0, 3.0, 0.0)


def test_kinematic_view_gathers_canonical_state_into_newton_coordinates() -> None:
    """One visible mapping fills roots and joints while preserving unmapped q defaults."""
    bank = _state_bank()
    kinematics = _kinematic_view()
    rows = torch.tensor([3, 1])

    joint_q = torch.empty(rows.numel(), kinematics.joint_q_default.numel())
    kinematics.joint_q_into(bank, rows, joint_q)

    assert torch.equal(joint_q[:, :14], bank.root_pose[rows].reshape(2, 14))
    assert torch.equal(joint_q[:, 14:16], bank.joint_position[rows])
    assert torch.equal(joint_q[:, 16:], torch.tensor([[7.0, 8.0], [7.0, 8.0]]))


def test_task_table_view_retains_state_and_evidence_without_copying() -> None:
    """The shared view owns no duplicate reset-state or evidence storage."""
    bank = _state_bank()
    points = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    endpoints = torch.arange(60, dtype=torch.float32).reshape(5, 2, 2, 3)
    quality = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    point_evidence = TaskTablePointEvidence("contacts", points, valid=torch.ones(5, 2, dtype=torch.bool))
    line_evidence = TaskTableLineEvidence("targets", endpoints)
    global_evidence = TaskTablePointEvidence("world_axes", torch.zeros(1, 3, 3), scope="global")
    table_view = TaskTableView(
        sequences=TaskTableSequenceIndex(
            offsets=torch.tensor([0, 2, 4]),
            state_indices=torch.tensor([0, 1, 3, 4]),
        ),
        state_bank=bank,
        kinematic_view=_kinematic_view(),
        points=(point_evidence, global_evidence),
        lines=(line_evidence,),
        quality=TaskTableQuality(("clearance", "stability"), quality),
    )

    assert table_view.state_bank is bank
    assert table_view.points[0].points.data_ptr() == points.data_ptr()
    assert table_view.lines[0].endpoints.data_ptr() == endpoints.data_ptr()
    assert table_view.quality is not None and table_view.quality.values.data_ptr() == quality.data_ptr()


def test_task_table_view_rejects_mapping_identity_and_row_mismatches() -> None:
    """Named mechanics and evidence must address the same canonical state bank."""
    bank = _state_bank()
    wrong_kinematics = TaskTableKinematicView(
        model_builder_state=object(),
        joint_q_default=torch.zeros(16),
        root_entity_names=("object", "robot"),
        root_state_indices=torch.tensor([0, 1]),
        root_q_indices=torch.arange(14).reshape(2, 7),
        joint_coordinate_names=(("robot", "hip"), ("robot", "knee")),
        joint_state_indices=torch.tensor([0, 1]),
        joint_q_indices=torch.tensor([14, 15]),
    )
    sequences = TaskTableSequenceIndex(offsets=torch.tensor([0, 2]), state_indices=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="root names"):
        TaskTableView(sequences, bank, wrong_kinematics)

    with pytest.raises(ValueError, match="declared scope"):
        TaskTableView(
            sequences,
            bank,
            _kinematic_view(),
            points=(TaskTablePointEvidence("contacts", torch.zeros(4, 1, 3)),),
        )

    with pytest.raises(ValueError, match="declared scope"):
        TaskTableView(
            sequences,
            bank,
            _kinematic_view(),
            quality=TaskTableQuality(("yield",), torch.zeros(3, 1), scope="sequence"),
        )


def test_task_table_view_module_has_no_simulator_imports() -> None:
    """The shared data contract may type Newton models but cannot import simulation packages."""
    module_path = Path(__file__).parents[1] / "mdp" / "commands" / "state_command" / "task_table_view.py"
    tree = ast.parse(module_path.read_text())
    forbidden = ("isaacsim", "omni", "carb", "pxr", "isaaclab.app", "isaaclab.envs", "isaaclab.scene", "isaaclab.sim")
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names] + [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not [name for name in imports if name.startswith(forbidden)]
