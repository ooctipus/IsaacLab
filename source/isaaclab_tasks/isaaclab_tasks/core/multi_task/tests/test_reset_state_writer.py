# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared canonical reset-state runtime writer."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.core.multi_task.mdp.commands.state_command import ResetStateBank, ResetStateLayout


def _states() -> ResetStateBank:
    layout = ResetStateLayout(
        names=("robot", "object"),
        kinds=("articulation", "rigid_object"),
        joint_names=(("joint_a", "joint_b"), ()),
        joint_offsets=(0, 2, 2),
    )
    root_pose = torch.zeros(3, 2, 7)
    root_pose[..., 6] = 1.0
    root_pose[:, 0, 0] = torch.tensor((1.0, 2.0, 3.0))
    root_pose[:, 1, 0] = torch.tensor((4.0, 5.0, 6.0))
    return ResetStateBank(
        layout,
        root_pose,
        torch.arange(36, dtype=torch.float32).reshape(3, 2, 6),
        torch.tensor(((0.1, 0.2), (1.1, 1.2), (2.1, 2.2))),
        torch.tensor(((0.3, 0.4), (1.3, 1.4), (2.3, 2.4))),
    )


def test_reset_state_writer_binds_layout_once_and_writes_named_fields(monkeypatch) -> None:
    """One writer handles mixed entity counts, relative roots, and articulation joints."""
    from isaaclab_tasks.core.multi_task.mdp.commands.state_command import reset_state_writer

    class _Entity:
        def write_root_pose_to_sim_index(
            self, *, root_pose: torch.Tensor, env_ids: torch.Tensor, skip_forward: bool = False
        ) -> None:
            self.root_pose_call = (root_pose.clone(), env_ids.clone(), skip_forward)

        def write_root_velocity_to_sim_index(
            self, *, root_velocity: torch.Tensor, env_ids: torch.Tensor, skip_forward: bool = False
        ) -> None:
            self.root_velocity_call = (root_velocity.clone(), env_ids.clone(), skip_forward)

    class _Articulation(_Entity):
        joint_names = ("joint_a", "joint_b")

        def write_joint_position_to_sim_index(
            self, *, position: torch.Tensor, env_ids: torch.Tensor, skip_forward: bool = False
        ) -> None:
            self.joint_position_call = (position.clone(), env_ids.clone(), skip_forward)

        def write_joint_velocity_to_sim_index(
            self, *, velocity: torch.Tensor, env_ids: torch.Tensor, skip_forward: bool = False
        ) -> None:
            self.joint_velocity_call = (velocity.clone(), env_ids.clone(), skip_forward)

    class _RigidObject(_Entity):
        pass

    monkeypatch.setattr(reset_state_writer, "_runtime_asset_types", lambda: (_Articulation, _RigidObject))
    robot = _Articulation()
    obj = _RigidObject()

    class _Scene:
        env_origins = torch.tensor(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))

        def __getitem__(self, name: str):
            return {"robot": robot, "object": obj}[name]

    states = _states()
    env = SimpleNamespace(scene=_Scene(), num_envs=2, device="cpu")
    writer = reset_state_writer.ResetStateWriter(env, states, states.layout.names, states_relative=True)
    env_ids = torch.tensor((1, 0))
    state_rows = torch.tensor((2, 1))
    writer.write(env_ids, state_rows)

    for entity_index, entity in enumerate((robot, obj)):
        expected_pose = states.root_pose[state_rows, entity_index].clone()
        expected_pose[:, :3] += env.scene.env_origins[env_ids]
        torch.testing.assert_close(entity.root_pose_call[0], expected_pose)
        torch.testing.assert_close(entity.root_velocity_call[0], states.root_velocity[state_rows, entity_index])
        torch.testing.assert_close(entity.root_pose_call[1], env_ids)
        assert entity.root_pose_call[2]
        assert entity.root_velocity_call[2] == (entity is robot)
    torch.testing.assert_close(robot.joint_position_call[0], states.joint_position[state_rows])
    torch.testing.assert_close(robot.joint_velocity_call[0], states.joint_velocity[state_rows])
    assert robot.joint_position_call[2]
    assert not robot.joint_velocity_call[2]

    scratch_pointers = (
        writer._root_pose.data_ptr(),
        writer._root_velocity.data_ptr(),
        writer._joint_position.data_ptr(),
        writer._joint_velocity.data_ptr(),
        writer._origins.data_ptr(),
    )
    writer.write(env_ids.flip(0), state_rows.flip(0))
    assert scratch_pointers == (
        writer._root_pose.data_ptr(),
        writer._root_velocity.data_ptr(),
        writer._joint_position.data_ptr(),
        writer._joint_velocity.data_ptr(),
        writer._origins.data_ptr(),
    )


def test_reset_state_writer_rejects_layout_or_runtime_order_mismatch(monkeypatch) -> None:
    """Entity identity, kind, and articulation order are validated at the runtime boundary."""
    from isaaclab_tasks.core.multi_task.mdp.commands.state_command import reset_state_writer

    class _Articulation:
        joint_names = ("joint_b", "joint_a")

    class _RigidObject:
        pass

    monkeypatch.setattr(reset_state_writer, "_runtime_asset_types", lambda: (_Articulation, _RigidObject))
    env = SimpleNamespace(scene={"robot": _Articulation(), "object": _RigidObject()}, num_envs=2, device="cpu")
    states = _states()

    with pytest.raises(ValueError, match="exactly match"):
        reset_state_writer.ResetStateWriter(env, states, ("object", "robot"), states_relative=False)
    with pytest.raises(ValueError, match="joint order"):
        reset_state_writer.ResetStateWriter(env, states, states.layout.names, states_relative=False)
