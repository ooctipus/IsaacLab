# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.curriculum import (
    get_reset_state,
    set_reset_state,
)


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


class _MockArticulation:
    def __init__(self, num_envs: int, num_joints: int):
        self.num_joints = num_joints
        self.root_state = torch.arange(num_envs * 13, dtype=torch.float32).view(num_envs, 13)
        self.root_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.joint_pos = torch.arange(num_envs * num_joints, dtype=torch.float32).view(num_envs, num_joints) * 0.1
        self.joint_vel = torch.ones(num_envs, num_joints)
        self.data = SimpleNamespace(
            root_state_w=wp.from_torch(self.root_state),
            joint_pos=wp.from_torch(self.joint_pos),
            joint_vel=wp.from_torch(self.joint_vel),
        )
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def write_root_link_pose_to_sim_index(self, root_pose: torch.Tensor, env_ids: torch.Tensor):
        self.calls.append(("root_pose", root_pose, env_ids, None))
        self.root_state[env_ids, :7] = root_pose

    def write_root_com_velocity_to_sim_index(self, root_velocity: torch.Tensor, env_ids: torch.Tensor):
        self.calls.append(("root_velocity", root_velocity, env_ids, None))
        self.root_state[env_ids, 7:] = root_velocity

    def write_joint_state_to_sim_index(self, position: torch.Tensor, velocity: torch.Tensor, env_ids: torch.Tensor):
        self.calls.append(("joint", position, env_ids, velocity))
        self.joint_pos[env_ids] = position
        self.joint_vel[env_ids] = velocity


class _MockRigidObject:
    def __init__(self, num_envs: int):
        self.root_state = torch.arange(num_envs * 13, dtype=torch.float32).view(num_envs, 13) + 100.0
        self.root_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.data = SimpleNamespace(root_state_w=wp.from_torch(self.root_state))
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def write_root_link_pose_to_sim_index(self, root_pose: torch.Tensor, env_ids: torch.Tensor):
        self.calls.append(("root_pose", root_pose, env_ids))
        self.root_state[env_ids, :7] = root_pose

    def write_root_com_velocity_to_sim_index(self, root_velocity: torch.Tensor, env_ids: torch.Tensor):
        self.calls.append(("root_velocity", root_velocity, env_ids))
        self.root_state[env_ids, 7:] = root_velocity


def _make_env(num_envs: int = 3, num_joints: int = 4):
    return SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(
            _articulations={"robot": _MockArticulation(num_envs, num_joints)},
            _rigid_objects={"box": _MockRigidObject(num_envs)},
            env_origins=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 20.0, 30.0],
                    [-1.0, -2.0, -3.0],
                ],
                dtype=torch.float32,
            ),
        ),
    )


def test_get_and_set_reset_state_round_trip_relative():
    env = _make_env()
    reset_assets = ["robot", "box"]
    env_ids = torch.tensor([1, 2])

    state = get_reset_state(env, env_ids, reset_assets, is_relative=True)
    assert state.shape == (2, 13 + 2 * env.scene._articulations["robot"].num_joints + 13)

    robot_root = state[:, :13]
    box_root_start = 13 + 2 * env.scene._articulations["robot"].num_joints
    box_root = state[:, box_root_start : box_root_start + 13]
    torch.testing.assert_close(
        robot_root[:, :3],
        env.scene._articulations["robot"].root_state[env_ids, :3] - env.scene.env_origins[env_ids],
    )
    torch.testing.assert_close(
        box_root[:, :3],
        env.scene._rigid_objects["box"].root_state[env_ids, :3] - env.scene.env_origins[env_ids],
    )

    edited = state.clone()
    edited[:, :3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    edited_before = edited.clone()
    set_reset_state(env, edited, env_ids, reset_assets, is_relative=True)

    torch.testing.assert_close(edited, edited_before)
    torch.testing.assert_close(
        env.scene._articulations["robot"].root_state[env_ids, :3],
        edited[:, :3] + env.scene.env_origins[env_ids],
    )


def test_set_absolute_state_passes_views_without_copy():
    env = _make_env()
    reset_assets = ["robot", "box"]
    robot = env.scene._articulations["robot"]
    box = env.scene._rigid_objects["box"]
    env_ids = torch.tensor([0, 2])

    width = 13 + 2 * robot.num_joints + 13
    states = torch.arange(env_ids.numel() * width, dtype=torch.float32).view(env_ids.numel(), width)
    states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    box_root_start = 13 + 2 * robot.num_joints
    states[:, box_root_start + 3 : box_root_start + 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

    set_reset_state(env, states, env_ids, reset_assets, is_relative=False)

    root_pose_call = next(call for call in robot.calls if call[0] == "root_pose")
    root_velocity_call = next(call for call in robot.calls if call[0] == "root_velocity")
    joint_call = next(call for call in robot.calls if call[0] == "joint")
    box_pose_call = next(call for call in box.calls if call[0] == "root_pose")
    box_velocity_call = next(call for call in box.calls if call[0] == "root_velocity")

    assert root_pose_call[1].data_ptr() == states[:, :7].data_ptr()
    assert root_velocity_call[1].data_ptr() == states[:, 7:13].data_ptr()
    assert joint_call[1].data_ptr() == states[:, 13 : 13 + robot.num_joints].data_ptr()
    assert joint_call[3].data_ptr() == states[:, 13 + robot.num_joints : 13 + 2 * robot.num_joints].data_ptr()
    assert box_pose_call[1].data_ptr() == states[:, box_root_start : box_root_start + 7].data_ptr()
    assert box_velocity_call[1].data_ptr() == states[:, box_root_start + 7 : box_root_start + 13].data_ptr()
