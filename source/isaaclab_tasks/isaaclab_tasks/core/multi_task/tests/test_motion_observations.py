# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact numerical tests for motion observation equations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab_tasks.core.multi_task.motion.frames import append_g1_head_runtime_frame
from isaaclab_tasks.core.multi_task.motion.mdp.observations import (
    g1_privileged_body_observation,
    g1_privileged_observation,
    motion_projected_gravity,
    smpl_humenv_observation,
)


def _phase3_fixtures() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts/reinforcement_learning/forward_backward/phase3/fixtures"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Phase 3 fixtures were not found from the repository test path.")


def test_smpl_observation_matches_native_trace_elementwise() -> None:
    """The runtime SMPL observation must preserve every frozen HumEnv scalar."""
    path = _phase3_fixtures() / "meta_humenv_next_step_trace_v1.npz"
    with np.load(path, allow_pickle=False) as tensors:
        actual = smpl_humenv_observation(
            torch.from_numpy(tensors["current_body_pos"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_quat"])[..., (1, 2, 3, 0)].flatten(0, 1),
            torch.from_numpy(tensors["current_body_lin_vel"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_ang_vel"]).flatten(0, 1),
        ).unflatten(0, (5, 2))
        torch.testing.assert_close(
            actual,
            torch.from_numpy(tensors["current_observation"]),
            rtol=5.0e-7,
            atol=5.0e-7,
        )


def test_g1_privileged_observation_matches_native_trace_elementwise() -> None:
    """The runtime G1 privileged observation must preserve every frozen BFM scalar."""
    path = _phase3_fixtures() / "g1_lafan_same_step_trace_v1.npz"
    with np.load(path, allow_pickle=False) as tensors:
        actual = g1_privileged_observation(
            torch.from_numpy(tensors["current_body_position"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_rotation_xyzw"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_linear_velocity"]).flatten(0, 1),
            torch.from_numpy(tensors["current_body_angular_velocity"]).flatten(0, 1),
        ).unflatten(0, (5, 2))
        torch.testing.assert_close(
            actual,
            torch.from_numpy(tensors["current_privileged_state"]),
            rtol=1.0e-5,
            atol=1.0e-5,
        )


def test_g1_privileged_wrapper_maps_live_body_ids_into_declared_behavior_order() -> None:
    """The manager-resolved behavior axis must not be mistaken for raw articulation order."""
    num_envs = 2
    position = torch.arange(num_envs * 30 * 3, dtype=torch.float32).view(num_envs, 30, 3) * 0.001
    rotation = torch.zeros(num_envs, 30, 4)
    rotation[..., 3] = 1.0
    linear_velocity = position + 0.2
    angular_velocity = position - 0.1
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=SimpleNamespace(torch=position),
            body_link_quat_w=SimpleNamespace(torch=rotation),
            body_com_lin_vel_w=SimpleNamespace(torch=linear_velocity),
            body_com_ang_vel_w=SimpleNamespace(torch=angular_velocity),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    behavior_body_ids = [0, *range(2, 30, 2), *range(1, 30, 2)]
    parent_live_id = 10
    asset_cfg = SimpleNamespace(name="robot", body_ids=behavior_body_ids)

    actual = g1_privileged_body_observation(env, asset_cfg, behavior_body_ids.index(parent_live_id))
    selected = (
        position[:, behavior_body_ids],
        rotation[:, behavior_body_ids],
        linear_velocity[:, behavior_body_ids],
        angular_velocity[:, behavior_body_ids],
    )
    expected = g1_privileged_observation(
        *append_g1_head_runtime_frame(
            *selected,
            parent_body_index=behavior_body_ids.index(parent_live_id),
        )
    )

    torch.testing.assert_close(actual, expected)


def test_projected_gravity_returns_existing_articulation_fact_without_copy() -> None:
    """Projected gravity must preserve the backend-owned normalized body-frame tensor."""
    projected = torch.tensor(((0.0, 0.0, -1.0), (0.5, 0.0, -0.8660254)))
    robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=SimpleNamespace(torch=projected)))
    env = SimpleNamespace(scene={"robot": robot})

    actual = motion_projected_gravity(env)

    assert actual.data_ptr() == projected.data_ptr()
    torch.testing.assert_close(actual, projected)
