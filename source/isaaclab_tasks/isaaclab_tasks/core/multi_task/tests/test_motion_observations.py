# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact numerical tests for motion observation equations."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.utils.math import matrix_from_quat, quat_apply

from isaaclab_tasks.core.multi_task.motion.robots.g1.actions import G1JointPositionAction
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import G1_BEHAVIOR_BODY_NAMES
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import (
    G1_HEAD_PARENT_BODY_NAME,
    append_g1_head_runtime_frame,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.observations import (
    g1_bfm_privileged_body_observation,
    g1_bfm_privileged_observation,
    g1_joint_pos_rel,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.observations import smpl_humenv_observation
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionObservationsCfg


def _phase3_fixtures() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts/reinforcement_learning/forward_backward/phase3/fixtures"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Phase 3 fixtures were not found from the repository test path.")


def test_rotation_matrix_basis_columns_match_bfm_quaternion_projection() -> None:
    """Matrix x/z columns preserve BFM-Zero's tangent-normal quaternion projection."""
    generator = torch.Generator().manual_seed(7)
    rotation = torch.randn(128, 4, dtype=torch.float64, generator=generator)
    rotation /= torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    tangent = torch.zeros(128, 3, dtype=torch.float64)
    tangent[:, 0] = 1.0
    normal = torch.zeros_like(tangent)
    normal[:, 2] = 1.0

    expected = torch.cat((quat_apply(rotation, tangent), quat_apply(rotation, normal)), dim=-1)
    matrix = matrix_from_quat(rotation)
    actual = torch.cat((matrix[..., 0], matrix[..., 2]), dim=-1)

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_robot_wrappers_share_heading_local_geometry() -> None:
    """Robot wrappers must retain only their layout and heading convention."""
    g1_source = inspect.getsource(g1_bfm_privileged_observation)
    smpl_source = inspect.getsource(smpl_humenv_observation)

    assert "body_heading_local_observation" in g1_source
    assert "body_heading_local_observation" in smpl_source
    assert "matrix_from_quat" not in g1_source + smpl_source
    assert "isinstance" not in inspect.getsource(g1_joint_pos_rel)


def test_g1_privileged_layout_is_validated_by_root_config() -> None:
    """The root config must fully declare the invariant body layout removed from the hot path."""
    term = MotionObservationsCfg.G1Cfg.PrivilegedStateCfg().value
    asset_cfg = term.params["asset_cfg"]

    assert asset_cfg.name == "robot"
    assert asset_cfg.body_names == list(G1_BEHAVIOR_BODY_NAMES)
    assert asset_cfg.preserve_order is True
    assert term.params["parent_idx"] == G1_BEHAVIOR_BODY_NAMES.index(G1_HEAD_PARENT_BODY_NAME)


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
        actual = g1_bfm_privileged_observation(
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

    actual = g1_bfm_privileged_body_observation(env, asset_cfg, behavior_body_ids.index(parent_live_id))
    selected = (
        position[:, behavior_body_ids],
        rotation[:, behavior_body_ids],
        linear_velocity[:, behavior_body_ids],
        angular_velocity[:, behavior_body_ids],
    )
    expected = g1_bfm_privileged_observation(
        *append_g1_head_runtime_frame(
            *selected,
            parent_body_index=behavior_body_ids.index(parent_live_id),
        )
    )

    torch.testing.assert_close(actual, expected)


def test_g1_joint_pos_rel_composes_core_state_with_episode_offset() -> None:
    """G1 actor joint position must apply only the episodic offset to the common relative term."""
    joint_position = torch.tensor(((10.0, 20.0, 30.0),))
    default_joint_position = torch.tensor(((1.0, 2.0, 3.0),))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_position),
            default_joint_pos=SimpleNamespace(torch=default_joint_position),
        )
    )
    action = object.__new__(G1JointPositionAction)
    action.default_joint_offset = torch.tensor(((0.5, -0.25),))
    env = SimpleNamespace(
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[2, 0])

    actual = g1_joint_pos_rel(env, "joint_position", asset_cfg)

    torch.testing.assert_close(actual, torch.tensor(((26.5, 9.25),)))


def test_g1_tracking_joint_position_uses_canonical_default_with_nonzero_episode_offset() -> None:
    """Tracking coordinates must not include the actor's episodic action offset."""
    joint_position = torch.tensor(((10.0, 20.0, 30.0),))
    default_joint_position = torch.tensor(((1.0, 2.0, 3.0),))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_position),
            default_joint_pos=SimpleNamespace(torch=default_joint_position),
        )
    )
    action = object.__new__(G1JointPositionAction)
    action.default_joint_offset = torch.tensor(((0.5, -0.25),))
    env = SimpleNamespace(
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[2, 0])
    tracking_cfg = MotionObservationsCfg.G1Cfg.JointPositionUnnoisedCfg().value

    actor = g1_joint_pos_rel(env, "joint_position", asset_cfg)
    tracking = isaaclab_mdp.joint_pos_rel(env, asset_cfg)

    torch.testing.assert_close(actor, torch.tensor(((26.5, 9.25),)))
    torch.testing.assert_close(tracking, torch.tensor(((27.0, 9.0),)))
    assert tracking_cfg.func is isaaclab_mdp.joint_pos_rel
    assert set(tracking_cfg.params) == {"asset_cfg"}


def test_last_action_selects_raw_or_processed_named_action() -> None:
    """The common action observation must expose processed terms without changing its raw default."""
    complete = torch.tensor(((1.0, 2.0, 3.0),))
    raw = torch.tensor(((4.0, 5.0),))
    processed = torch.tensor(((6.0, 7.0),))
    action = SimpleNamespace(raw_actions=raw, processed_actions=processed)
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            action=complete,
            get_term=lambda name: action if name == "joint_position" else None,
        )
    )

    assert isaaclab_mdp.last_action(env).data_ptr() == complete.data_ptr()
    assert isaaclab_mdp.last_action(env, "joint_position").data_ptr() == raw.data_ptr()
    assert isaaclab_mdp.last_action(env, "joint_position", processed=True).data_ptr() == processed.data_ptr()
    with pytest.raises(ValueError, match="requires an action_name"):
        isaaclab_mdp.last_action(env, processed=True)
