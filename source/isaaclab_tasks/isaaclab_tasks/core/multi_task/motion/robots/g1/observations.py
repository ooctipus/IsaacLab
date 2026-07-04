# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 policy, privileged, and transition observations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from ....mdp.observations import body_heading_local_observation
from .articulation import G1_BEHAVIOR_BODY_NAMES, G1_BEHAVIOR_JOINT_NAMES
from .frames import G1_HEAD_FRAME_NAME, append_g1_head_runtime_frame

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ...data import MotionFrameSource
    from .actions import G1JointPositionAction


def g1_bfm_privileged_observation(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    body_angular_velocity_world: torch.Tensor,
) -> torch.Tensor:
    """Return BFM-Zero's 463-wide G1 heading-local body observation."""
    return body_heading_local_observation(
        body_position_world,
        body_rotation_xyzw,
        body_linear_velocity_world,
        body_angular_velocity_world,
        body_rotation_xyzw[:, 0],
    )


def g1_bfm_expert_observation_fields(
    root_rotation_xyzw: torch.Tensor,
    root_angular_velocity_world: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    default_joint_position: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return BFM-Zero expert observations under their learner field names."""
    gravity_world = torch.zeros_like(root_angular_velocity_world)
    gravity_world[..., 2] = -1.0
    projected_gravity = quat_apply_inverse(root_rotation_xyzw, gravity_world)
    return {
        "joint_position": joint_position - default_joint_position,
        "joint_velocity": joint_velocity,
        "projected_gravity": projected_gravity,
        "base_angular_velocity": root_angular_velocity_world,
    }


def g1_bfm_observation_state_pose(joint_position: torch.Tensor) -> torch.Tensor:
    """Return the BFM-Zero 23-joint observation-state pose used by diagnostic tracking."""
    if joint_position.ndim != 2 or joint_position.shape[1] != 29:
        raise ValueError("G1 BFM-Zero observation-state tracking requires one 29-joint pose per row.")
    return joint_position[:, :23]


def g1_joint_pos_rel(env: ManagerBasedRLEnv, action_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return behavior-ordered joint displacement from the episodic default pose [rad]."""
    action = env.action_manager.get_term(action_name)
    return isaaclab_mdp.joint_pos_rel(env, asset_cfg) - action.default_joint_offset


def g1_bfm_privileged_body_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    parent_idx: int,
) -> torch.Tensor:
    """Return BFM-Zero's 463-wide G1 physical-plus-derived body observation."""
    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_position = robot.data.body_link_pos_w.torch[:, body_ids]
    body_rotation = robot.data.body_link_quat_w.torch[:, body_ids]
    body_linear_velocity = robot.data.body_com_lin_vel_w.torch[:, body_ids]
    body_angular_velocity = robot.data.body_com_ang_vel_w.torch[:, body_ids]

    body_frames = append_g1_head_runtime_frame(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
        parent_body_index=parent_idx,
    )
    return g1_bfm_privileged_observation(*body_frames)


def g1_bfm_expert_target(
    robot: Articulation,
    action: G1JointPositionAction,
    table: MotionFrameSource,
    field: Callable[[str], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], object]:
    """Project physical G1 table fields onto BFM-Zero's expert target."""
    if table.joint_names != tuple(robot.joint_names):
        raise ValueError("G1 expert trajectory joints differ from the live articulation order.")
    if table.reference_frame_names != (*robot.body_names, G1_HEAD_FRAME_NAME):
        raise ValueError("G1 expert reference frames must be the live body order followed by head_link.")
    expected_shapes = {
        "joint_position": (29,),
        "joint_velocity": (29,),
        "body_position": (31, 3),
        "body_rotation": (31, 4),
        "body_linear_velocity": (31, 3),
        "body_angular_velocity": (31, 3),
    }
    actual_shapes = {name: table.field(name).shape[1:] for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise ValueError(f"G1 expert trajectory shapes differ: expected {expected_shapes}, got {actual_shapes}.")
    behavior_joint_names = tuple(action.joint_names)
    if behavior_joint_names != G1_BEHAVIOR_JOINT_NAMES or set(table.joint_names) != set(behavior_joint_names):
        raise ValueError("G1 trajectory, action, and behavior joint names differ.")
    joint_indices = torch.tensor(
        [table.joint_names.index(name) for name in behavior_joint_names], dtype=torch.int64, device=table.device
    )
    behavior_frame_names = (*G1_BEHAVIOR_BODY_NAMES, G1_HEAD_FRAME_NAME)
    if set(table.reference_frame_names) != set(behavior_frame_names):
        raise ValueError("G1 trajectory and behavior reference-frame names differ.")
    body_indices = torch.tensor(
        [table.reference_frame_names.index(name) for name in behavior_frame_names],
        dtype=torch.int64,
        device=table.device,
    )
    default_joint_position = action.joint_default_position.to(device=table.device)
    if default_joint_position.shape != (29,) or default_joint_position.device != table.device:
        raise ValueError("G1 expert projection requires 29 behavior-ordered joint defaults on the table device.")

    joint_position = field("joint_position").index_select(1, joint_indices)
    joint_velocity = field("joint_velocity").index_select(1, joint_indices)
    body_rotation = field("body_rotation").index_select(1, body_indices)
    body_angular_velocity = field("body_angular_velocity").index_select(1, body_indices)
    expert_fields = g1_bfm_expert_observation_fields(
        body_rotation[:, 0],
        body_angular_velocity[:, 0],
        joint_position,
        joint_velocity,
        default_joint_position,
    )
    body_position = field("body_position").index_select(1, body_indices)
    body_linear_velocity = field("body_linear_velocity").index_select(1, body_indices)
    expert_fields["privileged_state"] = g1_bfm_privileged_observation(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
    )
    identity = {
        "version": "bfm_behavior_axes_v2",
        "joint_names": behavior_joint_names,
        "body_names": G1_BEHAVIOR_BODY_NAMES,
        "joint_default_position": default_joint_position.detach().cpu().tolist(),
    }
    return expert_fields, identity
