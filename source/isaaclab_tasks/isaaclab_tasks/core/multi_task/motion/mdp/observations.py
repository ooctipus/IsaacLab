# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named motion observations shared by both native robot profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, yaw_quat

from ..frames import append_g1_head_runtime_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_SMPL_BASE_ROTATION_CONJUGATE_XYZW = (-0.5, -0.5, -0.5, 0.5)


def _rotation_tangent_normal_xyzw(rotation: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros((rotation.shape[0], 3), dtype=rotation.dtype, device=rotation.device)
    tangent[:, 0] = 1.0
    normal = torch.zeros_like(tangent)
    normal[:, 2] = 1.0
    return torch.cat(
        (
            quat_apply(rotation, tangent),
            quat_apply(rotation, normal),
        ),
        dim=-1,
    )


def smpl_humenv_observation(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    body_angular_velocity_world: torch.Tensor,
) -> torch.Tensor:
    """Return the native 358-wide heading-local HumEnv proprioception."""
    if (
        body_position_world.shape[-2:] != (24, 3)
        or body_rotation_xyzw.shape[-2:] != (24, 4)
        or body_linear_velocity_world.shape[-2:] != (24, 3)
        or body_angular_velocity_world.shape[-2:] != (24, 3)
    ):
        raise ValueError("HumEnv observation inputs must contain the native 24 SMPL bodies.")
    batch = body_position_world.shape[0]
    root_position = body_position_world[:, 0]
    root_rotation = body_rotation_xyzw[:, 0]
    base = root_rotation.new_tensor(_SMPL_BASE_ROTATION_CONJUGATE_XYZW).expand(batch, 4)
    root_rotation = quat_mul(root_rotation, base)
    heading_inverse = quat_conjugate(yaw_quat(root_rotation))
    flat_heading = heading_inverse[:, None].expand(batch, 24, 4).reshape(-1, 4)

    local_position = body_position_world - root_position[:, None]
    local_position = quat_apply(
        flat_heading,
        local_position.reshape(-1, 3),
    ).reshape(batch, 72)[:, 3:]

    local_rotation = quat_mul(
        flat_heading,
        body_rotation_xyzw.reshape(-1, 4),
    )
    local_rotation = _rotation_tangent_normal_xyzw(local_rotation).reshape(batch, 144)

    local_linear_velocity = quat_apply(
        flat_heading,
        body_linear_velocity_world.reshape(-1, 3),
    ).reshape(batch, 72)
    local_angular_velocity = quat_apply(
        flat_heading,
        body_angular_velocity_world.reshape(-1, 3),
    ).reshape(batch, 72)
    return torch.cat(
        (
            root_position[:, 2:3],
            local_position,
            local_rotation,
            local_linear_velocity,
            local_angular_velocity,
        ),
        dim=-1,
    )


def g1_privileged_observation(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    body_angular_velocity_world: torch.Tensor,
) -> torch.Tensor:
    """Return the released 463-wide G1 heading-local body observation."""
    if (
        body_position_world.shape[-2:] != (31, 3)
        or body_rotation_xyzw.shape[-2:] != (31, 4)
        or body_linear_velocity_world.shape[-2:] != (31, 3)
        or body_angular_velocity_world.shape[-2:] != (31, 3)
    ):
        raise ValueError("G1 privileged observation inputs must contain 31 physical and derived reference frames.")
    batch = body_position_world.shape[0]
    root_position = body_position_world[:, 0]
    heading = quat_conjugate(yaw_quat(body_rotation_xyzw[:, 0]))
    flat_heading = heading[:, None].expand(batch, 31, 4).reshape(-1, 4)
    local_position = quat_apply(
        flat_heading,
        (body_position_world - root_position[:, None]).reshape(-1, 3),
    ).reshape(batch, 93)[:, 3:]
    local_rotation = quat_mul(
        flat_heading,
        body_rotation_xyzw.reshape(-1, 4),
    )
    local_rotation = _rotation_tangent_normal_xyzw(local_rotation).reshape(batch, 186)
    local_linear_velocity = quat_apply(
        flat_heading,
        body_linear_velocity_world.reshape(-1, 3),
    ).reshape(batch, 93)
    local_angular_velocity = quat_apply(
        flat_heading,
        body_angular_velocity_world.reshape(-1, 3),
    ).reshape(batch, 93)
    return torch.cat(
        (
            root_position[:, 2:3],
            local_position,
            local_rotation,
            local_linear_velocity,
            local_angular_velocity,
        ),
        dim=-1,
    )


def g1_released_expert_state(
    root_rotation_xyzw: torch.Tensor,
    root_angular_velocity_world: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    default_joint_position: torch.Tensor,
) -> torch.Tensor:
    """Return the released 64-wide BFM expert state, including its world-angular convention."""
    gravity_world = torch.zeros_like(root_angular_velocity_world)
    gravity_world[..., 2] = -1.0
    projected_gravity = quat_apply(quat_conjugate(root_rotation_xyzw), gravity_world)
    return torch.cat(
        (
            joint_position - default_joint_position,
            joint_velocity,
            projected_gravity,
            root_angular_velocity_world,
        ),
        dim=-1,
    )


def _payload(env: ManagerBasedRLEnv, command_name: str):
    """Return the motion payload at the public command-term boundary."""
    return env.command_manager.get_term(command_name).payload


def smpl_body_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the native 358-wide SMPL heading-local body observation."""
    robot = env.scene[asset_cfg.name]
    return smpl_humenv_observation(
        robot.data.body_link_pos_w.torch[:, asset_cfg.body_ids],
        robot.data.body_link_quat_w.torch[:, asset_cfg.body_ids],
        robot.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids],
        robot.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids],
    )


def motion_joint_position(
    env: ManagerBasedRLEnv,
    action_name: str,
) -> torch.Tensor:
    """Return behavior-ordered joint displacement from the episodic default pose [rad]."""
    from .actions import MotionJointPositionAction

    action = env.action_manager.get_term(action_name)
    if not isinstance(action, MotionJointPositionAction):
        raise TypeError("motion_joint_position requires MotionJointPositionAction.")
    return action.joint_position - action.joint_default_position - action.default_joint_offset


def motion_joint_velocity(
    env: ManagerBasedRLEnv,
    action_name: str,
) -> torch.Tensor:
    """Return joint velocity [rad/s] in behavior-action order."""
    from .actions import MotionJointPositionAction

    action = env.action_manager.get_term(action_name)
    if not isinstance(action, MotionJointPositionAction):
        raise TypeError("motion_joint_velocity requires MotionJointPositionAction.")
    return action.joint_velocity


def motion_projected_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return unit gravity expressed in the root frame."""
    return env.scene[asset_cfg.name].data.projected_gravity_b.torch


def motion_root_angular_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root angular velocity [rad/s] in the root frame."""
    return env.scene[asset_cfg.name].data.root_ang_vel_b.torch


def motion_last_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return the controller-normalized action from the preceding physical edge."""
    return env.action_manager.get_term(action_name).processed_actions


def motion_history(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the payload-owned field-major applied-transition history."""
    return _payload(env, command_name).history_value


def g1_privileged_body_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    parent_body_index: int,
) -> torch.Tensor:
    """Return the native 463-wide G1 physical-plus-synthetic body observation."""
    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if (
        not isinstance(body_ids, list)
        or len(body_ids) != 30
        or len(set(body_ids)) != 30
        or parent_body_index < 0
        or parent_body_index >= len(body_ids)
    ):
        raise ValueError("G1 privileged observation requires 30 behavior bodies and one valid parent index.")

    body_position = robot.data.body_link_pos_w.torch[:, body_ids]
    body_rotation = robot.data.body_link_quat_w.torch[:, body_ids]
    body_linear_velocity = robot.data.body_com_lin_vel_w.torch[:, body_ids]
    body_angular_velocity = robot.data.body_com_ang_vel_w.torch[:, body_ids]

    (
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
    ) = append_g1_head_runtime_frame(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
        parent_body_index=parent_body_index,
    )
    return g1_privileged_observation(
        body_position,
        body_rotation,
        body_linear_velocity,
        body_angular_velocity,
    )
