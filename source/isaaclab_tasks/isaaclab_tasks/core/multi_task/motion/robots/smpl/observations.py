# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SMPL HumEnv observations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul

from ....mdp.observations import body_heading_local_observation

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ...data import MotionFrameSource


_SMPL_BASE_ROTATION_CONJUGATE_XYZW = (-0.5, -0.5, -0.5, 0.5)


def smpl_humenv_observation(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    body_angular_velocity_world: torch.Tensor,
) -> torch.Tensor:
    """Return the native 358-wide heading-local HumEnv proprioception."""
    batch = body_position_world.shape[0]
    base = body_rotation_xyzw.new_tensor(_SMPL_BASE_ROTATION_CONJUGATE_XYZW).expand(batch, 4)
    heading_rotation = quat_mul(body_rotation_xyzw[:, 0], base)
    return body_heading_local_observation(
        body_position_world,
        body_rotation_xyzw,
        body_linear_velocity_world,
        body_angular_velocity_world,
        heading_rotation,
    )


def smpl_humenv_body_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the native 358-wide SMPL heading-local body observation."""
    robot = env.scene[asset_cfg.name]
    body_state = (
        robot.data.body_link_pos_w.torch[:, asset_cfg.body_ids],
        robot.data.body_link_quat_w.torch[:, asset_cfg.body_ids],
        robot.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids],
        robot.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids],
    )
    return smpl_humenv_observation(*body_state)


def smpl_humenv_tracking_pose(observation: torch.Tensor) -> torch.Tensor:
    """Return the 214-wide HumEnv pose used by uniform-assignment tracking."""
    if observation.ndim != 2 or observation.shape[1] != 358:
        raise ValueError("SMPL HumEnv tracking requires one 358-wide observation per row.")
    return observation[:, :214]


def smpl_expert_target(
    robot: Articulation,
    table: MotionFrameSource,
    field: Callable[[str], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], object]:
    """Project physical SMPL table fields onto the learner's expert target."""
    if table.reference_frame_names != tuple(robot.body_names):
        raise ValueError("SMPL expert reference frames must match the live articulation body order.")
    expected_shapes = {
        "body_position": (24, 3),
        "body_rotation": (24, 4),
        "body_linear_velocity": (24, 3),
        "body_angular_velocity": (24, 3),
    }
    actual_shapes = {name: table.field(name).shape[1:] for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise ValueError(f"SMPL expert body shapes differ: expected {expected_shapes}, got {actual_shapes}.")
    frames = smpl_humenv_observation(
        field("body_position"),
        field("body_rotation"),
        field("body_linear_velocity"),
        field("body_angular_velocity"),
    )
    return {"policy": frames}, "smpl_heading_local_physical_body_projection_v1"
