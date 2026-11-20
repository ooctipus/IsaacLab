# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the in-hand reorientation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ..clearance import ObjectClearance
from ..utils import sample_joint_positions_within_limits

if TYPE_CHECKING:
    import torch

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def reset_reorient_hand(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    joint_position_noise: float,
    joint_velocity_noise: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the hand's joints and the position targets tracking them.

    Task-local rather than :func:`~isaaclab.envs.mdp.reset_joints_by_offset`: the
    hand's PD targets must be re-seeded alongside the joint state, and the framework
    terms write joint state only.

    Args:
        env: Environment containing the robot.
        env_ids: Environment indices to reset.
        joint_position_noise: Scale applied to sampled joint-position deltas.
        joint_velocity_noise: Joint-velocity noise half-width [rad/s].
        robot_cfg: Robot scene entity.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    default_position = robot.data.default_joint_pos.torch[env_ids]
    limits = robot.data.joint_limits.torch[env_ids]
    joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
    velocity_sample = math_utils.sample_uniform(-1.0, 1.0, default_position.shape, device=env.device)
    joint_velocity = robot.data.default_joint_vel.torch[env_ids] + joint_velocity_noise * velocity_sample
    robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)


class reset_object_clear_of_hand(ManagerTermBase):
    """Reset the object to a jittered default position with a uniformly random orientation, clear of the hand.

    The draw is lifted against gravity by the least distance that clears every collider of the hand at the joint
    positions the hand's reset just wrote (:class:`~isaaclab_tasks.core.reorient.clearance.ObjectClearance`), so an
    episode never starts with the object inside a finger; the object then drops into the hand. Apply it after the
    hand's reset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self._robot: Articulation = env.scene[robot_cfg.name]
        self._object: RigidObject = env.scene[object_cfg.name]
        self._clearance = ObjectClearance(self._robot, self._object, env.sim.cfg.gravity)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> None:
        """Draw, clear and write the object's pose; zero its velocity.

        Args:
            env: Environment containing the hand and the object.
            env_ids: Environment indices to reset.
            pose_range: Position ranges about the default position, keyed ``x``, ``y``, ``z`` [m].
            robot_cfg: Hand scene entity.
            object_cfg: Object scene entity.
        """
        pose = self._object.data.default_root_pose.torch[env_ids].clone()
        offset = math_utils.sample_uniform_from_ranges(pose_range, ("x", "y", "z"), len(pose), device=env.device)
        pose[:, :3] += env.scene.env_origins[env_ids] + offset
        pose[:, 3:7] = math_utils.random_orientation(len(pose), device=env.device)
        self._clearance.raise_clear(pose, env_ids, env.sim)
        self._object.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._object.write_root_velocity_to_sim_index(
            root_velocity=self._object.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
