# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg

from . import states

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import RelativeStateCommand

"""
MDP terminations.
"""


def success(
    env: ManagerBasedRLEnv,
    thresh: list[float, float, float, float],
    command: str = "goal_point",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[robot_cfg.name]
    term: RelativeStateCommand = env.command_manager.get_term(command)
    err = term.get_state_error()
    speed = wp.to_torch(asset.data.body_lin_vel_w)[:, robot_cfg.body_ids].norm(2, dim=-1).amax(dim=1)
    joint_pos = (
        wp.to_torch(asset.data.joint_pos)[:, robot_cfg.joint_ids]
        - wp.to_torch(asset.data.default_joint_pos)[:, robot_cfg.joint_ids]
    )
    joint_pos_diff = torch.abs(joint_pos).amax(dim=1)
    return ((err[:, 0] < thresh[0]) & (err[:, 1] < thresh[1])) & (speed < thresh[2]) & (joint_pos_diff < thresh[3])


def success_terminate(env: ManagerBasedRLEnv, command_name: str = "goal_point"):
    command_term: RelativeStateCommand = env.command_manager.get_term(command_name)
    return command_term.get_task_done()


def abnormal_robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_body_lin_vel: float = 100.0,
    max_body_ang_vel: float = 100.0,
) -> torch.Tensor:
    """Terminate environments when robot velocities indicate unstable physics.

    Args:
        env: The manager-based RL environment.
        asset_cfg: The robot asset configuration.
        max_body_lin_vel: Maximum body linear speed [m/s].
        max_body_ang_vel: Maximum body angular speed [rad/s].

    Returns:
        Boolean tensor indicating environments with abnormal robot state.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    joint_vel = wp.to_torch(robot.data.joint_vel)[:, asset_cfg.joint_ids]
    joint_vel_limits = wp.to_torch(robot.data.joint_vel_limits)[:, asset_cfg.joint_ids]
    joint_vel_abnormal = (joint_vel.abs() > (joint_vel_limits * 2.0)).any(dim=1)

    # body_lin_speed = wp.to_torch(robot.data.body_lin_vel_w)[:, asset_cfg.body_ids].norm(dim=-1).amax(dim=1)
    # body_ang_speed = wp.to_torch(robot.data.body_ang_vel_w)[:, asset_cfg.body_ids].norm(dim=-1).amax(dim=1)
    # # print(body_lin_speed)
    # # print(body_ang_speed)
    # body_vel_abnormal = (body_lin_speed > max_body_lin_vel) | (body_ang_speed > max_body_ang_vel)

    return joint_vel_abnormal  # | body_vel_abnormal


def speed_terminate(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot"), speed_limit=2.0) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    speeding = (torch.norm(wp.to_torch(robot.data.root_vel_w), dim=-1) > speed_limit) & (
        env.episode_length_buf * env.step_dt > 0.5
    )
    return speeding


class joint_reaction_overload(ManagerTermBase):
    r"""Terminate when any joint's internal reaction force exceeds a body-weight multiple.

    The :class:`~isaaclab.sensors.JointWrenchSensor` reports the reaction wrench transmitted
    through each joint -- the internal load that propagates up the kinematic chain on impact,
    which the foot-ground contact sensor does not capture. A hard landing (e.g. jumping off a
    platform instead of taking the stairs) drives this reaction force far above the level seen in
    normal locomotion.

    The threshold is grounded as a multiple of total bodyweight, mirroring the foot-contact impact
    gate (:func:`illegal_contact_ratio`). In-vivo biomechanics measures internal joint contact
    forces of ~2-3.5x bodyweight in walking/stairs, ~5x in running, and ~8-9x in stumbles, so a
    cutoff around 6x bodyweight sits above vigorous gait but inside the "stumble/abuse" band.
    Runtime probing on this task agreed: normal gait peaks near ~4x bodyweight while jump-downs
    spike past 10x.

    Args:
        force_ratio: Multiple of total bodyweight a single joint's reaction force may reach before
            the episode terminates.
        sensor_cfg: Joint-wrench sensor to read. Defaults to ``SceneEntityCfg("joint_wrench")``.
        asset_cfg: Articulation whose total mass defines bodyweight. Defaults to
            ``SceneEntityCfg("robot")``.
    """

    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        force_ratio = float(cfg.params["force_ratio"])
        sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg", SceneEntityCfg("joint_wrench"))
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._sensor = env.scene[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        # [num_envs, 1] for broadcast against per-joint reaction force [num_envs, num_joints].
        total_mass = wp.to_torch(asset.data.body_mass).sum(dim=-1)
        self._threshold = (force_ratio * total_mass * 9.81).unsqueeze(-1)
        # Manager will not pass kwargs back to ``__call__`` if cfg.params is empty.
        cfg.params = {}

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        force = wp.to_torch(self._sensor.data.force)  # [num_envs, num_joints, 3] reaction force [N]
        max_force = torch.linalg.norm(force, dim=-1)  # [num_envs, num_joints] per-joint magnitude [N]
        return torch.any(max_force > self._threshold, dim=1)


class log(ManagerTermBase):
    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        log_key = cfg.params.get("log_key")
        category = cfg.params.get("category", None)
        prefix = "Info"
        if isinstance(log_key, str):
            if log_key.startswith("eval"):
                log_key = eval(log_key[5:])
        if category is not None and isinstance(category, str):
            prefix = f"{prefix}/{category}"
        self.return_val = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        if isinstance(log_key, str):
            self.log = f"{prefix}/{log_key}"
        elif isinstance(log_key, list):
            self.log = [f"{prefix}/{key}" for key in log_key]
        else:
            raise KeyError("input key is neither str or list of str")
        self.func: callable = cfg.params.get("func")
        self.params: dict = {key: val for key, val in cfg.params.items() if key not in ["func", "log_key", "category"]}
        cfg.params = {}

    def __call__(self, env: ManagerBasedRLEnv):
        val = self.func(env, **self.params)
        env_log = env.extras["log"]
        if isinstance(self.log, str):
            env_log[self.log] = val
        elif isinstance(self.log, list):
            for i, key in enumerate(self.log):
                env_log[key] = float(val[i])
        return self.return_val


def mean_mech_energy_per_joint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    applied_torque = wp.to_torch(robot.data.applied_torque)[:, asset_cfg.joint_ids]
    joint_vel = wp.to_torch(robot.data.joint_vel)[:, asset_cfg.joint_ids]
    work_per_joint = states.mechanical_work_per_joint(applied_torque, joint_vel, env.step_dt)
    return work_per_joint.mean(dim=0)


def total_average_mech_energy_per_joint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    applied_torque = wp.to_torch(robot.data.applied_torque)[:, asset_cfg.joint_ids]
    joint_vel = wp.to_torch(robot.data.joint_vel)[:, asset_cfg.joint_ids]
    work_per_joint = states.mechanical_work_per_joint(applied_torque, joint_vel, env.step_dt)
    return work_per_joint.mean(dim=0).sum()


def mean_per_body_shock(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    per_body_incoming_wrench = torch.norm(wp.to_torch(robot.data.body_incoming_joint_wrench_b), dim=-1)
    return per_body_incoming_wrench.mean(dim=0)


def total_body_shock(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    per_body_incoming_wrench = torch.norm(wp.to_torch(robot.data.body_incoming_joint_wrench_b), dim=-1)
    return per_body_incoming_wrench.mean(dim=0).sum()


def forwardness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[asset_cfg.name]
    base_velocity = wp.to_torch(robot.data.root_lin_vel_b)  # Robot's current base velocity vector
    speed = torch.linalg.vector_norm(base_velocity, ord=2, dim=-1)
    forward_comp = base_velocity[:, 0]
    forward_weight = forward_comp / (speed + 1e-6)
    return forward_weight.mean(dim=0)
