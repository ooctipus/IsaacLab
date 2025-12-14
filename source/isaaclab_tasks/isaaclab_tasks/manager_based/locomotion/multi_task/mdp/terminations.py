# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils.math import (quat_from_euler_xyz, quat_inv, quat_mul, euler_xyz_from_quat)
from . import states

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from .commands import MultiTaskCommand

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
    term: MultiTaskCommand = env.command_manager.get_term(command)
    err = term.get_state_error()
    speed = asset.data.body_lin_vel_w[:, robot_cfg.body_ids].norm(2, dim=-1).amax(dim=1)
    joint_pos = asset.data.joint_pos[:, robot_cfg.joint_ids] - asset.data.default_joint_pos[:, robot_cfg.joint_ids]
    joint_pos_diff = torch.abs(joint_pos).amax(dim=1)
    return ((err[:, 0] < thresh[0]) & (err[:, 1] < thresh[1])) & (speed < thresh[2]) & (joint_pos_diff < thresh[3])


def success_terminate(env: ManagerBasedRLEnv, command_name: str = "goal_point"):
    command_term: MultiTaskCommand = env.command_manager.get_term(command_name)
    return command_term.get_task_done()


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)


def speed_terminate(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot"), speed_limit=2.0) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    speeding = (torch.norm(robot.data.root_vel_w, dim=-1) > speed_limit) & (env.episode_length_buf * env.step_dt > 0.5)
    return speeding


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
    applied_torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    work_per_joint = states.mechanical_work_per_joint(applied_torque, joint_vel, env.step_dt)
    return work_per_joint.mean(dim=0)


def total_average_mech_energy_per_joint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    applied_torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    work_per_joint = states.mechanical_work_per_joint(applied_torque, joint_vel, env.step_dt)
    return work_per_joint.mean(dim=0).sum()


def mean_per_body_shock(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    per_body_incoming_wrench = torch.norm(robot.data.body_incoming_joint_wrench_b, dim=-1)
    return per_body_incoming_wrench.mean(dim=0)


def total_body_shock(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    per_body_incoming_wrench = torch.norm(robot.data.body_incoming_joint_wrench_b, dim=-1)
    return per_body_incoming_wrench.mean(dim=0).sum()


def forwardness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[asset_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    speed = torch.linalg.vector_norm(base_velocity, ord=2, dim=-1)
    forward_comp = base_velocity[:, 0]
    forward_weight = forward_comp / (speed + 1e-6)
    return forward_weight.mean(dim=0)
