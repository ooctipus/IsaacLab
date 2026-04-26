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


def speed_terminate(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot"), speed_limit=2.0) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    speeding = (torch.norm(wp.to_torch(robot.data.root_vel_w), dim=-1) > speed_limit) & (
        env.episode_length_buf * env.step_dt > 0.5
    )
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


def command_task_done(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Expose :attr:`MultiTaskCommand.task_done` as a :class:`DoneTerm` predicate.

    Fires when the command term reports success — all active-instant subtasks for the
    env's assigned task have been achieved. Bind with ``time_out=False`` so rsl_rl
    does not bootstrap on top of the terminal multiplicative reward.
    """
    return env.command_manager.get_term(command_name).task_done


def time_out_reach_truncate(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Timeout predicate for envs whose current task contains ≥1 instant subtask.

    Fires when ``episode_length_buf >= max_episode_length`` AND the env's task
    has an instant subtask (pure-reach or mixed). Bind with ``time_out=True``
    so rsl_rl treats this as a truncation and bootstraps ``γ·V(s_T)`` onto
    the last reward — the reach was incomplete only because the artificial
    episode cap ran out, and value should propagate through partial progress.

    Paired with :func:`time_out_track_terminate` (which covers pure-tracking
    envs with ``time_out=False``). Together they replace the single
    ``mdp.time_out`` DoneTerm.

    Reach/mixed envs always use the env's global ``max_episode_length`` — the
    adaptive curriculum only affects pure-tracking envs.
    """
    cmd = env.command_manager.get_term(command_name)
    timeout = env.episode_length_buf >= env.max_episode_length
    return timeout & cmd.spec.task_has_instant[cmd.task_samples]


def time_out_track_terminate(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Timeout predicate for envs whose current task has NO instant subtask.

    Fires when ``episode_length_buf >= effective_max_episode_length`` AND the
    env's task is pure-tracking. Bind with ``time_out=False`` so rsl_rl treats
    this as a real termination — the episode cap is the task's natural
    endpoint, and the composer's ``G = transit_mean`` is the complete
    episodic return. A bootstrap would double-count.

    The effective cap is per-env — under the adaptive episode-length
    curriculum (see :attr:`MultiTaskCfg.tracking_adaptive_err_threshold`) it
    shortens when tracking error is high and lengthens when error is low.
    When the curriculum is disabled the cap is just ``max_episode_length``.
    """
    cmd = env.command_manager.get_term(command_name)
    timeout = env.episode_length_buf >= cmd.effective_max_episode_length
    return timeout & ~cmd.spec.task_has_instant[cmd.task_samples]
