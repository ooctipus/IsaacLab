# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg

from isaaclab_tasks.manager_based.locomotion.position.config.rl_cfg import (
    RslRlCommanderActorModelCfg,
    RslRlTaskEasingActorModelCfg,
)
from isaaclab_tasks.utils import PresetCfg


def get_error(env, cmd_proposed: torch.Tensor, cmd_target: torch.Tensor):
    err = (cmd_proposed - cmd_target).clip(max=0.2)
    log = env.env.unwrapped.extras["log"]
    log["Kinematic/lin_vel_error"] = torch.linalg.vector_norm(err[:, 0:3], dim=-1)
    log["Kinematic/ang_vel_error"] = torch.linalg.vector_norm(err[:, 3:6], dim=-1)


def get_base_state(env):
    robot = env.unwrapped.scene["robot"]
    extras = env.unwrapped.extras

    current_state_w = wp.to_torch(robot.data.root_state_w).clone()
    current_time_stamp = env.unwrapped.common_step_counter
    episode_starts = (env.unwrapped.episode_length_buf == 0).unsqueeze(-1)

    if "last_root_state" not in extras:
        last_state_w = current_state_w
        last_diff = torch.zeros_like(current_state_w)
        extras["last_root_state"] = last_state_w
        extras["last_diff"] = last_diff
        extras["last_time_stamp"] = current_time_stamp
        return last_diff[:, 7:]

    last_state_w = extras["last_root_state"]
    last_diff = extras["last_diff"]
    last_time_stamp = extras["last_time_stamp"]

    last_state_w = torch.where(episode_starts, current_state_w, last_state_w)
    last_diff = torch.where(episode_starts, torch.zeros_like(last_diff), last_diff)

    if current_time_stamp <= last_time_stamp:
        extras["last_root_state"] = last_state_w
        extras["last_diff"] = last_diff
        return last_diff[:, 7:]

    state_diff_b = current_state_w - last_state_w
    last_root_quat_w = last_state_w[:, 3:7]

    state_diff_b[:, 7:10] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 7:10])
    state_diff_b[:, 10:13] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 10:13])

    extras["last_root_state"] = current_state_w
    extras["last_time_stamp"] = current_time_stamp
    extras["last_diff"] = state_diff_b
    return state_diff_b[:, 7:]


@configclass
class MultiTaskActorPresetCfg(PresetCfg):
    """Actor presets selectable via ``agent.actor=<name>``."""

    default = RslRlCommanderActorModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        commander_hidden_dims=[256, 256],
        commander_obs_normalization=True,
        kinematic_reward_weight=0.001,
        commander_loss_coef=0.1,
        get_command_target_fn=get_base_state,
        log_error_fn=get_error,
    )
    commander = RslRlCommanderActorModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        commander_hidden_dims=[256, 256],
        commander_obs_normalization=True,
        kinematic_reward_weight=0.005,
        commander_loss_coef=0.1,
        get_command_target_fn=get_base_state,
        log_error_fn=get_error,
    )
    task_easing = RslRlTaskEasingActorModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        task_easing_constraint_fn="relu",
        task_easing_loss_coef=0.0,
        task_easing_margin=0.0,
        task_easing_network="mlp",
        num_goal_refinements=2,
        goal_hidden_dims=[256, 256],
    )
    lstm = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        rnn_num_layers=1,
        rnn_hidden_dim=128,
        rnn_type="lstm",
    )
    mlp = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
    )


@configclass
class MultiTaskCriticPresetCfg(PresetCfg):
    """Critic presets selectable via ``agent.critic=<name>``."""

    default = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
    )
    lstm = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        rnn_num_layers=1,
        rnn_hidden_dim=128,
        rnn_type="lstm",
    )


@configclass
class MultiTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 200
    resume = False
    experiment_name = "position_command"
    obs_groups = {"actor": ["policy", "task"], "critic": ["policy", "task"]}

    actor = MultiTaskActorPresetCfg()  # type: ignore
    critic = MultiTaskCriticPresetCfg()  # type: ignore
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
