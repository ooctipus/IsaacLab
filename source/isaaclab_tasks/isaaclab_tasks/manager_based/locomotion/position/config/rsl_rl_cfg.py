# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoCommanderActorCriticCfg
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_inv


def get_base_state(env):
    robot = env.unwrapped.scene["robot"]
    if "last_root_state" in env.unwrapped.extras:
        last_state_w = env.unwrapped.extras["last_root_state"]
        last_diff = env.unwrapped.extras["last_diff"]
        last_time_stamp = env.unwrapped.extras["last_time_stamp"]
    else:
        last_state_w = robot.data.root_state_w.clone()
        last_time_stamp = -1
        last_diff = torch.zeros_like(last_state_w)
        env.unwrapped.extras["last_root_state"] = last_state_w
        env.unwrapped.extras["last_time_stamp"] = last_time_stamp
        env.unwrapped.extras["last_diff"] = last_diff

    current_time_stamp = env.unwrapped.common_step_counter
    if current_time_stamp <= last_time_stamp:
        return last_diff

    current_state_w = robot.data.root_state_w.clone()

    state_diff_b = (current_state_w - last_state_w)
    last_root_quat_w = last_state_w[:, 3:7]
    state_diff_b[:, :3] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, :3])
    state_diff_b[:, 3:7] = quat_mul(quat_inv(last_root_quat_w), current_state_w[:, 3:7])
    state_diff_b[:, 7:10] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 7:10])
    state_diff_b[:, 10:13] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 10:13])

    env.unwrapped.extras["last_root_state"] = current_state_w
    env.unwrapped.extras["last_time_stamp"] = current_time_stamp
    env.unwrapped.extras["last_diff"] = state_diff_b
    return state_diff_b


@configclass
class PositionLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 200
    resume = False
    experiment_name = "position_command"
    obs_groups = {"policy": ["policy", "task"], "critic": ["policy", "task"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 256, 128],
        critic_hidden_dims=[512, 256, 256, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
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
    variants = {
        "policy": {
            "lstm": RslRlPpoActorCriticRecurrentCfg(
                init_noise_std=1.0,
                actor_hidden_dims=[512, 256, 256, 128],
                critic_hidden_dims=[512, 256, 256, 128],
                actor_obs_normalization=True,
                critic_obs_normalization=True,
                activation="elu",
                rnn_num_layers=1,
                rnn_hidden_dim=128,
                rnn_type="lstm",
            ),
            "commander": RslRlPpoCommanderActorCriticCfg(
                init_noise_std=1.0,
                actor_hidden_dims=[512, 256, 256, 128],
                critic_hidden_dims=[512, 256, 256, 128],
                commander_hidden_dims=[256, 256, 256],
                commander_obs_normalization=True,
                actor_obs_normalization=True,
                critic_obs_normalization=True,
                get_command_target_fn=get_base_state,
                activation="elu",
                kinematic_reward_weight=0.001
            )
        }
    }
