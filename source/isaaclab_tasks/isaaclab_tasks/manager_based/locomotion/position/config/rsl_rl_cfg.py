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
    extras = env.unwrapped.extras

    current_state_w = robot.data.root_state_w.clone()
    current_time_stamp = env.unwrapped.common_step_counter
    episode_starts = (env.unwrapped.episode_length_buf == 0).unsqueeze(-1)  # [N, 1] for broadcasting

    # First call: initialize buffers
    if "last_root_state" not in extras:
        last_state_w = current_state_w
        last_diff = torch.zeros_like(current_state_w)
        extras["last_root_state"] = last_state_w
        extras["last_diff"] = last_diff
        extras["last_time_stamp"] = current_time_stamp
        return last_diff

    # Load previous state
    last_state_w = extras["last_root_state"]
    last_diff = extras["last_diff"]
    last_time_stamp = extras["last_time_stamp"]

    # Handle per-env new episodes: reset baseline and diff to zero for those envs
    last_state_w = torch.where(episode_starts, current_state_w, last_state_w)
    last_diff = torch.where(episode_starts, torch.zeros_like(last_diff), last_diff)

    # If we've already computed diff for this global step, just reuse it
    if current_time_stamp <= last_time_stamp:
        # Keep extras consistent with the masked versions
        extras["last_root_state"] = last_state_w
        extras["last_diff"] = last_diff
        return last_diff

    # Normal case: compute diff in body frame
    state_diff_b = (current_state_w - last_state_w)
    last_root_quat_w = last_state_w[:, 3:7]

    # position
    state_diff_b[:, :3] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, :3])
    # rotation
    state_diff_b[:, 3:7] = quat_mul(quat_inv(last_root_quat_w), current_state_w[:, 3:7])
    # lin vel
    state_diff_b[:, 7:10] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 7:10])
    # ang vel
    state_diff_b[:, 10:13] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 10:13])

    extras["last_root_state"] = current_state_w
    extras["last_time_stamp"] = current_time_stamp
    extras["last_diff"] = state_diff_b
    return state_diff_b


@configclass
class PositionLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 200
    resume = False
    experiment_name = "position_command"
    obs_groups = {"policy": ["policy", "task"], "critic": ["policy", "task"]}
    policy = RslRlPpoCommanderActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 256, 128],
        critic_hidden_dims=[512, 256, 256, 128],
        commander_hidden_dims=[256, 256, 256],
        commander_obs_normalization=True,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        get_command_target_fn=get_base_state,
        activation="elu",
        kinematic_reward_weight=0.001,
        commander_loss_coef=1.0
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
                kinematic_reward_weight=0.01,
                commander_loss_coef=5.0
            )
        }
    }
