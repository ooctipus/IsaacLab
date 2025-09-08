# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg
)

@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5_reach_agent"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
