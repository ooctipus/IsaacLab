# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import actor_critic_vision_cfg as encoder_cfg

@configclass
class DexsuiteKukaAllegroPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 36
    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"]
    }
    max_iterations = 15000
    save_interval = 250
    experiment_name = "dexsuite_kuka_allegro"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        encoder=encoder_cfg.ActorCriticVisionAdapterCfg(
            normalize=False,
            activation='elu',
            encoder_cfg=encoder_cfg.MLPEncoderCfg(layers=[1024, 512, 256])
        )
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
