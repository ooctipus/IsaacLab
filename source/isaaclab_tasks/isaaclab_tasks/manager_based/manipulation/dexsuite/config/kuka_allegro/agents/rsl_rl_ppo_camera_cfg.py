# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.ext.actor_critic_vision_cfg import ActorCriticVisionAdapterCfg, CNNEncoderCfg, MLPEncoderCfg

@configclass
class DexsuiteKukaAllegroPPORunnerCameraCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 36
    obs_groups = {
        "policy": ["policy", "base_image", "wrist_image"],
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
        encoders=ActorCriticVisionAdapterCfg(
            encoder_cfgs={
                "depth_image" : CNNEncoderCfg(
                    encoding_groups=["base_image", "wrist_image"],
                    channels=[32, 64, 128],
                    kernel_sizes=[3, 3, 3],
                    strides=[2, 2, 2],
                    paddings=[1, 1, 1],
                    use_maxpool=True,
                    pool_size=2,
                    activation='elu'
                ),
                "point_cloud" : MLPEncoderCfg(
                    encoding_groups=["privileged"],
                    layers=[512, 256, 128],
                    activation='elu'
                )
            }
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
