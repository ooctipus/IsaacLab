# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg, preset

from .estimator_ppo_algorithm_cfg import SuccessEstimatorPpoAlgorithmCfg


@configclass
class PpoAlgorithmCfg(PresetCfg):
    actor_critic = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=6e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        returns_method="gae",
    )
    success_estimator = SuccessEstimatorPpoAlgorithmCfg(
        class_name="SuccessEstimatorPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=6e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        returns_method="gae",
        success_estimator_learning_rate=1.0e-4,
        success_loss_coef=1.0,
        success_returns_method="hindsight_mc",
    )
    default = actor_critic


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "factory"
    obs_groups = preset(
        default={"actor": ["policy"], "critic": ["policy"]},
        actor_critic={"actor": ["policy"], "critic": ["policy"]},
        success_estimator={"actor": ["policy"], "critic": ["policy"], "success_estimator": ["success"]},
    )  # type: ignore
    actor = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    success_estimator_bind = preset(
        default=None,
        success_estimator=(
            'env.unwrapped.termination_manager.get_term_cfg("predictor_truncation").func.bind(alg.success_predictions)'
            ' or alg._loss_callback.append(env.unwrapped.termination_manager.get_term_cfg("predictor_truncation").func.update_loss)'
        ),
    )
    state_buffer_bind = 'setattr(alg, "_state_buffer", env.unwrapped.event_manager.get_term_cfg("reset_strategies").func._shared_buffer)'
    success_estimator = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = PpoAlgorithmCfg()  # type: ignore
