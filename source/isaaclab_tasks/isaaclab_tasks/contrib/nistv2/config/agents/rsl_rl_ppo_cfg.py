# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for complete-board Factory training."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
    RslRlStateCurriculumCfg,
    RslRlSuccessEstimatorCfg,
    RslRlValueShiftCfg,
)

from isaaclab_tasks.contrib.nist.config.agents.models import SimBaModelCfg
from isaaclab_tasks.contrib.nist.config.agents.rsl_rl_ppo_cfg import FactoryPPORunnerCfg
from isaaclab_tasks.utils import preset


@configclass
class FactoryBoardPPORunnerCfg(FactoryPPORunnerCfg):
    """PPO runner using the Variant policy and perception contract."""

    init_at_random_ep_len = False
    obs_groups = preset(
        default={"actor": ["policy", "perception"], "critic": ["policy", "perception"]},
        actor_critic={"actor": ["policy", "perception"], "critic": ["policy", "perception"]},
    )  # type: ignore

    actor = SimBaModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dim=256,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
        encoder_cfg={
            "perception": SimBaModelCfg.MLPEncoderCfg(
                hidden_dims=[256], output_dim=128, activation="elu", last_activation="elu"
            )
        },
    )
    critic = SimBaModelCfg(
        obs_normalization=True,
        hidden_dim=256,
        num_blocks=2,
        expansion_factor=4,
        activation="swish",
        encoder_cfg={
            "perception": SimBaModelCfg.MLPEncoderCfg(
                hidden_dims=[256], output_dim=128, activation="elu", last_activation="elu"
            )
        },
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=6e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.90,
        desired_kl=0.01,
        max_grad_norm=1.0,
        state_curriculum_cfg=RslRlStateCurriculumCfg(
            value_shift_cfg=preset(
                default=None,
                value_shift=RslRlValueShiftCfg(),
                value_shift_005=RslRlValueShiftCfg(),
                value_shift_05=RslRlValueShiftCfg(),
            ),
            success_estimator_cfg=preset(
                default=None,
                success_estimator=RslRlSuccessEstimatorCfg(
                    hidden_dims=[256, 256],
                    activation="elu",
                    learning_rate=1.0e-4,
                    optimizer="adam",
                    max_grad_norm=1.0,
                ),
            ),
        ),
    )
