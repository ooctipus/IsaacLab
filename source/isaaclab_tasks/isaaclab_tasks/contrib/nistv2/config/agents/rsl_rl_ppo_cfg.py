# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg, preset


@configclass
class SimBaModelCfg:
    """Configuration for a SimBa model with optional observation-group encoders."""

    @configclass
    class EncoderCfg:
        class_name: str = MISSING
        """Encoder class or constructor."""

        output_dim: int = MISSING
        """Output dimension of the encoder."""

    @configclass
    class MLPEncoderCfg(EncoderCfg):
        class_name: str = "isaaclab_tasks.contrib.nistv2.config.agents.models:MLPEncoder"

        hidden_dims: list[int] = MISSING
        """Hidden dimensions of the encoder MLP."""

        activation: str = MISSING
        """Activation function of the encoder MLP."""

        last_activation: str | None = None
        """Optional activation after the encoder output layer."""

    class_name: str = "isaaclab_tasks.contrib.nistv2.config.agents.models:SimBaModel"
    """Model class name."""

    hidden_dim: int = MISSING
    """Width of the residual pathway."""

    num_blocks: int = 2
    """Number of residual blocks."""

    expansion_factor: int = 4
    """Expansion factor inside each residual block."""

    activation: str = "relu"
    """Activation function used inside each residual block."""

    norm: bool = True
    """Whether to apply layer normalization inside the SimBa head."""

    obs_normalization: bool = False
    """Whether to normalize observation groups that bypass encoders."""

    encoder_normalization: bool = False
    """Whether to normalize each encoder input independently."""

    encoder_cfg: dict[str, EncoderCfg] | None = None
    """Encoders keyed by observation group."""

    distribution_cfg: RslRlMLPModelCfg.DistributionCfg | None = None
    """Optional output-distribution configuration."""


# Shared PPO hyper-parameters reused by both the plain-PPO and value-shift variants.
_FACTORY_PPO_KWARGS = dict(
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
)


@configclass
class PpoAlgorithmCfg(PresetCfg):
    actor_critic = RslRlPpoAlgorithmCfg(class_name="PPO", **_FACTORY_PPO_KWARGS)
    default = actor_critic


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "factory"
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
    algorithm = PpoAlgorithmCfg()  # type: ignore
