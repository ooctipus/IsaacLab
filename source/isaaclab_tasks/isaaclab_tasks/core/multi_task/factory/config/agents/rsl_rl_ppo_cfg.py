# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSuccessorCfg,
    RslRlValueShiftCfg,
)

from isaaclab_tasks.core.multi_task.rl.rsl_rl import (
    RslRlResidualMLPEncoderModelCfg,
    RslRlSuccessorFeatureCriticModelCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

# Shared PPO hyper-parameters reused by both the plain-PPO and value-shift variants.
_FACTORY_PPO_KWARGS: dict[str, Any] = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=6e-3,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.999,
    lam=0.85,
    desired_kl=0.01,
    max_grad_norm=1.0,
    optimizer=preset(default="adamw", adamw="adamw", adam="adam"),
    weight_decay=preset(default=0.01, adamw=0.01, adam=0.0),
)


@configclass
class FactoryActorPresetCfg(PresetCfg):
    """Actor presets selectable via ``agent.actor=<name>``."""

    actor_critic = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    simba = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        encoder_cfg={},
    )
    simba_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=512,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        encoder_cfg={},
    )
    default = actor_critic


@configclass
class FactoryCriticPresetCfg(PresetCfg):
    """Critic presets selectable via ``agent.critic=<name>``."""

    actor_critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    simba = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg={},
    )
    simba_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=1024,
        num_blocks=4,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg={},
    )
    # Successor-representation critic: two heads (``psi`` free, ``phi`` hard-normed to sqrt(d)) vs a scalar value;
    # the value is derived in the ``successor`` algorithm extension as ``V = <psi, w>``. Pair with the
    # ``successor`` algorithm preset, whose ``feature_dim`` must match the one set here.
    successor = RslRlSuccessorFeatureCriticModelCfg(
        feature_dim=128,
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg={},
    )
    default = actor_critic


@configclass
class ValueShiftAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with post-update critic drift bound to the Factory curriculum."""

    value_shift_cfg: RslRlValueShiftCfg = RslRlValueShiftCfg(
        observation_bind=(
            "setattr(self, 'obs_cache', env.unwrapped.curriculum_manager"
            ".get_term('reset_sampler').value_shift.observation_cache)"
        ),
        current_value_bind=(
            "setattr(self, 'cur_val', env.unwrapped.curriculum_manager.get_term('reset_sampler').value_shift.cur_val)"
        ),
        value_diff_bind=(
            "setattr(self, 'diff_val', env.unwrapped.curriculum_manager.get_term('reset_sampler').value_shift.diff_val)"
        ),
    )


@configclass
class SuccessorLatentAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm cfg + successor-latent value loss coefficient."""

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:SuccessorLatentPPO"
    successor_loss_coef: float = 0.01
    """Coefficient for the vector-valued successor-latent loss."""


@configclass
class SuccessorAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with the dynamics-anchored successor-representation value
    (:class:`~rsl_rl.extensions.SuccessorFeatures`).

    The critic learns forward/backward maps from the reward-free occupancy objective. The goal cache supplies
    ``z = B(goal)``; actor and value share that immutable behavior context, with
    ``V(s, z) = <F(s, z), z>``. Select this through global ``presets=successor``, which installs the matching
    actor, critic, observation groups, and environment-side ``goal_observations`` cache. Selecting only
    ``agent.algorithm=successor`` is incomplete. ``feature_dim`` must match the critic preset.
    """

    successor_cfg: RslRlSuccessorCfg = RslRlSuccessorCfg(
        feature_dim=128,
        goal_observation_bind=("env.unwrapped.curriculum_manager.get_term('goal_observations').observations"),
        goal_indices_bind="env.unwrapped.command_manager.get_term('reset_state').cmd_indices",
    )


@configclass
class PpoAlgorithmCfg(PresetCfg):
    """Algorithm preset selectable via ``agent.algorithm=<name>``."""

    actor_critic = RslRlPpoAlgorithmCfg(class_name="PPO", **_FACTORY_PPO_KWARGS)
    default = actor_critic
    value_shift = ValueShiftAlgorithmCfg(**_FACTORY_PPO_KWARGS)
    beta_value_shift = value_shift
    successor_latent = SuccessorLatentAlgorithmCfg(**_FACTORY_PPO_KWARGS)
    successor = SuccessorAlgorithmCfg(**_FACTORY_PPO_KWARGS)


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "factory"
    obs_groups = preset(
        default={"actor": ["policy"], "critic": ["policy"]},
        actor_critic={"actor": ["policy"], "critic": ["policy"]},
        simba={"actor": ["policy"], "critic": ["policy"]},
        simba_big={"actor": ["policy"], "critic": ["policy"]},
    )  # type: ignore
    actor = FactoryActorPresetCfg()  # type: ignore
    critic = FactoryCriticPresetCfg()  # type: ignore
    algorithm = PpoAlgorithmCfg()  # type: ignore
