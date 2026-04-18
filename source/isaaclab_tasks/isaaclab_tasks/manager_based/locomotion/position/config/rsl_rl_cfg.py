# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

from isaaclab_tasks.utils import PresetCfg

from ..mdp_presets import ExperimentNameCfg


@configclass
class PositionActorPresetCfg(PresetCfg):
    """Actor presets selectable via ``agent.actor=<name>``."""

    lstm = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        rnn_num_layers=1,
        rnn_hidden_dim=128,
        rnn_type="lstm",
    )
    flat = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    default = flat


@configclass
class PositionCriticPresetCfg(PresetCfg):
    """Critic presets selectable via ``agent.critic=<name>``."""

    flat = RslRlMLPModelCfg(
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
    default = flat


@configclass
class PositionObsGroupsPresetCfg(PresetCfg):
    """Observation-group-mapping presets selectable via ``agent.obs_groups=<name>``.

    Switches together with the env observations preset of the same name (e.g. ``presets=encoder``
    flips both at once).
    """

    flat: dict[str, list[str]] = {
        "actor": ["policy", "task", "height_scan"],
        "critic": ["policy", "task", "height_scan"],
    }
    encoder: dict[str, list[str]] = {
        "actor": ["policy", "task", "height_scan"],
        "critic": ["policy", "task", "height_scan"],
    }
    default: dict[str, list[str]] = encoder


@configclass
class PositionLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 200
    resume = False
    experiment_name: str = ExperimentNameCfg()  # type: ignore
    obs_groups: PositionObsGroupsPresetCfg = PositionObsGroupsPresetCfg()  # type: ignore
    actor = PositionActorPresetCfg()  # type: ignore
    critic = PositionCriticPresetCfg()  # type: ignore
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
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
