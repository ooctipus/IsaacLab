# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg, preset


@configclass
class PointCloudModelCfg(RslRlMLPModelCfg):
    """Configuration for the ordered Factory point-cloud encoder."""

    @configclass
    class MLPCfg:
        hidden_dims: list[int] = MISSING
        """Hidden dimensions of the point-cloud MLP."""

        output_dim: int = MISSING
        """Output dimension of the point-cloud MLP."""

        activation: str = MISSING
        """Activation function of the point-cloud MLP."""

    class_name: str = "isaaclab_tasks.contrib.nistv2.config.agents.models:PointCloudModel"
    point_cloud_group: str = "perception"
    point_cloud_mlp_cfg: MLPCfg = MISSING


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
    actor = PointCloudModelCfg(
        distribution_cfg=PointCloudModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        point_cloud_mlp_cfg=PointCloudModelCfg.MLPCfg(
            hidden_dims=[256],
            output_dim=128,
            activation="elu",
        ),
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    critic = PointCloudModelCfg(
        obs_normalization=True,
        point_cloud_mlp_cfg=PointCloudModelCfg.MLPCfg(
            hidden_dims=[256],
            output_dim=128,
            activation="elu",
        ),
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = PpoAlgorithmCfg()  # type: ignore
