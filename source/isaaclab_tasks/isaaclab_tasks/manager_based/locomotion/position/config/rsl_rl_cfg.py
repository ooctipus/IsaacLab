# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import (
    RslRlMLPEncoderModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlResidualMLPEncoderModelCfg,
    RslRlRNNModelCfg,
)

from isaaclab_tasks.utils import PresetCfg

from .rl_cfg import RslRlCommanderActorModelCfg, RslRlTaskEasingActorModelCfg

MLP_ENCODER_CFG: dict[str, RslRlMLPEncoderModelCfg.EncoderCfg] = {
    "height_scan": RslRlMLPEncoderModelCfg.EncoderCfg(
        output_dim=64,
        hidden_dims=[128, 64],
        activation="elu",
    ),
}


def get_error(env, cmd_proposed: torch.Tensor, cmd_target: torch.Tensor):
    err = (cmd_proposed - cmd_target).clip(max=0.2)
    log = env.env.unwrapped.extras["log"]
    log["Kinematic/lin_vel_error"] = torch.linalg.vector_norm(err[:, 0:3], dim=-1)
    log["Kinematic/ang_vel_error"] = torch.linalg.vector_norm(err[:, 3:6], dim=-1)


def get_base_state(env):
    robot = env.unwrapped.scene["robot"]
    extras = env.unwrapped.extras

    current_state_w = wp.to_torch(robot.data.root_state_w).clone()
    current_time_stamp = env.unwrapped.common_step_counter
    episode_starts = (env.unwrapped.episode_length_buf == 0).unsqueeze(-1)

    if "last_root_state" not in extras:
        last_state_w = current_state_w
        last_diff = torch.zeros_like(current_state_w)
        extras["last_root_state"] = last_state_w
        extras["last_diff"] = last_diff
        extras["last_time_stamp"] = current_time_stamp
        return last_diff[:, 7:]

    last_state_w = extras["last_root_state"]
    last_diff = extras["last_diff"]
    last_time_stamp = extras["last_time_stamp"]

    last_state_w = torch.where(episode_starts, current_state_w, last_state_w)
    last_diff = torch.where(episode_starts, torch.zeros_like(last_diff), last_diff)

    if current_time_stamp <= last_time_stamp:
        extras["last_root_state"] = last_state_w
        extras["last_diff"] = last_diff
        return last_diff[:, 7:]

    state_diff_b = current_state_w - last_state_w
    last_root_quat_w = last_state_w[:, 3:7]

    state_diff_b[:, 7:10] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 7:10])
    state_diff_b[:, 10:13] = quat_apply_inverse(last_root_quat_w, state_diff_b[:, 10:13])

    extras["last_root_state"] = current_state_w
    extras["last_time_stamp"] = current_time_stamp
    extras["last_diff"] = state_diff_b
    return state_diff_b[:, 7:]


@configclass
class PositionActorPresetCfg(PresetCfg):
    """Actor presets selectable via ``agent.actor=<name>``."""

    commander = RslRlCommanderActorModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        commander_hidden_dims=[256, 256],
        commander_obs_normalization=True,
        kinematic_reward_weight=0.005,
        commander_loss_coef=0.1,
        get_command_target_fn=get_base_state,
        log_error_fn=get_error,
    )
    task_easing = RslRlTaskEasingActorModelCfg(
        hidden_dims=[512, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        task_easing_constraint_fn="relu",
        task_easing_loss_coef=0.0,
        task_easing_margin=0.0,
        task_easing_network="mlp",
        num_goal_refinements=2,
        goal_hidden_dims=[256, 256],
    )
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
    encoder = RslRlMLPEncoderModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        encoder_cfg=MLP_ENCODER_CFG,
    )
    # Per-obs-group encoders remain plain MLPs; only the shared actor/critic main *body* becomes a
    # SimBa-style residual stack: pre-norm residual blocks with 4x inverted-bottleneck FFNs and a
    # post-stack LayerNorm, per equations (5)-(7) of Lee et al. 2024.
    #
    # Defaults restore the working configuration we had before the architecture correctness rewrite:
    # 2 residual blocks and ``swish`` activation. These values match the nonlinear capacity and
    # smooth-activation choice of the previous (post-norm) implementation that was training well.
    # Use ``agent.actor.num_blocks=1 agent.actor.activation=relu`` to explore the paper's exact
    # PPO defaults (Table 10) as an ablation.
    simba = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        encoder_cfg=MLP_ENCODER_CFG,
    )
    # Wider SimBa actor for capacity-scaling experiments. Same architecture as ``simba``, doubled
    # width. Critic should also switch to ``simba_big`` (which is wider AND deeper than the actor
    # per the SimBa paper's finding that critic capacity matters more than actor capacity).
    simba_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=512,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        encoder_cfg=MLP_ENCODER_CFG,
    )
    default = encoder


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
    encoder = RslRlMLPEncoderModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg=MLP_ENCODER_CFG,
    )
    # SimBa critic defaults: match the actor's width/depth/activation for a clean apples-to-apples
    # baseline. The paper's Table 10 suggests a wider critic (512) and ReLU; explore that with
    # ``agent.critic.hidden_dim=512 agent.critic.activation=relu`` as an ablation once the base
    # config is verified to learn.
    simba = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg=MLP_ENCODER_CFG,
    )
    # Scaled-up SimBa critic: 4 residual blocks of 1024 width (4x expansion -> 4096 inner width).
    # Per the SimBa paper's scaling analysis (Figure 4), critic capacity helps more than actor
    # capacity; this preset realizes that by scaling the critic more aggressively than the actor.
    simba_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=1024,
        num_blocks=4,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg=MLP_ENCODER_CFG,
    )
    default = encoder


@configclass
class PositionObsGroupsPresetCfg(PresetCfg):
    """Observation-group-mapping presets selectable via ``agent.obs_groups=<name>``.

    Switches together with the env observations preset of the same name (e.g. ``presets=encoder``
    flips both at once).
    """

    flat: dict[str, list[str]] = {"actor": ["policy", "task"], "critic": ["policy", "task"]}
    encoder: dict[str, list[str]] = {
        "actor": ["policy", "task", "height_scan"],
        "critic": ["policy", "task", "height_scan"],
    }
    # SimBa variants reuse the encoder variant's obs layout; only the main body architecture differs.
    simba: dict[str, list[str]] = encoder
    simba_big: dict[str, list[str]] = encoder
    default: dict[str, list[str]] = encoder


@configclass
class PositionLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 200
    resume = False
    experiment_name = "position_command"

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
