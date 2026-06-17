# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor / critic model configurations for the position-locomotion env.

Pure "model zoo" — module-level constants of rsl_rl model cfgs, no preset
wrappers. The preset dispatchers that compose them (``PositionActorPresetCfg``,
``PositionCriticPresetCfg``) live in :mod:`rsl_rl_cfg`, keeping model definitions
separate from preset / runner assembly.

Shared pieces (``MLP_ENCODER_CFG``, the commander-actor helper functions) are
defined here too since they only get referenced by the model cfgs.
"""

import torch
import warp as wp

from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)

from isaaclab_tasks.core.multi_task.rl.rsl_rl.rl_cfg import (
    RslRlCommanderActorModelCfg,
    RslRlMLPEncoderModelCfg,
    RslRlResidualMLPEncoderModelCfg,
    RslRlTaskEasingActorModelCfg,
)
from isaaclab_tasks.utils import preset

# ---------------------------------------------------------------------------
# Shared pieces.
# ---------------------------------------------------------------------------

MLP_ENCODER_CFG: dict[str, RslRlMLPEncoderModelCfg.EncoderCfg] = {
    "height_scan": RslRlMLPEncoderModelCfg.EncoderCfg(
        output_dim=64,
        hidden_dims=[128, 64],
        activation="elu",
    ),
}

CNN_ENCODER_CFG: dict[str, RslRlCNNModelCfg.CNNCfg] = {
    "height_scan": RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32],
        kernel_size=[3, 3],
        stride=[2, 2],
        activation="elu",
    ),
}

CNN_MODEL_ENCODER_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[16, 32, 64],
    kernel_size=[5, 5, 4],
    stride=[2, 2, 1],
    activation="elu",
)


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


# ---------------------------------------------------------------------------
# Actor model cfgs.
# ---------------------------------------------------------------------------

COMMANDER_ACTOR = RslRlCommanderActorModelCfg(
    hidden_dims=[512, 256, 128, 64],
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

TASK_EASING_ACTOR = RslRlTaskEasingActorModelCfg(
    hidden_dims=[512, 256, 128, 64],
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

LSTM_ACTOR = RslRlRNNModelCfg(
    hidden_dims=[512, 256, 128, 64],
    activation="elu",
    obs_normalization=True,
    stochastic=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    rnn_num_layers=1,
    rnn_hidden_dim=128,
    rnn_type="lstm",
)

FLAT_ACTOR = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128, 64],
    activation="elu",
    obs_normalization=True,
    stochastic=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
)

ENCODER_ACTOR = RslRlMLPEncoderModelCfg(
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
SIMBA_MLP_ACTOR = RslRlResidualMLPEncoderModelCfg(
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

# Wider SimBa actor for capacity-scaling experiments. Same architecture as ``SIMBA_ACTOR``, doubled
# width. Critic should also switch to ``SIMBA_BIG_CRITIC`` (which is wider AND deeper than the actor
# per the SimBa paper's finding that critic capacity matters more than actor capacity).
SIMBA_MLP_BIG_ACTOR = RslRlResidualMLPEncoderModelCfg(
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

# Recurrent SimBa actor: byte-for-byte the same SimBa head as ``SIMBA_ACTOR`` with an LSTM inserted
# between the encoder latent and the residual head, via the ``memory`` cfg. This is the clean "memory
# vs no memory" partner of ``SIMBA_ACTOR`` -- the only difference is the populated ``memory`` field, so
# any performance delta is attributable to memory rather than to backbone quality. Switch to GRU with
# ``agent.actor.memory.rnn_type=gru`` as an ablation.
SIMBA_CNN_ACTOR = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=256,
    num_blocks=2,
    expand=4,
    activation="swish",
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    encoder_cfg=CNN_ENCODER_CFG,
)

SIMBA_CNN_BIG_ACTOR = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=512,
    num_blocks=2,
    expand=4,
    activation="swish",
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    encoder_cfg=CNN_ENCODER_CFG,
)

SIMBA_ACTOR = SIMBA_MLP_ACTOR
SIMBA_BIG_ACTOR = SIMBA_MLP_BIG_ACTOR

SIMBA_RNN_ACTOR = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=256,
    num_blocks=2,
    expand=4,
    activation="swish",
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    head_layer_norm=True,
    memory=RslRlResidualMLPEncoderModelCfg.MemoryCfg(rnn_type="lstm", hidden_dim=256, num_layers=1, forget_bias=1.0),
    stochastic=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    encoder_cfg=MLP_ENCODER_CFG,
)

CNN_ACTOR_CFG = RslRlCNNModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
    cnn_cfg=CNN_MODEL_ENCODER_CFG,
    activation="elu",
)

# ---------------------------------------------------------------------------
# Critic model cfgs.
# ---------------------------------------------------------------------------

FLAT_CRITIC = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128, 64],
    activation="elu",
    obs_normalization=True,
    stochastic=False,
)

LSTM_CRITIC = RslRlRNNModelCfg(
    hidden_dims=[512, 256, 128, 64],
    activation="elu",
    obs_normalization=True,
    stochastic=False,
    rnn_num_layers=1,
    rnn_hidden_dim=128,
    rnn_type="lstm",
)

MLP_ENCODER_CRITIC = RslRlMLPEncoderModelCfg(
    hidden_dims=[256, 256, 128],
    activation="elu",
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=False,
    encoder_cfg=MLP_ENCODER_CFG,
)

CNN_CRITIC_CFG = RslRlCNNModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    cnn_cfg=CNN_MODEL_ENCODER_CFG,
    activation="elu",
)

# SimBa critic defaults: match the actor's width/depth/activation for a clean apples-to-apples
# baseline. The paper's Table 10 suggests a wider critic (512) and ReLU; explore that with
# ``agent.critic.hidden_dim=512 agent.critic.activation=relu`` as an ablation once the base
# config is verified to learn.
SIMBA_MLP_CRITIC = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=256,
    num_blocks=2,
    expand=4,
    activation=preset(relu="relu", swish="swish", default="swish"),
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=False,
    encoder_cfg=MLP_ENCODER_CFG,
)

# Scaled-up SimBa critic: 4 residual blocks of 1024 width (4x expansion -> 4096 inner width).
# Per the SimBa paper's scaling analysis (Figure 4), critic capacity helps more than actor
# capacity; this preset realizes that by scaling the critic more aggressively than the actor.
SIMBA_MLP_BIG_CRITIC = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=1024,
    num_blocks=4,
    expand=4,
    activation=preset(relu="relu", swish="swish", default="swish"),
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=False,
    encoder_cfg=MLP_ENCODER_CFG,
)

# Recurrent SimBa critic: matches ``SIMBA_CRITIC`` with an LSTM before the residual head (via the
# ``memory`` cfg), mirroring ``SIMBA_RNN_ACTOR``. A recurrent critic gives the value function the same
# temporal context as the actor, which is generally required for the recurrent actor to learn a
# consistent advantage.
SIMBA_CNN_CRITIC = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=256,
    num_blocks=2,
    expand=4,
    activation=preset(relu="relu", swish="swish", default="swish"),
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=False,
    encoder_cfg=CNN_ENCODER_CFG,
)

SIMBA_CNN_BIG_CRITIC = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=1024,
    num_blocks=4,
    expand=4,
    activation=preset(relu="relu", swish="swish", default="swish"),
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    stochastic=False,
    encoder_cfg=CNN_ENCODER_CFG,
)

SIMBA_CRITIC = SIMBA_MLP_CRITIC
SIMBA_BIG_CRITIC = SIMBA_MLP_BIG_CRITIC

SIMBA_RNN_CRITIC = RslRlResidualMLPEncoderModelCfg(
    hidden_dim=256,
    num_blocks=2,
    expand=4,
    activation=preset(relu="relu", swish="swish", default="swish"),
    norm=True,
    obs_normalization=True,
    encoder_normalization=True,
    head_layer_norm=True,
    memory=RslRlResidualMLPEncoderModelCfg.MemoryCfg(rnn_type="lstm", hidden_dim=256, num_layers=1, forget_bias=1.0),
    stochastic=False,
    encoder_cfg=MLP_ENCODER_CFG,
)
