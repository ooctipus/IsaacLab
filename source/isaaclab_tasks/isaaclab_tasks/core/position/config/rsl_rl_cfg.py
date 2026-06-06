# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp

from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

from isaaclab_tasks.core.multi_task.rl.rsl_rl import (
    RslRlCommanderActorModelCfg,
    RslRlMLPEncoderModelCfg,
    RslRlResidualMLPEncoderModelCfg,
    RslRlTaskEasingActorModelCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

MLP_ENCODER_CFG: dict[str, RslRlMLPEncoderModelCfg.EncoderCfg] = {
    "height_scan": RslRlMLPEncoderModelCfg.EncoderCfg(
        output_dim=64,
        hidden_dims=[128, 64],
        activation="elu",
    ),
}


# CNN encoder for the 2D ``(1, H, W)`` height scan from ``vision_obs``. Sized for the default scanner
# grid (``size=(2.5, 1.5)``, ``resolution=0.075`` -> ``(1, 21, 34)``): two stride-2 convs downsample
# ~5x to a ``(32, 4, 7)`` map (896-d latent), keeping enough spatial structure for the conv to be worth
# more than the flattened MLP encoder. The 21x34 scan is already low-res, so a larger early stride
# (e.g. ``s=4``) would crush it to ~2x3 and throw away the locality that motivates a CNN. ``k3/s2`` also
# stays valid on smaller per-robot grids (e.g. Spot-with-arm's ``(1, 11, 24)`` -> ``(32, 2, 5)``).
CNN_ENCODER_CFG: dict[str, RslRlCNNModelCfg.CNNCfg] = {
    "height_scan": RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32],
        kernel_size=[3, 3],
        stride=[2, 2],
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
    # SimBa actor with a per-group MLP encoder for the (flattened 1D) height scan.
    simba_mlp = RslRlResidualMLPEncoderModelCfg(
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
    # SimBa actor with a CNN encoder for the 2D ``(1, H, W)`` height scan (``vision_obs``). Identical
    # residual head to ``simba_mlp``; only the height-scan encoder differs (CNN vs flattened MLP).
    simba_cnn = RslRlResidualMLPEncoderModelCfg(
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
    # Wider SimBa actors for capacity-scaling experiments. Same architecture, doubled width. The
    # critic should also switch to the matching ``*_big`` (wider AND deeper than the actor per the
    # SimBa paper's finding that critic capacity matters more than actor capacity).
    simba_mlp_big = RslRlResidualMLPEncoderModelCfg(
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
    simba_cnn_big = RslRlResidualMLPEncoderModelCfg(
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
    # Back-compat aliases: ``simba`` / ``simba_big`` are the MLP-encoder variants.
    simba = simba_mlp
    simba_big = simba_mlp_big
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
    simba_mlp = RslRlResidualMLPEncoderModelCfg(
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
    simba_cnn = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg=CNN_ENCODER_CFG,
    )
    # Scaled-up SimBa critics: 4 residual blocks of 1024 width (4x expansion -> 4096 inner width).
    # Per the SimBa paper's scaling analysis (Figure 4), critic capacity helps more than actor
    # capacity; these presets scale the critic more aggressively than the actor.
    simba_mlp_big = RslRlResidualMLPEncoderModelCfg(
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
    simba_cnn_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=1024,
        num_blocks=4,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg=CNN_ENCODER_CFG,
    )
    # Back-compat aliases: ``simba`` / ``simba_big`` are the MLP-encoder variants.
    simba = simba_mlp
    simba_big = simba_mlp_big
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
    # SimBa variants reuse the encoder variant's obs layout (``policy``, ``task``, ``height_scan``);
    # only the encoder type and the env-side ``height_scan`` shape differ (1D for mlp, 2D for cnn).
    simba: dict[str, list[str]] = encoder
    simba_big: dict[str, list[str]] = encoder
    simba_mlp: dict[str, list[str]] = encoder
    simba_mlp_big: dict[str, list[str]] = encoder
    simba_cnn: dict[str, list[str]] = encoder
    simba_cnn_big: dict[str, list[str]] = encoder
    default: dict[str, list[str]] = encoder


@configclass
class ValueShiftAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm cfg + bind expressions for the value-shift curriculum.

    Bind expressions are popped off this cfg by
    :meth:`ValueShiftPPO.construct_algorithm` (so they never reach
    ``PPO.__init__``) and ``eval``-ed against a namespace of
    ``{env, alg, setattr}``. They wire :class:`ValueShiftPPO`'s three buffers
    to the matching :class:`ValueShiftSamplingStrategy` attribute on the
    active terrain-level curriculum term -- so this cfg only makes sense when
    the sampler preset on that curriculum includes value-shift scoring.
    """

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:ValueShiftPPO"
    # ``env`` here is the ``RslRlVecEnvWrapper`` from the runner -- managers
    # live on the underlying ``ManagerBasedRLEnv`` accessed via ``.unwrapped``.
    bind_observation_exp: str = (
        "setattr(alg, '_obs_cache',"
        " env.unwrapped.curriculum_manager.get_term('terrain_levels')"
        "._sampler._impl.value_shift_strategy.observation_cache)"
    )
    bind_current_value_exp: str = (
        "setattr(alg, '_cur_buf',"
        " env.unwrapped.curriculum_manager.get_term('terrain_levels')"
        "._sampler._impl.value_shift_strategy.cur_val)"
    )
    bind_value_diff_exp: str = (
        "setattr(alg, '_diff_buf',"
        " env.unwrapped.curriculum_manager.get_term('terrain_levels')"
        "._sampler._impl.value_shift_strategy.diff_val)"
    )


POSITION_PPO_ALGORITHM_KWARGS = {
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.005,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-4,
    "schedule": "adaptive",
    "gamma": 0.999,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
    "optimizer": preset(default="adam", weight_decay="adamw"),
    "weight_decay": preset(default=0.0, weight_decay=0.01),
    # Actor and critic each build their own per-group encoders (MLP or CNN). Set True to share the
    # actor's encoders with the critic via rsl-rl's ``share_cnn_encoders`` hook (couples their training).
    "share_cnn_encoders": False,
}


POSITION_PPO_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(**POSITION_PPO_ALGORITHM_KWARGS)

POSITION_VALUE_SHIFT_ALGORITHM_CFG = ValueShiftAlgorithmCfg(**POSITION_PPO_ALGORITHM_KWARGS)


@configclass
class PositionAlgorithmPresetCfg(PresetCfg):
    """Algorithm preset selectable via ``agent.algorithm=<name>``.

    The value-shift variants require a sampler preset that includes
    :class:`ValueShiftSamplingStrategy`; the bind expressions on
    :class:`ValueShiftAlgorithmCfg` reach into that strategy's buffers.
    """

    default: RslRlPpoAlgorithmCfg = POSITION_PPO_ALGORITHM_CFG
    value_shift: ValueShiftAlgorithmCfg = POSITION_VALUE_SHIFT_ALGORITHM_CFG
    beta_value_shift: ValueShiftAlgorithmCfg = value_shift


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
    algorithm: PositionAlgorithmPresetCfg = PositionAlgorithmPresetCfg()  # type: ignore
