# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlCNNModelCfg, RslRlMLPModelCfg


@configclass
class RslRlMLPEncoderModelCfg(RslRlMLPModelCfg):
    """Configuration for MLP models with per-group encoders."""

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.models.mlp_encoder_model:MLPEncoderModel"
    """The model class name."""

    @configclass
    class EncoderCfg:
        """Per-observation-group encoder MLP configuration."""

        output_dim: int = MISSING
        """Output dimension of the encoder."""

        hidden_dims: list[int] = MISSING
        """Hidden dimensions of the encoder MLP."""

        activation: str = MISSING
        """Activation function for the encoder MLP."""

        last_activation: str | None = None
        """Optional activation after the last encoder layer."""

    encoder_cfg: dict[str, EncoderCfg] = MISSING
    """Mapping from observation group name to encoder MLP configuration."""

    encoder_normalization: bool = False
    """Whether to normalize each encoded observation group before its encoder."""

    head_layer_norm: bool = True
    """Whether to apply LayerNorm after concatenating raw and encoded features."""


@configclass
class RslRlResidualMLPEncoderModelCfg(RslRlMLPEncoderModelCfg):
    """Configuration for encoder models with a residual MLP head and optional recurrent memory."""

    class_name: str = (
        "isaaclab_tasks.core.multi_task.rl.rsl_rl.models.residual_mlp_encoder_model:ResidualMLPEncoderModel"
    )
    """The model class name."""

    hidden_dim: int = MISSING
    """Width of the residual pathway."""

    num_blocks: int = 2
    """Number of residual blocks."""

    expand: int = 4
    """Expansion ratio inside each residual block."""

    norm: bool = True
    """Whether to apply LayerNorm inside the residual head."""

    last_activation: str | None = None
    """Optional activation after the final residual-head layer."""

    head_layer_norm: bool = False
    """Whether to apply LayerNorm before the residual head (and before the RNN, if memory is used)."""

    @configclass
    class MemoryCfg:
        """Recurrent memory inserted between the encoder latent and the residual head."""

        rnn_type: str = "lstm"
        """The type of recurrent cell to use. Either ``"lstm"`` or ``"gru"``."""

        hidden_dim: int = 256
        """Dimension of the recurrent hidden state."""

        num_layers: int = 1
        """Number of stacked recurrent layers."""

        forget_bias: float | None = 1.0
        """Initial LSTM forget-gate bias. ``None`` keeps PyTorch defaults; ignored for GRU."""

    memory: MemoryCfg | None = None
    """Optional recurrent memory. ``None`` (default) yields a feedforward SimBa model; set a
    :class:`MemoryCfg` to insert an RNN, turning this into the recurrent SimBa model. This is the only
    field that distinguishes the memory and no-memory variants."""

    encoder_cfg: dict[str, RslRlMLPEncoderModelCfg.EncoderCfg | RslRlCNNModelCfg.CNNCfg] | None = None  # type: ignore[assignment]
    """Per-group encoders. Each entry is either an MLP encoder (:class:`RslRlMLPEncoderModelCfg.EncoderCfg`,
    which flattens its input) or a CNN encoder (:class:`RslRlCNNModelCfg.CNNCfg`, which consumes a
    ``(C, H, W)`` image such as a 2D height scan). An entry is treated as a CNN when it declares
    ``output_channels``. Must be set unless encoders are shared via ``cnns``."""

    hidden_dims: list[int] = [0]  # type: ignore[assignment]
    """Unused plain-MLP head field retained for inherited config shape."""


@configclass
class RslRlSuccessorFeatureCriticModelCfg(RslRlResidualMLPEncoderModelCfg):
    """SimBa encoder + two (free-norm) successor-representation heads (``psi``, ``phi``).

    The command is folded into the encoded state; ``psi``/``phi`` branch off the shared latent and feed the
    dynamics-anchored successor value ``V = <psi, w>`` (the read-out ``w`` lives in the
    :class:`~rsl_rl.extensions.SuccessorFeatures` extension). Use with the ``successor`` algorithm preset.
    """

    class_name: str = (
        "isaaclab_tasks.core.multi_task.rl.rsl_rl.models.successor_feature_critic_model:SuccessorFeatureCriticModel"
    )
    """The model class name."""

    feature_dim: int = 128
    """Width ``d`` of the ``psi`` / ``phi`` heads (and of ``w``)."""


@configclass
class RslRlSuccessorActorModelCfg(RslRlResidualMLPEncoderModelCfg):
    """SimBa encoder + a z-conditioned Gaussian policy head ``pi(a | [h(s), z])``.

    The goal is absent from the actor's observation; it enters only via the goal embedding ``z = B(goal)``
    (supplied by the :class:`~rsl_rl.extensions.SuccessorFeatures` extension), so the policy must consult ``z``.
    Use with the ``successor`` algorithm + critic presets; ``feature_dim`` must match the critic's.
    """

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.models.successor_actor_model:SuccessorActorModel"
    """The model class name."""

    feature_dim: int = 128
    """Width ``d`` of the goal embedding ``z`` (must equal the critic's ``feature_dim``)."""


@configclass
class RslRlResidualMLPCfg:
    """Configuration for a residual MLP backbone."""

    hidden_dim: int = 256
    """Width of the residual pathway."""

    depth: int = 4
    """Total dense-layer depth."""

    num_layers_per_block: int = 4
    """Dense layers per residual block."""

    expand: int = 1
    """Expansion ratio inside residual blocks."""

    activation: str = "swish"
    """Activation function name."""

    norm: bool = True
    """Whether to apply LayerNorm inside residual blocks."""

    repr_dim: int | None = None
    """Output representation dimension. If ``None``, the task output dimension is used."""

    encoder_cfg: dict[str, dict] | None = None
    """Optional per-group MLP encoder configuration."""


@configclass
class RslRlHerCfg:
    """Configuration for Hindsight Experience Replay."""

    gamma: float = 0.99
    """Geometric discount for future-timestep sampling."""

    target_state: str = "target_state"
    """Observation group containing the target state."""

    current_state: str = "current_state"
    """Observation group containing the achieved state."""


@configclass
class RslRlCrlAlgorithmCfg:
    """Configuration for the CRL algorithm."""

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:CRL"
    """The algorithm class name."""

    actor_lr: float = 3e-4
    """Learning rate for the actor optimizer."""

    critic_lr: float = 3e-4
    """Learning rate for the critic optimizer."""

    alpha_lr: float = 3e-4
    """Learning rate for the entropy coefficient optimizer."""

    max_replay_size: int = 10000
    """Maximum replay-buffer capacity."""

    min_replay_size: int = 1000
    """Minimum buffer fill before training starts."""

    replay_ratio: float = 0.1
    """Fraction of the replay buffer used per update."""

    num_sgd_steps: int = 800
    """Number of gradient steps per update."""

    logsumexp_penalty_coeff: float = 0.1
    """Regularization coefficient for the logsumexp term."""

    entropy_param: float = 0.5
    """Entropy coefficient multiplier."""

    her_cfg: RslRlHerCfg | None = RslRlHerCfg()
    """HER configuration. Set to ``None`` to disable HER."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graph capture for the SGD loop."""

    share_encoders: bool = True
    """Whether actor and critic share per-group observation encoders."""


@configclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for off-policy runners."""

    class_type: type[Any] | str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.runners:OffPolicyRunner"
    """The runner class."""

    class_name: str = "OffPolicyRunner"
    """The runner class name."""

    actor: RslRlResidualMLPCfg = MISSING
    """The actor trunk configuration."""

    critic: RslRlResidualMLPCfg = MISSING
    """The critic trunk configuration."""

    algorithm: RslRlCrlAlgorithmCfg = MISSING
    """The algorithm configuration."""


@configclass
class RslRlCommanderActorModelCfg(RslRlMLPModelCfg):
    """Configuration for the commander actor model.

    This extends :class:`RslRlMLPModelCfg` with parameters specific to
    :class:`CommanderActorModel` from :mod:`model`. The commander network maps policy
    observations (excluding ``cmd_feat``) to a command feature vector that is
    injected into the observation dictionary before the actor MLP runs.
    """

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.models.commander_actor_model:CommanderActorModel"
    """The model class name. Defaults to CommanderActorModel."""

    commander_hidden_dims: list[int] = [256, 256, 256]
    """Hidden dimensions for the commander MLP."""

    commander_activation: str = "elu"
    """Activation function for the commander MLP. Defaults to elu."""

    commander_obs_normalization: bool = False
    """Whether to normalise commander inputs and targets. Defaults to False."""

    kinematic_reward_weight: float = 0.1
    """Scale applied to the kinematic tracking reward [1]. Defaults to 0.1."""

    commander_loss_coef: float = 0.1
    """Coefficient for the commander L2 regularisation loss [1]. Defaults to 0.1."""

    get_command_target_fn: Callable | None = None
    """Callable returning the ground-truth command target tensor.

    The callable should accept one argument (the environment) and return a tensor.
    """

    log_error_fn: Callable | None = None
    """Optional callable for logging command errors.

    The callable should accept ``(env, cmd_proposed, cmd_target)``.
    """


@configclass
class RslRlTaskEasingActorModelCfg(RslRlMLPModelCfg):
    """Configuration for the task-easing actor model.

    This extends :class:`RslRlMLPModelCfg` with parameters specific to
    :class:`TaskEasingActorModel` from :mod:`model`. A chain of learned goal blocks
    progressively refines the task observation before passing it to the actor MLP.
    """

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.models.task_easing_actor_model:TaskEasingActorModel"
    """The model class name. Defaults to TaskEasingActorModel."""

    task_easing_constraint_fn: Literal["relu", "softplus"] = "relu"
    """Activation used as the constraint function. Defaults to relu."""

    task_easing_loss_coef: float = 0.0
    """Coefficient for the monotonic-value loss [1]. Defaults to 0.0."""

    task_easing_margin: float = 0.0
    """Margin in the monotonic-value constraint [1]. Defaults to 0.0."""

    task_easing_network: Literal["mlp", "residual"] = "mlp"
    """Backbone for goal blocks. Defaults to mlp."""

    num_goal_refinements: int = 2
    """Number of refinement steps. Defaults to 2."""

    goal_hidden_dims: list[int] = [256, 256]
    """Hidden dimensions for each goal block."""

    goal_activation: str | None = None
    """Activation for goal blocks. Defaults to the actor activation if not specified."""
