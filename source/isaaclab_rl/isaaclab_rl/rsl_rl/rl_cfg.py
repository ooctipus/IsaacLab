# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils.configclass import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Model configurations #
#########################


@configclass
class RslRlMLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = "MLPModel"
    """The model class name. Defaults to MLPModel."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP network."""

    activation: str = MISSING
    """The activation function for the MLP network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""

    stochastic: bool = MISSING
    """Whether the model output is stochastic.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and set it to None
    for deterministic output or to a valid configuration class, e.g., `GaussianDistributionCfg` for stochastic output.
    """

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the model.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `init_std` field of the distribution configuration to specify the initial noise standard deviation.
    """

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the model. Defaults to scalar.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `std_type` field of the distribution configuration to specify the type of noise standard deviation.
    """

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Defaults to False.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use
    the `HeteroscedasticGaussianDistributionCfg` if state-dependent standard deviation is desired.
    """


@configclass
class RslRlRNNModelCfg(RslRlMLPModelCfg):
    """Configuration for RNN model."""

    class_name: str = "RNNModel"
    """The model class name. Defaults to RNNModel."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlCNNModelCfg(RslRlMLPModelCfg):
    """Configuration for CNN model."""

    class_name: str = "CNNModel"
    """The model class name. Defaults to CNNModel."""

    @configclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = MISSING
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = 1
        """The stride for the CNN. Defaults to 1."""

        dilation: int | tuple[int] | list[int] = 1
        """The dilation for the CNN. Defaults to 1."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
        """The padding for the CNN. Defaults to none."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN. Defaults to none."""

        activation: str = MISSING
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = False
        """Whether to use max pooling for the CNN. Defaults to False."""

        global_pool: Literal["none", "max", "avg"] = "none"
        """The global pooling for the CNN. Defaults to none."""

        flatten: bool = True
        """Whether to flatten the output of the CNN. Defaults to True."""

        channels_last: bool = False
        """Pin conv weights and forward inputs to ``torch.channels_last`` on CUDA. Defaults to False."""

    cnn_cfg: CNNCfg = MISSING
    """The configuration for the CNN(s)."""


@configclass
class RslRlMLPEncoderModelCfg(RslRlMLPModelCfg):
    """Configuration for an MLP model with per-obs-group MLP encoders.

    Extends :class:`RslRlMLPModelCfg` with a mapping of observation group names to encoder
    MLP configurations. The keys of :attr:`encoder_cfg` determine which observation groups
    are routed through dedicated MLP encoders; all other groups in the active obs set are
    concatenated directly with the encoder outputs and fed to the main MLP head.
    """

    class_name: str = "MLPEncoderModel"
    """The model class name. Defaults to MLPEncoderModel."""

    @configclass
    class EncoderCfg:
        """Per-obs-group encoder MLP configuration.

        Forwarded as keyword arguments to :class:`rsl_rl.modules.MLP`, which determines the
        encoder's hidden stack and output dimension. ``input_dim`` is inferred from the
        observation group at model construction time and should not be set here.
        """

        output_dim: int = MISSING
        """Output dimension of the encoder (the compact feature size fed to the main MLP head)."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the encoder MLP."""

        activation: str = MISSING
        """The activation function for the encoder MLP."""

        last_activation: str | None = None
        """Optional activation applied after the last linear layer of the encoder. Defaults to None (linear output)."""

    encoder_cfg: dict[str, EncoderCfg] = MISSING
    """Mapping from observation group name to a per-group encoder configuration.

    Only observation groups present as keys in this mapping are routed through encoders.
    All other groups in ``obs_groups[obs_set]`` pass through to the main MLP head directly.
    """

    encoder_normalization: bool = False
    """Whether to apply running-statistic :class:`EmpiricalNormalization` to each encoded observation
    group before it enters its per-group encoder MLP. Independent of :attr:`obs_normalization`, which
    controls only the passthrough path. Defaults to False."""

    head_layer_norm: bool = True
    """Whether to apply :class:`torch.nn.LayerNorm` on the concatenated latent
    ``[passthrough || encoder_features]`` before the main MLP head. Per-sample, no running stats, no
    train/eval mismatch. Stabilizes the main MLP head against scale drift in encoder outputs during
    training. Aligned with modern ML practice (transformers, multi-modal fusion, MLP-Mixer, SimBa-style
    stable RL). Defaults to True."""


@configclass
class RslRlResidualMLPEncoderModelCfg(RslRlMLPEncoderModelCfg):
    """Configuration for an MLP encoder model with a SimBa-style residual main head.

    Inherits the encoder machinery from :class:`RslRlMLPEncoderModelCfg` but replaces the plain
    :class:`~rsl_rl.modules.MLP` main head with :class:`~rsl_rl.modules.ResidualMLP`, a faithful
    implementation of equations (5)-(7) in SimBa (Lee et al. 2024): input Linear projection,
    ``num_blocks`` pre-norm residual feedforward blocks with 4x inverted bottlenecks, post-Layer
    normalization, and a linear output head.
    """

    class_name: str = "ResidualMLPEncoderModel"
    """The model class name. Defaults to ResidualMLPEncoderModel."""

    hidden_dim: int = MISSING
    """Width of the residual pathway (``d_h`` in the SimBa paper)."""

    num_blocks: int = 2
    """Number of pre-norm residual blocks. The paper uses 1 for actors and 2 for critics in its PPO
    experiments. Defaults to 2."""

    expand: int = 4
    """Inverted-bottleneck expansion ratio inside each residual block. The paper uses 4. Defaults to 4."""

    norm: bool = True
    """Whether the residual blocks apply LayerNorm internally and whether a post-LayerNorm is applied
    before the output linear. Defaults to True."""

    last_activation: str | None = None
    """Optional activation applied after the final linear layer of the residual main head."""

    head_layer_norm: bool = False
    """Whether to apply an additional :class:`torch.nn.LayerNorm` on the concatenated latent before
    the residual head. Defaults to False because the residual head already contains an internal
    pre-norm inside its first block; enable only if you want a second norm step."""

    # The plain-MLP head fields from the parent are unused in this variant.
    hidden_dims: list[int] = [0]  # type: ignore[assignment]
    """Unused for :class:`RslRlResidualMLPEncoderModelCfg`; retained to satisfy the parent dataclass
    signature. Use :attr:`hidden_dim` and :attr:`num_blocks` instead."""


@configclass
class RslRlResidualMLPCfg:
    """Configuration for a :class:`~rsl_rl.modules.ResidualMLP` backbone.

    Reusable across algorithms — specifies only the network architecture,
    not algorithm-specific details like distributions or normalization.
    """

    hidden_dim: int = 256
    """Width of the residual pathway [neurons]."""

    depth: int = 4
    """Total depth (Dense layers inside residual blocks). ``depth // num_layers_per_block`` blocks."""

    num_layers_per_block: int = 4
    """Dense layers per residual block. 4 = scaling-crl, 2 = SimBa."""

    expand: int = 1
    """Expansion ratio inside residual blocks. 1 = scaling-crl, 4 = SimBa."""

    activation: str = "swish"
    """Activation function name."""

    norm: bool = True
    """Whether to apply LayerNorm inside residual blocks."""

    repr_dim: int | None = None
    """Output representation dimensionality (used by CRL critic). If ``None``,
    the output dim is inferred from the task (e.g., action_dim for actors)."""

    encoder_cfg: dict[str, dict] | None = None
    """Optional per-group MLP encoder configuration for high-dim observations.

    Keys are obs-group names (e.g. ``"height_scan"``); values are kwargs for
    :class:`~rsl_rl.modules.MLP`. Each named group is routed through its own
    encoder MLP before concatenation; other groups pass through. Used by CRL's
    :class:`~rsl_rl.algorithms.crl.SquashedGaussianActor` and
    :class:`~rsl_rl.algorithms.crl.BilinearCritic` so the policy and the
    contrastive critic share a compressed view of high-dim state. Example::

        encoder_cfg={"height_scan": {"hidden_dims": [256, 256], "output_dim": 256}}
    """


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Defaults to PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use. Defaults to adam."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Defaults to False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    returns_method: Literal["gae", "hindsight_mc"] = "gae"
    """Method for computing critic return targets.

    - ``"gae"``: Generalized Advantage Estimation (standard). Blends TD and MC
      returns via the lambda parameter. Good default for dense reward settings.
    - ``"hindsight_mc"``: Hindsight Monte Carlo. Computes exact discounted returns
      for completed episodes; bootstraps with V(s) only at the horizon cutoff
      (end of the rollout buffer). Equivalent to GAE with lambda=1. Best suited
      for sparse/binary reward tasks where bootstrap bias is more harmful than
      MC variance.
    """

    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False.

    When ``True``, actor and critic hold the *same* :class:`torch.nn.ModuleDict` of CNN encoders.
    PPO additionally computes the encoder forward exactly once per minibatch and threads the
    resulting features to both heads, halving the conv compute (the dominant cost of the PPO
    update on CNN-encoded configurations). Mathematically identical to two separate forwards
    through the shared module, since the same module on the same input produces the same
    outputs — no user-facing flag for the compute-sharing path, it's automatic when the
    encoder modules are shared.
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Defaults to None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Defaults to None, in which case symmetry is not used."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    class_type: type[Any] | str = MISSING
    """The runner class."""

    seed: int = 42
    """The seed for the experiment. Defaults to 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Defaults to cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """This parameter is deprecated and will be removed in the future.

    For rsl-rl < 4.0.0, use `actor_obs_normalization` and `critic_obs_normalization` of the policy instead.
    For rsl-rl >= 4.0.0, use `obs_normalization` of the model instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "actor": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the actor will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    obs_storage_dtypes: dict[str, str] | None = None
    """Optional per-key dtype overrides for the rollout-buffer observations storage.

    Keys are observation names (e.g. ``"height_scan"``); values are torch dtype names
    (e.g. ``"float16"``, ``"bfloat16"``). When set, the matching key is allocated in the
    reduced-precision dtype in the rollout buffer and upcast back to the env-side dtype
    when the storage yields mini-batches. This is intended to cut memory of large
    image-like obs (e.g. CNN height-scans) without changing the policy's input precision.
    """

    check_for_nan: bool = True
    """Whether to check for NaN values coming from the environment."""

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Defaults to empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """
    run_id: str | None = None
    """The run ID (e.g. for Weights & Biases). This is the unique identifier for the run as set by the logger."""

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Defaults to tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Defaults to "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Defaults to "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Defaults to False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Defaults to ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Defaults to ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_type: type[Any] | str = "rsl_rl.runners:OnPolicyRunner"
    """The runner class. Defaults to OnPolicyRunner."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    actor: RslRlMLPModelCfg = MISSING
    """The actor configuration."""

    critic: RslRlMLPModelCfg = MISSING
    """The critic configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is is deprecated. Please use `actor` and `critic` model configurations
    instead.
    """


###########################
# CRL (off-policy) config #
###########################


@configclass
class RslRlHerCfg:
    """Configuration for Hindsight Experience Replay.

    ``goal_start_idx``, ``goal_end_idx``, and ``obs_dim`` are resolved at runtime
    from the environment's observation groups via :func:`resolve_her_config`.
    """

    gamma: float = 0.99
    """Geometric discount for future-timestep sampling in HER."""

    target_state: str = "target_state"
    """Observation group containing the target/commanded state."""

    current_state: str = "current_state"
    """Observation group containing the current achieved state (for HER relabeling)."""


@configclass
class RslRlCrlAlgorithmCfg:
    """Configuration for the CRL algorithm.

    Two primary knobs control the training budget:

    - ``replay_ratio``: fraction of the buffer to train on per update (0.0-1.0).
    - ``num_sgd_steps``: number of gradient steps per update.

    ``batch_size`` is auto-derived:
    ``batch_size = (max_replay_size * replay_ratio * num_envs) / num_sgd_steps``.
    It scales automatically with ``num_envs``.
    """

    class_name: str = "CRL"
    """The algorithm class name. Defaults to CRL."""

    actor_lr: float = 3e-4
    """Learning rate for the actor optimizer."""

    critic_lr: float = 3e-4
    """Learning rate for the critic optimizer."""

    alpha_lr: float = 3e-4
    """Learning rate for the entropy coefficient optimizer."""

    max_replay_size: int = 10000
    """Maximum capacity of the replay buffer [timesteps per env]."""

    min_replay_size: int = 1000
    """Minimum buffer fill before training starts [timesteps per env]."""

    replay_ratio: float = 0.1
    """Fraction of the buffer to train on per update (0.0-1.0).
    Total samples = ``max_replay_size * replay_ratio * num_envs``.
    ``batch_size = total_samples / num_sgd_steps`` (auto-derived)."""

    num_sgd_steps: int = 800
    """Number of gradient steps per ``update()`` call.
    Together with ``replay_ratio``, determines ``batch_size``."""

    logsumexp_penalty_coeff: float = 0.1
    """Regularization coefficient for the logsumexp term in the InfoNCE loss."""

    entropy_param: float = 0.5
    """Entropy coefficient multiplied by action_dim to derive ``target_entropy``."""

    her_cfg: RslRlHerCfg | None = RslRlHerCfg()
    """HER configuration. Default ON. Set to ``None`` to disable (for ablation)."""

    use_cuda_graph: bool = True
    """Enable CUDA graph capture for the SGD loop. Set ``False`` to force eager
    execution (useful for debugging)."""

    share_encoders: bool = True
    """Share per-group obs encoders between the actor and the critic's SA branch.

    When ``True`` (default), the same :class:`~rsl_rl.algorithms.crl._SharedStateEncoder`
    instance is wired into both the policy network and the bilinear critic, so
    parameters are tied and the contrastive Q's view of state matches the
    policy's. The shared encoder is registered with the critic optimizer; the
    actor's loss back-propagates through it but only the critic step updates
    its parameters. Set to ``False`` for an ablation that gives actor and
    critic independent encoder copies (no parameter sharing)."""


@configclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for off-policy algorithms (e.g. CRL)."""

    class_type: type[Any] | str = "rsl_rl.runners:OffPolicyRunner"
    """The runner class. Defaults to OffPolicyRunner."""

    class_name: str = "OffPolicyRunner"
    """The runner class name. Defaults to OffPolicyRunner."""

    actor: RslRlResidualMLPCfg = MISSING
    """The actor trunk configuration."""

    critic: RslRlResidualMLPCfg = MISSING
    """The critic trunk configuration."""

    algorithm: RslRlCrlAlgorithmCfg = MISSING
    """The algorithm configuration."""


#############################
# Deprecated configurations #
#############################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead.
    """

    class_name: str = "ActorCritic"
    """The policy class name. Defaults to ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Defaults to scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Defaults to False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlRNNModelCfg` instead.
    """

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Defaults to ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""
