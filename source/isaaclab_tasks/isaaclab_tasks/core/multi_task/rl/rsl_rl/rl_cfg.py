# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils.configclass import configclass

import isaaclab_rl.rsl_rl
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


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
class RslRlForwardBackwardRunnerCfg(isaaclab_rl.rsl_rl.RslRlOffPolicyRunnerCfg):
    """Typed connector from IsaacLab configuration to the RSL-RL FB runner."""

    @configclass
    class ObservationRoutesCfg:
        """Fixed observation routes consumed by the FB learner."""

        actor: list[str] = MISSING
        forward: list[str] = MISSING
        backward: list[str] = MISSING
        discriminator: list[str] = MISSING
        critic_discriminator: list[str] = MISSING

    @configclass
    class NetworkCfg:
        """Architecture of one FB residual or plain MLP."""

        hidden_dim: int = MISSING
        hidden_layers: int = MISSING
        embedding_layers: int = MISSING
        residual: bool = MISSING

    @configclass
    class ScheduleCfg:
        """One coherent collection, update, and checkpoint schedule."""

        num_envs: int = MISSING
        num_steps_per_env: int = MISSING
        num_updates_per_iteration: int = MISSING
        random_action_steps: int = MISSING
        max_iterations: int = MISSING
        save_interval: int = MISSING

    @configclass
    class ModelTopologyCfg:
        """Actor and forward-map architectures selected as one model policy."""

        actor: RslRlForwardBackwardRunnerCfg.NetworkCfg = MISSING
        forward: RslRlForwardBackwardRunnerCfg.NetworkCfg = MISSING

    @configclass
    class ReplayPolicyCfg:
        """Capacity and sampling semantics selected as one replay policy."""

        capacity_transitions: int = MISSING
        terminal_capacity_per_env: int = MISSING
        sampling: Literal["transition_uniform", "episode_uniform"] = MISSING

    @configclass
    class ExpertClockCfg:
        """Relation between source time and sampled expert rows."""

        sampling_mode: Literal["source_rows", "uniform_before_source_end"] = MISSING
        sampling_step_seconds: float | None = MISSING

    @configclass
    class OptimizationCfg:
        """Coordinated representation and value-helper optimization policy."""

        learning_rate: float = MISSING
        implied_value_coefficient: float = MISSING

    @configclass
    class ContextPolicyCfg:
        """Expert windows and online context relabeling selected as one policy."""

        expert_window_lengths: tuple[int, ...] = MISSING
        buffer_capacity: int = MISSING
        refresh_steps: int = MISSING
        rollout_expert_fraction: float = MISSING

    @configclass
    class ExplorationCfg:
        """Actor distribution and random warm-up action range."""

        distribution: RslRlForwardBackwardRunnerCfg.DistributionCfg = MISSING
        random_action_range: tuple[float, float] = MISSING

    @configclass
    class DistributionCfg:
        """Clipped Gaussian actor distribution."""

        class_name: str = "ClippedGaussianDistribution"
        init_std: float = MISSING
        action_range: tuple[float, float] = (-1.0, 1.0)
        noise_clip: float = 0.3

    @configclass
    class ValueSpecCfg:
        """Identity and reward routing for one value helper."""

        name: str = MISSING
        kind: Literal["critic"] = "critic"
        route: str = MISSING
        reward_channels: list[str] = MISSING
        ensemble_size: int = 2
        has_target: bool = True

    @configclass
    class ScalarValueSpecCfg(ValueSpecCfg):
        """Value-helper specification with scalar reward composition."""

        reward_composition: Literal["scalar"] = "scalar"

    @configclass
    class ValueHeadCfg:
        """One value-helper specification and optional architecture override."""

        spec: RslRlForwardBackwardRunnerCfg.ValueSpecCfg = MISSING
        network: RslRlForwardBackwardRunnerCfg.NetworkCfg | None = None

    @configclass
    class ModelCfg:
        """Unified FB model configuration."""

        @configclass
        class NormalizationGroupCfg:
            """One ordered set of semantic fields sharing normalization statistics."""

            name: str = MISSING
            fields: tuple[str, ...] = MISSING

        class_name: str = "rsl_rl.models.forward_backward_model:ForwardBackwardModel"
        context_dim: int = 256
        topology: RslRlForwardBackwardRunnerCfg.ModelTopologyCfg = MISSING
        backward_hidden_dims: list[int] = [256]
        backward_normalization: bool = True
        discriminator_hidden_dims: list[int] = [1024, 1024, 1024]
        initialization_type: str = "orthogonal"
        normalization_type: str = "exponential"
        normalization_eps: float = 1.0e-5
        normalization_momentum: float = 0.01
        normalization_groups: tuple[NormalizationGroupCfg, ...] = ()
        context_normalization: bool = True

    @configclass
    class RewardChannelCfg:
        """One named reward channel and its temporal semantics."""

        name: str = MISSING
        provider_name: str = MISSING
        source: Literal["environment", "recomputed", "stored_evidence"] = MISSING
        timing: Literal["state", "next_state", "transition"] = MISSING
        context_dependent: bool = MISSING
        sign: Literal[-1, 1] = MISSING

    @configclass
    class HistorySourceCfg:
        """One complete named observation packed into replay history."""

        observation_name: str = MISSING

    @configclass
    class HistoryLayoutCfg:
        """Field-major reached-transition history layout."""

        history_field: str = MISSING
        history_length: int = MISSING
        include_seed_observations: bool = False
        sources: list[RslRlForwardBackwardRunnerCfg.HistorySourceCfg] = MISSING

    @configclass
    class ReplayCfg:
        """Replay storage and reward-channel contract."""

        class_name: str = "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay"
        policy: RslRlForwardBackwardRunnerCfg.ReplayPolicyCfg = MISSING
        autoreset_mode: Literal["disabled", "same_step", "next_step"] = MISSING
        seed: int | None = None
        history_layout: RslRlForwardBackwardRunnerCfg.HistoryLayoutCfg | None = None

    @configclass
    class ExpertCfg:
        """Base expert-corpus provider configuration."""

        provider: str = MISSING
        seed: int | None = None

    @configclass
    class SequenceExpertCfg(ExpertCfg):
        """Expert provider backed by a bound sequence source."""

        source_bind: str = MISSING
        clock: RslRlForwardBackwardRunnerCfg.ExpertClockCfg = MISSING
        target_projection: str = MISSING
        target_projection_binds: tuple[str, ...] = MISSING

    @configclass
    class ValueObjectiveCfg:
        """Optimization settings for one value helper."""

        pessimism: float = MISSING
        reward_coefficients: tuple[float, ...] = MISSING
        actor_coefficient: float = MISSING
        target_tau: float = MISSING

    @configclass
    class NormalizedValueObjectiveCfg(ValueObjectiveCfg):
        """Value objective with online reward normalization."""

        normalize_rewards: bool = True
        reward_normalization_decay: float = MISSING
        reward_normalization_epsilon: float = MISSING

    @configclass
    class ValueTermCfg:
        """One ordered reward term consumed by a value helper."""

        name: str = MISSING
        coefficient: float = MISSING
        source: Literal["environment", "recomputed", "stored_evidence"] = MISSING
        timing: Literal["state", "next_state", "transition"] = MISSING
        context_dependent: bool = MISSING
        sign: Literal[-1, 1] = MISSING

    @configclass
    class ValueHelperCfg:
        """One value helper and its complete ordered reward algebra."""

        name: str = MISSING
        route: str = MISSING
        terms: tuple[RslRlForwardBackwardRunnerCfg.ValueTermCfg, ...] = MISSING
        reward_composition: Literal["scalar"] | None = None
        pessimism: float = MISSING
        actor_coefficient: float = MISSING
        normalize_rewards: bool = False
        reward_normalization_decay: float | None = None
        reward_normalization_epsilon: float | None = None
        target_tau: float = MISSING

    @configclass
    class AlgorithmCfg:
        """Unified FB optimization configuration."""

        class_name: str = "rsl_rl.algorithms.forward_backward:ForwardBackward"
        batch_size: int = MISSING
        expert_sequence_length: int = MISSING
        gamma: float = MISSING
        optimization: RslRlForwardBackwardRunnerCfg.OptimizationCfg = MISSING
        backward_learning_rate: float = MISSING
        discriminator_learning_rate: float = MISSING
        optimizer: str = MISSING
        weight_decay: float = MISSING
        discriminator_weight_decay: float = MISSING
        fb_pessimism: float = MISSING
        actor_pessimism: float = MISSING
        orthogonality_coefficient: float = MISSING
        implied_reward_ridge: float = MISSING
        discriminator_gradient_penalty_coefficient: float = MISSING
        context_goal_fraction: float = MISSING
        context_expert_fraction: float = MISSING
        relabel_fraction: float = MISSING
        fb_target_tau: float = MISSING
        scale_actor_helpers: bool = MISSING
        max_grad_norm: float | None = None
        seed: int | None = None
        rollout_expert_steps: int = MISSING
        rollout_expert_context_steps: int = MISSING

    @configclass
    class LifecycleCfg:
        """Base transition-count lifecycle extension."""

        class_name: str = MISSING
        transition_interval: int = MISSING

    @configclass
    class TrackingLifecycleCfg(LifecycleCfg):
        """Generic sequence-tracking lifecycle configured through bindings."""

        @configclass
        class ProjectionCfg:
            """One expert-target to reached-observation metric projection."""

            metric_name: str = MISSING
            target_name: str = MISSING
            observation_name: str = MISSING
            projection: str | None = None
            assignment_metric: str = "uniform_assignment"

        command_bind: str = MISSING
        sequence_ids_bind: str = MISSING
        sequence_start_rows_bind: str = MISSING
        sampling_priorities_bind: str = MISSING
        evaluation_scope_bind: str = MISSING
        projections: tuple[ProjectionCfg, ...] = MISSING
        context_window_length: int = 1
        include_reset_frame: bool = True
        allow_horizon_truncation: bool = True
        shuffle_assignments: bool = True
        priority_metric_name: str = MISSING
        priority_metric_minimum: float = MISSING
        priority_metric_maximum: float = MISSING
        priority_exponent_scale: float = MISSING
        priority_exponent_base: float = MISSING
        reset_source_name: str = MISSING
        evaluation_seed: int = 0

    obs_groups: ObservationRoutesCfg = MISSING  # type: ignore[assignment]
    """Typed FB observation routes."""

    model: ModelCfg = MISSING
    """Unified FB model."""

    replay: ReplayCfg = MISSING
    """Replay storage contract."""

    expert: ExpertCfg = MISSING
    """Expert-corpus provider."""

    algorithm: AlgorithmCfg = MISSING  # type: ignore[assignment]
    """Unified FB optimization."""

    lifecycle_extension: LifecycleCfg | None = None  # type: ignore[assignment]
    """Optional completed-transition lifecycle extension."""

    schedule: ScheduleCfg = MISSING
    """Structured collection, update, and checkpoint schedule."""

    context_policy: ContextPolicyCfg = MISSING
    """Structured expert-window and online-context policy."""

    exploration: ExplorationCfg = MISSING
    """Structured actor and random warm-up exploration policy."""

    value_helpers: list[ValueHelperCfg] = MISSING
    """Ordered value helpers that derive model, replay, and objective runtime views."""

    def to_dict(self) -> dict[str, object]:
        """Serialize the typed connector into the ordinary RSL-RL runtime contract."""
        values = super().to_dict()

        schedule = values.pop("schedule")
        schedule_names = (
            "num_envs",
            "num_steps_per_env",
            "num_updates_per_iteration",
            "random_action_steps",
            "max_iterations",
            "save_interval",
        )
        for name in schedule_names:
            values[name] = schedule.pop(name)
        if schedule:
            raise ValueError(f"Unknown forward-backward schedule fields: {tuple(schedule)}.")
        if self.num_envs is not None:
            values["num_envs"] = self.num_envs
        if not isinstance(self.max_iterations, type(MISSING)):
            values["max_iterations"] = self.max_iterations

        exploration = values.pop("exploration")
        context_policy = values.pop("context_policy")
        value_helpers = values.pop("value_helpers")

        model = values["model"]
        topology = model.pop("topology")
        model["actor_cfg"] = topology.pop("actor")
        model["forward_cfg"] = topology.pop("forward")
        model["distribution_cfg"] = exploration.pop("distribution")
        if topology:
            raise ValueError(f"Unknown forward-backward topology fields: {tuple(topology)}.")

        replay = values["replay"]
        replay.update(replay.pop("policy"))

        expert = values["expert"]
        expert.update(expert.pop("clock"))
        expert["window_lengths"] = context_policy.pop("expert_window_lengths")

        algorithm = values["algorithm"]
        optimization = algorithm.pop("optimization")
        algorithm["learning_rate"] = optimization.pop("learning_rate")
        algorithm["implied_value_coefficient"] = optimization.pop("implied_value_coefficient")
        algorithm["context_buffer_capacity"] = context_policy.pop("buffer_capacity")
        algorithm["rollout_context_refresh_steps"] = context_policy.pop("refresh_steps")
        algorithm["rollout_expert_fraction"] = context_policy.pop("rollout_expert_fraction")
        algorithm["random_action_range"] = exploration.pop("random_action_range")
        if optimization or context_policy or exploration:
            raise ValueError("Unknown forward-backward optimization, context, or exploration fields.")

        if not value_helpers:
            raise ValueError("At least one value helper must be configured.")
        model["value_heads"] = []
        replay["reward_channels"] = []
        algorithm["value_cfg"] = {}
        helper_names: set[str] = set()
        term_names: set[str] = set()
        environment_reward_name: str | None = None
        stored_evidence_names: list[str] = []
        for helper in value_helpers:
            helper_name = helper.pop("name")
            terms = helper.pop("terms")
            names = [term["name"] for term in terms]
            coefficients = tuple(term.pop("coefficient") for term in terms)
            if (
                not helper_name
                or helper_name in helper_names
                or not names
                or any(not name for name in names)
                or term_names.intersection(names)
                or len(names) != len(set(names))
            ):
                raise ValueError("Value helper and reward-term names must be non-empty and globally unique.")
            helper_names.add(helper_name)
            term_names.update(names)

            reward_composition = helper.pop("reward_composition")
            if reward_composition not in (None, "scalar"):
                raise ValueError(f"Unsupported reward composition: {reward_composition!r}.")
            spec_type = (
                RslRlForwardBackwardRunnerCfg.ValueSpecCfg
                if reward_composition is None
                else RslRlForwardBackwardRunnerCfg.ScalarValueSpecCfg
            )
            model["value_heads"].append(
                RslRlForwardBackwardRunnerCfg.ValueHeadCfg(
                    spec=spec_type(
                        name=helper_name,
                        route=helper.pop("route"),
                        reward_channels=names,
                    ),
                ).to_dict()
            )
            for term in terms:
                name = term.pop("name")
                source = term.pop("source")
                timing = term.pop("timing")
                if source == "environment":
                    if timing != "transition":
                        raise ValueError("Environment rewards must describe a completed transition.")
                    if environment_reward_name is not None:
                        raise ValueError("At most one environment reward term may be configured.")
                    environment_reward_name = name
                elif source == "stored_evidence":
                    if timing != "transition":
                        raise ValueError("Stored evidence must describe a completed transition.")
                    stored_evidence_names.append(name)
                replay["reward_channels"].append(
                    RslRlForwardBackwardRunnerCfg.RewardChannelCfg(
                        name=name,
                        provider_name=name,
                        source=source,
                        timing=timing,
                        context_dependent=term.pop("context_dependent"),
                        sign=term.pop("sign"),
                    ).to_dict()
                )
                if term:
                    raise ValueError(f"Unknown value term fields: {tuple(term)}.")

            normalize_rewards = helper.pop("normalize_rewards")
            normalization_decay = helper.pop("reward_normalization_decay")
            normalization_epsilon = helper.pop("reward_normalization_epsilon")
            objective_type = (
                RslRlForwardBackwardRunnerCfg.NormalizedValueObjectiveCfg
                if normalize_rewards
                else RslRlForwardBackwardRunnerCfg.ValueObjectiveCfg
            )
            objective_args = dict(
                pessimism=helper.pop("pessimism"),
                actor_coefficient=helper.pop("actor_coefficient"),
                reward_coefficients=coefficients,
                target_tau=helper.pop("target_tau"),
            )
            if normalize_rewards:
                if normalization_decay is None or normalization_epsilon is None:
                    raise ValueError("Normalized value helpers require decay and epsilon.")
                objective_args.update(
                    normalize_rewards=True,
                    reward_normalization_decay=normalization_decay,
                    reward_normalization_epsilon=normalization_epsilon,
                )
            elif normalization_decay is not None or normalization_epsilon is not None:
                raise ValueError("Unnormalized value helpers cannot declare normalization parameters.")
            objective = objective_type(**objective_args).to_dict()
            objective["learning_rate"] = algorithm["learning_rate"]
            algorithm["value_cfg"][helper_name] = objective
            if helper:
                raise ValueError(f"Unknown value helper fields: {tuple(helper)}.")
        replay["environment_reward_name"] = environment_reward_name
        replay["auxiliary_evidence_names"] = stored_evidence_names
        replay["auxiliary_evidence_observation_group"] = "transition" if stored_evidence_names else None

        root_seed = values["seed"]
        if type(root_seed) is not int:
            raise TypeError("Forward-backward runner seed must be an integer.")
        for name, section in (("replay", replay), ("expert", expert), ("algorithm", algorithm)):
            configured_seed = section.get("seed")
            if configured_seed not in (None, root_seed):
                raise ValueError(f"{name} seed must be omitted or equal the runner seed.")
            section["seed"] = root_seed
        return values

    def resolve_num_envs(self, cli_num_envs: int | None, env_num_envs: int) -> int:
        """Resolve CLI, explicit runner, then structured-schedule environment count."""
        del env_num_envs
        if cli_num_envs is not None:
            return cli_num_envs
        return self.schedule.num_envs if self.num_envs is None else self.num_envs

    def resolve_max_iterations(self, cli_max_iterations: int | None) -> int:
        """Resolve CLI, explicit runner, then structured-schedule iteration count."""
        if cli_max_iterations is not None:
            return cli_max_iterations
        return self.schedule.max_iterations if isinstance(self.max_iterations, type(MISSING)) else self.max_iterations


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
class RslRlSuccessEstimatorAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with an explicitly bound completed-transition success estimator."""

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:SuccessEstimatorPPO"
    """Success-estimator PPO implementation."""

    success_outcome_bind: str = MISSING
    """Expression resolving to the environment-owned success outcome tensor."""

    success_train_mask_bind: str | None = None
    """Optional expression resolving to the success-estimator training mask."""

    success_estimator_learning_rate: float = 1.0e-4
    """Success-estimator optimizer learning rate."""

    success_loss_coef: float = 1.0
    """Success-estimator loss coefficient."""

    success_returns_method: Literal["bootstrap", "hindsight_mc"] = "hindsight_mc"
    """Return target construction used by the success estimator."""


@configclass
class RslRlSuccessEstimatorRunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy runner configuration for :class:`RslRlSuccessEstimatorAlgorithmCfg`."""

    algorithm: RslRlSuccessEstimatorAlgorithmCfg = MISSING
    """Success-estimator PPO configuration."""

    success_estimator: RslRlMLPModelCfg = MISSING
    """Network predicting the completed-episode success probability."""

    success_estimator_bind: str | None = None
    """Optional expression binding predictions to an external consumer."""

    state_buffer_bind: str | None = None
    """Optional expression binding a curriculum state buffer for diagnostics."""


@configclass
class RslRlCrlRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for the CRL-specific replay and prefill lifecycle."""

    class_type: type[Any] | str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.runners:CrlRunner"
    """The CRL runner class."""

    class_name: str = "CrlRunner"
    """The runner class name."""

    actor: RslRlResidualMLPCfg = MISSING
    """The actor trunk configuration."""

    critic: RslRlResidualMLPCfg = MISSING
    """The critic trunk configuration."""

    algorithm: RslRlCrlAlgorithmCfg = MISSING
    """The algorithm configuration."""


@configclass
class RslRlOffPolicyRunnerCfg(RslRlCrlRunnerCfg):
    """Deprecated task-local name for :class:`RslRlCrlRunnerCfg`.

    .. deprecated:: 8.0.2
        Use :class:`RslRlCrlRunnerCfg`. Generic off-policy algorithms should
        use :class:`isaaclab_rl.rsl_rl.RslRlOffPolicyRunnerCfg` instead.
    """

    class_type: type[Any] | str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.runners:OffPolicyRunner"
    """Deprecated runner wrapper retained for one release."""

    class_name: str = "OffPolicyRunner"
    """Deprecated runner class name retained for one release."""

    def __post_init__(self) -> None:
        """Warn that the task-local off-policy name now belongs to CRL."""
        warnings.warn(
            "RslRlOffPolicyRunnerCfg is deprecated; use RslRlCrlRunnerCfg.",
            DeprecationWarning,
            stacklevel=2,
        )


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
