# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified RSL-RL forward-backward reference configurations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOffPolicyRunnerCfg

from isaaclab_tasks.utils import PresetCfg

from ...rsl_rl import motion_expert_buffer_g1, motion_expert_buffer_smpl_cmu
from ..presets import G1_CMU_PROFILE_CFG, G1_LAFAN_PROFILE_CFG, SMPL_CMU_PROFILE_CFG
from ..profiles import MotionProfileCfg

_ENVIRONMENT_REWARD = "environment"
_DISCRIMINATOR_REWARD = "discriminator"


@dataclass(frozen=True, slots=True)
class _ReferenceProfile:
    """Source-measured differences consumed by one shared config builder."""

    motion: MotionProfileCfg
    expert_provider: Callable[..., object]
    seed: int
    reference_num_envs: int
    num_steps_per_env: int
    num_updates_per_iteration: int
    random_action_steps: int
    training_transitions: int
    replay_capacity_transitions: int
    terminal_capacity_per_env: int
    replay_sampling: str
    save_interval: int
    actor_hidden_dim: int
    actor_hidden_layers: int
    actor_embedding_layers: int
    forward_hidden_dim: int
    forward_hidden_layers: int
    forward_embedding_layers: int
    residual: bool
    actor_std: float
    learning_rate: float
    implied_value_coefficient: float
    helper_actor_coefficients: tuple[float, ...]
    auxiliary_reward_coefficients: tuple[float, ...]
    context_buffer_capacity: int
    rollout_context_refresh_steps: int
    rollout_expert_fraction: float
    expert_window_lengths: tuple[int, ...]
    random_action_range: tuple[float, float]
    history_sources: tuple[tuple[str | None, int, int], ...] = ()

    def __post_init__(self) -> None:
        """Reject reference counts that cannot map exactly to vector rows."""
        collection = self.reference_num_envs * self.num_steps_per_env
        if self.training_transitions % collection:
            raise ValueError("Reference training transitions must contain complete collection blocks.")
        if self.replay_capacity_transitions % self.reference_num_envs:
            raise ValueError("Reference replay transitions must contain complete vector rows.")
        if len(self.helper_actor_coefficients) != 1 + bool(self.motion.routes.auxiliary_evidence):
            raise ValueError("One helper actor coefficient is required per configured value head.")
        if len(self.auxiliary_reward_coefficients) != len(self.motion.routes.auxiliary_evidence):
            raise ValueError("Auxiliary reward coefficients must follow the profile evidence order.")
        if bool(self.history_sources) != (self.motion.routes.history is not None):
            raise ValueError("Replay history reconstruction must match the motion observation profile.")

    @property
    def max_iterations(self) -> int:
        """Number of source-comparable collection blocks."""
        return self.training_transitions // (self.reference_num_envs * self.num_steps_per_env)


def _reward_channel(
    name: str,
    *,
    source: str,
    timing: str,
    context_dependent: bool,
    sign: int,
) -> dict[str, object]:
    """Return one ordered reward-channel declaration."""
    return {
        "name": name,
        "provider_name": name,
        "source": source,
        "timing": timing,
        "context_dependent": context_dependent,
        "sign": sign,
    }


def _observation_routes(profile: MotionProfileCfg) -> dict[str, list[str]]:
    """Derive learner routes from the environment's named information sets."""
    routes = profile.routes
    groups = {
        "actor": list(routes.actor_fields),
        "forward": list(routes.forward_fields),
        "backward": list(routes.expert_fields),
        "discriminator": list(routes.expert_fields),
        "critic_discriminator": list(routes.forward_fields),
    }
    if routes.auxiliary_evidence:
        groups["critic_auxiliary"] = list(routes.forward_fields)
    return groups


def _history_layout(reference: _ReferenceProfile) -> dict[str, object] | None:
    """Return compact reconstruction of the profile's emitted G1 history."""
    history = reference.motion.routes.history
    if history is None:
        return None
    return {
        "history_field": "history_actor",
        "history_length": history.length,
        # The environment emits controller-normalized last_action, whereas replay actions
        # remain in the actor's normalized [-1, 1] coordinates. Store last_action once and
        # reconstruct history from that field rather than silently changing coordinates.
        "last_action_field": None,
        "include_seed_observations": history.include_reset_seed,
        "sources": [
            {"observation_name": name, "start": start, "stop": stop} for name, start, stop in reference.history_sources
        ],
    }


def _model(reference: _ReferenceProfile) -> dict[str, object]:
    """Build one composite model from profile data."""
    actor = {
        "hidden_dim": reference.actor_hidden_dim,
        "hidden_layers": reference.actor_hidden_layers,
        "embedding_layers": reference.actor_embedding_layers,
        "residual": reference.residual,
    }
    forward = {
        "hidden_dim": reference.forward_hidden_dim,
        "hidden_layers": reference.forward_hidden_layers,
        "embedding_layers": reference.forward_embedding_layers,
        "residual": reference.residual,
    }
    value_heads: list[dict[str, object]] = [
        {
            "spec": {
                "name": _DISCRIMINATOR_REWARD,
                "kind": "critic",
                "route": "critic_discriminator",
                "reward_channels": [_DISCRIMINATOR_REWARD],
                "ensemble_size": 2,
                "has_target": True,
            },
            "network": dict(forward),
        }
    ]
    auxiliary = reference.motion.routes.auxiliary_evidence
    if auxiliary:
        value_heads.append(
            {
                "spec": {
                    "name": "auxiliary",
                    "kind": "critic",
                    "route": "critic_auxiliary",
                    "reward_channels": list(auxiliary),
                    "reward_composition": "scalar",
                    "ensemble_size": 2,
                    "has_target": True,
                },
                "network": dict(forward),
            }
        )
    return {
        "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
        "context_dim": 256,
        "actor_cfg": actor,
        "forward_cfg": forward,
        "backward_hidden_dims": [256],
        "backward_normalization": True,
        "discriminator_hidden_dims": [1024, 1024, 1024],
        "distribution_cfg": {
            "class_name": "ClippedGaussianDistribution",
            "init_std": reference.actor_std,
            "action_range": (-1.0, 1.0),
            "noise_clip": 0.3,
        },
        "initialization_type": "orthogonal",
        "normalization_type": "exponential",
        "normalization_eps": 1.0e-5,
        "normalization_momentum": 0.01,
        "context_normalization": True,
        "value_heads": value_heads,
    }


def _replay(reference: _ReferenceProfile) -> dict[str, object]:
    """Build one exact-terminal GPU replay declaration."""
    auxiliary = reference.motion.routes.auxiliary_evidence
    reward_channels = [
        _reward_channel(
            _ENVIRONMENT_REWARD,
            source="environment",
            timing="transition",
            context_dependent=False,
            sign=1,
        ),
        _reward_channel(
            _DISCRIMINATOR_REWARD,
            source="recomputed",
            # Both released critics use the pre-action state as their immediate
            # density-ratio reward. Same-Step only changes reached-state storage.
            timing="state",
            context_dependent=True,
            sign=1,
        ),
    ]
    reward_channels.extend(
        _reward_channel(
            name,
            source="stored_evidence",
            timing="transition",
            context_dependent=False,
            sign=-1,
        )
        for name in auxiliary
    )
    replay: dict[str, object] = {
        "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
        "capacity_transitions": reference.replay_capacity_transitions,
        "terminal_capacity_per_env": reference.terminal_capacity_per_env,
        "sampling": reference.replay_sampling,
        "autoreset_mode": "same_step",
        "environment_reward_name": _ENVIRONMENT_REWARD,
        "auxiliary_evidence_names": list(auxiliary),
        "reward_channels": reward_channels,
        "seed": reference.seed,
    }
    history_layout = _history_layout(reference)
    if history_layout is not None:
        replay["history_layout"] = history_layout
    return replay


def _algorithm(reference: _ReferenceProfile) -> dict[str, object]:
    """Build the shared learner with source-measured profile values."""
    value_cfg: dict[str, dict[str, object]] = {
        _DISCRIMINATOR_REWARD: {
            "learning_rate": reference.learning_rate,
            "pessimism": 0.5,
            "actor_coefficient": reference.helper_actor_coefficients[0],
            "reward_coefficients": (1.0,),
            "target_tau": 0.005,
        }
    }
    if reference.motion.routes.auxiliary_evidence:
        value_cfg["auxiliary"] = {
            "learning_rate": reference.learning_rate,
            "pessimism": 0.5,
            "actor_coefficient": reference.helper_actor_coefficients[1],
            "reward_coefficients": reference.auxiliary_reward_coefficients,
            "normalize_rewards": True,
            "reward_normalization_decay": 0.99,
            "reward_normalization_epsilon": 1.0e-8,
            "target_tau": 0.005,
        }
    return {
        "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
        "batch_size": 1024,
        "expert_sequence_length": 8,
        "gamma": 0.98,
        "learning_rate": reference.learning_rate,
        "backward_learning_rate": 1.0e-5,
        "discriminator_learning_rate": 1.0e-5,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "discriminator_weight_decay": 0.0,
        "fb_pessimism": 0.0,
        "actor_pessimism": 0.5,
        "orthogonality_coefficient": 100.0,
        "implied_value_coefficient": reference.implied_value_coefficient,
        "implied_reward_ridge": 0.0,
        "discriminator_gradient_penalty_coefficient": 10.0,
        "context_goal_fraction": 0.2,
        "context_expert_fraction": 0.6,
        "relabel_fraction": 0.8,
        "context_buffer_capacity": reference.context_buffer_capacity,
        "fb_target_tau": 0.01,
        "scale_actor_helpers": True,
        "max_grad_norm": None,
        "random_action_range": reference.random_action_range,
        "seed": reference.seed,
        "rollout_context_refresh_steps": reference.rollout_context_refresh_steps,
        "rollout_expert_fraction": reference.rollout_expert_fraction,
        "rollout_expert_steps": 250,
        "rollout_expert_context_steps": 8,
        "value_cfg": value_cfg,
    }


def _sections(reference: _ReferenceProfile) -> dict[str, object]:
    """Return every nested RSL-RL section from one profile record."""
    return {
        "obs_groups": _observation_routes(reference.motion),
        "model": _model(reference),
        "replay": _replay(reference),
        "expert": {
            "provider": reference.expert_provider,
            "command_name": "motion",
            "window_lengths": reference.expert_window_lengths,
            "seed": reference.seed,
        },
        "algorithm": _algorithm(reference),
    }


_SMPL_CMU_REFERENCE = _ReferenceProfile(
    motion=SMPL_CMU_PROFILE_CFG,
    expert_provider=motion_expert_buffer_smpl_cmu,
    seed=0,
    reference_num_envs=50,
    num_steps_per_env=10,
    num_updates_per_iteration=50,
    random_action_steps=50_000,
    training_transitions=5_000_000,
    replay_capacity_transitions=2_000_000,
    terminal_capacity_per_env=384,
    replay_sampling="transition_uniform",
    save_interval=1_000,
    actor_hidden_dim=1024,
    actor_hidden_layers=2,
    actor_embedding_layers=2,
    forward_hidden_dim=1024,
    forward_hidden_layers=2,
    forward_embedding_layers=2,
    residual=False,
    actor_std=0.2,
    learning_rate=1.0e-4,
    implied_value_coefficient=0.1,
    helper_actor_coefficients=(0.01,),
    auxiliary_reward_coefficients=(),
    context_buffer_capacity=10_000,
    rollout_context_refresh_steps=150,
    rollout_expert_fraction=0.0,
    expert_window_lengths=(8,),
    random_action_range=(-1.0, 1.0),
)

_G1_LAFAN_REFERENCE = _ReferenceProfile(
    motion=G1_LAFAN_PROFILE_CFG,
    expert_provider=motion_expert_buffer_g1,
    seed=4728,
    reference_num_envs=1024,
    num_steps_per_env=1,
    num_updates_per_iteration=16,
    random_action_steps=10_240,
    training_transitions=9_600_000,
    replay_capacity_transitions=5_120_000,
    terminal_capacity_per_env=17,
    replay_sampling="episode_uniform",
    save_interval=9_375,
    actor_hidden_dim=1024,
    actor_hidden_layers=6,
    actor_embedding_layers=2,
    forward_hidden_dim=1024,
    forward_hidden_layers=6,
    forward_embedding_layers=6,
    residual=True,
    actor_std=0.05,
    learning_rate=3.0e-4,
    implied_value_coefficient=0.0,
    helper_actor_coefficients=(0.05, 0.02),
    auxiliary_reward_coefficients=(0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0),
    context_buffer_capacity=8192,
    rollout_context_refresh_steps=100,
    rollout_expert_fraction=0.5,
    expert_window_lengths=(8, 257),
    random_action_range=(-5.0, 5.0),
    history_sources=(
        ("last_action", 0, 29),
        ("state", 61, 64),
        ("state", 0, 29),
        ("state", 29, 58),
        ("state", 58, 61),
    ),
)

_G1_CMU_EXPERIMENT = replace(_G1_LAFAN_REFERENCE, motion=G1_CMU_PROFILE_CFG)


_SMPL_CMU_SECTIONS = _sections(_SMPL_CMU_REFERENCE)
_G1_LAFAN_SECTIONS = _sections(_G1_LAFAN_REFERENCE)
_G1_CMU_SECTIONS = _sections(_G1_CMU_EXPERIMENT)


@configclass
class SmplCmuForwardBackwardRunnerCfg(RslRlOffPolicyRunnerCfg):
    """Phase 2-comparable SMPL-CMU reference learner for 50 environments."""

    seed = _SMPL_CMU_REFERENCE.seed
    num_steps_per_env = _SMPL_CMU_REFERENCE.num_steps_per_env
    num_envs = _SMPL_CMU_REFERENCE.reference_num_envs
    num_updates_per_iteration = _SMPL_CMU_REFERENCE.num_updates_per_iteration
    random_action_steps = _SMPL_CMU_REFERENCE.random_action_steps
    init_at_random_ep_len = False
    max_iterations = _SMPL_CMU_REFERENCE.max_iterations
    save_interval = _SMPL_CMU_REFERENCE.save_interval
    experiment_name = "motion_smpl_cmu_forward_backward"
    obs_groups = _SMPL_CMU_SECTIONS["obs_groups"]
    algorithm = _SMPL_CMU_SECTIONS["algorithm"]
    model: dict[str, object] = _SMPL_CMU_SECTIONS["model"]  # type: ignore[assignment]
    replay: dict[str, object] = _SMPL_CMU_SECTIONS["replay"]  # type: ignore[assignment]
    expert: dict[str, object] = _SMPL_CMU_SECTIONS["expert"]  # type: ignore[assignment]
    torch_compile_mode = None


@configclass
class G1LafanForwardBackwardRunnerCfg(RslRlOffPolicyRunnerCfg):
    """Selected 6x1024 G1-LAFAN Phase 2 reference learner for 1024 environments."""

    seed = _G1_LAFAN_REFERENCE.seed
    num_steps_per_env = _G1_LAFAN_REFERENCE.num_steps_per_env
    num_envs = _G1_LAFAN_REFERENCE.reference_num_envs
    num_updates_per_iteration = _G1_LAFAN_REFERENCE.num_updates_per_iteration
    random_action_steps = _G1_LAFAN_REFERENCE.random_action_steps
    init_at_random_ep_len = False
    max_iterations = _G1_LAFAN_REFERENCE.max_iterations
    save_interval = _G1_LAFAN_REFERENCE.save_interval
    experiment_name = "motion_g1_lafan_forward_backward"
    obs_groups = _G1_LAFAN_SECTIONS["obs_groups"]
    algorithm = _G1_LAFAN_SECTIONS["algorithm"]
    model: dict[str, object] = _G1_LAFAN_SECTIONS["model"]  # type: ignore[assignment]
    replay: dict[str, object] = _G1_LAFAN_SECTIONS["replay"]  # type: ignore[assignment]
    expert: dict[str, object] = _G1_LAFAN_SECTIONS["expert"]  # type: ignore[assignment]
    torch_compile_mode = None
    lifecycle_extension = {
        "class_name": "isaaclab_tasks.core.multi_task.motion.tracking:MotionTrackingCurriculum",
        "transition_interval": 9_600_000,
        "evaluator": "isaaclab_tasks.core.multi_task.motion.tracking:g1_motion_tracking_evaluator",
        "evaluation_seed": 0,
    }


@configclass
class G1CmuForwardBackwardRunnerCfg(RslRlOffPolicyRunnerCfg):
    """G1-CMU composition experiment initialized from the G1-LAFAN learner."""

    seed = _G1_CMU_EXPERIMENT.seed
    num_steps_per_env = _G1_CMU_EXPERIMENT.num_steps_per_env
    num_envs = _G1_CMU_EXPERIMENT.reference_num_envs
    num_updates_per_iteration = _G1_CMU_EXPERIMENT.num_updates_per_iteration
    random_action_steps = _G1_CMU_EXPERIMENT.random_action_steps
    init_at_random_ep_len = False
    max_iterations = _G1_CMU_EXPERIMENT.max_iterations
    save_interval = _G1_CMU_EXPERIMENT.save_interval
    experiment_name = "motion_g1_cmu_composition_forward_backward"
    obs_groups = _G1_CMU_SECTIONS["obs_groups"]
    algorithm = _G1_CMU_SECTIONS["algorithm"]
    model: dict[str, object] = _G1_CMU_SECTIONS["model"]  # type: ignore[assignment]
    replay: dict[str, object] = _G1_CMU_SECTIONS["replay"]  # type: ignore[assignment]
    expert: dict[str, object] = _G1_CMU_SECTIONS["expert"]  # type: ignore[assignment]
    torch_compile_mode = None
    lifecycle_extension = {
        "class_name": "isaaclab_tasks.core.multi_task.motion.tracking:MotionTrackingCurriculum",
        "transition_interval": 9_600_000,
        "evaluator": "isaaclab_tasks.core.multi_task.motion.tracking:g1_motion_tracking_evaluator",
        "evaluation_seed": 0,
    }


@configclass
class MotionForwardBackwardRunnerPresetsCfg(PresetCfg):
    """Select the learner paired with each complete motion-environment preset."""

    smpl_cmu = SmplCmuForwardBackwardRunnerCfg()
    g1_lafan = G1LafanForwardBackwardRunnerCfg()
    default = smpl_cmu
    g1_cmu = G1CmuForwardBackwardRunnerCfg()


__all__ = [
    "G1LafanForwardBackwardRunnerCfg",
    "G1CmuForwardBackwardRunnerCfg",
    "MotionForwardBackwardRunnerPresetsCfg",
    "SmplCmuForwardBackwardRunnerCfg",
]
