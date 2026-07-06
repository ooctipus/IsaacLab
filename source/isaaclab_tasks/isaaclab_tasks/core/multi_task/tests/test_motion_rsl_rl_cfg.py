# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct runtime-contract tests for motion forward-backward configuration."""

from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.rl.rsl_rl import (
    RslRlForwardBackwardAlgorithmCfg,
    RslRlForwardBackwardExpertCfg,
    RslRlForwardBackwardModelCfg,
    RslRlForwardBackwardReplayCfg,
    RslRlForwardBackwardRunnerCfg,
    RslRlForwardBackwardValueHelperCfg,
)
from isaaclab_tasks.utils import resolve_presets

_META_TOKENS = {
    "smpl",
    "cmu",
    "helpers_discriminator",
    "tracking_off",
    "model_plain_2x1024",
    "replay_transition_uniform_2m",
    "schedule_50x10_5m",
    "optimization_lr1e4_implied0p1_actor0p01",
    "context_online_10k",
    "exploration_std0p2_range1",
    "seed_0",
    "expert_clock_source_rows",
}
_BFM_TOKENS = {
    "g1",
    "lafan",
    "helpers_discriminator_auxiliary",
    "tracking_reset_frame",
    "tracking_interval_9p6m",
    "model_residual_6x1024",
    "replay_episode_uniform_5120k",
    "schedule_1024x1_211p2m",
    "optimization_lr3e4_implied0_actor0p05",
    "context_expert_half_8192",
    "exploration_std0p05_range5",
    "seed_4728",
    "expert_clock_50hz",
}


def _runner(tokens: set[str]) -> MotionForwardBackwardRunnerCfg:
    cfg = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=tokens)
    assert isinstance(cfg, MotionForwardBackwardRunnerCfg)
    return cfg


def test_fb_config_is_the_runtime_contract() -> None:
    """Generic serialization should need no connector-owned schema compiler."""
    cfg = _runner(_BFM_TOKENS)
    values = cfg.to_dict()

    assert isinstance(cfg, RslRlForwardBackwardRunnerCfg)
    assert isinstance(cfg.model, RslRlForwardBackwardModelCfg)
    assert isinstance(cfg.replay, RslRlForwardBackwardReplayCfg)
    assert isinstance(cfg.expert, RslRlForwardBackwardExpertCfg)
    assert isinstance(cfg.algorithm, RslRlForwardBackwardAlgorithmCfg)
    assert all(isinstance(helper, RslRlForwardBackwardValueHelperCfg) for helper in cfg.value_helpers)
    assert {"schedule", "context_policy", "lifecycle_extension"}.isdisjoint(values)
    assert {"capacity_transitions", "terminal_capacity_per_env", "sampling"}.isdisjoint(values["replay"])
    assert set(values["replay"]["policy"]) == {
        "capacity_transitions",
        "terminal_capacity_per_env",
        "sampling",
    }
    assert {"sampling_mode", "sampling_step_seconds"}.isdisjoint(values["expert"])
    assert set(values["expert"]["clock"]) == {"sampling_mode", "sampling_step_seconds"}
    assert set(values["algorithm"]["optimization"]) == {
        "learning_rate",
        "backward_learning_rate",
        "discriminator_learning_rate",
        "optimizer",
        "weight_decay",
        "discriminator_weight_decay",
        "max_grad_norm",
    }
    assert set(values["algorithm"]["context"]) == {
        "goal_fraction",
        "expert_fraction",
        "relabel_fraction",
        "buffer_capacity",
        "refresh_steps",
        "rollout_expert_fraction",
        "rollout_expert_steps",
        "rollout_expert_context_steps",
    }
    assert set(values["algorithm"]["exploration"]) == {
        "random_action_transitions",
        "random_action_range",
    }
    assert {
        "learning_rate",
        "context_buffer_capacity",
        "rollout_context_refresh_steps",
        "random_action_transitions",
    }.isdisjoint(values["algorithm"])
    assert {"value_heads"}.isdisjoint(values["model"])
    assert {
        "reward_channels",
        "environment_reward_name",
        "auxiliary_evidence_names",
        "auxiliary_evidence_observation_group",
    }.isdisjoint(values["replay"])
    assert {"value_cfg"}.isdisjoint(values["algorithm"])


def test_metamotivo_profile_keeps_its_faithful_direct_values() -> None:
    cfg = _runner(_META_TOKENS)

    assert (cfg.seed, cfg.num_envs, cfg.num_steps_per_env, cfg.num_updates_per_iteration) == (0, 50, 10, 50)
    assert cfg.max_iterations * cfg.num_envs * cfg.num_steps_per_env == 5_000_000
    assert cfg.algorithm.exploration.random_action_transitions == 50_000
    assert cfg.algorithm.optimization.learning_rate == 1.0e-4
    assert cfg.algorithm.implied_value_coefficient == 0.1
    assert cfg.algorithm.context.buffer_capacity == 10_000
    assert cfg.algorithm.context.rollout_expert_fraction == 0.0
    assert cfg.replay.policy.capacity_transitions == 2_000_000
    assert cfg.replay.policy.sampling == "transition_uniform"
    assert cfg.replay.history_layout is None
    assert cfg.expert.clock.sampling_mode == "source_rows"
    assert cfg.expert.clock.sampling_step_seconds is None
    assert cfg.expert.window_lengths == (8,)
    assert cfg.model.actor_cfg.hidden_layers == cfg.model.forward_cfg.hidden_layers == 2
    assert cfg.model.actor_cfg.residual is cfg.model.forward_cfg.residual is False
    assert cfg.tracking_curriculum is None
    assert cfg.class_type == "rsl_rl.runners:ForwardBackwardRunner"
    assert tuple(helper.name for helper in cfg.value_helpers) == ("discriminator",)
    assert cfg.value_helpers[0].route == "critic_value"
    assert "critic_discriminator" not in cfg.obs_groups
    assert cfg.value_helpers[0].learning_rate == 1.0e-4
    assert cfg.value_helpers[0].actor_coefficient == 0.01


def test_bfm_profile_keeps_its_faithful_direct_values() -> None:
    cfg = _runner(_BFM_TOKENS)

    assert (cfg.seed, cfg.num_envs, cfg.num_steps_per_env, cfg.num_updates_per_iteration) == (4728, 1024, 1, 16)
    assert cfg.max_iterations * cfg.num_envs * cfg.num_steps_per_env == 211_200_000
    assert cfg.algorithm.exploration.random_action_transitions == 10_240
    assert cfg.algorithm.optimization.learning_rate == 3.0e-4
    assert cfg.algorithm.implied_value_coefficient == 0.0
    assert cfg.algorithm.context.buffer_capacity == 8192
    assert cfg.algorithm.context.rollout_expert_fraction == 0.5
    assert cfg.replay.policy.capacity_transitions == 5_120_000
    assert cfg.replay.policy.terminal_capacity_per_env == 17
    assert cfg.replay.policy.sampling == "episode_uniform"
    assert cfg.replay.history_layout is not None
    assert tuple(source.observation_name for source in cfg.replay.history_layout.sources) == (
        "last_action",
        "base_angular_velocity",
        "joint_position",
        "joint_velocity",
        "projected_gravity",
    )
    assert cfg.expert.clock.sampling_mode == "uniform_before_source_end"
    assert cfg.expert.clock.sampling_step_seconds == 0.02
    assert cfg.expert.window_lengths == (8, 257)
    assert cfg.class_type == "rsl_rl.runners:ForwardBackwardRunner"
    assert cfg.model.actor_cfg.hidden_layers == cfg.model.forward_cfg.hidden_layers == 6
    assert cfg.model.actor_cfg.embedding_layers == 2
    assert cfg.model.forward_cfg.embedding_layers == 6
    assert cfg.model.normalization_groups[0].fields == (
        "joint_position",
        "joint_velocity",
        "projected_gravity",
        "base_angular_velocity",
    )
    assert tuple(helper.name for helper in cfg.value_helpers) == ("discriminator", "auxiliary")
    assert {helper.route for helper in cfg.value_helpers} == {"critic_value"}
    assert "critic_discriminator" not in cfg.obs_groups
    assert tuple(helper.learning_rate for helper in cfg.value_helpers) == (3.0e-4, 3.0e-4)
    assert cfg.value_helpers[0].actor_coefficient == 0.05
    assert cfg.tracking_curriculum is not None
    assert cfg.tracking_curriculum.class_name.endswith(":ForwardBackwardTrackingCurriculum")
    assert cfg.tracking_curriculum.sequence_ids_bind.endswith(".table.clip_ids")
    assert cfg.tracking_curriculum.sequence_start_rows_bind.endswith(".table.clip_start_rows")
    assert cfg.tracking_curriculum.evaluation_scope_bind.endswith(".payload.sampler.reset_sampling_scope")
    assert cfg.tracking_curriculum.interval_transitions == 9_600_000


def test_reward_algebra_has_one_declared_order() -> None:
    cfg = _runner(_BFM_TOKENS)
    discriminator, auxiliary = cfg.value_helpers

    assert tuple(term.name for term in discriminator.terms) == ("discriminator",)
    assert discriminator.terms[0].source == "recomputed"
    assert discriminator.reward_composition == "vector"
    assert auxiliary.reward_composition == "scalar"
    assert tuple((term.name, term.coefficient) for term in auxiliary.terms) == (
        ("penalty_torques", 0.0),
        ("penalty_action_rate", 0.1),
        ("limits_dof_pos", 10.0),
        ("limits_torque", 0.0),
        ("penalty_undesired_contact", 1.0),
        ("penalty_feet_ori", 0.4),
        ("penalty_ankle_roll", 4.0),
        ("penalty_slippage", 2.0),
    )


def test_helper_presence_and_optimization_profiles_remain_independent() -> None:
    meta_with_auxiliary = _runner((_META_TOKENS - {"helpers_discriminator"}) | {"helpers_discriminator_auxiliary"})
    bfm_without_auxiliary = _runner((_BFM_TOKENS - {"helpers_discriminator_auxiliary"}) | {"helpers_discriminator"})

    assert tuple(helper.name for helper in meta_with_auxiliary.value_helpers) == ("discriminator", "auxiliary")
    assert tuple(helper.learning_rate for helper in meta_with_auxiliary.value_helpers) == (1.0e-4, 1.0e-4)
    assert tuple(helper.name for helper in bfm_without_auxiliary.value_helpers) == ("discriminator",)
    assert bfm_without_auxiliary.value_helpers[0].learning_rate == 3.0e-4


def test_robot_and_dataset_axes_remain_independent() -> None:
    lafan = _runner(_BFM_TOKENS)
    cmu = _runner((_BFM_TOKENS - {"lafan"}) | {"cmu"})

    assert lafan.to_dict() == cmu.to_dict()
    assert "g1_lafan" not in repr(lafan.to_dict())
    assert "g1_cmu" not in repr(cmu.to_dict())
