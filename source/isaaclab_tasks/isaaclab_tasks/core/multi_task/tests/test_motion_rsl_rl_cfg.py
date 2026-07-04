# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the independently composed motion forward-backward runner."""

from __future__ import annotations

import ast
import dataclasses
import enum
import hashlib
import json
from pathlib import Path

import pytest

import isaaclab_tasks.core.multi_task.motion.config.agents.rsl_rl_fb_cfg as rsl_rl_fb_cfg
import isaaclab_tasks.core.multi_task.rl.rsl_rl.rl_cfg as connector_cfg
from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.rl.rsl_rl import RslRlForwardBackwardRunnerCfg
from isaaclab_tasks.utils import PresetCfg, resolve_presets

_META_RUNNER_SHA256 = "2e8f1753cf34ea4802b16e76d324ea08d566e57becf8217a5316303838f74565"
_BFM_RUNNER_SHA256 = "b14b4658f31cfaf4264decbfa0161c5858cc84d2e198507d1ee913ae33947295"

_META_RUNNER_TOKENS = frozenset(
    (
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
    )
)
_BFM_RUNNER_TOKENS = frozenset(
    (
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
    )
)
_G1_CMU_RUNNER_TOKENS = (_BFM_RUNNER_TOKENS - {"lafan"}) | {"cmu"}


def _runner(tokens: frozenset[str]) -> MotionForwardBackwardRunnerCfg:
    """Resolve one explicit set of independent runner-policy axes."""
    cfg = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=tokens)
    assert isinstance(cfg, MotionForwardBackwardRunnerCfg)
    return cfg


def _channel(values: dict[str, object], name: str) -> dict[str, object]:
    channels = values["replay"]["reward_channels"]  # type: ignore[index]
    return next(channel for channel in channels if channel["name"] == name)  # type: ignore[index,union-attr]


def _json_default(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    raise TypeError(type(value))


def _runner_sha256(cfg: MotionForwardBackwardRunnerCfg) -> str:
    """Hash the complete resolved runner dictionary."""
    values = cfg.to_dict()
    encoded = json.dumps(
        values,
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_values(cfg: MotionForwardBackwardRunnerCfg) -> dict[str, object]:
    """Return the strict runtime dictionary produced by the typed connector."""
    return cfg.to_dict()


def test_runner_root_has_independent_literal_axes_and_no_product_classes() -> None:
    """The module must expose one literal root and reject the removed product language."""
    source = Path(rsl_rl_fb_cfg.__file__).read_text()
    assert "reset_source_name=preset(" not in source
    assert 'reset_source_name="motion"' not in source
    tree = ast.parse(source)
    top_level_classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    forbidden = {
        "MetaMotivoForwardBackwardRunnerCfg",
        "BfmZeroForwardBackwardRunnerCfg",
        "MotionForwardBackwardRunnerPresetsCfg",
        "_ReferenceProfile",
        "_reward_channel",
        "_observation_routes",
        "_history_layout",
        "_model",
        "_replay",
        "_algorithm",
        "_sections",
    }

    definitions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assignments = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert set(top_level_classes) == {"MotionForwardBackwardRunnerCfg"}
    assert forbidden.isdisjoint(definitions | assignments)
    assert not any(isinstance(node, ast.FunctionDef) for node in tree.body)
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("rsl_rl_expert") for node in tree.body
    )
    assert not any(isinstance(node, ast.Dict) for node in ast.walk(tree))

    root = top_level_classes["MotionForwardBackwardRunnerCfg"]
    nested = {node.name: node for node in root.body if isinstance(node, ast.ClassDef)}
    assert {
        "ObservationRoutesCfg",
        "ModelTopologyCfg",
        "ScheduleCfg",
        "OptimizationCfg",
        "ExpertClockCfg",
        "ExplorationCfg",
        "ValueHelpersCfg",
        "ReplayCfg",
        "ContextPolicyCfg",
        "LifecycleCfg",
    } <= set(nested)
    assert {"ExpertCfg", "TrackingCfg", "ValueHeadsCfg", "ValueObjectivesCfg"}.isdisjoint(nested)
    assert issubclass(MotionForwardBackwardRunnerCfg, RslRlForwardBackwardRunnerCfg)
    assert issubclass(MotionForwardBackwardRunnerCfg.ObservationRoutesCfg, PresetCfg)
    assert issubclass(MotionForwardBackwardRunnerCfg.ModelTopologyCfg, PresetCfg)
    assert issubclass(MotionForwardBackwardRunnerCfg.ReplayCfg, RslRlForwardBackwardRunnerCfg.ReplayCfg)
    assert not issubclass(MotionForwardBackwardRunnerCfg.ReplayCfg, PresetCfg)
    assert issubclass(MotionForwardBackwardRunnerCfg.ValueHelpersCfg, PresetCfg)

    preset_classes = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "PresetCfg" for base in node.bases)
    )
    for preset_class in preset_classes:
        members = {
            target.id
            for node in preset_class.body
            for target in (
                [node.target]
                if isinstance(node, ast.AnnAssign)
                else node.targets
                if isinstance(node, ast.Assign)
                else []
            )
            if isinstance(target, ast.Name)
        }
        assert "default" in members
        assert len(members) > 1

    assert not any(isinstance(node, ast.ClassDef) and node.name == "ForwardNetworkCfg" for node in ast.walk(tree))

    for call in (node for node in ast.walk(root) if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Name) and call.func.id == "preset":
            assert all(not isinstance(keyword.value, ast.Dict | ast.List) for keyword in call.keywords)


def test_runner_presets_have_independent_semantic_owners() -> None:
    """Preset names describe one concern and never encode a paper or product."""
    source = Path(rsl_rl_fb_cfg.__file__).read_text()
    tree = ast.parse(source)
    fields = {
        target.id
        for node in ast.walk(tree)
        for target in (
            [node.target]
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            else node.targets
            if isinstance(node, ast.Assign)
            else []
        )
        if isinstance(target, ast.Name)
    }
    fields.update(
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "preset"
        for keyword in node.keywords
        if keyword.arg is not None
    )
    required = {
        "helpers_discriminator",
        "helpers_discriminator_auxiliary",
        "tracking_off",
        "tracking_source_edge",
        "tracking_reset_frame",
        "tracking_interval_9p6m",
        "model_plain_2x1024",
        "model_residual_6x1024",
        "replay_transition_uniform_2m",
        "replay_episode_uniform_5120k",
        "schedule_50x10_5m",
        "schedule_1024x1_211p2m",
        "optimization_lr1e4_implied0p1_actor0p01",
        "optimization_lr3e4_implied0_actor0p05",
        "context_online_10k",
        "context_expert_half_8192",
        "exploration_std0p2_range1",
        "exploration_std0p05_range5",
        "seed_0",
        "seed_4728",
        "expert_clock_source_rows",
        "expert_clock_50hz",
    }

    assert required <= fields
    declared_names = fields | {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    lowered_names = {name.lower() for name in declared_names}
    assert {"fb_metamotivo", "fb_bfm_zero"}.isdisjoint(lowered_names)
    assert all(
        token not in name
        for name in lowered_names
        for token in ("metamotivo", "bfm", "g1_tracking", "g1_lafan", "g1_cmu", "smpl_cmu")
    )

    root = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionForwardBackwardRunnerCfg"
    )
    nested = {node.name: node for node in root.body if isinstance(node, ast.ClassDef)}
    for owner in ("ValueHelpersCfg", "LifecycleCfg"):
        members = {
            target.id
            for node in nested[owner].body
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if isinstance(target, ast.Name)
        }
        assert {"smpl", "g1"}.isdisjoint(members)


def test_fb_connector_types_are_generic_config_only_owners() -> None:
    """Fixed RSL-RL schemas live in the typed connector without importing motion."""
    source = Path(connector_cfg.__file__).read_text()
    tree = ast.parse(source)
    imports = [node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None]
    top_level_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    connector = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RslRlForwardBackwardRunnerCfg"
    )
    nested = {node.name for node in connector.body if isinstance(node, ast.ClassDef)}

    assert not any("multi_task.motion" in module for module in imports)
    assert not any(
        name.startswith("RslRlForwardBackward") and name != "RslRlForwardBackwardRunnerCfg"
        for name in top_level_classes
    )
    assert {
        "ObservationRoutesCfg",
        "ScheduleCfg",
        "ModelTopologyCfg",
        "ReplayPolicyCfg",
        "ExpertClockCfg",
        "OptimizationCfg",
        "ContextPolicyCfg",
        "ExplorationCfg",
        "ValueTermCfg",
        "ValueHelperCfg",
        "ModelCfg",
        "ReplayCfg",
        "ExpertCfg",
        "SequenceExpertCfg",
        "AlgorithmCfg",
        "LifecycleCfg",
        "TrackingLifecycleCfg",
    } <= nested
    assert {
        "AuxiliaryObservationRoutesCfg",
        "HistoryReplayCfg",
        "AuxiliaryHelperCfg",
        "ValueObjectivesCfg",
        "AuxiliaryValueObjectivesCfg",
    }.isdisjoint(nested)
    assert "value_helpers" in RslRlForwardBackwardRunnerCfg.__dataclass_fields__
    assert "value_heads" not in RslRlForwardBackwardRunnerCfg.ModelCfg.__dataclass_fields__
    assert {
        "environment_reward_name",
        "auxiliary_evidence_names",
        "auxiliary_evidence_observation_group",
        "reward_channels",
    }.isdisjoint(RslRlForwardBackwardRunnerCfg.ReplayCfg.__dataclass_fields__)
    assert "value_cfg" not in RslRlForwardBackwardRunnerCfg.AlgorithmCfg.__dataclass_fields__
    assert issubclass(MotionForwardBackwardRunnerCfg, RslRlForwardBackwardRunnerCfg)

    meta = _runner(_META_RUNNER_TOKENS)
    bfm = _runner(_BFM_RUNNER_TOKENS)
    assert isinstance(meta.obs_groups, RslRlForwardBackwardRunnerCfg.ObservationRoutesCfg)
    assert not hasattr(meta.model, "value_heads")
    assert not hasattr(bfm.model, "value_heads")
    assert all(isinstance(helper, RslRlForwardBackwardRunnerCfg.ValueHelperCfg) for helper in meta.value_helpers)
    assert all(isinstance(helper, RslRlForwardBackwardRunnerCfg.ValueHelperCfg) for helper in bfm.value_helpers)
    assert isinstance(bfm.obs_groups, RslRlForwardBackwardRunnerCfg.ObservationRoutesCfg)
    assert isinstance(meta.model, RslRlForwardBackwardRunnerCfg.ModelCfg)
    assert isinstance(meta.replay, RslRlForwardBackwardRunnerCfg.ReplayCfg)
    assert isinstance(bfm.replay, RslRlForwardBackwardRunnerCfg.ReplayCfg)
    assert meta.replay.history_layout is None
    assert isinstance(bfm.replay.history_layout, RslRlForwardBackwardRunnerCfg.HistoryLayoutCfg)
    assert isinstance(meta.expert, RslRlForwardBackwardRunnerCfg.SequenceExpertCfg)
    assert isinstance(bfm.expert, RslRlForwardBackwardRunnerCfg.SequenceExpertCfg)
    assert isinstance(bfm.lifecycle_extension, RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg)
    assert isinstance(meta.algorithm, RslRlForwardBackwardRunnerCfg.AlgorithmCfg)
    assert not hasattr(bfm.algorithm, "value_cfg")


@pytest.mark.parametrize(
    ("tokens", "training_transitions"),
    (
        (_META_RUNNER_TOKENS, 5_000_000),
        (_BFM_RUNNER_TOKENS, 211_200_000),
    ),
)
def test_runner_budgets_are_exact(tokens: frozenset[str], training_transitions: int) -> None:
    cfg = _runner(tokens)

    schedule = cfg.schedule
    assert schedule.num_envs * schedule.num_steps_per_env * schedule.max_iterations == training_transitions
    assert cfg.replay.policy.capacity_transitions % schedule.num_envs == 0
    assert schedule.random_action_steps % schedule.num_envs == 0


def test_structured_schedule_serializes_once_without_mutating_the_config() -> None:
    meta = _runner(_META_RUNNER_TOKENS)
    g1 = _runner(_BFM_RUNNER_TOKENS)
    meta_schedule = meta.schedule.to_dict()
    g1_schedule = g1.schedule.to_dict()
    meta_unset_steps = meta.num_steps_per_env
    g1_unset_steps = g1.num_steps_per_env

    assert (meta.resolve_num_envs(None, 7), meta.resolve_max_iterations(None)) == (50, 10_000)
    assert (g1.resolve_num_envs(None, 7), g1.resolve_max_iterations(None)) == (1024, 206_250)
    assert g1.resolve_num_envs(16, 7) == 16
    assert g1.resolve_max_iterations(12) == 12

    meta_values = _runtime_values(meta)
    g1_values = _runtime_values(g1)
    assert meta.schedule.to_dict() == meta_schedule
    assert g1.schedule.to_dict() == g1_schedule
    assert meta.num_envs is None and g1.num_envs is None
    assert meta.num_steps_per_env is meta_unset_steps
    assert g1.num_steps_per_env is g1_unset_steps
    assert hasattr(meta.model, "topology") and not hasattr(meta.model, "actor_cfg")
    assert hasattr(meta.replay, "policy") and not hasattr(meta.replay, "capacity_transitions")
    assert hasattr(meta.expert, "clock") and not hasattr(meta.expert, "sampling_mode")
    assert hasattr(meta.algorithm, "optimization") and not hasattr(meta.algorithm, "learning_rate")
    assert "schedule" not in meta_values and "schedule" not in g1_values
    assert "context_policy" not in meta_values and "exploration" not in meta_values
    assert "topology" not in meta_values["model"]
    assert "policy" not in meta_values["replay"]
    assert "clock" not in meta_values["expert"]
    assert "optimization" not in meta_values["algorithm"]
    assert tuple(
        meta_values[name]
        for name in (
            "num_envs",
            "num_steps_per_env",
            "num_updates_per_iteration",
            "random_action_steps",
            "max_iterations",
            "save_interval",
        )
    ) == (50, 10, 50, 50_000, 10_000, 1_000)
    assert tuple(
        g1_values[name]
        for name in (
            "num_envs",
            "num_steps_per_env",
            "num_updates_per_iteration",
            "random_action_steps",
            "max_iterations",
            "save_interval",
        )
    ) == (1024, 1, 16, 10_240, 206_250, 9_375)


def test_effective_cli_scale_serializes_without_mutating_the_schedule() -> None:
    cfg = _runner(_BFM_RUNNER_TOKENS)
    schedule = cfg.schedule.to_dict()
    cfg.num_envs = cfg.resolve_num_envs(16, 7)
    cfg.max_iterations = cfg.resolve_max_iterations(12)

    values = cfg.to_dict()

    assert (values["num_envs"], values["max_iterations"]) == (16, 12)
    assert cfg.schedule.to_dict() == schedule
    assert (cfg.schedule.num_envs, cfg.schedule.max_iterations) == (1024, 206_250)


def test_explicit_runner_scale_precedes_the_structured_schedule() -> None:
    cfg = _runner(_BFM_RUNNER_TOKENS)
    cfg.num_envs = 32
    cfg.max_iterations = 24

    assert cfg.resolve_num_envs(None, 7) == 32
    assert cfg.resolve_max_iterations(None) == 24
    assert cfg.resolve_num_envs(16, 7) == 16
    assert cfg.resolve_max_iterations(12) == 12
    values = cfg.to_dict()
    assert (values["num_envs"], values["max_iterations"]) == (32, 24)


def test_train_and_play_record_effective_scale_before_serialization() -> None:
    repository_root = Path(__file__).parents[6]
    scripts = repository_root / "scripts/reinforcement_learning/rsl_rl"
    train_source = (scripts / "train.py").read_text()
    phase3_train_source = (scripts / "train_rsl_rl.py").read_text()
    play_source = (scripts / "play.py").read_text()

    assert "agent_cfg.num_envs = resolved_num_envs" in train_source
    assert "agent_cfg.num_envs = resolved_num_envs" in phase3_train_source
    assert "agent_cfg.max_iterations = agent_cfg.resolve_max_iterations" in phase3_train_source
    assert "agent_cfg.num_envs = env_cfg.scene.num_envs" in play_source


def test_canonical_compositions_match_frozen_final_dictionaries() -> None:
    """Every resolved scalar, structure, and runtime path remains exact."""
    meta = _runner(_META_RUNNER_TOKENS)
    bfm = _runner(_BFM_RUNNER_TOKENS)

    assert _runner_sha256(meta) == _META_RUNNER_SHA256
    assert _runner_sha256(bfm) == _BFM_RUNNER_SHA256
    provider = "isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert:forward_backward_expert_buffer"
    assert meta.expert.provider == bfm.expert.provider == provider
    assert meta.expert.target_projection.endswith("motion.robots.smpl.observations:smpl_expert_target")
    assert bfm.expert.target_projection.endswith("motion.robots.g1.observations:g1_bfm_expert_target")


def test_expert_projection_owners_are_resolved_by_robot_axis() -> None:
    """Expert projections must serialize the exact robot-owned bindings they consume."""
    meta = _runner(_META_RUNNER_TOKENS)
    bfm = _runner(_BFM_RUNNER_TOKENS)

    robot_bind = "env.unwrapped.scene['robot']"
    action_bind = "env.unwrapped.action_manager.get_term('joint_position')"
    assert meta.expert.target_projection_binds == (robot_bind,)
    assert bfm.expert.target_projection_binds == (robot_bind, action_bind)
    assert bfm.to_dict()["expert"]["target_projection_binds"] == (robot_bind, action_bind)


def test_runner_default_is_smpl_cmu_metamotivo() -> None:
    """The no-token task default must be one complete, trainable runner."""
    default = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=set())
    explicit = _runner(_META_RUNNER_TOKENS)

    assert isinstance(default, MotionForwardBackwardRunnerCfg)
    assert default.to_dict() == explicit.to_dict()
    assert _runner_sha256(default) == _META_RUNNER_SHA256


def test_runner_instances_do_not_share_mutable_sections() -> None:
    first = _runner(_META_RUNNER_TOKENS)
    second = _runner(_META_RUNNER_TOKENS)
    bfm = _runner(_BFM_RUNNER_TOKENS)

    first.model.topology.actor.hidden_layers = -1
    first.value_helpers[0].terms[0].sign = -1
    first.value_helpers[0].actor_coefficient = -1.0

    assert second.model.topology.actor.hidden_layers == 2
    assert second.value_helpers[0].terms[0].sign == 1
    assert second.value_helpers[0].actor_coefficient == 0.01
    assert bfm.model.topology.actor.hidden_layers == 6


def test_smpl_metamotivo_composition_matches_reference_contract() -> None:
    cfg = _runner(_META_RUNNER_TOKENS)
    values = _runtime_values(cfg)

    assert cfg.class_name == "OffPolicyRunner"
    assert cfg.seed == 0
    assert values["num_envs"] == 50
    assert values["num_steps_per_env"] == 10
    assert values["num_updates_per_iteration"] == 50
    assert values["random_action_steps"] == 50_000
    assert not cfg.init_at_random_ep_len
    assert values["max_iterations"] == 10_000
    assert cfg.lifecycle_extension is None
    assert values["replay"]["capacity_transitions"] == 2_000_000
    assert values["replay"]["autoreset_mode"] == "same_step"
    assert values["replay"]["sampling"] == "transition_uniform"
    assert values["replay"]["auxiliary_evidence_observation_group"] is None
    assert values["replay"]["history_layout"] is None
    assert _channel(values, "discriminator")["timing"] == "state"
    assert values["expert"]["window_lengths"] == (8,)
    assert values["expert"]["sampling_mode"] == "source_rows"
    assert values["expert"]["sampling_step_seconds"] is None
    assert values["algorithm"]["random_action_range"] == (-1.0, 1.0)
    assert values["algorithm"]["implied_value_coefficient"] == 0.1
    assert values["model"]["actor_cfg"] == {
        "hidden_dim": 1024,
        "hidden_layers": 2,
        "embedding_layers": 2,
        "residual": False,
    }
    assert values["model"]["forward_cfg"] == values["model"]["actor_cfg"]
    assert values["obs_groups"]["actor"] == ["policy"]
    assert values["obs_groups"]["backward"] == ["policy"]
    assert "critic_auxiliary" not in values["obs_groups"]
    assert not hasattr(cfg.obs_groups, "critic_auxiliary")


def test_g1_bfm_zero_composition_matches_reference_contract() -> None:
    cfg = _runner(_BFM_RUNNER_TOKENS)
    values = _runtime_values(cfg)

    assert cfg.seed == 4728
    assert values["num_envs"] == 1024
    assert values["num_steps_per_env"] == 1
    assert values["num_updates_per_iteration"] == 16
    assert values["random_action_steps"] == 10_240
    assert not cfg.init_at_random_ep_len
    assert values["max_iterations"] == 206_250
    assert values["lifecycle_extension"] == {
        "class_name": (
            "isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking:ForwardBackwardTrackingLifecycle"
        ),
        "transition_interval": 9_600_000,
        "command_bind": "env.unwrapped.command_manager.get_term('motion')",
        "sequence_ids_bind": "command.table.clip_ids",
        "sequence_start_rows_bind": "command.table.clip_start_rows",
        "sampling_priorities_bind": "command.payload.sampler.clip_priorities",
        "evaluation_scope_bind": "command.payload.sampler.reset_sampling_scope",
        "projections": (
            {
                "metric_name": "emd",
                "target_name": "joint_position",
                "observation_name": "joint_position_unnoised",
                "projection": None,
                "assignment_metric": "uniform_assignment",
            },
            {
                "metric_name": "obs_state_emd",
                "target_name": "joint_position",
                "observation_name": "joint_position",
                "projection": (
                    "isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_observation_state_pose"
                ),
                "assignment_metric": "uniform_assignment",
            },
        ),
        "context_window_length": 1,
        "include_reset_frame": True,
        "allow_horizon_truncation": True,
        "shuffle_assignments": True,
        "priority_metric_name": "emd",
        "priority_metric_minimum": 0.5,
        "priority_metric_maximum": 2.0,
        "priority_exponent_scale": 2.0,
        "priority_exponent_base": 2.0,
        "reset_source_name": "reference",
        "evaluation_seed": 0,
    }
    assert values["replay"]["capacity_transitions"] == 5_120_000
    assert values["replay"]["environment_reward_name"] is None
    assert values["replay"]["terminal_capacity_per_env"] == 17
    assert values["replay"]["sampling"] == "episode_uniform"
    assert values["replay"]["auxiliary_evidence_observation_group"] == "transition"
    assert _channel(values, "discriminator")["timing"] == "state"
    assert values["expert"]["window_lengths"] == (8, 257)
    assert values["expert"]["sampling_mode"] == "uniform_before_source_end"
    assert values["expert"]["sampling_step_seconds"] == 0.02
    assert values["algorithm"]["random_action_range"] == (-5.0, 5.0)
    assert values["algorithm"]["rollout_expert_fraction"] == 0.5
    assert values["model"]["actor_cfg"] == {
        "hidden_dim": 1024,
        "hidden_layers": 6,
        "embedding_layers": 2,
        "residual": True,
    }
    assert values["model"]["forward_cfg"] == {
        "hidden_dim": 1024,
        "hidden_layers": 6,
        "embedding_layers": 6,
        "residual": True,
    }
    assert values["model"]["normalization_groups"] == (
        {
            "name": "state",
            "fields": ("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"),
        },
    )
    state_route = ["joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"]
    evaluator_route = [*state_route, "privileged_state", "last_action", "history_actor"]
    assert values["obs_groups"]["forward"] == evaluator_route
    assert values["obs_groups"]["critic_discriminator"] == evaluator_route
    assert "critic_auxiliary" not in values["obs_groups"]
    assert values["model"]["value_heads"][1]["spec"]["route"] == "critic_discriminator"
    assert values["obs_groups"]["actor"] == [*state_route, "last_action", "history_actor"]
    assert values["obs_groups"]["backward"] == [*state_route, "privileged_state"]


def test_tracking_reset_source_follows_the_independent_robot_axis() -> None:
    """Robot projection and lifecycle protocol remain independent preset axes."""
    smpl = _runner((_META_RUNNER_TOKENS - {"tracking_off"}) | {"tracking_source_edge"})
    g1 = _runner(_BFM_RUNNER_TOKENS)

    assert isinstance(smpl.lifecycle_extension, RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg)
    assert isinstance(g1.lifecycle_extension, RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg)
    assert smpl.lifecycle_extension.reset_source_name == "reference"
    assert g1.lifecycle_extension.reset_source_name == "reference"
    assert (
        smpl.lifecycle_extension.context_window_length,
        smpl.lifecycle_extension.include_reset_frame,
        smpl.lifecycle_extension.allow_horizon_truncation,
        smpl.lifecycle_extension.shuffle_assignments,
    ) == (8, False, False, False)
    assert (
        g1.lifecycle_extension.context_window_length,
        g1.lifecycle_extension.include_reset_frame,
        g1.lifecycle_extension.allow_horizon_truncation,
        g1.lifecycle_extension.shuffle_assignments,
    ) == (1, True, True, True)
    assert all(
        projection.assignment_metric == "uniform_assignment" for projection in smpl.lifecycle_extension.projections
    )
    assert all(
        projection.assignment_metric == "uniform_assignment" for projection in g1.lifecycle_extension.projections
    )
    assert smpl.lifecycle_extension.projections[0].projection.endswith(":smpl_humenv_tracking_pose")
    assert g1.lifecycle_extension.projections[1].projection.endswith(":g1_bfm_observation_state_pose")


def test_expert_windows_follow_context_policy_not_tracking_policy() -> None:
    g1_without_tracking = _runner((_BFM_RUNNER_TOKENS - {"tracking_reset_frame"}) | {"tracking_off"})
    smpl_with_tracking = _runner((_META_RUNNER_TOKENS - {"tracking_off"}) | {"tracking_source_edge"})

    assert g1_without_tracking.lifecycle_extension is None
    assert g1_without_tracking.context_policy.expert_window_lengths == (8, 257)
    assert smpl_with_tracking.lifecycle_extension is not None
    assert smpl_with_tracking.context_policy.expert_window_lengths == (8,)


def test_runner_seed_is_the_single_stochastic_seed_owner() -> None:
    meta = _runner(_META_RUNNER_TOKENS)
    g1 = _runner(_BFM_RUNNER_TOKENS)

    assert (meta.seed, meta.replay.seed, meta.expert.seed, meta.algorithm.seed) == (0, None, None, None)
    assert (g1.seed, g1.replay.seed, g1.expert.seed, g1.algorithm.seed) == (4728, None, None, None)


def test_g1_history_layout_uses_complete_semantic_fields() -> None:
    values = _runtime_values(_runner(_BFM_RUNNER_TOKENS))
    history = values["replay"]["history_layout"]

    assert history == {
        "history_field": "history_actor",
        "history_length": 4,
        "include_seed_observations": False,
        "sources": [
            {"observation_name": "last_action"},
            {"observation_name": "base_angular_velocity"},
            {"observation_name": "joint_position"},
            {"observation_name": "joint_velocity"},
            {"observation_name": "projected_gravity"},
        ],
    }
    widths = {
        "last_action": 29,
        "base_angular_velocity": 3,
        "joint_position": 29,
        "joint_velocity": 29,
        "projected_gravity": 3,
    }
    assert sum(widths[source["observation_name"]] for source in history["sources"]) * 4 == 372
    assert all(set(source) == {"observation_name"} for source in history["sources"])


def test_value_helpers_are_the_single_order_owner() -> None:
    """One typed helper algebra must derive every ordered runtime view."""
    cfg = _runner(_BFM_RUNNER_TOKENS)
    expected = (
        ("discriminator", (("discriminator", 1.0),)),
        (
            "auxiliary",
            (
                ("penalty_torques", 0.0),
                ("penalty_action_rate", 0.1),
                ("limits_dof_pos", 10.0),
                ("limits_torque", 0.0),
                ("penalty_undesired_contact", 1.0),
                ("penalty_feet_ori", 0.4),
                ("penalty_ankle_roll", 4.0),
                ("penalty_slippage", 2.0),
            ),
        ),
    )
    assert (
        tuple(
            (helper.name, tuple((term.name, term.coefficient) for term in helper.terms)) for helper in cfg.value_helpers
        )
        == expected
    )

    environment_term = cfg.value_helpers[0].terms[0]
    environment_term.name = "renamed_environment"
    environment_term.source = "environment"
    environment_term.timing = "transition"
    environment_term.context_dependent = False
    cfg.value_helpers[1].terms = tuple(reversed(cfg.value_helpers[1].terms))
    values = _runtime_values(cfg)
    declared_names = [term.name for helper in cfg.value_helpers for term in helper.terms]
    stored_names = [
        term.name for helper in cfg.value_helpers for term in helper.terms if term.source == "stored_evidence"
    ]

    assert [channel["name"] for channel in values["replay"]["reward_channels"]] == declared_names
    assert values["replay"]["auxiliary_evidence_names"] == stored_names
    assert values["replay"]["environment_reward_name"] == "renamed_environment"
    assert list(values["algorithm"]["value_cfg"]) == [helper.name for helper in cfg.value_helpers]
    for helper, value_head in zip(cfg.value_helpers, values["model"]["value_heads"], strict=True):
        term_names = [term.name for term in helper.terms]
        assert value_head["spec"]["name"] == helper.name
        assert value_head["spec"]["reward_channels"] == term_names
        assert values["algorithm"]["value_cfg"][helper.name]["reward_coefficients"] == tuple(
            term.coefficient for term in helper.terms
        )

    source = Path(rsl_rl_fb_cfg.__file__).read_text()
    auxiliary_terms = expected[1][1]
    assert all(source.count(f'"{name}"') == 1 for name, _ in auxiliary_terms)


def test_g1_cmu_uses_the_same_robot_and_bfm_axes_as_g1_lafan() -> None:
    """Changing only the source leaves the robot/algorithm runner tree coherent and unchanged."""
    native = _runner(_BFM_RUNNER_TOKENS)
    cross = _runner(_G1_CMU_RUNNER_TOKENS)

    assert cross.to_dict() == native.to_dict()
    assert cross.expert.provider.endswith(":forward_backward_expert_buffer")
    assert cross.expert.target_projection.endswith(":g1_bfm_expert_target")
    assert cross.expert.clock.sampling_mode == "uniform_before_source_end"
    assert cross.expert.target_projection_binds == native.expert.target_projection_binds
    assert len(cross.expert.target_projection_binds) == 2
    assert cross.expert.clock.sampling_step_seconds == 0.02
    assert cross.replay.history_layout.history_field == "history_actor"
    assert [helper.name for helper in cross.value_helpers] == ["discriminator", "auxiliary"]
