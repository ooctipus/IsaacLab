# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for unified motion forward-backward agent configurations."""

from __future__ import annotations

from isaaclab_tasks.core.multi_task.motion.config.agents import (
    G1CmuForwardBackwardRunnerCfg,
    G1LafanForwardBackwardRunnerCfg,
    MotionForwardBackwardRunnerPresetsCfg,
    SmplCmuForwardBackwardRunnerCfg,
)
from isaaclab_tasks.core.multi_task.motion.config.presets import G1_LAFAN_PROFILE_CFG, SMPL_CMU_PROFILE_CFG
from isaaclab_tasks.core.multi_task.motion.rsl_rl import (
    motion_expert_buffer_g1,
    motion_expert_buffer_smpl_cmu,
)


def _channel(cfg: dict[str, object], name: str) -> dict[str, object]:
    channels = cfg["replay"]["reward_channels"]  # type: ignore[index]
    return next(channel for channel in channels if channel["name"] == name)  # type: ignore[index,union-attr]


def test_smpl_cfg_matches_same_step_phase2_reference_contract() -> None:
    cfg = SmplCmuForwardBackwardRunnerCfg()
    values = cfg.to_dict()

    assert cfg.class_name == "OffPolicyRunner"
    assert cfg.seed == 0
    assert cfg.num_envs == 50
    assert cfg.num_steps_per_env == 10
    assert cfg.num_updates_per_iteration == 50
    assert cfg.random_action_steps == 50_000
    assert not cfg.init_at_random_ep_len
    assert cfg.max_iterations == 10_000
    assert cfg.lifecycle_extension is None
    assert values["replay"]["capacity_transitions"] == 2_000_000
    assert "capacity_steps" not in values["replay"]
    assert values["replay"]["autoreset_mode"] == "same_step"
    assert values["replay"]["sampling"] == "transition_uniform"
    assert "history_layout" not in values["replay"]
    assert _channel(values, "discriminator")["timing"] == "state"
    assert values["expert"]["provider"] == (
        f"{motion_expert_buffer_smpl_cmu.__module__}:{motion_expert_buffer_smpl_cmu.__name__}"
    )
    assert values["expert"]["command_name"] == "motion"
    assert values["expert"]["window_lengths"] == (8,)
    assert values["algorithm"]["random_action_range"] == (-1.0, 1.0)
    assert values["algorithm"]["implied_value_coefficient"] == 0.1
    assert values["model"]["actor_cfg"] == {
        "hidden_dim": 1024,
        "hidden_layers": 2,
        "embedding_layers": 2,
        "residual": False,
    }
    assert values["model"]["forward_cfg"] == values["model"]["actor_cfg"]
    assert values["obs_groups"]["actor"] == list(SMPL_CMU_PROFILE_CFG.routes.actor_fields)
    assert values["obs_groups"]["backward"] == list(SMPL_CMU_PROFILE_CFG.routes.expert_fields)


def test_g1_cfg_matches_selected_same_step_phase2_reference_contract() -> None:
    cfg = G1LafanForwardBackwardRunnerCfg()
    values = cfg.to_dict()

    assert cfg.seed == 4728
    assert cfg.num_envs == 1024
    assert cfg.num_steps_per_env == 1
    assert cfg.num_updates_per_iteration == 16
    assert cfg.random_action_steps == 10_240
    assert not cfg.init_at_random_ep_len
    assert cfg.max_iterations == 9_375
    assert cfg.lifecycle_extension == {
        "class_name": "isaaclab_tasks.core.multi_task.motion.tracking:MotionTrackingCurriculum",
        "transition_interval": 9_600_000,
        "evaluator": "isaaclab_tasks.core.multi_task.motion.tracking:g1_motion_tracking_evaluator",
        "evaluation_seed": 0,
    }
    collection_block = cfg.num_envs * cfg.num_steps_per_env
    assert cfg.lifecycle_extension["transition_interval"] % collection_block == 0
    assert cfg.lifecycle_extension["transition_interval"] == cfg.max_iterations * collection_block
    assert values["replay"]["capacity_transitions"] == 5_120_000
    assert "capacity_steps" not in values["replay"]
    assert values["replay"]["terminal_capacity_per_env"] == 17
    assert values["replay"]["sampling"] == "episode_uniform"
    assert values["replay"]["autoreset_mode"] == "same_step"
    assert values["replay"]["auxiliary_evidence_names"] == list(G1_LAFAN_PROFILE_CFG.routes.auxiliary_evidence)
    assert _channel(values, "discriminator")["timing"] == "state"
    assert values["expert"]["provider"] == (f"{motion_expert_buffer_g1.__module__}:{motion_expert_buffer_g1.__name__}")
    assert values["expert"]["window_lengths"] == (8, 257)
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
    evaluator_route = ["state", "privileged_state", "last_action", "history_actor"]
    assert list(G1_LAFAN_PROFILE_CFG.routes.forward_fields) == evaluator_route
    assert values["obs_groups"]["forward"] == evaluator_route
    assert values["obs_groups"]["critic_discriminator"] == evaluator_route
    assert values["obs_groups"]["critic_auxiliary"] == evaluator_route
    assert values["obs_groups"]["actor"] == ["state", "last_action", "history_actor"]
    assert values["obs_groups"]["backward"] == ["state", "privileged_state"]


def test_g1_cfg_reconstructs_emitted_processed_action_history_without_seed_rows() -> None:
    values = G1LafanForwardBackwardRunnerCfg().to_dict()
    history = values["replay"]["history_layout"]

    assert history["history_field"] == "history_actor"
    assert history["history_length"] == 4
    assert history["last_action_field"] is None
    assert history["include_seed_observations"] is False
    assert history["sources"] == [
        {"observation_name": "last_action", "start": 0, "stop": 29},
        {"observation_name": "state", "start": 61, "stop": 64},
        {"observation_name": "state", "start": 0, "stop": 29},
        {"observation_name": "state", "start": 29, "stop": 58},
        {"observation_name": "state", "start": 58, "stop": 61},
    ]
    source_width = sum(source["stop"] - source["start"] for source in history["sources"])
    assert source_width * history["history_length"] == G1_LAFAN_PROFILE_CFG.routes.history.width


def test_runner_presets_pair_names_and_defaults_with_motion_presets() -> None:
    """The learner selector should expose the same profile names as the environment."""
    presets = MotionForwardBackwardRunnerPresetsCfg()

    assert isinstance(presets.default, SmplCmuForwardBackwardRunnerCfg)
    assert isinstance(presets.smpl_cmu, SmplCmuForwardBackwardRunnerCfg)
    assert isinstance(presets.g1_lafan, G1LafanForwardBackwardRunnerCfg)
    assert presets.default.num_envs == presets.smpl_cmu.num_envs == 50
    assert isinstance(presets.g1_cmu, G1CmuForwardBackwardRunnerCfg)
    assert presets.g1_lafan.num_envs == 1024

    assert presets.g1_cmu.num_envs == 1024
    assert "composition" in presets.g1_cmu.experiment_name


def test_g1_cfg_preserves_auxiliary_evidence_order_through_value_composition() -> None:
    values = G1LafanForwardBackwardRunnerCfg().to_dict()
    names = list(G1_LAFAN_PROFILE_CFG.routes.auxiliary_evidence)
    channels = values["replay"]["reward_channels"]
    auxiliary_channels = channels[2:]
    value_head = values["model"]["value_heads"][1]
    value_cfg = values["algorithm"]["value_cfg"]["auxiliary"]

    assert [channel["name"] for channel in auxiliary_channels] == names
    assert all(channel["source"] == "stored_evidence" for channel in auxiliary_channels)
    assert all(channel["timing"] == "transition" for channel in auxiliary_channels)
    assert all(channel["sign"] == -1 for channel in auxiliary_channels)
    assert value_head["spec"]["reward_channels"] == names
    assert value_head["spec"]["reward_composition"] == "scalar"
    assert value_cfg["reward_coefficients"] == (0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0)
    assert value_cfg["normalize_rewards"] is True
