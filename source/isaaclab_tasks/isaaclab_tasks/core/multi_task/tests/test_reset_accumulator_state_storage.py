# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.managers import ManagerTermBaseCfg

import isaaclab_tasks.core.multi_task.curriculum.event_combinators as event_combinators
from isaaclab_tasks.core.multi_task.curriculum.sampling import SamplerCfg, UniformSamplingStrategyCfg
from isaaclab_tasks.core.multi_task.curriculum.state_layout import StateLayoutCfg
from isaaclab_tasks.core.multi_task.curriculum.success_monitor_cfg import SuccessMonitorCfg
from isaaclab_tasks.core.multi_task.mdp.curriculums import success_rate_sampler

reset_accumulator = event_combinators.reset_accumulator


def _make_accumulator(capacity: int, state_dim: int, target_size: int | None = None) -> reset_accumulator:
    acc = reset_accumulator.__new__(reset_accumulator)
    acc.state_data = torch.zeros((capacity, state_dim))
    acc.state_tag_indices = torch.full((capacity,), -1, dtype=torch.int64)
    acc.state_tag_names = None
    acc.state_coords = None
    acc.slot_indices = None
    acc._state_target_size = capacity if target_size is None else target_size
    acc._state_fps_features = None
    acc._state_tag_names_bind = None
    acc._tag_indices_bind = None
    acc.reset_assets = []
    acc.acceptance_conditions = {}
    acc.monitor_success_rate = None
    acc.precollecting_phase = False
    return acc


def _termination_manager(num_envs: int) -> SimpleNamespace:
    return SimpleNamespace(
        get_term_cfg=lambda _name: SimpleNamespace(
            func=SimpleNamespace(is_success=torch.zeros(num_envs, dtype=torch.bool))
        )
    )


def test_accumulator_init_accepts_resolved_manager_term_cfg_conditions(monkeypatch) -> None:
    """Resolved nested manager-term cfgs should be valid acceptance conditions."""

    class Condition:
        def __call__(self, _env, env_ids):
            return torch.ones_like(env_ids, dtype=torch.bool)

    monkeypatch.setattr(event_combinators.reset_state, "get_reset_state", lambda *_args, **_kwargs: torch.zeros(1, 2))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=1,
        scene=SimpleNamespace(_articulations={}, _rigid_objects={}),
    )
    condition = ManagerTermBaseCfg(func=Condition())
    cfg = SimpleNamespace(
        params={
            "acceptance_conditions": {"ok": condition},
            "reset_assets": [],
            "state_table_size": 4,
            "reset_term": SimpleNamespace(func=lambda *_args, **_kwargs: None, params={}),
        }
    )

    acc = reset_accumulator(cfg, env)

    assert acc.acceptance_conditions["ok"] is condition


def test_accumulator_precollect_appends_direct_data_and_tag_indices(monkeypatch):
    acc = _make_accumulator(capacity=5, state_dim=2)
    acc._env = SimpleNamespace()
    acc._state_tag_names_bind = "env.tag_names"
    acc._tag_indices_bind = "env.tag_indices"

    rows = torch.arange(6).reshape(3, 2).float()
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=3,
        extras={},
        tag_names=["a", "b", "c", "d", "e", "f", "g", "h"],
        tag_indices=torch.arange(3, dtype=torch.int64),
        state_rows=rows,
        termination_manager=_termination_manager(3),
    )

    reset_term = SimpleNamespace(func=lambda *_args, **_kwargs: None, params={})

    def get_reset_state(_env, env_ids, _reset_assets, is_relative=False):
        return env.state_rows[env_ids]

    monkeypatch.setattr(event_combinators.reset_state, "get_reset_state", get_reset_state)

    acc._precollect_state_table(env, reset_term)
    acc._finalize_state_table(env)

    assert acc.state_tag_names == env.tag_names
    torch.testing.assert_close(acc.state_data, torch.cat([rows, rows[:2]], dim=0))
    torch.testing.assert_close(acc.state_tag_indices, torch.tensor([0, 1, 2, 0, 1]))
    assert acc.monitor_success_rate.shape == (5,)
    torch.testing.assert_close(acc.state_coords, acc.state_data)
    torch.testing.assert_close(acc.slot_indices, torch.arange(5))


def test_accumulator_state_compact_precedes_monitor_materialization(monkeypatch):
    torch.manual_seed(0)
    capacity = 9
    target = 3
    acc = _make_accumulator(capacity=capacity, state_dim=3, target_size=target)
    acc._env = SimpleNamespace()
    acc._tag_indices_bind = "env.tag_indices"

    states = torch.cat([torch.arange(capacity).float().unsqueeze(-1), torch.zeros(capacity, 2)], dim=-1)
    tag_indices = torch.arange(capacity, dtype=torch.int64) + 30
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=capacity,
        extras={},
        state_rows=states,
        tag_indices=tag_indices,
        termination_manager=_termination_manager(capacity),
    )
    reset_term = SimpleNamespace(func=lambda *_args, **_kwargs: None, params={})

    def get_reset_state(_env, env_ids, _reset_assets, is_relative=False):
        return env.state_rows[env_ids]

    monkeypatch.setattr(event_combinators.reset_state, "get_reset_state", get_reset_state)
    acc._precollect_state_table(env, reset_term)
    acc._finalize_state_table(env)

    assert acc.state_data.shape[0] == target
    assert acc.state_data.shape == (target, 3)
    assert acc.state_tag_indices.shape == (target,)
    for row, tag in zip(acc.state_data, acc.state_tag_indices, strict=True):
        matches = (states == row).all(dim=1).nonzero(as_tuple=False).squeeze(-1)
        assert matches.numel() == 1
        assert tag.item() == tag_indices[matches[0]].item()
    assert acc.monitor_success_rate.shape == (target,)
    assert acc.state_coords.shape == (target, 3)
    torch.testing.assert_close(acc.slot_indices, torch.arange(target))


def test_accumulator_applies_curriculum_selected_slots(monkeypatch):
    acc = _make_accumulator(capacity=4, state_dim=4)
    acc._env = SimpleNamespace(extras={})
    acc.state_data[:] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 11.0],
            [2.0, 0.0, 0.0, 20.0],
            [3.0, 0.0, 0.0, 21.0],
        ]
    )
    acc.sampled_slots = torch.tensor([3, 1], dtype=torch.long)
    feature_calls = {"n": 0}

    def features(states):
        feature_calls["n"] += 1
        return states[:, [0, 3]]

    acc._state_fps_features = features
    acc._finalize_state_table(SimpleNamespace(device=torch.device("cpu")))

    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        extras={},
        termination_manager=_termination_manager(2),
    )
    reset_term = SimpleNamespace(func=lambda *_args, **_kwargs: None, params={})
    applied = {}

    def set_reset_state(_env, states, env_ids, _reset_assets, is_relative=False):
        applied["states"] = states
        applied["env_ids"] = env_ids

    monkeypatch.setattr(event_combinators.reset_state, "set_reset_state", set_reset_state)
    acc(
        env,
        torch.tensor([0, 1]),
        reset_term,
        reset_assets=[],
        acceptance_conditions={},
        state_table_size=acc._state_target_size,
    )

    assert feature_calls["n"] == 1
    torch.testing.assert_close(acc.state_coords, acc.state_data[:, [0, 3]])
    torch.testing.assert_close(applied["states"], acc.state_data[torch.tensor([3, 1])])
    assert applied["states"].shape == (2, 4)
    torch.testing.assert_close(applied["env_ids"], torch.tensor([0, 1]))


def test_accumulator_external_sampling_applies_existing_slots_without_resampling(monkeypatch):
    acc = _make_accumulator(capacity=3, state_dim=2)
    acc._env = SimpleNamespace(extras={})
    acc.precollecting_phase = False
    acc.state_data[:] = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    acc.sampled_slots = torch.tensor([2, 0], dtype=torch.long)
    acc._finalize_state_table(SimpleNamespace(device=torch.device("cpu")))

    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        extras={},
        termination_manager=_termination_manager(2),
    )
    reset_term = SimpleNamespace(func=lambda *_args, **_kwargs: None, params={})
    applied = {}

    def set_reset_state(_env, states, env_ids, _reset_assets, is_relative=False):
        applied["states"] = states.clone()
        applied["env_ids"] = env_ids.clone()

    monkeypatch.setattr(event_combinators.reset_state, "set_reset_state", set_reset_state)

    acc(
        env,
        torch.tensor([0, 1]),
        reset_term,
        reset_assets=[],
        acceptance_conditions={},
        state_table_size=acc._state_target_size,
    )

    torch.testing.assert_close(acc.state_coords, acc.state_data)
    torch.testing.assert_close(acc.slot_indices, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(applied["states"], torch.tensor([[30.0, 0.0], [10.0, 0.0]]))
    torch.testing.assert_close(applied["env_ids"], torch.tensor([0, 1]))


def test_success_rate_sampler_binds_existing_source_tensors():
    class Source:
        def __init__(self):
            self.monitor_success_rate = torch.zeros(3)
            self.sampled_slots = torch.zeros(2, dtype=torch.long)
            self.state_coords = torch.arange(3, dtype=torch.float32).unsqueeze(-1)
            self.slot_indices = torch.arange(3, dtype=torch.long)

    source = Source()
    env = SimpleNamespace(device=torch.device("cpu"), num_envs=2, source=source, success=torch.tensor([True, False]))
    cfg = SimpleNamespace(
        params={
            "success_rates_bind": "env.source.monitor_success_rate",
            "sample_indices_bind": "env.source.sampled_slots",
            "success_bind": "env.success",
            "layout": StateLayoutCfg(
                coords_bind="env.source.state_coords",
                spawn_index_bind="env.source.slot_indices",
            ),
            "sampling": SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=2),
        }
    )

    term = success_rate_sampler(cfg, env)
    assert term.success_rates is source.monitor_success_rate
    assert term.sample_indices is source.sampled_slots

    result = term(env, torch.tensor([0, 1]), **cfg.params)

    assert source.sampled_slots.shape == (2,)
    assert result["success"].shape == ()


def test_factory_sampler_config_lives_on_command_term():
    from isaaclab_tasks.core.multi_task.factory.config.agents.rsl_rl_ppo_cfg import ValueShiftAlgorithmCfg
    from isaaclab_tasks.core.multi_task.factory.reset_env_cfg import FACTORY_RESET_SAMPLER_PRESETS
    from isaaclab_tasks.core.multi_task.factory_env_cfg import FactoryCommandsCfg, FactoryCurriculumsCfg

    commands = FactoryCommandsCfg()
    curriculum = FactoryCurriculumsCfg()

    assert list(commands.reset_state.commands.keys()) == ["assembly_asset"]
    assert commands.reset_state.commands["assembly_asset"].position_threshold == 0.005
    assert commands.reset_state.commands["assembly_asset"].duration == (0.0, 1.0)
    assert commands.reset_state.payload.reset_assets == ["nistboard", "fixed_asset", "held_asset", "robot"]
    assert commands.reset_state.task_table.pipeline_cfg is not None
    assert commands.reset_state.task_table.pipeline_cfg.board.num_boards == 64
    assert commands.reset_state.task_table.settle_steps == 0
    assert commands.reset_state.task_table.rows_per_board == 30
    assert commands.reset_state.task_table.targets_per_board == 30
    assert commands.reset_state.task_table.stash_viz_geometry is True
    assert commands.reset_state.randomize_command_indices is False
    assert curriculum.reset_sampler.params["sampling"].default.eps == FACTORY_RESET_SAMPLER_PRESETS.default.eps
    assert len(curriculum.reset_sampler.params["sampling"].beta_value_shift.strategies) == 2
    assert curriculum.reset_sampler.params["sample_indices_bind"].endswith(".cmd_indices")
    assert "command_manager.get_term('reset_state')" in curriculum.reset_sampler.params["success_rates_bind"]
    assert curriculum.reset_sampler.params["layout"].target_index_bind is not None
    value_shift_strategy = curriculum.reset_sampler.params["sampling"].beta_value_shift.strategies[1]
    assert "command_manager.get_term('reset_state')" in value_shift_strategy.state_buffer_bind
    assert "curriculum_manager.get_term('reset_sampler')" in ValueShiftAlgorithmCfg().bind_observation_exp
