# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.managers import ManagerTermBaseCfg

import isaaclab_tasks.core.multi_task.curriculum.event_combinators as event_combinators
from isaaclab_tasks.core.multi_task.curriculum.sampling import FrontierSamplingStrategyCfg, SamplerCfg
from isaaclab_tasks.core.multi_task.curriculum.success_monitor_cfg import SuccessMonitorCfg

reset_accumulator = event_combinators.reset_accumulator


def _make_accumulator(capacity: int, state_dim: int, target_size: int | None = None) -> reset_accumulator:
    acc = reset_accumulator.__new__(reset_accumulator)
    acc.state_data = torch.zeros((capacity, state_dim))
    acc.state_tag_indices = torch.full((capacity,), -1, dtype=torch.int64)
    acc.state_tag_names = None
    acc._state_target_size = capacity if target_size is None else target_size
    acc._state_fps_features = None
    acc._state_tag_names_bind = None
    acc._tag_indices_bind = None
    acc._requested_reset_assets = []
    acc.reset_assets = []
    acc.acceptance_conditions = {}
    acc._success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=2)
    acc.monitor_success_rate = None
    acc.success_monitor = None
    acc.precollecting_phase = True
    acc._sampler = None
    acc._wandb_3d_log_state = {}
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

    monkeypatch.setattr(event_combinators.reset_state, "get_reset_state", lambda *_args: torch.zeros(1, 2))
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
            "sampling": SimpleNamespace(max_samples=None),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=2),
        }
    )

    acc = reset_accumulator(cfg, env)

    assert acc.acceptance_conditions["ok"] is condition


def test_accumulator_precollect_appends_direct_data_and_tag_indices(monkeypatch):
    acc = _make_accumulator(capacity=5, state_dim=2)
    acc._env = SimpleNamespace()
    acc._state_tag_names_bind = "env.tag_names"
    acc._tag_indices_bind = "env.tag_indices"
    acc._sampling_cfg = SimpleNamespace()

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

    acc.sampled_slots = torch.zeros(env.num_envs, dtype=torch.long)
    acc(
        env,
        torch.empty(0, dtype=torch.long),
        reset_term,
        reset_assets=[],
        acceptance_conditions={},
        state_table_size=acc._state_target_size,
        success_monitor_cfg=acc._success_monitor_cfg,
        sampling=acc._sampling_cfg,
    )

    assert acc.state_tag_names == env.tag_names
    torch.testing.assert_close(acc.state_data, torch.cat([rows, rows[:2]], dim=0))
    torch.testing.assert_close(acc.state_tag_indices, torch.tensor([0, 1, 2, 0, 1]))
    assert acc.monitor_success_rate.shape == (5,)
    assert acc.success_monitor.success_buf.shape == (5, 2)


def test_accumulator_state_compact_precedes_monitor_materialization(monkeypatch):
    torch.manual_seed(0)
    capacity = 9
    target = 3
    acc = _make_accumulator(capacity=capacity, state_dim=3, target_size=target)
    acc._env = SimpleNamespace()
    acc._tag_indices_bind = "env.tag_indices"
    acc._sampling_cfg = SimpleNamespace()

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
    acc.sampled_slots = torch.zeros(env.num_envs, dtype=torch.long)
    acc(
        env,
        torch.empty(0, dtype=torch.long),
        reset_term,
        reset_assets=[],
        acceptance_conditions={},
        state_table_size=acc._state_target_size,
        success_monitor_cfg=acc._success_monitor_cfg,
        sampling=acc._sampling_cfg,
    )

    assert acc.state_data.shape[0] == target
    assert acc.state_data.shape == (target, 3)
    assert acc.state_tag_indices.shape == (target,)
    for row, tag in zip(acc.state_data, acc.state_tag_indices, strict=True):
        matches = (states == row).all(dim=1).nonzero(as_tuple=False).squeeze(-1)
        assert matches.numel() == 1
        assert tag.item() == tag_indices[matches[0]].item()
    assert acc.monitor_success_rate.shape == (target,)
    assert acc.success_monitor.success_buf.shape == (target, 2)


def test_accumulator_frontier_samples_from_state_feature_space_without_wandb(monkeypatch):
    acc = _make_accumulator(capacity=4, state_dim=4)
    acc._env = SimpleNamespace(extras={})
    acc.precollecting_phase = False
    acc.state_data[:] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 11.0],
            [2.0, 0.0, 0.0, 20.0],
            [3.0, 0.0, 0.0, 21.0],
        ]
    )
    feature_calls = {"n": 0}

    def features(states):
        feature_calls["n"] += 1
        return states[:, [0, 3]]

    acc._state_fps_features = features
    acc._sampling_cfg = SamplerCfg(
        strategies=[FrontierSamplingStrategyCfg(k=2, dilation_steps=1, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    )

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
    acc.sampled_slots = torch.zeros(env.num_envs, dtype=torch.long)
    acc(
        env,
        torch.tensor([0, 1]),
        reset_term,
        reset_assets=[],
        acceptance_conditions={},
        state_table_size=acc._state_target_size,
        success_monitor_cfg=acc._success_monitor_cfg,
        sampling=acc._sampling_cfg,
    )

    assert acc._sampler is not None
    assert feature_calls["n"] == 1
    assert applied["states"].shape == (2, 4)
    torch.testing.assert_close(applied["env_ids"], torch.tensor([0, 1]))
