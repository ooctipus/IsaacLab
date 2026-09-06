# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for cold, expression-bound task observation caches."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


class _Sensor:
    def __init__(self) -> None:
        self.calls: list[tuple[float, bool]] = []

    def update(self, dt: float, force_recompute: bool = False) -> None:
        self.calls.append((dt, force_recompute))


class _ObservationManager:
    def __init__(self, term) -> None:
        self.term = term
        self.update_history: list[bool] = []

    def compute(self, update_history: bool = False):
        self.update_history.append(update_history)
        rows = self.term.cmd_indices.float()
        physical = self.term.physical_rows.float()
        return {
            "policy": torch.stack((rows, physical), dim=-1),
            "nested": {"task": rows.unsqueeze(-1)},
        }


class _StateCommand:
    def __init__(self, num_envs: int, num_tasks: int) -> None:
        self.table = SimpleNamespace(num_tasks=num_tasks)
        self.cmd_indices = torch.zeros(num_envs, dtype=torch.long)
        self.physical_rows = torch.zeros(num_envs, dtype=torch.long)
        self.command = torch.zeros(num_envs, 1)
        self.error = torch.zeros(num_envs, 1)
        self.command_updates: list[float] = []
        self.spawn_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.target_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def bind_rows(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        self.spawn_calls.append((env_ids.clone(), task_rows.clone()))
        self.cmd_indices[env_ids] = task_rows
        self.physical_rows[env_ids] = task_rows + 10

    def bind_rows_target(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        self.target_calls.append((env_ids.clone(), task_rows.clone()))
        self.cmd_indices[env_ids] = task_rows
        self.physical_rows[env_ids] = task_rows + 100

    def materialize(self) -> None:
        self.command_updates.append(0.0)
        self.command.copy_(self.cmd_indices.unsqueeze(-1))
        self.error.zero_()


def _make_env(num_envs: int = 2, num_tasks: int = 5):
    term = _StateCommand(num_envs, num_tasks)
    sensor = _Sensor()
    scene = SimpleNamespace(sensors={"state": sensor}, update_calls=[])
    scene.update = lambda dt: scene.update_calls.append(dt)
    sim = SimpleNamespace(forward_calls=0)
    sim.forward = lambda: setattr(sim, "forward_calls", sim.forward_calls + 1)
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        sim=sim,
        scene=scene,
        observation_manager=_ObservationManager(term),
        command_manager=SimpleNamespace(get_term=lambda _name: term),
    )
    return env, term, sensor


def test_spawn_observation_cache_covers_rows_and_matching_commands() -> None:
    """Every row should pair its spawn state with the same command row."""
    from isaaclab_tasks.core.multi_task.curriculum.observation_cache import evaluate_observation_cache_bind

    env, term, sensor = _make_env()
    cache = evaluate_observation_cache_bind(
        "materialize_state_command_observations(env, 'goal_point')",
        env,
    )

    assert isinstance(cache, TensorDict)
    assert cache.batch_size == torch.Size([5])
    assert cache.is_locked
    torch.testing.assert_close(cache["policy"][:, 0], torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(cache["policy"][:, 1], torch.arange(10, 15, dtype=torch.float32))
    torch.testing.assert_close(cache["nested", "task"].squeeze(-1), torch.arange(5, dtype=torch.float32))
    assert [rows.tolist() for _, rows in term.spawn_calls] == [[0, 1], [2, 3], [4]]
    assert env.sim.forward_calls == 3
    assert env.scene.update_calls == [0.0, 0.0, 0.0]
    assert sensor.calls == [(0.0, True), (0.0, True), (0.0, True)]
    assert env.observation_manager.update_history == [False, False, False]
    assert term.command_updates == [0.0, 0.0, 0.0]


def test_target_observation_cache_uses_domain_target_binding() -> None:
    """Goal observations should use target physics with the matching task row."""
    from isaaclab_tasks.core.multi_task.curriculum.observation_cache import evaluate_observation_cache_bind

    env, term, _sensor = _make_env()
    cache = evaluate_observation_cache_bind(
        "materialize_state_command_target_observations(env, 'goal_point')",
        env,
    )

    torch.testing.assert_close(cache["policy"][:, 0], torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(cache["policy"][:, 1], torch.arange(100, 105, dtype=torch.float32))
    assert [rows.tolist() for _, rows in term.target_calls] == [[0, 1], [2, 3], [4]]
    assert term.command_updates == [0.0, 0.0, 0.0]


def test_observation_cache_uses_only_state_command_public_boundary() -> None:
    """Cold cache construction must not mutate payload or row storage directly."""
    from isaaclab_tasks.core.multi_task.curriculum import observation_cache

    source = inspect.getsource(observation_cache)
    assert ".payload" not in source
    assert ".cmd_indices" not in source


def test_observation_cache_rejects_materialization_after_first_reset() -> None:
    """Cold materialization must fail instead of mutating a running rollout."""
    from isaaclab_tasks.core.multi_task.curriculum.observation_cache import evaluate_observation_cache_bind

    env, _term, _sensor = _make_env()
    env.obs_buf = {}

    with pytest.raises(RuntimeError, match="before the first environment reset"):
        evaluate_observation_cache_bind(
            "materialize_state_command_observations(env, 'goal_point')",
            env,
        )


def test_observation_cache_materialization_preserves_global_rng() -> None:
    """Building an optional cache must not shift training random streams."""
    from isaaclab_tasks.core.multi_task.curriculum.observation_cache import evaluate_observation_cache_bind

    env, _term, _sensor = _make_env()
    torch.manual_seed(7123)
    before = torch.get_rng_state().clone()

    evaluate_observation_cache_bind(
        "materialize_state_command_observations(env, 'goal_point')",
        env,
    )

    torch.testing.assert_close(torch.get_rng_state(), before)


def test_observation_cache_owner_accepts_arbitrary_tensor_dict_expression() -> None:
    """The cache owner should not impose command or task-state configuration fields."""
    from isaaclab_tasks.core.multi_task.curriculum import ObservationCache

    observations = TensorDict({"policy": torch.zeros(3, 2)}, batch_size=[3])
    env = SimpleNamespace(precomputed=observations)
    cfg = SimpleNamespace(params={"observations_bind": "env.precomputed"})

    cache = ObservationCache(cfg, env)

    assert cache.observations is observations
    assert cache.observations.is_locked


def test_observation_cache_evaluator_accepts_full_sampler_bindings() -> None:
    """Cache expressions may use any binding injected by the owning sampler."""
    from isaaclab_tasks.core.multi_task.curriculum.observation_cache import evaluate_observation_cache_bind

    observations = TensorDict({"policy": torch.zeros(3, 2)}, batch_size=[3])
    result = evaluate_observation_cache_bind(
        "precomputed",
        SimpleNamespace(),
        {"precomputed": observations},
    )

    assert result is observations
    assert result.is_locked


def test_value_shift_rejects_cache_layout_mismatch() -> None:
    """A cache row must correspond to exactly one sampler layout item."""
    from isaaclab_tasks.core.multi_task.curriculum import StateLayout, ValueShiftSamplingStrategyCfg
    from isaaclab_tasks.core.multi_task.curriculum.sampling.sampling_strategies import ValueShiftSamplingStrategy

    cache = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    layout = StateLayout(coords=torch.zeros(3, 1), spawn_index=torch.arange(3))
    cfg = ValueShiftSamplingStrategyCfg(obs_cache_bind="cache")

    with pytest.raises(ValueError, match=r"rows \(2\).*items \(3\)"):
        ValueShiftSamplingStrategy(cfg, layout, env=SimpleNamespace(), cache=cache)


def test_sampler_exposes_configured_value_shift_strategy() -> None:
    """The public sampler boundary should expose the actual configured strategy."""
    from isaaclab_tasks.core.multi_task.curriculum import (
        Sampler,
        SamplerCfg,
        StateLayout,
        ValueShiftSamplingStrategyCfg,
    )

    cache = TensorDict({"policy": torch.zeros(3, 2)}, batch_size=[3])
    layout = StateLayout(coords=torch.zeros(3, 1), spawn_index=torch.arange(3))
    sampler = Sampler(
        SamplerCfg(strategies=[ValueShiftSamplingStrategyCfg(obs_cache_bind="cache")]),
        layout,
        env=SimpleNamespace(),
        cache=cache,
    )

    assert sampler.value_shift.observation_cache is cache
