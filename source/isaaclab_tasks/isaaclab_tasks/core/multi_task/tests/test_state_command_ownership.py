# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for StateCommand and curriculum state ownership."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp.proxy_array import ProxyArray

from isaaclab_tasks.core.multi_task.curriculum import (
    SamplerCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
    UniformSamplingStrategyCfg,
)
from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command import StateCommand
from isaaclab_tasks.core.multi_task.mdp.curriculums import success_rate_sampler


class _BindingPayload:
    command_dim = 2
    error_dim = 1
    error_names = ("distance",)

    def __init__(self) -> None:
        self.bound: tuple[torch.Tensor, torch.Tensor] | None = None
        self.update_steps: list[float] = []

    def bind(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        self.bound = (env_ids.clone(), task_rows.clone())

    def update(self, step_dt: float, command: torch.Tensor, error: torch.Tensor) -> None:
        self.update_steps.append(step_dt)
        command.fill_(3.0)
        error.fill_(4.0)


def test_state_command_delegates_selected_rows_without_interpreting_table_data() -> None:
    """The generic shell must not require a spawn/target table shape or write reset state."""
    payload = _BindingPayload()
    term = object.__new__(StateCommand)
    term._env = SimpleNamespace(step_dt=0.02, device=torch.device("cpu"))
    term.table = SimpleNamespace(num_tasks=5)
    term._payload = payload
    term.randomize_command_indices = False
    term.cmd_indices = torch.tensor([4, 2, 1], dtype=torch.long)
    term._command = torch.zeros(3, payload.command_dim)
    term._err = torch.zeros(3, payload.error_dim)
    term._debug_vis_handle = None

    env_ids = torch.tensor([0, 2], dtype=torch.long)
    term._resample_command(env_ids)

    assert payload.bound is not None
    torch.testing.assert_close(payload.bound[0], env_ids)
    torch.testing.assert_close(payload.bound[1], torch.tensor([4, 1]))
    assert payload.update_steps == [0.0]
    torch.testing.assert_close(term.command, torch.full_like(term.command, 3.0))
    torch.testing.assert_close(term.error, torch.full_like(term.error, 4.0))


def test_state_command_bind_rows_owns_selection_and_delegates_semantics() -> None:
    """Cold consumers should bind exact rows through the normal command lifecycle."""
    payload = _BindingPayload()
    term = object.__new__(StateCommand)
    term._env = SimpleNamespace(step_dt=0.02, device=torch.device("cpu"))
    term.table = SimpleNamespace(num_tasks=5)
    term._payload = payload
    term.cmd_indices = torch.zeros(3, dtype=torch.long)
    term._command = torch.zeros(3, payload.command_dim)
    term._err = torch.zeros(3, payload.error_dim)

    env_ids = torch.tensor([0, 2], dtype=torch.long)
    rows = torch.tensor([4, 1], dtype=torch.long)
    term.bind_rows(env_ids, rows)

    torch.testing.assert_close(term.cmd_indices, torch.tensor([4, 0, 1]))
    assert payload.bound is not None
    torch.testing.assert_close(payload.bound[0], env_ids)
    torch.testing.assert_close(payload.bound[1], rows)
    assert payload.update_steps == [0.0]


def test_state_command_delegates_random_row_sampling_to_table() -> None:
    """The table must own the row distribution selected during resampling."""

    class _SentinelTable:
        num_tasks = 100

        def __init__(self) -> None:
            self.sample_counts: list[int] = []

        def sample_rows(self, count: int) -> torch.Tensor:
            self.sample_counts.append(count)
            return torch.tensor([7, 3], dtype=torch.long)

    payload = _BindingPayload()
    table = _SentinelTable()
    term = object.__new__(StateCommand)
    term._env = SimpleNamespace(step_dt=0.02, device=torch.device("cpu"))
    term.table = table
    term._payload = payload
    term.randomize_command_indices = True
    term.cmd_indices = torch.zeros(3, dtype=torch.long)
    term._command = torch.zeros(3, payload.command_dim)
    term._err = torch.zeros(3, payload.error_dim)

    env_ids = torch.tensor([0, 2], dtype=torch.long)
    term._resample_command(env_ids)

    assert table.sample_counts == [2]
    torch.testing.assert_close(term.cmd_indices, torch.tensor([7, 0, 3]))
    assert payload.bound is not None
    torch.testing.assert_close(payload.bound[1], torch.tensor([7, 3]))


def test_success_rate_sampler_owns_exact_layout_sized_rate_storage() -> None:
    """Curriculum statistics must not alias an unrelated command-owned tensor."""
    source = SimpleNamespace(
        sampled_rows=torch.zeros(2, dtype=torch.long),
        coords=torch.arange(3, dtype=torch.float32).unsqueeze(-1),
        row_indices=torch.arange(3, dtype=torch.long),
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        source=source,
        success=torch.tensor([True, False]),
    )
    params = {
        "sample_indices_bind": "env.source.sampled_rows",
        "success_bind": "env.success",
        "layout": StateLayoutCfg(
            coords_bind="env.source.coords",
            spawn_index_bind="env.source.row_indices",
        ),
        "sampling": SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
        "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=2),
    }
    term = success_rate_sampler(SimpleNamespace(params=params), env)
    rate_ptr = term.success_rates.data_ptr()

    assert term.success_rates.shape == (3,)
    assert term.success_monitor.success_rate.data_ptr() == rate_ptr
    assert term.sample_indices.data_ptr() == source.sampled_rows.data_ptr()

    result = term(env, torch.tensor([0, 1]), **params)

    assert term.success_rates.data_ptr() == rate_ptr
    assert result["success"].shape == ()


def test_factory_payload_owns_descriptor_binding_and_relative_reset(monkeypatch) -> None:
    """Factory framing and reset writes must stay behind its payload boundary."""
    from isaaclab_tasks.core.multi_task.factory.mdp import reset_state_command_payloads as factory_payloads

    spawn = torch.arange(30, dtype=torch.float32).reshape(2, 15)
    target = spawn + 100.0
    task_rows = torch.tensor([3, 1], dtype=torch.long)
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    origins = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    calls: dict[str, tuple] = {}

    class _Table:
        @staticmethod
        def gather(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            torch.testing.assert_close(rows, task_rows)
            return spawn, target

    payload = object.__new__(factory_payloads.FactoryAssemblyPayload)
    payload.table = _Table()
    payload._env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    payload._states_relative = True
    payload.reset_assets = ["robot", "held_asset"]

    def bind_target(
        bound_env_ids: torch.Tensor,
        bound_task_rows: torch.Tensor,
        target_states: torch.Tensor,
        target_origin: torch.Tensor,
    ) -> None:
        calls["target"] = (bound_env_ids, bound_task_rows, target_states, target_origin)

    def set_reset_state(env, states, bound_env_ids, reset_assets, is_relative):
        calls["reset"] = (env, states, bound_env_ids, reset_assets, is_relative)

    payload._bind_target = bind_target
    monkeypatch.setattr(factory_payloads, "set_reset_state", set_reset_state)

    payload.bind(env_ids, task_rows)

    target_call = calls["target"]
    torch.testing.assert_close(target_call[0], env_ids)
    torch.testing.assert_close(target_call[1], task_rows)
    torch.testing.assert_close(target_call[2], target)
    torch.testing.assert_close(target_call[3], origins[env_ids])
    reset_call = calls["reset"]
    torch.testing.assert_close(reset_call[1], spawn)
    torch.testing.assert_close(reset_call[2], env_ids)
    assert reset_call[3:] == (payload.reset_assets, True)

    payload.bind_target(env_ids, task_rows)
    reset_call = calls["reset"]
    torch.testing.assert_close(reset_call[1], target)
    torch.testing.assert_close(reset_call[2], env_ids)
    assert reset_call[3:] == (payload.reset_assets, True)


def test_position_payload_bind_target_writes_target_state(monkeypatch) -> None:
    """Position goal materialization must write target rows rather than spawn rows."""
    from isaaclab_tasks.core.multi_task.terrain.mdp.commands import state_command_payloads

    spawn = torch.arange(30, dtype=torch.float32).reshape(2, 15)
    target = spawn + 100.0
    task_rows = torch.tensor([1, 0], dtype=torch.long)
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    origins = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    calls: dict[str, tuple] = {}

    class _Table:
        @staticmethod
        def gather(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            torch.testing.assert_close(rows, task_rows)
            return spawn, target

    payload = object.__new__(state_command_payloads.CommandPayloadBase)
    payload.table = _Table()
    payload._env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    payload._states_relative = True
    payload.reset_assets = ["robot"]

    def bind_target(
        bound_env_ids: torch.Tensor,
        bound_task_rows: torch.Tensor,
        target_states: torch.Tensor,
        target_origin: torch.Tensor,
    ) -> None:
        calls["target"] = (bound_env_ids, bound_task_rows, target_states, target_origin)

    def set_reset_state(env, states, bound_env_ids, reset_assets, is_relative):
        calls["reset"] = (env, states, bound_env_ids, reset_assets, is_relative)

    payload._bind_target = bind_target
    monkeypatch.setattr(state_command_payloads, "set_reset_state", set_reset_state)

    payload.bind_target(env_ids, task_rows)

    torch.testing.assert_close(calls["target"][2], target)
    torch.testing.assert_close(calls["target"][3], origins[env_ids])
    torch.testing.assert_close(calls["reset"][1], target)
    assert calls["reset"][3:] == (payload.reset_assets, True)


def test_state_command_deprecated_views_delegate_to_new_owners() -> None:
    """Deprecated command APIs should delegate without restoring materialization ownership."""
    term = object.__new__(StateCommand)
    term.cmd_indices = torch.zeros(2, dtype=torch.long)
    term.cfg = SimpleNamespace(states_relative=True)
    rates = torch.tensor([0.25, 0.75])
    spawn_cache = object()
    target_cache = object()
    sampler_term = SimpleNamespace(
        sample_indices=term.cmd_indices,
        success_rates=rates,
        value_shift=SimpleNamespace(observation_cache=spawn_cache),
    )
    goal_term = SimpleNamespace(observations=target_cache)
    terms = {"terrain_levels": sampler_term, "goal_observations": goal_term}
    manager = SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)
    term._env = SimpleNamespace(curriculum_manager=manager)

    with pytest.warns(DeprecationWarning, match="states_relative"):
        assert term.states_relative is True
    with pytest.warns(DeprecationWarning, match="success_rates"):
        assert term.success_rates is rates
    with pytest.warns(DeprecationWarning, match="get_spawn_obs_cache"):
        assert term.get_spawn_obs_cache() is spawn_cache
    with pytest.warns(DeprecationWarning, match="get_target_obs_cache"):
        assert term.get_target_obs_cache() is target_cache


def test_state_command_source_has_no_domain_or_learner_ownership() -> None:
    """The shared shell must not interpret rows, frames, resets, or learner caches."""
    source = inspect.getsource(StateCommand)

    for forbidden in (
        "set_reset_state",
        ".gather(",
        "env_origins",
        "_build_obs_cache",
        "observation_manager",
        "sim.forward",
        "scene.update",
    ):
        assert forbidden not in source


def test_factory_selected_rows_match_frozen_lifecycle_oracle(monkeypatch) -> None:
    """Factory binding preserves reset, target, command, error, timer, and success tensors."""
    from isaaclab_tasks.core.multi_task.factory.mdp import reset_state_command_payloads as factory_payloads

    class _Table:
        num_tasks = 2

        def __init__(self) -> None:
            self.spawn = torch.zeros(2, 13)
            self.spawn[:, 6] = 1.0
            self.target = self.spawn.clone()
            self.target[0, :3] = torch.tensor([4.0, 5.0, 6.0])
            self.target[1, :3] = torch.tensor([1.0, 2.0, 3.0])

        def gather(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.spawn[rows], self.target[rows]

    class _IdentitySymmetry:
        @staticmethod
        def reduce_orientation(held, target, _type_id, error, nearest) -> None:
            wp.to_torch(nearest).copy_(wp.to_torch(target))
            wp.to_torch(error).zero_()

    num_envs = 3
    origins = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins), extras={})
    table = _Table()
    payload = object.__new__(factory_payloads.FactoryAssemblyPayload)
    payload._env = env
    payload._device = torch.device("cpu")
    payload._states_relative = True
    payload.table = table
    payload.reset_assets = ["held_asset"]
    payload._held_asset_root_offset = 0
    payload.randomize_command_indices = False
    payload._command_names = ["precise", "loose"]
    payload.command_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    payload._command_masks = torch.ones(2, 2, dtype=torch.bool)
    payload._command_thresholds = torch.tensor([[0.05, 0.05], [0.10, 0.10]])
    payload._duration_ranges = torch.tensor([[0.2, 0.2], [0.3, 0.3]])
    payload.cmd_mask = torch.zeros(num_envs, 2, dtype=torch.bool)
    payload.command_thresholds = torch.full((num_envs, 2), 1.0)
    payload.orientation_aligned = torch.zeros(num_envs, dtype=torch.bool)
    payload.position_reached = torch.zeros(num_envs, dtype=torch.bool)
    payload.is_success = torch.zeros(num_envs, dtype=torch.bool)
    payload.duration_required = torch.zeros(num_envs)
    payload.duration_held = torch.zeros(num_envs)

    held_pos = torch.tensor([[11.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-0.8, 5.0, 6.0]])
    identity_quat = torch.zeros(num_envs, 4)
    identity_quat[:, 3] = 1.0
    payload.held_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=ProxyArray(wp.from_torch(held_pos, dtype=wp.vec3)),
            root_quat_w=ProxyArray(wp.from_torch(identity_quat.clone(), dtype=wp.quatf)),
        )
    )
    payload.robot = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=ProxyArray(wp.from_torch(identity_quat, dtype=wp.quatf)))
    )
    payload._symmetry = _IdentitySymmetry()
    payload._type_id = wp.zeros(num_envs, dtype=wp.int32, device="cpu")
    payload.target_pos = ProxyArray(wp.zeros(num_envs, dtype=wp.vec3, device="cpu"))
    target_quat = wp.zeros(num_envs, dtype=wp.quatf, device="cpu")
    target_quat.fill_(wp.quatf(0.0, 0.0, 0.0, 1.0))
    payload.target_quat = ProxyArray(target_quat)
    payload.orientation_error = ProxyArray(wp.zeros(num_envs, dtype=wp.float32, device="cpu"))
    payload.position_distance = ProxyArray(wp.zeros(num_envs, dtype=wp.float32, device="cpu"))
    payload._nearest_quat = ProxyArray(wp.zeros(num_envs, dtype=wp.quatf, device="cpu"))

    reset_calls: list[tuple[torch.Tensor, torch.Tensor, list[str], bool]] = []

    def set_reset_state(_env, states, env_ids, reset_assets, is_relative):
        reset_calls.append((states.clone(), env_ids.clone(), reset_assets, is_relative))

    monkeypatch.setattr(factory_payloads, "set_reset_state", set_reset_state)
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    task_rows = torch.tensor([1, 0], dtype=torch.long)
    payload.bind(env_ids, task_rows)

    expected_target = torch.tensor([[11.0, 2.0, 3.0], [-1.0, 5.0, 6.0]])
    torch.testing.assert_close(payload.target_pos.torch[env_ids], expected_target)
    torch.testing.assert_close(payload.target_quat.torch[env_ids], identity_quat[env_ids])
    torch.testing.assert_close(payload.command_thresholds[env_ids], torch.tensor([[0.05, 0.05], [0.10, 0.10]]))
    torch.testing.assert_close(payload.duration_required[env_ids], torch.tensor([0.2, 0.3]))
    torch.testing.assert_close(payload.duration_held[env_ids], torch.zeros(2))
    torch.testing.assert_close(reset_calls[0][0], table.spawn[task_rows])
    torch.testing.assert_close(reset_calls[0][1], env_ids)
    assert reset_calls[0][2:] == (payload.reset_assets, True)

    command = torch.empty(num_envs, payload.command_dim)
    error = torch.empty(num_envs, payload.error_dim)
    payload.update(0.1, command, error)

    torch.testing.assert_close(command[0], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(command[2], torch.tensor([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(error[env_ids], torch.tensor([[0.0, 0.0], [0.0, 0.2]]))
    torch.testing.assert_close(payload.duration_held[env_ids], torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(payload.is_success[env_ids], torch.tensor([False, False]))

    payload.update(0.1, command, error)

    torch.testing.assert_close(payload.duration_held[env_ids], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(payload.is_success[env_ids], torch.tensor([True, False]))
    assert env.extras["successes"] is payload.is_success
