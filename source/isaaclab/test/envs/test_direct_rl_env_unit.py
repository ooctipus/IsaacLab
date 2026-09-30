# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for direct reinforcement-learning environment lifecycle boundaries."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import isaaclab.envs.direct_marl_env as direct_marl_env_module
import isaaclab.envs.direct_rl_env as direct_rl_env_module
import isaaclab.envs.manager_based_env as manager_based_env_module
from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv

pytestmark = pytest.mark.unit

_ENV_MODULES = [
    (DirectRLEnv, direct_rl_env_module),
    (DirectMARLEnv, direct_marl_env_module),
    (ManagerBasedEnv, manager_based_env_module),
]


@pytest.mark.parametrize(
    ("env_type", "module"),
    _ENV_MODULES,
)
@pytest.mark.parametrize("seed_fails", [False, True])
def test_initial_seed_runs_after_simulation_context_stage_exists(
    env_type, module, seed_fails: bool, monkeypatch: pytest.MonkeyPatch
):
    """Replicator is seeded against the real SimulationContext stage, before scene creation."""
    events = []

    class _SimulationContext:
        _instance = None

        def __init__(self, _cfg):
            self.stage = object()
            type(self)._instance = self
            events.append("stage")

        @classmethod
        def instance(cls):
            return cls._instance

        def clear_instance(self):
            type(self)._instance = None

    class _Env(env_type):
        @staticmethod
        def seed(seed):
            assert _SimulationContext.instance().stage is not None
            events.append("seed")
            if seed_fails:
                raise RuntimeError("seed failed")
            return seed

        def _init_sim(self, *_args, **_kwargs):
            events.append("scene")

    monkeypatch.setattr(module, "SimulationContext", _SimulationContext)
    cfg = SimpleNamespace(validate=lambda: None, seed=42, sim=object())

    if seed_fails:
        with pytest.raises(RuntimeError, match="seed failed"):
            _Env(cfg)
        assert events == ["stage", "seed"]
        assert _SimulationContext.instance() is None
        return

    env = _Env(cfg)
    assert events == ["stage", "seed", "scene"]
    env._is_closed = True
    _SimulationContext._instance = None


@pytest.mark.parametrize(("env_type", "module"), _ENV_MODULES)
def test_random_seed_is_resolved_before_replicator(env_type, module, monkeypatch: pytest.MonkeyPatch):
    """Replicator receives the concrete seed used by every other random generator."""
    events = []
    core = ModuleType("omni.replicator.core")
    core.set_global_seed = lambda seed: events.append(("replicator", seed))
    replicator = ModuleType("omni.replicator")
    replicator.core = core
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)
    monkeypatch.setattr(module, "configure_seed", lambda seed: events.append(("core", seed)) or 2718)

    assert env_type.seed(-1) == 2718
    assert events == [("core", -1), ("replicator", 2718)]


def _make_step_env(events: list[object], reset_mask: torch.Tensor, manual_reset: bool) -> DirectRLEnv:
    env = object.__new__(DirectRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(dt=0.01, render_interval=2),
        action_noise_model=None,
        observation_noise_model=None,
        compute_final_obs=False,
        num_rerenders_on_reset=1,
        events=True,
    )
    env.scene = SimpleNamespace(
        num_envs=len(reset_mask),
        write_data_to_sim=lambda: events.append("write"),
        update=lambda dt: None,
    )
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=True,
        step=lambda render: None,
        forward=lambda: events.append("forward"),
        render=lambda **kwargs: events.append("render"),
        consume_reset_request=lambda: manual_reset,
    )
    env.event_manager = SimpleNamespace(available_modes=["interval"], apply=lambda **kwargs: events.append("interval"))
    env.video_recorders = [SimpleNamespace(step=lambda: events.append("video"))]
    env._physics_handles_decimation = True
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(len(reset_mask), dtype=torch.long)
    env.common_step_counter = 0
    env.reset_terminated = torch.zeros_like(reset_mask)
    env.reset_time_outs = torch.zeros_like(reset_mask)
    env.reset_buf = torch.zeros_like(reset_mask)
    env.render_enabled = True
    env.has_rtx_sensors = True
    env.extras = {}
    env._pre_physics_step = lambda action: None
    env._apply_action = lambda: None
    env._get_dones = lambda: events.append("dones") or (reset_mask, torch.zeros_like(reset_mask))
    env._get_rewards = lambda: events.append("rewards") or torch.zeros(len(reset_mask))
    env._refresh_task_state = lambda: events.append("refresh")
    env._reset_idx = lambda env_ids: events.append(("reset", tuple(env_ids.tolist())))
    env._get_observations = lambda: events.append("observation") or {"policy": torch.zeros((len(reset_mask), 1))}
    return env


def test_step_forwards_automatic_and_ui_resets_once_before_consumers():
    """Automatic and UI resets share one write-forward-render boundary."""
    events = []
    env = _make_step_env(events, torch.tensor([True, False]), manual_reset=True)

    env.step(torch.zeros((2, 1)))

    assert events.count("write") == 2
    assert events.count("forward") == 1
    assert events.count("refresh") == 2
    assert events.index("refresh") < events.index("dones") < events.index("rewards")
    assert events[-9:] == [
        ("reset", (0,)),
        ("reset", (1,)),
        "write",
        "forward",
        "refresh",
        "render",
        "interval",
        "video",
        "observation",
    ]


def test_step_refreshes_task_state_once_without_reset():
    """The steady-state path refreshes once before termination and reward consumers."""
    events = []
    env = _make_step_env(events, torch.tensor([False, False]), manual_reset=False)

    env.step(torch.zeros((2, 1)))

    assert events.count("refresh") == 1
    assert events.index("refresh") < events.index("dones") < events.index("rewards")


def test_step_conservatively_forwards_mask_native_reset_before_consumers():
    """A mask-native reset without host indices still establishes the reset boundary."""
    events = []
    env = _make_step_env(events, torch.tensor([False, False]), manual_reset=False)
    env._reset_envs_from_buffer = lambda: None

    env.step(torch.zeros((2, 1)))

    assert events.count("write") == 2
    assert events.count("forward") == 1
    assert events.count("refresh") == 2
    assert events[-7:] == ["write", "forward", "refresh", "render", "interval", "video", "observation"]


def test_reset_refreshes_task_state_after_forward_before_observations():
    """A public reset refreshes derived task state from reconciled physics."""
    events = []
    env = object.__new__(DirectRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(num_rerenders_on_reset=0)
    env.scene = SimpleNamespace(num_envs=2, write_data_to_sim=lambda: events.append("write"))
    env.sim = SimpleNamespace(device="cpu", forward=lambda: events.append("forward"))
    env.has_rtx_sensors = False
    env.extras = {}
    env._reset_idx = lambda env_ids: events.append(("reset", tuple(env_ids.tolist())))
    env._refresh_task_state = lambda: events.append("refresh")
    env._get_observations = lambda: events.append("observation") or {"policy": torch.zeros((2, 1))}

    env.reset()

    assert events == [("reset", (0, 1)), "write", "forward", "refresh", "observation"]
