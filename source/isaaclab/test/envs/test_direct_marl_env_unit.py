# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Unit tests for direct multi-agent reinforcement-learning environments."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.test.env_cfgs import make_empty_direct_marl_env_cfg

pytestmark = pytest.mark.unit


class _StubMARLEnv(DirectMARLEnv):
    """Direct MARL environment stub that skips simulator initialization."""

    def __init__(self, cfg: DirectMARLEnvCfg) -> None:
        self._is_closed = True
        self.cfg = cfg
        self.scene = SimpleNamespace(num_envs=cfg.scene.num_envs)
        self.sim = SimpleNamespace(device=cfg.sim.device)


def test_agent_and_space_configuration():
    """Agent counts and spaces are configured without initializing the simulator."""
    env = _StubMARLEnv(make_empty_direct_marl_env_cfg(device="cpu"))

    env._configure_env_spaces()

    assert env.agents == ["agent_0", "agent_1"]
    assert env.possible_agents == ["agent_0", "agent_1"]
    assert env.num_agents == 2
    assert env.max_num_agents == 2
    assert len(env.observation_spaces) == 2
    assert len(env.action_spaces) == 2
    assert all(isinstance(space, gym.spaces.Box) for space in env.observation_spaces.values())
    assert all(isinstance(space, gym.spaces.Box) for space in env.action_spaces.values())
    assert env.observation_spaces["agent_0"].shape == (3,)
    assert env.observation_spaces["agent_1"].shape == (4,)
    assert env.action_spaces["agent_0"].shape == (1,)
    assert env.action_spaces["agent_1"].shape == (2,)
    assert isinstance(env.state_space, gym.spaces.Box)
    assert env.state_space.shape == (7,)


class _DebugVisStubMARLEnv(_StubMARLEnv):
    """Stub whose debug visualization is implemented, so ``set_debug_vis`` runs its handle logic."""

    def __init__(self, cfg: DirectMARLEnvCfg) -> None:
        super().__init__(cfg)
        # mirrors what DirectMARLEnv.__init__ derives, which the stub skips
        self.has_debug_vis_implementation = "NotImplementedError" not in inspect.getsource(self._set_debug_vis_impl)
        self._debug_vis_handle = None
        self.sim = SimpleNamespace(device=cfg.sim.device, vis_marker_registry=VisMarkerRegistry())
        self.callback_count = 0

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        self.callback_count += 1


def _make_step_env(events: list[object], reset_mask: torch.Tensor, manual_reset: bool) -> DirectMARLEnv:
    zeros = torch.zeros_like(reset_mask)
    env = object.__new__(DirectMARLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(dt=0.01, render_interval=2),
        action_noise_model=None,
        observation_noise_model=None,
        compute_final_obs=False,
        events=True,
    )
    env.scene = SimpleNamespace(
        num_envs=len(reset_mask),
        write_data_to_sim=lambda: events.append("write"),
        update=lambda dt: None,
    )
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        step=lambda render: None,
        forward=lambda: events.append("forward"),
        consume_reset_request=lambda: manual_reset,
    )
    env.event_manager = SimpleNamespace(available_modes=["interval"], apply=lambda **kwargs: events.append("interval"))
    env.video_recorders = [SimpleNamespace(step=lambda: events.append("video"))]
    env._physics_handles_decimation = True
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(len(reset_mask), dtype=torch.long)
    env.common_step_counter = 0
    env.reset_buf = zeros.clone()
    env.render_enabled = True
    env.extras = {"agent": {}}
    env.possible_agents = ["agent"]
    env._pre_physics_step = lambda actions: None
    env._apply_action = lambda: None
    env._get_dones = lambda: events.append("dones") or ({"agent": reset_mask}, {"agent": zeros})
    env._get_rewards = lambda: events.append("rewards") or {"agent": torch.zeros(len(reset_mask))}
    env._refresh_task_state = lambda: events.append("refresh")
    env._reset_idx = lambda env_ids: events.append(("reset", tuple(env_ids.tolist())))
    env._get_observations = lambda: events.append("observation") or {"agent": torch.zeros((len(reset_mask), 1))}
    return env


def test_set_debug_vis_registers_without_kit():
    """Debug visualization registers through the marker registry, so it needs no Kit application.

    Guards against reintroducing the deprecated ``IApp.get_post_update_event_stream`` subscription,
    which raised ``NameError`` in kitless mode because ``omni.kit.app`` is only imported when Kit is
    present.
    """
    env = _DebugVisStubMARLEnv(make_empty_direct_marl_env_cfg(device="cpu"))
    registry = env.sim.vis_marker_registry

    assert env.set_debug_vis(True) is True
    assert isinstance(env._debug_vis_handle, str)

    registry.dispatch_callbacks()
    assert env.callback_count == 1

    env.set_debug_vis(False)
    assert env._debug_vis_handle is None

    registry.dispatch_callbacks()
    assert env.callback_count == 1


def test_reset_forwards_scene_writes_before_observations():
    """A public reset exposes reset state only after the simulator boundary."""
    events = []
    env = object.__new__(DirectMARLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=2, write_data_to_sim=lambda: events.append("write"))
    env.sim = SimpleNamespace(device="cpu", forward=lambda: events.append("forward"))
    env.possible_agents = ["agent"]
    env.extras = {"agent": {}}
    env._reset_idx = lambda env_ids: events.append(("reset", tuple(env_ids.tolist())))
    env._refresh_task_state = lambda: events.append("refresh")
    env._get_observations = lambda: events.append("observation") or {"agent": torch.zeros((2, 1))}

    env.reset()

    assert events == [("reset", (0, 1)), "write", "forward", "refresh", "observation"]


def test_step_forwards_automatic_and_ui_resets_once_before_consumers():
    """Automatic and UI resets share one simulator boundary before post-reset consumers."""
    events = []
    reset_mask = torch.tensor([True, False])
    env = _make_step_env(events, reset_mask, manual_reset=True)

    env.step({"agent": torch.zeros((2, 1))})

    assert events.count("write") == 2
    assert events.count("forward") == 1
    assert events.count("refresh") == 2
    assert events.index("refresh") < events.index("dones") < events.index("rewards")
    assert events[-8:] == [
        ("reset", (0,)),
        ("reset", (1,)),
        "write",
        "forward",
        "refresh",
        "interval",
        "video",
        "observation",
    ]


def test_step_refreshes_task_state_once_without_reset():
    """The steady-state path refreshes once before termination and reward consumers."""
    events = []
    env = _make_step_env(events, torch.tensor([False, False]), manual_reset=False)

    env.step({"agent": torch.zeros((2, 1))})

    assert events.count("refresh") == 1
    assert events.index("refresh") < events.index("dones") < events.index("rewards")
