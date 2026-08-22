# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

pytestmark = pytest.mark.unit


def _make_env_with_policy_obs_terms(
    num_envs: int,
    terms: list[tuple[str, tuple[int, ...], tuple[float, float] | None]],
) -> ManagerBasedRLEnv:
    """Build an uninitialized env whose observation manager stubs drive space setup.

    Args:
        num_envs: Number of vectorized environments.
        terms: Non-concatenated policy observation terms as ``(name, shape, clip)``
            where ``clip`` is ``(low, high)`` or ``None``.

    Returns:
        An uninitialized :class:`~isaaclab.envs.ManagerBasedRLEnv` ready for
        :meth:`~isaaclab.envs.ManagerBasedRLEnv._configure_gym_env_spaces`.
    """
    term_names = [name for name, _, _ in terms]
    term_dims = [shape for _, shape, _ in terms]
    term_cfgs = [SimpleNamespace(clip=clip) for _, _, clip in terms]

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.observation_manager = SimpleNamespace(
        active_terms={"policy": term_names},
        group_obs_concatenate={"policy": False},
        group_obs_dim={"policy": term_dims},
        _group_obs_term_cfgs={"policy": term_cfgs},
    )
    env.action_manager = SimpleNamespace(action_term_dim=[0])
    return env


def _make_step_env(reset_mask: torch.Tensor, manual_reset: bool = False) -> tuple[ManagerBasedRLEnv, list[str]]:
    """Build an uninitialized environment with a recorded step lifecycle."""
    events: list[str] = []
    num_envs = len(reset_mask)
    zeros = torch.zeros(num_envs, dtype=torch.bool)

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        decimation=1,
        sim=SimpleNamespace(dt=0.01, render_interval=1),
        compute_final_obs=False,
        num_rerenders_on_reset=0,
    )
    env.scene = SimpleNamespace(
        num_envs=num_envs,
        write_data_to_sim=lambda: None,
        update=lambda dt: None,
    )
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        step=lambda render: None,
        forward=lambda: events.append("forward"),
        consume_reset_request=lambda: manual_reset,
    )
    env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
    env.observation_manager = SimpleNamespace(compute=lambda **kwargs: events.append("observation") or {})
    env.termination_manager = SimpleNamespace(compute=lambda: reset_mask, terminated=reset_mask, time_outs=zeros)
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(num_envs))
    env.command_manager = SimpleNamespace(compute=lambda dt: None)
    env.event_manager = SimpleNamespace(available_modes=[])
    env.recorder_manager = SimpleNamespace(
        active_terms=[],
        record_pre_step=lambda: None,
        record_post_physics_decimation_step=lambda: None,
        record_pre_reset=lambda env_ids: events.append("pre_reset"),
        record_post_reset=lambda env_ids: events.append("post_reset"),
    )
    env.video_recorders = []
    env._render_video_recorders = []
    env._physics_handles_decimation = True
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    env.common_step_counter = 0
    env.render_enabled = False
    env.has_rtx_sensors = False
    env.extras = {}
    env._reset_idx = lambda env_ids: events.append("reset")
    return env, events


def test_non_concatenated_obs_groups_contain_all_terms():
    """Non-concatenated observation groups expose every term in the Dict space (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    terms = [
        ("image", (4, 5, 3), (0.0, 1.0)),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("vector", (3,), None),
    ]
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict)

    expected_policy_terms = ["image", "matrix", "vector"]
    assert list(policy_space.spaces) == expected_policy_terms
    for term_name in expected_policy_terms:
        assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)


def test_obs_space_follows_clip_constraint():
    """Observation space bounds reflect the clip constraint on each non-concatenated term."""
    terms = [
        ("vector", (3,), None),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("image", (4, 5, 3), (0.0, 1.0)),
    ]
    expected_shapes = {
        "vector": (2, 3),
        "matrix": (2, 2, 3),
        "image": (2, 4, 5, 3),
    }
    term_clips = {name: clip for name, _, clip in terms}
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    for group_name, group_space in env.observation_space.spaces.items():
        assert isinstance(group_space, gym.spaces.Dict)
        for term_name, term_space in group_space.spaces.items():
            clip = term_clips[term_name]
            low = -np.inf if clip is None else clip[0]
            high = np.inf if clip is None else clip[1]
            assert isinstance(term_space, gym.spaces.Box), (
                f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            )
            assert term_space.shape == expected_shapes[term_name]
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)


@pytest.mark.parametrize(
    ("reset_mask", "manual_reset", "expected_rate", "expected_forward_calls"),
    [
        (torch.tensor([False, False]), False, 0.0, 0),
        (torch.tensor([True, False]), False, 0.5, 1),
        (torch.tensor([True, False]), True, 1.0, 1),
    ],
)
def test_step_reconciles_reset_state_before_post_reset(
    reset_mask: torch.Tensor, manual_reset: bool, expected_rate: float, expected_forward_calls: int
):
    """Finalize all same-step resets once before post-reset consumers run."""
    env, events = _make_step_env(reset_mask, manual_reset)

    env.step(torch.empty((len(reset_mask), 0)))

    assert events.count("forward") == expected_forward_calls
    if expected_forward_calls:
        assert (
            events.index("reset") < events.index("forward") < events.index("post_reset") < events.index("observation")
        )
    assert env.extras["info"]["reset_rate"] == expected_rate
    assert env.extras["log"]["Info/ResetRate"] == expected_rate
