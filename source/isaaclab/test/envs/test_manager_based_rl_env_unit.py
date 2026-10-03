# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import Mock, call

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("ending", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_final_observation_hook_runs_only_for_enabled_automatic_resets(ending, enabled, override):
    """Keep terminal observation timing with the environment and preview ownership with the task."""
    done = torch.tensor([ending, False])
    ordinary, successor = {"policy": torch.zeros(2, 1)}, {"policy": torch.ones(2, 1)}
    env = SimpleNamespace(
        cfg=SimpleNamespace(decimation=1, sim=SimpleNamespace(render_interval=1), compute_final_obs=enabled),
        device="cpu",
        num_envs=2,
        physics_dt=0.01,
        step_dt=0.01,
        render_enabled=False,
        video_recorders=[],
        _physics_handles_decimation=True,
        _sim_step_counter=0,
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        extras={},
        action_manager=Mock(),
        scene=Mock(),
        recorder_manager=Mock(active_terms=[]),
        sim=Mock(is_rendering=False, consume_reset_request=Mock(return_value=False)),
        termination_manager=Mock(compute=Mock(return_value=done), terminated=done, time_outs=torch.zeros_like(done)),
        reward_manager=Mock(compute=Mock(return_value=torch.zeros(2))),
        observation_manager=Mock(compute=Mock(return_value=ordinary), preview=Mock(return_value=successor)),
        command_manager=Mock(),
        event_manager=Mock(available_modes=[]),
        _reset_idx=Mock(),
    )
    operation = (
        env.observation_manager.preview if override else MethodType(ManagerBasedRLEnv._compute_final_observations, env)
    )
    env._compute_final_observations = Mock(side_effect=operation)
    ManagerBasedRLEnv.step(env, torch.zeros(2, 1))
    expected = int(ending and enabled)
    assert env._compute_final_observations.call_count == expected
    assert env.observation_manager.preview.call_count == expected * int(override)
    assert env.observation_manager.compute.call_args_list == (
        ([call()] if expected and not override else []) + [call(update_history=True)]
    )
    if expected:
        assert env.extras["final_obs"] is (successor if override else ordinary)
    else:
        assert "final_obs" not in env.extras


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
