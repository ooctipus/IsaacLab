# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RSL-RL same-step final-observation conversion."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


class _NestedFinalObservationEnv:
    """Minimal environment returning nested ordinary and final observations."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(is_finite_horizon=False)
        self.final_obs_valid = torch.tensor((False, True))
        self.truncated = torch.tensor((False, True))
        self.extras: dict[str, object] = {}

    @property
    def unwrapped(self) -> _NestedFinalObservationEnv:
        return self

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        del actions
        observations = {
            "policy": {"joint_position": torch.tensor(((1.0, 2.0), (3.0, 4.0)))},
            "transition": {"penalty": torch.tensor(((0.1,), (0.2,)))},
        }
        final_observations = {
            "policy": {"joint_position": torch.tensor(((5.0, 6.0), (7.0, 8.0)))},
            "transition": {"penalty": torch.tensor(((0.3,), (0.4,)))},
        }
        rewards = torch.tensor((0.0, 1.0))
        terminated = torch.zeros(2, dtype=torch.bool)
        self.extras = {"final_obs": final_observations, "final_obs_valid": self.final_obs_valid}
        return observations, rewards, terminated, self.truncated, self.extras


def test_step_converts_nested_final_observations_like_ordinary_observations() -> None:
    """Nested final observations should use the public RSL observation contract without changing masks."""
    env = _NestedFinalObservationEnv()
    wrapper = RslRlVecEnvWrapper.__new__(RslRlVecEnvWrapper)
    wrapper.env = env
    wrapper.clip_actions = None
    wrapper.num_envs = 2

    observations, _rewards, dones, extras = wrapper.step(torch.zeros(2, 1))

    final_observations = extras["final_obs"]
    assert isinstance(observations, TensorDict)
    assert isinstance(final_observations, TensorDict)
    assert observations.batch_size == final_observations.batch_size == torch.Size((2,))
    assert set(observations.keys(include_nested=True, leaves_only=True)) == set(
        final_observations.keys(include_nested=True, leaves_only=True)
    )
    expected_final_joint_position = torch.tensor(((5.0, 6.0), (7.0, 8.0)))
    torch.testing.assert_close(final_observations["policy", "joint_position"], expected_final_joint_position)
    assert extras["final_obs_valid"] is env.final_obs_valid
    assert extras["time_outs"] is env.truncated
    torch.testing.assert_close(dones, env.truncated.to(dtype=torch.long))
