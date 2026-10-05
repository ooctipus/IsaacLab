# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL's structural environment boundary without launching a simulator."""

import builtins
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlEnv, RslRlVecEnvWrapper

pytestmark = pytest.mark.unit


class TensorGymEnv(gym.Env):
    """A vector Gym environment with tensor groups and same-step autoreset."""

    def __init__(self, *, finite_horizon=False):
        self.num_envs, self.device, self.max_episode_length = 2, "cpu", 3
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.cfg = SimpleNamespace(is_finite_horizon=finite_horizon)
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.observation_space = gym.spaces.Dict(policy=gym.spaces.Box(-np.inf, np.inf, shape=(2, 3)))
        self.render_mode = None
        self.obs_buf = {"policy": torch.zeros(2, 3)}
        self.reset_count, self.closed = 0, False

    def reset(self, **kwargs):
        self.reset_count += 1
        self.episode_length_buf.zero_()
        self.obs_buf = {"policy": torch.full((2, 3), float(self.reset_count))}
        return self.obs_buf, {"reset_count": self.reset_count}

    def step(self, actions):
        self.last_actions = actions.clone()
        self.episode_length_buf.zero_()
        self.obs_buf = {"policy": self.obs_buf["policy"] + 1}
        return (
            self.obs_buf,
            actions.sum(dim=1),
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            {"log": {"sample": 1.0}},
        )

    def seed(self, seed=-1):
        self.last_seed = seed
        return seed

    def close(self):
        self.closed = True


@pytest.mark.parametrize("finite_horizon", [False, True])
def test_structural_gym_environment_reset_step_and_clipping(finite_horizon):
    """Keep tensor groups, autoreset flags, clipping and timeout bootstrapping intact."""
    raw = TensorGymEnv(finite_horizon=finite_horizon)
    assert isinstance(raw, RslRlEnv)
    wrapped = RslRlVecEnvWrapper(gym.Wrapper(raw), clip_actions=0.5)
    assert wrapped.unwrapped is raw and raw.reset_count == 1
    assert wrapped.num_actions == 2
    np.testing.assert_array_equal(raw.single_action_space.low, [-0.5, -0.5])
    np.testing.assert_array_equal(raw.action_space.high, np.full((2, 2), 0.5))
    obs, extras = wrapped.reset()
    assert isinstance(obs, TensorDict) and obs.batch_size == torch.Size([2])
    assert extras == {"reset_count": 2}
    torch.testing.assert_close(wrapped.get_observations()["policy"], raw.obs_buf["policy"])
    lengths = torch.tensor([1, 2])
    wrapped.episode_length_buf = lengths
    assert raw.episode_length_buf is lengths
    actions = torch.tensor([[2.0, -2.0], [0.1, 0.2]])
    obs, rewards, dones, extras = wrapped.step(actions)
    torch.testing.assert_close(raw.last_actions, actions.clamp(-0.5, 0.5))
    torch.testing.assert_close(actions, torch.tensor([[2.0, -2.0], [0.1, 0.2]]))
    torch.testing.assert_close(rewards, raw.last_actions.sum(dim=1))
    torch.testing.assert_close(dones, torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(obs["policy"], wrapped.get_observations()["policy"])
    assert ("time_outs" in extras) is not finite_horizon
    if not finite_horizon:
        torch.testing.assert_close(extras["time_outs"], torch.tensor([False, True]))
    assert wrapped.seed(42) == raw.last_seed == 42
    assert not hasattr(raw, "scene") and not hasattr(raw, "action_manager")
    wrapped.close()
    assert raw.closed


def test_structural_environment_does_not_load_nominal_simulation_classes(monkeypatch):
    """Do not require simulation-class imports to recognize a complete Gym contract."""
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "isaaclab.envs" or name.startswith("isaaclab.envs."):
            raise AssertionError("Structural wrapping loaded concrete environment classes.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    RslRlVecEnvWrapper(TensorGymEnv()).close()


def test_structural_contract_rejects_non_gym_and_incomplete_environments():
    """Require a real Gym environment and every structural field, rather than accepting a facade."""
    raw = TensorGymEnv()
    duck = SimpleNamespace(**vars(raw), reset=raw.reset, step=raw.step, seed=raw.seed, close=raw.close)
    duck.unwrapped = duck
    assert isinstance(duck, RslRlEnv)
    with pytest.raises(ValueError, match="SimpleNamespace"):
        RslRlVecEnvWrapper(duck)
    del raw.obs_buf
    with pytest.raises(ValueError, match="TensorGymEnv"):
        RslRlVecEnvWrapper(raw)
