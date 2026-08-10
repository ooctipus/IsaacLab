# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Factory containment of a Newton solver reset."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.envs import mdp as newton_mdp
from isaaclab_newton.envs.mdp import rewards as newton_rewards

from isaaclab_tasks.contrib.nist import mdp
from isaaclab_tasks.contrib.nist.factory_env_cfg import FactoryRewardsCfg, FactoryTerminationsCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_zero_reward_on_solver_reset_overrides_non_finite_total(monkeypatch: pytest.MonkeyPatch):
    """The final reward term should overwrite prior non-finite reward values."""
    reset_required = torch.tensor([False, True, True])
    monkeypatch.setattr(newton_rewards, "solver_reset_required", lambda env: reset_required)
    reward_buf = torch.tensor([2.0, torch.nan, torch.inf])
    env = SimpleNamespace(num_envs=3, reward_manager=SimpleNamespace(_reward_buf=reward_buf))

    term_reward = newton_mdp.zero_reward_on_solver_reset(env)

    torch.testing.assert_close(reward_buf, torch.tensor([2.0, 0.0, 0.0]))
    torch.testing.assert_close(term_reward, torch.zeros(3))


def test_factory_cfg_wires_newton_solver_reset_and_reward_override():
    """Factory should consume Newton's reset mask and override the terminal reward."""
    terminations = resolve_presets(FactoryTerminationsCfg(), ("newton_mjwarp",))
    rewards = resolve_presets(FactoryRewardsCfg(), ("newton_mjwarp",))
    physx_terminations = resolve_presets(FactoryTerminationsCfg())
    physx_rewards = resolve_presets(FactoryRewardsCfg())

    assert terminations.solver_reset_required.func is newton_mdp.solver_reset_required
    assert physx_terminations.solver_reset_required is None
    assert not any(
        hasattr(terminations, name) for name in ("non_finite_robot", "non_finite_held_asset", "non_finite_fixed_asset")
    )
    assert rewards.joint_effort.func is mdp.joint_torques_l2
    assert rewards.early_termination.params["term_keys"] == "abnormal"
    assert rewards.solver_reset_reward.func is newton_mdp.zero_reward_on_solver_reset
    assert rewards.solver_reset_reward.weight == 1.0
    assert list(vars(rewards))[-1] == "solver_reset_reward"
    assert physx_rewards.solver_reset_reward is None
