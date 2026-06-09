# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the successor-latent PPO auxiliary value objective."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "dep" / "rsl_rl"))
sys.path.insert(0, str(_REPO_ROOT / "source" / "isaaclab_tasks"))

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms import SuccessorLatentPPO


def _make_obs(values: torch.Tensor) -> TensorDict:
    return TensorDict({"policy": values}, batch_size=[values.shape[0]], device=values.device)


def _make_algorithm(
    num_envs: int = 2,
    num_steps: int = 3,
    obs_dim: int = 2,
    num_actions: int = 1,
    gamma: float = 0.5,
    lam: float = 1.0,
) -> SuccessorLatentPPO:
    obs = _make_obs(torch.zeros(num_envs, obs_dim))
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        num_actions,
        hidden_dims=[8],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[8], activation="elu")
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], device="cpu")
    return SuccessorLatentPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        gamma=gamma,
        lam=lam,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=None,
        device="cpu",
        successor_loss_coef=0.01,
    )


def _collect_rollout(alg: SuccessorLatentPPO, obs_values: torch.Tensor, dones: torch.Tensor) -> None:
    for step in range(alg.storage.num_transitions_per_env):
        alg.act(_make_obs(obs_values[step]))
        alg.process_env_step(
            _make_obs(obs_values[step + 1]),
            rewards=torch.zeros(alg.storage.num_envs),
            dones=dones[step],
            extras={},
        )


def test_successor_returns_follow_value_lambda_return_indexing() -> None:
    """Vector targets should mirror scalar value lambda-return recursion."""
    alg = _make_algorithm()
    with torch.no_grad():
        alg.successor_head.weight.zero_()
        alg.successor_head.bias.zero_()

    obs_values = torch.tensor(
        [
            [[1.0, 0.0], [10.0, 0.0]],
            [[2.0, 0.0], [20.0, 0.0]],
            [[4.0, 0.0], [40.0, 0.0]],
            [[8.0, 0.0], [80.0, 0.0]],
        ]
    )
    dones = torch.tensor(
        [
            [False, False],
            [False, True],
            [False, False],
        ]
    )

    _collect_rollout(alg, obs_values, dones)
    alg.compute_returns(_make_obs(obs_values[-1]))

    expected = torch.tensor(
        [
            [[3.0, 0.0], [20.0, 0.0]],
            [[4.0, 0.0], [20.0, 0.0]],
            [[4.0, 0.0], [40.0, 0.0]],
        ]
    )
    torch.testing.assert_close(alg._successor_returns, expected)


def test_successor_latent_update_smoke_returns_metrics() -> None:
    """A PPO update should include successor-latent metrics and clear storage."""
    alg = _make_algorithm()
    obs_values = torch.randn(alg.storage.num_transitions_per_env + 1, alg.storage.num_envs, 2)
    dones = torch.zeros(alg.storage.num_transitions_per_env, alg.storage.num_envs, dtype=torch.bool)

    _collect_rollout(alg, obs_values, dones)
    alg.compute_returns(_make_obs(obs_values[-1]))
    losses = alg.update()

    assert "successor_latent" in losses
    assert "successor_latent_target_norm" in losses
    assert "successor_latent_prediction_norm" in losses
    assert alg.storage.step == 0
    assert alg._successor_returns is None


def test_factory_successor_latent_algorithm_preset_resolves() -> None:
    """Factory config should expose successor-latent PPO as an opt-in preset."""
    from isaaclab_tasks.core.multi_task.factory.config.agents.rsl_rl_ppo_cfg import (
        FactoryPPORunnerCfg,
        SuccessorLatentAlgorithmCfg,
    )
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = FactoryPPORunnerCfg()

    resolve_presets(cfg, selected=("successor_latent",))

    assert isinstance(cfg.algorithm, SuccessorLatentAlgorithmCfg)
    assert cfg.algorithm.successor_loss_coef == 0.01
