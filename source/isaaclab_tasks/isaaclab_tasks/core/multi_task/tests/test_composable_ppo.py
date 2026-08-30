# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")

import torch.nn as nn  # noqa: E402
from rsl_rl.algorithms.ppo import PPO  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from rsl_rl.storage import RolloutStorage  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms.composable_ppo import (  # noqa: E402
    ComposablePPO,
    DiscountedReturnScaler,
    project_categorical_targets,
)
from isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms.value_shift_ppo import ValueShiftPPO  # noqa: E402
from isaaclab_tasks.core.multi_task.rl.rsl_rl.models.categorical_value import (  # noqa: E402
    CategoricalResidualMLPEncoderModel,
)


class _ProjectableModel(nn.Module):
    is_recurrent = False
    hyperspherical_backbone = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0, 4.0]))

    def forward(self, value):
        return value * self.weight[0]

    @torch.no_grad()
    def project_hyperspherical_weights_(self) -> None:
        self.weight.div_(torch.linalg.vector_norm(self.weight))


def test_two_hot_projection_preserves_expectation_and_clamps_support():
    support = torch.linspace(-2.0, 2.0, 5)
    returns = torch.tensor([[-3.0], [-0.5], [0.0], [1.25], [3.0]])

    probabilities = project_categorical_targets(returns, support)

    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(5))
    expected = torch.sum(probabilities * support, dim=-1)
    torch.testing.assert_close(expected, returns.squeeze(-1).clamp(-2.0, 2.0))
    assert torch.count_nonzero(probabilities[1]) == 2


def test_discounted_return_scaler_resets_before_terminal_reward_and_round_trips_state():
    scaler = DiscountedReturnScaler(3, gamma=0.9, max_normalized_return=5.0, epsilon=1.0e-8, device="cpu")
    scaled = scaler.update_and_scale(
        torch.tensor([1.0, 2.0, -1.0]),
        torch.zeros(3, dtype=torch.bool),
    )
    scaler.update_and_scale(
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([False, True, False]),
    )

    assert torch.isfinite(scaled).all()
    torch.testing.assert_close(scaler.discounted_returns, torch.tensor([1.4, 0.5, -0.4]))
    assert scaler.count > 6.0

    restored = DiscountedReturnScaler(3, gamma=0.9, max_normalized_return=5.0, epsilon=1.0e-8, device="cpu")
    restored.load_state_dict(scaler.state_dict())
    torch.testing.assert_close(restored.discounted_returns, scaler.discounted_returns)
    torch.testing.assert_close(restored.variance, scaler.variance)
    torch.testing.assert_close(restored.max_abs_return, scaler.max_abs_return)


def test_optimizer_hook_projects_hyperspherical_models_after_every_step():
    actor = _ProjectableModel()
    critic = _ProjectableModel()
    storage = type("Storage", (), {"num_envs": 2})()
    algorithm = ComposablePPO(actor, critic, storage, optimizer="adam", learning_rate=0.1)

    loss = actor.weight.sum() + critic.weight.sum()
    algorithm.optimizer.zero_grad()
    loss.backward()
    algorithm.optimizer.step()

    torch.testing.assert_close(torch.linalg.vector_norm(actor.weight), torch.tensor(1.0))
    torch.testing.assert_close(torch.linalg.vector_norm(critic.weight), torch.tensor(1.0))


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_simba_v2_rejects_non_adam_optimizers(optimizer):
    actor = _ProjectableModel()
    critic = _ProjectableModel()
    storage = type("Storage", (), {"num_envs": 2})()
    with pytest.raises(ValueError, match="requires .*Adam"):
        ComposablePPO(actor, critic, storage, optimizer=optimizer)


def test_checkpoint_load_revalidates_weight_decay():
    actor = _ProjectableModel()
    critic = _ProjectableModel()
    storage = type("Storage", (), {"num_envs": 2})()
    algorithm = ComposablePPO(actor, critic, storage, optimizer="adam")
    checkpoint = algorithm.save()
    checkpoint["optimizer_state_dict"]["param_groups"][0]["weight_decay"] = 0.1

    with pytest.raises(ValueError, match="zero optimizer weight decay"):
        algorithm.load(
            checkpoint,
            load_cfg={"actor": False, "critic": False, "optimizer": True, "iteration": False},
            strict=True,
        )


def test_scalar_path_delegates_to_upstream_ppo(monkeypatch: pytest.MonkeyPatch):
    class _ScalarModel(nn.Module):
        is_recurrent = False

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    storage = type("Storage", (), {"num_envs": 2})()
    algorithm = ComposablePPO(_ScalarModel(), _ScalarModel(), storage)
    monkeypatch.setattr(PPO, "update", lambda self: {"upstream": 1.0})

    assert algorithm.update() == {"upstream": 1.0}


def test_categorical_ppo_runs_one_update_and_saves_reward_scaler():
    num_envs = 4
    num_steps = 3
    obs_dim = 6
    action_dim = 2
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    def make_obs() -> TensorDict:
        return TensorDict({"policy": torch.randn(num_envs, obs_dim)}, batch_size=[num_envs])

    initial_obs = make_obs()
    actor = MLPModel(
        initial_obs,
        obs_groups,
        "actor",
        action_dim,
        hidden_dims=[16, 16],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "log"},
    )
    critic = CategoricalResidualMLPEncoderModel(
        initial_obs,
        obs_groups,
        "critic",
        output_dim=1,
        hidden_dim=16,
        num_blocks=1,
        expand=2,
        activation="relu",
        norm=True,
        encoder_cfg={},
        num_bins=11,
        value_min=-2.0,
        value_max=2.0,
    )
    storage = RolloutStorage("rl", num_envs, num_steps, initial_obs, [action_dim], "cpu")
    algorithm = ValueShiftPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        schedule="fixed",
        desired_kl=None,
    )
    algorithm._obs_cache = initial_obs[:2]
    algorithm._cur_buf = torch.zeros(2)
    algorithm._diff_buf = torch.zeros(2)

    obs = initial_obs
    for _ in range(num_steps):
        algorithm.act(obs)
        next_obs = make_obs()
        algorithm.process_env_step(
            next_obs,
            torch.randn(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )
        obs = next_obs
    algorithm.compute_returns(obs)
    losses = algorithm.update()

    assert all(math.isfinite(value) for value in losses.values())
    assert "reward_scale" in losses
    assert torch.isfinite(algorithm._diff_buf).all()
    assert torch.count_nonzero(algorithm._diff_buf) > 0
    assert algorithm.reward_scaler is not None
    checkpoint = algorithm.save()
    scaler_state = checkpoint["reward_scaler_state_dict"]
    algorithm.reward_scaler.mean.zero_()
    algorithm.load(checkpoint, load_cfg=None, strict=True)
    torch.testing.assert_close(algorithm.reward_scaler.mean, scaler_state["mean"])
