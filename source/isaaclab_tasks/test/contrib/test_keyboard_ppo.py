# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PPO storage and returns across same-step keyboard autoresets."""

import ast
import inspect
import textwrap

import pytest
import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.contrib.keyboard.agents.models import SharedEncoderMLPModel, SharedEncoderPPO


def _observations(values):
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
    return TensorDict({"value": values, "policy": torch.zeros_like(values)}, batch_size=[len(values)])


def _algorithm(observations, device="cpu"):
    groups = {"actor": ["value", "policy"], "critic": ["value", "policy"]}
    config = {"hidden_dims": [1], "encoder_cfg": {"policy": {"hidden_dims": [1], "latent_dim": 1}}}
    actor = SharedEncoderMLPModel(
        observations, groups, "actor", 1, distribution_cfg={"class_name": "GaussianDistribution"}, **config
    )
    critic = SharedEncoderMLPModel(observations, groups, "critic", 1, **config)
    actor, critic = actor.to(device), critic.to(device)
    storage = RolloutStorage("rl", len(observations), 1, observations, [1], device=device)
    algorithm = SharedEncoderPPO(actor, critic, storage, gamma=0.9, device=device)
    assert not actor.is_recurrent and not critic.is_recurrent
    with torch.no_grad():
        critic.mlp[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.mlp[0].bias.zero_()
        critic.mlp[2].weight.fill_(1.0)
        critic.mlp[2].bias.zero_()
    return algorithm


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("final_as_dict", [False, True])
def test_final_observation_bootstrap_preserves_real_ppo_storage_and_returns(final_as_dict, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    initial = _observations([2, 3, 4, 5]).to(device)
    final = _observations([20, 30, 40, 50])
    # Timeout, true termination, continuing episode, and termination coinciding
    # with an administrative boundary. The environment excludes real terminals
    # from its effective timeout mask. Only reset rows have post-reset values.
    post_reset = _observations([200, 300, 40, 500]).to(device)
    dones = torch.tensor([1, 1, 0, 1], device=device)
    timeouts = torch.tensor([True, False, False, False])
    extras = {"time_outs": timeouts, "final_obs": final.to_dict() if final_as_dict else final, "log": {"x": 1}}
    reward = torch.ones(4, device=device)
    algorithm = _algorithm(initial, device)
    with torch.inference_mode():
        algorithm.act(initial)
        algorithm.process_env_step(post_reset, reward, dones, extras)
        algorithm.compute_returns(post_reset)

    torch.testing.assert_close(algorithm.storage.values[0, :, 0].cpu(), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0].cpu(), torch.tensor([19.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(algorithm.storage.returns[0, :, 0].cpu(), torch.tensor([19.0, 1.0, 37.0, 1.0]))
    torch.testing.assert_close(reward.cpu(), torch.ones(4))
    assert extras["time_outs"] is timeouts and "final_obs" in extras and extras["log"] == {"x": 1}


@pytest.mark.parametrize("with_timeouts,with_final", [(True, False), (False, True), (False, False)])
def test_missing_final_observation_or_timeout_preserves_base_ppo_contract(with_timeouts, with_final):
    initial, post_reset = _observations([2, 3]), _observations([200, 300])
    algorithm = _algorithm(initial)
    extras = {}
    if with_timeouts:
        extras["time_outs"] = torch.tensor([True, False])
    if with_final:
        extras["final_obs"] = _observations([20, 30]).to_dict()
    with torch.inference_mode():
        algorithm.act(initial)
        algorithm.process_env_step(post_reset, torch.ones(2), torch.ones(2), extras)
        algorithm.compute_returns(post_reset)
    expected = torch.tensor([2.8 if with_timeouts else 1.0, 1.0])
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected)
    torch.testing.assert_close(algorithm.storage.returns[0, :, 0], expected)


@pytest.mark.parametrize("timeout_value", [20.0, float("nan")])
def test_nonfinite_final_values_only_affect_actual_timeouts(timeout_value):
    initial, post_reset = _observations([2, 3]), _observations([200, 300])
    algorithm = _algorithm(initial)
    extras = {
        "time_outs": torch.tensor([True, False]),
        "final_obs": _observations([timeout_value, float("nan")]),
    }
    with torch.inference_mode():
        algorithm.act(initial)
        algorithm.process_env_step(post_reset, torch.ones(2), torch.ones(2), extras)
        algorithm.compute_returns(post_reset)
    expected = torch.tensor([1.0 + 0.9 * timeout_value, 1.0])
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected, equal_nan=True)
    torch.testing.assert_close(algorithm.storage.returns[0, :, 0], expected, equal_nan=True)


def test_bootstrap_extension_leaves_rollout_ownership_in_base_ppo():
    tree = ast.parse(textwrap.dedent(inspect.getsource(SharedEncoderPPO.process_env_step)))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"transition", "storage", "add_transition", "update_normalization", "reset"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert sum(isinstance(node.func, ast.Attribute) and node.func.attr == "process_env_step" for node in calls) == 1
