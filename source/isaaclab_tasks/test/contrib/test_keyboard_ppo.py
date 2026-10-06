# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PPO storage and returns across same-step keyboard autoresets."""

import ast
import inspect
import textwrap

import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.contrib.keyboard.agents.models import SharedArmMLPModel, SharedEncoderMLPModel, SharedEncoderPPO


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


def _multi_arm_observations():
    generator = torch.Generator().manual_seed(17)
    states = torch.randn(2, 2, 25, generator=generator)
    states[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    return TensorDict(
        {
            "policy": torch.randn(2, 12, generator=generator),
            "robot_state": states,
            "robot_active": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            "key_positions": torch.randn(2, 4, 3, generator=generator),
            "key_active": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
        },
        batch_size=[2],
    )


def _multi_arm_algorithm(observations):
    groups = {"actor": list(observations.keys()), "critic": list(observations.keys())}
    config = {
        "hidden_dims": [16],
        "obs_normalization": True,
        "encoder_cfg": {"policy": {"hidden_dims": [8], "latent_dim": 4}},
    }
    actor = SharedArmMLPModel(
        observations, groups, "actor", 12, distribution_cfg={"class_name": "GaussianDistribution"}, **config
    )
    critic = SharedArmMLPModel(observations, groups, "critic", 1, **config)
    storage = RolloutStorage("rl", len(observations), 2, observations, [12], device="cpu")
    return SharedEncoderPPO(actor, critic, storage, gamma=0.9, num_learning_epochs=1, num_mini_batches=1)


def test_shared_arm_inference_tracks_active_count_and_is_permutation_equivariant():
    observations = _multi_arm_observations()
    algorithm = _multi_arm_algorithm(observations)
    actor, critic = algorithm.actor, algorithm.critic
    arm_batches, world_batches = [], []
    head_hook = actor.mlp.register_forward_pre_hook(lambda _, args: arm_batches.append(len(args[0])))
    encoder_hook = actor.encoders["policy"].register_forward_pre_hook(
        lambda _, args: world_batches.append(len(args[0]))
    )
    try:
        actions, values = actor(observations).reshape(2, 2, 6), critic(observations)
        swapped = observations.clone()
        swapped["robot_state"] = swapped["robot_state"].flip(1)
        swapped["robot_active"] = swapped["robot_active"].flip(1)
        torch.testing.assert_close(actor(swapped).reshape(2, 2, 6), actions.flip(1))
        torch.testing.assert_close(critic(swapped), values)
        observations["robot_active"].fill_(1.0)
        actor(observations)
        observations["robot_active"][:, 1] = 0.0
        actor(observations)
    finally:
        head_hook.remove()
        encoder_hook.remove()
    assert arm_batches == [3, 3, 4, 2]
    # Actor and critic share the world encoder, but each encodes once per model evaluation.
    assert world_batches == [2] * 6
    assert values.shape == (2, 1) and torch.equal(actions[0, 1], torch.zeros(6))
    assert actor.distribution.std_param.shape == (6,)
    parameters = algorithm.optimizer.param_groups[0]["params"]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)


def test_shared_arm_padding_has_no_effect_on_outputs_normalization_or_gradients():
    observations = _multi_arm_observations()
    actor = _multi_arm_algorithm(observations).actor
    expected = actor(observations)
    contaminated = observations.clone()
    inactive_robots = ~contaminated["robot_active"].bool()
    inactive_keys = ~contaminated["key_active"].bool()
    contaminated["robot_state"][inactive_robots] = float("nan")
    contaminated["key_positions"][inactive_keys] = float("nan")
    contaminated["robot_state"].requires_grad_()
    contaminated["key_positions"].requires_grad_()
    actual = actor(contaminated)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    for group, inactive in (("robot_state", inactive_robots), ("key_positions", inactive_keys)):
        gradients = contaminated[group].grad
        assert torch.isfinite(gradients).all()
        assert torch.count_nonzero(gradients[inactive]) == 0
    # The first arm in a dual world can respond to the other arm's root, joints, and motion.
    state = observations["robot_state"].clone().requires_grad_()
    observations["robot_state"] = state
    other_gradient = torch.autograd.grad(actor(observations)[1, :6].sum(), state)[0][1, 1]
    assert all(torch.count_nonzero(other_gradient[part]) for part in (slice(0, 7), slice(7, 13), slice(13, 19)))
    actor.update_normalization(contaminated)
    assert actor.obs_normalizer.count == 3
    assert torch.isfinite(actor.obs_normalizer.mean).all()


def test_shared_arm_world_likelihood_entropy_and_kl_exclude_absent_arms():
    observations = _multi_arm_observations()
    actor = _multi_arm_algorithm(observations).actor
    actions = actor(observations, stochastic_output=True)
    means, stds, active = actor.output_distribution_params
    expected = torch.distributions.Normal(means, stds).log_prob(actions.reshape(2, 2, 6)).sum(-1)
    expected = torch.stack((expected[0, 0], expected[1].sum()))
    actions[0, 6:] = float("nan")
    log_prob = actor.get_output_log_prob(actions)
    torch.testing.assert_close(log_prob, expected)
    mean_gradient = torch.autograd.grad(log_prob.sum(), means, retain_graph=True)[0]
    assert torch.count_nonzero(mean_gradient[~active]) == 0
    unit_entropy = torch.distributions.Normal(0.0, 1.0).entropy()
    torch.testing.assert_close(actor.output_entropy, torch.tensor([6.0, 12.0]) * unit_entropy)
    shifted_means = means.detach().clone() + 1.0
    shifted_means[~active] = float("nan")
    shifted_means.requires_grad_()
    kl = actor.get_kl_divergence((means.detach(), stds.detach(), active), (shifted_means, stds.detach(), active))
    torch.testing.assert_close(kl, torch.tensor([3.0, 6.0]))
    kl.sum().backward()
    assert torch.isfinite(shifted_means.grad).all()
    assert torch.count_nonzero(shifted_means.grad[~active]) == 0
    # One shared std receives entropy gradients for three active six-DOF distributions.
    actor.distribution.std_param.grad = None
    actor.output_entropy.sum().backward()
    torch.testing.assert_close(actor.distribution.std_param.grad, torch.full((6,), 3.0))


def test_shared_arm_policy_uses_scalar_first_root_frames_and_other_joint_motion():
    observations = _multi_arm_observations()
    actor = _multi_arm_algorithm(observations).actor
    actor.obs_normalizer = torch.nn.Identity()
    # A calibrated head reads own x, other relative x/quaternion w/q/qd, and the first key's relative x.
    with torch.no_grad():
        for parameter in actor.mlp.parameters():
            parameter.zero_()
        for output, feature in enumerate((0, 25, 28, 32, 38, 51)):
            actor.mlp[0].weight[output, feature] = 1.0
            actor.mlp[-1].weight[output, output] = 1.0
    # The first arm faces +y; the second faces -x and sits 2 m along that first arm's local +x.
    half = 2.0**-0.5
    observations["robot_state"][1, :, :7] = torch.tensor(
        [[1.0, 2.0, 3.0, half, 0.0, 0.0, half], [1.0, 4.0, 3.0, 0.0, 0.0, 0.0, 1.0]]
    )
    observations["robot_state"][1, 1, 7] = 5.0
    observations["robot_state"][1, 1, 13] = 7.0
    observations["key_positions"][1, 0] = torch.tensor([1.0, 3.0, 3.0])
    torch.testing.assert_close(actor(observations)[1, :6], torch.tensor([1.0, 2.0, half, 5.0, 7.0, 1.0]))


def test_shared_arm_real_ppo_keeps_world_rows_across_resets_and_updates_weights():
    observations = _multi_arm_observations()
    algorithm = _multi_arm_algorithm(observations)
    with torch.no_grad():
        for parameter in algorithm.critic.mlp.parameters():
            parameter.zero_()
        algorithm.critic.mlp[-1].bias.fill_(2.0)
    before = algorithm.actor.mlp[0].weight.detach().clone()
    with torch.inference_mode():
        algorithm.act(observations)
        next_observations = observations.clone()
        next_observations["robot_active"] = observations["robot_active"].flip(0)
        algorithm.process_env_step(
            next_observations,
            torch.ones(2),
            torch.tensor([1, 0]),
            {"time_outs": torch.tensor([True, False]), "final_obs": observations},
        )
        algorithm.act(next_observations)
        algorithm.process_env_step(observations, torch.tensor([0.5, -0.5]), torch.tensor([0, 1]), {})
        algorithm.compute_returns(observations)
    assert algorithm.storage.actions.shape == (2, 2, 12)
    assert algorithm.storage.values.shape == (2, 2, 1)
    torch.testing.assert_close(algorithm.storage.values, torch.full((2, 2, 1), 2.0))
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([2.8, 1.0]))
    assert algorithm.storage.distribution_params[0].shape == (2, 2, 2, 6)
    torch.testing.assert_close(algorithm.storage.distribution_params[2][0], observations["robot_active"])
    torch.testing.assert_close(algorithm.storage.distribution_params[2][1], next_observations["robot_active"])
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses.values())
    assert not torch.equal(algorithm.actor.mlp[0].weight, before)
    assert algorithm.actor.obs_normalizer.count == 6
    restored = _multi_arm_algorithm(observations)
    restored.load(algorithm.save(), load_cfg=None, strict=True)
    torch.testing.assert_close(restored.actor(observations), algorithm.actor(observations))
    torch.testing.assert_close(restored.critic(observations), algorithm.critic(observations))


def test_shared_arm_exports_preserve_dynamic_active_rows_and_world_outputs(tmp_path):
    observations = _multi_arm_observations()
    algorithm = _multi_arm_algorithm(observations)
    for model in (algorithm.actor, algorithm.critic):
        model.update_normalization(observations)
        scripted = torch.jit.script(model.as_jit())
        onnx_model = model.as_onnx()
        for active in (torch.tensor([[1.0, 0.0], [1.0, 1.0]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])):
            observations["robot_active"] = active
            inputs = tuple(observations[group] for group in onnx_model.input_names)
            expected = model(observations)
            torch.testing.assert_close(scripted(*inputs), expected)
            torch.testing.assert_close(onnx_model(*inputs), expected)
    export = algorithm.actor.as_onnx().eval()
    path = tmp_path / "policy.onnx"
    torch.onnx.export(
        export,
        export.get_dummy_inputs(),
        path,
        export_params=True,
        opset_version=18,
        input_names=export.input_names,
        output_names=export.output_names,
    )
    onnx.checker.check_model(path)
    reference = ReferenceEvaluator(str(path))
    one_world = observations[:1].clone()
    for active in ([1.0, 1.0], [1.0, 0.0], [0.0, 1.0]):
        one_world["robot_active"][:] = torch.tensor(active)
        inputs = {name: one_world[name].numpy() for name in export.input_names}
        actual = torch.from_numpy(reference.run(None, inputs)[0])
        torch.testing.assert_close(actual, algorithm.actor(one_world))
