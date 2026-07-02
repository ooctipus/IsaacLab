# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for packed multi-checkpoint SMPL-CMU evidence production."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardDualNetworkCfg, ForwardBackwardModel
from tensordict import TensorDict

MODULE_PATH = Path(__file__).with_name("smpl_cmu_emd_evidence.py")
_ROUTES = {name: ("state",) for name in ("actor", "forward", "backward")}


@pytest.fixture(scope="module")
def module():
    """Load the evidence producer without launching Isaac Sim."""
    spec = importlib.util.spec_from_file_location("smpl_cmu_emd_evidence_batch", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _observations(seed: int, *batch_size: int) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    return TensorDict(
        {"state": torch.randn(*batch_size, 7, generator=generator)},
        batch_size=[*batch_size],
    )


def _model(seed: int) -> ForwardBackwardModel:
    torch.manual_seed(seed)
    model = ForwardBackwardModel(
        _observations(seed, 2),
        _ROUTES,
        action_dim=2,
        context_dim=4,
        actor_cfg=ForwardBackwardDualNetworkCfg(hidden_dim=16, hidden_layers=2, embedding_layers=2),
        forward_cfg=ForwardBackwardDualNetworkCfg(hidden_dim=16, hidden_layers=2, embedding_layers=2),
        backward_hidden_dims=(8, 8),
        normalization_type="empirical",
        distribution_cfg={
            "class_name": "ClippedGaussianDistribution",
            "init_std": 0.2,
            "action_range": (-0.8, 0.9),
        },
        context_normalization=True,
    )
    model.update_normalization(_observations(seed + 100, 11))
    model.eval()
    return model


def _checkpoint(path: Path, module, *, iteration: int, transition: int, value: float) -> Path:
    torch.save({"model_state_dict": {"weight": torch.full((3,), value)}}, path)
    manifest = {
        "schema": module._COMPACT_SCHEMA,
        "iteration": iteration,
        "collected_transitions": transition,
        "output": {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": module._sha256(path),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    return path


def _model_checkpoint(path: Path, module, model: ForwardBackwardModel, *, transition: int) -> Path:
    torch.save({"model_state_dict": model.state_dict()}, path)
    manifest = {
        "schema": module._COMPACT_SCHEMA,
        "iteration": transition // 100,
        "collected_transitions": transition,
        "output": {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": module._sha256(path),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    return path


def _inference_state_bytes(model: ForwardBackwardModel) -> int:
    return sum(value.numel() * value.element_size() for value in model.as_inference_model().state_dict().values())


def test_stacked_inference_matches_checkpoint_major_independent_models(module) -> None:
    """One functional state stack must preserve the explicit [K, rows] layout."""
    models = tuple(_model(seed) for seed in (11, 22, 33))
    views = tuple(model.as_inference_model() for model in models)
    adapter = module._StackedForwardBackwardInference(views, device="cpu")
    observations = _observations(44, 3, 5)

    expected_backward = torch.stack([view.backward_map(observations[index]) for index, view in enumerate(views)])
    actual_backward = adapter.backward_map(observations)
    context = adapter.context_project(actual_backward)
    expected_action = torch.stack(
        [view.action_deterministic(observations[index], context[index]) for index, view in enumerate(views)]
    )
    actual_action = adapter.action_deterministic(observations, context)

    torch.testing.assert_close(actual_backward, expected_backward, rtol=1.0e-5, atol=2.0e-7)
    torch.testing.assert_close(actual_action, expected_action, rtol=1.0e-5, atol=2.0e-7)
    assert actual_backward.shape == (3, 5, 4)
    assert actual_action.shape == (3, 5, 2)
    assert adapter.policy_count == 3
    assert adapter.state_bytes_per_policy == tuple(
        sum(value.numel() * value.element_size() for value in view.state_dict().values()) for view in views
    )
    assert adapter.stacked_state_bytes == sum(adapter.state_bytes_per_policy)


def test_stacked_inference_reuses_rsl_modules_without_model_algebra(module) -> None:
    """The producer adapter must only functionalize the reusable RSL inference view."""
    del module
    producer = MODULE_PATH.read_text()
    source = producer[
        producer.index("class _StackedForwardBackwardInference") : producer.index("\ndef _canonical_checkpoint_batch")
    ]

    assert "torch.func.stack_module_state" in source
    assert "torch.func.functional_call" in source
    assert "torch.vmap" in source
    assert "MLP(" not in source
    assert "Linear(" not in source


def test_checkpoint_batch_has_canonical_transition_digest_order(tmp_path: Path, module) -> None:
    """Caller argument order must not enter packed evaluation semantics."""
    third = _checkpoint(tmp_path / "model_3.pt", module, iteration=3, transition=300, value=3.0)
    first = _checkpoint(tmp_path / "model_1.pt", module, iteration=1, transition=100, value=1.0)
    second = _checkpoint(tmp_path / "model_2.pt", module, iteration=2, transition=200, value=2.0)

    identities = module._canonical_checkpoint_batch((third, first, second))

    assert [identity["transition"] for identity in identities] == [100, 200, 300]
    assert identities == tuple(sorted(identities, key=lambda value: (value["transition"], value["sha256"])))
    members, digest = module._checkpoint_batch_identity(identities)
    serialized_members = json.loads(json.dumps(members))
    assert [member["transition"] for member in serialized_members] == [100, 200, 300]
    assert digest == module._canonical_sha256(serialized_members)
    with pytest.raises(ValueError, match="unique"):
        module._canonical_checkpoint_batch((first, first))
    same_transition = _checkpoint(tmp_path / "model_4.pt", module, iteration=4, transition=100, value=4.0)
    with pytest.raises(ValueError, match="transitions must be unique"):
        module._canonical_checkpoint_batch((first, same_transition))


def test_batch_output_preflight_is_collision_free_and_happens_before_runtime(tmp_path: Path, module) -> None:
    """Every immutable destination must be free before simulator construction."""
    first = _checkpoint(tmp_path / "model_a.pt", module, iteration=1, transition=100, value=1.0)
    second = _checkpoint(tmp_path / "model_b.pt", module, iteration=2, transition=200, value=2.0)
    identities = module._canonical_checkpoint_batch((second, first))
    output_dir = tmp_path / "evidence"

    outputs = module._batch_output_paths(output_dir, identities)

    assert len(outputs) == 2
    assert len(set(outputs)) == 2
    assert [output.name for output in outputs] == ["100.json", "200.json"]
    outputs[1].parent.mkdir(parents=True)
    outputs[1].write_text("occupied")
    with pytest.raises(FileExistsError, match="already exists"):
        module._batch_output_paths(output_dir, identities)


def test_request_normalization_keeps_evaluator_mode_independent_of_cardinality(tmp_path: Path, module) -> None:
    """Artifact cardinality must not implicitly choose the scientific evaluator."""
    later = _checkpoint(tmp_path / "model_2.pt", module, iteration=2, transition=200, value=2.0)
    earlier = _checkpoint(tmp_path / "model_1.pt", module, iteration=1, transition=100, value=1.0)
    packed = argparse.Namespace(
        checkpoints=(later, earlier),
        output_dir=tmp_path / "packed",
        num_envs=None,
        evaluator_mode="packed",
    )

    request = module._prepare_request(packed)

    assert [identity["transition"] for identity in request.identities] == [100, 200]
    assert [path.name for path in request.outputs] == ["100.json", "200.json"]
    assert request.lanes_per_policy_override is None
    assert request.evaluator_mode == "packed"

    faithful = argparse.Namespace(
        checkpoints=(earlier,),
        output_dir=tmp_path / "faithful",
        num_envs=37,
        evaluator_mode="faithful",
    )
    request = module._prepare_request(faithful)
    assert request.identities[0]["transition"] == 100
    assert request.outputs == ((tmp_path / "faithful" / "100.json").resolve(),)
    assert request.lanes_per_policy_override == 37
    assert request.evaluator_mode == "faithful"


def test_request_normalization_rejects_implicit_or_multicheckpoint_faithful_mode(tmp_path: Path, module) -> None:
    """Faithful semantics require an explicit mode and exactly one checkpoint."""
    first = _checkpoint(tmp_path / "model_1.pt", module, iteration=1, transition=100, value=1.0)
    second = _checkpoint(tmp_path / "model_2.pt", module, iteration=2, transition=200, value=2.0)
    with pytest.raises(ValueError, match="exactly one checkpoint"):
        module._prepare_request(
            argparse.Namespace(
                checkpoints=(first, second),
                output_dir=tmp_path / "faithful",
                num_envs=None,
                evaluator_mode="faithful",
            )
        )
    with pytest.raises(ValueError, match="evaluator_mode"):
        module._prepare_request(
            argparse.Namespace(
                checkpoints=(first,),
                output_dir=tmp_path / "batch",
                num_envs=None,
                evaluator_mode=None,
            )
        )


def test_compact_checkpoints_load_strictly_into_one_functional_stack(tmp_path: Path, module) -> None:
    """Sequential full-model loads must retain only inference state in the packed adapter."""
    models = tuple(_model(seed) for seed in (101, 202, 303))
    paths = tuple(
        _model_checkpoint(tmp_path / f"model_{index}.pt", module, model, transition=index * 100)
        for index, model in enumerate(models, start=1)
    )
    identities = module._canonical_checkpoint_batch(tuple(reversed(paths)))
    adapter = module._load_stacked_inference(_model(999), identities, device="cpu")
    observations = _observations(404, 3, 6)

    expected_views = tuple(model.as_inference_model() for model in models)
    expected_backward = torch.stack(
        [view.backward_map(observations[index]) for index, view in enumerate(expected_views)]
    )
    actual_backward = adapter.backward_map(observations)
    torch.testing.assert_close(actual_backward, expected_backward, rtol=1.0e-5, atol=2.0e-7)
    contexts = adapter.context_project(actual_backward)
    expected_actions = torch.stack(
        [view.action_deterministic(observations[index], contexts[index]) for index, view in enumerate(expected_views)]
    )
    actual_actions = adapter.action_deterministic(observations, contexts)
    torch.testing.assert_close(actual_actions, expected_actions, rtol=1.0e-5, atol=2.0e-7)
    assert adapter.stacked_state_bytes == sum(_inference_state_bytes(model) for model in models)
    assert adapter.state_bytes_per_policy == tuple(_inference_state_bytes(model) for model in models)
