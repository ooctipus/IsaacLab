# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the fast native G1 checkpoint semantic preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent
PREFLIGHT = ROOT / "g1_checkpoint_semantic_preflight.py"
TRACE = ROOT / "fixtures" / "g1_lafan_same_step_trace_v1.npz"
BEHAVIOR_JOINT_NAMES = tuple(f"joint_{index}" for index in range(29))
PHYSICAL_JOINT_NAMES = (
    *BEHAVIOR_JOINT_NAMES[::3],
    *BEHAVIOR_JOINT_NAMES[1::3],
    *BEHAVIOR_JOINT_NAMES[2::3],
)
BEHAVIOR_BODY_NAMES = tuple(f"body_{index}" for index in range(30))
PHYSICAL_BODY_NAMES = (
    BEHAVIOR_BODY_NAMES[0],
    *BEHAVIOR_BODY_NAMES[1::2],
    *BEHAVIOR_BODY_NAMES[2::2],
)


def _module():
    spec = importlib.util.spec_from_file_location("g1_checkpoint_semantic_preflight", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(names: tuple[str, ...], physical: tuple[str, ...]) -> torch.Tensor:
    return torch.tensor([physical.index(name) for name in names], dtype=torch.int64)


def test_preflight_rejects_old_identity_ids_for_nonidentity_live_axis() -> None:
    """The old raw-live ordering must fail before a long checkpoint evaluation starts."""
    module = _module()
    body_ids = _ids(BEHAVIOR_BODY_NAMES, PHYSICAL_BODY_NAMES)

    with pytest.raises(ValueError, match="behavior-to-physical joint map"):
        module._behavior_axis_identity(
            physical_joint_names=PHYSICAL_JOINT_NAMES,
            behavior_joint_names=BEHAVIOR_JOINT_NAMES,
            behavior_joint_ids=torch.arange(29),
            physical_body_names=PHYSICAL_BODY_NAMES,
            behavior_body_names=BEHAVIOR_BODY_NAMES,
            behavior_body_ids=body_ids,
        )


def test_preflight_accepts_exact_nonidentity_behavior_axes() -> None:
    """Both checkpoint-facing axes must retain their explicit semantic permutations."""
    module = _module()
    joint_ids = _ids(BEHAVIOR_JOINT_NAMES, PHYSICAL_JOINT_NAMES)
    body_ids = _ids(BEHAVIOR_BODY_NAMES, PHYSICAL_BODY_NAMES)

    identity = module._behavior_axis_identity(
        physical_joint_names=PHYSICAL_JOINT_NAMES,
        behavior_joint_names=BEHAVIOR_JOINT_NAMES,
        behavior_joint_ids=joint_ids,
        physical_body_names=PHYSICAL_BODY_NAMES,
        behavior_body_names=BEHAVIOR_BODY_NAMES,
        behavior_body_ids=body_ids,
    )

    assert identity["joint"]["behavior_to_physical"] == joint_ids.tolist()
    assert identity["joint"]["is_identity"] is False
    assert identity["body"]["behavior_to_physical"] == body_ids.tolist()
    assert identity["body"]["is_identity"] is False


def test_raw_physical_counterexample_permutes_every_joint_bearing_actor_field() -> None:
    """The sensitivity probe must reproduce the former raw-live semantic error exactly."""
    module = _module()
    joint_ids = _ids(BEHAVIOR_JOINT_NAMES, PHYSICAL_JOINT_NAMES)
    state = torch.arange(64, dtype=torch.float32).view(1, -1)
    last_action = (100.0 + torch.arange(29, dtype=torch.float32)).view(1, -1)
    history = (200.0 + torch.arange(372, dtype=torch.float32)).view(1, -1)
    observations = {
        "state": state,
        "last_action": last_action,
        "history_actor": history,
        "privileged_state": torch.zeros(1, 463),
    }

    raw = module._raw_physical_actor_observations(observations, joint_ids)

    expected_state = state.clone()
    expected_state[:, :29].scatter_(1, joint_ids.view(1, -1), state[:, :29])
    expected_state[:, 29:58].scatter_(1, joint_ids.view(1, -1), state[:, 29:58])
    torch.testing.assert_close(raw["state"], expected_state)
    expected_last_action = last_action.clone()
    expected_last_action.scatter_(1, joint_ids.view(1, -1), last_action)
    torch.testing.assert_close(raw["last_action"], expected_last_action)

    expected_history = history.clone()
    for offset in (0, 116, 128, 244):
        field_width = 29 if offset == 0 else (3 if offset == 116 else 29)
        if field_width != 29:
            continue
        for lag in range(4):
            start = offset + lag * 29
            expected_history[:, start : start + 29].scatter_(
                1,
                joint_ids.view(1, -1),
                history[:, start : start + 29],
            )
    torch.testing.assert_close(raw["history_actor"], expected_history)
    assert raw["privileged_state"] is observations["privileged_state"]


def test_native_trace_loader_returns_the_exact_checkpoint_fields() -> None:
    """The preflight input must come from the frozen native BFM trace, not synthetic values."""
    module = _module()

    observations, identity = module._load_native_trace_observations(TRACE, torch.device("cpu"))

    assert tuple(observations.batch_size) == (2,)
    assert {name: tuple(value.shape) for name, value in observations.items()} == {
        "state": (2, 64),
        "last_action": (2, 29),
        "history_actor": (2, 372),
        "privileged_state": (2, 463),
    }
    assert identity["step"] == 0
    assert identity["batch_size"] == 2
    assert len(identity["sha256"]) == 64


@pytest.mark.parametrize(
    "change,error",
    (
        ({"runtime_seconds": 120.001}, "runtime"),
        ({"checkpoint_strict_load": False}, "checkpoint"),
        ({"axis_identity": False}, "axis"),
        ({"route_max_abs_error": 1.0e-7}, "route"),
        ({"raw_order_action_delta": 0.0}, "sensitive"),
        ({"raw_order_backward_delta": 0.0}, "sensitive"),
        ({"rollout_finite": False}, "finite"),
        ({"action_request_max_abs_error": 1.0e-7}, "action boundary"),
        ({"processed_action_max_abs_error": 1.0e-7}, "action boundary"),
        ({"action_target_max_abs_error": 1.0e-7}, "action boundary"),
        ({"done_rows": 1}, "terminal"),
    ),
)
def test_preflight_decision_fails_every_mandatory_gate(change: dict[str, object], error: str) -> None:
    """No shape-only or partial result may authorize a long policy-quality run."""
    module = _module()
    values = {
        "runtime_seconds": 10.0,
        "runtime_limit_seconds": 120.0,
        "checkpoint_strict_load": True,
        "axis_identity": True,
        "route_max_abs_error": 0.0,
        "raw_order_action_delta": 1.0,
        "raw_order_backward_delta": 1.0,
        "rollout_finite": True,
        "action_request_max_abs_error": 0.0,
        "processed_action_max_abs_error": 0.0,
        "action_target_max_abs_error": 0.0,
        "done_rows": 0,
    }
    values.update(change)

    with pytest.raises(RuntimeError, match=error):
        module._require_preflight_pass(**values)


def test_preflight_decision_accepts_only_complete_semantic_evidence() -> None:
    """The fast gate should return one explicit pass after all semantic checks hold."""
    module = _module()

    result = module._require_preflight_pass(
        runtime_seconds=10.0,
        runtime_limit_seconds=120.0,
        checkpoint_strict_load=True,
        axis_identity=True,
        route_max_abs_error=0.0,
        raw_order_action_delta=1.0,
        raw_order_backward_delta=1.0,
        rollout_finite=True,
        action_request_max_abs_error=0.0,
        processed_action_max_abs_error=0.0,
        action_target_max_abs_error=0.0,
        done_rows=0,
    )

    assert result == {"status": "passed", "passed": True}
