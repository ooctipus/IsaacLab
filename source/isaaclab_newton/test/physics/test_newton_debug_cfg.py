# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the public Newton physics debug configuration."""

from __future__ import annotations

import pytest
from isaaclab_newton.physics import NewtonCfg, NewtonDebugCaptureCfg, NewtonDebugReplayCfg


def test_replay_record_operations_requires_enabled_replay():
    """Replay operation capture cannot be configured for a disabled replay recorder."""
    debug_capture = NewtonDebugCaptureCfg(replay=NewtonDebugReplayCfg(enabled=False, record_operations=True))

    with pytest.raises(
        ValueError,
        match=r"record_operations=True requires .*replay\.enabled=True",
    ):
        NewtonCfg(debug_capture=debug_capture)


def test_incident_operation_capture_is_independent_of_replay():
    """Cold incident operation capture does not enable transition replay."""
    debug_capture = NewtonDebugCaptureCfg(record_operations=True)

    cfg = NewtonCfg(use_cuda_graph=False, debug_capture=debug_capture)

    assert cfg.debug_capture.record_operations is True
    assert cfg.debug_capture.replay.enabled is False
    assert cfg.debug_capture.replay.record_operations is False


def test_incident_operation_capture_is_opt_in():
    """Transient operations are not retained in cold incidents by default."""
    assert NewtonDebugCaptureCfg().record_operations is False


def test_incident_operation_capture_requires_a_boolean():
    """A truthy non-boolean cannot silently require an operation provider."""
    with pytest.raises(TypeError, match=r"record_operations.*bool"):
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(record_operations=1))


def test_replay_only_operation_capture_remains_valid():
    """Transition history can retain operations without cold incident capture."""
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=False,
        record_contacts=False,
        record_operations=True,
    )

    cfg = NewtonCfg(
        use_cuda_graph=False,
        debug_capture=NewtonDebugCaptureCfg(record_operations=False, replay=replay),
    )

    assert cfg.debug_capture.record_operations is False
    assert cfg.debug_capture.replay.record_operations is True


def test_collision_pipeline_capture_is_opt_in():
    """External collision internals are not recorded unless explicitly requested."""
    assert NewtonDebugCaptureCfg().record_collision_pipeline is False


def test_scene_capture_is_explicit_and_disabled_by_default():
    """Static full-stage USD export is opt-in while arrays remain authoritative."""
    assert NewtonDebugCaptureCfg().record_scene is False
    assert NewtonCfg(debug_capture=NewtonDebugCaptureCfg(record_scene=True)).debug_capture.record_scene is True


def test_cold_control_capture_is_comprehensive_by_default():
    """Applied control is retained even when transition replay is disabled."""
    assert NewtonDebugCaptureCfg().record_control is True


def test_cold_control_capture_requires_a_boolean():
    """A truthy non-boolean cannot silently alter required control binding."""
    with pytest.raises(TypeError, match=r"record_control.*bool"):
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(record_control=1))


def test_contact_capture_is_explicit_and_disabled_by_default():
    """Solver-specific contact evidence must be requested with known live semantics."""
    assert NewtonDebugCaptureCfg().record_contacts is False


def test_scene_capture_requires_a_boolean():
    """A truthy non-boolean cannot accidentally request cold-path USD export."""
    with pytest.raises(TypeError, match=r"record_scene.*bool"):
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(record_scene=1))


def test_collision_pipeline_capture_requires_a_boolean():
    """A truthy non-boolean cannot accidentally enable required provider binding."""
    debug_capture = NewtonDebugCaptureCfg(record_collision_pipeline=1)

    with pytest.raises(TypeError, match=r"record_collision_pipeline.*bool"):
        NewtonCfg(debug_capture=debug_capture)


def test_nonfinite_detection_defaults_to_state():
    """The low-overhead default scans only physics state."""
    assert NewtonDebugCaptureCfg().detect_nonfinite_in == ("state",)


@pytest.mark.parametrize(
    ("providers", "error_type", "message"),
    [
        (["state"], TypeError, "tuple of strings"),
        ((), ValueError, "must not be empty"),
        (("state", "state"), ValueError, "duplicates"),
        (("unknown",), ValueError, "unsupported providers"),
        ((1,), ValueError, "non-empty strings"),
    ],
)
def test_nonfinite_detection_rejects_invalid_provider_selections(
    providers,
    error_type,
    message: str,
):
    """Detection providers use one exact, deterministic tuple vocabulary."""
    debug_capture = NewtonDebugCaptureCfg(detect_nonfinite_in=providers)

    with pytest.raises(error_type, match=message):
        NewtonCfg(debug_capture=debug_capture)


@pytest.mark.parametrize(
    ("provider", "record_flag"),
    [
        ("model", "record_model"),
        ("control", "record_control"),
        ("contacts", "record_contacts"),
        ("solver", "record_solver"),
        ("collision_pipeline", "record_collision_pipeline"),
    ],
)
def test_nonfinite_detection_requires_recorded_retained_providers(
    provider: str,
    record_flag: str,
):
    """A provider cannot trigger incidents while being omitted from artifacts."""
    debug_capture = NewtonDebugCaptureCfg(
        detect_nonfinite_in=(provider,),
        **{record_flag: False},
    )

    with pytest.raises(ValueError, match=record_flag):
        NewtonCfg(debug_capture=debug_capture)


@pytest.mark.parametrize(
    "replay",
    [
        NewtonDebugReplayCfg(),
        NewtonDebugReplayCfg(enabled=True, record_operations=False),
    ],
)
def test_operation_nonfinite_detection_requires_incident_operation_capture(replay: NewtonDebugReplayCfg):
    """Replay retention alone cannot make operations an incident detector."""
    debug_capture = NewtonDebugCaptureCfg(
        record_operations=False,
        detect_nonfinite_in=("operations",),
        replay=replay,
    )

    with pytest.raises(ValueError, match=r"operations.*requires record_operations=True"):
        NewtonCfg(use_cuda_graph=False, debug_capture=debug_capture)


def test_context_nonfinite_detection_requires_no_record_flag():
    """Registered workflow context can trigger without a dedicated record flag."""
    debug_capture = NewtonDebugCaptureCfg(detect_nonfinite_in=("context",))

    cfg = NewtonCfg(debug_capture=debug_capture)

    assert cfg.debug_capture.detect_nonfinite_in == ("context",)


def test_nonfinite_detection_accepts_every_supported_provider():
    """All retained, context, and transient provider names compose without hidden aliases."""
    providers = (
        "state",
        "model",
        "control",
        "contacts",
        "solver",
        "collision_pipeline",
        "context",
        "operations",
    )
    debug_capture = NewtonDebugCaptureCfg(
        record_contacts=True,
        record_collision_pipeline=True,
        record_operations=True,
        detect_nonfinite_in=providers,
    )

    cfg = NewtonCfg(use_cuda_graph=False, debug_capture=debug_capture)

    assert cfg.debug_capture.detect_nonfinite_in == providers


def test_replay_collision_pipeline_capture_is_opt_in():
    """Rolling external collision evidence is disabled by default."""
    assert NewtonDebugReplayCfg().record_collision_pipeline is False


def test_replay_collision_pipeline_capture_requires_a_boolean():
    """A truthy non-boolean cannot enable a large replay provider."""
    replay = NewtonDebugReplayCfg(record_collision_pipeline=1)
    debug_capture = NewtonDebugCaptureCfg(replay=replay)

    with pytest.raises(TypeError, match=r"replay.record_collision_pipeline.*bool"):
        NewtonCfg(debug_capture=debug_capture)


def test_replay_collision_pipeline_capture_requires_enabled_replay():
    """Rolling collision capture cannot run while replay is disabled."""
    replay = NewtonDebugReplayCfg(enabled=False, record_collision_pipeline=True)
    debug_capture = NewtonDebugCaptureCfg(replay=replay)

    with pytest.raises(
        ValueError,
        match=r"record_collision_pipeline=True requires .*replay.enabled=True",
    ):
        NewtonCfg(debug_capture=debug_capture)


def test_replay_collision_pipeline_counts_as_a_recorded_provider():
    """Collision internals alone form a valid replay schema selection."""
    replay = NewtonDebugReplayCfg(
        enabled=True,
        record_state=False,
        record_control=False,
        record_solver=False,
        record_contacts=False,
        record_collision_pipeline=True,
        record_operations=False,
    )
    debug_capture = NewtonDebugCaptureCfg(replay=replay)

    cfg = NewtonCfg(use_cuda_graph=False, debug_capture=debug_capture)

    assert cfg.debug_capture.replay.record_collision_pipeline is True


def test_nonfinite_scan_filters_default_to_all_recorded_fields():
    """Scan-only filters preserve comprehensive detection by default."""
    capture = NewtonDebugCaptureCfg()

    assert capture.detect_nonfinite_include_fields == ("*",)
    assert capture.detect_nonfinite_exclude_fields == ()


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("detect_nonfinite_include_fields", ["*"], TypeError, "tuple of strings"),
        ("detect_nonfinite_include_fields", (), ValueError, "must not be empty"),
        ("detect_nonfinite_include_fields", ("state.*", "state.*"), ValueError, "duplicates"),
        ("detect_nonfinite_exclude_fields", ["state.x"], TypeError, "tuple of strings"),
        ("detect_nonfinite_exclude_fields", ("",), ValueError, "non-empty strings"),
        ("detect_nonfinite_exclude_fields", ("state.x", "state.x"), ValueError, "duplicates"),
    ],
)
def test_nonfinite_scan_filters_reject_invalid_pattern_tuples(
    field_name: str,
    value,
    error_type,
    message: str,
):
    """Scan filters use the same strict deterministic tuple contract as archives."""
    capture = NewtonDebugCaptureCfg(**{field_name: value})

    with pytest.raises(error_type, match=message):
        NewtonCfg(debug_capture=capture)
