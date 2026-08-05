# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused CPU tests for the mechanics-neutral retarget evaluation contract."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).with_name("benchmark_neutral_retargeting.py")
SELECTED = "a" * 64
NATIVE = "b" * 64


@pytest.fixture(scope="module")
def benchmark():
    """Load the pure evidence script without simulator imports."""
    spec = importlib.util.spec_from_file_location("neutral_retarget_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target(module, identity: str, limit: float = 2.0):
    return module.RetargetEvaluationTarget(
        geometry_sha256=identity,
        joint_names=("hip", "knee"),
        joint_lower_limit_rad=torch.tensor((-limit, -limit), dtype=torch.float64),
        joint_upper_limit_rad=torch.tensor((limit, limit), dtype=torch.float64),
        joint_velocity_limit_rad_s=torch.tensor((5.0, 5.0), dtype=torch.float64),
        support_patch_roles=("left_foot", "right_foot"),
        support_patch_offsets=(0, 3, 6),
        collision_probe_count=6,
    )


def _source(module):
    time = torch.arange(4, dtype=torch.float64)
    pelvis = torch.stack((0.2 * time, 0.1 * time, torch.full_like(time, 0.8)), dim=-1)
    landmarks = torch.stack(
        (pelvis, pelvis + pelvis.new_tensor((-0.2, 0.15, -0.7)), pelvis + pelvis.new_tensor((0.0, 0.0, 0.5))),
        dim=1,
    )
    rotations = torch.zeros((4, 1, 4), dtype=torch.float64)
    rotations[..., 3] = 1.0
    return module.RetargetEvaluationSourceClip(
        clip_id="dance1_subject1_clip0",
        source_content_sha256="c" * 64,
        semantic_projection_sha256="d" * 64,
        selected_target_geometry_sha256=SELECTED,
        frame_dt_s=1.0 / 30.0,
        frame_indices=torch.arange(4, dtype=torch.int64),
        landmark_roles=("pelvis", "left_ankle", "torso"),
        root_role="pelvis",
        semantic_edges=(("pelvis", "left_ankle"), ("pelvis", "torso")),
        landmark_position_m=landmarks,
        rotation_roles=("pelvis",),
        rotation_xyzw=rotations,
        support_patch_roles=("left_foot", "right_foot"),
        stance_active=torch.tensor(((True, True), (True, True), (True, False), (False, False))),
    )


def _view(module, source, target, *, yaw=0.0, xy=(0.0, 0.0), z=0.0, q=0.0, rows=(0, 1, 2, 3), dt=None):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = source.landmark_position_m.new_tensor(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    translation = source.landmark_position_m.new_tensor((*xy, 0.0))
    landmarks = (source.landmark_position_m - translation) @ rotation
    triangle = torch.tensor(((-0.1, -0.05, 0.0), (0.1, -0.05, 0.0), (0.0, 0.1, 0.0)), dtype=torch.float64)
    support = torch.cat((triangle + triangle.new_tensor((0.0, 0.2, 0.0)), triangle - triangle.new_tensor((0.0, 0.2, 0.0))))
    support = (support.unsqueeze(0).expand(4, -1, -1) - translation) @ rotation
    landmarks[..., 2] += z
    support[..., 2] += z
    quaternions = torch.zeros((4, 1, 4), dtype=torch.float64)
    quaternions[..., 2] = -math.sin(0.5 * yaw)
    quaternions[..., 3] = math.cos(0.5 * yaw)
    selected_rows = torch.tensor(rows, dtype=torch.int64)
    positions = torch.full((4, 2), q, dtype=torch.float64)
    velocities = torch.zeros_like(positions)
    return module.RetargetEvaluationView(
        geometry_sha256=target.geometry_sha256,
        clip_id=source.clip_id,
        frame_dt_s=source.frame_dt_s if dt is None else dt,
        frame_indices=source.frame_indices[selected_rows],
        landmark_roles=source.landmark_roles,
        landmark_position_m=landmarks[selected_rows],
        rotation_roles=source.rotation_roles,
        rotation_xyzw=quaternions[selected_rows],
        support_point_position_m=support[selected_rows],
        collision_probe_position_m=support[selected_rows],
        joint_names=target.joint_names,
        joint_position_rad=positions[selected_rows],
        joint_velocity_rad_s=velocities[selected_rows],
    )


def _method(module, name, native_target, native, selected):
    return module.RetargetEvaluationMethod(
        name=name,
        native_target=native_target,
        native_clips=(native,),
        selected_target_clips=(selected,),
        runtime=module.RetargetEvaluationRuntime(
            scope="source_decode_retarget_and_native_output",
            included_stages=("source_decode", "retarget", "native_materialization"),
            wall_seconds=2.0,
            input_frame_count=4,
            output_frame_count=native.frame_indices.numel(),
            device="cpu",
        ),
    )


def test_alignment_removes_only_constant_yaw_and_xy(benchmark) -> None:
    """Vertical fidelity and grounding must survive the allowed horizontal gauge quotient."""
    source, target = _source(benchmark), _target(benchmark, SELECTED)
    output = _view(benchmark, source, target, yaw=0.7, xy=(2.0, -3.0), z=0.1)
    report = benchmark.evaluate_neutral_retargeting(
        (source,), target, (_method(benchmark, "ours", target, output, output),)
    )
    aggregate = report["methods"]["ours"]["aggregate"]["selected_target_mechanics"]
    alignment = report["methods"]["ours"]["clips"][0]["selected_target_mechanics"]["source_fidelity"]["alignment"]
    assert alignment["translation_m"][2] == 0.0
    assert aggregate["source_fidelity"]["landmark_error_m"]["mean"] == pytest.approx(0.1)
    assert aggregate["contacts_ground"]["stance_patch_hover_m"]["mean"] == pytest.approx(0.1)
    assert report["contract"]["alignment"]["scale"] is False
    assert report["contract"]["quality_acceptance"] == "none_measurements_only"


def test_native_and_selected_limits_are_reported_separately(benchmark) -> None:
    """Tight native limits cannot be hidden by wider selected-target limits."""
    source = _source(benchmark)
    native_target, selected_target = _target(benchmark, NATIVE, 0.5), _target(benchmark, SELECTED, 2.0)
    native, selected = _view(benchmark, source, native_target, q=1.0), _view(benchmark, source, selected_target, q=1.0)
    report = benchmark.evaluate_neutral_retargeting(
        (source,), selected_target, (_method(benchmark, "protomotions", native_target, native, selected),)
    )
    aggregate = report["methods"]["protomotions"]["aggregate"]
    assert aggregate["native_mechanics"]["hard_target_constraints"]["position_excess_rad"]["max"] == 0.5
    assert aggregate["native_mechanics"]["hard_target_constraints"]["position_violating_samples"] == 8
    assert aggregate["selected_target_mechanics"]["hard_target_constraints"]["position_excess_rad"]["max"] == 0.0
    assert report["methods"]["protomotions"]["mechanics"]["same_geometry"] is False


def test_partial_frames_are_coverage_not_resampling(benchmark) -> None:
    """Missing rows remain explicit and the method cannot own the stance mask."""
    source, target = _source(benchmark), _target(benchmark, SELECTED)
    output = _view(benchmark, source, target, rows=(0, 2, 3))
    report = benchmark.evaluate_neutral_retargeting(
        (source,), target, (_method(benchmark, "soma", target, output, output),)
    )
    coverage = report["methods"]["soma"]["clips"][0]["native_mechanics"]["coverage"]
    assert coverage["matched_fraction"] == 0.75
    assert coverage["complete"] is False
    assert report["contract"]["time"]["resampling"] is False
    assert "stance_active" not in {field.name for field in dataclasses.fields(benchmark.RetargetEvaluationView)}
    changed_clock = _view(benchmark, source, target, dt=1.0 / 24.0)
    with pytest.raises(ValueError, match="resampling is forbidden"):
        benchmark.evaluate_neutral_retargeting(
            (source,), target, (_method(benchmark, "soma", target, changed_clock, changed_clock),)
        )


def test_native_view_cannot_masquerade_as_selected_target(benchmark) -> None:
    """The selected deployment view must carry the selected target identity."""
    source = _source(benchmark)
    native_target, selected_target = _target(benchmark, NATIVE), _target(benchmark, SELECTED)
    native = _view(benchmark, source, native_target)
    with pytest.raises(ValueError, match="declared target mechanics"):
        benchmark.evaluate_neutral_retargeting(
            (source,), selected_target, (_method(benchmark, "soma", native_target, native, native),)
        )
