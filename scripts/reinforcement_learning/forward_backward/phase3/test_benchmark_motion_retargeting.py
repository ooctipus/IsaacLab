# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the intrinsic four-cell motion-retargeting benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).with_name("benchmark_motion_retargeting.py")


@pytest.fixture(scope="module")
def benchmark():
    """Load analysis helpers without importing the environment composition root."""
    spec = importlib.util.spec_from_file_location("benchmark_motion_retargeting_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quality() -> SimpleNamespace:
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _QUALITY_NAMES

    names = tuple(_QUALITY_NAMES)
    values = torch.full((3, len(names)), float("nan"), dtype=torch.float32)
    for name, column in zip(names, values.T, strict=True):
        if name == "trajectory_route":
            column.copy_(torch.tensor((0.0, 1.0, 1.0)))
        elif name == "accepted":
            column.copy_(torch.tensor((1.0, 1.0, 0.0)))
        elif name == "base_priority":
            column.fill_(1.0)
        elif name.endswith("_feasible"):
            column[1:] = torch.tensor((1.0, 0.0))
        else:
            column[1:] = torch.tensor((0.1, 0.2))
    for name in names:
        if name.startswith("dynamics_"):
            values[2, names.index(name)] = float("nan")
    return SimpleNamespace(scope="sequence", names=names, values=values)


def test_quality_summary_consumes_builder_schema_dynamically(benchmark) -> None:
    """All non-semantic columns come directly from the table-owned quality schema."""
    quality = _quality()
    summary, stage_quality, accepted, trajectory = benchmark._quality_summary(
        quality, ("stored", "accepted", "rejected")
    )

    assert summary["accepted_clips"] == 2
    assert summary["rejected_clips"] == 1
    assert summary["stored_coordinate_clips"] == 1
    assert summary["trajectory_clips"] == 2
    assert set(summary["metrics"]) == set(quality.names) - {"trajectory_route", "accepted"}
    route_metric = next(
        name
        for name in quality.names
        if name not in ("trajectory_route", "accepted")
        and not torch.isfinite(quality.values[0, quality.names.index(name)])
    )
    assert summary["metrics"][route_metric]["count"] == 2
    assert summary["metrics"][route_metric]["max"] == pytest.approx(0.2)
    assert accepted.tolist() == [True, True, False]
    assert trajectory.tolist() == [False, True, True]
    assert stage_quality is None

    quality.names = (*quality.names, "future_metric")
    quality.values = torch.cat((quality.values, quality.values.new_tensor(((float("nan"),), (3.0,), (4.0,)))), dim=1)
    summary, _, _, _ = benchmark._quality_summary(quality, ("stored", "accepted", "rejected"))
    assert summary["metrics"]["future_metric"]["max"] == pytest.approx(4.0)


def test_quality_summary_partitions_complete_inspection_stage_schema(benchmark) -> None:
    """Inspection stages are complete, namespaced, and separate from final quality."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _TRAJECTORY_INSPECTION_QUALITY_PREFIX,
        _TRAJECTORY_INSPECTION_STAGE_NAMES,
        _TRAJECTORY_METRIC_NAMES,
    )

    quality = _quality()
    stage_names = tuple(
        f"{_TRAJECTORY_INSPECTION_QUALITY_PREFIX}{stage}/{metric}"
        for stage in _TRAJECTORY_INSPECTION_STAGE_NAMES
        for metric in _TRAJECTORY_METRIC_NAMES
    )
    stage_values = torch.arange(3 * len(stage_names), dtype=torch.float32).view(3, -1)
    contact_column = stage_names.index(f"{_TRAJECTORY_INSPECTION_QUALITY_PREFIX}frame_seed/contact_gap_max_m")
    stage_values[0, contact_column] = float("nan")
    quality.names = (*quality.names, *stage_names)
    quality.values = torch.cat((quality.values, stage_values), dim=1)

    summary, stage_quality, accepted, trajectory = benchmark._quality_summary(
        quality, ("stored", "accepted", "rejected")
    )

    assert not set(stage_names) & set(summary["metrics"])
    assert tuple(stage_quality) == _TRAJECTORY_INSPECTION_STAGE_NAMES
    assert stage_quality["frame_seed"]["metrics"]["contact_gap_max_m"]["missing_clips"] == 1
    assert accepted.tolist() == [True, True, False]
    assert trajectory.tolist() == [False, True, True]

    quality.names = quality.names[:-1]
    quality.values = quality.values[:, :-1]
    with pytest.raises(ValueError, match="inspection stage quality"):
        benchmark._quality_summary(quality, ("stored", "accepted", "rejected"))


def test_quality_summary_rejects_nonbinary_semantics(benchmark) -> None:
    """Route and acceptance remain the only benchmark-owned quality semantics."""
    quality = _quality()
    quality.values[1, quality.names.index("trajectory_route")] = 0.5
    with pytest.raises(ValueError, match="finite binary"):
        benchmark._quality_summary(quality, ("stored", "accepted", "rejected"))


def test_root_motion_never_differences_across_clip_boundaries(benchmark) -> None:
    """Motion rates and accelerations preserve independent clip clocks."""
    yaw = torch.tensor((0.0, torch.pi / 2.0, torch.pi, 0.0, -torch.pi / 2.0))
    root_pose = torch.zeros(5, 1, 7)
    root_pose[:, 0, 0] = torch.tensor((0.0, 1.0, 2.0, 100.0, 101.0))
    root_pose[:, 0, 5] = torch.sin(yaw / 2.0)
    root_pose[:, 0, 6] = torch.cos(yaw / 2.0)
    root_velocity = torch.zeros(5, 1, 6)
    root_velocity[:, 0, 0] = torch.tensor((2.0, 2.0, 2.0, 1.0, 1.0))
    view = SimpleNamespace(
        sequences=SimpleNamespace(
            sequence_count=2,
            frame_count=5,
            offsets=torch.tensor((0, 3, 5), dtype=torch.int64),
            state_indices=None,
            frame_dt=torch.tensor((0.5, 1.0)),
        ),
        state_bank=SimpleNamespace(
            root_pose=root_pose,
            root_velocity=root_velocity,
            joint_velocity=torch.tensor(((0.0,), (1.0,), (2.0,), (10.0,), (11.0,))),
        ),
    )

    summary, mean_speeds = benchmark._root_motion(view, ("turn", "return"))

    assert mean_speeds == pytest.approx((2.0, 1.0))
    assert summary["horizontal_path_m"]["max"] == pytest.approx(2.0)
    assert summary["net_yaw_rad"]["max"] == pytest.approx(torch.pi)
    assert summary["derived_root_speed_mps"]["max"] == pytest.approx(2.0)
    assert summary["root_linear_acceleration_mps2"]["max"] == pytest.approx(0.0)
    assert summary["joint_acceleration_abs_radps2"]["max"] == pytest.approx(2.0)


def test_landmark_summary_quotients_translation_and_heading(benchmark) -> None:
    """Heading-aligned landmark error removes only planar rigid gauge freedoms."""
    target = torch.tensor(
        (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
        )
    )
    solved = target.clone()
    solved[..., 0] = -target[..., 1] + 10.0
    solved[..., 1] = target[..., 0] + 2.0
    view = SimpleNamespace(
        sequences=SimpleNamespace(state_indices=None),
        points=(
            SimpleNamespace(name="target_landmarks", points=target),
            SimpleNamespace(name="solved_robot_landmarks", points=solved),
        ),
    )

    summary = benchmark._landmark_summary(view)

    assert summary["world_error_m"]["max"] > 1.0
    assert summary["root_relative_error_m"]["max"] > 1.0
    assert summary["heading_aligned_error_m"]["max"] == pytest.approx(0.0, abs=1.0e-6)


def test_contact_summary_preserves_intervals_at_clip_boundaries(benchmark) -> None:
    """A contact spanning two adjacent stored clips remains two physical intervals."""
    left = torch.tensor((False, True, True, True, True, False))
    right = torch.tensor((False, True, False, False, True, False))
    valid = torch.cat((left[:, None].expand(-1, 3), right[:, None].expand(-1, 3)), dim=1)
    view = SimpleNamespace(
        sequences=SimpleNamespace(
            offsets=torch.tensor((0, 3, 6), dtype=torch.int64),
            state_indices=None,
            frame_dt=torch.tensor((0.1, 0.2)),
        ),
        points=(SimpleNamespace(name="contact_points", valid=valid),),
    )
    patches = (
        SimpleNamespace(channel="left", points_per_body=3),
        SimpleNamespace(channel="right", points_per_body=3),
    )

    summary, patterns = benchmark._contact_summary(view, patches, ("first", "second"))

    assert patterns == ["simultaneous", "simultaneous"]
    left_summary = summary["channels"]["left"]
    assert left_summary["interval_count"] == 2
    assert left_summary["interval_duration_s"]["mean"] == pytest.approx(0.3)
    assert left_summary["interval_duration_s"]["max"] == pytest.approx(0.4)
    assert left_summary["clip_coverage_fraction"]["mean"] == pytest.approx(2.0 / 3.0)


def test_contact_summary_rejects_stale_anchor_alias(benchmark) -> None:
    """Only the builder-owned contact_points evidence name carries contact activity."""
    view = SimpleNamespace(
        sequences=SimpleNamespace(
            offsets=torch.tensor((0, 2), dtype=torch.int64),
            state_indices=None,
            frame_dt=torch.tensor((0.1,)),
        ),
        points=(SimpleNamespace(name="contact_anchors", valid=torch.ones((2, 3), dtype=torch.bool)),),
    )
    patches = (SimpleNamespace(channel="left", points_per_body=3),)

    summary, patterns = benchmark._contact_summary(view, patches, ("clip",))

    assert summary is None
    assert patterns == ["not_applicable"]


def test_contact_summary_requires_one_activity_mask_per_patch(benchmark) -> None:
    """Per-point validity cannot disagree inside one configured support patch."""
    valid = torch.tensor(((True, True, False), (True, True, True)))
    view = SimpleNamespace(
        sequences=SimpleNamespace(
            offsets=torch.tensor((0, 2), dtype=torch.int64),
            state_indices=None,
            frame_dt=torch.tensor((0.1,)),
        ),
        points=(SimpleNamespace(name="contact_points", valid=valid),),
    )
    patches = (SimpleNamespace(channel="left", points_per_body=3),)

    with pytest.raises(ValueError, match="repeat one source-owned activity mask"):
        benchmark._contact_summary(view, patches, ("clip",))


def test_rejection_summary_uses_only_comparison_dimensions(benchmark) -> None:
    """Rejection analysis stays limited to route, skeleton, speed, and contact."""
    index = SimpleNamespace(
        clips=(
            SimpleNamespace(clip_id="one", skeleton_id=0),
            SimpleNamespace(clip_id="two", skeleton_id=0),
            SimpleNamespace(clip_id="three", skeleton_id=1),
        )
    )
    accepted = torch.tensor((True, False, True))
    trajectory = torch.tensor((False, True, True))
    summary = benchmark._rejection_summary(
        index.clips, accepted, trajectory, [0.0, 0.5, 2.0], ["none", "alternating", "none"]
    )

    assert set(summary["by"]) == {"route", "skeleton", "speed", "contact"}
    assert summary["by"]["route"]["trajectory"]["accepted_fraction"] == pytest.approx(0.5)
    assert summary["rejected_clips"] == [
        {"clip_id": "two", "route": "trajectory", "speed": "ordinary", "contact": "alternating"}
    ]


def test_full_report_defaults_to_every_declared_source_sequence(benchmark) -> None:
    """The comparison defaults to the complete split without mutating production scope."""
    assert benchmark._build_report.__defaults__ == (None,)
    source = MODULE_PATH.read_text()
    assert "table_cfg.inspection_limit" not in source
    assert "sequence_limit=selected_count" in source


def test_report_builds_and_counts_only_the_requested_prefix(benchmark, monkeypatch, tmp_path: Path) -> None:
    """A bring-up cap constrains construction, analysis ids, frame counts, and throughput together."""
    clips = (
        SimpleNamespace(clip_id="first", frame_count=2, skeleton_id=0),
        SimpleNamespace(clip_id="second", frame_count=3, skeleton_id=0),
        SimpleNamespace(clip_id="third", frame_count=7, skeleton_id=1),
    )
    index = SimpleNamespace(clips=clips, clip_ids=tuple(clip.clip_id for clip in clips), offsets=(0, 2, 5, 12))
    split = SimpleNamespace(name="evaluation")
    clip_source = SimpleNamespace(inspect=lambda: index, close=lambda: None)
    source_cfg = SimpleNamespace(
        identifier="fake_motion",
        train=split,
        evaluation=split,
        open_split=lambda *_args: clip_source,
    )
    calls = {}

    from isaaclab_tasks.core.multi_task.kinematics import IKTrajectorySolver

    solver_calls = []

    def solve(_solver, *_args, **kwargs):
        solver_calls.append(kwargs)

    monkeypatch.setattr(IKTrajectorySolver, "solve", solve)

    def build_inspection_view(*_args, sequence_limit):
        calls["sequence_limit"] = sequence_limit
        IKTrajectorySolver.solve(object(), residual_activity=None, inequalities=object())
        IKTrajectorySolver.solve(object(), residual_activity=object(), inequalities=None)
        IKTrajectorySolver.solve(object(), residual_activity=object(), inequalities=object())
        return SimpleNamespace(quality=object(), sequences=SimpleNamespace(frame_count=5))

    table_cfg = SimpleNamespace(
        source=source_cfg,
        target_kinematics=SimpleNamespace(contact_patches=(), calibration=None),
        build_inspection_view=build_inspection_view,
    )
    cfg = SimpleNamespace(
        commands=SimpleNamespace(motion=SimpleNamespace(task_table=table_cfg)),
        scene=object(),
    )
    monkeypatch.setattr(benchmark, "resolve_presets", lambda *_args, **_kwargs: cfg)
    clock = iter((10.0, 12.0, 20.0, 21.0))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        benchmark,
        "_quality_summary",
        lambda _quality, clip_ids: (
            {"accepted_clips": 1, "rejected_clips": 1},
            None,
            torch.tensor((True, False)),
            torch.tensor((False, True)),
        ),
    )
    monkeypatch.setattr(benchmark, "_root_motion", lambda _view, _clip_ids: ({}, [0.0, 0.0]))
    monkeypatch.setattr(benchmark, "_landmark_summary", lambda _view: None)
    monkeypatch.setattr(benchmark, "_contact_summary", lambda _view, _patches, _clip_ids: (None, ["none", "none"]))

    report = benchmark._build_report("g1", "cmu", tmp_path, None, "evaluation", torch.device("cpu"), inspection_limit=2)

    assert not hasattr(table_cfg, "inspection_limit")
    assert calls["sequence_limit"] == 2
    assert report["run"] == {"scope": "limited_inspection", "accepting": False}
    assert report["corpus"] == {
        "source_identifier": "fake_motion",
        "split": "evaluation",
        "declared_clips": 3,
        "declared_frames": 12,
        "inspection_limit": 2,
        "clips": 2,
        "input_frames": 5,
        "output_frames": 5,
    }
    assert report["performance"]["input_frames_per_construction_second"] == pytest.approx(2.5)
    assert report["performance"]["trajectory_nonlinear_iterations"] == {"source": 1, "contact": 2, "total": 3}
    assert report["performance"]["retarget_stages"]["source"]["calls"] == 1
    assert report["performance"]["retarget_stages"]["contact"]["calls"] == 2
    assert report["performance"]["retarget_stages"]["total"]["iterations"] == 3
    assert len(solver_calls) == 3
    assert IKTrajectorySolver.solve is solve


def test_nonlinear_iteration_counter_restores_solver_after_failure(benchmark, monkeypatch) -> None:
    """The benchmark wrapper never leaks diagnostic state into later production work."""
    from isaaclab_tasks.core.multi_task.kinematics import IKTrajectorySolver

    def solve(_solver, *_args, **_kwargs):
        return None

    monkeypatch.setattr(IKTrajectorySolver, "solve", solve)
    with pytest.raises(RuntimeError, match="construction failed"):
        with benchmark.count_trajectory_nonlinear_iterations() as counts:
            IKTrajectorySolver.solve(object(), residual_activity=None, inequalities=None)
            raise RuntimeError("construction failed")

    assert counts.report() == {"source": 1, "contact": 0, "total": 1}
    assert IKTrajectorySolver.solve is solve


def test_solver_instrumentation_classifies_stages_and_restores_methods(benchmark, monkeypatch, capsys) -> None:
    """Benchmark-only wrappers count every solve class and restore all methods."""
    from newton import ik

    from isaaclab_tasks.core.multi_task.kinematics import IKTrajectorySolver

    def frame_solve(_solver, *_args, **_kwargs):
        return SimpleNamespace(iterations=7)

    def frame_step(_solver, *_args, **_kwargs):
        return None

    def trajectory_solve(_solver, *_args, **_kwargs):
        return None

    monkeypatch.setattr(ik.IKSolver, "solve", frame_solve)
    monkeypatch.setattr(ik.IKSolver, "step", frame_step)
    monkeypatch.setattr(IKTrajectorySolver, "solve", trajectory_solve)
    with benchmark.count_trajectory_nonlinear_iterations(progress_interval_seconds=0.0) as counts:
        ik.IKSolver.solve(SimpleNamespace(n_seeds=64), None, None, max_iterations=200)
        ik.IKSolver.step(SimpleNamespace(n_seeds=1), None, None, iterations=24)
        IKTrajectorySolver.solve(object(), residual_activity=None, inequalities=object(), feasibility_only=True)
        IKTrajectorySolver.solve(object(), residual_activity=None, inequalities=object())
        IKTrajectorySolver.solve(object(), residual_activity=object(), inequalities=object())

    stages = counts.stage_report()
    assert counts.report() == {"source": 2, "contact": 1, "total": 3}
    assert stages["frame_global"]["calls"] == 1
    assert stages["frame_global"]["iterations"] == 7
    assert stages["frame_local"]["calls"] == 1
    assert stages["frame_local"]["iterations"] == 24
    assert stages["feasibility"]["iterations"] == 1
    assert stages["source"]["iterations"] == 1
    assert stages["contact"]["iterations"] == 1
    assert stages["total"]["calls"] == 5
    stage_names = ("frame_global", "frame_local", "feasibility", "source", "contact", "total")
    assert all(stages[name]["gpu_seconds"] is None for name in stage_names)
    assert "[motion benchmark] stage=" in capsys.readouterr().err
    assert ik.IKSolver.solve is frame_solve
    assert ik.IKSolver.step is frame_step
    assert IKTrajectorySolver.solve is trajectory_solve


def test_main_requires_strict_json_for_file_and_stdout(benchmark, monkeypatch, tmp_path: Path, capsys) -> None:
    """Persisted evidence and stdout must reject non-standard JSON numbers."""
    output = tmp_path / "report.json"
    report = {
        "quality": {"accepted_clips": 1, "rejected_clips": 0},
        "performance": {"construction_seconds": 1.0},
    }
    monkeypatch.setattr(
        benchmark,
        "_parse_args",
        lambda: SimpleNamespace(
            robot="smpl",
            source="cmu",
            source_artifact_root=tmp_path,
            target_artifact_root=tmp_path / "target",
            motion_split="evaluation",
            inspection_limit=2,
            device="cpu",
            output=output,
        ),
    )
    build_call = {}

    def build_report(*args, **kwargs):
        build_call["args"] = args
        build_call["kwargs"] = kwargs
        return report

    monkeypatch.setattr(benchmark, "_build_report", build_report)
    original_dumps = benchmark.json.dumps
    calls = []

    def strict_dumps(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(benchmark.json, "dumps", strict_dumps)
    benchmark.main()

    assert len(calls) == 2
    assert all(call.get("allow_nan") is False for call in calls)
    assert build_call["kwargs"] == {"inspection_limit": 2}
    assert benchmark.json.loads(output.read_text()) == report
    assert benchmark.json.loads(capsys.readouterr().out)["quality"] == report["quality"]


def test_cli_exposes_independent_robot_and_source_axes(benchmark, tmp_path: Path) -> None:
    """The producer does not encode composite dataset-robot product names."""
    args = benchmark._parse_args(
        [
            "--robot",
            "smpl",
            "--source",
            "lafan",
            "--source_artifact_root",
            str(tmp_path),
            "--target_artifact_root",
            str(tmp_path / "target"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.robot == "smpl"
    assert args.source == "lafan"
    assert args.target_artifact_root == tmp_path / "target"
    assert args.motion_split == "evaluation"
    assert args.inspection_limit is None
    source = MODULE_PATH.read_text()
    assert all(name not in source for name in ("g1_cmu", "smpl_cmu", "g1_lafan", "smpl_lafan"))


def test_cli_accepts_only_positive_inspection_limit(benchmark, tmp_path: Path, capsys) -> None:
    """The optional bring-up cap is positive and otherwise leaves full-corpus behavior selected."""
    base = [
        "--robot",
        "g1",
        "--source",
        "cmu",
        "--source_artifact_root",
        str(tmp_path),
        "--output",
        str(tmp_path / "report.json"),
    ]
    assert benchmark._parse_args([*base, "--inspection_limit", "1"]).inspection_limit == 1
    with pytest.raises(SystemExit) as error:
        benchmark._parse_args([*base, "--inspection_limit", "0"])
    assert error.value.code == 2
    assert "--inspection_limit must be positive" in capsys.readouterr().err
