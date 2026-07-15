# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the released-LAFAN semantic comparison oracle."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).with_name("benchmark_lafan_retargeting.py")


@pytest.fixture(scope="module")
def benchmark():
    """Load only the pure comparator; table-building imports remain lazy."""
    spec = importlib.util.spec_from_file_location("benchmark_lafan_retargeting_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_clip(benchmark, clip_id: str = "dance1_subject1"):
    time = torch.arange(4, dtype=torch.float64)
    pelvis = torch.stack((0.2 * time, 0.1 * time, torch.ones_like(time)), dim=-1)
    left_ankle = pelvis + pelvis.new_tensor((-0.25, 0.15, -0.9))
    torso = pelvis + pelvis.new_tensor((0.05, 0.0, 0.5))
    landmarks = torch.stack((pelvis, left_ankle, torso), dim=1)
    support = torch.tensor(
        [
            ((0.0, 0.1, -0.01), (0.0, -0.1, 0.0)),
            ((0.0, 0.1, -0.01), (0.0, -0.1, 0.0)),
            ((0.0, 0.1, -0.01), (0.0, -0.1, 0.0)),
            ((0.2, 0.1, 0.1), (0.0, -0.1, 0.0)),
        ],
        dtype=torch.float64,
    )
    stance = torch.tensor(((True, False), (True, False), (True, False), (False, False)))
    rotation_roles = ("pelvis", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
    rotation = torch.zeros((4, len(rotation_roles), 4), dtype=torch.float64)
    rotation[..., 3] = 1.0
    joint_position = torch.zeros((4, 2), dtype=torch.float64)
    joint_velocity = torch.zeros_like(joint_position)
    return benchmark.SemanticTrajectoryClip(
        clip_id=clip_id,
        frame_dt_s=1.0 / 30.0,
        landmark_roles=("pelvis", "left_ankle", "torso"),
        semantic_edges=(("pelvis", "left_ankle"), ("pelvis", "torso")),
        landmark_position_m=landmarks,
        rotation_roles=rotation_roles,
        rotation_xyzw=rotation,
        support_roles=("left_toe", "right_toe"),
        support_position_m=support,
        stance_active=stance,
        joint_names=("hip", "knee"),
        joint_position_rad=joint_position,
        joint_velocity_rad_s=joint_velocity,
        joint_lower_limit_rad=torch.tensor((-1.0, -1.0), dtype=torch.float64),
        joint_upper_limit_rad=torch.tensor((1.0, 1.0), dtype=torch.float64),
        joint_velocity_limit_rad_s=torch.tensor((2.0, 2.0), dtype=torch.float64),
    )


def _inverse_rigid(points: torch.Tensor, yaw: float, translation: torch.Tensor) -> torch.Tensor:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = points.new_tensor(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    return (points - translation) @ rotation


def test_build_performance_contract_requires_positive_complete_fields(benchmark) -> None:
    """The full production build record must retain positive rates and exact CUDA peak semantics."""
    report = benchmark._build_performance(
        scope="full_build",
        included_stages=("decode", "solve"),
        wall_seconds=2.0,
        input_clip_count=10,
        input_frame_count=100,
        output_clip_count=8,
        output_frame_count=80,
        device="cuda:0",
        cuda_allocated_bytes_before=100,
        cuda_peak_allocated_bytes=160,
        trajectory_nonlinear_iterations={"source": 3, "contact": 2, "total": 5},
    )
    required = {
        "scope",
        "included_stages",
        "device",
        "input_clip_count",
        "input_frame_count",
        "output_clip_count",
        "output_frame_count",
        "wall_seconds",
        "input_frames_per_second",
        "output_frames_per_second",
        "cuda_allocated_bytes_before",
        "cuda_peak_allocated_bytes",
        "cuda_peak_incremental_allocated_bytes",
        "trajectory_nonlinear_iterations",
    }
    assert set(report) == required
    assert report["wall_seconds"] == 2.0
    assert report["input_frames_per_second"] == 50.0
    assert report["output_frames_per_second"] == 40.0
    assert report["cuda_peak_incremental_allocated_bytes"] == 60
    assert report["trajectory_nonlinear_iterations"] == {"source": 3, "contact": 2, "total": 5}
    with pytest.raises(ValueError, match="positive duration"):
        benchmark._build_performance(
            scope="full_build",
            included_stages=("decode",),
            wall_seconds=0.0,
            input_clip_count=1,
            input_frame_count=1,
            output_clip_count=1,
            output_frame_count=1,
            device="cpu",
            cuda_allocated_bytes_before=None,
            cuda_peak_allocated_bytes=None,
            trajectory_nonlinear_iterations={"source": 0, "contact": 0, "total": 0},
        )


def test_max_clips_drives_id_aligned_inspection_prefix_builds(benchmark) -> None:
    """The benchmark builds the raw prefix and enough released rows to cover the same IDs."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex

    tree = ast.parse(inspect.getsource(benchmark._build_report))
    inspection_calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build_inspection_view"
        ),
        key=lambda node: node.lineno,
    )
    assert len(inspection_calls) == 2
    assert [
        ast.unparse(next(keyword.value for keyword in call.keywords if keyword.arg == "sequence_limit"))
        for call in inspection_calls
    ] == ["selected_count", "1 + max(released_sequence_indices)"]

    paired_prefix_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_paired_prefix_indices"
    )
    assert [ast.unparse(argument) for argument in paired_prefix_call.args] == [
        "raw_index",
        "released_index",
        "max_clips",
    ]

    performance_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_build_performance"
    )
    performance_fields = {keyword.arg: ast.unparse(keyword.value) for keyword in performance_call.keywords}
    assert performance_fields["input_clip_count"] == "selected_count"
    assert performance_fields["input_frame_count"] == "raw_index.offsets[selected_count]"

    index = MotionClipIndex(
        source_content_sha256="0" * 64,
        skeleton_identity_sha256s=("1" * 64,),
        clips=(
            MotionClipIndex.Clip("clip_0", 2, 10.0, "2" * 64, 0),
            MotionClipIndex.Clip("clip_1", 3, 20.0, "3" * 64, 0),
        ),
    )
    prefix_view = SimpleNamespace(
        sequences=SimpleNamespace(
            offsets=torch.tensor((0, 2), dtype=torch.int64),
            frame_dt=torch.tensor((0.1,), dtype=torch.float32),
            sequence_count=1,
        )
    )
    benchmark._validate_inspection_index(prefix_view, index)


def test_raw_prefix_maps_to_differently_ordered_released_rows(benchmark) -> None:
    """Equal corpus IDs need not share one native source order."""
    raw = SimpleNamespace(
        clip_ids=("dance", "walk", "run"),
        clips=(
            SimpleNamespace(clip_id="dance", frame_count=4, source_fps=30.0),
            SimpleNamespace(clip_id="walk", frame_count=5, source_fps=30.0),
            SimpleNamespace(clip_id="run", frame_count=6, source_fps=30.0),
        ),
    )
    released = SimpleNamespace(
        clip_ids=("run", "dance", "walk"),
        clips=(
            SimpleNamespace(clip_id="run", frame_count=6, source_fps=30.0),
            SimpleNamespace(clip_id="dance", frame_count=4, source_fps=30.0),
            SimpleNamespace(clip_id="walk", frame_count=5, source_fps=30.0),
        ),
    )

    raw_rows, released_rows = benchmark._paired_prefix_indices(raw, released, max_clips=2)

    assert raw_rows == (0, 1)
    assert released_rows == (1, 2)

    for changed in (
        SimpleNamespace(clip_id="dance", frame_count=5, source_fps=30.0),
        SimpleNamespace(clip_id="dance", frame_count=4, source_fps=24.0),
    ):
        mismatched = SimpleNamespace(
            clip_ids=released.clip_ids,
            clips=(released.clips[0], changed, released.clips[2]),
        )
        with pytest.raises(ValueError, match="exact released frame/time correspondence"):
            benchmark._paired_prefix_indices(raw, mismatched, max_clips=1)


def test_oracle_quotients_only_one_constant_yaw_and_translation_per_clip(benchmark) -> None:
    released = _reference_clip(benchmark)
    yaw = 0.7
    translation = torch.tensor((1.2, -0.8, 0.35), dtype=torch.float64)
    candidate_landmarks = _inverse_rigid(released.landmark_position_m, yaw, translation)
    candidate_support = _inverse_rigid(released.support_position_m, yaw, translation)
    candidate_rotation = (
        released.rotation_xyzw.new_tensor((0.0, 0.0, -math.sin(0.5 * yaw), math.cos(0.5 * yaw)))
        .expand_as(released.rotation_xyzw)
        .clone()
    )
    candidate = replace(
        released,
        landmark_roles=("torso", "pelvis", "left_ankle"),
        landmark_position_m=candidate_landmarks[:, (2, 0, 1)],
        rotation_roles=tuple(reversed(released.rotation_roles)),
        rotation_xyzw=candidate_rotation.flip(1),
        support_roles=("right_toe", "left_toe"),
        support_position_m=candidate_support[:, (1, 0)],
        stance_active=released.stance_active[:, (1, 0)],
    )

    report = benchmark.compare_lafan_retargeting((candidate,), (released,))

    assert report["comparison_policy"]["scale_fitted"] is False
    assert report["comparison_policy"]["time_warped_or_resampled"] is False
    assert report["aggregate"]["landmark_error_m"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert report["aggregate"]["semantic_edge_angle_rad"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    orientation = report["aggregate"]["selected_orientation_error_rad"]
    assert orientation["all"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert tuple(orientation["by_role"]) == released.rotation_roles
    assert all(metric["max"] == pytest.approx(0.0, abs=1.0e-12) for metric in orientation["by_role"].values())
    assert report["aggregate"]["stance"]["anchor_error_m"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert report["clips"][0]["alignment"]["yaw_rad"] == pytest.approx(yaw)
    assert report["clips"][0]["alignment"]["translation_m"] == pytest.approx(translation.tolist())


def test_oracle_does_not_hide_scale_or_time_correspondence_changes(benchmark) -> None:
    released = _reference_clip(benchmark)
    scaled = replace(
        released,
        landmark_position_m=2.0 * released.landmark_position_m,
        support_position_m=2.0 * released.support_position_m,
    )
    report = benchmark.compare_lafan_retargeting((scaled,), (released,))

    assert report["aggregate"]["landmark_error_m"]["mean"] > 0.1
    assert report["aggregate"]["semantic_edge_angle_rad"]["max"] == pytest.approx(0.0, abs=1.0e-12)

    quantized = benchmark.compare_lafan_retargeting((replace(released, frame_dt_s=0.033333),), (released,))
    assert quantized["comparison_policy"]["frame_period_absolute_tolerance_s"] == 0.5e-6
    assert quantized["clips"][0]["candidate_frame_dt_s"] == 0.033333
    assert quantized["clips"][0]["released_frame_dt_s"] == released.frame_dt_s

    with pytest.raises(ValueError, match="frame periods.*resampling is forbidden"):
        benchmark.compare_lafan_retargeting((replace(released, frame_dt_s=1.0 / 50.0),), (released,))

    shortened = replace(
        released,
        landmark_position_m=released.landmark_position_m[:-1],
        rotation_xyzw=released.rotation_xyzw[:-1],
        support_position_m=released.support_position_m[:-1],
        stance_active=released.stance_active[:-1],
        joint_position_rad=released.joint_position_rad[:-1],
        joint_velocity_rad_s=released.joint_velocity_rad_s[:-1],
    )
    with pytest.raises(ValueError, match="frame counts; time warp is forbidden"):
        benchmark.compare_lafan_retargeting((shortened,), (released,))


def test_oracle_pairs_exact_clip_ids_and_common_semantic_geometry(benchmark) -> None:
    released = _reference_clip(benchmark)
    extra_position = released.landmark_position_m[:, 2:3] + released.landmark_position_m.new_tensor((0.0, 0.0, 0.25))
    candidate = replace(
        released,
        landmark_roles=(*released.landmark_roles, "head"),
        semantic_edges=(*released.semantic_edges, ("torso", "head")),
        landmark_position_m=torch.cat((released.landmark_position_m, extra_position), dim=1),
    )
    report = benchmark.compare_lafan_retargeting((candidate,), (released,))

    assert report["clips"][0]["common_semantics"]["landmark_roles"] == list(released.landmark_roles)
    assert report["clips"][0]["common_semantics"]["edges"] == [list(edge) for edge in released.semantic_edges]
    assert report["clips"][0]["common_semantics"]["rotation_roles"] == list(released.rotation_roles)

    with pytest.raises(ValueError, match="identical clip IDs"):
        benchmark.compare_lafan_retargeting((replace(candidate, clip_id="walk1_subject1"),), (released,))
    with pytest.raises(ValueError, match="selected orientation roles"):
        benchmark.compare_lafan_retargeting(
            (
                replace(
                    candidate,
                    rotation_roles=candidate.rotation_roles[:-1],
                    rotation_xyzw=candidate.rotation_xyzw[:, :-1],
                ),
            ),
            (released,),
        )


def test_oracle_reports_geometry_stance_penetration_and_limits_separately(benchmark) -> None:
    released = _reference_clip(benchmark)
    landmarks = released.landmark_position_m.clone()
    landmarks[:, 1, 2] += 0.2
    landmarks[:, 2, 2] -= 0.2
    support = released.support_position_m.clone()
    support[:3, 0] = torch.tensor(((0.2, 0.1, -0.05), (0.3, 0.1, -0.05), (0.4, 0.1, -0.05)), dtype=torch.float64)
    rotation = released.rotation_xyzw.clone()
    rotation[:, 0] = released.rotation_xyzw.new_tensor((math.sin(0.1), 0.0, 0.0, math.cos(0.1)))
    joint_position = released.joint_position_rad.clone()
    joint_velocity = released.joint_velocity_rad_s.clone()
    joint_position[0, 0] = 1.5
    joint_velocity[1, 1] = 3.0
    candidate = replace(
        released,
        landmark_position_m=landmarks,
        rotation_xyzw=rotation,
        support_position_m=support,
        joint_position_rad=joint_position,
        joint_velocity_rad_s=joint_velocity,
    )

    report = benchmark.compare_lafan_retargeting((candidate,), (released,))
    metrics = report["clips"][0]

    assert metrics["landmark_error_m"]["all"]["max"] > 0.0
    assert metrics["semantic_edge_angle_rad"]["all"]["max"] > 0.0
    orientation = metrics["selected_orientation_error_rad"]
    assert orientation["all"]["max"] > 0.1
    assert orientation["by_role"]["pelvis"]["max"] > 0.1
    assert all(orientation["by_role"][role]["max"] == pytest.approx(0.0) for role in released.rotation_roles[1:])
    assert metrics["stance"]["interval_count"] == 1
    assert metrics["stance"]["anchor_error_m"]["max"] > 0.2
    assert metrics["stance"]["candidate_slip_m"]["max"] == pytest.approx(0.2)
    assert metrics["stance"]["released_reference_slip_m"]["max"] == pytest.approx(0.0)
    assert metrics["frame_max_support_penetration_m"]["candidate"]["max"] == pytest.approx(0.05)
    assert metrics["frame_max_support_penetration_m"]["released_reference"]["max"] == pytest.approx(0.01)
    candidate_limits = metrics["hard_limits"]["candidate"]
    reference_limits = metrics["hard_limits"]["released_reference"]
    assert candidate_limits["position"]["violating_samples"] == 1
    assert candidate_limits["position"]["excess_rad"]["max"] == pytest.approx(0.5)
    assert candidate_limits["velocity"]["violating_samples"] == 1
    assert candidate_limits["velocity"]["excess_rad_s"]["max"] == pytest.approx(1.0)
    assert reference_limits["position"]["violating_samples"] == 0
    assert reference_limits["velocity"]["violating_samples"] == 0


def test_oracle_rejects_ambiguous_stance_labels(benchmark) -> None:
    released = _reference_clip(benchmark)
    candidate_stance = released.stance_active.clone()
    candidate_stance[0, 0] = False

    with pytest.raises(ValueError, match="one explicit stance mask"):
        benchmark.compare_lafan_retargeting((replace(released, stance_active=candidate_stance),), (released,))


def test_inspection_adapter_keeps_rejected_sequences_and_reuses_planned_fk_workspace(benchmark, monkeypatch) -> None:
    """Production rejection must remain metadata, not remove comparison candidates."""

    class KinematicView:
        joint_q_default = torch.tensor((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))

        def joint_q_into(self, state_bank, state_rows, out) -> None:
            out.copy_(KinematicView.joint_q_default)
            out[:, :7] = state_bank.root_pose[state_rows, 0]
            out[:, 7] = state_bank.joint_position[state_rows, 0]

    class Reference:
        device = "cpu"
        model = SimpleNamespace(joint_coord_count=8, joint_dof_count=7, body_count=3)

        def __init__(self) -> None:
            self.batch_sizes = []
            self.storage_pointers = []

        def eval_fk_batched_torch(self, joint_q, _joint_qd, body_q, body_qd):
            self.batch_sizes.append(joint_q.shape[0])
            self.storage_pointers.append(joint_q.untyped_storage().data_ptr())
            body_q.zero_()
            body_q[..., 6] = 1.0
            body_q[:, 0, :3] = joint_q[:, :3]
            body_q[:, 0, 3:] = joint_q[:, 3:7]
            body_q[:, 1, :3] = joint_q[:, :3] + joint_q.new_tensor((1.0, 0.0, 0.0))
            body_q[:, 2, :3] = joint_q[:, :3] + joint_q.new_tensor((0.0, 1.0, 0.0))
            body_q[:, 2, 5:] = joint_q.new_tensor((2.0**-0.5, 2.0**-0.5))
            body_qd.zero_()
            return body_q, body_qd

    frame_count = 5
    root_pose = torch.zeros((frame_count, 1, 7))
    root_pose[:, 0, 0] = torch.arange(frame_count)
    root_pose[:, 0, 6] = 1.0
    state_bank = SimpleNamespace(
        layout=SimpleNamespace(names=("robot",), joint_names=(("joint",),)),
        root_pose=root_pose,
        joint_position=torch.arange(frame_count, dtype=torch.float32).reshape(-1, 1),
        joint_velocity=10.0 * torch.arange(frame_count, dtype=torch.float32).reshape(-1, 1),
    )
    view = SimpleNamespace(
        sequences=SimpleNamespace(
            offsets=torch.tensor((0, 2, 5), dtype=torch.int64),
            frame_dt=torch.tensor((0.1, 0.2), dtype=torch.float32),
            state_indices=None,
        ),
        state_bank=state_bank,
        kinematic_view=KinematicView(),
        quality=SimpleNamespace(
            scope="sequence",
            names=("accepted",),
            values=torch.tensor(((0.0,), (1.0,))),
        ),
    )
    clips = (
        SimpleNamespace(clip_id="rejected", frame_count=2, source_fps=10.0),
        SimpleNamespace(clip_id="accepted", frame_count=3, source_fps=5.0),
    )
    index = SimpleNamespace(clips=clips, offsets=(0, 2, 5))
    trajectory = SimpleNamespace(
        position_body_index_tensor=torch.tensor((0, 1), dtype=torch.int64),
        root_body_index=0,
        support_body_indices=torch.tensor((2,), dtype=torch.int64),
        support_point_body_m=torch.tensor(((0.0, 0.0, -0.1),)),
    )

    def planned_memory(problem_count, device, estimate_memory):
        assert problem_count == frame_count
        assert torch.device(device) == torch.device("cpu")
        assert estimate_memory(2) == 624
        return SimpleNamespace(
            problem_count=problem_count,
            max_safe_capacity=2,
            batch_capacity=2,
            fixed_bytes=0,
            bytes_per_problem=312,
            device_free_bytes=None,
            safety_reserve_bytes=0,
            memory_budget_bytes=None,
            peak_additional_workspace_bytes=624,
        )

    monkeypatch.setattr(benchmark, "plan_ik_memory", planned_memory)
    reference = Reference()
    corpus, memory_plan, measured_peak = benchmark._inspection_corpus(view, index, (1, 0), reference, trajectory)
    acceptance = benchmark._inspection_acceptance(view, index, (1, 0))

    assert corpus.landmark_position_m[:, 0, 0].tolist() == [2.0, 3.0, 4.0, 0.0, 1.0]
    assert corpus.rotation_xyzw.shape == (frame_count, 1, 4)
    assert corpus.rotation_xyzw[:, 0, 3].tolist() == pytest.approx([1.0] * frame_count)
    assert corpus.support_position_m[:, 0, 2].tolist() == pytest.approx([-0.1] * frame_count)
    assert corpus.joint_position_rad[:, 0].tolist() == [2.0, 3.0, 4.0, 0.0, 1.0]
    assert reference.batch_sizes == [2, 2, 1]
    assert len(set(reference.storage_pointers)) == 1
    assert memory_plan.batch_capacity == 2
    assert measured_peak is None
    assert benchmark._fk_memory_report(memory_plan, measured_peak) == {
        "problem_count": 5,
        "max_safe_capacity": 2,
        "batch_capacity": 2,
        "batch_count": 3,
        "fixed_bytes": 0,
        "bytes_per_frame": 312,
        "device_free_bytes": None,
        "safety_reserve_bytes": 0,
        "memory_budget_bytes": None,
        "estimated_peak_additional_workspace_bytes": 624,
        "measured_peak_incremental_bytes": None,
    }
    assert acceptance == {"accepted": True, "rejected": False}


def test_inspection_stance_uses_one_candidate_source_mask_per_target_patch(benchmark) -> None:
    """Candidate contact evidence supplies the shared mask for both compared trajectories."""
    point_valid = torch.tensor(
        (
            (True, True, False, False, False),
            (True, True, True, True, True),
            (False, False, True, True, True),
            (False, False, False, False, False),
            (True, True, False, False, False),
        )
    )
    view = SimpleNamespace(
        sequences=SimpleNamespace(offsets=torch.tensor((0, 3, 5), dtype=torch.int64), state_indices=None),
        points=(
            SimpleNamespace(name="landmarks", scope="state", valid=None),
            SimpleNamespace(name="contact_points", scope="state", valid=point_valid),
        ),
    )
    index = SimpleNamespace(clips=(SimpleNamespace(clip_id="first"), SimpleNamespace(clip_id="second")))

    stance = benchmark._inspection_stance_by_id(view, index, (1, 0), (0, 2, 5))

    assert tuple(stance) == ("second", "first")
    torch.testing.assert_close(stance["first"], point_valid[:3])
    torch.testing.assert_close(stance["second"], point_valid[3:])

    inconsistent = point_valid.clone()
    inconsistent[0, 1] = False
    broken_view = SimpleNamespace(
        sequences=view.sequences,
        points=(SimpleNamespace(name="contact_points", scope="state", valid=inconsistent),),
    )
    with pytest.raises(ValueError, match="one source-contact mask"):
        benchmark._inspection_stance_by_id(broken_view, index, (0,), (0, 2, 5))
