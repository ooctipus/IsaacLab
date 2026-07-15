# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for typed motion sources and the command-owned trajectory table."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.motion.data import (
    MotionClipIndex,
    MotionFrames,
    MotionSkeleton,
    MotionSourceCfg,
)
from isaaclab_tasks.core.multi_task.motion.data import source as motion_source_module
from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
from isaaclab_tasks.core.multi_task.motion.data.sources import cmu_humenv_smpl as cmu_humenv_smpl_module
from isaaclab_tasks.core.multi_task.motion.data.sources import lafan_g1_29dof as lafan_g1_29dof_module
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_sampler import MotionSampler
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import MotionTaskTable
from isaaclab_tasks.core.multi_task.tests.motion_table_test_utils import motion_task_table

_SYNTHETIC_JOINT_NAMES = ("joint_a", "joint_b")
_SYNTHETIC_REFERENCE_FRAME_NAMES: tuple[str, ...] = ()
_SYNTHETIC_RESET_SOURCES = (("reference", 0.7), ("fall", 0.3))


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _ordered_file_hashes(entries: tuple[tuple[str, str], ...]) -> str:
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _phase3_fixtures() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts/reinforcement_learning/forward_backward/phase3/fixtures"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Phase 3 fixtures were not found from the repository test path.")


def _synthetic_skeleton() -> MotionSkeleton:
    return MotionSkeleton(
        identifier="synthetic_two_body",
        content_sha256=_hash("synthetic-skeleton"),
        body_names=("root", "child"),
        parent_indices=(-1, 0),
        rest_translation_m=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        joint_names=("hinge",),
        joint_axes=((0.0, 1.0, 0.0),),
        joint_child_body_indices=(1,),
        root_translation_frame="world",
        root_rotation_convention="wxyz",
        landmark_rotation_policy="calibrated_body",
    )


def _synthetic_index(*, source_hash: str | None = None) -> MotionClipIndex:
    return MotionClipIndex(
        source_content_sha256=source_hash or _hash("source-table"),
        skeleton_identity_sha256s=(_synthetic_skeleton().identity_sha256,),
        clips=(
            MotionClipIndex.Clip(
                clip_id="clip_a",
                frame_count=5,
                source_fps=30.0,
                content_sha256=_hash("clip-a"),
                skeleton_id=0,
            ),
            MotionClipIndex.Clip(
                clip_id="clip_b",
                frame_count=4,
                source_fps=20.0,
                content_sha256=_hash("clip-b"),
                skeleton_id=0,
            ),
        ),
    )


def _empty_frames(frame_count: int) -> MotionFrames:
    return MotionFrames(
        root_position=torch.empty(frame_count, 3),
        root_rotation=torch.empty(frame_count, 4),
        root_linear_velocity=torch.empty(frame_count, 3),
        root_angular_velocity=torch.empty(frame_count, 3),
        joint_position=torch.empty(frame_count, 2),
        joint_velocity=torch.empty(frame_count, 2),
    )


def _clip_frames(clip_index: int, frame_count: int) -> MotionFrames:
    base = torch.arange(frame_count, dtype=torch.float32) + 100.0 * clip_index
    angle = base * 0.2
    zeros = torch.zeros_like(angle)
    root_rotation = torch.stack(
        (zeros, zeros, torch.sin(0.5 * angle), torch.cos(0.5 * angle)),
        dim=-1,
    )
    return MotionFrames(
        root_position=torch.stack((base, base + 0.25, base + 0.5), dim=-1),
        root_rotation=root_rotation,
        root_linear_velocity=torch.stack((base + 10.0, base + 20.0, base + 30.0), dim=-1),
        root_angular_velocity=torch.stack((base + 40.0, base + 50.0, base + 60.0), dim=-1),
        joint_position=torch.stack((base, base + 0.25), dim=-1),
        joint_velocity=torch.stack((base + 10.0, base + 20.0), dim=-1),
    )


def _synthetic_table() -> MotionTaskTable:
    index = _synthetic_index()
    frames = _empty_frames(index.total_frames)
    for clip_number, clip in enumerate(index.clips):
        start, end = index.offsets[clip_number : clip_number + 2]
        frames._copy_clip_(start, end, _clip_frames(clip_number, clip.frame_count))
    return motion_task_table(
        index,
        frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        "synthetic_decoder_v1",
    )


def test_frozen_g1_skeleton_and_explicit_frames_construct_exact_typed_contracts() -> None:
    """Physical skeleton provenance and task tensors remain separate owners."""
    fixture = json.loads((_phase3_fixtures() / "g1_lafan_50hz_skeleton_v1.json").read_text())
    physical = fixture["physical_skeleton"]
    skeleton = MotionSkeleton(
        identifier="g1_29dof_physical_v1",
        content_sha256=fixture["source"]["sha256"],
        body_names=tuple(physical["body_names"]),
        parent_indices=tuple(physical["parent_indices"]),
        rest_translation_m=tuple(tuple(value) for value in physical["rest_translation_m"]),
        rest_rotation_wxyz=tuple(tuple(value) for value in physical["rest_rotation_wxyz"]),
        joint_names=tuple(physical["joint_names"]),
        joint_axes=tuple(tuple(value) for value in physical["joint_axes"]),
        joint_child_body_indices=tuple(range(1, len(physical["body_names"]))),
        root_translation_frame=physical["root"]["translation_frame"],
        root_rotation_convention=physical["root"]["rotation_convention"],
        landmark_rotation_policy="calibrated_body",
    )
    frames = MotionFrames(
        joint_position=torch.zeros(2, 29),
        joint_velocity=torch.zeros(2, 29),
        body_position=torch.zeros(2, 31, 3),
        body_rotation=torch.zeros(2, 31, 4),
        body_linear_velocity=torch.zeros(2, 31, 3),
        body_angular_velocity=torch.zeros(2, 31, 3),
    )

    assert skeleton.num_bodies == 30
    assert skeleton.num_joints == 29
    assert frames.body_position is not None and frames.body_position.shape[1] == fixture["reference_frame_count"] == 31
    assert "observation" not in {field.name for field in dataclasses.fields(MotionFrames)}
    assert not hasattr(frames, "observation")
    assert frames.stored_fields == (
        "joint_position",
        "joint_velocity",
        "body_position",
        "body_rotation",
        "body_linear_velocity",
        "body_angular_velocity",
    )
    assert frames.root_storage == "body_row_zero"
    assert set(frames.available_fields) == set(frames.stored_fields) | set(frames._ROOT_FIELDS)
    for root_name, body_name in zip(
        frames._ROOT_FIELDS,
        ("body_position", "body_rotation", "body_linear_velocity", "body_angular_velocity"),
        strict=True,
    ):
        assert frames.field(root_name).data_ptr() == frames.field(body_name).data_ptr()
    frame_bytes = sum(frames.field(name).numel() * frames.field(name).element_size() for name in frames.stored_fields)
    assert frame_bytes == 2 * 461 * torch.tensor([], dtype=torch.float32).element_size()
    assert skeleton.identity_sha256 == dataclasses.replace(skeleton).identity_sha256
    assert not hasattr(MotionTaskTable, "FrameLayout")


def test_motion_table_view_addresses_exact_stored_integer_frames() -> None:
    """The shared inspector boundary reads the stored robot state without interpolation."""
    table = _synthetic_table()
    sequence_ids = torch.tensor((0, 1), dtype=torch.int64)
    frame_ids = torch.tensor((2, 1), dtype=torch.int64)
    rows = table.view.sequences.state_rows(sequence_ids, frame_ids)

    torch.testing.assert_close(table.view.state_bank.root_pose[:, 0, :3], table.field("root_position"))
    torch.testing.assert_close(table.view.state_bank.root_pose[:, 0, 3:], table.field("root_rotation"))
    torch.testing.assert_close(table.view.state_bank.joint_position, table.field("joint_position"))
    joint_q = torch.empty(rows.numel(), table.view.kinematic_view.joint_q_default.numel())
    table.view.kinematic_view.joint_q_into(table.view.state_bank, rows, joint_q)
    torch.testing.assert_close(joint_q[:, :7], table.view.state_bank.root_pose[rows, 0])
    torch.testing.assert_close(joint_q[:, 7:], table.view.state_bank.joint_position[rows])


def test_frames_reject_partial_groups_and_noncontiguous_columns() -> None:
    with pytest.raises(ValueError, match="required together"):
        MotionFrames(
            joint_position=torch.zeros(2, 1),
            root_position=torch.zeros(2, 3),
            root_rotation=torch.zeros(2, 4),
            root_linear_velocity=torch.zeros(2, 3),
            root_angular_velocity=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError, match="all present or all absent"):
        MotionFrames(
            root_position=torch.zeros(2, 3),
            root_rotation=torch.zeros(2, 4),
            root_linear_velocity=torch.zeros(2, 3),
            root_angular_velocity=torch.zeros(2, 3),
            joint_position=torch.zeros(2, 1),
            joint_velocity=torch.zeros(2, 1),
            body_position=torch.zeros(2, 3, 3),
        )
    with pytest.raises(ValueError, match="contiguous"):
        MotionFrames(
            joint_position=torch.zeros(1, 2).expand(2, 2),
            joint_velocity=torch.zeros(2, 2),
            root_position=torch.zeros(2, 3),
            root_rotation=torch.zeros(2, 4),
            root_linear_velocity=torch.zeros(2, 3),
            root_angular_velocity=torch.zeros(2, 3),
        )


@pytest.mark.parametrize("field_name", ("joint_position", "root_rotation"))
def test_table_rejects_nonfinite_and_nonunit_physical_rows(field_name: str) -> None:
    """Invalid physical values must fail before entering immutable table storage."""
    index = _synthetic_index()
    frames = _empty_frames(index.total_frames)
    for clip_number, clip in enumerate(index.clips):
        start, end = index.offsets[clip_number : clip_number + 2]
        frames._copy_clip_(start, end, _clip_frames(clip_number, clip.frame_count))
    if field_name == "joint_position":
        assert frames.joint_position is not None
        frames.joint_position[1, 0] = torch.nan
        match = "finite"
    else:
        assert frames.root_rotation is not None
        frames.root_rotation[1].zero_()
        match = "unit quaternion"

    with pytest.raises(ValueError, match=match):
        motion_task_table(
            index,
            frames,
            _SYNTHETIC_JOINT_NAMES,
            _SYNTHETIC_REFERENCE_FRAME_NAMES,
            "synthetic_builder_v1",
            _hash("synthetic-builder-construction-v1"),
            "source_frames",
            "synthetic_decoder_v1",
        )


def test_skeleton_landmarks_are_validated_and_identity_bearing() -> None:
    """Semantic roles bind existing position/orientation bodies and affect source identity."""
    skeleton = _synthetic_skeleton()
    landmark = MotionSkeleton.Landmark("tip", "child", "child")
    semantic = dataclasses.replace(skeleton, landmarks=(landmark,))

    assert semantic.landmarks == (landmark,)
    assert semantic.identity_sha256 != skeleton.identity_sha256
    assert semantic.coordinate_identity_sha256 == skeleton.coordinate_identity_sha256
    with pytest.raises(ValueError, match="landmark bodies"):
        dataclasses.replace(skeleton, landmarks=(MotionSkeleton.Landmark("tip", "missing", "child"),))
    with pytest.raises(ValueError, match="landmark names"):
        dataclasses.replace(skeleton, landmarks=(landmark, landmark))


def test_skeleton_landmark_rotation_policy_is_validated_and_identity_bearing() -> None:
    """Rotation evidence changes semantic identity without changing native coordinates."""
    calibrated = _synthetic_skeleton()
    anatomical = dataclasses.replace(calibrated, landmark_rotation_policy="anatomical_root")

    assert calibrated.identity_sha256 != anatomical.identity_sha256
    assert calibrated.coordinate_identity_sha256 == anatomical.coordinate_identity_sha256
    with pytest.raises(ValueError, match="landmark_rotation_policy"):
        dataclasses.replace(calibrated, landmark_rotation_policy="unknown")
    with pytest.raises(ValueError, match="landmark_rotation_policy"):
        dataclasses.replace(calibrated, landmark_rotation_policy="root_only")


def test_skeleton_represents_hinge_and_unrestricted_source_joints() -> None:
    hinge = _synthetic_skeleton()
    unrestricted = dataclasses.replace(hinge, joint_axes=(None,))
    assert unrestricted.joint_axes == (None,)
    assert unrestricted.identity_sha256 != hinge.identity_sha256


def test_skeleton_represents_multiple_joint_coordinates_on_one_body() -> None:
    skeleton = MotionSkeleton(
        identifier="three_hinges_one_body",
        content_sha256=_hash("three-hinges-one-body"),
        body_names=("root", "child"),
        parent_indices=(-1, 0),
        rest_translation_m=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        joint_names=("child_x", "child_y", "child_z"),
        joint_child_body_indices=(1, 1, 1),
        joint_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        root_translation_frame="world",
        root_rotation_convention="wxyz",
        landmark_rotation_policy="calibrated_body",
    )

    assert skeleton.num_bodies == 2
    assert skeleton.num_joints == 3
    assert skeleton.joint_child_body_indices == (1, 1, 1)


def test_skeleton_rejects_nonfinite_and_nonunit_geometry() -> None:
    skeleton = _synthetic_skeleton()

    with pytest.raises(ValueError, match="rest translations must be finite"):
        dataclasses.replace(
            skeleton,
            rest_translation_m=((float("nan"), 0.0, 0.0), skeleton.rest_translation_m[1]),
        )
    with pytest.raises(ValueError, match="rest rotations must be finite unit quaternions"):
        dataclasses.replace(
            skeleton,
            rest_rotation_wxyz=((2.0, 0.0, 0.0, 0.0), skeleton.rest_rotation_wxyz[1]),
        )
    with pytest.raises(ValueError, match="hinge axes must be finite unit vectors"):
        dataclasses.replace(skeleton, joint_axes=((0.0, 2.0, 0.0),))


def test_clip_index_retains_only_runtime_fields_and_changes_with_clip_metadata() -> None:
    index = _synthetic_index()
    changed_fps = dataclasses.replace(
        index,
        clips=(dataclasses.replace(index.clips[0], source_fps=60.0), index.clips[1]),
    )

    assert tuple(field.name for field in dataclasses.fields(MotionClipIndex.Clip)) == (
        "clip_id",
        "frame_count",
        "source_fps",
        "content_sha256",
        "skeleton_id",
        "source_clip_id",
        "source_frame_start",
    )
    assert tuple(field.name for field in dataclasses.fields(MotionClipIndex)) == (
        "source_content_sha256",
        "skeleton_identity_sha256s",
        "clips",
        "clip_ids",
        "skeleton_ids",
        "offsets",
    )
    assert index.clip_ids == ("clip_a", "clip_b")
    assert index.skeleton_ids == (0,)
    assert index.for_skeleton(0) == (0, 1)
    assert index.total_frames == 9
    assert index != changed_fps
    assert index.clips[0].source_clip_id is None
    assert index.clips[0].source_frame_start == 0
    assert index.clips[0].source_frame_stop == 5
    derived = dataclasses.replace(
        index.clips[0],
        clip_id="clip_a_segment",
        frame_count=3,
        source_clip_id="clip_a",
        source_frame_start=1,
    )
    assert derived.source_frame_stop == 4


def test_clip_index_groups_dense_source_skeletons_in_first_occurrence_order() -> None:
    base = _synthetic_index()
    clips = (base.clips[0], dataclasses.replace(base.clips[1], skeleton_id=1))
    index = MotionClipIndex(
        source_content_sha256=base.source_content_sha256,
        skeleton_identity_sha256s=(_hash("skeleton-a"), _hash("skeleton-b")),
        clips=clips,
    )

    assert index.skeleton_ids == (0, 1)
    assert index.for_skeleton(0) == (0,)
    assert index.for_skeleton(1) == (1,)
    with pytest.raises(ValueError, match="dense in first-occurrence order"):
        dataclasses.replace(index, clips=(clips[1], clips[0]))
    with pytest.raises(ValueError, match="Every declared skeleton identity"):
        dataclasses.replace(index, clips=(clips[0],))


def test_frozen_g1_training_index_has_exact_capacity_without_loading_joblib() -> None:
    fixture = json.loads((_phase3_fixtures() / "native_motion_data_v1.json").read_text())
    g1 = fixture["collections"]["g1_lafan_50hz"]
    training = g1["training"]
    source_fps = float(g1["fields"]["fps"]["value"])
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=f"lafan_{clip_index:04d}",
            frame_count=training["frames_per_clip"],
            source_fps=source_fps,
            content_sha256=_hash(f"lafan-{clip_index}"),
            skeleton_id=0,
        )
        for clip_index in range(training["clips"])
    )
    index = MotionClipIndex(
        source_content_sha256=training["sha256"],
        skeleton_identity_sha256s=(_hash("frozen-g1-skeleton"),),
        clips=clips,
    )

    assert len(index.clips) == 862
    assert index.total_frames == training["frames"] == 258_600
    assert index.offsets[0] == 0 and index.offsets[-1] == 258_600
    assert all(right - left == 300 for left, right in zip(index.offsets, index.offsets[1:]))


def test_table_identity_invalidates_scientific_inputs_without_robot_schema() -> None:
    table = _synthetic_table()
    index = _synthetic_index()
    same = motion_task_table(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        "synthetic_decoder_v1",
    )
    changed_source = motion_task_table(
        _synthetic_index(source_hash=_hash("changed-source")),
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        "synthetic_decoder_v1",
    )
    changed_builder = motion_task_table(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v2",
        _hash("synthetic-builder-construction-v2"),
        "source_frames",
        "synthetic_decoder_v1",
    )
    changed_construction = motion_task_table(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("changed-reference-kinematics-and-order"),
        "source_frames",
        "synthetic_decoder_v1",
    )
    changed_mode = motion_task_table(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "clip_time_ranges",
        "synthetic_decoder_v1",
    )

    assert same.cache_identity == table.cache_identity
    assert changed_source.cache_identity != table.cache_identity
    assert changed_builder.cache_identity != table.cache_identity
    assert changed_construction.cache_identity != table.cache_identity
    assert changed_mode.cache_identity != table.cache_identity


def test_sampler_policy_is_excluded_from_immutable_table_identity() -> None:
    table = _synthetic_table()
    first = MotionSampler(table, _SYNTHETIC_RESET_SOURCES, capacity=table.num_tasks, seed=7)
    changed = MotionSampler(
        table,
        (("motion", 0.6), ("fall", 0.4)),
        capacity=table.num_tasks,
        seed=11,
    )

    assert first.table.cache_identity == changed.table.cache_identity == table.cache_identity
    assert first.reset_source_names != changed.reset_source_names


def test_task_row_modes_generate_exact_rows_and_expose_reset_source_names() -> None:
    source_frames = _synthetic_table()
    expected_times = torch.tensor((0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0, 4.0 / 30.0, 0.0, 0.05, 0.1, 0.15))

    source_sampler = MotionSampler(source_frames, _SYNTHETIC_RESET_SOURCES, capacity=source_frames.num_tasks, seed=7)
    assert source_sampler.reset_source_names == ("reference", "fall")
    torch.testing.assert_close(source_frames.clip_indices, torch.tensor((0, 0, 0, 0, 0, 1, 1, 1, 1)))
    torch.testing.assert_close(source_frames.clip_start_rows, torch.tensor((0, 5)))
    torch.testing.assert_close(source_frames.reset_time_ranges_seconds[:, 0], expected_times)
    torch.testing.assert_close(source_frames.reset_time_ranges_seconds[:, 1], expected_times)

    clip_ranges = motion_task_table(
        source_frames.clip_index,
        source_frames.frames,
        source_frames.joint_names,
        source_frames.reference_frame_names,
        source_frames.construction_version,
        source_frames.construction_identity_sha256,
        "clip_time_ranges",
        "synthetic_decoder_v1",
    )
    torch.testing.assert_close(clip_ranges.clip_indices, torch.tensor((0, 1)))
    torch.testing.assert_close(clip_ranges.clip_start_rows, torch.tensor((0, 1)))
    torch.testing.assert_close(
        clip_ranges.reset_time_ranges_seconds,
        torch.tensor(((0.0, 4.0 / 30.0), (0.0, 3.0 / 20.0))),
    )


def test_table_binds_preallocated_storage_without_copy() -> None:
    index = _synthetic_index()
    frames = _empty_frames(index.total_frames)
    pointers = {name: frames.field(name).data_ptr() for name in frames.stored_fields}
    for clip_number, clip in enumerate(index.clips):
        start, end = index.offsets[clip_number : clip_number + 2]
        frames._copy_clip_(start, end, _clip_frames(clip_number, clip.frame_count))
        assert {name: frames.field(name).data_ptr() for name in frames.stored_fields} == pointers
    table = motion_task_table(
        index,
        frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        "synthetic_decoder_v1",
    )

    assert table.frames is frames
    assert {name: table.field(name).data_ptr() for name in frames.stored_fields} == pointers
    assert table.clip_offsets.tolist() == [0, 5, 9]
    assert table.frame_counts.tolist() == [5, 4]
    assert table.source_fps.tolist() == [30.0, 20.0]
    assert table.clip_ids == index.clip_ids
    sampler = MotionSampler(table, _SYNTHETIC_RESET_SOURCES, capacity=table.num_tasks, seed=7)
    table_metadata = (
        table.clip_offsets,
        table.clip_start_rows,
        table.frame_counts,
        table.source_fps,
        table.clip_indices,
        table.reset_time_ranges_seconds,
    )
    trajectory_bytes = sum(
        table.frames.field(name).numel() * table.frames.field(name).element_size()
        for name in table.frames.stored_fields
    )
    resident_bytes = trajectory_bytes + sum(tensor.numel() * tensor.element_size() for tensor in table_metadata)
    assert resident_bytes == 820
    sampler_metadata = (
        sampler._sampling_row_starts,
        sampler._sampling_row_counts,
        sampler.reset_source_probabilities,
        sampler.clip_priorities,
    )
    sampler_bytes = sum(tensor.numel() * tensor.element_size() for tensor in sampler_metadata)
    assert sampler_bytes == 48


def test_storage_binding_reuses_tensors_and_seals_metadata() -> None:
    table = _synthetic_table()
    rebound = motion_task_table(
        table.clip_index,
        table.frames,
        table.joint_names,
        table.reference_frame_names,
        table.construction_version,
        table.construction_identity_sha256,
        "source_frames",
        "synthetic_decoder_v1",
    )

    assert rebound.frames is table.frames
    assert rebound.cache_identity == table.cache_identity
    with pytest.raises(AttributeError, match="immutable"):
        rebound._task_row_mode = "clip_time_ranges"


def test_storage_binding_rejects_invalid_physical_values() -> None:
    """Prebuilt storage must satisfy the same value contract as streamed clips."""
    table = _synthetic_table()
    assert table.frames.root_rotation is not None
    table.frames.root_rotation[0].zero_()

    with pytest.raises(ValueError, match="unit quaternion"):
        motion_task_table(
            table.clip_index,
            table.frames,
            table.joint_names,
            table.reference_frame_names,
            table.construction_version,
            table.construction_identity_sha256,
            "source_frames",
            "synthetic_decoder_v1",
        )


def test_reference_view_interpolates_and_reports_tail_without_wrapping() -> None:
    table = _synthetic_table()
    view = table.reference_view(
        torch.tensor([0, 1, 0]),
        torch.tensor([0.5 / 30.0, 100.0, -0.1]),
    )

    assert view.table is table
    assert view.local_frame0.tolist() == [0, 3, 0]
    assert view.local_frame1.tolist() == [1, 3, 1]
    assert view.tail_valid.tolist() == [True, False, False]
    torch.testing.assert_close(view.field("joint_position")[0], torch.tensor([0.5, 0.75]))
    torch.testing.assert_close(view.field("joint_position")[1], torch.tensor([103.0, 103.25]))
    torch.testing.assert_close(
        view.field("joint_velocity"),
        torch.tensor(
            ((10.5, 20.5), (113.0, 123.0), (10.0, 20.0)),
            dtype=torch.float32,
        ),
    )
    expected_half_angle = torch.tensor(0.05)
    expected_rotation = torch.tensor([0.0, 0.0, torch.sin(expected_half_angle), torch.cos(expected_half_angle)])
    torch.testing.assert_close(view.field("root_rotation")[0], expected_rotation)


def test_reference_view_slerp_uses_shortest_arc_without_mutating_table_frames() -> None:
    """Signed quaternion storage must interpolate by rotation without rewriting immutable table data."""
    table = _synthetic_table()
    first = torch.tensor([0.0, 0.0, 0.0, 1.0])
    second = -torch.tensor([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)])
    table.frames.root_rotation[0].copy_(first)
    table.frames.root_rotation[1].copy_(second)
    frames_before = table.frames.root_rotation.clone()
    view = table.reference_view(
        torch.zeros(3, dtype=torch.int64),
        torch.tensor([0.0, 0.5 / 30.0, 1.0 / 30.0]),
    )

    result = view.field("root_rotation")

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)],
            [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
        ]
    )
    torch.testing.assert_close(result[:2], expected[:2])
    torch.testing.assert_close(torch.abs(torch.dot(result[2], expected[2])), torch.tensor(1.0))
    torch.testing.assert_close(table.frames.root_rotation, frames_before)


def test_table_owns_source_row_and_uniform_sample_clocks() -> None:
    table = _synthetic_table()
    source_rows = table.sample("source_rows", None)
    uniform = table.sample("uniform_before_source_end", 0.02)

    assert source_rows.source is table
    assert source_rows.clip_ids == table.clip_ids
    assert source_rows.clip_offsets == (0, 5, 9)
    assert uniform.source is table
    assert uniform.clip_ids == table.clip_ids
    assert uniform.clip_offsets == (0, 7, 15)
    assert source_rows.field("joint_position").shape == (9, 2)
    assert uniform.field("joint_position").shape == (15, 2)


def test_runtime_package_has_no_source_format_dependencies() -> None:
    package = Path(__file__).parents[1] / "motion" / "data"
    imported_roots: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint({"h5py", "joblib", "scipy"})


def _humenv_arrays(frame_count: int, offset: float) -> dict[str, np.ndarray]:
    values = np.arange(frame_count, dtype=np.float32) + offset
    truncated = np.zeros((frame_count, 1), dtype=np.bool_)
    truncated[-1] = True
    return {
        "motion_id": np.full((frame_count, 1), int(offset), dtype=np.int64),
        "observation": np.repeat(values[:, None], 358, axis=1).astype(np.float64),
        "qpos": np.repeat(values[:, None], 76, axis=1),
        "qvel": np.repeat(values[:, None], 75, axis=1),
        "terminated": np.zeros((frame_count, 1), dtype=np.bool_),
        "truncated": truncated,
    }


def _bfm_arrays(frame_count: int, offset: float, motion_name: str) -> dict[str, np.ndarray | int | str]:
    values = np.arange(frame_count, dtype=np.float32) + offset
    joint_position = np.repeat(values[:, None], 29, axis=1)
    pose_axis_angle = np.zeros((frame_count, 30, 3), dtype=np.float32)
    pose_axis_angle[:, 0] = values[:, None]
    pose_axis_angle[:, 1:] = joint_position[..., None] * np.asarray(lafan_g1_29dof_module.LAFAN_G1_JOINT_AXES)
    return {
        "root_trans_offset": np.repeat(values[:, None], 3, axis=1),
        "pose_aa": pose_axis_angle,
        "dof": joint_position,
        "root_rot": np.repeat(values[:, None], 4, axis=1),
        "smpl_joints": np.repeat(values[:, None, None], 24 * 3, axis=1).reshape(frame_count, 24, 3),
        "fps": 30,
        "motion_name": motion_name,
    }


def test_source_boundary_hashes_artifact_once_and_passes_explicit_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "motion_deployment"
    artifact = root / "indexes" / "train.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("clip.hdf5\n", encoding="utf-8")
    artifact_sha256 = _file_hash(artifact)
    split = MotionSourceCfg.SplitCfg(
        name="train",
        artifact="indexes/train.txt",
        artifact_sha256=artifact_sha256,
        source_content_sha256=_hash("content"),
        clip_count=1,
        frame_count=1,
    )
    opened = object()
    received: tuple[Path, Path, MotionSourceCfg.SplitCfg, MotionSourceCfg, str] | None = None

    def open_source(
        artifact_path: Path,
        source_root: Path,
        selected_split: MotionSourceCfg.SplitCfg,
        source_cfg: MotionSourceCfg,
        verified_artifact_sha256: str,
    ):
        nonlocal received
        received = artifact_path, source_root, selected_split, source_cfg, verified_artifact_sha256
        return opened

    cfg = MotionSourceCfg(
        identifier="synthetic",
        open_source=open_source,
        format="split_list",
        semantic_level="test",
        decoder_version="test_v1",
        source_fps=30.0,
        license="test-only",
        clip_directory=None,
        train=split,
        evaluation=dataclasses.replace(split, name="evaluation"),
    )
    original_file_sha256 = motion_source_module.file_sha256
    scanned_paths: list[Path] = []

    def record_scan(path: str | Path) -> str:
        scanned_paths.append(Path(path))
        return original_file_sha256(path)

    monkeypatch.setattr(motion_source_module, "file_sha256", record_scan)

    assert cfg.open_split(root, cfg.train) is opened
    assert scanned_paths == [artifact]
    assert received == (artifact, root, cfg.train, cfg, artifact_sha256)


def test_cmu_source_uses_explicit_clip_directory_and_hashes_each_clip_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "motion_deployment"
    split_path = root / "indexes" / "nested" / "train.txt"
    clip_root = root / "native" / "cmu"
    split_path.parent.mkdir(parents=True)
    clip_root.mkdir(parents=True)
    clip_names = ("later_name.hdf5", "earlier_name.hdf5")
    paths = tuple(clip_root / name for name in clip_names)
    for path, values in zip(paths, (_humenv_arrays(4, 10.0), _humenv_arrays(3, 20.0)), strict=True):
        with h5py.File(path, "w") as stream:
            episode = stream.create_group("ep_0")
            for name, value in values.items():
                episode.create_dataset(name, data=value)
    split_path.write_text("".join(f"{name}\n" for name in clip_names), encoding="utf-8")
    content_sha256 = _ordered_file_hashes(tuple(zip(clip_names, map(_file_hash, paths), strict=True)))
    split = MotionSourceCfg.SplitCfg(
        name="train",
        artifact="indexes/nested/train.txt",
        artifact_sha256=_file_hash(split_path),
        source_content_sha256=content_sha256,
        clip_count=2,
        frame_count=7,
    )
    cfg = MotionSourceCfg(
        identifier="cmu_test",
        open_source=cmu_humenv_smpl_module.open_cmu_humenv_smpl_source,
        format="one_hdf5_file_per_clip_with_group_ep_0",
        semantic_level="smpl_robot_state_and_observation",
        decoder_version="test_v1",
        source_fps=30.0,
        license="test-only",
        clip_directory="native/cmu",
        train=split,
        evaluation=dataclasses.replace(split, name="evaluation"),
    )
    original_file_sha256 = cmu_humenv_smpl_module.file_sha256
    scanned_paths: list[Path] = []

    def record_scan(path: Path) -> str:
        scanned_paths.append(path)
        return original_file_sha256(path)

    monkeypatch.setattr(cmu_humenv_smpl_module, "file_sha256", record_scan)
    source = cfg.open_split(root, cfg.train)

    assert scanned_paths == list(paths)
    assert source.inspect().source_content_sha256 == content_sha256
    assert [source.inspect().clip_ids[clip_index] for clip_index, _ in source.clips((0, 1))] == list(clip_names)
    assert scanned_paths == list(paths)
    with paths[0].open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="source content hash differs"):
        cfg.open_split(root, cfg.train)


def test_humenv_hdf5_import_preserves_explicit_caller_order_and_streams_files(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    paths = (tmp_path / "later_name.hdf5", tmp_path / "earlier_name.hdf5")
    for path, values in zip(paths, (_humenv_arrays(4, 10.0), _humenv_arrays(3, 20.0)), strict=True):
        with h5py.File(path, "w") as stream:
            episode = stream.create_group("ep_0")
            for name, value in values.items():
                episode.create_dataset(name, data=value)

    source = CmuHumEnvSmplClips(
        paths,
        file_sha256s=tuple(map(_file_hash, paths)),
        clip_ids=("caller_first", "caller_second"),
        source_fps=30.0,
    )
    index = source.inspect()
    assert index.clip_ids == ("caller_first", "caller_second")
    assert index.offsets == (0, 4, 7)
    assert index.skeleton_identity_sha256s == (source.skeleton(0).identity_sha256,)
    with pytest.raises(ValueError, match="Unknown skeleton id"):
        source.skeleton(1)
    decoded = list(source.clips((0, 1)))
    assert [clip_index for clip_index, _ in decoded] == [0, 1]
    assert decoded[0][1].generalized_position.dtype == np.float32
    assert decoded[0][1].generalized_velocity.dtype == np.float32
    assert decoded[1][1].generalized_position.shape == (3, 76)


def test_humenv_inspect_validates_unconsumed_native_fields(tmp_path: Path) -> None:
    """Inspection validates the full native schema even when decoded clips retain only qpos/qvel."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "invalid_observation.hdf5"
    fields = _humenv_arrays(3, 0.0)
    fields["observation"] = fields["observation"].astype(np.float32)
    with h5py.File(path, "w") as stream:
        episode = stream.create_group("ep_0")
        for name, value in fields.items():
            episode.create_dataset(name, data=value)
    source = CmuHumEnvSmplClips(
        (path,),
        file_sha256s=(_file_hash(path),),
        source_fps=30.0,
    )

    with pytest.raises(ValueError, match="observation.*wrong dtype"):
        source.inspect()


def test_native_motion_sources_expose_only_consumed_runtime_state() -> None:
    """Native validation metadata must not survive into decoded runtime clips or facades."""
    fields = tuple(field.name for field in dataclasses.fields(cmu_humenv_smpl_module.CmuHumEnvSmplClip))

    assert fields == ("generalized_position", "generalized_velocity", "source_fps")
    assert not hasattr(cmu_humenv_smpl_module.CmuHumEnvSmplClip, "root_translation")
    assert not hasattr(LafanG1JoblibClips, "remaining_clips")
    assert not hasattr(LafanG1JoblibClips, "remaining_frames")
    assert not hasattr(LafanG1JoblibClips, "__enter__")
    assert not hasattr(LafanG1JoblibClips, "__exit__")


def test_lafan_joblib_import_preserves_insertion_order_and_yields_typed_clips(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm.pkl"
    joblib.dump(
        {
            "z_motion_clip0": _bfm_arrays(5, 10.0, "z_motion"),
            "a_motion_clip0": _bfm_arrays(4, 20.0, "a_motion"),
        },
        path,
    )
    source = LafanG1JoblibClips.load(
        path,
        verified_artifact_sha256=_file_hash(path),
    )
    try:
        index = source.inspect()
        assert index.clip_ids == ("z_motion_clip0", "a_motion_clip0")
        assert index.offsets == (0, 5, 9)
        assert index.skeleton_identity_sha256s == (source.skeleton(0).identity_sha256,)
        assert tuple((clip.source_clip_id, clip.source_frame_start) for clip in index.clips) == (
            ("z_motion", 0),
            ("a_motion", 0),
        )

        clips = source.clips((0, 1))
        first_id, first = next(clips)
        assert first_id == 0
        assert first.pose_axis_angle.shape == (5, 30, 3)
        assert type(first.pose_axis_angle) is np.ndarray
        assert first.pose_axis_angle.flags.writeable
        second_id, second = next(clips)
        assert second_id == 1
        assert second.frame_count == 4
        with pytest.raises(StopIteration):
            next(clips)
    finally:
        source.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(source.clips((0,)))


def test_bfm_training_windows_retain_original_clip_intervals_and_reject_numbering(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "windows.pkl"
    joblib.dump(
        {
            "motion_clip0": _bfm_arrays(3, 10.0, "motion"),
            "motion_clip1": _bfm_arrays(4, 20.0, "motion"),
        },
        path,
    )
    source = LafanG1JoblibClips.load(path, verified_artifact_sha256=_file_hash(path))
    try:
        index = source.inspect()
    finally:
        source.close()

    assert tuple((clip.source_clip_id, clip.source_frame_start, clip.source_frame_stop) for clip in index.clips) == (
        ("motion", 0, 3),
        ("motion", 3, 7),
    )

    malformed = tmp_path / "malformed_windows.pkl"
    joblib.dump({"motion_clip1": _bfm_arrays(3, 10.0, "motion")}, malformed)
    source = LafanG1JoblibClips.load(malformed, verified_artifact_sha256=_file_hash(malformed))
    try:
        with pytest.raises(ValueError, match="numbered contiguously from zero"):
            source.inspect()
    finally:
        source.close()


def test_source_boundary_rejects_lafan_artifact_corruption(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm.pkl"
    joblib.dump({"motion_clip0": _bfm_arrays(3, 10.0, "motion")}, path)
    split = MotionSourceCfg.SplitCfg(
        name="train",
        artifact="bfm.pkl",
        artifact_sha256=_hash("wrong-artifact"),
        source_content_sha256=_file_hash(path),
        clip_count=1,
        frame_count=3,
    )
    cfg = MotionSourceCfg(
        identifier="lafan_test",
        open_source=lafan_g1_29dof_module.open_lafan_g1_source,
        format="joblib_pickle_mapping_clip_name_to_field_mapping",
        semantic_level="robot_pose",
        decoder_version="test_v1",
        source_fps=30.0,
        license="test-only",
        clip_directory=None,
        train=split,
        evaluation=dataclasses.replace(split, name="evaluation"),
    )

    with pytest.raises(ValueError, match="artifact hash differs"):
        cfg.open_split(tmp_path, cfg.train)


def test_bfm_joblib_normalizes_native_variants_to_frame_builder_fields(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm_variants.pkl"
    training = _bfm_arrays(4, 10.0, "training_motion")
    evaluation = _bfm_arrays(3, 20.0, "unused")
    del evaluation["motion_name"]
    joblib.dump({"training_motion_clip0": training, "evaluation_clip": evaluation}, path)

    source = LafanG1JoblibClips.load(
        path,
        verified_artifact_sha256=_file_hash(path),
    )
    try:
        index = source.inspect()
        assert index.clip_ids == ("training_motion_clip0", "evaluation_clip")
        decoded = list(source.clips((0, 1)))
    finally:
        source.close()

    assert [clip_index for clip_index, _ in decoded] == [0, 1]
    assert all(fields.source_fps == 30.0 and fields.pose_axis_angle.shape[1:] == (30, 3) for _, fields in decoded)


def test_bfm_loaded_source_never_rescans_the_boundary_verified_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm_verified.pkl"
    joblib.dump({"motion_clip0": _bfm_arrays(3, 10.0, "motion")}, path)
    artifact_sha256 = _file_hash(path)

    def reject_scan(_source_path: Path) -> str:
        raise AssertionError("The decoder rescanned a boundary-verified artifact.")

    monkeypatch.setattr(lafan_g1_29dof_module, "file_sha256", reject_scan, raising=False)
    source = LafanG1JoblibClips.load(
        path,
        verified_artifact_sha256=artifact_sha256,
    )

    source.inspect()
    list(source.clips((0,)))
    source.close()
