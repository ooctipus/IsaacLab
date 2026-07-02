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
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex, MotionSampleGrid, MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.data.importers import bfm_g1_joblib as bfm_g1_joblib_module
from isaaclab_tasks.core.multi_task.motion.data.importers.bfm_g1_joblib import BfmG1JoblibClips
from isaaclab_tasks.core.multi_task.motion.data.importers.humenv_hdf5 import HumEnvHdf5Clips
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import MotionTaskTable

_SYNTHETIC_JOINT_NAMES = ("joint_a", "joint_b")
_SYNTHETIC_REFERENCE_FRAME_NAMES: tuple[str, ...] = ()
_SYNTHETIC_RESET_SOURCES = (("reference", 0.7), ("fall", 0.3))
_SYNTHETIC_EXPERT_GRID = MotionSampleGrid.source_rows()


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    )


def _synthetic_index(*, first_path: str = "clip_a.tensor", source_hash: str | None = None) -> MotionClipIndex:
    return MotionClipIndex(
        source_content_sha256=source_hash or _hash("source-table"),
        skeleton_sha256=_synthetic_skeleton().identity_sha256,
        semantic_level="robot_state",
        license="test-only",
        clips=(
            MotionClipIndex.Clip(
                clip_id="clip_a",
                source_path=first_path,
                frame_count=5,
                source_fps=30.0,
                split="train",
                tags=("walk",),
                content_sha256=_hash("clip-a"),
            ),
            MotionClipIndex.Clip(
                clip_id="clip_b",
                source_path="clip_b.tensor",
                frame_count=4,
                source_fps=20.0,
                split="test",
                tags=("turn",),
                content_sha256=_hash("clip-b"),
            ),
        ),
    )


def _empty_frames(frame_count: int) -> MotionTaskTable.Frames:
    return MotionTaskTable.Frames(
        root_position=torch.empty(frame_count, 3),
        root_rotation=torch.empty(frame_count, 4),
        root_linear_velocity=torch.empty(frame_count, 3),
        root_angular_velocity=torch.empty(frame_count, 3),
        joint_position=torch.empty(frame_count, 2),
        joint_velocity=torch.empty(frame_count, 2),
        observation=torch.empty(frame_count, 1),
    )


def _clip_frames(clip_index: int, frame_count: int) -> MotionTaskTable.Frames:
    base = torch.arange(frame_count, dtype=torch.float32) + 100.0 * clip_index
    angle = base * 0.2
    zeros = torch.zeros_like(angle)
    root_rotation = torch.stack(
        (zeros, zeros, torch.sin(0.5 * angle), torch.cos(0.5 * angle)),
        dim=-1,
    )
    return MotionTaskTable.Frames(
        root_position=torch.stack((base, base + 0.25, base + 0.5), dim=-1),
        root_rotation=root_rotation,
        root_linear_velocity=torch.stack((base + 10.0, base + 20.0, base + 30.0), dim=-1),
        root_angular_velocity=torch.stack((base + 40.0, base + 50.0, base + 60.0), dim=-1),
        joint_position=torch.stack((base, base + 0.25), dim=-1),
        joint_velocity=torch.stack((base + 10.0, base + 20.0), dim=-1),
        observation=torch.full((frame_count, 1), float(clip_index)),
    )


def _synthetic_table() -> MotionTaskTable:
    index = _synthetic_index()
    writer = MotionTaskTable.writer(
        index,
        _empty_frames(index.total_frames),
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    for clip_number, clip in enumerate(index.clips):
        writer.write_clip(clip.clip_id, _clip_frames(clip_number, clip.frame_count))
    return writer.finish()


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
    )
    frames = MotionTaskTable.Frames(
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
    assert frames.memory_bytes == 2 * 461 * torch.tensor([], dtype=torch.float32).element_size()
    assert skeleton.identity_sha256 == dataclasses.replace(skeleton).identity_sha256
    assert not hasattr(MotionTaskTable, "FrameLayout")


def test_frames_reject_partial_groups_and_noncontiguous_columns() -> None:
    with pytest.raises(ValueError, match="required together"):
        MotionTaskTable.Frames(
            joint_position=torch.zeros(2, 1),
            root_position=torch.zeros(2, 3),
            root_rotation=torch.zeros(2, 4),
            root_linear_velocity=torch.zeros(2, 3),
            root_angular_velocity=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError, match="all present or all absent"):
        MotionTaskTable.Frames(
            root_position=torch.zeros(2, 3),
            root_rotation=torch.zeros(2, 4),
            root_linear_velocity=torch.zeros(2, 3),
            root_angular_velocity=torch.zeros(2, 3),
            joint_position=torch.zeros(2, 1),
            joint_velocity=torch.zeros(2, 1),
            body_position=torch.zeros(2, 3, 3),
        )
    with pytest.raises(ValueError, match="contiguous"):
        MotionTaskTable.Frames(
            joint_position=torch.zeros(1, 2).expand(2, 2),
            joint_velocity=torch.zeros(2, 2),
            root_position=torch.zeros(2, 3),
            root_rotation=torch.zeros(2, 4),
            root_linear_velocity=torch.zeros(2, 3),
            root_angular_velocity=torch.zeros(2, 3),
        )


@pytest.mark.parametrize("field_name", ("joint_position", "root_rotation"))
def test_stream_writer_rejects_nonfinite_and_nonunit_physical_rows(field_name: str) -> None:
    """Invalid physical values must fail before entering immutable table storage."""
    index = _synthetic_index()
    writer = MotionTaskTable.writer(
        index,
        _empty_frames(index.total_frames),
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    frames = _clip_frames(0, index.clips[0].frame_count)
    if field_name == "joint_position":
        assert frames.joint_position is not None
        frames.joint_position[1, 0] = torch.nan
        match = "finite"
    else:
        assert frames.root_rotation is not None
        frames.root_rotation[1].zero_()
        match = "unit quaternion"

    writer.write_clip(index.clips[0].clip_id, frames)
    writer.write_clip(index.clips[1].clip_id, _clip_frames(1, index.clips[1].frame_count))
    with pytest.raises(ValueError, match=match):
        writer.finish()


def test_public_build_rejects_a_clip_index_from_another_source() -> None:
    """Stored provenance must come from the same source that yields the frames."""
    declared_index = _synthetic_index()
    actual_index = _synthetic_index(source_hash=_hash("different-source"))

    class Source:
        def inspect(self) -> MotionClipIndex:
            return actual_index

        def clips(self):
            raise AssertionError("Mismatched source provenance must fail before clip decoding.")

    class Builder:
        source_skeleton = _synthetic_skeleton()
        version = "synthetic_builder_v1"
        construction_identity_sha256 = _hash("synthetic-builder-construction-v1")
        joint_names = _SYNTHETIC_JOINT_NAMES
        reference_frame_names = _SYNTHETIC_REFERENCE_FRAME_NAMES

        def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
            raise AssertionError("Mismatched source provenance must fail before allocation.")

        def build_frames(self, fields, *, device: str | torch.device) -> MotionTaskTable.Frames:
            raise AssertionError("Mismatched source provenance must fail before frame construction.")

    with pytest.raises(ValueError, match="clip index identity"):
        MotionTaskTable.build(
            Source(),
            declared_index,
            Builder(),
            "source_frames",
            _SYNTHETIC_RESET_SOURCES,
            _SYNTHETIC_EXPERT_GRID,
            seed=7,
            device="cpu",
        )


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


def test_clip_index_retains_provenance_but_content_identity_ignores_path() -> None:
    index = _synthetic_index()
    moved = _synthetic_index(first_path="/different/root/clip_a.tensor")
    changed_fps = dataclasses.replace(
        index,
        clips=(dataclasses.replace(index.clips[0], source_fps=60.0), index.clips[1]),
    )

    assert index.clip_ids == ("clip_a", "clip_b")
    assert index.total_frames == 9
    assert index.identity_sha256 != moved.identity_sha256
    assert index.content_identity_sha256 == moved.content_identity_sha256
    assert index.content_identity_sha256 != changed_fps.content_identity_sha256


def test_frozen_g1_training_index_has_exact_capacity_without_loading_joblib() -> None:
    fixture = json.loads((_phase3_fixtures() / "native_motion_data_v1.json").read_text())
    g1 = fixture["collections"]["g1_lafan_50hz"]
    training = g1["training"]
    source_fps = float(g1["fields"]["fps"]["value"])
    clips = tuple(
        MotionClipIndex.Clip(
            clip_id=f"lafan_{clip_index:04d}",
            source_path=f"lafan/{clip_index:04d}",
            frame_count=training["frames_per_clip"],
            source_fps=source_fps,
            split="train",
            tags=(),
            content_sha256=_hash(f"lafan-{clip_index}"),
        )
        for clip_index in range(training["clips"])
    )
    index = MotionClipIndex(
        source_content_sha256=training["sha256"],
        skeleton_sha256=_hash("g1-29dof-source-skeleton"),
        semantic_level=g1["semantic_level"],
        license=g1["license"]["redistribution_gate"],
        clips=clips,
    )

    assert len(index.clips) == 862
    assert index.total_frames == training["frames"] == 258_600
    assert index.offsets[0] == 0 and index.offsets[-1] == 258_600
    assert all(right - left == 300 for left, right in zip(index.offsets, index.offsets[1:]))


def test_table_identity_invalidates_scientific_inputs_without_robot_schema() -> None:
    table = _synthetic_table()
    index = _synthetic_index()
    same = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    changed_source = MotionTaskTable.from_storage(
        _synthetic_index(source_hash=_hash("changed-source")),
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    changed_builder = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v2",
        _hash("synthetic-builder-construction-v2"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    changed_construction = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("changed-reference-kinematics-and-order"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    changed_mode = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "clip_time_ranges",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )

    changed_reset_name = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        (("motion", 0.7), ("fall", 0.3)),
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    changed_reset_probability = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        (("reference", 0.6), ("fall", 0.4)),
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )
    changed_grid = MotionTaskTable.from_storage(
        index,
        table.frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        MotionSampleGrid.uniform_before_source_end(step_seconds=0.02),
        seed=7,
    )

    assert same.cache_identity == table.cache_identity
    assert changed_source.cache_identity != table.cache_identity
    assert changed_builder.cache_identity != table.cache_identity
    assert changed_construction.cache_identity != table.cache_identity
    assert changed_mode.cache_identity != table.cache_identity
    assert changed_reset_name.cache_identity != table.cache_identity
    assert changed_reset_probability.cache_identity != table.cache_identity
    assert changed_grid.cache_identity != table.cache_identity


def test_task_row_modes_generate_exact_rows_and_expose_reset_source_names() -> None:
    source_frames = _synthetic_table()
    expected_times = torch.tensor((0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0, 4.0 / 30.0, 0.0, 0.05, 0.1, 0.15))

    assert source_frames.task_row_mode == "source_frames"
    assert source_frames.task_sampling_law == "clip_categorical_then_discrete_source_frame_v1"
    assert source_frames.reset_source_names == ("reference", "fall")
    torch.testing.assert_close(source_frames.clip_indices, torch.tensor((0, 0, 0, 0, 0, 1, 1, 1, 1)))
    torch.testing.assert_close(source_frames.clip_start_rows, torch.tensor((0, 5)))
    torch.testing.assert_close(source_frames.reset_time_ranges_seconds[:, 0], expected_times)
    torch.testing.assert_close(source_frames.reset_time_ranges_seconds[:, 1], expected_times)

    clip_ranges = MotionTaskTable.from_storage(
        source_frames.clip_index,
        source_frames.frames,
        source_frames.joint_names,
        source_frames.reference_frame_names,
        source_frames.frame_builder_version,
        source_frames.frame_builder_identity_sha256,
        "clip_time_ranges",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=source_frames.seed,
    )
    assert clip_ranges.task_row_mode == "clip_time_ranges"
    assert clip_ranges.task_sampling_law == "clip_categorical_then_continuous_time_v1"
    torch.testing.assert_close(clip_ranges.clip_indices, torch.tensor((0, 1)))
    torch.testing.assert_close(clip_ranges.clip_start_rows, torch.tensor((0, 1)))
    torch.testing.assert_close(
        clip_ranges.reset_time_ranges_seconds,
        torch.tensor(((0.0, 4.0 / 30.0), (0.0, 3.0 / 20.0))),
    )


def test_stream_writer_preallocates_once_and_finishes_without_copy() -> None:
    index = _synthetic_index()
    frames = _empty_frames(index.total_frames)
    pointers = {name: frames.field(name).data_ptr() for name in frames.stored_fields}
    writer = MotionTaskTable.writer(
        index,
        frames,
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )

    for clip_number, clip in enumerate(index.clips):
        writer.write_clip(clip.clip_id, _clip_frames(clip_number, clip.frame_count))
        assert {name: frames.field(name).data_ptr() for name in frames.stored_fields} == pointers
    table = writer.finish()

    assert table.frames is frames
    assert {name: table.field(name).data_ptr() for name in frames.stored_fields} == pointers
    assert table.clip_offsets.tolist() == [0, 5, 9]
    assert table.frame_counts.tolist() == [5, 4]
    assert table.source_fps.tolist() == [30.0, 20.0]
    assert table.clip_ids == index.clip_ids
    assert table.splits == ("train", "test")
    assert table.clip_valid.tolist() == [True, True]
    metadata = (
        table.clip_offsets,
        table.clip_start_rows,
        table._sampling_row_starts,
        table._sampling_row_counts,
        table.frame_counts,
        table.source_fps,
        table.clip_valid,
        table.clip_indices,
        table.reset_time_ranges_seconds,
        table.reset_source_probabilities,
        table.clip_priorities,
    )
    assert table.memory_bytes == table.frames.memory_bytes + sum(
        tensor.numel() * tensor.element_size() for tensor in metadata
    )


def test_storage_binding_reuses_tensors_and_seals_metadata() -> None:
    table = _synthetic_table()
    rebound = MotionTaskTable.from_storage(
        table.clip_index,
        table.frames,
        table.joint_names,
        table.reference_frame_names,
        table.frame_builder_version,
        table.frame_builder_identity_sha256,
        table.task_row_mode,
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=table.seed,
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
        MotionTaskTable.from_storage(
            table.clip_index,
            table.frames,
            table.joint_names,
            table.reference_frame_names,
            table.frame_builder_version,
            table.frame_builder_identity_sha256,
            table.task_row_mode,
            _SYNTHETIC_RESET_SOURCES,
            _SYNTHETIC_EXPERT_GRID,
            seed=table.seed,
        )


def test_stream_writer_rejects_order_shape_and_incomplete_tables() -> None:
    index = _synthetic_index()
    writer = MotionTaskTable.writer(
        index,
        _empty_frames(index.total_frames),
        _SYNTHETIC_JOINT_NAMES,
        _SYNTHETIC_REFERENCE_FRAME_NAMES,
        "synthetic_builder_v1",
        _hash("synthetic-builder-construction-v1"),
        "source_frames",
        _SYNTHETIC_RESET_SOURCES,
        _SYNTHETIC_EXPERT_GRID,
        seed=7,
    )

    with pytest.raises(ValueError, match="expected clip 'clip_a'"):
        writer.write_clip("clip_b", _clip_frames(1, 4))
    with pytest.raises(ValueError, match="incomplete"):
        writer.finish()

    wrong = dataclasses.replace(
        _clip_frames(0, 5),
        joint_position=torch.zeros(5, 1),
        joint_velocity=torch.zeros(5, 1),
    )
    with pytest.raises(ValueError, match="joint_position"):
        writer.write_clip("clip_a", wrong)


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
    torch.testing.assert_close(view.field("observation").squeeze(-1), torch.tensor([0.0, 1.0, 0.0]))
    expected_half_angle = torch.tensor(0.05)
    expected_rotation = torch.tensor([0.0, 0.0, torch.sin(expected_half_angle), torch.cos(expected_half_angle)])
    torch.testing.assert_close(view.field("root_rotation")[0], expected_rotation)


def test_expert_sample_grid_reproduces_native_meta_and_bfm_cardinality() -> None:
    meta = MotionSampleGrid.source_rows()
    bfm = MotionSampleGrid.uniform_before_source_end(step_seconds=0.02)

    assert meta.sample_count(frame_count=472, source_fps=30.0) == 472
    assert meta.window_count(frame_count=472, source_fps=30.0, length=8) == 464
    assert bfm.sample_count(frame_count=300, source_fps=30.0) == 499
    assert bfm.window_count(frame_count=300, source_fps=30.0, length=8) == 491


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
    return {
        "root_trans_offset": np.repeat(values[:, None], 3, axis=1),
        "pose_aa": np.repeat(values[:, None, None], 30 * 3, axis=1).reshape(frame_count, 30, 3),
        "dof": np.repeat(values[:, None], 29, axis=1),
        "root_rot": np.repeat(values[:, None], 4, axis=1),
        "smpl_joints": np.repeat(values[:, None, None], 24 * 3, axis=1).reshape(frame_count, 24, 3),
        "fps": 30,
        "motion_name": motion_name,
    }


def test_humenv_hdf5_import_preserves_explicit_caller_order_and_streams_files(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    paths = (tmp_path / "later_name.hdf5", tmp_path / "earlier_name.hdf5")
    for path, values in zip(paths, (_humenv_arrays(4, 10.0), _humenv_arrays(3, 20.0)), strict=True):
        with h5py.File(path, "w") as stream:
            episode = stream.create_group("ep_0")
            for name, value in values.items():
                episode.create_dataset(name, data=value)

    source = HumEnvHdf5Clips(
        paths,
        clip_ids=("caller_first", "caller_second"),
        source_fps=30.0,
        skeleton_sha256=_hash("smpl-skeleton"),
        split="train",
        license="test-only",
    )
    index = source.inspect()
    assert index.clip_ids == ("caller_first", "caller_second")
    assert index.offsets == (0, 4, 7)
    decoded = list(source.clips())
    assert [clip_id for clip_id, _ in decoded] == ["caller_first", "caller_second"]
    assert decoded[0][1]["observation"].dtype == np.float64
    assert decoded[1][1]["qpos"].shape == (3, 76)


def test_bfm_joblib_import_preserves_insertion_order_and_releases_each_clip(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm.pkl"
    joblib.dump(
        {
            "z_clip": _bfm_arrays(5, 10.0, "z_motion"),
            "a_clip": _bfm_arrays(4, 20.0, "a_motion"),
        },
        path,
    )

    with BfmG1JoblibClips.load(
        path,
        artifact_sha256=_file_hash(path),
        skeleton_sha256=_hash("g1-skeleton"),
        split="train",
        license="test-only",
    ) as source:
        index = source.inspect()
        assert index.clip_ids == ("z_clip", "a_clip")
        assert index.offsets == (0, 5, 9)
        assert source.remaining_clips == 2
        assert source.remaining_frames == 9

        clips = source.clips()
        first_id, first = next(clips)
        assert first_id == "z_clip"
        assert first["pose_aa"].shape == (5, 30, 3)
        assert type(first["pose_aa"]) is np.ndarray
        assert first["pose_aa"].flags.writeable
        assert source.remaining_clips == 1
        assert source.remaining_frames == 4
        second_id, _ = next(clips)
        assert second_id == "a_clip"
        assert source.remaining_clips == 0
        assert source.remaining_frames == 0
        with pytest.raises(StopIteration):
            next(clips)


def test_bfm_loaded_source_rejects_source_changes_before_consumption(tmp_path: Path) -> None:
    """The native source artifact must remain identical through frame building."""
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm.pkl"
    joblib.dump({"clip": _bfm_arrays(3, 10.0, "motion")}, path)
    source = BfmG1JoblibClips.load(
        path,
        artifact_sha256=_file_hash(path),
        skeleton_sha256=_hash("g1-skeleton"),
        split="train",
        license="test-only",
    )
    source.inspect()
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="changed"):
        next(source.clips())
    source.close()


def test_bfm_joblib_normalizes_native_variants_to_frame_builder_fields(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm_variants.pkl"
    training = _bfm_arrays(4, 10.0, "training_motion")
    evaluation = _bfm_arrays(3, 20.0, "unused")
    del evaluation["motion_name"]
    joblib.dump({"training_clip": training, "evaluation_clip": evaluation}, path)

    with BfmG1JoblibClips.load(
        path,
        artifact_sha256=_file_hash(path),
        skeleton_sha256=_hash("g1-skeleton"),
        split="train",
        license="test-only",
    ) as source:
        index = source.inspect()
        assert index.clips[0].tags == ("training_motion",)
        assert index.clips[1].tags == ()
        decoded = list(source.clips())

    assert [clip_id for clip_id, _ in decoded] == ["training_clip", "evaluation_clip"]
    assert all(tuple(fields) == ("root_trans_offset", "pose_aa", "fps") for _, fields in decoded)


def test_bfm_load_reuses_the_verified_artifact_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading must retain the after-load mutation check without rescanning first."""
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "bfm_verified.pkl"
    joblib.dump({"clip": _bfm_arrays(3, 10.0, "motion")}, path)
    artifact_sha256 = _file_hash(path)
    original_file_sha256 = bfm_g1_joblib_module.file_sha256
    scanned_paths: list[Path] = []

    def record_scan(source_path: Path) -> str:
        scanned_paths.append(source_path)
        return original_file_sha256(source_path)

    monkeypatch.setattr(bfm_g1_joblib_module, "file_sha256", record_scan)
    source = BfmG1JoblibClips.load(
        path,
        artifact_sha256=artifact_sha256,
        skeleton_sha256=_hash("g1-skeleton"),
        split="train",
        license="test-only",
    )

    assert scanned_paths == [path]
    source.close()
