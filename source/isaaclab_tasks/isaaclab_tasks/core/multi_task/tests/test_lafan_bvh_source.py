# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the simulator-free raw LAFAN1 BVH decoder."""

from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import pytest
import torch

from isaaclab.utils.math import convert_quat

import isaaclab_tasks.core.multi_task.motion.data.sources.lafan_bvh as lafan_bvh_module
from isaaclab_tasks.core.multi_task.kinematics import (
    kinematic_pose_forward,
    kinematic_tree_forward,
    ordered_hinge_rotation,
)
from isaaclab_tasks.core.multi_task.motion.data.sources.lafan_bvh import (
    LafanBvhHierarchy,
    LafanBvhZipClips,
    LafanClipRow,
    lafan_bvh_skeleton,
    lafan_clip_rows_content_sha256,
    load_lafan_bvh,
    write_lafan_clip_rows,
)

_MINIMAL_BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 100 200 300
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Knee
    {
        OFFSET 0 100 0
        CHANNELS 3 Zrotation Yrotation Xrotation
        End Site
        {
            OFFSET 0 50 0
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.033333
100 200 300 0 0 0 0 0 0
200 300 400 90 0 0 0 0 90
"""


_LAFAN_BIND_BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftUpLeg
    {
        OFFSET 0 0 10
        CHANNELS 3 Zrotation Yrotation Xrotation
    }
    JOINT RightUpLeg
    {
        OFFSET 0 0 -10
        CHANNELS 3 Zrotation Yrotation Xrotation
    }
    JOINT Spine
    {
        OFFSET 10 0 0
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT Spine1
        {
            OFFSET 10 0 0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT Spine2
            {
                OFFSET 10 0 0
                CHANNELS 3 Zrotation Yrotation Xrotation
            }
        }
    }
}
MOTION
Frames: 1
Frame Time: 0.033333
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""


def test_lafan_zero_channels_match_declared_rest_mechanics(tmp_path) -> None:
    """Zero BVH channels and declared source rest transforms describe one pose."""
    path = tmp_path / "bind.bvh"
    path.write_text(_LAFAN_BIND_BVH, encoding="utf-8")
    clip = load_lafan_bvh(path)
    skeleton = lafan_bvh_skeleton(clip.hierarchy)
    rest_translation = torch.tensor(skeleton.rest_translation_m, dtype=torch.float32)
    rest_rotation = convert_quat(torch.tensor(skeleton.rest_rotation_wxyz, dtype=torch.float32), to="xyzw")
    rest_position, rest_world_rotation = kinematic_tree_forward(
        rest_translation, rest_rotation, skeleton.parent_indices
    )
    root_position, local_rotation = clip.local_pose(skeleton, device="cpu")
    pose_position, pose_world_rotation = kinematic_pose_forward(
        rest_translation, rest_rotation, local_rotation, root_position, skeleton.parent_indices
    )

    torch.testing.assert_close(pose_position[0], rest_position, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(pose_world_rotation[0], rest_world_rotation, atol=1.0e-6, rtol=0.0)


def test_lafan_torso_semantics_bind_the_first_spine_joint(tmp_path) -> None:
    """The common torso role denotes the first articulated torso frame, not the upper chest."""
    path = tmp_path / "torso.bvh"
    path.write_text(_LAFAN_BIND_BVH, encoding="utf-8")
    skeleton = lafan_bvh_skeleton(load_lafan_bvh(path).hierarchy)
    torso = next(landmark for landmark in skeleton.landmarks if landmark.name == "torso")

    assert torso.position_body_name == "Spine"
    assert torso.rotation_body_name == "Spine"
    landmark_bodies = dict(lafan_bvh_module._LANDMARK_BODIES)
    assert {role: landmark_bodies[role] for role in ("spine", "chest", "neck", "left_thorax", "right_thorax")} == {
        "spine": "Spine1",
        "chest": "Spine2",
        "neck": "Neck",
        "left_thorax": "LeftShoulder",
        "right_thorax": "RightShoulder",
    }


def test_lafan_terminal_wrists_declare_orientation_without_source_endpoint_geometry() -> None:
    """Terminal wrists provide observed orientation while endpoint geometry remains target-owned."""
    hierarchy = LafanBvhHierarchy(
        body_names=(
            "Hips",
            "LeftFoot",
            "LeftToe",
            "RightFoot",
            "RightToe",
            "LeftForeArm",
            "LeftHand",
            "RightForeArm",
            "RightHand",
        ),
        parent_indices=(-1, 0, 1, 0, 3, 0, 5, 0, 7),
        rest_translation_m=(
            (0.0, 0.0, 0.0),
            (-0.1, -0.8, 0.0),
            (0.0, -0.2, 0.1),
            (0.1, -0.8, 0.0),
            (0.0, -0.2, 0.1),
            (-0.3, 0.0, 0.5),
            (0.25, 0.01, -0.02),
            (0.3, 0.0, 0.5),
            (0.24, -0.03, 0.04),
        ),
        channels=(
            ("Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation"),
            *((("Zrotation", "Yrotation", "Xrotation"),) * 8),
        ),
        end_sites=(
            LafanBvhHierarchy.EndSite(6, (0.0, 0.0, 0.0)),
            LafanBvhHierarchy.EndSite(8, (0.0, 0.0, 0.0)),
        ),
        source_sha256="0" * 64,
    )
    skeleton = lafan_bvh_skeleton(hierarchy)
    landmarks = {landmark.name: landmark for landmark in skeleton.landmarks}

    assert landmarks["left_wrist"].position_body_name == "LeftHand"
    assert landmarks["left_wrist"].rotation_body_name == "LeftHand"
    assert landmarks["right_wrist"].position_body_name == "RightHand"
    assert landmarks["right_wrist"].rotation_body_name == "RightHand"
    assert not hasattr(skeleton, "distal_points")


def test_load_lafan_bvh_preserves_hierarchy_channels_and_source_identity(tmp_path) -> None:
    """Per-file topology, offsets, channel order, and bytes remain explicit."""
    path = tmp_path / "minimal.bvh"
    path.write_text(_MINIMAL_BVH, encoding="utf-8")

    clip = load_lafan_bvh(path)

    assert clip.hierarchy.body_names == ("Hips", "Knee")
    assert clip.hierarchy.parent_indices == (-1, 0)
    assert clip.hierarchy.rotation_orders == ("zyx", "zyx")
    assert clip.hierarchy.channels[0] == (
        "Xposition",
        "Yposition",
        "Zposition",
        "Zrotation",
        "Yrotation",
        "Xrotation",
    )
    assert clip.hierarchy.rest_translation_m == ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert clip.hierarchy.end_sites == (clip.hierarchy.EndSite(1, (0.0, 0.5, 0.0)),)
    assert clip.hierarchy.source_sha256 == hashlib.sha256(_MINIMAL_BVH.encode()).hexdigest()


def test_load_lafan_bvh_converts_only_world_coordinates(tmp_path) -> None:
    """World axes become Z-up while non-root rotations remain parent-local."""
    path = tmp_path / "minimal.bvh"
    path.write_text(_MINIMAL_BVH, encoding="utf-8")

    clip = load_lafan_bvh(path)

    np.testing.assert_allclose(clip.root_position_m, ((1.0, -3.0, 2.0), (2.0, -4.0, 3.0)))
    half = 2.0**-0.5
    np.testing.assert_allclose(clip.local_rotation_xyzw[0, 0], (half, 0.0, 0.0, half), atol=1.0e-12)
    np.testing.assert_allclose(clip.local_rotation_xyzw[0, 1], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12)
    np.testing.assert_allclose(clip.local_rotation_xyzw[1, 1], (half, 0.0, 0.0, half), atol=1.0e-12)
    assert clip.frame_count == 2
    assert clip.source_fps == pytest.approx(30.000300003)
    assert not clip.root_position_m.flags.writeable
    assert not clip.local_rotation_xyzw.flags.writeable


def test_load_lafan_bvh_rejects_motion_rows_with_wrong_width(tmp_path) -> None:
    """A frame that violates its declared channel mechanics fails at the source boundary."""
    path = tmp_path / "broken.bvh"
    path.write_text(_MINIMAL_BVH.replace("0 0 90\n", "0 90\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong width"):
        load_lafan_bvh(path)


def test_lafan_bvh_coordinate_and_semantic_views_share_one_pose(tmp_path) -> None:
    """Raw Euler coordinates reconstruct the same local rotations exposed semantically."""
    path = tmp_path / "minimal.bvh"
    path.write_text(_MINIMAL_BVH, encoding="utf-8")
    clip = load_lafan_bvh(path)
    skeleton = lafan_bvh_skeleton(clip.hierarchy)

    root_position, local_rotation = clip.local_pose(skeleton, device="cpu")
    coordinates, velocity = clip.free_root_coordinates(skeleton, device="cpu")
    axes = torch.tensor(skeleton.joint_axes, dtype=torch.float32).view(skeleton.num_bodies - 1, 3, 3)
    reconstructed = ordered_hinge_rotation(coordinates[:, 7:].view(clip.frame_count, skeleton.num_bodies - 1, 3), axes)

    assert velocity is None
    torch.testing.assert_close(coordinates[:, :3], root_position)
    torch.testing.assert_close(coordinates[:, 3:7], local_rotation[:, 0])
    torch.testing.assert_close(reconstructed, local_rotation[:, 1:], atol=1.0e-6, rtol=1.0e-6)


def test_lafan_bvh_skeleton_excludes_root_offset_but_detects_mechanics_changes(tmp_path) -> None:
    """Starting root placement is clip state; non-root offsets and channel order are mechanics."""
    original = tmp_path / "original.bvh"
    root_changed = tmp_path / "root_changed.bvh"
    offset_changed = tmp_path / "offset_changed.bvh"
    channels_changed = tmp_path / "channels_changed.bvh"
    original.write_text(_MINIMAL_BVH, encoding="utf-8")
    root_changed.write_text(_MINIMAL_BVH.replace("OFFSET 100 200 300", "OFFSET -4 12 99"), encoding="utf-8")
    offset_changed.write_text(_MINIMAL_BVH.replace("OFFSET 0 100 0", "OFFSET 0 101 0"), encoding="utf-8")
    channels_changed.write_text(
        _MINIMAL_BVH.replace("CHANNELS 3 Zrotation Yrotation Xrotation", "CHANNELS 3 Xrotation Yrotation Zrotation"),
        encoding="utf-8",
    )

    original_skeleton = lafan_bvh_skeleton(load_lafan_bvh(original).hierarchy)
    root_skeleton = lafan_bvh_skeleton(load_lafan_bvh(root_changed).hierarchy)
    offset_skeleton = lafan_bvh_skeleton(load_lafan_bvh(offset_changed).hierarchy)
    channels_skeleton = lafan_bvh_skeleton(load_lafan_bvh(channels_changed).hierarchy)

    assert original_skeleton.identity_sha256 == root_skeleton.identity_sha256
    assert original_skeleton.identity_sha256 != offset_skeleton.identity_sha256
    assert original_skeleton.identity_sha256 != channels_skeleton.identity_sha256


def _bvh_frames(frame_count: int, *, root_offset: str = "100 200 300", knee_offset: str = "0 100 0") -> str:
    """Return one test BVH with observable motion and independently variable mechanics."""
    hierarchy = _MINIMAL_BVH.split("MOTION", 1)[0]
    hierarchy = hierarchy.replace("100 200 300", root_offset, 1).replace("0 100 0", knee_offset, 1)
    rows = "\n".join(f"{frame} 200 300 0 0 0 0 0 0" for frame in range(frame_count))
    return f"{hierarchy}MOTION\nFrames: {frame_count}\nFrame Time: 0.033333\n{rows}\n"


def _write_clip_rows(archive_path, rows_path, ranges):
    """Write concrete test rows and return their verified source identity."""
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        facts = {}
        for member in sorted({member for _, member, _, _ in ranges}):
            source_bytes = archive.read(member)
            clip = lafan_bvh_module.decode_lafan_bvh(source_bytes)
            facts[member] = (hashlib.sha256(source_bytes).hexdigest(), clip.frame_count, clip.source_fps)
        for clip_id, member, start, stop in ranges:
            source_sha256, frame_count, source_fps = facts[member]
            rows.append(LafanClipRow(clip_id, member, source_sha256, frame_count, source_fps, start, stop))
    rows = tuple(rows)
    write_lafan_clip_rows(rows_path, rows)
    zip_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    rows_sha256 = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    return rows, zip_sha256, rows_sha256, lafan_clip_rows_content_sha256(zip_sha256, rows)


def test_lafan_zip_source_slices_concrete_rows_and_decodes_each_member_once(tmp_path, monkeypatch) -> None:
    """Ordered windows share one member decode and retain exact source-frame provenance."""
    archive_path = tmp_path / "lafan1.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dance1_subject1.bvh", _bvh_frames(605))
        archive.writestr("walk1_subject1.bvh", _bvh_frames(301, root_offset="-4 12 99"))
    rows_path = tmp_path / "lafan_ground_train.csv"
    rows, zip_sha256, rows_sha256, source_sha256 = _write_clip_rows(
        archive_path,
        rows_path,
        (
            ("dance1_subject1_clip0", "dance1_subject1.bvh", 0, 300),
            ("dance1_subject1_clip1", "dance1_subject1.bvh", 300, 600),
            ("walk1_subject1_clip0", "walk1_subject1.bvh", 0, 300),
        ),
    )
    original_decode = lafan_bvh_module.decode_lafan_bvh
    decode_count = 0

    def counted_decode(source_bytes):
        nonlocal decode_count
        decode_count += 1
        return original_decode(source_bytes)

    monkeypatch.setattr(lafan_bvh_module, "decode_lafan_bvh", counted_decode)
    source = LafanBvhZipClips(
        archive_path,
        rows_path,
        verified_zip_sha256=zip_sha256,
        verified_rows_sha256=rows_sha256,
        expected_source_sha256=source_sha256,
    )
    index = source.inspect()
    decoded = list(source.clips((0, 1, 2)))

    assert index.clip_ids == tuple(row.clip_id for row in rows)
    expected_source_clips = ("dance1_subject1", "dance1_subject1", "walk1_subject1")
    assert tuple(clip.source_clip_id for clip in index.clips) == expected_source_clips
    assert tuple(clip.source_frame_start for clip in index.clips) == (0, 300, 0)
    assert [index for index, _ in decoded] == [0, 1, 2]
    assert decode_count == 2
    assert decoded[0][1].root_position_m.base is decoded[1][1].root_position_m.base
    np.testing.assert_allclose(decoded[0][1].root_position_m[[0, -1], 0], (0.0, 2.99))
    np.testing.assert_allclose(decoded[1][1].root_position_m[[0, -1], 0], (3.0, 5.99))


def test_lafan_zip_source_allows_one_bounded_window_without_redecoding_its_member(tmp_path) -> None:
    """Inspection may request a strict row subset while member decoding remains single-use."""
    archive_path = tmp_path / "lafan1.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dance1_subject1.bvh", _bvh_frames(605))
    rows_path = tmp_path / "lafan_ground_train.csv"
    _, zip_sha256, rows_sha256, source_sha256 = _write_clip_rows(
        archive_path,
        rows_path,
        (
            ("dance1_subject1_clip0", "dance1_subject1.bvh", 0, 300),
            ("dance1_subject1_clip1", "dance1_subject1.bvh", 300, 600),
        ),
    )
    source = LafanBvhZipClips(
        archive_path,
        rows_path,
        verified_zip_sha256=zip_sha256,
        verified_rows_sha256=rows_sha256,
        expected_source_sha256=source_sha256,
    )

    decoded = list(source.clips((0,)))

    assert [clip_index for clip_index, _clip in decoded] == [0]
    assert decoded[0][1].frame_count == 300
    with pytest.raises(ValueError, match="members may be decoded only once"):
        list(source.clips((1,)))


def test_lafan_inspect_reads_bounded_headers_and_streams_multiple_mechanics_groups(tmp_path, monkeypatch) -> None:
    """Inspect scales with member headers while source groups decode each member once."""
    archive_path = tmp_path / "lafan1.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dance1_subject1.bvh", _bvh_frames(10_001))
        archive.writestr("fight1_subject1.bvh", _bvh_frames(11_001, knee_offset="0 101 0"))
        archive.writestr("walk1_subject1.bvh", _bvh_frames(12_001, root_offset="-4 12 99"))
    rows_path = tmp_path / "lafan_ground_evaluation.csv"
    _, zip_sha256, rows_sha256, source_sha256 = _write_clip_rows(
        archive_path,
        rows_path,
        (
            ("dance1_subject1", "dance1_subject1.bvh", 0, 10_001),
            ("fight1_subject1", "fight1_subject1.bvh", 0, 11_001),
            ("walk1_subject1", "walk1_subject1.bvh", 0, 12_001),
        ),
    )
    original_decode = lafan_bvh_module.decode_lafan_bvh
    original_header = lafan_bvh_module._read_lafan_bvh_hierarchy
    original_read = zipfile.ZipFile.read
    header_bytes = []

    def reject_full_decode(_source_bytes):
        raise AssertionError("inspect must not decode motion arrays")

    def reject_full_read(*_args, **_kwargs):
        raise AssertionError("inspect must not read a complete zip member")

    def measured_header(stream, source_sha256):
        start = stream.tell()
        hierarchy = original_header(stream, source_sha256)
        header_bytes.append(stream.tell() - start)
        return hierarchy

    monkeypatch.setattr(lafan_bvh_module, "decode_lafan_bvh", reject_full_decode)
    monkeypatch.setattr(lafan_bvh_module, "_read_lafan_bvh_hierarchy", measured_header)
    monkeypatch.setattr(zipfile.ZipFile, "read", reject_full_read)
    source = LafanBvhZipClips(
        archive_path,
        rows_path,
        verified_zip_sha256=zip_sha256,
        verified_rows_sha256=rows_sha256,
        expected_source_sha256=source_sha256,
    )
    index = source.inspect()

    assert tuple(clip.skeleton_id for clip in index.clips) == (0, 1, 0)
    assert index.skeleton_identity_sha256s[0] != index.skeleton_identity_sha256s[1]
    assert len(header_bytes) == 3
    assert max(header_bytes) < 2_000
    assert sum(header_bytes) * 100 < sum(info.file_size for info in zipfile.ZipFile(archive_path).infolist())

    decode_count = 0

    def counted_decode(source_bytes):
        nonlocal decode_count
        decode_count += 1
        return original_decode(source_bytes)

    monkeypatch.setattr(zipfile.ZipFile, "read", original_read)
    monkeypatch.setattr(lafan_bvh_module, "decode_lafan_bvh", counted_decode)
    first = list(source.clips((1,)))
    second = list(source.clips((0, 2)))

    assert [index for index, _clip in first] == [1]
    assert [index for index, _clip in second] == [0, 2]
    assert sorted(index for index, _clip in (*first, *second)) == [0, 1, 2]
    assert decode_count == 3


def test_lafan_clip_rows_reject_all_member_and_unordered_window_behavior(tmp_path) -> None:
    """Concrete rows, not archive membership, define selected clips and their stable order."""
    archive_path = tmp_path / "lafan1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("aiming1_subject1.bvh", _bvh_frames(301))
        archive.writestr("walk1_subject1.bvh", _bvh_frames(605))
    rows_path = tmp_path / "lafan_ground_evaluation.csv"
    _, zip_sha256, rows_sha256, source_sha256 = _write_clip_rows(
        archive_path,
        rows_path,
        (("walk1_subject1", "walk1_subject1.bvh", 0, 605),),
    )
    source = LafanBvhZipClips(
        archive_path,
        rows_path,
        verified_zip_sha256=zip_sha256,
        verified_rows_sha256=rows_sha256,
        expected_source_sha256=source_sha256,
    )

    assert source.inspect().clip_ids == ("walk1_subject1",)
    with pytest.raises(ValueError, match="sorted, and unique"):
        list(source.clips((0, 0)))


def test_raw_lafan_config_declares_ground_rows_and_official_zip_dependency() -> None:
    """The old 77-member train/evaluation behavior cannot satisfy the frozen source config."""
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionSourcesCfg

    source = MotionSourcesCfg().lafan
    source_artifacts = (source.train.artifact, source.evaluation.artifact, source.dependencies[0].artifact)
    assert all(".pkl" not in artifact for artifact in source_artifacts)
    assert (source.train.clip_count, source.train.frame_count) == (862, 258_600)
    assert (source.evaluation.clip_count, source.evaluation.frame_count) == (40, 264_705)
    assert source.decoder_version == "lafan_bvh_clip_rows_v4"
    assert source.train.artifact == "lafan_ground_train.csv"
    assert source.evaluation.artifact == "lafan_ground_evaluation.csv"
    assert tuple((item.name, item.artifact) for item in source.dependencies) == (("lafan_zip", "lafan1.zip"),)
