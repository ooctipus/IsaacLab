# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the native AMASS SMPL-H decoder."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import isaaclab_tasks.core.multi_task.motion.data.sources.amass_smplh as amass_module
from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsBuildCfg, ordered_hinge_rotation
from isaaclab_tasks.core.multi_task.motion.data import MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.data.frames import (
    MotionGeneralizedCoordinates,
    MotionSourceProjectionAnalytic,
    MotionSourceProjectionExact,
    MotionSourceProjectionTrajectory,
)
from isaaclab_tasks.core.multi_task.motion.data.smpl import SmplLbsModel, load_smpl_lbs_model
from isaaclab_tasks.core.multi_task.motion.data.sources.amass_smplh import (
    AmassClipRow,
    AmassSmplhClips,
    load_amass_smplh_clip,
    smpl_body_core_mechanics,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionTaskTableCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
    MotionAnalyticFamilyCfg,
    MotionExactFamilyCfg,
    MotionTrajectoryFamilyCfg,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import _plan_motion_group
from isaaclab_tasks.core.multi_task.motion.retarget import motion_contact_probe_offsets


def _skeleton() -> MotionSkeleton:
    body_names = tuple(f"body_{index}" for index in range(52))
    joint_names = tuple(f"body_{body}_{axis}" for body in range(1, 52) for axis in "xyz")
    xyz = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return MotionSkeleton(
        identifier="synthetic_smplh",
        content_sha256=hashlib.sha256(b"synthetic-smplh").hexdigest(),
        body_names=body_names,
        parent_indices=(-1, *range(51)),
        rest_translation_m=((0.0, 0.0, 0.0), *((0.0, 0.0, 0.1),) * 51),
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0),) * 52,
        joint_names=joint_names,
        joint_child_body_indices=tuple(body for body in range(1, 52) for _ in range(3)),
        joint_axes=xyz * 51,
        root_translation_frame="world",
        root_rotation_convention="xyzw",
        landmark_rotation_policy="calibrated_body",
    )


def _write_clip(
    path: Path,
    *,
    frame_count: int = 5,
    source_fps: float = 120.0,
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    translation = np.arange(frame_count * 3, dtype=np.float64).reshape(frame_count, 3) * 0.01
    pose = np.linspace(-0.2, 0.2, frame_count * 156, dtype=np.float64).reshape(frame_count, 156)
    if betas is None:
        betas = np.linspace(-0.1, 0.1, 16, dtype=np.float64)
    np.savez(
        path,
        trans=translation,
        gender=np.array("male"),
        mocap_framerate=np.array(source_fps, dtype=np.float64),
        betas=betas,
        dmpls=np.zeros((frame_count, 8), dtype=np.float64),
        poses=pose,
    )
    return translation, pose


def _amass_source(
    paths: tuple[str | Path, ...],
    models_by_gender: dict[str, SmplLbsModel] | None = None,
) -> AmassSmplhClips:
    """Build one concrete-row test source without a compatibility constructor."""
    source_paths = tuple(Path(path) for path in paths)
    data_root = source_paths[0].parent
    if any(path.parent != data_root for path in source_paths):
        raise ValueError("Synthetic AMASS clips must share one test data root.")
    rows = tuple(AmassClipRow.from_file(path.name, path) for path in source_paths)
    models = (
        {gender: _smpl_model(gender) for gender in ("female", "male", "neutral")}
        if models_by_gender is None
        else models_by_gender
    )
    return AmassSmplhClips(data_root, rows, models_by_gender=models)


_SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
# Twelve raw/prepared CMU pairs bound ordinary float32 conversion error below
# these values. Root angular velocity is less tight because the released law
# applies acos(2*w^2-1) near w=1, amplifying matrix-to-quaternion roundoff.
_HUMENV_QPOS_ATOL = 2.0e-6
_HUMENV_ROOT_LINEAR_ATOL = 2.0e-5
_HUMENV_ROOT_ANGULAR_ATOL = 8.0e-3
_HUMENV_DOF_VELOCITY_ATOL = 1.0e-4


def _smpl_model(gender: str) -> SmplLbsModel:
    vertex_count = 24
    template = torch.zeros((vertex_count, 3), dtype=torch.float32)
    template[:, 0] = torch.linspace(0.0, 0.4, vertex_count)
    template[:, 2] = torch.linspace(0.0, 1.0, vertex_count)
    shape_directions = torch.zeros((vertex_count, 3, 10), dtype=torch.float32)
    shape_directions[:, 1, 0] = torch.linspace(0.0, 0.1, vertex_count)
    return SmplLbsModel(
        gender=gender,
        source_sha256=hashlib.sha256(f"synthetic-{gender}".encode()).hexdigest(),
        artifact_sha256=hashlib.sha256(f"synthetic-artifact-{gender}".encode()).hexdigest(),
        vertex_template_m=template,
        shape_blend_directions_m=shape_directions,
        pose_blend_directions_m=torch.zeros((vertex_count, 3, 207), dtype=torch.float32),
        joint_regressor=torch.eye(vertex_count, dtype=torch.float32),
        skinning_weights=torch.eye(vertex_count, dtype=torch.float32),
        parent_indices=torch.tensor(_SMPL_PARENTS, dtype=torch.int64),
    )


def _write_smpl_model(path: Path, model: SmplLbsModel) -> None:
    np.savez(
        path,
        format_version=np.array("smpl_lbs_v1"),
        gender=np.array(model.gender),
        source_sha256=np.array(model.source_sha256),
        vertex_template_m=model.vertex_template_m.numpy(),
        shape_blend_directions_m=model.shape_blend_directions_m.numpy(),
        pose_blend_directions_m=model.pose_blend_directions_m.numpy(),
        joint_regressor=model.joint_regressor.numpy(),
        skinning_weights=model.skinning_weights.numpy(),
        parent_indices=model.parent_indices.numpy(),
    )


def _write_target_neutral_calibration(root: Path, model: SmplLbsModel | None = None) -> tuple[str, str]:
    """Write one target-owned neutral calibration artifact and return its relative identity."""
    path = root / "target_smpl_neutral.npz"
    _write_smpl_model(path, _smpl_model("neutral") if model is None else model)
    return path.name, hashlib.sha256(path.read_bytes()).hexdigest()


def _target_calibration(
    artifact: str,
    artifact_sha256: str,
) -> MotionTaskTableCfg.TargetKinematicsCfg.CalibrationCfg:
    """Declare one target-owned calibration test artifact."""
    return MotionTaskTableCfg.TargetKinematicsCfg.CalibrationCfg(
        artifact=artifact,
        artifact_sha256=artifact_sha256,
    )


@pytest.mark.parametrize("source_fps", (60.0, 120.0))
def test_load_amass_smplh_clip_preserves_native_rows(tmp_path, source_fps: float):
    path = tmp_path / "clip_poses.npz"
    translation, pose = _write_clip(path, source_fps=source_fps)

    clip = load_amass_smplh_clip(path)

    assert clip.frame_count == translation.shape[0]
    assert clip.source_fps == source_fps
    np.testing.assert_array_equal(clip.root_translation_m, translation)
    np.testing.assert_array_equal(clip.local_axis_angle_rad, pose.reshape(-1, 52, 3))


def test_amass_smplh_semantic_and_hinge_views_describe_the_same_pose(tmp_path):
    path = tmp_path / "clip_poses.npz"
    _write_clip(path)
    clip = load_amass_smplh_clip(path)
    skeleton = _skeleton()

    root_position, local_rotation = clip.local_pose(skeleton, device="cpu")
    coordinates, velocity = clip.free_root_coordinates(skeleton, device="cpu")
    fitted = ordered_hinge_rotation(
        coordinates[:, 7:].reshape(clip.frame_count, 51, 3),
        torch.eye(3, dtype=torch.float32),
    )

    assert velocity is None
    torch.testing.assert_close(coordinates[:, :3], root_position)
    torch.testing.assert_close(coordinates[:, 3:7], local_rotation[:, 0])
    torch.testing.assert_close(fitted, local_rotation[:, 1:], atol=2.0e-5, rtol=2.0e-5)


def test_load_amass_smplh_clip_rejects_non_native_fields(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, trans=np.zeros((2, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="fields differ"):
        load_amass_smplh_clip(path)


def test_smpl_lbs_zero_pose_preserves_the_shaped_template():
    model = _smpl_model("neutral")
    betas = torch.zeros((1, 10), dtype=torch.float32)

    vertices = model.vertices(
        torch.zeros((1, 24, 3), dtype=torch.float32),
        betas,
        torch.zeros((1, 3), dtype=torch.float32),
    )

    torch.testing.assert_close(vertices[0], model.vertex_template_m, atol=2.0e-7, rtol=0.0)


def test_smpl_body_core_mechanics_reuses_the_compact_shape_model():
    model = _smpl_model("male")
    mechanics = smpl_body_core_mechanics(model)
    zero = mechanics.skeleton(np.zeros(16, dtype=np.float64))
    shaped_betas = np.zeros(16, dtype=np.float64)
    shaped_betas[0] = 1.0
    shaped = mechanics.skeleton(shaped_betas)
    landmarks = {landmark.name: landmark for landmark in zero.landmarks}

    assert mechanics.content_sha256 != model.source_sha256
    assert mechanics.pose_body_indices == tuple(range(22))
    assert zero.body_names == shaped.body_names
    assert zero.identity_sha256 != shaped.identity_sha256
    assert zero.rest_translation_m != shaped.rest_translation_m
    assert landmarks["left_ankle"].rotation_body_name == "L_Ankle"
    assert landmarks["right_ankle"].rotation_body_name == "R_Ankle"
    assert landmarks["left_wrist"].rotation_body_name == "L_Wrist"
    assert landmarks["right_wrist"].rotation_body_name == "R_Wrist"
    assert {
        role: landmarks[role].position_body_name for role in ("spine", "chest", "neck", "left_thorax", "right_thorax")
    } == {
        "spine": "Spine",
        "chest": "Chest",
        "neck": "Neck",
        "left_thorax": "L_Thorax",
        "right_thorax": "R_Thorax",
    }
    assert not hasattr(zero, "distal_points")


def test_amass_source_groups_shapes_and_streams_requested_clips(tmp_path):
    """Shape-specific source skeletons stay indexed while raw 52-row poses stream lazily."""
    first = tmp_path / "first_poses.npz"
    second = tmp_path / "second_poses.npz"
    third = tmp_path / "third_poses.npz"
    _write_clip(first)
    _write_clip(second)
    changed_betas = np.linspace(-0.1, 0.1, 16, dtype=np.float64)
    changed_betas[0] += 0.25
    _write_clip(third, betas=changed_betas)
    paths = (first, second, third)
    source = _amass_source(paths)

    index = source.inspect()
    decoded = list(source.clips((2, 0)))

    assert index.clip_ids == ("first_poses.npz", "second_poses.npz", "third_poses.npz")
    assert tuple(clip.skeleton_id for clip in index.clips) == (0, 0, 1)
    assert index.skeleton_ids == (0, 1)
    assert index.for_skeleton(0) == (0, 1)
    assert [clip_index for clip_index, _ in decoded] == [2, 0]
    for clip_index, clip in decoded:
        skeleton_id = index.clips[clip_index].skeleton_id
        skeleton = source.skeleton(skeleton_id)
        root_position, local_rotation = clip.local_pose(skeleton, device="cpu")
        assert clip.local_axis_angle_rad.shape[1:] == (52, 3)
        assert clip.pose_body_indices == tuple(range(22))
        assert root_position.shape == (clip.frame_count, 3)
        assert local_rotation.shape == (clip.frame_count, 22, 4)


def test_amass_source_inspection_does_not_decode_motion(tmp_path):
    """Concrete source metadata makes inspection independent of full motion arrays."""
    first = tmp_path / "first_poses.npz"
    second = tmp_path / "second_poses.npz"
    _write_clip(first)
    _write_clip(second, frame_count=7)
    paths = (first, second)
    source = _amass_source(paths)
    first.unlink()
    second.unlink()

    index = source.inspect()

    assert index.clip_ids == ("first_poses.npz", "second_poses.npz")
    assert index.total_frames == 12


def test_amass_source_decodes_only_selected_rows_once(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Unselected files remain untouched and one row cannot be decoded twice."""
    first = tmp_path / "first_poses.npz"
    second = tmp_path / "second_poses.npz"
    _write_clip(first)
    _write_clip(second)
    source = _amass_source((first, second))
    load_verified = amass_module._load_verified_amass_smplh_clip
    loaded = []

    def count_load(path: Path, expected_sha256: str):
        loaded.append(path)
        return load_verified(path, expected_sha256)

    monkeypatch.setattr(amass_module, "_load_verified_amass_smplh_clip", count_load)

    decoded = list(source.clips((1,)))

    assert [index for index, _ in decoded] == [1]
    assert loaded == [second]
    with pytest.raises(ValueError, match="decoded only once"):
        list(source.clips((1,)))


def _motion_contact_channels() -> tuple[MotionTaskTableCfg.ContactChannelCfg, ...]:
    """Return the canonical two-probe foot contact channels."""
    return (
        MotionTaskTableCfg.ContactChannelCfg(name="left_foot", source_probe_roles=("left_ankle", "left_toe")),
        MotionTaskTableCfg.ContactChannelCfg(name="right_foot", source_probe_roles=("right_ankle", "right_toe")),
    )


def _motion_contact_offsets(target) -> torch.Tensor:
    """Build canonical source-probe offsets on the target device."""
    channels = _motion_contact_channels()
    return motion_contact_probe_offsets(channels, target.kinematics.device)


def _smpl_contact_patches() -> tuple[MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg, ...]:
    """Return the real SMPL contact features used by motion presets."""
    return (
        MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
            channel="left_foot",
            body_name="L_Ankle",
        ),
        MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
            channel="right_foot",
            body_name="R_Ankle",
        ),
    )


def _smpl_scene_kinematics() -> NewtonKinematics:
    """Build target mechanics from the same scene-owned articulation used by the environment."""
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import SMPL_MOTION_ARTICULATION_CFG

    return NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), SMPL_MOTION_ARTICULATION_CFG, "cpu"
    )


def test_raw_amass_smpl_uses_compatible_analytic_route_at_target_clock(tmp_path) -> None:
    """Raw AMASS converts compatible poses directly at the declared target clock."""
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        smpl_frame_target,
        smpl_source_projection,
    )

    path = tmp_path / "raw_clip_poses.npz"
    _write_clip(path, frame_count=12, source_fps=120.0)
    source = _amass_source((path,))
    index = source.inspect()
    calibration_artifact, calibration_sha256 = _write_target_neutral_calibration(tmp_path)
    target = smpl_frame_target(
        _smpl_scene_kinematics(),
        _smpl_contact_patches(),
        calibration_artifact_root=str(tmp_path),
        calibration=_target_calibration(calibration_artifact, calibration_sha256),
    )
    table_cfg = SimpleNamespace(
        target_kinematics=SimpleNamespace(source_projection_factory=smpl_source_projection),
        contact_channels=_motion_contact_channels(),
        families=(MotionExactFamilyCfg(), MotionAnalyticFamilyCfg(), MotionTrajectoryFamilyCfg()),
    )
    plan = _plan_motion_group(
        table_cfg,
        source,
        index,
        0,
        source.skeleton(0),
        target,
        _motion_contact_offsets(target),
    )

    _, clip = next(source.clips((0,)))
    coordinates = plan.projection.convert_clip(clip)

    assert isinstance(plan.projection, MotionSourceProjectionAnalytic)
    assert plan.family_name == "analytic"
    assert plan.source_clip_indices == (0,)
    assert plan.output_index.clips[0].frame_count == 3
    assert plan.output_index.clips[0].source_fps == 30.0
    assert plan.output_index.clips[0].clip_id == index.clips[0].clip_id
    assert plan.output_index.clips[0].source_frame_start == 0
    assert coordinates.joint_q.shape == (3, 76)


def test_smpl_exact_route_precedes_the_compatible_source_protocol() -> None:
    """An exact skeleton must bypass compatible-pose mechanics and calibration."""
    from isaaclab_tasks.core.multi_task.motion.data.sources import cmu_humenv_smpl_skeleton
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        smpl_frame_target,
        smpl_source_projection,
    )

    class CompatibleSource:
        compatible_pose_profile_sha256 = AmassSmplhClips.compatible_pose_profile_sha256

        def smpl_subject_model(self, *_args):
            raise AssertionError("Exact projection must not resolve compatible-pose mechanics.")

    target = smpl_frame_target(_smpl_scene_kinematics(), _smpl_contact_patches())
    source = cmu_humenv_smpl_skeleton()
    projection = smpl_source_projection(
        source,
        target,
        CompatibleSource(),
        _motion_contact_channels(),
        _motion_contact_offsets(target),
    )

    assert isinstance(projection, MotionSourceProjectionExact)


def test_smpl_target_compatible_pose_map_resamples_the_native_clock(tmp_path):
    """The reference converter selects the HumEnv output clock and velocity law."""
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        _compatible_pose_coordinates,
        _compatible_pose_output_index,
        smpl_frame_target,
    )

    path = tmp_path / "clip_poses.npz"
    translation, pose = _write_clip(path, frame_count=8, source_fps=60.0)
    source = _amass_source((path,))
    index = source.inspect()
    calibration_artifact, calibration_sha256 = _write_target_neutral_calibration(tmp_path)
    target = smpl_frame_target(
        _smpl_scene_kinematics(),
        _smpl_contact_patches(),
        calibration_artifact_root=str(tmp_path),
        calibration=_target_calibration(calibration_artifact, calibration_sha256),
    )
    _, clip = next(source.clips((0,)))
    coordinates = _compatible_pose_coordinates(
        clip,
        subject_model=source.smpl_subject_model(source.skeleton(0).identity_sha256, target.kinematics.device),
        neutral_model=target.neutral_calibration_model(),
        target_body_names=tuple(target.kinematics.body_names),
        target_coordinate_names=target.reference_coordinate_names,
    )
    output_index = _compatible_pose_output_index(index)
    qpos = coordinates.joint_q
    qvel = coordinates.joint_qd

    assert qvel is not None
    assert output_index.clips[0].frame_count == 4
    assert qpos.shape == (4, 76)
    assert qvel.shape == (4, 75)
    torch.testing.assert_close(qpos[:, :2], torch.from_numpy(translation[::2, :2]).float())
    root_rotation = torch.from_numpy(pose[::2, :3].reshape(4, 3)).float()
    root_xyzw = torch.cat(
        (
            root_rotation
            * (torch.sin(0.5 * root_rotation.norm(dim=-1, keepdim=True)) / root_rotation.norm(dim=-1, keepdim=True)),
            torch.cos(0.5 * root_rotation.norm(dim=-1, keepdim=True)),
        ),
        dim=-1,
    )
    torch.testing.assert_close(qpos[:, 3:7], torch.cat((root_xyzw[:, 3:], root_xyzw[:, :3]), dim=-1))
    expected_dof_velocity = (qpos[1:, 7:] - qpos[:-1, 7:]) * 30.0
    torch.testing.assert_close(qvel[:-1, 6:], expected_dof_velocity)
    torch.testing.assert_close(qvel[-1, :3], qvel[-2, :3])
    torch.testing.assert_close(qvel[-1, 6:], qvel[-2, 6:])


def test_smpl_analytic_projection_identity_tracks_target_calibration(tmp_path):
    """The analytic identity includes its target-owned neutral calibration mechanics."""
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        smpl_frame_target,
        smpl_source_projection,
    )

    path = tmp_path / "clip_poses.npz"
    _write_clip(path)
    source = _amass_source((path,))
    reference = _smpl_scene_kinematics()

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_artifact, first_sha256 = _write_target_neutral_calibration(first_root)
    second_model = _smpl_model("neutral")
    second_model = replace(second_model, vertex_template_m=(second_model.vertex_template_m + 0.001).contiguous())
    second_artifact, second_sha256 = _write_target_neutral_calibration(second_root, second_model)
    first_target = smpl_frame_target(
        reference,
        _smpl_contact_patches(),
        calibration_artifact_root=str(first_root),
        calibration=_target_calibration(first_artifact, first_sha256),
    )
    second_target = smpl_frame_target(
        reference,
        _smpl_contact_patches(),
        calibration_artifact_root=str(second_root),
        calibration=_target_calibration(second_artifact, second_sha256),
    )

    assert first_target.construction_identity_sha256 == second_target.construction_identity_sha256
    first_projection = smpl_source_projection(
        source.skeleton(0), first_target, source, _motion_contact_channels(), _motion_contact_offsets(first_target)
    )
    second_projection = smpl_source_projection(
        source.skeleton(0), second_target, source, _motion_contact_channels(), _motion_contact_offsets(second_target)
    )
    assert isinstance(first_projection, MotionSourceProjectionAnalytic)
    assert isinstance(second_projection, MotionSourceProjectionAnalytic)
    assert first_projection.construction_identity_sha256 != second_projection.construction_identity_sha256


def test_smpl_wrong_compatible_pose_digest_uses_trajectory_fallback(tmp_path):
    """A pose-compatible-looking source must match the exact profile digest."""
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        smpl_frame_target,
        smpl_source_projection,
    )

    path = tmp_path / "clip_poses.npz"
    _write_clip(path)
    source = _amass_source((path,))
    calibration_artifact, calibration_sha256 = _write_target_neutral_calibration(tmp_path)
    target = smpl_frame_target(
        _smpl_scene_kinematics(),
        _smpl_contact_patches(),
        calibration_artifact_root=str(tmp_path),
        calibration=_target_calibration(calibration_artifact, calibration_sha256),
    )
    wrong_profile = SimpleNamespace(
        compatible_pose_profile_sha256="0" * 64,
        smpl_subject_model=source.smpl_subject_model,
    )

    projection = smpl_source_projection(
        source.skeleton(0), target, wrong_profile, _motion_contact_channels(), _motion_contact_offsets(target)
    )

    assert isinstance(projection, MotionSourceProjectionTrajectory)


def test_smpl_analytic_route_requires_target_neutral_calibration(tmp_path):
    """Compatible-pose conversion fails clearly when target mechanics are absent."""
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        smpl_frame_target,
        smpl_source_projection,
    )

    path = tmp_path / "clip_poses.npz"
    _write_clip(path)
    source = _amass_source((path,))
    target = smpl_frame_target(_smpl_scene_kinematics(), _smpl_contact_patches())

    with pytest.raises(ValueError, match="target-owned neutral calibration mechanics"):
        smpl_source_projection(
            source.skeleton(0),
            target,
            source,
            _motion_contact_channels(),
            _motion_contact_offsets(target),
        )


def test_load_compact_smpl_model_round_trips_without_body_model_dependency(tmp_path):
    expected = _smpl_model("neutral")
    path = tmp_path / "neutral.npz"
    _write_smpl_model(path, expected)
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    actual = load_smpl_lbs_model(path, artifact_sha256=artifact_sha256)
    repeated = load_smpl_lbs_model(path, artifact_sha256=artifact_sha256)

    assert repeated is actual
    assert actual.gender == expected.gender
    assert actual.source_sha256 == expected.source_sha256
    assert actual.artifact_sha256 == artifact_sha256
    torch.testing.assert_close(actual.vertex_template_m, expected.vertex_template_m)
    torch.testing.assert_close(actual.pose_blend_directions_m, expected.pose_blend_directions_m)


def test_load_compact_smpl_model_rejects_an_unverified_artifact(tmp_path):
    path = tmp_path / "neutral.npz"
    _write_smpl_model(path, _smpl_model("neutral"))

    with pytest.raises(ValueError, match="artifact hash differs"):
        load_smpl_lbs_model(path, artifact_sha256="0" * 64)


def test_raw_amass_matches_local_prepared_humenv_specimen():
    raw_path = os.environ.get("ISAACLAB_AMASS_RAW_CLIP")
    prepared_path = os.environ.get("ISAACLAB_HUMENV_PREPARED_CLIP")
    subject_model_path = os.environ.get("ISAACLAB_SMPL_SUBJECT_MODEL")
    neutral_model_path = os.environ.get("ISAACLAB_SMPL_NEUTRAL_MODEL")
    if None in (raw_path, prepared_path, subject_model_path, neutral_model_path):
        pytest.skip("Set the four ISAACLAB raw-AMASS parity paths to run the licensed-data integration oracle.")

    h5py = pytest.importorskip("h5py")
    subject_model = load_smpl_lbs_model(
        subject_model_path,
        artifact_sha256=hashlib.sha256(Path(subject_model_path).read_bytes()).hexdigest(),
    )
    neutral_model = load_smpl_lbs_model(
        neutral_model_path,
        artifact_sha256=hashlib.sha256(Path(neutral_model_path).read_bytes()).hexdigest(),
    )
    models = {gender: _smpl_model(gender) for gender in ("female", "male", "neutral")}
    models[subject_model.gender] = subject_model
    models["neutral"] = neutral_model
    source = _amass_source((raw_path,), models)
    index = source.inspect()
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
        _compatible_pose_coordinates,
        _compatible_pose_output_index,
        smpl_frame_target,
    )

    target = smpl_frame_target(
        _smpl_scene_kinematics(),
        _smpl_contact_patches(),
        calibration_artifact_root=str(Path(neutral_model_path).parent),
        calibration=_target_calibration(Path(neutral_model_path).name, neutral_model.artifact_sha256),
    )
    _, clip = next(source.clips((0,)))
    coordinates = _compatible_pose_coordinates(
        clip,
        subject_model=source.smpl_subject_model(source.skeleton(0).identity_sha256, target.kinematics.device),
        neutral_model=target.neutral_calibration_model(),
        target_body_names=tuple(target.kinematics.body_names),
        target_coordinate_names=target.reference_coordinate_names,
    )
    qpos = coordinates.joint_q
    qvel = coordinates.joint_qd
    output_index = _compatible_pose_output_index(index)
    assert qvel is not None
    with h5py.File(prepared_path, "r") as archive:
        expected_qpos = torch.from_numpy(archive["ep_0/qpos"][:])
        expected_qvel = torch.from_numpy(archive["ep_0/qvel"][:])

    assert output_index.clips[0].frame_count == expected_qpos.shape[0]
    assert output_index.clips[0].source_fps == 30.0
    torch.testing.assert_close(qpos, expected_qpos, atol=_HUMENV_QPOS_ATOL, rtol=0.0)
    torch.testing.assert_close(qvel[:, :3], expected_qvel[:, :3], atol=_HUMENV_ROOT_LINEAR_ATOL, rtol=0.0)
    torch.testing.assert_close(qvel[:, 6:], expected_qvel[:, 6:], atol=_HUMENV_DOF_VELOCITY_ATOL, rtol=0.0)
    expected_coordinates = MotionGeneralizedCoordinates(
        expected_qpos.to(torch.float32).contiguous(), expected_qvel.to(torch.float32).contiguous()
    )
    actual_frames = target.materialize_coordinates(coordinates, output_index)
    expected_frames = target.materialize_coordinates(expected_coordinates, output_index)
    torch.testing.assert_close(
        actual_frames.joint_velocity,
        expected_frames.joint_velocity,
        atol=_HUMENV_DOF_VELOCITY_ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(qvel[:, 3:6], expected_qvel[:, 3:6], atol=_HUMENV_ROOT_ANGULAR_ATOL, rtol=0.0)
