# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-builder oracles for native SMPL, native G1, and cross projection."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_from_rotation_vector,
    quat_mul,
)

from isaaclab_tasks.core.multi_task.kinematics import (
    KinematicTree,
    KinematicTreeRotationProjection,
    fit_ordered_hinge_coordinates,
    time_gaussian_filter,
    time_gradient,
    time_quaternion_angular_velocity,
)
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    CmuHumEnvSmplClip,
    CmuHumEnvSmplClips,
    LafanG1Clip,
    cmu_humenv_smpl_skeleton,
    lafan_g1_29dof_skeleton,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_OFFSET_M, G1_HEAD_PARENT_BODY_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1LocalBodyPoseFrameBuilder, G1PoseFrameBuilder
from isaaclab_tasks.core.multi_task.motion.robots.smpl.frames import smpl_live_joint_source_names
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder

from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

SMPL_LIVE_JOINT_NAMES = tuple(
    f"{body}_x_{body}_y_{body}_z:{component}" for body in MUJOCO_BODY_NAMES[1:] for component in range(3)
)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _ReferenceKinematics:
    """Small exact-contract FK stand-in for construction-boundary unit tests."""

    def __init__(self, skeleton, path: Path, *, smpl: bool = False) -> None:
        self.body_names = list(skeleton.body_names)
        self.joint_names = ["root", *skeleton.joint_names]
        self.mjcf_path = str(path)
        self.device = "cpu"
        self.model = SimpleNamespace(
            body_count=skeleton.num_bodies,
            joint_coord_count=(7 + skeleton.num_joints),
            joint_dof_count=(6 + skeleton.num_joints),
            body_com=wp.zeros(skeleton.num_bodies, dtype=wp.vec3, device="cpu"),
        )
        self.smpl = smpl

    def eval_fk_batched_torch(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        body_q: torch.Tensor,
        body_qd: torch.Tensor,
    ) -> None:
        """Fill deterministic world poses while preserving root orientation."""
        body_q.zero_()
        body_q[..., :3].copy_(joint_q[:, None, :3])
        body_q[..., 0].add_(torch.arange(body_q.shape[1], dtype=torch.float32)[None] * 0.001)
        body_q[..., 3:].copy_(joint_q[:, None, 3:7])
        body_qd.zero_()

        body_qd[..., :3].copy_(joint_qd[:, None, :3])
        body_qd[..., 3:].copy_(joint_qd[:, None, 3:6])


@pytest.fixture
def reference_path(tmp_path: Path) -> Path:
    path = tmp_path / "reference.xml"
    path.write_text("<mujoco/>", encoding="utf-8")
    return path


def _g1_builder(reference_path: Path, *, reverse_joints: bool = False) -> G1PoseFrameBuilder:
    skeleton = lafan_g1_29dof_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path)
    joint_names = skeleton.joint_names[::-1] if reverse_joints else skeleton.joint_names
    return G1PoseFrameBuilder(
        target_tree=_kinematic_tree(skeleton),
        pose_coordinate_identity_sha256=skeleton.identity_sha256,
        reference_kinematics=reference,
        reference_mjcf_sha256=_file_sha256(reference_path),
        live_joint_names=joint_names,
        live_body_names=skeleton.body_names,
    )


def test_g1_builder_rejects_invalid_pose_coordinate_digest(reference_path: Path) -> None:
    """A pose-coordinate contract must carry one canonical SHA-256 identity."""
    with pytest.raises(ValueError, match="pose_coordinate_identity_sha256"):
        replace(_g1_builder(reference_path), pose_coordinate_identity_sha256="invalid")


def _smpl_builder(reference_path: Path) -> SmplGeneralizedCoordinateFrameBuilder:
    skeleton = cmu_humenv_smpl_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path, smpl=True)
    return SmplGeneralizedCoordinateFrameBuilder(
        source_skeleton=skeleton,
        reference_kinematics=reference,
        reference_mjcf_sha256=_file_sha256(reference_path),
        live_joint_names=SMPL_LIVE_JOINT_NAMES,
        live_body_names=skeleton.body_names,
    )


def _cmu_clip(qpos: np.ndarray, qvel: np.ndarray | None = None) -> CmuHumEnvSmplClip:
    frame_count = qpos.shape[0]
    return CmuHumEnvSmplClip(
        generalized_position=qpos,
        generalized_velocity=np.zeros((frame_count, 75), dtype=np.float32) if qvel is None else qvel,
        source_fps=30.0,
    )


def _kinematic_tree(skeleton) -> KinematicTree:
    return KinematicTree(
        body_names=skeleton.body_names,
        joint_names=skeleton.joint_names,
        parent_indices=skeleton.parent_indices,
        joint_child_body_indices=skeleton.joint_child_body_indices,
        joint_axes=tuple(axis for axis in skeleton.joint_axes if axis is not None),
    )


def test_trajectory_runtime_has_no_scipy_or_materializer_dependency() -> None:
    """Construction uses Torch and exact kinematics without retired package dependencies."""
    motion = Path(__file__).parents[1] / "motion"
    packages = (motion / "data" / "sources", motion / "robots")
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for package in packages:
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])
                    imported_modules.add(node.module)
    assert "scipy" not in imported_roots
    assert not any("materializer" in module for module in imported_modules)


def test_quaternion_angular_velocity_differentiates_each_batch_along_time() -> None:
    step_seconds = 0.1
    angles = torch.tensor(
        (
            (0.0, 0.1, 0.2, 0.3),
            (0.0, 0.2, 0.4, 0.6),
        ),
        dtype=torch.float32,
    )
    rotation_vector = torch.zeros(2, 4, 1, 3)
    rotation_vector[..., 0, 2] = angles
    rotation = quat_from_rotation_vector(rotation_vector)

    angular_velocity = time_quaternion_angular_velocity(rotation, step_seconds)

    expected = torch.zeros(2, 4, 1, 3)
    expected[0, :-1, 0, 2] = 1.0
    expected[1, :-1, 0, 2] = 2.0
    torch.testing.assert_close(angular_velocity, expected, rtol=1.0e-5, atol=1.0e-5)


def test_g1_builder_emits_live_joint_and_reference_frame_axes(reference_path: Path) -> None:
    builder = _g1_builder(reference_path, reverse_joints=True)
    target_tree = builder.target_tree

    assert builder.joint_names == target_tree.joint_names[::-1]
    assert builder.reference_frame_names[:-1] == target_tree.body_names
    assert builder.reference_frame_names[-1] == "head_link"
    assert len(builder.construction_identity_sha256) == 64

    allocated = builder.allocate(7, device="cpu")
    assert allocated.field("root_position").shape == (7, 3)
    assert allocated.joint_position.shape == (7, 29)
    assert allocated.body_position.shape == (7, 31, 3)


def test_g1_builder_reorders_reference_bodies_once_before_head_derivation(reference_path: Path) -> None:
    skeleton = lafan_g1_29dof_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path)
    live_body_names = (skeleton.body_names[0], *skeleton.body_names[:0:-1])
    builder = G1PoseFrameBuilder(
        target_tree=_kinematic_tree(skeleton),
        pose_coordinate_identity_sha256=skeleton.identity_sha256,
        reference_kinematics=reference,
        reference_mjcf_sha256=_file_sha256(reference_path),
        live_joint_names=skeleton.joint_names,
        live_body_names=live_body_names,
    )
    canonical = _g1_builder(reference_path)

    assert builder.construction_identity_sha256 != canonical.construction_identity_sha256
    assert builder.reference_frame_names == (*live_body_names, "head_link")
    frames = builder.build_pose_frames(torch.zeros(5, 30, 3), torch.zeros(5, 3), 30.0)
    expected_reference_indices = torch.tensor(builder._live_body_from_reference, dtype=torch.float32)
    torch.testing.assert_close(frames.body_position[0, :-1, 0], expected_reference_indices * 0.001)
    parent = live_body_names.index(G1_HEAD_PARENT_BODY_NAME)
    torch.testing.assert_close(frames.body_position[:, -1, :2], frames.body_position[:, parent, :2])
    torch.testing.assert_close(
        frames.body_position[:, -1, 2],
        frames.body_position[:, parent, 2] + G1_HEAD_OFFSET_M[2],
    )


def test_g1_builder_restores_bfm_reference_head_derivative_law(reference_path: Path) -> None:
    """Reference head velocity differentiates augmented pose, not the live cross shortcut."""
    builder = _g1_builder(reference_path, reverse_joints=True)
    frame_count = 17
    pose = torch.zeros(frame_count, 30, 3)
    pose[:, 0, 1] = torch.linspace(0.0, 1.2, frame_count)
    translation = torch.zeros(frame_count, 3)
    frames = builder.build_pose_frames(pose, translation, 30.0)

    expected = time_gaussian_filter(time_gradient(frames.body_position[:, -1].unsqueeze(0), 1.0 / 30.0)).squeeze(0)
    torch.testing.assert_close(frames.body_linear_velocity[:, -1], expected)

    parent = builder.live_body_names.index(G1_HEAD_PARENT_BODY_NAME)
    unrotated = frames.body_position.new_tensor(G1_HEAD_OFFSET_M).expand(frame_count, 3)
    live_shortcut = frames.body_linear_velocity[:, parent] + torch.cross(
        frames.body_angular_velocity[:, parent],
        unrotated,
        dim=-1,
    )
    assert not torch.allclose(frames.body_linear_velocity[4:-4, -1], live_shortcut[4:-4])


def test_g1_builder_maps_source_joints_once_and_exposes_body_row_zero_root(reference_path: Path) -> None:
    builder = _g1_builder(reference_path, reverse_joints=True)
    frame_count = 5
    pose = torch.zeros(frame_count, 30, 3)
    pose[:, 1:, 0] = torch.arange(29, dtype=torch.float32)
    translation = torch.arange(frame_count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
    frames = builder.build_pose_frames(pose, translation, 30.0)

    assert frames.root_position is None
    assert frames.root_storage == "body_row_zero"
    stored_values = sum(frames.field(name).numel() for name in frames.stored_fields)
    assert stored_values // frames.frame_count == 461
    torch.testing.assert_close(frames.joint_position[0], torch.arange(29, dtype=torch.float32).flip(0))
    torch.testing.assert_close(frames.field("root_position"), frames.body_position[:, 0])
    torch.testing.assert_close(frames.field("root_rotation"), frames.body_rotation[:, 0])
    torch.testing.assert_close(frames.field("root_linear_velocity"), frames.body_linear_velocity[:, 0])
    torch.testing.assert_close(frames.field("root_angular_velocity"), frames.body_angular_velocity[:, 0])


def test_lafan_typed_clip_rejects_source_dtype_instead_of_repairing_it() -> None:
    with pytest.raises(ValueError, match="float32"):
        LafanG1Clip(
            root_translation=np.zeros((5, 3), dtype=np.float64),
            pose_axis_angle=np.zeros((5, 30, 3), dtype=np.float32),
            source_fps=30.0,
        )


def test_smpl_builder_maps_live_axes_and_materializes_physical_body_fields(reference_path: Path) -> None:
    builder = _smpl_builder(reference_path)
    skeleton = builder.source_skeleton
    frame_count = 4
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qvel = np.zeros((frame_count, 75), dtype=np.float32)
    qpos[:, 2] = 1.0
    qpos[:, 3] = np.cos(0.25 * np.pi)
    qpos[:, 6] = np.sin(0.25 * np.pi)
    qpos[:, 7:] = np.arange(69, dtype=np.float32)
    qvel[:, 3] = 2.0
    clip = _cmu_clip(qpos, qvel)
    frames = builder.build_frames(clip, device="cpu")

    live_source_names = smpl_live_joint_source_names(SMPL_LIVE_JOINT_NAMES)
    live_indices = torch.tensor([skeleton.joint_names.index(name) for name in live_source_names])
    torch.testing.assert_close(frames.joint_position[0], torch.arange(69, dtype=torch.float32)[live_indices])
    expected_rotation = convert_quat(torch.from_numpy(qpos[:, 3:7]), to="xyzw")
    torch.testing.assert_close(frames.field("root_rotation"), expected_rotation)
    torch.testing.assert_close(
        frames.field("root_angular_velocity"),
        quat_apply(expected_rotation, torch.from_numpy(qvel[:, 3:6])),
    )
    assert builder.joint_names == SMPL_LIVE_JOINT_NAMES
    assert builder.reference_frame_names == skeleton.body_names
    assert frames.body_position.shape == (frame_count, skeleton.num_bodies, 3)
    assert frames.root_storage == "body_row_zero"
    assert not hasattr(frames, "observation")


def test_smpl_builder_converts_nonzero_com_velocities_between_root_and_link_frames(reference_path: Path) -> None:
    """Root-COM FK velocities must return to link-origin velocities with the correct world-frame signs."""
    builder = _smpl_builder(reference_path)
    body_count = builder.source_skeleton.num_bodies
    body_index = torch.arange(body_count, dtype=torch.float32)
    body_com = torch.stack(
        (
            0.13 + 0.011 * body_index,
            -0.09 + 0.007 * body_index,
            0.04 - 0.005 * body_index,
        ),
        dim=-1,
    )
    builder._body_com.copy_(body_com)

    frame_count = 3
    root_rotation = torch.tensor((0.23, -0.31, 0.17, 0.88), dtype=torch.float32)
    root_rotation /= torch.linalg.vector_norm(root_rotation)
    root_rotation = root_rotation.expand(frame_count, 4).clone()
    root_linear_velocity = torch.tensor(((1.2, -0.4, 0.7), (-0.3, 0.9, 1.1), (0.6, 0.2, -0.8)), dtype=torch.float32)
    root_angular_velocity_local = torch.tensor(
        ((0.4, -0.7, 1.1), (-1.2, 0.3, 0.5), (0.8, 1.0, -0.6)), dtype=torch.float32
    )
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qpos[:, 3:7] = convert_quat(root_rotation, to="wxyz").numpy()
    qvel = np.zeros((frame_count, 75), dtype=np.float32)
    qvel[:, :3] = root_linear_velocity.numpy()
    qvel[:, 3:6] = root_angular_velocity_local.numpy()

    frames = builder.build_frames(_cmu_clip(qpos, qvel), device="cpu")

    root_angular_velocity = quat_apply(root_rotation, root_angular_velocity_local)
    root_com_world = quat_apply(root_rotation, body_com[0].expand(frame_count, 3))
    body_rotation = root_rotation[:, None, :].expand(frame_count, body_count, 4)
    body_com_world = quat_apply(body_rotation, body_com[None].expand(frame_count, body_count, 3))
    expected = root_linear_velocity[:, None, :] + torch.cross(
        root_angular_velocity[:, None, :], root_com_world[:, None, :] - body_com_world, dim=-1
    )
    torch.testing.assert_close(frames.body_linear_velocity, expected)
    torch.testing.assert_close(frames.body_angular_velocity, root_angular_velocity[:, None].expand_as(expected))


def test_humenv_importer_keeps_unconsumed_native_fields_out_of_smpl_table(
    tmp_path: Path,
    reference_path: Path,
) -> None:
    """The full native schema is inspected while decoded clips retain only physical inputs."""
    h5py = pytest.importorskip("h5py")
    builder = _smpl_builder(reference_path)
    frame_count = 4
    observation = np.arange(frame_count * 358, dtype=np.float64).reshape(frame_count, 358)
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qpos[:, 3] = 1.0
    terminated = np.zeros((frame_count, 1), dtype=np.bool_)
    terminated[-1] = True
    native_fields = {
        "motion_id": np.arange(frame_count, dtype=np.int64)[:, None],
        "observation": observation,
        "qpos": qpos,
        "qvel": np.zeros((frame_count, 75), dtype=np.float32),
        "terminated": terminated,
        "truncated": np.zeros((frame_count, 1), dtype=np.bool_),
    }
    path = tmp_path / "native_humenv.hdf5"
    with h5py.File(path, "w") as stream:
        episode = stream.create_group("ep_0")
        for name, value in native_fields.items():
            episode.create_dataset(name, data=value)

    source = CmuHumEnvSmplClips(
        (path,),
        file_sha256s=(_file_sha256(path),),
        source_fps=30.0,
    )
    _, decoded = next(source.clips())
    frames = builder.build_frames(decoded, device="cpu")

    assert decoded.generalized_position.dtype == np.float32
    assert decoded.generalized_velocity.dtype == np.float32
    assert frames.body_position.dtype == torch.float32
    assert frames.body_position.shape == (frame_count, builder.source_skeleton.num_bodies, 3)
    assert not hasattr(frames, "observation")


def test_ordered_hinge_fit_is_exact_for_three_axes_and_optimal_when_underactuated() -> None:
    axes = torch.eye(3, dtype=torch.float32)
    coordinates = torch.tensor(((0.3, -0.4, 0.2), (-0.8, 0.5, 0.9)), dtype=torch.float32)
    rotation = torch.zeros(2, 4)
    rotation[:, 3] = 1.0
    for index in range(3):
        rotation = quat_mul(rotation, quat_from_rotation_vector(coordinates[:, index, None] * axes[index]))
    fitted, residual = fit_ordered_hinge_coordinates(rotation, axes)

    reconstructed = torch.zeros_like(rotation)
    reconstructed[:, 3] = 1.0
    for index in range(3):
        reconstructed = quat_mul(
            reconstructed,
            quat_from_rotation_vector(fitted[:, index, None] * axes[index]),
        )
    torch.testing.assert_close(reconstructed, rotation, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-6, rtol=0.0)

    one_axis, one_residual = fit_ordered_hinge_coordinates(rotation, axes[:1])
    assert one_axis.shape == (2, 1)
    assert torch.all(one_residual >= 0.0)


def test_humenv_xyz_coordinates_reconstruct_before_g1_projection() -> None:
    skeleton = cmu_humenv_smpl_skeleton()
    qpos = torch.zeros(3, 76)
    qpos[:, 3] = 1.0
    qpos[:, 7:10] = torch.tensor((0.2, -0.3, 0.4))
    local_wxyz = _cmu_clip(qpos.numpy()).local_body_rotation_wxyz(skeleton, device="cpu")

    axes = torch.eye(3)
    expected = torch.zeros(3, 4)
    expected[:, 3] = 1.0
    for index in range(3):
        expected = quat_mul(
            expected,
            quat_from_rotation_vector(qpos[:, 7 + index, None] * axes[index]),
        )
    torch.testing.assert_close(convert_quat(local_wxyz[:, 1], to="xyzw"), expected)


def test_cross_builder_preserves_target_axis_identity(reference_path: Path) -> None:
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path, reverse_joints=True)
    source_by_name = {name: index for index, name in enumerate(source.body_names)}
    source_names = (
        "L_Hip",
        "L_Hip",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "L_Ankle",
        "R_Hip",
        "R_Hip",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "R_Ankle",
        "Torso",
        "Torso",
        "Torso",
        "L_Shoulder",
        "L_Shoulder",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "L_Wrist",
        "L_Wrist",
        "R_Shoulder",
        "R_Shoulder",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Wrist",
        "R_Wrist",
    )
    projection = KinematicTreeRotationProjection(
        source_body_count=source.num_bodies,
        target_tree=target.target_tree,
        target_joint_source_body_indices=tuple(source_by_name[name] for name in source_names),
        device=target.reference_kinematics.device,
    )
    builder = G1LocalBodyPoseFrameBuilder(
        source_skeleton=source,
        target_builder=target,
        projection=projection,
        target_tree_identity_sha256=target.pose_coordinate_identity_sha256,
    )

    assert builder.joint_names == target.joint_names
    assert builder.reference_frame_names == target.reference_frame_names
    assert len(builder.construction_identity_sha256) == 64


def test_cross_builder_builds_finite_continuous_target_g1_frames(reference_path: Path) -> None:
    """The full HumEnv decode and projection path must emit known target-G1 values."""
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path)
    source_by_name = {name: index for index, name in enumerate(source.body_names)}
    source_names = (
        "L_Hip",
        "L_Hip",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "L_Ankle",
        "R_Hip",
        "R_Hip",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "R_Ankle",
        "Torso",
        "Torso",
        "Torso",
        "L_Shoulder",
        "L_Shoulder",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "L_Wrist",
        "L_Wrist",
        "R_Shoulder",
        "R_Shoulder",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Wrist",
        "R_Wrist",
    )
    projection = KinematicTreeRotationProjection(
        source_body_count=source.num_bodies,
        target_tree=target.target_tree,
        target_joint_source_body_indices=tuple(source_by_name[name] for name in source_names),
        device=target.reference_kinematics.device,
    )
    builder = G1LocalBodyPoseFrameBuilder(
        source_skeleton=source,
        target_builder=target,
        projection=projection,
        target_tree_identity_sha256=target.pose_coordinate_identity_sha256,
    )

    source_angles = np.asarray((3.0, 3.1, -3.1, -3.0, -2.9), dtype=np.float32)
    expected_angles = torch.tensor((3.0, 3.1, 2.0 * np.pi - 3.1, 2.0 * np.pi - 3.0, 2.0 * np.pi - 2.9))
    frame_count = source_angles.shape[0]
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qpos[:, :3] = np.stack(
        (
            np.linspace(0.0, 0.4, frame_count, dtype=np.float32),
            np.zeros(frame_count, dtype=np.float32),
            np.ones(frame_count, dtype=np.float32),
        ),
        axis=-1,
    )
    qpos[:, 3] = 1.0
    qpos[:, 8] = source_angles
    clip = _cmu_clip(qpos)

    frames = builder.build_frames(clip, device="cpu")

    torch.testing.assert_close(frames.joint_position[:, 0], expected_angles, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(frames.joint_position[:, 1:], torch.zeros(frame_count, 28), atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(frames.field("root_position"), torch.from_numpy(qpos[:, :3]))
    assert torch.all(torch.isfinite(frames.joint_velocity))
    assert torch.max(torch.abs(torch.diff(frames.joint_position[:, 0]))) < 0.2
    for name in frames.stored_fields:
        assert torch.all(torch.isfinite(frames.field(name)))
    assert frames.body_rotation is not None
    torch.testing.assert_close(
        torch.linalg.vector_norm(frames.body_rotation, dim=-1),
        torch.ones(frame_count, len(builder.reference_frame_names)),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
