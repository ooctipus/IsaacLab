# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-builder oracles for native SMPL, native G1, and cross projection."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_from_rotation_vector,
    quat_mul,
)

from isaaclab_tasks.core.multi_task.motion.config.robots.smpl import (
    _SMPL_SIMULATOR_JOINT_NAMES as SMPL_LIVE_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.source_skeletons import (
    g1_lafan_source_skeleton,
    smpl_humenv_source_skeleton,
)
from isaaclab_tasks.core.multi_task.motion.data.importers.humenv_hdf5 import HumEnvHdf5Clips
from isaaclab_tasks.core.multi_task.motion.frames import G1_HEAD_OFFSET_M, G1_HEAD_PARENT_BODY_NAME
from isaaclab_tasks.core.multi_task.motion.trajectory._time import gaussian_filter_time, gradient_time
from isaaclab_tasks.core.multi_task.motion.trajectory.g1 import G1LafanFrameBuilder
from isaaclab_tasks.core.multi_task.motion.trajectory.g1_smpl import (
    G1HumanFrameProjection,
    G1SmplHumEnvFrameBuilder,
    fit_ordered_hinge_coordinates,
    smpl_humenv_local_rotation_wxyz,
)
from isaaclab_tasks.core.multi_task.motion.trajectory.smpl import SmplHumEnvFrameBuilder, smpl_live_joint_source_names


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
        del joint_qd
        body_q.zero_()
        body_q[..., :3].copy_(joint_q[:, None, :3])
        body_q[..., 0].add_(torch.arange(body_q.shape[1], dtype=torch.float32)[None] * 0.001)
        body_q[..., 3:].copy_(joint_q[:, None, 3:7])
        body_qd.zero_()


@pytest.fixture
def reference_path(tmp_path: Path) -> Path:
    path = tmp_path / "reference.xml"
    path.write_text("<mujoco/>", encoding="utf-8")
    return path


def _g1_builder(reference_path: Path, *, reverse_joints: bool = False) -> G1LafanFrameBuilder:
    skeleton = g1_lafan_source_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path)
    joint_names = skeleton.joint_names[::-1] if reverse_joints else skeleton.joint_names
    return G1LafanFrameBuilder(
        source_skeleton=skeleton,
        reference_kinematics=reference,
        live_joint_names=joint_names,
        live_body_names=skeleton.body_names,
    )


def _smpl_builder(reference_path: Path) -> SmplHumEnvFrameBuilder:
    skeleton = smpl_humenv_source_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path, smpl=True)
    return SmplHumEnvFrameBuilder(
        source_skeleton=skeleton,
        reference_kinematics=reference,
        live_joint_names=SMPL_LIVE_JOINT_NAMES,
        live_body_names=skeleton.body_names,
    )


def test_trajectory_runtime_has_no_scipy_or_materializer_dependency() -> None:
    """Construction uses Torch and exact kinematics without retired package dependencies."""
    package = Path(__file__).parents[1] / "motion" / "trajectory"
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
                imported_modules.add(node.module)
    assert "scipy" not in imported_roots
    assert not any("materializer" in module for module in imported_modules)


def test_g1_builder_emits_live_joint_and_reference_frame_axes(reference_path: Path) -> None:
    builder = _g1_builder(reference_path, reverse_joints=True)
    skeleton = builder.source_skeleton

    assert builder.joint_names == skeleton.joint_names[::-1]
    assert builder.reference_frame_names[:-1] == skeleton.body_names
    assert builder.reference_frame_names[-1] == "head_link"
    assert len(builder.construction_identity_sha256) == 64

    allocated = builder.allocate(7, device="cpu")
    assert allocated.field("root_position").shape == (7, 3)
    assert allocated.joint_position.shape == (7, 29)
    assert allocated.body_position.shape == (7, 31, 3)


def test_g1_builder_reorders_reference_bodies_once_before_head_derivation(reference_path: Path) -> None:
    skeleton = g1_lafan_source_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path)
    live_body_names = (skeleton.body_names[0], *skeleton.body_names[:0:-1])
    builder = G1LafanFrameBuilder(
        source_skeleton=skeleton,
        reference_kinematics=reference,
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


def test_g1_builder_restores_released_reference_head_derivative_law(reference_path: Path) -> None:
    """Reference head velocity differentiates augmented pose, not the live cross shortcut."""
    builder = _g1_builder(reference_path, reverse_joints=True)
    frame_count = 17
    pose = torch.zeros(frame_count, 30, 3)
    pose[:, 0, 1] = torch.linspace(0.0, 1.2, frame_count)
    translation = torch.zeros(frame_count, 3)
    frames = builder.build_pose_frames(pose, translation, 30.0)

    expected = gaussian_filter_time(gradient_time(frames.body_position[:, -1].unsqueeze(0), 1.0 / 30.0)).squeeze(0)
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
    assert frames.memory_bytes // frames.frame_count // torch.tensor([], dtype=torch.float32).element_size() == 461
    torch.testing.assert_close(frames.joint_position[0], torch.arange(29, dtype=torch.float32).flip(0))
    torch.testing.assert_close(frames.field("root_position"), frames.body_position[:, 0])
    torch.testing.assert_close(frames.field("root_rotation"), frames.body_rotation[:, 0])
    torch.testing.assert_close(frames.field("root_linear_velocity"), frames.body_linear_velocity[:, 0])
    torch.testing.assert_close(frames.field("root_angular_velocity"), frames.body_angular_velocity[:, 0])


def test_g1_builder_rejects_source_dtype_instead_of_repairing_it(reference_path: Path) -> None:
    builder = _g1_builder(reference_path)
    fields = {
        "root_trans_offset": np.zeros((5, 3), dtype=np.float64),
        "pose_aa": np.zeros((5, 30, 3), dtype=np.float32),
        "fps": 30,
    }
    with pytest.raises(ValueError, match="float32"):
        builder.build_frames(fields, device="cpu")


def test_smpl_builder_maps_nonselfinverse_live_order_and_world_angular_velocity(reference_path: Path) -> None:
    builder = _smpl_builder(reference_path)
    skeleton = builder.source_skeleton
    frame_count = 4
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qvel = np.zeros((frame_count, 75), dtype=np.float32)
    observation = np.arange(frame_count * 358, dtype=np.float64).reshape(frame_count, 358)
    qpos[:, 2] = 1.0
    qpos[:, 3] = np.cos(0.25 * np.pi)
    qpos[:, 6] = np.sin(0.25 * np.pi)
    qpos[:, 7:] = np.arange(69, dtype=np.float32)
    qvel[:, 3] = 2.0
    fields = {
        "motion_id": np.asarray("clip"),
        "observation": observation,
        "qpos": qpos,
        "qvel": qvel,
        "terminated": np.zeros(frame_count, dtype=np.bool_),
        "truncated": np.zeros(frame_count, dtype=np.bool_),
    }
    frames = builder.build_frames(fields, device="cpu")

    live_source_names = smpl_live_joint_source_names(SMPL_LIVE_JOINT_NAMES)
    live_indices = torch.tensor([skeleton.joint_names.index(name) for name in live_source_names])
    torch.testing.assert_close(frames.joint_position[0], torch.arange(69, dtype=torch.float32)[live_indices])
    expected_rotation = convert_quat(torch.from_numpy(qpos[:, 3:7]), to="xyzw")
    torch.testing.assert_close(frames.root_rotation, expected_rotation)
    torch.testing.assert_close(
        frames.root_angular_velocity,
        quat_apply(expected_rotation, torch.from_numpy(qvel[:, 3:6])),
    )
    assert builder.joint_names == SMPL_LIVE_JOINT_NAMES
    assert frames.observation.dtype == torch.float32
    torch.testing.assert_close(frames.observation, torch.from_numpy(observation).float())
    assert builder.reference_frame_names == ()


def test_smpl_builder_rejects_non_native_float32_observation(reference_path: Path) -> None:
    builder = _smpl_builder(reference_path)
    fields = {
        "motion_id": np.asarray("clip"),
        "observation": np.zeros((3, 358), dtype=np.float32),
        "qpos": np.zeros((3, 76), dtype=np.float32),
        "qvel": np.zeros((3, 75), dtype=np.float32),
        "terminated": np.zeros(3, dtype=np.bool_),
        "truncated": np.zeros(3, dtype=np.bool_),
    }
    with pytest.raises(ValueError, match="observation must be a float64"):
        builder.build_frames(fields, device="cpu")


def test_humenv_importer_native_observation_builds_float32_smpl_table(
    tmp_path: Path,
    reference_path: Path,
) -> None:
    """The importer-to-builder seam preserves native dtype and casts once into table storage."""
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

    source = HumEnvHdf5Clips(
        (path,),
        source_fps=30.0,
        skeleton_sha256=builder.source_skeleton.identity_sha256,
        split="train",
        license="test-only",
    )
    _, decoded = next(source.clips())
    frames = builder.build_frames(decoded, device="cpu")

    assert decoded["observation"].dtype == np.float64
    assert frames.observation.dtype == torch.float32
    torch.testing.assert_close(frames.observation, torch.from_numpy(observation).float())


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
    skeleton = smpl_humenv_source_skeleton()
    qpos = torch.zeros(3, 76)
    qpos[:, 3] = 1.0
    qpos[:, 7:10] = torch.tensor((0.2, -0.3, 0.4))
    local_wxyz = smpl_humenv_local_rotation_wxyz(qpos, skeleton)

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
    source = smpl_humenv_source_skeleton()
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
    projection = G1HumanFrameProjection(
        source_skeleton=source,
        target_builder=target,
        target_joint_source_body_indices=tuple(source_by_name[name] for name in source_names),
    )
    builder = G1SmplHumEnvFrameBuilder(source, projection)

    assert builder.joint_names == target.joint_names
    assert builder.reference_frame_names == target.reference_frame_names
    assert len(builder.construction_identity_sha256) == 64


def test_cross_builder_builds_finite_continuous_target_g1_frames(reference_path: Path) -> None:
    """The full HumEnv decode and projection path must emit known target-G1 values."""
    source = smpl_humenv_source_skeleton()
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
    projection = G1HumanFrameProjection(
        source_skeleton=source,
        target_builder=target,
        target_joint_source_body_indices=tuple(source_by_name[name] for name in source_names),
    )
    builder = G1SmplHumEnvFrameBuilder(source, projection)

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
    fields = {
        "motion_id": np.zeros((frame_count, 1), dtype=np.int64),
        "observation": np.zeros((frame_count, 358), dtype=np.float64),
        "qpos": qpos,
        "qvel": np.zeros((frame_count, 75), dtype=np.float32),
        "terminated": np.zeros((frame_count, 1), dtype=np.bool_),
        "truncated": np.asarray(((False,), (False,), (False,), (False,), (True,)), dtype=np.bool_),
    }

    frames = builder.build_frames(fields, device="cpu")

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
