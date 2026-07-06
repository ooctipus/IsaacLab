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
    fit_ordered_hinge_coordinates,
    kinematic_tree_forward,
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
from isaaclab_tasks.core.multi_task.motion.retarget import MotionSemanticProjection
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_OFFSET_M, G1_HEAD_PARENT_BODY_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import (
    _G1_RETARGET_MATH_VERSION,
    _G1_RETARGET_TARGETS,
    _G1_ROOT_BASIS_ROLES,
    _G1_SUPPORT_ROLES,
    _g1_coordinates_match,
    _G1TargetFrameBuilder,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import smpl_live_joint_mujoco_names
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
    _SMPL_RETARGET_MATH_VERSION,
    _SMPL_RETARGET_TARGETS,
    _SMPL_ROOT_BASIS_ROLES,
    _SMPL_SUPPORT_ROLES,
    _smpl_coordinates_match,
    _SmplTargetFrameBuilder,
)

from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

SMPL_LIVE_JOINT_NAMES = tuple(
    f"{body}_x_{body}_y_{body}_z:{component}" for body in MUJOCO_BODY_NAMES[1:] for component in range(3)
)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _ReferenceKinematics:
    """Small exact-contract FK stand-in for construction-boundary unit tests."""

    def __init__(
        self,
        skeleton,
        path: Path,
        *,
        smpl: bool = False,
        body_com: torch.Tensor | None = None,
        smpl_joint_body_indices: tuple[int, ...] | None = None,
        device: str = "cpu",
    ) -> None:
        self.body_names = list(skeleton.body_names)
        self.mjcf_path = str(path)
        self.device = device
        if smpl:
            joint_body_indices = smpl_joint_body_indices or tuple(range(1, skeleton.num_bodies))
            self.joint_names = ["root", *(f"{skeleton.body_names[index]}_xyz" for index in joint_body_indices)]
            self.joint_q_names = [
                *(f"root_{index}" for index in range(7)),
                *(f"{skeleton.body_names[index]}_{axis}" for index in joint_body_indices for axis in "xyz"),
            ]
            joint_q_start = [0, *range(7, 7 + 3 * (len(joint_body_indices) + 1), 3)]
            joint_qd_start = [0, *range(6, 6 + 3 * (len(joint_body_indices) + 1), 3)]
            joint_child = [0, *joint_body_indices]
            joint_parent = [-1, *(skeleton.parent_indices[index] for index in joint_body_indices)]
            joint_dof_dim = [(0, 6), *((0, 3),) * len(joint_body_indices)]
            joint_axis = torch.cat(
                (
                    torch.zeros(6, 3, dtype=torch.float32),
                    torch.eye(3, dtype=torch.float32).repeat(len(joint_body_indices), 1),
                )
            )
        else:
            self.joint_names = ["root", *skeleton.joint_names]
            self.joint_q_names = [*(f"root_{index}" for index in range(7)), *skeleton.joint_names]
            joint_q_start = [0, *range(7, 7 + skeleton.num_joints + 1)]
            joint_qd_start = [0, *range(6, 6 + skeleton.num_joints + 1)]
            joint_child = [0, *skeleton.joint_child_body_indices]
            joint_parent = [-1, *(skeleton.parent_indices[index] for index in skeleton.joint_child_body_indices)]
            joint_dof_dim = [(0, 6), *((0, 1),) * skeleton.num_joints]
            joint_axis = torch.cat(
                (
                    torch.zeros(6, 3, dtype=torch.float32),
                    torch.tensor(skeleton.joint_axes, dtype=torch.float32),
                )
            )
        rest_position, rest_rotation = kinematic_tree_forward(
            torch.tensor(skeleton.rest_translation_m, dtype=torch.float32),
            convert_quat(torch.tensor(skeleton.rest_rotation_wxyz, dtype=torch.float32), to="xyzw"),
            skeleton.parent_indices,
        )
        self.default_body_q = torch.cat((rest_position, rest_rotation), dim=-1).numpy()
        self.model = SimpleNamespace(
            body_count=skeleton.num_bodies,
            joint_count=len(self.joint_names),
            joint_coord_count=(7 + skeleton.num_joints),
            joint_dof_count=(6 + skeleton.num_joints),
            joint_parent=wp.array(joint_parent, dtype=wp.int32, device=device),
            joint_child=wp.array(joint_child, dtype=wp.int32, device=device),
            joint_q_start=wp.array(joint_q_start, dtype=wp.int32, device=device),
            joint_qd_start=wp.array(joint_qd_start, dtype=wp.int32, device=device),
            joint_dof_dim=wp.array(joint_dof_dim, dtype=wp.vec2i, device=device),
            joint_axis=wp.from_torch(joint_axis.to(device), dtype=wp.vec3),
            joint_limit_lower=wp.full(6 + skeleton.num_joints, -float("inf"), dtype=wp.float32, device=device),
            joint_limit_upper=wp.full(6 + skeleton.num_joints, float("inf"), dtype=wp.float32, device=device),
            body_com=(
                wp.zeros(skeleton.num_bodies, dtype=wp.vec3, device=device)
                if body_com is None
                else wp.from_torch(body_com.to(device), dtype=wp.vec3)
            ),
        )
        self.topology = SimpleNamespace(
            body_count=skeleton.num_bodies,
            joint_count=len(self.joint_names),
            coordinate_count=7 + skeleton.num_joints,
            joint_parent=np.asarray(joint_parent, dtype=np.int32),
            joint_child=np.asarray(joint_child, dtype=np.int32),
            joint_q_start=np.asarray(joint_q_start, dtype=np.int32),
            joint_qd_start=np.asarray(joint_qd_start, dtype=np.int32),
            joint_dof_dim=np.asarray(joint_dof_dim, dtype=np.int32),
            joint_axis=joint_axis.numpy().copy(),
            joint_limit_lower=np.full(6 + skeleton.num_joints, -np.inf, dtype=np.float32),
            joint_limit_upper=np.full(6 + skeleton.num_joints, np.inf, dtype=np.float32),
            body_parent=np.asarray(skeleton.parent_indices, dtype=np.int32),
            body_com=(
                np.zeros((skeleton.num_bodies, 3), dtype=np.float32)
                if body_com is None
                else body_com.detach().cpu().numpy().copy()
            ),
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
        body_q[..., 0].add_(torch.arange(body_q.shape[1], dtype=torch.float32, device=body_q.device)[None] * 0.001)
        body_q[..., 3:].copy_(joint_q[:, None, 3:7])
        body_qd.zero_()

        body_qd[..., :3].copy_(joint_qd[:, None, :3])
        body_qd[..., 3:].copy_(joint_qd[:, None, 3:6])


@pytest.fixture
def reference_path(tmp_path: Path) -> Path:
    path = tmp_path / "reference.xml"
    path.write_text("<mujoco/>", encoding="utf-8")
    return path


def _g1_builder(reference_path: Path, *, reverse_joints: bool = False, device: str = "cpu") -> _G1TargetFrameBuilder:
    skeleton = lafan_g1_29dof_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path, device=device)
    joint_names = skeleton.joint_names[::-1] if reverse_joints else skeleton.joint_names
    return _G1TargetFrameBuilder(
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


def test_lafan_clip_decodes_declared_g1_hinges_as_local_rotations() -> None:
    """The source boundary exposes the G1-retargeted rows as local rotations without target policy."""
    skeleton = lafan_g1_29dof_skeleton()
    pose = np.zeros((3, 30, 3), dtype=np.float32)
    pose[:, 0, 2] = np.array((0.0, 0.1, 0.2), dtype=np.float32)
    pose[:, 1] = np.array((0.0, 0.3, 0.0), dtype=np.float32)
    clip = LafanG1Clip(
        root_translation=np.zeros((3, 3), dtype=np.float32),
        pose_axis_angle=pose,
        source_fps=30.0,
    )

    _, local_xyzw = clip.semantic_local_pose(skeleton, device="cpu")
    expected = quat_from_rotation_vector(torch.from_numpy(pose))

    torch.testing.assert_close(local_xyzw, expected)


def _smpl_builder(
    reference_path: Path, *, body_com: torch.Tensor | None = None, device: str = "cpu"
) -> _SmplTargetFrameBuilder:
    skeleton = cmu_humenv_smpl_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path, smpl=True, body_com=body_com, device=device)
    return _SmplTargetFrameBuilder(
        reference_kinematics=reference,
        reference_mjcf_sha256=_file_sha256(reference_path),
        live_joint_names=SMPL_LIVE_JOINT_NAMES,
        live_body_names=skeleton.body_names,
    )


def _g1_pose_frames(
    builder: _G1TargetFrameBuilder,
    pose_axis_angle: torch.Tensor,
    root_translation: torch.Tensor,
    source_fps: float,
):
    joint_q = torch.empty(pose_axis_angle.shape[0], 36)
    joint_q[:, :3].copy_(root_translation)
    joint_q[:, 3:7].copy_(quat_from_rotation_vector(pose_axis_angle[:, 0]))
    joint_q[:, 7:].copy_(pose_axis_angle.sum(dim=-1)[:, 1:])
    return builder.build_generalized_frames(joint_q, source_fps)


def _smpl_exact_frames(builder: _SmplTargetFrameBuilder, clip: CmuHumEnvSmplClip):
    joint_q, joint_qd = clip.free_root_coordinates(cmu_humenv_smpl_skeleton(), device="cpu")
    assert joint_qd is not None
    generalized_position = torch.cat((joint_q[:, :3], convert_quat(joint_q[:, 3:7], to="wxyz"), joint_q[:, 7:]), dim=-1)
    return builder.build_generalized_frames(generalized_position, joint_qd)


def test_smpl_builder_derives_scalar_coordinates_from_grouped_newton_joints(reference_path: Path) -> None:
    """Canonical SMPL coordinates come from grouped Newton child, range, and axis metadata."""
    builder = _smpl_builder(reference_path)

    assert builder.reference_coordinate_names == cmu_humenv_smpl_skeleton().joint_names


def test_smpl_builder_rejects_an_unrelated_source_schema(reference_path: Path) -> None:
    """The native SMPL builder owns its exact source-coordinate contract."""
    assert not _smpl_coordinates_match(lafan_g1_29dof_skeleton(), _smpl_builder(reference_path))


def test_exact_route_requires_the_complete_robot_coordinate_profile(reference_path: Path) -> None:
    """Rest geometry and topology changes must route through semantic retargeting."""
    g1_source = lafan_g1_29dof_skeleton()
    g1_target = _g1_builder(reference_path).target_tree
    changed_g1 = replace(
        g1_source,
        rest_translation_m=((9.0, 9.0, 9.0), *g1_source.rest_translation_m[1:]),
    )
    assert _g1_coordinates_match(g1_source, g1_target)
    assert not _g1_coordinates_match(changed_g1, g1_target)

    smpl_source = cmu_humenv_smpl_skeleton()
    smpl_target = _smpl_builder(reference_path)
    changed_smpl = replace(
        smpl_source,
        parent_indices=(-1, 0, 1, 1, *smpl_source.parent_indices[4:]),
    )
    assert _smpl_coordinates_match(smpl_source, smpl_target)
    assert not _smpl_coordinates_match(changed_smpl, smpl_target)


def _cmu_clip(qpos: np.ndarray, qvel: np.ndarray | None = None) -> CmuHumEnvSmplClip:
    frame_count = qpos.shape[0]
    return CmuHumEnvSmplClip(
        generalized_position=qpos,
        generalized_velocity=np.zeros((frame_count, 75), dtype=np.float32) if qvel is None else qvel,
        source_fps=30.0,
    )


def _kinematic_tree(skeleton) -> KinematicTree:
    coordinate_count = skeleton.num_joints
    return KinematicTree(
        body_names=skeleton.body_names,
        parent_indices=skeleton.parent_indices,
        joint_names=skeleton.joint_names,
        joint_child_body_indices=skeleton.joint_child_body_indices,
        joint_coordinate_ranges=tuple((index, index + 1) for index in range(coordinate_count)),
        coordinate_names=skeleton.joint_names,
        coordinate_axes=tuple(axis for axis in skeleton.joint_axes if axis is not None),
        coordinate_q_indices=tuple(range(7, 7 + coordinate_count)),
        coordinate_qd_indices=tuple(range(6, 6 + coordinate_count)),
        coordinate_lower_limits_rad=(-float("inf"),) * coordinate_count,
        coordinate_upper_limits_rad=(float("inf"),) * coordinate_count,
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
    builder = _G1TargetFrameBuilder(
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
    frames = _g1_pose_frames(builder, torch.zeros(5, 30, 3), torch.zeros(5, 3), 30.0)
    expected_reference_indices = torch.tensor(
        [reference.body_names.index(name) for name in live_body_names], dtype=torch.float32
    )
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
    frames = _g1_pose_frames(builder, pose, translation, 30.0)

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
    frames = _g1_pose_frames(builder, pose, translation, 30.0)

    assert frames.root_position is None
    assert frames.root_storage == "body_row_zero"
    stored_values = sum(frames.field(name).numel() for name in frames.stored_fields)
    assert stored_values // frames.frame_count == 461
    torch.testing.assert_close(frames.joint_position[0], torch.arange(29, dtype=torch.float32).flip(0))
    torch.testing.assert_close(frames.field("root_position"), frames.body_position[:, 0])
    torch.testing.assert_close(frames.field("root_rotation"), frames.body_rotation[:, 0])
    torch.testing.assert_close(frames.field("root_linear_velocity"), frames.body_linear_velocity[:, 0])
    torch.testing.assert_close(frames.field("root_angular_velocity"), frames.body_angular_velocity[:, 0])


def test_exact_coordinate_materialization_preserves_native_joint_and_root_state(reference_path: Path) -> None:
    """Native G1 and SMPL source coordinates reach target storage without semantic projection."""
    g1_skeleton = lafan_g1_29dof_skeleton()
    g1_coordinates = torch.linspace(-0.4, 0.4, 4 * g1_skeleton.num_joints).view(4, -1)
    g1_pose = torch.zeros(4, g1_skeleton.num_bodies, 3)
    g1_pose[:, 1:] = g1_coordinates[..., None] * torch.tensor(g1_skeleton.joint_axes)[None]
    g1_root = torch.arange(12, dtype=torch.float32).view(4, 3) * 0.01
    g1_clip = LafanG1Clip(g1_root.numpy(), g1_pose.numpy(), 30.0)
    g1_joint_q, _ = g1_clip.free_root_coordinates(g1_skeleton, device="cpu")
    g1_builder = _g1_builder(reference_path)
    g1_frames = g1_builder.build_generalized_frames(g1_joint_q, g1_clip.source_fps)

    torch.testing.assert_close(g1_frames.joint_position, g1_coordinates)
    torch.testing.assert_close(g1_frames.field("root_position"), g1_root)

    smpl_skeleton = cmu_humenv_smpl_skeleton()
    smpl_q = torch.zeros(4, 7 + smpl_skeleton.num_joints)
    smpl_q[:, :3].copy_(g1_root)
    smpl_q[:, 3] = 1.0
    smpl_q[:, 7:] = torch.linspace(-0.3, 0.3, 4 * smpl_skeleton.num_joints).view(4, -1)
    smpl_qd = torch.zeros(4, 6 + smpl_skeleton.num_joints)
    smpl_builder = _smpl_builder(reference_path)
    smpl_frames = smpl_builder.build_generalized_frames(smpl_q, smpl_qd)
    live_names = smpl_live_joint_mujoco_names(SMPL_LIVE_JOINT_NAMES)
    live_indices = torch.tensor([smpl_skeleton.joint_names.index(name) for name in live_names])

    torch.testing.assert_close(smpl_frames.joint_position, smpl_q[:, 7:].index_select(1, live_indices))
    torch.testing.assert_close(smpl_frames.field("root_position"), g1_root)


def test_lafan_typed_clip_rejects_source_dtype_instead_of_repairing_it() -> None:
    with pytest.raises(ValueError, match="float32"):
        LafanG1Clip(
            root_translation=np.zeros((5, 3), dtype=np.float64),
            pose_axis_angle=np.zeros((5, 30, 3), dtype=np.float32),
            source_fps=30.0,
        )


def test_smpl_builder_maps_live_axes_and_materializes_physical_body_fields(reference_path: Path) -> None:
    builder = _smpl_builder(reference_path)
    skeleton = cmu_humenv_smpl_skeleton()
    frame_count = 4
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qvel = np.zeros((frame_count, 75), dtype=np.float32)
    qpos[:, 2] = 1.0
    qpos[:, 3] = np.cos(0.25 * np.pi)
    qpos[:, 6] = np.sin(0.25 * np.pi)
    qpos[:, 7:] = np.arange(69, dtype=np.float32)
    qvel[:, 3] = 2.0
    clip = _cmu_clip(qpos, qvel)
    frames = _smpl_exact_frames(builder, clip)

    live_source_names = smpl_live_joint_mujoco_names(SMPL_LIVE_JOINT_NAMES)
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
    body_count = cmu_humenv_smpl_skeleton().num_bodies
    body_index = torch.arange(body_count, dtype=torch.float32)
    body_com = torch.stack(
        (
            0.13 + 0.011 * body_index,
            -0.09 + 0.007 * body_index,
            0.04 - 0.005 * body_index,
        ),
        dim=-1,
    )
    builder = _smpl_builder(reference_path, body_com=body_com)

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

    frames = _smpl_exact_frames(builder, _cmu_clip(qpos, qvel))

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
    frames = _smpl_exact_frames(builder, decoded)

    assert decoded.generalized_position.dtype == np.float32
    assert decoded.generalized_velocity.dtype == np.float32
    assert frames.body_position.dtype == torch.float32
    assert frames.body_position.shape == (frame_count, cmu_humenv_smpl_skeleton().num_bodies, 3)
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
    _, local_xyzw = _cmu_clip(qpos.numpy()).semantic_local_pose(skeleton, device="cpu")

    axes = torch.eye(3)
    expected = torch.zeros(3, 4)
    expected[:, 3] = 1.0
    for index in range(3):
        expected = quat_mul(
            expected,
            quat_from_rotation_vector(qpos[:, 7 + index, None] * axes[index]),
        )
    torch.testing.assert_close(local_xyzw[:, 1], expected)


def test_cross_builder_preserves_target_axis_identity(reference_path: Path) -> None:
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path, reverse_joints=True)
    builder = MotionSemanticProjection(
        source_skeleton=source,
        target=target,
        target_landmarks=_G1_RETARGET_TARGETS,
        root_basis_roles=_G1_ROOT_BASIS_ROLES,
        support_roles=_G1_SUPPORT_ROLES,
        version=_G1_RETARGET_MATH_VERSION,
    )

    assert builder.target is target
    assert builder.version == _G1_RETARGET_MATH_VERSION
    assert len(builder.construction_identity_sha256) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise TF32 matmul policy.")
@pytest.mark.parametrize("target_robot", ("g1", "smpl"), ids=("g1_from_cmu", "smpl_from_lafan"))
def test_cross_projection_stays_finite_with_training_tf32(reference_path: Path, target_robot: str) -> None:
    """Training's TF32 policy must preserve finite unit semantic targets for both cross-compositions."""
    if target_robot == "g1":
        source = cmu_humenv_smpl_skeleton()
        target = _g1_builder(reference_path, device="cuda:0")
        landmarks = _G1_RETARGET_TARGETS
        root_basis_roles = _G1_ROOT_BASIS_ROLES
        support_roles = _G1_SUPPORT_ROLES
        version = _G1_RETARGET_MATH_VERSION
    else:
        source = lafan_g1_29dof_skeleton()
        target = _smpl_builder(reference_path, device="cuda:0")
        landmarks = _SMPL_RETARGET_TARGETS
        root_basis_roles = _SMPL_ROOT_BASIS_ROLES
        support_roles = _SMPL_SUPPORT_ROLES
        version = _SMPL_RETARGET_MATH_VERSION

    frame_count = 128
    rotation_vector = torch.linspace(
        -2.8,
        2.8,
        frame_count * source.num_bodies * 3,
        dtype=torch.float32,
        device="cuda",
    ).reshape(frame_count, source.num_bodies, 3)
    if target_robot == "smpl":
        rotation_vector[:, 1:].zero_()
        joint_angles = torch.linspace(
            -2.8,
            2.8,
            frame_count * source.num_joints,
            dtype=torch.float32,
            device="cuda",
        ).reshape(frame_count, source.num_joints)
        child_indices = torch.tensor(source.joint_child_body_indices, dtype=torch.int64, device="cuda")
        joint_axes = torch.tensor(source.joint_axes, dtype=torch.float32, device="cuda")
        rotation_vector[:, child_indices] = joint_angles[..., None] * joint_axes[None]
    root_position = torch.zeros(frame_count, 3, dtype=torch.float32, device="cuda")
    root_position[:, 0] = torch.linspace(-0.4, 0.4, frame_count, device="cuda")
    root_position[:, 2] = 1.0

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        projection = MotionSemanticProjection(source, target, landmarks, root_basis_roles, support_roles, version)
        targets = projection.generate_targets(root_position, quat_from_rotation_vector(rotation_vector))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    assert version.endswith("_v4")
    prior_version = f"{version[:-1]}3"
    prior_projection = MotionSemanticProjection(
        source, target, landmarks, root_basis_roles, support_roles, prior_version
    )
    assert projection.construction_identity_sha256 != prior_projection.construction_identity_sha256
    assert torch.isfinite(targets.position_m).all()
    assert torch.isfinite(targets.rotation_xyzw).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(targets.rotation_xyzw, dim=-1),
        torch.ones(targets.rotation_xyzw.shape[:-1], device="cuda"),
        atol=5.0e-6,
        rtol=5.0e-6,
    )
