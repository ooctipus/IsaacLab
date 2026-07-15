# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-builder oracles for native SMPL, native G1, and cross projection."""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import newton
import newton.ik as ik
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_from_rotation_vector,
    quat_mul,
)

from isaaclab_tasks.core.multi_task.kinematics import (
    IKTrajectorySolver,
    KinematicTree,
    fit_ordered_hinge_coordinates,
    kinematic_pose_forward,
    kinematic_root_basis,
    kinematic_tree_forward,
    ordered_hinge_rotation,
    time_gaussian_filter,
    time_gradient,
    time_quaternion_angular_velocity,
)
from isaaclab_tasks.core.multi_task.kinematics.newton_kinematics import NewtonKinematics
from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex
from isaaclab_tasks.core.multi_task.motion.data.frames import MotionGeneralizedCoordinates
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    CmuHumEnvSmplClip,
    CmuHumEnvSmplClips,
    LafanG1Clip,
    cmu_humenv_smpl_skeleton,
    lafan_g1_29dof_skeleton,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
    MotionSourceDirectionPointObjectiveCfg,
    MotionSourceGlobalPositionObjectiveCfg,
    MotionSourceRotationObjectiveCfg,
    MotionTaskTableCfg,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _TRAJECTORY_METRIC_NAMES
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
    _FRAME_SEED_LOCAL_CANDIDATES,
    _METRIC_CONTACT_APPLICABLE,
    _METRIC_CONTACT_CUMULATIVE_DRIFT,
    _METRIC_CONTACT_GAP,
    _METRIC_CONTACT_SLIP_SPEED,
    _METRIC_CONTACT_STABLE_COUNT,
    _METRIC_CONTACT_TILT,
    _METRIC_SOURCE_ALL_DISTAL_DIRECTION,
    _METRIC_SOURCE_ALL_DISTAL_POSITION,
    _METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
    _METRIC_SOURCE_ALL_POSITION,
    _METRIC_SOURCE_CONTACT_CONFIDENCE,
    _METRIC_SOURCE_NONROOT_ROTATION,
    _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
    _METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
    _METRIC_SOURCE_REQUIRED_POSITION,
    _METRIC_SOURCE_ROOT_ROTATION,
    _motion_frame_seed_targets,
    _motion_workspace_targets,
    _trajectory_tensor_storage_bytes,
    motion_objective_source_direction_point,
    motion_objective_source_global_position,
    motion_objective_source_rotation,
)
from isaaclab_tasks.core.multi_task.motion.retarget import MotionTrajectoryProjection, motion_contact_probe_offsets
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_OFFSET_M, G1_HEAD_PARENT_BODY_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import (
    _G1_RETARGET_DIRECTION_POINTS,
    _G1_RETARGET_MATH_VERSION,
    _g1_coordinates_match,
    _G1FrameTarget,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import smpl_live_joint_mujoco_names
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
    _SMPL_RETARGET_MATH_VERSION,
    _smpl_coordinates_match,
    _SmplFrameTarget,
)

from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

SMPL_LIVE_JOINT_NAMES = tuple(
    f"{body}_x_{body}_y_{body}_z:{component}" for body in MUJOCO_BODY_NAMES[1:] for component in range(3)
)


_CONTACT_CHANNELS = (
    MotionTaskTableCfg.ContactChannelCfg(name="left_foot", source_probe_roles=("left_ankle", "left_toe")),
    MotionTaskTableCfg.ContactChannelCfg(name="right_foot", source_probe_roles=("right_ankle", "right_toe")),
)


def _trajectory_projection(source, target, contact_channels=_CONTACT_CHANNELS) -> MotionTrajectoryProjection:
    """Build one projection with a target-device canonical contact layout."""
    offsets = motion_contact_probe_offsets(contact_channels, target.frame_target.kinematics.device)
    return MotionTrajectoryProjection(source, target, contact_channels, offsets)


def _contact_patches(left_body: str, right_body: str):
    return (
        MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
            channel="left_foot",
            body_name=left_body,
        ),
        MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
            channel="right_foot",
            body_name=right_body,
        ),
    )


_G1_CONTACT_PATCHES = _contact_patches("left_ankle_roll_link", "right_ankle_roll_link")
_SMPL_CONTACT_PATCHES = _contact_patches("L_Ankle", "R_Ankle")


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
        self.mechanics_identity_sha256 = _file_sha256(path)
        self.device = device
        self.n_root_coords = 7
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
        self.default_joint_q = np.zeros(7 + skeleton.num_joints, dtype=np.float32)
        self.default_joint_q[6] = 1.0
        self.default_body_q = torch.cat((rest_position, rest_rotation), dim=-1).numpy()
        self.builder = SimpleNamespace(
            body_count=skeleton.num_bodies,
            shape_body=list(range(skeleton.num_bodies)),
            shape_flags=[int(newton.ShapeFlags.COLLIDE_SHAPES)] * skeleton.num_bodies,
            shape_type=[int(newton.GeoType.BOX)] * skeleton.num_bodies,
            shape_scale=[np.full(3, 0.02, dtype=np.float32)] * skeleton.num_bodies,
            shape_transform=[np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)] * skeleton.num_bodies,
            shape_source=[None] * skeleton.num_bodies,
        )
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
        body_joint = np.full(skeleton.num_bodies, -1, dtype=np.int32)
        body_joint[np.asarray(joint_child, dtype=np.int32)] = np.arange(len(joint_child), dtype=np.int32)
        joint_transform = np.zeros((len(joint_child), 7), dtype=np.float32)
        joint_transform[:, 6] = 1.0
        self.topology = SimpleNamespace(
            body_count=skeleton.num_bodies,
            joint_count=len(self.joint_names),
            coordinate_count=7 + skeleton.num_joints,
            joint_parent=np.asarray(joint_parent, dtype=np.int32),
            joint_child=np.asarray(joint_child, dtype=np.int32),
            joint_transform_parent=joint_transform.copy(),
            joint_transform_child=joint_transform.copy(),
            joint_q_start=np.asarray(joint_q_start, dtype=np.int32),
            joint_qd_start=np.asarray(joint_qd_start, dtype=np.int32),
            joint_dof_dim=np.asarray(joint_dof_dim, dtype=np.int32),
            joint_axis=joint_axis.numpy().copy(),
            joint_limit_lower=np.full(6 + skeleton.num_joints, -np.inf, dtype=np.float32),
            joint_limit_upper=np.full(6 + skeleton.num_joints, np.inf, dtype=np.float32),
            body_parent=np.asarray(skeleton.parent_indices, dtype=np.int32),
            body_joint=body_joint,
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


def _g1_builder(reference_path: Path, *, reverse_joints: bool = False, device: str = "cpu") -> _G1FrameTarget:
    skeleton = lafan_g1_29dof_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path, device=device)
    joint_names = skeleton.joint_names[::-1] if reverse_joints else skeleton.joint_names
    return _G1FrameTarget(
        kinematic_tree=_kinematic_tree(skeleton),
        pose_coordinate_identity_sha256=skeleton.identity_sha256,
        kinematics=reference,
        reference_mechanics_sha256=reference.mechanics_identity_sha256,
        live_joint_names=joint_names,
        live_body_names=skeleton.body_names,
        contact_patches=_G1_CONTACT_PATCHES,
    )


def test_g1_builder_rejects_invalid_pose_coordinate_digest(reference_path: Path) -> None:
    """A pose-coordinate contract must carry one canonical SHA-256 identity."""
    with pytest.raises(ValueError, match="pose_coordinate_identity_sha256"):
        replace(_g1_builder(reference_path), pose_coordinate_identity_sha256="invalid")


def test_g1_wrist_landmarks_and_hand_endpoints_are_explicit(reference_path: Path) -> None:
    """G1 endpoints use ProtoMotions' foot, hand-auxiliary, and physical rubber-hand axes."""
    target = _g1_builder(reference_path).trajectory_target
    landmarks = {landmark.role: landmark for landmark in target.landmarks}

    assert landmarks["left_wrist"].position_body_name == "left_wrist_yaw_link"
    assert landmarks["right_wrist"].position_body_name == "right_wrist_yaw_link"
    assert target.direction_points == _G1_RETARGET_DIRECTION_POINTS
    assert target.required_direction_roles == ("left_foot", "right_foot", "left_hand_endpoint", "right_hand_endpoint")
    assert target.required_position_roles == ("pelvis", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
    assert target.required_position_rows == tuple(target.roles.index(role) for role in target.required_position_roles)
    assert target.required_direction_rows == tuple(
        target.direction_roles.index(role) for role in target.required_direction_roles
    )
    torch.testing.assert_close(target.required_position_row_tensor, torch.tensor(target.required_position_rows))
    assert target.contact_direction_rows == (0, 1)
    assert target.required_direction_rows == (0, 1, 4, 5)
    assert tuple(target.direction_contact_channel_slots.tolist()) == (0, 1, -1, -1, -1, -1)
    torch.testing.assert_close(target.contact_direction_row_tensor, torch.tensor((0, 1)))
    torch.testing.assert_close(target.required_direction_row_tensor, torch.tensor((0, 1, 4, 5)))
    # fmt: off
    assert tuple(
        (
            point.name,
            point.base_role,
            point.source_from_role,
            point.source_to_role,
            point.source_direction_law,
            point.body_name,
            point.point_body_m,
        )
        for point in target.direction_points
    ) == (
        (
            "left_foot", "left_ankle", "left_ankle", "left_toe", "between_positions",
            "left_ankle_roll_link", (0.15, 0.0, 0.0),
        ),
        (
            "right_foot", "right_ankle", "right_ankle", "right_toe", "between_positions",
            "right_ankle_roll_link", (0.15, 0.0, 0.0),
        ),
        (
            "left_hand", "left_wrist", "left_elbow", "left_wrist", "wrist_forward",
            "left_wrist_yaw_link", (0.0, 0.0, 0.14),
        ),
        (
            "right_hand", "right_wrist", "right_elbow", "right_wrist", "wrist_forward",
            "right_wrist_yaw_link", (0.0, 0.0, 0.14),
        ),
        (
            "left_hand_endpoint", "left_wrist", "left_elbow", "left_wrist", "between_positions",
            "left_wrist_yaw_link", (0.12, 0.0, 0.0),
        ),
        (
            "right_hand_endpoint", "right_wrist", "right_elbow", "right_wrist", "between_positions",
            "right_wrist_yaw_link", (0.12, 0.0, 0.0),
        ),
    )
    # fmt: on

    points = list(target.direction_points)
    points[0] = replace(points[0], source_direction_law="unknown")
    with pytest.raises(ValueError, match="duplicate/missing roles"):
        replace(target, direction_points=tuple(points))


def test_g1_hand_endpoint_objective_observes_wrist_yaw(reference_path: Path) -> None:
    """The physical hand axis supplies the wrist-yaw Jacobian missing from the auxiliary axis."""
    target = _g1_builder(reference_path).trajectory_target
    points = {point.name: point for point in target.direction_points}
    tree = target.frame_target.kinematic_tree
    yaw_row = tree.coordinate_names.index("right_wrist_yaw_joint")
    yaw_axis = torch.tensor(tree.coordinate_axes[yaw_row], dtype=torch.float32)
    torch.testing.assert_close(yaw_axis, torch.tensor((0.0, 0.0, 1.0)))

    endpoint = torch.tensor(points["right_hand_endpoint"].point_body_m)
    auxiliary = torch.tensor(points["right_hand"].point_body_m)
    yaw_rotation = quat_from_rotation_vector(0.2 * yaw_axis)
    endpoint_displacement = torch.linalg.vector_norm(quat_apply(yaw_rotation, endpoint) - endpoint)
    auxiliary_displacement = torch.linalg.vector_norm(quat_apply(yaw_rotation, auxiliary) - auxiliary)
    assert endpoint_displacement > 0.023
    torch.testing.assert_close(auxiliary_displacement, torch.tensor(0.0), atol=1.0e-7, rtol=0.0)

    epsilon = 1.0e-3
    positive = quat_apply(quat_from_rotation_vector(epsilon * yaw_axis), endpoint)
    negative = quat_apply(quat_from_rotation_vector(-epsilon * yaw_axis), endpoint)
    endpoint_yaw_jacobian = (positive - negative) / (2.0 * epsilon)
    torch.testing.assert_close(
        torch.linalg.vector_norm(endpoint_yaw_jacobian), torch.tensor(0.12), atol=1.0e-5, rtol=0.0
    )


def test_g1_trajectory_target_uses_only_protomotions_root_orientation(reference_path: Path) -> None:
    """Raw-keypoint G1 fitting preserves root orientation without uncalibrated nonroot BVH rotations."""
    target = _g1_builder(reference_path).trajectory_target

    assert target.source_root_policy == "optimized"
    assert target.root_body_index == target.frame_target.kinematic_tree.root_body_index
    assert target.position_body_indices[0] == target.root_body_index
    assert tuple((item.role, item.body_name, item.weight) for item in target.rotation_landmarks) == (
        ("pelvis", "pelvis", 2.0),
    )
    assert target.rotation_roles[0] == target.roles[0] == "pelvis"
    assert target.rotation_body_indices[0] == target.root_body_index
    assert target.rotation_weights == tuple(item.weight for item in target.rotation_landmarks)


def test_smpl_trajectory_target_uses_the_protomotions_rotation_map(reference_path: Path) -> None:
    """SMPL keeps ProtoMotions' mapped rotations and fixes its authored root during source refinement."""
    target = _smpl_builder(reference_path).trajectory_target

    assert target.source_root_policy == "fixed"
    assert tuple((item.role, item.body_name, item.weight) for item in target.rotation_landmarks) == (
        ("pelvis", "Pelvis", 10.0),
        ("left_hip", "L_Hip", 1.0),
        ("left_knee", "L_Knee", 1.0),
        ("left_ankle", "L_Ankle", 1.0),
        ("left_toe", "L_Toe", 1.0),
        ("right_hip", "R_Hip", 1.0),
        ("right_knee", "R_Knee", 1.0),
        ("right_ankle", "R_Ankle", 1.0),
        ("right_toe", "R_Toe", 1.0),
        ("torso", "Torso", 1.0),
        ("spine", "Spine", 1.0),
        ("chest", "Chest", 1.0),
        ("neck", "Neck", 1.0),
        ("head", "Head", 1.0),
        ("left_thorax", "L_Thorax", 1.0),
        ("left_shoulder", "L_Shoulder", 1.0),
        ("left_elbow", "L_Elbow", 1.0),
        ("left_wrist", "L_Wrist", 1.0),
        ("right_thorax", "R_Thorax", 1.0),
        ("right_shoulder", "R_Shoulder", 1.0),
        ("right_elbow", "R_Elbow", 1.0),
        ("right_wrist", "R_Wrist", 1.0),
    )
    assert target.rotation_roles[0] == "pelvis"
    assert target.rotation_body_indices[0] == target.root_body_index
    assert target.direction_roles == ("left_foot", "right_foot")
    # fmt: off
    assert target.required_position_roles == (
        "pelvis", "head", "left_ankle", "right_ankle", "left_toe", "right_toe", "left_wrist", "right_wrist"
    )
    # fmt: on
    assert target.required_direction_roles == target.direction_roles
    assert target.required_position_rows == tuple(target.roles.index(role) for role in target.required_position_roles)
    assert target.contact_direction_rows == target.required_direction_rows == (0, 1)
    assert tuple(target.direction_contact_channel_slots.tolist()) == (0, 1)
    torch.testing.assert_close(target.required_position_row_tensor, torch.tensor(target.required_position_rows))
    torch.testing.assert_close(target.contact_direction_row_tensor, torch.tensor((0, 1)))
    torch.testing.assert_close(target.required_direction_row_tensor, torch.tensor((0, 1)))


def test_smpl_calibrated_seed_improves_rotations_while_anatomical_root_keeps_default_nonroot(
    reference_path: Path,
) -> None:
    """The Proto seed fits calibrated globals while anatomical evidence cannot invent nonroot pose."""
    source = cmu_humenv_smpl_skeleton()
    frame_target = _smpl_builder(reference_path)
    frame_count = 9
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    root_position[:, 0] = torch.linspace(0.0, 0.08, frame_count)
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    body_by_name = {name: index for index, name in enumerate(source.body_names)}
    for body_name, rotation_vector in (
        ("L_Hip", (0.18, -0.08, 0.05)),
        ("Spine", (-0.06, 0.12, 0.04)),
        ("L_Shoulder", (0.10, 0.02, -0.09)),
        ("R_Elbow", (-0.04, 0.15, 0.03)),
    ):
        vector = torch.tensor(rotation_vector, dtype=torch.float32).expand(frame_count, -1)
        local_rotation[:, body_by_name[body_name]] = quat_from_rotation_vector(vector)

    calibrated = _trajectory_projection(source, frame_target.trajectory_target).generate_targets(
        root_position, local_rotation
    )
    anatomical_source = replace(source, landmark_rotation_policy="anatomical_root")
    anatomical = _trajectory_projection(anatomical_source, frame_target.trajectory_target).generate_targets(
        root_position, local_rotation
    )
    default_joint_q = torch.tensor(frame_target.kinematics.default_joint_q, dtype=torch.float32)
    calibrated_baseline = default_joint_q.expand(frame_count, -1).clone()
    calibrated_baseline[:, :7].copy_(calibrated.initial_joint_q[:, :7])
    anatomical_baseline = default_joint_q.expand(frame_count, -1).clone()
    anatomical_baseline[:, :7].copy_(anatomical.initial_joint_q[:, :7])

    def mapped_rotation_error(joint_q: torch.Tensor) -> torch.Tensor:
        reference_coordinate = joint_q[:, 7:]
        canonical_coordinate = torch.empty_like(reference_coordinate)
        canonical_coordinate.index_copy_(1, frame_target._reference_from_canonical_indices, reference_coordinate)
        joint_rotation = ordered_hinge_rotation(
            canonical_coordinate.view(frame_count, -1, 3),
            torch.eye(3, dtype=torch.float32),
        )
        local = torch.empty((frame_count, frame_target.kinematic_tree.num_bodies, 4), dtype=torch.float32)
        local[:, 0].copy_(joint_q[:, 3:7])
        target_rest_local = frame_target._target_default_local_rotation_xyzw[None, 1:].expand(frame_count, -1, -1)
        local[:, 1:].copy_(quat_mul(target_rest_local, joint_rotation))
        world = local.clone()
        for body, parent in enumerate(frame_target.kinematic_tree.parent_indices[1:], start=1):
            world[:, body] = quat_mul(world[:, parent], local[:, body])
        actual = world[:, list(calibrated.rotation_body_indices)]
        desired = calibrated.source_landmark_rotation_xyzw.transpose(0, 1)
        alignment = torch.sum(actual * desired, dim=-1).abs().clamp(max=1.0)
        return (2.0 * torch.acos(alignment)).mean()

    assert calibrated.source_landmark_rotation_xyzw.shape == (22, frame_count, 4)
    assert anatomical.source_landmark_rotation_xyzw.shape == (1, frame_count, 4)
    assert calibrated.initial_joint_q.shape[0] == anatomical.initial_joint_q.shape[0] == frame_count
    assert torch.max(torch.abs(calibrated.initial_joint_q[:, 7:] - calibrated_baseline[:, 7:])) > 1.0e-3
    torch.testing.assert_close(anatomical.initial_joint_q[:, 7:], anatomical_baseline[:, 7:])
    torch.testing.assert_close(calibrated.initial_joint_q[:, :7], calibrated_baseline[:, :7])
    torch.testing.assert_close(anatomical.initial_joint_q[:, :7], anatomical_baseline[:, :7])
    assert mapped_rotation_error(calibrated.initial_joint_q) < 0.1 * mapped_rotation_error(calibrated_baseline)


@pytest.mark.parametrize("rotation_policy", ("anatomical_root", "calibrated_body"))
def test_rotation_policy_aligns_bases_and_freezes_the_seed_root(reference_path: Path, rotation_policy: str) -> None:
    """Every policy uses the anatomical root while calibrated body rows retain relative rotations."""
    source = replace(cmu_humenv_smpl_skeleton(), landmark_rotation_policy=rotation_policy)
    target = _smpl_builder(reference_path).trajectory_target
    projection = _trajectory_projection(source, target)
    frame_count = 7
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    root_position[:, 0] = torch.linspace(0.0, 0.12, frame_count)
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    root_vector = torch.zeros((frame_count, 3), dtype=torch.float32)
    root_vector[:, 2] = torch.linspace(-0.4, 0.35, frame_count)
    local_rotation[:, 0] = quat_from_rotation_vector(root_vector)
    body_by_name = {name: index for index, name in enumerate(source.body_names)}
    hip_vector = torch.zeros((frame_count, 3), dtype=torch.float32)
    hip_vector[:, 0] = torch.linspace(-0.15, 0.18, frame_count)
    local_rotation[:, body_by_name["L_Hip"]] = quat_from_rotation_vector(hip_vector)
    spine_vector = torch.zeros((frame_count, 3), dtype=torch.float32)
    spine_vector[:, 1] = torch.linspace(0.12, -0.1, frame_count)
    local_rotation[:, body_by_name["Torso"]] = quat_from_rotation_vector(spine_vector)

    targets = projection.generate_targets(root_position, local_rotation)
    source_position, source_world_rotation = kinematic_pose_forward(
        projection._source_rest_translation_m,
        projection._source_rest_rotation_xyzw,
        local_rotation,
        root_position,
        source.parent_indices,
    )
    source_anatomical_rotation = quat_from_matrix(
        kinematic_root_basis(source_position, *projection._source_anatomical_basis_body_indices)
    )
    target_anatomical_rotation = target.anatomical_rotation_xyzw.expand(frame_count, 4)
    generated_world_anatomy = quat_mul(targets.source_landmark_rotation_xyzw[0], target_anatomical_rotation)
    alignment = torch.sum(generated_world_anatomy * source_anatomical_rotation, dim=-1).abs()

    assert len(set(target.anatomical_basis_body_indices)) == 4
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=2.0e-6, rtol=0.0)
    assert torch.equal(targets.initial_joint_q[:, 3:7], targets.source_landmark_rotation_xyzw[0])
    if rotation_policy == "anatomical_root":
        assert projection._source_rotation_body_indices.numel() == 0
        assert projection._source_to_target_rotation_xyzw.shape == (0, 4)
        return

    mapped_rotation = quat_mul(
        source_world_rotation.index_select(1, projection._source_rotation_body_indices),
        projection._source_to_target_rotation_xyzw.unsqueeze(0).expand(frame_count, -1, -1),
    )
    generated_rotation = targets.source_landmark_rotation_xyzw.transpose(0, 1)
    mapped_relative = quat_mul(quat_conjugate(mapped_rotation[:, :1].expand_as(mapped_rotation)), mapped_rotation)
    generated_relative = quat_mul(
        quat_conjugate(generated_rotation[:, :1].expand_as(generated_rotation)), generated_rotation
    )
    relative_alignment = torch.sum(mapped_relative * generated_relative, dim=-1).abs()
    relative_alignment = relative_alignment / (
        torch.linalg.vector_norm(mapped_relative, dim=-1) * torch.linalg.vector_norm(generated_relative, dim=-1)
    )
    torch.testing.assert_close(relative_alignment, torch.ones_like(relative_alignment), atol=2.0e-6, rtol=0.0)


def test_rotation_and_direction_construction_facts_change_identity(reference_path: Path) -> None:
    """Rotation, direction, and publication-role facts all participate in target identity."""
    target = _smpl_builder(reference_path).trajectory_target

    landmarks = list(target.landmarks)
    landmarks[1] = replace(landmarks[1], weight=0.5)
    position_weight_target = replace(target, landmarks=tuple(landmarks))

    rotations = list(target.rotation_landmarks)
    rotations[1] = replace(rotations[1], weight=0.5)
    rotation_weight_target = replace(target, rotation_landmarks=tuple(rotations))

    rotations = list(target.rotation_landmarks)
    shoulder_row = target.rotation_roles.index("left_shoulder")
    rotations[shoulder_row] = replace(rotations[shoulder_row], body_name="L_Hand")
    body_target = replace(target, rotation_landmarks=tuple(rotations))

    points = list(target.direction_points)
    points[0] = replace(points[0], point_body_m=(0.01, 0.0, 0.0))
    point_target = replace(target, direction_points=tuple(points))

    points = list(target.direction_points)
    points[0] = replace(points[0], weight=0.5)
    direction_weight_target = replace(target, direction_points=tuple(points))

    points = list(target.direction_points)
    points[0] = replace(points[0], base_role="left_elbow")
    with pytest.raises(ValueError, match="must start at its contact-patch"):
        replace(target, direction_points=tuple(points))

    points = list(target.direction_points)
    points[0] = replace(points[0], source_from_role="left_knee")
    source_role_target = replace(target, direction_points=tuple(points))

    required_position_target = replace(target, required_position_roles=target.required_position_roles[:-1])

    target_identities = {
        item.construction_identity_sha256
        for item in (
            target,
            position_weight_target,
            rotation_weight_target,
            body_target,
            point_target,
            direction_weight_target,
            source_role_target,
            required_position_target,
        )
    }
    assert len(target_identities) == 8

    g1_target = _g1_builder(reference_path).trajectory_target
    points = list(g1_target.direction_points)
    hand_row = g1_target.direction_roles.index("left_hand")
    points[hand_row] = replace(points[hand_row], source_direction_law="between_positions")
    law_target = replace(g1_target, direction_points=tuple(points))
    required_direction_target = replace(
        g1_target, required_direction_roles=(*g1_target.required_direction_roles, "left_hand")
    )
    with pytest.raises(ValueError, match="duplicate/missing roles"):
        replace(g1_target, required_direction_roles=("left_foot", *g1_target.required_direction_roles[2:]))
    assert required_direction_target.contact_direction_rows == g1_target.contact_direction_rows
    assert required_direction_target.required_direction_rows != g1_target.required_direction_rows
    assert len({item.construction_identity_sha256 for item in (g1_target, law_target, required_direction_target)}) == 3

    source = cmu_humenv_smpl_skeleton()
    projection = _trajectory_projection(source, target)
    landmarks = list(source.landmarks)
    toe_row = tuple(item.name for item in landmarks).index("left_toe")
    landmarks[toe_row] = replace(landmarks[toe_row], position_body_name="L_Hand")
    changed_source = replace(source, landmarks=tuple(landmarks))
    changed_projection = _trajectory_projection(changed_source, target)

    assert projection.construction_identity_sha256 != changed_projection.construction_identity_sha256


def test_target_objective_weights_must_be_finite_positive(reference_path: Path) -> None:
    """Invalid target-owned objective weights fail at target construction."""
    target = _smpl_builder(reference_path).trajectory_target

    landmarks = list(target.landmarks)
    landmarks[1] = replace(landmarks[1], weight=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(target, landmarks=tuple(landmarks))

    points = list(target.direction_points)
    points[0] = replace(points[0], weight=float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        replace(target, direction_points=tuple(points))


def test_target_contact_patches_are_three_point_channel_geometry(reference_path: Path) -> None:
    """Each source contact channel owns one finite non-collinear three-point robot patch."""
    target = _g1_builder(reference_path).trajectory_target

    assert target.support_up_frame == "root"
    torch.testing.assert_close(target.support_channel_slots, torch.tensor((0, 0, 0, 1, 1, 1)))
    assert target.contact_channel_names == ("left_foot", "right_foot")
    assert target.support_point_body_m.shape == (6, 3)
    assert torch.all(torch.isfinite(target.support_point_body_m))
    for start in (0, 3):
        patch = target.support_point_body_m[start : start + 3]
        area = torch.linalg.vector_norm(torch.linalg.cross(patch[1] - patch[0], patch[2] - patch[0]))
        assert area > 0.0


def _contact_objectives(frame_count: int, channel_count: int = 1):
    wp.init()
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
        MotionContactObjectiveCfg,
        MotionTrajectorySolveCfg,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import motion_objective_contact

    patch = torch.tensor(((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.3, 0.0)), dtype=torch.float32)
    points = torch.cat(tuple(patch + torch.tensor((float(channel), 0.0, 0.0)) for channel in range(channel_count)))
    targets = SimpleNamespace(
        contact_body_indices=torch.arange(channel_count, dtype=torch.int64),
        support_point_body_m=points,
        contact_normal_body=torch.tensor(((0.0, 0.0, 1.0),) * channel_count, dtype=torch.float32),
        support_patch_offsets=tuple(3 * channel for channel in range(channel_count + 1)),
    )
    support_pose = torch.zeros((frame_count, 7), dtype=torch.float32)
    support_pose[:, 6] = 1.0
    reference = SimpleNamespace(topology=SimpleNamespace(body_dof_ancestry=np.ones((channel_count, 6), dtype=np.uint8)))
    objectives = motion_objective_contact(
        MotionContactObjectiveCfg(), targets, reference, support_pose, MotionTrajectorySolveCfg().contact
    )
    return objectives, support_pose


def test_motion_contact_objectives_are_soft_support_features() -> None:
    """Each channel contributes one cardinality-normalized soft support patch."""
    from isaaclab_tasks.core.multi_task.kinematics import IKObjectiveSupportPatch

    objectives, support_pose = _contact_objectives(frame_count=4, channel_count=2)

    assert [type(objective) for objective in objectives] == [IKObjectiveSupportPatch, IKObjectiveSupportPatch]
    assert [objective.residual_dim() for objective in objectives] == [13, 13]
    assert all(objective._support_pose_t is support_pose for objective in objectives)
    expected_inverse = 1.0 / (0.15 * math.sqrt(3.0))
    assert all(math.isclose(objective.inverse_slip_scale, expected_inverse) for objective in objectives)
    assert all(objective.supports_analytic() for objective in objectives)


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

    _, local_xyzw = clip.local_pose(skeleton, device="cpu")
    expected = quat_from_rotation_vector(torch.from_numpy(pose))

    torch.testing.assert_close(local_xyzw, expected)


def _smpl_builder(
    reference_path: Path, *, body_com: torch.Tensor | None = None, device: str = "cpu"
) -> _SmplFrameTarget:
    skeleton = cmu_humenv_smpl_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path, smpl=True, body_com=body_com, device=device)
    return _SmplFrameTarget(
        kinematics=reference,
        reference_mechanics_sha256=reference.mechanics_identity_sha256,
        live_joint_names=SMPL_LIVE_JOINT_NAMES,
        live_body_names=skeleton.body_names,
        contact_patches=_SMPL_CONTACT_PATCHES,
    )


def test_humenv_smpl_source_declares_shared_torso_roles() -> None:
    """HumEnv SMPL coordinates expose the same intermediate torso semantics as raw AMASS and LAFAN."""
    landmarks = {landmark.name: landmark for landmark in cmu_humenv_smpl_skeleton().landmarks}

    assert {
        role: landmarks[role].position_body_name for role in ("spine", "chest", "neck", "left_thorax", "right_thorax")
    } == {
        "spine": "Spine",
        "chest": "Chest",
        "neck": "Neck",
        "left_thorax": "L_Thorax",
        "right_thorax": "R_Thorax",
    }


def test_smpl_foot_directions_use_toe_origins_independently_of_contact_patches(reference_path: Path) -> None:
    """SMPL foot direction semantics end at toe origins while contact remains ankle collider geometry."""
    frame_target = _smpl_builder(reference_path)
    target = frame_target.trajectory_target
    directions = {point.name: point for point in target.direction_points}

    assert target.support_up_frame == "anatomy"
    assert (directions["left_foot"].body_name, directions["left_foot"].point_body_m) == (
        "L_Toe",
        (0.0, 0.0, 0.0),
    )
    assert (directions["right_foot"].body_name, directions["right_foot"].point_body_m) == (
        "R_Toe",
        (0.0, 0.0, 0.0),
    )
    assert tuple(patch.body_name for patch in frame_target.contact_patches) == ("L_Ankle", "R_Ankle")

    rest_position = torch.from_numpy(frame_target.kinematics.default_body_q[:, :3])
    body_by_name = {name: index for index, name in enumerate(frame_target.kinematics.body_names)}
    body_slots = target.body_normal_channel_slots.cpu()
    position_slots = target.position_normal_channel_slots.cpu()
    role_rows = {role: row for row, role in enumerate(target.roles)}
    assert body_slots[body_by_name["L_Ankle"]] == body_slots[body_by_name["L_Toe"]] == 0
    assert body_slots[body_by_name["R_Ankle"]] == body_slots[body_by_name["R_Toe"]] == 1
    assert body_slots[body_by_name["Pelvis"]] == -1
    assert tuple(position_slots[role_rows[role]].item() for role in ("left_ankle", "left_toe")) == (0, 0)
    assert tuple(position_slots[role_rows[role]].item() for role in ("right_ankle", "right_toe")) == (1, 1)
    assert position_slots[role_rows["pelvis"]] == -1
    expected_lengths = torch.stack(
        (
            torch.linalg.vector_norm(rest_position[body_by_name["L_Toe"]] - rest_position[body_by_name["L_Ankle"]]),
            torch.linalg.vector_norm(rest_position[body_by_name["R_Toe"]] - rest_position[body_by_name["R_Ankle"]]),
        )
    )
    torch.testing.assert_close(target.direction_lengths_m[:2], expected_lengths)


def test_generated_support_patches_preserve_target_root_seed_geometry(reference_path: Path) -> None:
    """Generated patches are target-owned pose seeds before source contact alignment."""
    source = cmu_humenv_smpl_skeleton()
    frame_count = 3
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    target = _smpl_builder(reference_path).trajectory_target
    targets = _trajectory_projection(source, target).generate_targets(root_position, local_rotation)

    support_root = target.support_point_root_m[:, None].expand(-1, frame_count, -1)
    root_rotation = targets.initial_joint_q[:, 3:7][None].expand(len(support_root), -1, -1)
    expected = targets.initial_joint_q[None, :, :3] + quat_apply(
        root_rotation.reshape(-1, 4),
        support_root.reshape(-1, 3),
    ).view_as(support_root)
    torch.testing.assert_close(targets.target_support_position_m, expected, atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("source_name", ("lafan", "cmu"))
def test_projection_uses_complete_target_owned_direction_geometry(reference_path: Path, source_name: str) -> None:
    """Every source drives the same target-owned foot and hand endpoint layout."""
    source = lafan_g1_29dof_skeleton() if source_name == "lafan" else cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path).trajectory_target
    projection = _trajectory_projection(source, target)
    root_position = torch.zeros((2, 3), dtype=torch.float32)
    local_rotation = torch.zeros((2, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0

    targets = projection.generate_targets(root_position, local_rotation)

    assert not hasattr(source, "distal_points")
    assert target.direction_roles == (
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
        "left_hand_endpoint",
        "right_hand_endpoint",
    )
    assert targets.position_weights == target.position_weights
    assert targets.direction_body_indices == target.direction_body_indices
    assert targets.direction_position_rows == target.direction_position_rows
    assert targets.direction_weights == target.direction_weights
    assert targets.direction_length_values_m == target.direction_length_values_m
    torch.testing.assert_close(targets.direction_body_index_tensor, target.direction_body_index_tensor)
    assert targets.required_position_rows == target.required_position_rows
    assert targets.contact_direction_rows == target.contact_direction_rows == (0, 1)
    assert targets.required_direction_rows == target.required_direction_rows == (0, 1, 4, 5)
    assert tuple(target.direction_roles[row] for row in targets.required_direction_rows) == (
        "left_foot",
        "right_foot",
        "left_hand_endpoint",
        "right_hand_endpoint",
    )
    assert tuple(targets.direction_contact_channel_slots.tolist()) == (0, 1, -1, -1, -1, -1)
    torch.testing.assert_close(targets.required_position_row_tensor, target.required_position_row_tensor)
    torch.testing.assert_close(targets.contact_direction_row_tensor, target.contact_direction_row_tensor)
    torch.testing.assert_close(targets.direction_contact_channel_slots, target.direction_contact_channel_slots)
    torch.testing.assert_close(targets.required_direction_row_tensor, target.required_direction_row_tensor)
    torch.testing.assert_close(targets.direction_position_row_tensor, target.direction_rows)
    torch.testing.assert_close(targets.direction_point_body_m, target.direction_point_body_m)
    assert targets.source_direction_point_position_m.shape == (6, 2, 3)


def test_g1_required_hand_endpoints_do_not_expand_contact_evidence(reference_path: Path) -> None:
    """Required noncontact endpoints build strict 3-D objectives from foot-only contact evidence."""
    wp.init()
    source = lafan_g1_29dof_skeleton()
    target = _g1_builder(reference_path).trajectory_target
    root_position = torch.zeros((2, 3), dtype=torch.float32)
    local_rotation = torch.zeros((2, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    targets = _trajectory_projection(source, target).generate_targets(root_position, local_rotation)
    contact_evidence = torch.zeros((2, len(targets.contact_direction_rows)), dtype=torch.float32)

    objectives = motion_objective_source_direction_point(
        MotionSourceDirectionPointObjectiveCfg(),
        targets,
        source_channel_normal_owned=contact_evidence,
        source_channel_confidence=contact_evidence,
    )

    assert len(objectives) == len(targets.direction_body_indices) == 6
    assert tuple(objective.contact_channel for objective in objectives) == (0, 1, -1, -1, -1, -1)


def test_anatomical_root_wrist_forward_is_position_derived_and_orthogonal(reference_path: Path) -> None:
    """Raw LAFAN-style wrist channels cannot affect geometry-derived forward hand evidence."""
    source = replace(lafan_g1_29dof_skeleton(), landmark_rotation_policy="anatomical_root")
    target = _g1_builder(reference_path).trajectory_target
    projection = _trajectory_projection(source, target)
    frame_count = 3
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0

    baseline = projection.generate_targets(root_position, local_rotation)
    rotated_local = local_rotation.clone()
    source_landmarks = {landmark.name: landmark for landmark in source.landmarks}
    for role, vector in (
        ("left_wrist", (0.4, -0.2, 0.1)),
        ("right_wrist", (-0.3, 0.1, 0.2)),
    ):
        body = source.body_names.index(source_landmarks[role].rotation_body_name)
        rotated_local[:, body] = quat_from_rotation_vector(
            torch.tensor(vector, dtype=torch.float32).expand(frame_count, -1)
        )
    rotated = projection.generate_targets(root_position, rotated_local)

    hand_rows = projection._source_wrist_forward_rows
    assert tuple(target.direction_roles[row] for row in hand_rows.tolist()) == ("left_hand", "right_hand")
    assert projection._source_wrist_forward_rotation_body_indices.numel() == 0
    assert projection._source_wrist_forward_local_axis.shape == (0, 3)
    torch.testing.assert_close(
        baseline.source_direction_point_position_m.index_select(0, hand_rows),
        rotated.source_direction_point_position_m.index_select(0, hand_rows),
        atol=1.0e-6,
        rtol=0.0,
    )

    source_position, _ = kinematic_pose_forward(
        projection._source_rest_translation_m,
        projection._source_rest_rotation_xyzw,
        local_rotation,
        root_position,
        source.parent_indices,
    )
    source_from = source_position.index_select(1, projection._source_direction_from_position_indices)
    source_to = source_position.index_select(1, projection._source_direction_to_position_indices)
    forearm = (source_to - source_from).transpose(0, 1).index_select(0, hand_rows)
    forearm = forearm / torch.linalg.vector_norm(forearm, dim=-1, keepdim=True)
    anatomy_forward = kinematic_root_basis(source_position, *projection._source_anatomical_basis_body_indices)[..., 0]
    hand_axis = baseline.source_direction_point_position_m - baseline.source_landmark_position_m.index_select(
        0, target.direction_rows
    )
    hand_axis = hand_axis.index_select(0, hand_rows)
    hand_axis = hand_axis / torch.linalg.vector_norm(hand_axis, dim=-1, keepdim=True)
    torch.testing.assert_close(
        torch.sum(hand_axis * forearm, dim=-1), torch.zeros((2, frame_count)), atol=2.0e-6, rtol=0.0
    )
    assert torch.all(torch.sum(hand_axis * anatomy_forward.unsqueeze(0), dim=-1) > 0.0)


def test_calibrated_body_wrist_forward_transports_rest_anatomical_axis(reference_path: Path) -> None:
    """Calibrated SMPL wrists transport a coordinate-invariant rest-anatomy forward axis."""
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path).trajectory_target
    projection = _trajectory_projection(source, target)
    frame_count = 3
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    source_landmarks = {landmark.name: landmark for landmark in source.landmarks}
    for role, vector in (
        ("left_wrist", (0.25, -0.15, 0.1)),
        ("right_wrist", (-0.2, 0.1, 0.15)),
    ):
        body = source.body_names.index(source_landmarks[role].rotation_body_name)
        local_rotation[:, body] = quat_from_rotation_vector(
            torch.tensor(vector, dtype=torch.float32).expand(frame_count, -1)
        )

    targets = projection.generate_targets(root_position, local_rotation)
    hand_rows = projection._source_wrist_forward_rows
    wrist_bodies = projection._source_wrist_forward_rotation_body_indices
    assert tuple(target.direction_roles[row] for row in hand_rows.tolist()) == ("left_hand", "right_hand")
    assert wrist_bodies.shape == (2,)
    assert projection._source_wrist_forward_local_axis.shape == (2, 3)

    rest_position, rest_world_rotation = kinematic_tree_forward(
        projection._source_rest_translation_m,
        projection._source_rest_rotation_xyzw,
        source.parent_indices,
    )
    rest_forward = kinematic_root_basis(rest_position, *projection._source_anatomical_basis_body_indices)[..., 0]
    expected_local_axis = quat_apply(
        quat_conjugate(rest_world_rotation.index_select(0, wrist_bodies)), rest_forward.expand(2, -1)
    )
    expected_local_axis /= torch.linalg.vector_norm(expected_local_axis, dim=-1, keepdim=True)
    torch.testing.assert_close(projection._source_wrist_forward_local_axis, expected_local_axis, atol=1.0e-6, rtol=0.0)

    _, source_world_rotation = kinematic_pose_forward(
        projection._source_rest_translation_m,
        projection._source_rest_rotation_xyzw,
        local_rotation,
        root_position,
        source.parent_indices,
    )
    wrist_world_rotation = source_world_rotation.index_select(1, wrist_bodies).transpose(0, 1)
    local_axis = expected_local_axis[:, None].expand(-1, frame_count, -1)
    expected_world_axis = quat_apply(wrist_world_rotation.reshape(-1, 4), local_axis.reshape(-1, 3)).view_as(local_axis)
    expected_world_axis /= torch.linalg.vector_norm(expected_world_axis, dim=-1, keepdim=True)
    hand_axis = targets.source_direction_point_position_m - targets.source_landmark_position_m.index_select(
        0, target.direction_rows
    )
    hand_axis = hand_axis.index_select(0, hand_rows)
    hand_axis /= torch.linalg.vector_norm(hand_axis, dim=-1, keepdim=True)
    torch.testing.assert_close(hand_axis, expected_world_axis, atol=1.0e-6, rtol=0.0)


def test_projection_rejects_a_missing_required_semantic_landmark(reference_path: Path) -> None:
    """A source cannot omit a target position or rotation role behind fallback geometry."""
    source = lafan_g1_29dof_skeleton()
    source = replace(source, landmarks=tuple(item for item in source.landmarks if item.name != "left_knee"))
    target = _smpl_builder(reference_path).trajectory_target

    with pytest.raises(ValueError) as error:
        _trajectory_projection(source, target)

    assert "left_knee" in str(error.value)
    assert "rotation roles" in str(error.value)


def test_projection_rejects_nonroot_source_pelvis_carriers(reference_path: Path) -> None:
    """Pelvis translation and orientation cannot silently come from a nonroot source body."""
    source = cmu_humenv_smpl_skeleton()
    pelvis = source.landmarks[0]
    malformed_pelvis = replace(pelvis, position_body_name="L_Hip", rotation_body_name="L_Hip")
    source = replace(source, landmarks=(malformed_pelvis, *source.landmarks[1:]))
    target = _g1_builder(reference_path).trajectory_target

    with pytest.raises(ValueError, match="exact source root body"):
        _trajectory_projection(source, target)


def test_projection_rejects_noncanonical_contact_probe_offsets(reference_path: Path) -> None:
    """Contact prefix rows are derived from channel declarations, not caller-provided aliases."""
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path).trajectory_target
    offsets = motion_contact_probe_offsets(_CONTACT_CHANNELS, target.frame_target.kinematics.device)
    offsets = offsets.clone()
    offsets[1] += 1

    with pytest.raises(ValueError, match="canonical target-device int32 prefix tensor"):
        MotionTrajectoryProjection(source, target, _CONTACT_CHANNELS, offsets)


def test_projection_rejects_nonunit_source_quaternions(reference_path: Path) -> None:
    """The public projection boundary rejects rotations that would distort FK positions."""
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path).trajectory_target
    projection = _trajectory_projection(source, target)
    root_position = torch.zeros((2, 3), dtype=torch.float32)
    local_rotation = torch.zeros((2, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    local_rotation[0, 0, 3] = 2.0

    with pytest.raises(RuntimeError, match="finite unit quaternions"):
        projection.generate_targets(root_position, local_rotation)


def test_newton_factory_forwards_deterministic_multiseed_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kinematics boundary forwards only the sampler controls needed by global frame IK."""
    captured: dict[str, object] = {}
    expected = object()

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(ik, "IKSolver", fake_solver)
    kinematics = object.__new__(NewtonKinematics)
    kinematics.model = object()
    result = kinematics.create_ik_solver(
        [object()],
        3,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
        sampler=ik.IKSampler.GAUSS,
        n_seeds=64,
        noise_std=0.1,
        rng_seed=12345,
    )

    assert result is expected
    assert captured["model"] is kinematics.model
    assert captured["n_problems"] == 3
    assert captured["optimizer"] is ik.IKOptimizer.LM
    assert captured["jacobian_mode"] is ik.IKJacobianType.ANALYTIC
    assert captured["sampler"] is ik.IKSampler.GAUSS
    assert captured["n_seeds"] == 64
    assert captured["noise_std"] == 0.1
    assert captured["rng_seed"] == 12345


@pytest.mark.parametrize("robot", ("g1", "smpl"))
def test_trajectory_named_memory_matches_allocated_robot_tensors(robot: str, reference_path: Path) -> None:
    """The planned target/workspace bytes equal every allocated tensor for each robot shape."""
    if robot == "g1":
        source = lafan_g1_29dof_skeleton()
        builder = _g1_builder(reference_path)
    else:
        source = cmu_humenv_smpl_skeleton()
        builder = _smpl_builder(reference_path)
    frame_count = 3
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    targets = _trajectory_projection(source, builder.trajectory_target).generate_targets(root_position, local_rotation)

    frame_capacity = 11
    batch_clip_count = 2
    residual_count = 97
    physical_inequality_count = 17
    coordinate_count = builder.kinematics.model.joint_coord_count
    dof_count = builder.kinematics.model.joint_dof_count
    body_count = builder.kinematics.model.body_count
    position_count = len(targets.position_body_indices)
    direction_count = len(targets.direction_body_indices)
    rotation_count = len(targets.rotation_body_indices)
    source_probe_count = targets.source_contact_probe_position_m.shape[0]
    contact_channel_count = targets.contact_channel_probe_offsets.shape[0] - 1
    target_support_count = targets.target_support_position_m.shape[0]
    joint_reference_count = targets.coordinate_indices.numel()
    workspace_targets = _motion_workspace_targets(targets, frame_capacity)
    frame_seed_batch_count = batch_clip_count
    global_frame_targets = _motion_frame_seed_targets(targets, frame_seed_batch_count)
    local_frame_targets = _motion_frame_seed_targets(targets, frame_seed_batch_count)
    tensors = {
        "targets.source_landmark_position_m": workspace_targets.source_landmark_position_m,
        "targets.source_landmark_rotation_xyzw": workspace_targets.source_landmark_rotation_xyzw,
        "targets.source_direction_point_position_m": workspace_targets.source_direction_point_position_m,
        "targets.initial_joint_q": workspace_targets.initial_joint_q,
        "targets.source_contact_probe_position_m": workspace_targets.source_contact_probe_position_m,
        "targets.target_support_position_m": workspace_targets.target_support_position_m,
        "workspace.joint_q": torch.empty((frame_capacity, coordinate_count)),
        "workspace.certified_joint_q": torch.empty((frame_capacity, coordinate_count)),
        "workspace.segment_iteration_attempted": torch.empty(batch_clip_count, dtype=torch.int32),
        "workspace.segment_damping": torch.empty(batch_clip_count),
        "workspace.segment_recovery_count": torch.empty(batch_clip_count, dtype=torch.int32),
        "workspace.joint_qd": torch.empty((frame_capacity, dof_count)),
        "workspace.achieved_direction_position_m": torch.empty((direction_count, frame_capacity, 3)),
        "workspace.joint_reference": torch.empty((frame_capacity, joint_reference_count)),
        "workspace.velocity_reachable_lower": torch.empty((frame_capacity, joint_reference_count)),
        "workspace.velocity_reachable_upper": torch.empty((frame_capacity, joint_reference_count)),
        "workspace.body_q": torch.empty((frame_capacity, body_count, 7)),
        "workspace.body_qd": torch.empty((frame_capacity, body_count, 6)),
        "workspace.frame_quality": torch.empty((frame_capacity, len(_TRAJECTORY_METRIC_NAMES))),
        "workspace.source_plane_height_m": torch.empty(batch_clip_count),
        "workspace.source_probe_active": torch.empty((frame_capacity, source_probe_count), dtype=torch.uint8),
        "workspace.source_probe_stable": torch.empty((frame_capacity, source_probe_count), dtype=torch.uint8),
        "workspace.source_channel_normal_owned": torch.empty((frame_capacity, contact_channel_count)),
        "workspace.source_channel_clearance_lift_m": torch.empty((frame_capacity, contact_channel_count)),
        "workspace.source_channel_confidence": torch.empty((frame_capacity, contact_channel_count)),
        "workspace.source_channel_activity": torch.empty((frame_capacity, 2 * contact_channel_count)),
        "workspace.source_channel_stable": torch.empty((frame_capacity, contact_channel_count), dtype=torch.uint8),
        "workspace.source_channel_edge_stable": torch.empty((frame_capacity, contact_channel_count), dtype=torch.uint8),
        "targets.contact_distal_point_body_m": targets.contact_distal_point_body_m,
        "root.frozen_dof_indices": torch.empty(6, dtype=torch.int32),
        "workspace.obstacle_pose": torch.empty((frame_capacity, 7)),
        "workspace.segment_phase_attempted": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.rotation_body_indices": torch.empty(rotation_count, dtype=torch.int64),
        "workspace.segment_active": torch.empty(batch_clip_count, dtype=torch.int32),
        "workspace.segment_iteration_geometry_feasible": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.segment_iteration_inner_converged": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.segment_iteration_globalization_succeeded": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.segment_iteration_residual_constraints_satisfied": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.segment_phase_globalization_succeeded": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.segment_phase_converged": torch.empty(batch_clip_count, dtype=torch.bool),
        "workspace.segment_contact_refinement_required": torch.empty(batch_clip_count, dtype=torch.bool),
        "layout.activity_group_by_residual": torch.empty(residual_count, dtype=torch.int32),
        "layout.first_difference_group_by_residual": torch.empty(residual_count, dtype=torch.int32),
        "physical.inequality_indices": torch.empty(physical_inequality_count, dtype=torch.int32),
        "physical.inequality_upper": torch.empty(physical_inequality_count),
        "workspace.base_weights": torch.empty(residual_count),
        "workspace.temporal_weights": torch.empty((3, residual_count)),
        "workspace.velocity_lower": torch.empty(dof_count),
        "workspace.velocity_upper": torch.empty(dof_count),
        "workspace.source_velocity_lower": torch.empty(dof_count),
        "workspace.source_velocity_upper": torch.empty(dof_count),
        "frame_seed.global.source_landmark_position_m": global_frame_targets.source_landmark_position_m,
        "frame_seed.global.source_landmark_rotation_xyzw": global_frame_targets.source_landmark_rotation_xyzw,
        "frame_seed.global.source_direction_point_position_m": global_frame_targets.source_direction_point_position_m,
        "frame_seed.global.joint_q": global_frame_targets.initial_joint_q,
        "frame_seed.local.source_landmark_position_m": local_frame_targets.source_landmark_position_m,
        "frame_seed.local.source_landmark_rotation_xyzw": local_frame_targets.source_landmark_rotation_xyzw,
        "frame_seed.local.source_direction_point_position_m": local_frame_targets.source_direction_point_position_m,
        "frame_seed.local.joint_q": local_frame_targets.initial_joint_q,
        "frame_seed.local.candidate_joint_q": torch.empty(
            (_FRAME_SEED_LOCAL_CANDIDATES * frame_seed_batch_count, coordinate_count)
        ),
        "frame_seed.local.problem_indices": torch.empty(
            _FRAME_SEED_LOCAL_CANDIDATES * frame_seed_batch_count, dtype=torch.int32
        ),
        "frame_seed.frame_indices": torch.empty(frame_seed_batch_count, dtype=torch.int32),
        "frame_seed.frame_active": torch.empty(frame_seed_batch_count, dtype=torch.int32),
        "batch.clip_offsets": torch.empty(batch_clip_count + 1, dtype=torch.int32),
        "batch.step_seconds": torch.empty(batch_clip_count),
    }
    estimated = _trajectory_tensor_storage_bytes(
        frame_capacity=frame_capacity,
        batch_clip_count=batch_clip_count,
        coordinate_count=coordinate_count,
        frame_seed_batch_count=frame_seed_batch_count,
        dof_count=dof_count,
        body_count=body_count,
        residual_count=residual_count,
        position_count=position_count,
        direction_count=direction_count,
        source_probe_count=source_probe_count,
        rotation_count=rotation_count,
        contact_channel_count=contact_channel_count,
        target_support_count=target_support_count,
        physical_inequality_count=physical_inequality_count,
        joint_reference_count=joint_reference_count,
    )
    allocated = {name: tensor.numel() * tensor.element_size() for name, tensor in tensors.items()}
    assert estimated == allocated
    assert sum(estimated.values()) == sum(allocated.values())


def _g1_pose_frames(
    builder: _G1FrameTarget,
    pose_axis_angle: torch.Tensor,
    root_translation: torch.Tensor,
    source_fps: float,
):
    joint_q = torch.empty(pose_axis_angle.shape[0], 36)
    joint_q[:, :3].copy_(root_translation)
    joint_q[:, 3:7].copy_(quat_from_rotation_vector(pose_axis_angle[:, 0]))
    joint_q[:, 7:].copy_(pose_axis_angle.sum(dim=-1)[:, 1:])
    return builder.build_generalized_frames(joint_q, source_fps)


def _smpl_exact_frames(builder: _SmplFrameTarget, clip: CmuHumEnvSmplClip):
    joint_q, joint_qd = clip.free_root_coordinates(cmu_humenv_smpl_skeleton(), device="cpu")
    assert joint_qd is not None
    generalized_position = torch.cat((joint_q[:, :3], convert_quat(joint_q[:, 3:7], to="wxyz"), joint_q[:, 7:]), dim=-1)
    return builder.build_generalized_frames(generalized_position, joint_qd)


def test_smpl_builder_derives_scalar_coordinates_from_grouped_newton_joints(reference_path: Path) -> None:
    """Canonical SMPL coordinates come from grouped Newton child, range, and axis metadata."""
    builder = _smpl_builder(reference_path)

    assert builder.reference_coordinate_names == cmu_humenv_smpl_skeleton().joint_names


def test_smpl_coordinates_from_newton_derive_d6_velocity_across_angle_branch(
    reference_path: Path,
) -> None:
    """Equivalent hinge-angle branches must not create nonphysical D6 coordinate rates."""
    builder = _smpl_builder(reference_path)
    frame_count = 3
    joint_q = torch.zeros((frame_count, 76), dtype=torch.float32)
    joint_q[:, 6] = 1.0
    joint_q[:, 7] = torch.tensor((math.pi - 0.05, -math.pi + 0.05, -math.pi + 0.15))
    clip_index = MotionClipIndex(
        "0" * 64,
        ("1" * 64,),
        (MotionClipIndex.Clip("branch", frame_count, 30.0, "2" * 64, 0),),
    )

    coordinates = builder.coordinates_from_newton(joint_q, clip_index)
    frames = builder.materialize_coordinates(coordinates, clip_index)

    expected = torch.zeros_like(frames.joint_velocity)
    expected[:, 0] = 3.0
    torch.testing.assert_close(frames.joint_velocity, expected, atol=2.0e-5, rtol=2.0e-5)


def test_smpl_materialization_preserves_exact_generalized_velocity(reference_path: Path) -> None:
    """Exact source D6 rates must pass through materialization unchanged."""
    builder = _smpl_builder(reference_path)
    frame_count = 3
    generalized_position = torch.zeros((frame_count, 76), dtype=torch.float32)
    generalized_position[:, 3] = 1.0
    generalized_velocity = torch.zeros((frame_count, 75), dtype=torch.float32)
    generalized_velocity[:, 6] = torch.tensor((-188.0, 7.0, 11.0))
    coordinates = MotionGeneralizedCoordinates(generalized_position, generalized_velocity)
    clip_index = MotionClipIndex(
        "0" * 64,
        ("1" * 64,),
        (MotionClipIndex.Clip("exact", frame_count, 30.0, "2" * 64, 0),),
    )

    frames = builder.materialize_coordinates(coordinates, clip_index)

    torch.testing.assert_close(frames.joint_velocity[:, 0], generalized_velocity[:, 6])


def test_smpl_builder_rejects_an_unrelated_source_schema(reference_path: Path) -> None:
    """The native SMPL builder owns its exact source-coordinate contract."""
    assert not _smpl_coordinates_match(lafan_g1_29dof_skeleton(), _smpl_builder(reference_path))


def test_exact_route_requires_the_complete_robot_coordinate_profile(reference_path: Path) -> None:
    """Rest geometry and topology changes must route through semantic retargeting."""
    g1_source = lafan_g1_29dof_skeleton()
    g1_target = _g1_builder(reference_path).kinematic_tree
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
    g1_reference_source = (motion / "robots" / "g1" / "reference.py").read_text()
    assert "model_body_collider_extreme_points" not in g1_reference_source


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
    kinematic_tree = builder.kinematic_tree

    assert builder.joint_names == kinematic_tree.joint_names[::-1]
    assert builder.reference_frame_names[:-1] == kinematic_tree.body_names
    assert builder.reference_frame_names[-1] == "head_link"
    assert len(builder.construction_identity_sha256) == 64

    assert builder.kinematics.model.joint_coord_count == 36
    assert builder.kinematics.model.joint_dof_count == 35
    assert kinematic_tree.joint_coordinate_ranges == tuple((index, index + 1) for index in range(29))
    assert kinematic_tree.coordinate_q_indices == tuple(range(7, 36))
    assert kinematic_tree.coordinate_qd_indices == tuple(range(6, 35))


@pytest.mark.parametrize("field", ("joint_coordinate_ranges", "coordinate_q_indices", "coordinate_qd_indices"))
def test_g1_builder_rejects_noncanonical_scalar_coordinate_layout(reference_path: Path, field: str) -> None:
    """G1 construction rejects grouped or permuted q/qd layouts before velocity certification."""
    builder = _g1_builder(reference_path)
    tree = builder.kinematic_tree
    invalid = {
        "joint_coordinate_ranges": ((0, 2), (2, 2), *((index, index + 1) for index in range(2, 29))),
        "coordinate_q_indices": (8, 7, *range(9, 36)),
        "coordinate_qd_indices": (7, 6, *range(8, 35)),
    }[field]
    invalid_tree = replace(tree, **{field: invalid})

    with pytest.raises(ValueError, match="one contiguous scalar coordinate and dof"):
        replace(builder, kinematic_tree=invalid_tree)


def test_g1_builder_reorders_reference_bodies_once_before_head_derivation(reference_path: Path) -> None:
    skeleton = lafan_g1_29dof_skeleton()
    reference = _ReferenceKinematics(skeleton, reference_path)
    live_body_names = (skeleton.body_names[0], *skeleton.body_names[:0:-1])
    builder = _G1FrameTarget(
        kinematic_tree=_kinematic_tree(skeleton),
        pose_coordinate_identity_sha256=skeleton.identity_sha256,
        kinematics=reference,
        reference_mechanics_sha256=reference.mechanics_identity_sha256,
        live_joint_names=skeleton.joint_names,
        live_body_names=live_body_names,
        contact_patches=_G1_CONTACT_PATCHES,
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
    _, decoded = next(source.clips((0,)))
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
    _, local_xyzw = _cmu_clip(qpos.numpy()).local_pose(skeleton, device="cpu")

    axes = torch.eye(3)
    expected = torch.zeros(3, 4)
    expected[:, 3] = 1.0
    for index in range(3):
        expected = quat_mul(
            expected,
            quat_from_rotation_vector(qpos[:, 7 + index, None] * axes[index]),
        )
    torch.testing.assert_close(local_xyzw[:, 1], expected)


def test_landmark_objectives_use_target_owned_rotation_bodies_and_weights() -> None:
    """Position and rotation targets retain their independent target-owned weights."""
    position_indices = (10, 11, 12)
    rotation_indices = position_indices
    rotation_weights = (10.0, 4.0, 2.0)
    targets = SimpleNamespace(
        position_body_indices=position_indices,
        position_weights=(2.0, 3.0, 4.0),
        root_body_index=position_indices[0],
        source_landmark_position_m=torch.zeros(len(position_indices), 1, 3),
        rotation_body_indices=rotation_indices,
        rotation_weights=rotation_weights,
        source_landmark_rotation_xyzw=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 3),
        segment_length_values_m=(2.0, 0.5, 4.0),
    )
    wp.init()
    position_objectives = motion_objective_source_global_position(MotionSourceGlobalPositionObjectiveCfg(), targets)
    rotation_objectives = motion_objective_source_rotation(MotionSourceRotationObjectiveCfg(), targets)

    assert tuple(objective.link_index for objective in position_objectives) == (
        *position_indices[1:],
        position_indices[0],
    )
    assert tuple(objective.weight for objective in position_objectives) == (6.0, 1.0, 10.0)
    assert tuple(objective.link_index for objective in rotation_objectives) == rotation_indices
    assert tuple(objective.weight for objective in rotation_objectives) == rotation_weights


def test_distal_endpoint_uses_standard_absolute_point_objective() -> None:
    """Distal evidence is a target-weighted physical point objective the solver can satisfy."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0)
    root_joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([root_joint])
    model = builder.finalize(device="cpu", requires_grad=True)
    desired = torch.tensor(((0.4, -0.2, 0.8),), dtype=torch.float32)
    targets = SimpleNamespace(
        direction_body_indices=(body,),
        direction_weights=(2.0,),
        direction_point_body_m=torch.tensor(((0.1, 0.0, 0.0),), dtype=torch.float32),
        source_direction_point_position_m=desired.unsqueeze(0),
        direction_length_values_m=(0.1,),
    )
    objectives = motion_objective_source_direction_point(MotionSourceDirectionPointObjectiveCfg(), targets)
    assert len(objectives) == 1
    assert type(objectives[0]) is ik.IKObjectivePosition
    assert objectives[0].weight == pytest.approx(5.0)

    optimizer = ik.IKOptimizerLM(model, 1, objectives, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    initial = torch.zeros((1, model.joint_coord_count), dtype=torch.float32)
    initial[:, 6] = 1.0
    solved = torch.empty_like(initial)
    optimizer.step(wp.from_torch(initial), wp.from_torch(solved), iterations=20)
    residual = wp.to_torch(optimizer.linearize(wp.from_torch(solved))[0])
    assert torch.linalg.vector_norm(residual) < 1.0e-5


def test_cross_builder_preserves_target_axis_identity(reference_path: Path) -> None:
    source = cmu_humenv_smpl_skeleton()
    target = _g1_builder(reference_path, reverse_joints=True)
    builder = _trajectory_projection(source, target.trajectory_target)

    assert builder.target is target.trajectory_target
    assert builder.target.version == _G1_RETARGET_MATH_VERSION
    assert len(builder.construction_identity_sha256) == 64


@pytest.mark.parametrize("target_robot", ("g1", "smpl"))
def test_native_semantic_rotations_are_rest_exact_and_quaternion_sign_invariant(
    reference_path: Path, target_robot: str
) -> None:
    """Every source rest carrier reproduces target rest orientation independent of quaternion sign."""
    if target_robot == "g1":
        source = lafan_g1_29dof_skeleton()
        frame_target = _g1_builder(reference_path)
    else:
        source = cmu_humenv_smpl_skeleton()
        frame_target = _smpl_builder(reference_path)
    target = frame_target.trajectory_target
    projection = _trajectory_projection(source, target)
    root_position = torch.zeros(2, 3)
    local_rotation = torch.zeros(2, source.num_bodies, 4)
    local_rotation[..., 3] = 1.0
    targets = projection.generate_targets(root_position, local_rotation)

    expected_rotation = target.rotation_rest_xyzw.to(dtype=torch.float32)[:, None].expand_as(
        targets.source_landmark_rotation_xyzw
    )
    rest_alignment = torch.sum(targets.source_landmark_rotation_xyzw * expected_rotation, dim=-1).abs()
    torch.testing.assert_close(rest_alignment, torch.ones_like(rest_alignment), atol=1.0e-6, rtol=0.0)

    negated_source = replace(
        source,
        rest_rotation_wxyz=tuple(tuple(-component for component in rotation) for rotation in source.rest_rotation_wxyz),
    )
    negated_targets = _trajectory_projection(negated_source, target).generate_targets(root_position, local_rotation)
    sign_alignment = torch.sum(
        targets.source_landmark_rotation_xyzw * negated_targets.source_landmark_rotation_xyzw, dim=-1
    ).abs()
    torch.testing.assert_close(sign_alignment, torch.ones_like(sign_alignment), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        negated_targets.source_landmark_position_m, targets.source_landmark_position_m, atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        negated_targets.source_direction_point_position_m,
        targets.source_direction_point_position_m,
        atol=1.0e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("target_robot", ("g1", "smpl"))
def test_trajectory_target_factors_out_scene_root_pose(reference_path: Path, target_robot: str) -> None:
    """Arbitrary G1 and SMPL scene roots leave intrinsic target geometry and projection unchanged."""
    if target_robot == "g1":
        skeleton = lafan_g1_29dof_skeleton()
        intrinsic_target = _g1_builder(reference_path).trajectory_target
        reference = _ReferenceKinematics(skeleton, reference_path)
    else:
        skeleton = cmu_humenv_smpl_skeleton()
        intrinsic_target = _smpl_builder(reference_path).trajectory_target
        reference = _ReferenceKinematics(skeleton, reference_path, smpl=True)

    scene_rotation = quat_from_rotation_vector(torch.tensor((0.4, -0.2, 0.7)))
    scene_position = torch.tensor((1.5, -2.0, 0.8))
    body_pose = torch.tensor(reference.default_body_q)
    body_pose[:, :3] = scene_position + quat_apply(scene_rotation.expand(body_pose.shape[0], 4), body_pose[:, :3])
    body_pose[:, 3:7] = quat_mul(scene_rotation.expand(body_pose.shape[0], 4), body_pose[:, 3:7])
    reference.default_body_q = body_pose.numpy()
    joint_pose = torch.tensor(reference.default_joint_q)
    joint_pose[:3] = scene_position + quat_apply(scene_rotation, joint_pose[:3])
    joint_pose[3:7] = quat_mul(scene_rotation, joint_pose[3:7])
    reference.default_joint_q = joint_pose.numpy()
    if target_robot == "g1":
        scene_target = _G1FrameTarget(
            kinematic_tree=_kinematic_tree(skeleton),
            pose_coordinate_identity_sha256=skeleton.identity_sha256,
            kinematics=reference,
            reference_mechanics_sha256=reference.mechanics_identity_sha256,
            live_joint_names=skeleton.joint_names,
            live_body_names=skeleton.body_names,
            contact_patches=_G1_CONTACT_PATCHES,
        ).trajectory_target
    else:
        scene_target = _SmplFrameTarget(
            kinematics=reference,
            reference_mechanics_sha256=reference.mechanics_identity_sha256,
            live_joint_names=SMPL_LIVE_JOINT_NAMES,
            live_body_names=skeleton.body_names,
            contact_patches=_SMPL_CONTACT_PATCHES,
        ).trajectory_target

    assert scene_target.root_body_index == intrinsic_target.root_body_index
    assert scene_target.root_cluster_rows == intrinsic_target.root_cluster_rows
    torch.testing.assert_close(scene_target.root_cluster_offset_m, intrinsic_target.root_cluster_offset_m)
    torch.testing.assert_close(scene_target.segment_lengths_m, intrinsic_target.segment_lengths_m)
    torch.testing.assert_close(scene_target.direction_lengths_m, intrinsic_target.direction_lengths_m)
    torch.testing.assert_close(scene_target.direction_point_body_m, intrinsic_target.direction_point_body_m)
    torch.testing.assert_close(
        scene_target.rotation_rest_xyzw, intrinsic_target.rotation_rest_xyzw, atol=2.0e-7, rtol=0.0
    )
    torch.testing.assert_close(scene_target.contact_normal_body, intrinsic_target.contact_normal_body)
    assert scene_target.support_patch_offsets == intrinsic_target.support_patch_offsets
    for start, stop in zip(
        scene_target.support_patch_offsets[:-1], scene_target.support_patch_offsets[1:], strict=True
    ):
        torch.testing.assert_close(
            scene_target.support_point_root_m[start:stop].amin(dim=0),
            intrinsic_target.support_point_root_m[start:stop].amin(dim=0),
        )
        torch.testing.assert_close(
            scene_target.support_point_root_m[start:stop].amax(dim=0),
            intrinsic_target.support_point_root_m[start:stop].amax(dim=0),
        )

    root_position = torch.tensor(((0.0, 0.0, 0.8), (0.2, -0.1, 0.9)))
    local_rotation = torch.zeros((2, skeleton.num_bodies, 4))
    local_rotation[..., 3] = 1.0
    intrinsic = _trajectory_projection(skeleton, intrinsic_target).generate_targets(root_position, local_rotation)
    scene = _trajectory_projection(skeleton, scene_target).generate_targets(root_position, local_rotation)
    torch.testing.assert_close(scene.source_landmark_position_m, intrinsic.source_landmark_position_m)
    torch.testing.assert_close(scene.source_landmark_rotation_xyzw, intrinsic.source_landmark_rotation_xyzw)
    torch.testing.assert_close(scene.source_direction_point_position_m, intrinsic.source_direction_point_position_m)
    torch.testing.assert_close(scene.initial_joint_q, intrinsic.initial_joint_q)


@pytest.mark.parametrize(("source_name", "target_robot"), (("cmu", "g1"), ("cmu", "smpl"), ("lafan", "g1")))
def test_coordinate_projection_rotation_calibration_and_keypoints_are_rigid_world_equivariant(
    reference_path: Path, source_name: str, target_robot: str
) -> None:
    """Every coordinate projection follows the anatomical root gauge and rigid world equivariance."""
    source = cmu_humenv_smpl_skeleton() if source_name == "cmu" else lafan_g1_29dof_skeleton()
    frame_target = _g1_builder(reference_path) if target_robot == "g1" else _smpl_builder(reference_path)
    target = frame_target.trajectory_target
    projection = _trajectory_projection(source, target)

    frame_count = 3
    source_root_index = source.parent_indices.index(-1)
    root_position = torch.tensor(((0.2, -0.4, 0.8), (0.5, 0.1, 1.0), (-0.3, 0.2, 0.7)))
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    local_rotation[:, source_root_index].copy_(
        quat_from_rotation_vector(torch.tensor(((0.2, -0.1, 0.4), (-0.3, 0.5, 0.1), (0.6, 0.2, -0.4))))
    )
    local_rotation[:, 1].copy_(
        quat_from_rotation_vector(torch.tensor(((0.0, 0.2, -0.1), (0.1, -0.3, 0.2), (-0.2, 0.1, 0.3))))
    )
    baseline = projection.generate_targets(root_position, local_rotation)

    source_rest_translation = torch.tensor(source.rest_translation_m, dtype=torch.float32)
    source_rest_rotation = convert_quat(torch.tensor(source.rest_rotation_wxyz, dtype=torch.float32), to="xyzw")
    source_position, source_world_rotation = kinematic_pose_forward(
        source_rest_translation,
        source_rest_rotation,
        local_rotation,
        root_position,
        source.parent_indices,
    )
    source_anatomy = quat_from_matrix(
        kinematic_root_basis(source_position, *projection._source_anatomical_basis_body_indices)
    )
    expected_root = quat_mul(source_anatomy, quat_conjugate(target.anatomical_rotation_xyzw).expand(frame_count, 4))
    if source.landmark_rotation_policy == "anatomical_root":
        expected_rotation = expected_root.unsqueeze(0)
    else:
        mapped_rotation = quat_mul(
            source_world_rotation.index_select(1, projection._source_rotation_body_indices),
            projection._source_to_target_rotation_xyzw.unsqueeze(0).expand(frame_count, -1, -1),
        )
        mapped_rotation /= torch.linalg.vector_norm(mapped_rotation, dim=-1, keepdim=True)
        gauge = quat_mul(expected_root, quat_conjugate(mapped_rotation[:, 0]))
        expected_rotation = quat_mul(gauge.unsqueeze(1).expand_as(mapped_rotation), mapped_rotation)
        expected_rotation = expected_rotation.transpose(0, 1)
    expected_rotation /= torch.linalg.vector_norm(expected_rotation, dim=-1, keepdim=True)
    baseline_alignment = torch.sum(baseline.source_landmark_rotation_xyzw * expected_rotation, dim=-1).abs()
    torch.testing.assert_close(baseline_alignment, torch.ones_like(baseline_alignment), atol=1.0e-6, rtol=0.0)
    root_alignment = torch.sum(baseline.initial_joint_q[:, 3:7] * expected_rotation[0], dim=-1).abs()
    torch.testing.assert_close(root_alignment, torch.ones_like(root_alignment), atol=1.0e-6, rtol=0.0)

    delta = quat_from_rotation_vector(torch.tensor((0.2, 0.3, -0.5), dtype=torch.float32))
    translation = torch.tensor((1.1, -0.7, 0.3), dtype=torch.float32)
    delta_rows = delta.expand(frame_count, 4)
    transformed_root_position = translation + quat_apply(delta_rows, root_position)
    transformed_local_rotation = local_rotation.clone()
    transformed_local_rotation[:, source_root_index].copy_(
        quat_mul(delta_rows, transformed_local_rotation[:, source_root_index])
    )
    transformed = projection.generate_targets(transformed_root_position, transformed_local_rotation)
    expected_transformed_rotation = quat_mul(delta[None, None].expand_as(expected_rotation), expected_rotation)
    transformed_alignment = torch.sum(
        transformed.source_landmark_rotation_xyzw * expected_transformed_rotation, dim=-1
    ).abs()
    torch.testing.assert_close(transformed_alignment, torch.ones_like(transformed_alignment), atol=1.0e-6, rtol=0.0)
    transformed_root_alignment = torch.sum(
        transformed.initial_joint_q[:, 3:7] * expected_transformed_rotation[0], dim=-1
    ).abs()
    torch.testing.assert_close(
        transformed_root_alignment,
        torch.ones_like(transformed_root_alignment),
        atol=1.0e-6,
        rtol=0.0,
    )

    def rigid_transform(points: torch.Tensor) -> torch.Tensor:
        return translation + quat_apply(delta.expand(points.numel() // 3, 4), points.reshape(-1, 3)).view_as(points)

    torch.testing.assert_close(
        transformed.source_landmark_position_m,
        rigid_transform(baseline.source_landmark_position_m),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed.source_direction_point_position_m,
        rigid_transform(baseline.source_direction_point_position_m),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed.source_contact_probe_position_m,
        rigid_transform(baseline.source_contact_probe_position_m),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed.target_support_position_m,
        rigid_transform(baseline.target_support_position_m),
        atol=2.0e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("target_robot", ("g1", "smpl"))
def test_supported_coordinate_projections_respect_source_direction_laws(
    reference_path: Path, target_robot: str
) -> None:
    """Semantic edges use target morphology while each distal direction follows its declared law."""
    source = cmu_humenv_smpl_skeleton()
    frame_target = _g1_builder(reference_path) if target_robot == "g1" else _smpl_builder(reference_path)
    target = frame_target.trajectory_target
    root_position = torch.tensor(((0.0, 0.0, 0.8), (0.1, -0.2, 0.9), (0.3, 0.2, 1.0)))
    local_rotation = torch.zeros((3, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0

    targets = _trajectory_projection(source, target).generate_targets(root_position, local_rotation)

    root = targets.source_landmark_position_m[0]
    root_rotation = targets.source_landmark_rotation_xyzw[0]
    for index, row in enumerate(target.root_cluster_rows):
        relative_root = quat_apply(quat_conjugate(root_rotation), targets.source_landmark_position_m[row] - root)
        torch.testing.assert_close(
            relative_root, target.root_cluster_offset_m[index].expand_as(relative_root), atol=1.0e-6, rtol=0.0
        )

    cluster_rows = set(target.root_cluster_rows)
    for row, parent in enumerate(target.parent_rows[1:], start=1):
        if row in cluster_rows:
            continue
        edge = targets.source_landmark_position_m[row] - targets.source_landmark_position_m[parent]
        expected_length = target.segment_lengths_m[row].expand(edge.shape[0])
        torch.testing.assert_close(torch.linalg.vector_norm(edge, dim=-1), expected_length, atol=1.0e-6, rtol=0.0)

    source_rest_translation = torch.tensor(source.rest_translation_m, dtype=torch.float32)
    source_rest_rotation = convert_quat(torch.tensor(source.rest_rotation_wxyz, dtype=torch.float32), to="xyzw")
    source_position, _ = kinematic_pose_forward(
        source_rest_translation,
        source_rest_rotation,
        local_rotation,
        root_position,
        source.parent_indices,
    )
    source_landmarks = {item.name: item for item in source.landmarks}
    source_from_indices = torch.tensor(
        tuple(
            source.body_names.index(source_landmarks[point.source_from_role].position_body_name)
            for point in target.direction_points
        )
    )
    source_to_indices = torch.tensor(
        tuple(
            source.body_names.index(source_landmarks[point.source_to_role].position_body_name)
            for point in target.direction_points
        )
    )
    source_direction = (
        source_position.index_select(1, source_to_indices) - source_position.index_select(1, source_from_indices)
    ).transpose(0, 1)
    source_direction /= torch.linalg.vector_norm(source_direction, dim=-1, keepdim=True)
    distal = targets.source_direction_point_position_m - targets.source_landmark_position_m.index_select(
        0, target.direction_rows
    )
    expected_distal = target.direction_lengths_m[:, None, None] * source_direction
    between_rows = torch.tensor(
        tuple(
            row
            for row, point in enumerate(target.direction_points)
            if point.source_direction_law == "between_positions"
        )
    )
    torch.testing.assert_close(distal[between_rows], expected_distal[between_rows], atol=1.0e-6, rtol=0.0)
    expected_distal_length = target.direction_lengths_m[:, None].expand(-1, distal.shape[1])
    torch.testing.assert_close(torch.linalg.vector_norm(distal, dim=-1), expected_distal_length, atol=1.0e-6, rtol=0.0)


def test_cross_projection_root_translation_is_ground_relative(reference_path: Path) -> None:
    """Calibrated height uses the raw source floor while horizontal world gauge remains unchanged."""
    source = cmu_humenv_smpl_skeleton()
    projection = _trajectory_projection(source, _g1_builder(reference_path).trajectory_target)
    assert projection._root_translation_scale != 1.0
    root_position = torch.tensor(((4.0, -3.0, 0.8), (4.2, -2.9, 0.9), (4.5, -2.7, 1.0)))
    root_offset = torch.tensor((7.0, -11.0, 2.5))
    local_rotation = torch.zeros((3, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0

    targets = projection.generate_targets(root_position, local_rotation)
    shifted_targets = projection.generate_targets(root_position + root_offset, local_rotation)
    expected_root = root_position[:1] + projection._root_translation_scale * (root_position - root_position[:1])
    source_floor_m = torch.amin(targets.source_contact_probe_position_m[..., 2])
    expected_root[:, 2] = projection._root_translation_scale * (root_position[:, 2] - source_floor_m)
    torch.testing.assert_close(targets.initial_joint_q[:, :3], expected_root, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        shifted_targets.source_contact_probe_position_m - targets.source_contact_probe_position_m,
        root_offset.expand_as(targets.source_contact_probe_position_m),
        atol=4.0e-6,
        rtol=0.0,
    )
    calibrated_offset = root_offset.clone()
    calibrated_offset[2] = 0.0
    for actual, shifted in (
        (targets.initial_joint_q[:, :3], shifted_targets.initial_joint_q[:, :3]),
        (targets.source_landmark_position_m, shifted_targets.source_landmark_position_m),
        (targets.source_direction_point_position_m, shifted_targets.source_direction_point_position_m),
        (targets.target_support_position_m, shifted_targets.target_support_position_m),
    ):
        torch.testing.assert_close(shifted - actual, calibrated_offset.expand_as(actual), atol=4.0e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to compare construction devices.")
@pytest.mark.parametrize("target_robot", ("g1", "smpl"))
def test_projection_construction_identity_is_device_independent(reference_path: Path, target_robot: str) -> None:
    """Immutable source and target facts must produce one identity and calibration on CPU and CUDA."""
    if target_robot == "g1":
        target_builder = _g1_builder
    else:
        target_builder = _smpl_builder
    source = cmu_humenv_smpl_skeleton()

    cpu_target = target_builder(reference_path, device="cpu").trajectory_target
    cuda_target = target_builder(reference_path, device="cuda:0").trajectory_target
    cpu_projection = _trajectory_projection(source, cpu_target)
    cuda_projection = _trajectory_projection(source, cuda_target)

    assert cpu_target.frame_target.construction_identity_sha256 == cuda_target.frame_target.construction_identity_sha256
    assert cpu_target.construction_identity_sha256 == cuda_target.construction_identity_sha256
    assert cpu_projection.construction_identity_sha256 == cuda_projection.construction_identity_sha256
    assert cpu_target.segment_length_values_m == cuda_target.segment_length_values_m
    assert cpu_target.root_cluster_rows == cuda_target.root_cluster_rows
    assert cpu_projection._root_translation_scale == cuda_projection._root_translation_scale
    torch.testing.assert_close(
        cpu_target.root_cluster_offset_m, cuda_target.root_cluster_offset_m.cpu(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        cpu_target.direction_lengths_m, cuda_target.direction_lengths_m.cpu(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        cpu_target.direction_point_body_m, cuda_target.direction_point_body_m.cpu(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(cpu_target.rotation_rest_xyzw, cuda_target.rotation_rest_xyzw.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        cpu_projection._source_to_target_rotation_xyzw,
        cuda_projection._source_to_target_rotation_xyzw.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cpu_target.support_point_body_m, cuda_target.support_point_body_m.cpu(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(cpu_target.support_body_indices, cuda_target.support_body_indices.cpu())
    assert cpu_target.segment_lengths_m.device.type == "cpu"
    assert cuda_target.segment_lengths_m.device.type == "cuda"
    assert cpu_projection._source_rest_translation_m.device.type == "cpu"
    assert cuda_projection._source_rest_translation_m.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise TF32 matmul policy.")
@pytest.mark.parametrize("target_robot", ("g1", "smpl"), ids=("g1_from_cmu", "smpl_from_lafan"))
def test_cross_projection_stays_finite_with_training_tf32(reference_path: Path, target_robot: str) -> None:
    """Training's TF32 policy must preserve finite unit semantic targets for both cross-compositions."""
    if target_robot == "g1":
        source = cmu_humenv_smpl_skeleton()
        target = _g1_builder(reference_path, device="cuda:0")
        version = _G1_RETARGET_MATH_VERSION
    else:
        source = lafan_g1_29dof_skeleton()
        target = _smpl_builder(reference_path, device="cuda:0")
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
        projection = _trajectory_projection(source, target.trajectory_target)
        targets = projection.generate_targets(root_position, quat_from_rotation_vector(rotation_vector))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    changed_projection = _trajectory_projection(
        source, replace(target.trajectory_target, version=f"{version}_identity_probe")
    )
    assert projection.construction_identity_sha256 != changed_projection.construction_identity_sha256
    assert torch.isfinite(targets.source_landmark_position_m).all()
    assert torch.isfinite(targets.source_landmark_rotation_xyzw).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(targets.source_landmark_rotation_xyzw, dim=-1),
        torch.ones(targets.source_landmark_rotation_xyzw.shape[:-1], device="cuda"),
        atol=5.0e-6,
        rtol=5.0e-6,
    )


def _infer_source_probe_evidence(
    source: torch.Tensor,
    channel_offsets: torch.Tensor,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    source_plane_height_m: torch.Tensor,
    *,
    enter_height_m: float = 0.01,
    exit_height_m: float = 0.02,
    enter_speed_mps: float = 0.2,
    exit_speed_mps: float = 0.4,
    persistence_seconds: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer hysteretic activity and strict planted evidence for each source probe."""
    wp.init()
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_contact_probe_evidence_infer,
    )

    probe_active = torch.empty((source.shape[1], source.shape[0]), dtype=torch.uint8)
    probe_stable = torch.empty_like(probe_active)
    wp.launch(
        _motion_contact_probe_evidence_infer,
        dim=(clip_offsets.numel() - 1, source.shape[0]),
        inputs=[
            wp.from_torch(source),
            wp.from_torch(channel_offsets),
            channel_offsets.numel() - 1,
            wp.from_torch(clip_offsets),
            wp.from_torch(step_seconds),
            wp.from_torch(source_plane_height_m),
            enter_height_m,
            exit_height_m,
            enter_speed_mps,
            exit_speed_mps,
            persistence_seconds,
        ],
        outputs=[wp.from_torch(probe_active), wp.from_torch(probe_stable)],
        device="cpu",
    )
    return probe_active, probe_stable


def _aggregate_source_contact(
    source: torch.Tensor,
    probe_active: torch.Tensor,
    probe_stable: torch.Tensor,
    channel_offsets: torch.Tensor,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    confidence_window_seconds: float,
    *,
    enter_speed_mps: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate centered confidence and strict backward source edges."""
    wp.init()
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_contact_channel_aggregate,
    )

    channel_count = channel_offsets.numel() - 1
    confidence = torch.empty((probe_active.shape[0], channel_count), dtype=torch.float32)
    stable = torch.empty((probe_active.shape[0], channel_count), dtype=torch.uint8)
    edge_stable = torch.empty_like(stable)
    wp.launch(
        _motion_contact_channel_aggregate,
        dim=(clip_offsets.numel() - 1, channel_count),
        inputs=[
            wp.from_torch(source),
            wp.from_torch(probe_active),
            wp.from_torch(probe_stable),
            wp.from_torch(channel_offsets),
            wp.from_torch(clip_offsets),
            wp.from_torch(step_seconds),
            confidence_window_seconds,
            enter_speed_mps,
        ],
        outputs=[wp.from_torch(confidence), wp.from_torch(stable), wp.from_torch(edge_stable)],
        device="cpu",
    )
    return confidence, stable, edge_stable


def test_motion_contact_channel_proximity_uses_lowest_probe() -> None:
    """One low point establishes foot proximity while every stationary probe witnesses rigidity."""
    source = torch.zeros((2, 5, 3), dtype=torch.float32)
    source[1, :, 2] = 0.1
    channel_offsets = torch.tensor((0, 2), dtype=torch.int32)
    clip_offsets = torch.tensor((0, 5), dtype=torch.int32)
    step_seconds = torch.tensor((0.1,), dtype=torch.float32)
    active, stable = _infer_source_probe_evidence(source, channel_offsets, clip_offsets, step_seconds, torch.zeros(1))
    confidence, channel_stable, edge_stable = _aggregate_source_contact(
        source, active, stable, channel_offsets, clip_offsets, step_seconds, 0.0
    )

    torch.testing.assert_close(active, torch.ones_like(active))
    torch.testing.assert_close(stable, torch.ones_like(stable))
    torch.testing.assert_close(confidence[:, 0], torch.ones(5))
    torch.testing.assert_close(channel_stable[:, 0], torch.ones(5, dtype=torch.uint8))
    torch.testing.assert_close(edge_stable[:, 0], torch.tensor((0, 1, 1, 1, 1), dtype=torch.uint8))


def test_motion_contact_infers_each_probe_speed_before_channel_aggregation() -> None:
    """A fast landmark remains inactive independently after another point establishes proximity."""
    source = torch.zeros((2, 5, 3), dtype=torch.float32)
    source[1, :, 0] = torch.arange(5, dtype=torch.float32) * 0.1
    source[1, :, 2] = 0.1
    channel_offsets = torch.tensor((0, 2), dtype=torch.int32)
    clip_offsets = torch.tensor((0, 5), dtype=torch.int32)
    step_seconds = torch.tensor((0.1,), dtype=torch.float32)
    active, stable = _infer_source_probe_evidence(source, channel_offsets, clip_offsets, step_seconds, torch.zeros(1))
    confidence, channel_stable, edge_stable = _aggregate_source_contact(
        source, active, stable, channel_offsets, clip_offsets, step_seconds, 0.0
    )

    torch.testing.assert_close(active[:, 0], torch.ones(5, dtype=torch.uint8))
    torch.testing.assert_close(active[:, 1], torch.zeros(5, dtype=torch.uint8))
    torch.testing.assert_close(stable[:, 0], torch.ones(5, dtype=torch.uint8))
    torch.testing.assert_close(stable[:, 1], torch.zeros(5, dtype=torch.uint8))
    torch.testing.assert_close(confidence[:, 0], torch.full((5,), 0.5))
    torch.testing.assert_close(channel_stable[:, 0], torch.zeros(5, dtype=torch.uint8))
    torch.testing.assert_close(edge_stable[:, 0], torch.zeros(5, dtype=torch.uint8))


def test_motion_contact_hysteretic_activity_does_not_claim_strict_stability() -> None:
    """Exit-band evidence remains a soft contact while hard planted evidence is false."""
    source = torch.zeros((2, 6, 3), dtype=torch.float32)
    source[:, 3:, 2] = 0.04
    channel_offsets = torch.tensor((0, 2), dtype=torch.int32)
    clip_offsets = torch.tensor((0, 6), dtype=torch.int32)
    step_seconds = torch.tensor((0.1,), dtype=torch.float32)
    active, stable = _infer_source_probe_evidence(
        source,
        channel_offsets,
        clip_offsets,
        step_seconds,
        torch.zeros(1),
        enter_height_m=0.03,
        exit_height_m=0.06,
        enter_speed_mps=0.15,
        exit_speed_mps=0.3,
    )

    torch.testing.assert_close(active, torch.ones_like(active))
    torch.testing.assert_close(stable[:2], torch.ones_like(stable[:2]))
    torch.testing.assert_close(stable[3:], torch.zeros_like(stable[3:]))


def test_motion_contact_stability_requires_complete_probe_time_support() -> None:
    """A planted channel requires every active strict probe-time sample in its centered window."""
    source = torch.zeros((2, 3, 3), dtype=torch.float32)
    channel_offsets = torch.tensor((0, 2), dtype=torch.int32)
    clip_offsets = torch.tensor((0, 3), dtype=torch.int32)
    step_seconds = torch.tensor((0.1,), dtype=torch.float32)
    active = torch.tensor(((1, 0), (1, 0), (1, 0)), dtype=torch.uint8)
    confidence, stable, edge_stable = _aggregate_source_contact(
        source, active, active, channel_offsets, clip_offsets, step_seconds, 0.0
    )

    torch.testing.assert_close(confidence[:, 0], torch.full((3,), 0.5))
    torch.testing.assert_close(stable[:, 0], torch.zeros(3, dtype=torch.uint8))
    torch.testing.assert_close(edge_stable[:, 0], torch.zeros(3, dtype=torch.uint8))

    partial = torch.tensor(((1, 1, 0),), dtype=torch.uint8)
    partial_confidence, partial_stable, _ = _aggregate_source_contact(
        torch.zeros((3, 1, 3)),
        partial,
        partial,
        torch.tensor((0, 3), dtype=torch.int32),
        torch.tensor((0, 1), dtype=torch.int32),
        step_seconds,
        0.0,
    )
    torch.testing.assert_close(partial_confidence[:, 0], torch.tensor((2.0 / 3.0,)))
    torch.testing.assert_close(partial_stable[:, 0], torch.zeros(1, dtype=torch.uint8))

    complete = torch.ones((1, 3), dtype=torch.uint8)
    complete_confidence, complete_stable, complete_edge = _aggregate_source_contact(
        torch.zeros((3, 1, 3)),
        complete,
        complete,
        torch.tensor((0, 3), dtype=torch.int32),
        torch.tensor((0, 1), dtype=torch.int32),
        step_seconds,
        0.0,
    )
    torch.testing.assert_close(complete_confidence[:, 0], torch.ones(1))
    torch.testing.assert_close(complete_stable[:, 0], torch.ones(1, dtype=torch.uint8))
    torch.testing.assert_close(complete_edge[:, 0], torch.zeros(1, dtype=torch.uint8))


def test_motion_contact_confidence_window_is_clip_bounded() -> None:
    """A centered contact window never consumes probe evidence from an adjacent clip."""
    active = torch.tensor(((1,), (1,), (0,), (0,), (1,), (1,)), dtype=torch.uint8)
    confidence, stable, edge_stable = _aggregate_source_contact(
        torch.zeros((1, 6, 3)),
        active,
        active,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((0, 3, 6), dtype=torch.int32),
        torch.tensor((0.1, 0.1), dtype=torch.float32),
        0.4,
    )

    torch.testing.assert_close(confidence[:, 0], torch.full((6,), 2.0 / 3.0))
    torch.testing.assert_close(stable[:, 0], torch.zeros(6, dtype=torch.uint8))
    torch.testing.assert_close(edge_stable[:, 0], torch.zeros(6, dtype=torch.uint8))


def test_motion_contact_confidence_window_uses_physical_time() -> None:
    """The duration chooses nearest odd samples and resolves even-count ties upward."""
    active = torch.zeros((11, 1), dtype=torch.uint8)
    active[5, 0] = 1
    source = torch.zeros((1, 11, 3))
    channel_offsets = torch.tensor((0, 1), dtype=torch.int32)
    clip_offsets = torch.tensor((0, 11), dtype=torch.int32)
    confidence_60hz, _, _ = _aggregate_source_contact(
        source, active, active, channel_offsets, clip_offsets, torch.tensor((1.0 / 60.0,)), 5.0 / 60.0
    )
    confidence_30hz, _, _ = _aggregate_source_contact(
        source, active, active, channel_offsets, clip_offsets, torch.tensor((1.0 / 30.0,)), 5.0 / 60.0
    )
    confidence_120hz, _, _ = _aggregate_source_contact(
        source, active, active, channel_offsets, clip_offsets, torch.tensor((1.0 / 120.0,)), 5.0 / 60.0
    )

    torch.testing.assert_close(confidence_60hz[5, 0], torch.tensor(1.0 / 5.0))
    torch.testing.assert_close(confidence_30hz[5, 0], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(confidence_120hz[5, 0], torch.tensor(1.0 / 11.0))


def test_motion_contact_inference_uses_every_probe_and_keeps_clips_independent() -> None:
    """Probe evidence stays clip-local before channel confidence and planted-state aggregation."""
    wp.init()
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_contact_source_plane

    frame_count = 10
    source = torch.zeros((4, frame_count, 3), dtype=torch.float32)
    source[..., 2].fill_(-0.25)
    source[1, :5, 0] = torch.arange(5, dtype=torch.float32) * 0.1
    source[3, 5:, 0] = torch.arange(5, dtype=torch.float32) * 0.1
    channel_offsets = torch.tensor((0, 2, 4), dtype=torch.int32)
    clip_offsets = torch.tensor((0, 5, 10), dtype=torch.int32)
    step_seconds = torch.tensor((0.1, 0.1), dtype=torch.float32)
    source_plane_height_m = torch.empty(2, dtype=torch.float32)
    wp.launch(
        _motion_contact_source_plane,
        dim=2,
        inputs=[wp.from_torch(source), wp.from_torch(clip_offsets), 4],
        outputs=[wp.from_torch(source_plane_height_m)],
        device="cpu",
    )
    active, stable = _infer_source_probe_evidence(
        source, channel_offsets, clip_offsets, step_seconds, source_plane_height_m
    )
    confidence, channel_stable, edge_stable = _aggregate_source_contact(
        source, active, stable, channel_offsets, clip_offsets, step_seconds, 0.0
    )

    expected_confidence = torch.tensor((*([(0.5, 1.0)] * 5), *([(1.0, 0.5)] * 5)))
    expected_stable = torch.tensor((*([(0, 1)] * 5), *([(1, 0)] * 5)), dtype=torch.uint8)
    torch.testing.assert_close(confidence, expected_confidence)
    torch.testing.assert_close(channel_stable, expected_stable)
    torch.testing.assert_close(edge_stable[clip_offsets[:-1].long()], torch.zeros((2, 2), dtype=torch.uint8))
    torch.testing.assert_close(source_plane_height_m, torch.full((2,), -0.25))


def test_motion_contact_pre_enter_hover_stays_inactive() -> None:
    """Exit-band hover cannot become soft or hard contact before a persistent strict enter window."""
    frame_count = 6
    source = torch.zeros((2, frame_count, 3), dtype=torch.float32)
    source[:, :3, 2] = 0.015
    channel_offsets = torch.tensor((0, 2), dtype=torch.int32)
    clip_offsets = torch.tensor((0, frame_count), dtype=torch.int32)
    step_seconds = torch.tensor((1.0,), dtype=torch.float32)
    active, stable = _infer_source_probe_evidence(
        source,
        channel_offsets,
        clip_offsets,
        step_seconds,
        torch.zeros(1),
        enter_height_m=0.01,
        exit_height_m=0.02,
        enter_speed_mps=0.1,
        exit_speed_mps=0.2,
        persistence_seconds=2.0,
    )
    confidence, channel_stable, edge_stable = _aggregate_source_contact(
        source, active, stable, channel_offsets, clip_offsets, step_seconds, 0.0
    )

    expected = torch.tensor((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    torch.testing.assert_close(confidence[:, 0], expected)
    torch.testing.assert_close(channel_stable[:, 0], expected.to(torch.uint8))
    torch.testing.assert_close(edge_stable[:, 0], torch.tensor((0, 0, 0, 0, 1, 1), dtype=torch.uint8))


def test_motion_contact_edge_requires_endpoint_states_and_every_probe_speed() -> None:
    """Each strict edge independently requires two stable states and bounded backward source speed."""
    source = torch.zeros((1, 7, 3), dtype=torch.float32)
    source[0, :, 0] = torch.tensor((0.0, 0.01, 0.02, 0.031, 0.032, 0.0, 0.01))
    active = torch.tensor(((1,), (1,), (1,), (1,), (1,), (0,), (1,)), dtype=torch.uint8)
    stable = torch.ones_like(active)
    confidence, channel_stable, edge_stable = _aggregate_source_contact(
        source,
        active,
        stable,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((0, 7), dtype=torch.int32),
        torch.tensor((1.0,), dtype=torch.float32),
        0.0,
    )

    torch.testing.assert_close(confidence[:, 0], active[:, 0].float())
    torch.testing.assert_close(channel_stable[:, 0], active[:, 0])
    torch.testing.assert_close(edge_stable[:, 0], torch.tensor((0, 1, 1, 1, 1, 0, 0), dtype=torch.uint8))

    fast_source = torch.zeros((2, 3, 3), dtype=torch.float32)
    fast_source[0, :, 0] = torch.tensor((0.0, 0.02, 0.02))
    ones = torch.ones((3, 2), dtype=torch.uint8)
    _, fast_stable, fast_edges = _aggregate_source_contact(
        fast_source,
        ones,
        ones,
        torch.tensor((0, 2), dtype=torch.int32),
        torch.tensor((0, 3), dtype=torch.int32),
        torch.tensor((0.1,), dtype=torch.float32),
        0.0,
    )
    torch.testing.assert_close(fast_stable[:, 0], torch.ones(3, dtype=torch.uint8))
    torch.testing.assert_close(fast_edges[:, 0], torch.tensor((0, 0, 1), dtype=torch.uint8))


def _contact_quality(
    position_x: torch.Tensor,
    step_seconds: float,
    *,
    points_per_patch: int = 3,
    confidence: torch.Tensor | None = None,
    stable: torch.Tensor | None = None,
    edge_stable: torch.Tensor | None = None,
    position_z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Measure one flat-support contact clip without source-fidelity scaffolding."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _trajectory_quality_clips

    wp.init()

    frame_count = position_x.shape[0]
    body_q = torch.zeros((frame_count, 1, 7), dtype=torch.float32)
    body_q[..., 6] = 1.0
    body_q[:, 0, 0].copy_(position_x)
    if position_z is not None:
        body_q[:, 0, 2].copy_(position_z)
    patch = torch.tensor(((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.0)))
    repeats = math.ceil(points_per_patch / 3)
    support_points = patch.repeat(repeats, 1)[:points_per_patch].contiguous()
    support_bodies = torch.zeros(points_per_patch, dtype=torch.int64)
    support_channels = torch.zeros(points_per_patch, dtype=torch.int64)
    confidence = torch.ones(frame_count) if confidence is None else confidence
    stable = torch.ones(frame_count, dtype=torch.uint8) if stable is None else stable
    if edge_stable is None:
        edge_stable = torch.zeros_like(stable)
        edge_stable[1:] = stable[1:] & stable[:-1]
    obstacle_pose = torch.zeros((frame_count, 7), dtype=torch.float32)
    obstacle_pose[:, 6] = 1.0
    quality_by_frame = torch.zeros((frame_count, len(_TRAJECTORY_METRIC_NAMES)), dtype=torch.float32)
    quality = torch.empty((1, len(_TRAJECTORY_METRIC_NAMES)), dtype=torch.float32)
    wp.launch(
        _trajectory_quality_clips,
        dim=1,
        inputs=[
            wp.from_torch(quality_by_frame),
            wp.from_torch(body_q),
            wp.from_torch(support_bodies),
            wp.from_torch(support_points),
            wp.from_torch(support_channels),
            wp.from_torch(torch.zeros(1, dtype=torch.int64)),
            wp.from_torch(torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32)),
            wp.from_torch(confidence[:, None].contiguous()),
            wp.from_torch(stable[:, None].contiguous()),
            wp.from_torch(edge_stable[:, None].contiguous()),
            wp.from_torch(obstacle_pose),
            wp.from_torch(torch.tensor((0, frame_count), dtype=torch.int32)),
            wp.from_torch(torch.tensor((step_seconds,), dtype=torch.float32)),
            1,
            1,
            points_per_patch,
            _METRIC_SOURCE_REQUIRED_POSITION,
            _METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
            _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
            _METRIC_SOURCE_ROOT_ROTATION,
            _METRIC_SOURCE_ALL_POSITION,
            _METRIC_SOURCE_ALL_DISTAL_POSITION,
            _METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
            _METRIC_SOURCE_ALL_DISTAL_DIRECTION,
            _METRIC_SOURCE_NONROOT_ROTATION,
            _METRIC_CONTACT_GAP,
            _METRIC_CONTACT_TILT,
            _METRIC_CONTACT_SLIP_SPEED,
            _METRIC_CONTACT_CUMULATIVE_DRIFT,
            _METRIC_CONTACT_APPLICABLE,
            _METRIC_CONTACT_STABLE_COUNT,
            _METRIC_SOURCE_CONTACT_CONFIDENCE,
        ],
        outputs=[wp.from_torch(quality)],
        device="cpu",
    )
    return quality[0]


def test_motion_support_objective_is_signed_and_yaw_free() -> None:
    """Trajectory support residuals ignore yaw while distinguishing inverted normals."""
    from isaaclab_tasks.core.multi_task.kinematics import IKObjectiveSupportPatch

    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0)
    root_joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([root_joint])
    model = builder.finalize(device="cpu", requires_grad=True)
    tolerance = math.radians(15.0)
    support_pose = torch.zeros((4, 7), dtype=torch.float32)
    support_pose[:, 6] = 1.0
    objective = IKObjectiveSupportPatch(
        body=body,
        points_body=torch.tensor(((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.0))),
        normal_body=torch.tensor((0.0, 0.0, 1.0)),
        support_pose=support_pose,
        affects_dof=np.ones(model.joint_dof_count, dtype=np.uint8),
        gap_tolerance_m=0.03,
        tilt_tolerance_rad=tolerance,
        slip_speed_scale_mps=0.15,
    )
    optimizer = ik.IKOptimizerLM(model, 4, (objective,), jacobian_mode=ik.IKJacobianType.ANALYTIC)
    joint_q = torch.zeros((4, model.joint_coord_count), dtype=torch.float32)
    joint_q[:, 6] = 1.0
    yaw = 0.7
    joint_q[1, 5] = math.sin(0.5 * yaw)
    joint_q[1, 6] = math.cos(0.5 * yaw)
    tilt = 0.2
    joint_q[2, 3] = math.sin(0.5 * tilt)
    joint_q[2, 6] = math.cos(0.5 * tilt)
    joint_q[3, 3] = 1.0
    joint_q[3, 6] = 0.0
    residual = wp.to_torch(optimizer.linearize(wp.from_torch(joint_q))[0]).clone()

    torch.testing.assert_close(residual[0, :4], residual[1, :4], atol=1.0e-6, rtol=0.0)
    denominator = 2.0 * math.sin(0.5 * tolerance)
    torch.testing.assert_close(
        torch.linalg.vector_norm(residual[2, 1:4]),
        torch.tensor(2.0 * math.sin(0.5 * tilt) / denominator),
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(residual[3, 1:4]), torch.tensor(2.0 / denominator), atol=1.0e-5, rtol=0.0
    )


def test_motion_contact_certification_ignores_transition_outliers_but_keeps_stable_outliers() -> None:
    """Only exact stable states and stable-to-stable edges contribute physical contact maxima."""
    confidence = torch.tensor((0.5, 1.0, 1.0))
    transitional = _contact_quality(
        torch.tensor((100.0, 0.0, 0.0)),
        0.1,
        confidence=confidence,
        stable=torch.tensor((0, 1, 1), dtype=torch.uint8),
        position_z=torch.tensor((100.0, 0.0, 0.0)),
    )
    stable = _contact_quality(
        torch.tensor((100.0, 0.0, 0.0)),
        0.1,
        confidence=torch.ones(3),
        stable=torch.ones(3, dtype=torch.uint8),
        position_z=torch.tensor((100.0, 0.0, 0.0)),
    )

    torch.testing.assert_close(transitional[_METRIC_CONTACT_GAP], torch.tensor(0.0))
    torch.testing.assert_close(transitional[_METRIC_CONTACT_SLIP_SPEED], torch.tensor(0.0))
    torch.testing.assert_close(transitional[_METRIC_CONTACT_CUMULATIVE_DRIFT], torch.tensor(0.0))
    torch.testing.assert_close(stable[_METRIC_CONTACT_GAP], torch.tensor(100.0))
    assert stable[_METRIC_CONTACT_SLIP_SPEED] > 900.0
    assert stable[_METRIC_CONTACT_CUMULATIVE_DRIFT] > 90.0


def test_motion_contact_certification_uses_only_declared_planted_edges() -> None:
    """Only an explicit planted edge contributes slip and its connected-component drift."""
    quality = _contact_quality(
        torch.tensor((-2.0, 0.0, 0.01, 3.0)),
        0.1,
        confidence=torch.tensor((0.0, 1.0, 1.0, 0.0)),
        stable=torch.tensor((0, 1, 1, 0), dtype=torch.uint8),
    )
    torch.testing.assert_close(quality[_METRIC_CONTACT_SLIP_SPEED], torch.tensor(0.1), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(quality[_METRIC_CONTACT_CUMULATIVE_DRIFT], torch.tensor(0.01))

    disarmed = _contact_quality(
        torch.tensor((0.0, 1.0, 1.01)),
        0.1,
        stable=torch.ones(3, dtype=torch.uint8),
        edge_stable=torch.tensor((0, 0, 1), dtype=torch.uint8),
    )
    torch.testing.assert_close(disarmed[_METRIC_CONTACT_SLIP_SPEED], torch.tensor(0.1), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(disarmed[_METRIC_CONTACT_CUMULATIVE_DRIFT], torch.tensor(0.01))


def test_motion_contact_metrics_are_patch_cardinality_invariant() -> None:
    """Duplicating a support patch leaves gap, tilt, slip, and drift metrics unchanged."""
    position = torch.tensor((0.0, 0.01, 0.02), dtype=torch.float32)
    triangle = _contact_quality(position, 0.1, points_per_patch=3)
    duplicated = _contact_quality(position, 0.1, points_per_patch=6)
    rows = torch.tensor(
        (_METRIC_CONTACT_GAP, _METRIC_CONTACT_TILT, _METRIC_CONTACT_SLIP_SPEED, _METRIC_CONTACT_CUMULATIVE_DRIFT)
    )
    torch.testing.assert_close(triangle.index_select(0, rows), duplicated.index_select(0, rows))


def test_motion_contact_slip_is_invariant_to_sample_rate() -> None:
    """The same continuous support-frame speed measures equally at 30 and 60 Hz."""
    speed = 0.08
    time_30 = torch.arange(31, dtype=torch.float32) / 30.0
    time_60 = torch.arange(61, dtype=torch.float32) / 60.0
    quality_30 = _contact_quality(speed * time_30, 1.0 / 30.0)
    quality_60 = _contact_quality(speed * time_60, 1.0 / 60.0)
    torch.testing.assert_close(quality_30[_METRIC_CONTACT_SLIP_SPEED], torch.tensor(speed), atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(quality_60[_METRIC_CONTACT_SLIP_SPEED], torch.tensor(speed), atol=3.0e-6, rtol=0.0)


def test_motion_contact_cumulative_drift_is_independent_from_edge_slip() -> None:
    """A slow slide can pass the per-edge speed scale while failing cumulative drift."""
    quality = _contact_quality(torch.arange(10, dtype=torch.float32) * 0.004, 0.1)
    assert quality[_METRIC_CONTACT_SLIP_SPEED] < 0.15
    assert quality[_METRIC_CONTACT_CUMULATIVE_DRIFT] > 0.03


def test_exact_non_collinear_contact_planting_satisfies_publication_metrics() -> None:
    """One fixed non-collinear patch satisfies every planted-contact publication bound."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_contact_rows_accepted

    quality = _contact_quality(torch.zeros(6), 1.0 / 60.0)
    contact_rows = torch.tensor(
        (_METRIC_CONTACT_GAP, _METRIC_CONTACT_TILT, _METRIC_CONTACT_SLIP_SPEED, _METRIC_CONTACT_CUMULATIVE_DRIFT)
    )

    torch.testing.assert_close(quality.index_select(0, contact_rows), torch.zeros(4))
    assert bool(_motion_contact_rows_accepted(MotionTrajectorySolveCfg().acceptance.contact, quality.unsqueeze(0))[0])


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_rank_safe_contact_pose_equalities_restore_tilted_rigid_body(device_type: str) -> None:
    """Independent ankle position and rotation rows restore a tilted rigid support patch."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    wp.init()
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0)
    root_joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([root_joint])
    model = builder.finalize(device=str(device), requires_grad=True)
    patch = torch.tensor(
        ((0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.05, 0.0)),
        dtype=torch.float32,
        device=device,
    )
    objectives = (
        ik.IKObjectivePosition(
            link_index=body,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.from_torch(torch.zeros((1, 3), dtype=torch.float32, device=device), dtype=wp.vec3),
            weight=1.0,
        ),
        ik.IKObjectiveRotation(
            link_index=body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.from_torch(
                torch.tensor(((0.0, 0.0, 0.0, 1.0),), dtype=torch.float32, device=device), dtype=wp.vec4
            ),
            canonicalize_quat_err=True,
            weight=1.0,
        ),
    )
    optimizer = ik.IKOptimizerLM(model, 1, objectives, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    solver = IKTrajectorySolver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=6,
        damping=1.0e-6,
        krylov_max_iterations=128,
        krylov_relative_tolerance=1.0e-5,
        kkt_relative_tolerance=1.0e-5,
    )
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.ones((1, 2), dtype=torch.uint8, device=device),
        residual_starts_by_target=torch.tensor((0, 3), dtype=torch.int32, device=device),
    )
    tilt = 0.123
    joint_q = torch.tensor(
        ((0.01, -0.02, 0.03, math.sin(0.5 * tilt), 0.0, 0.0, math.cos(0.5 * tilt)),),
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(joint_q)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    direction_valid = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    coordinate_bounds = IKTrajectorySolver.CoordinateBounds(
        coordinate_indices=empty_i32,
        dof_indices=empty_i32,
        lower=empty_f32,
        upper=empty_f32,
    )
    for _ in range(20):
        solver.solve(
            joint_q,
            output,
            torch.tensor((0, 1), dtype=torch.int32, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
            torch.zeros(6, dtype=torch.float32, device=device),
            torch.zeros((3, 6), dtype=torch.float32, device=device),
            equalities=equalities,
            inequalities=None,
            coordinate_bounds=coordinate_bounds,
            joint_velocity=torch.zeros((1, 6), dtype=torch.float32, device=device),
            velocity_lower=torch.full((6,), -torch.inf, dtype=torch.float32, device=device),
            velocity_upper=torch.full((6,), torch.inf, dtype=torch.float32, device=device),
            segment_feasible=segment_feasible,
            segment_active=segment_active,
            segment_direction_valid=direction_valid,
            segment_globalization_succeeded=globalization_succeeded,
            feasibility_only=True,
            convergence_tolerance=1.0e-8,
        )
        joint_q.copy_(output)
        if not bool(segment_active[0]):
            break

    residuals, _ = optimizer.linearize(wp.from_torch(joint_q))
    assert bool(segment_feasible[0] and direction_valid[0] and globalization_succeeded[0])
    assert segment_active[0] == 0
    assert torch.max(torch.abs(wp.to_torch(residuals))) <= 64.0 * torch.finfo(torch.float32).eps
    solved_patch = joint_q[0, :3] + quat_apply(joint_q[0, 3:7].expand(patch.shape[0], 4), patch)
    torch.testing.assert_close(solved_patch, patch, atol=64.0 * torch.finfo(torch.float32).eps, rtol=0.0)


def test_motion_contact_inspection_exposes_solved_points_not_anchors() -> None:
    """Inspection publishes solved active patch points and no frozen onset target."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _trajectory_inspection_contacts

    frame_count = 3
    body_q = torch.zeros((frame_count, 1, 7), dtype=torch.float32)
    body_q[..., 6] = 1.0
    body_q[:, 0, 0] = torch.tensor((0.0, 0.1, 0.2))
    support_points = torch.tensor(((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.0)))
    stable = torch.tensor(((0,), (1,), (1,)), dtype=torch.uint8)
    points = torch.zeros((frame_count, 3, 3), dtype=torch.float32)
    valid = torch.zeros((frame_count, 3), dtype=torch.bool)
    wp.launch(
        _trajectory_inspection_contacts,
        dim=(3, frame_count),
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(torch.zeros(3, dtype=torch.int64)),
            wp.from_torch(support_points),
            wp.from_torch(torch.zeros(3, dtype=torch.int64)),
            wp.from_torch(stable),
            frame_count,
            3,
        ],
        outputs=[wp.from_torch(points), wp.from_torch(valid, dtype=wp.uint8)],
        device="cpu",
    )
    assert not valid[0].any()
    torch.testing.assert_close(points[1, valid[1]], support_points + torch.tensor((0.1, 0.0, 0.0)))
    torch.testing.assert_close(points[2, valid[2]], support_points + torch.tensor((0.2, 0.0, 0.0)))


def test_motion_trajectory_uses_one_monolithic_source_physical_contact_solve() -> None:
    """The trajectory root prepares one joint objective and keeps its final iterate."""
    import inspect

    from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTrajectorySolveCfg
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_monolithic_weights,
        _MotionTrajectoryResidualLayout,
        motion_solve_trajectory,
    )

    cfg = MotionTrajectorySolveCfg()
    layout = _MotionTrajectoryResidualLayout(
        source_global_position=slice(0, 3),
        source_rotation=slice(3, 6),
        source_direction_point=slice(6, 9),
        source_fidelity_guard=slice(24, 28),
        contact=slice(9, 16),
        activity_group_by_residual=torch.full((28,), -1, dtype=torch.int32),
        first_difference_group_by_residual=torch.full((28,), -1, dtype=torch.int32),
        joint_default=slice(16, 18),
        joint_reference=slice(18, 20),
        collision_objective=slice(20, 22),
        nonpenetration_objective=slice(22, 24),
    )
    targets = SimpleNamespace(required_position_rows=(0,), required_direction_rows=(0,), support_patch_offsets=(0, 1))
    base = torch.empty(layout.residual_count)
    temporal = torch.empty((3, layout.residual_count))

    _motion_monolithic_weights(layout, cfg, targets, base, temporal)
    assert torch.all(base[layout.source_global_position] == 1.0)
    assert torch.all(base[layout.source_rotation] == 1.0)
    assert torch.all(base[layout.source_direction_point] == 1.0)
    assert torch.all(base[layout.joint_default] == cfg.joint_default_position_weight)
    assert torch.all(base[layout.collision_objective] == 1.0)
    torch.testing.assert_close(base[layout.contact], torch.tensor((1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)))
    assert torch.count_nonzero(temporal[0, layout.contact.start + 4 : layout.contact.stop]) == 3
    assert torch.count_nonzero(base[layout.source_fidelity_guard]) == 0
    assert torch.count_nonzero(base[layout.joint_reference]) == 0
    assert torch.count_nonzero(temporal[:, layout.joint_reference]) == 0
    assert torch.all(base[layout.nonpenetration_objective] == 1.0)

    solve_source = inspect.getsource(motion_solve_trajectory)
    assert solve_source.count("IKTrajectorySolver(") == 1
    assert solve_source.count("solve_phase(") == 2  # Nested definition plus one call.
    assert solve_source.count("_motion_monolithic_weights(") == 1
    assert "_motion_source_weights(" not in solve_source
    assert "_motion_physical_projection_weights(" not in solve_source
    assert "_motion_contact_weights(" not in solve_source
    assert "source_inequalities" not in solve_source
    assert "_motion_phase_copy_selected(" not in solve_source
    gauge = solve_source.index("_motion_target_ground_gauge")
    witness = solve_source.index("_motion_scalar_velocity_box_witness", gauge)
    contact_targets = solve_source.index("prepare_contact_targets(frame_count, clip_offsets, phase_active)", witness)
    weights = solve_source.index("_motion_monolithic_weights(", contact_targets)
    solve = solve_source.index("solve_phase(", weights)
    finish = solve_source.index("_motion_phase_finish(", solve)
    assert gauge < witness < contact_targets < weights < solve < finish
    solve_call = solve_source[solve:finish]
    assert "residual_activity=residual_activity" in solve_call
    assert "inequalities=None" in solve_call
    assert "velocity_bounds=(workspace.source_velocity_lower, workspace.source_velocity_upper)" in solve_call
    assert "terminal_quality=" not in solve_call
    assert "terminal_acceptance=" not in solve_call
    assert "adaptive_recovery=" not in solve_call


def test_motion_trajectory_delegates_target_coordinate_bounds_to_solver() -> None:
    """The trajectory solver owns target bounds while Motion retains final certification."""
    import inspect

    from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_trajectory as module

    assert not hasattr(module, "_motion_target_coordinates_project")
    assert not hasattr(module, "_trajectory_project_coordinates")
    solve_source = inspect.getsource(module.motion_solve_trajectory)
    bounds_start = solve_source.index("solver_coordinate_bounds = IKTrajectorySolver.CoordinateBounds(")
    bounds_stop = solve_source.index("velocity_lower = ", bounds_start)
    bounds = solve_source[bounds_start:bounds_stop]
    assert "coordinate_q_indices" in bounds
    assert "coordinate_qd_indices" in bounds
    assert "lower=targets.coordinate_lower_limits_rad" in bounds
    assert "upper=targets.coordinate_upper_limits_rad" in bounds
    assert "coordinate_bounds=solver_coordinate_bounds" in solve_source
    assert "_motion_target_coordinates_project" not in solve_source
    assert "velocity_lower=velocity_lower" in solve_source
    assert "velocity_upper=velocity_upper" in solve_source
    assert "source_velocity_lower = torch.full_like(velocity_lower, -torch.inf)" in solve_source
    assert "source_velocity_upper = torch.full_like(velocity_upper, torch.inf)" in solve_source
    assert "physical_inequalities" not in solve_source
    assert "clearance_indices" not in solve_source
    assert "clearance_upper" not in solve_source
    assert "source_inequalities" not in solve_source
    assert "terminal_inequalities" not in solve_source
    assert "clearance_inequalities" not in solve_source
    assert "root_dof_indices = torch.arange(" in solve_source
    assert "frozen_dof_indices=root_dof_indices if source_root_fixed else None" in solve_source
    assert "contact_equalities = IKTrajectorySolver.ResidualEqualities(" not in solve_source
    assert "max_equality_residuals_per_frame=3 * contact_channel_count" not in solve_source
    assert solve_source.count("max_equality_residuals_per_frame=0") == 2
    assert solve_source.count("inequalities=None") == 1
    assert "inequalities=None" in solve_source
    assert not hasattr(module, "_motion_contact_align_clip_height")
    assert "IKObjectiveJointDefaultCfg" not in inspect.getsource(module._motion_frame_seed_objectives)


def test_motion_trajectory_builds_terms_in_declared_residual_order() -> None:
    """The composition root builds every objective and constraint in its declared row order."""
    import ast
    import inspect
    import textwrap

    from isaaclab_tasks.core.multi_task.motion.mdp.commands import motion_trajectory as module

    assert not hasattr(module, "motion_trajectory_objectives")
    assert not hasattr(module, "motion_trajectory_constraints")
    solve_source = inspect.getsource(module.motion_solve_trajectory)
    build_start = solve_source.index("    def build_system(")
    build_stop = solve_source.index("\n    residual_layout =", build_start)
    build_source = solve_source[build_start:build_stop]
    dedented_build_source = textwrap.dedent(build_source)
    build_tree = ast.parse(dedented_build_source)
    term_loop = next(
        node for node in ast.walk(build_tree) if isinstance(node, ast.For) and ast.unparse(node.target) == "term_cfg"
    )
    term_loop_source = ast.get_source_segment(dedented_build_source, term_loop)
    assert term_loop_source is not None
    assert "for term_cfg in cfg.objectives:" in build_source
    assert "features.extend(build.features)" in term_loop_source
    assert "constraint_builds.append(build)" in term_loop_source
    assert "motion_trajectory_objectives" not in build_source
    assert "motion_trajectory_constraints" not in build_source
