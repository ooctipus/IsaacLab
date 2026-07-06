# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared cross-skeleton kinematics foundations."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.utils.math import (
    convert_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

from isaaclab_tasks.core.multi_task.kinematics import (
    KinematicTree,
    fit_ordered_hinge_coordinates,
    kinematic_retarget_positions,
    kinematic_root_basis,
    kinematic_seed_target_rotations,
    kinematic_tree_forward,
    ordered_hinge_rotation,
)
from isaaclab_tasks.core.multi_task.motion.data import MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    cmu_humenv_smpl_skeleton,
    lafan_g1_29dof_skeleton,
)


def _rest_pose(skeleton: MotionSkeleton) -> tuple[torch.Tensor, torch.Tensor]:
    local_position = torch.tensor(skeleton.rest_translation_m, dtype=torch.float32)
    local_rotation = convert_quat(torch.tensor(skeleton.rest_rotation_wxyz, dtype=torch.float32), to="xyzw")
    return kinematic_tree_forward(local_position, local_rotation, skeleton.parent_indices)


def _body_index(skeleton: MotionSkeleton, name: str) -> int:
    return skeleton.body_names.index(name)


def _root_basis(skeleton: MotionSkeleton, positions: torch.Tensor) -> torch.Tensor:
    landmarks = {landmark.name: landmark for landmark in skeleton.landmarks}
    roles = ("pelvis", "left_hip", "right_hip", "torso")
    root, left, right, up = (_body_index(skeleton, landmarks[role].position_body_name) for role in roles)
    return kinematic_root_basis(positions, root, left, right, up)


def test_rest_forward_kinematics_derives_anatomical_root_alignment() -> None:
    """Rest transforms, not world-axis assumptions, define source-to-target root alignment."""
    source = lafan_g1_29dof_skeleton()
    target = cmu_humenv_smpl_skeleton()
    source_position, source_rotation = _rest_pose(source)
    target_position, target_rotation = _rest_pose(target)
    source_basis = _root_basis(source, source_position)
    target_basis = _root_basis(target, target_position)
    target_to_source = source_basis @ target_basis.transpose(-1, -2)

    torch.testing.assert_close(source_rotation.norm(dim=-1), torch.ones(source.num_bodies))
    torch.testing.assert_close(target_rotation.norm(dim=-1), torch.ones(target.num_bodies))
    torch.testing.assert_close(target_to_source @ target_basis, source_basis, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(target_to_source @ target_to_source.T, torch.eye(3), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(torch.linalg.det(target_to_source), torch.tensor(1.0), atol=1.0e-6, rtol=1.0e-6)


def test_semantic_edges_share_nonidentity_root_alignment_with_rotations() -> None:
    """Rest and rigid-root target edges must use the same non-identity alignment as orientation."""
    axis = torch.nn.functional.normalize(torch.tensor((0.3, -0.5, 0.8)), dim=0)
    root_delta = matrix_from_quat(quat_from_angle_axis(torch.tensor(0.7), axis))
    alignment = matrix_from_quat(
        quat_from_angle_axis(torch.tensor(-0.6), torch.nn.functional.normalize(torch.tensor((0.2, 0.9, -0.3)), dim=0))
    )
    source_rest_rotation = matrix_from_quat(
        quat_from_angle_axis(
            torch.tensor((0.2, -0.4, 0.3)),
            torch.nn.functional.normalize(torch.tensor(((1.0, 2.0, -1.0), (-2.0, 1.0, 0.5), (0.3, -0.7, 1.0))), dim=-1),
        )
    )
    target_rest_rotation = matrix_from_quat(
        quat_from_angle_axis(
            torch.tensor((-0.3, 0.5, -0.2)),
            torch.nn.functional.normalize(torch.tensor(((0.5, -1.0, 2.0), (1.0, 0.2, -0.4), (-0.8, 0.6, 0.1))), dim=-1),
        )
    )
    target_rest_edges = torch.tensor(((0.0, 0.0, 0.0), (0.2, 0.1, 0.4), (-0.1, 0.3, 0.2)))
    aligned_rest_rotation = alignment @ target_rest_rotation
    aligned_rest_edges = (alignment @ target_rest_edges.unsqueeze(-1)).squeeze(-1)
    source_rotation = torch.stack((source_rest_rotation, root_delta @ source_rest_rotation), dim=1)
    target_rotation = source_rotation @ source_rest_rotation[:, None].transpose(-1, -2) @ aligned_rest_rotation[:, None]
    root_position = torch.tensor(((0.3, -0.2, 0.8), (-0.4, 0.5, 1.1)))

    position = kinematic_retarget_positions(
        root_position, target_rotation, aligned_rest_rotation, aligned_rest_edges, (-1, 0, 1)
    )
    expected_rest = torch.stack(
        (
            root_position[0],
            root_position[0] + alignment @ target_rest_edges[1],
            root_position[0] + alignment @ (target_rest_edges[1] + target_rest_edges[2]),
        )
    )
    torch.testing.assert_close(position[:, 0], expected_rest, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(
        position[:, 1] - root_position[1],
        (root_delta @ (expected_rest - root_position[0]).T).T,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_g1_knee_flexion_reexpresses_as_smpl_knee_flexion() -> None:
    """An isolated G1 knee flexion maps to target flexion rather than a forbidden secondary axis."""
    source = lafan_g1_29dof_skeleton()
    target = cmu_humenv_smpl_skeleton()
    source_position, source_rotation = _rest_pose(source)
    target_position, target_rotation = _rest_pose(target)
    target_to_source = _root_basis(source, source_position) @ _root_basis(target, target_position).T

    source_parent = _body_index(source, "left_hip_yaw_link")
    target_parent = _body_index(target, "L_Hip")
    source_parent_rest = matrix_from_quat(source_rotation[source_parent])
    target_parent_rest = matrix_from_quat(target_rotation[target_parent])
    target_parent_to_source_parent = source_parent_rest.T @ target_to_source @ target_parent_rest
    source_knee_axis = torch.tensor(source.joint_axes[source.joint_names.index("left_knee_joint")])
    source_delta = matrix_from_quat(quat_from_angle_axis(torch.tensor(0.5), source_knee_axis))
    target_delta = target_parent_to_source_parent.T @ source_delta @ target_parent_to_source_parent
    target_coordinates, residual = fit_ordered_hinge_coordinates(quat_from_matrix(target_delta), torch.eye(3))

    torch.testing.assert_close(target_coordinates, torch.tensor([0.5, 0.0, 0.0]), atol=1.5e-2, rtol=0.0)
    torch.testing.assert_close(residual, torch.tensor(0.0), atol=1.0e-6, rtol=0.0)


def test_grouped_newton_joint_coordinates_preserve_indices_and_bounds() -> None:
    """Grouped target joints retain exact generalized-position, velocity, axis, and bound rows."""
    topology = SimpleNamespace(
        body_count=2,
        joint_count=2,
        coordinate_count=10,
        body_parent=np.asarray((-1, 0)),
        joint_child=np.asarray((0, 1)),
        joint_q_start=np.asarray((0, 7, 10)),
        joint_qd_start=np.asarray((0, 6, 9)),
        joint_dof_dim=np.asarray(((0, 6), (0, 3))),
        joint_axis=np.asarray(((1.0, 0.0, 0.0),) * 6 + ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        joint_limit_lower=np.asarray((-np.inf,) * 6 + (-1.0, -0.2, -0.1)),
        joint_limit_upper=np.asarray((np.inf,) * 6 + (1.0, 0.2, 0.1)),
    )
    kinematics = SimpleNamespace(
        topology=topology,
        body_names=("root", "knee"),
        joint_names=("root", "knee_joint"),
        joint_q_names=(
            "root:x",
            "root:y",
            "root:z",
            "root:qx",
            "root:qy",
            "root:qz",
            "root:qw",
            "knee:x",
            "knee:y",
            "knee:z",
        ),
    )
    tree = KinematicTree.from_newton(kinematics)

    assert tree.joint_names == ("knee_joint",)
    assert tree.joint_coordinate_ranges == ((0, 3),)
    assert tree.coordinate_names == ("knee:x", "knee:y", "knee:z")
    assert tree.coordinate_child_body_indices == (1, 1, 1)
    assert tree.coordinate_q_indices == (7, 8, 9)
    assert tree.coordinate_qd_indices == (6, 7, 8)
    assert tree.coordinate_axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert tree.coordinate_lower_limits_rad == (-1.0, -0.2, -0.1)
    assert tree.coordinate_upper_limits_rad == (1.0, 0.2, 0.1)
    torch.testing.assert_close(
        tree.coordinates_within_limits(torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, -0.2]])),
        torch.tensor([[True, True, True], [False, True, False]]),
    )

    with pytest.raises(ValueError, match="uniquely own every non-root body"):
        replace(
            tree,
            body_names=("root", "knee", "ankle"),
            parent_indices=(-1, 0, 1),
            joint_names=("knee_joint", "ankle_joint"),
            joint_child_body_indices=(1, 1),
            joint_coordinate_ranges=((0, 1), (1, 3)),
        )


def test_direct_semantic_joint_seed_uses_exact_parent_and_child_frames() -> None:
    """A direct semantic edge must seed the exact bounded target rotation instead of the default pose."""
    tree = KinematicTree(
        body_names=("root", "hand"),
        parent_indices=(-1, 0),
        joint_names=("wrist",),
        joint_child_body_indices=(1,),
        joint_coordinate_ranges=((0, 3),),
        coordinate_names=("wrist_x", "wrist_y", "wrist_z"),
        coordinate_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        coordinate_q_indices=(7, 8, 9),
        coordinate_qd_indices=(6, 7, 8),
        coordinate_lower_limits_rad=(2.0, 2.0, 2.0),
        coordinate_upper_limits_rad=(3.1, 3.1, 3.1),
    )
    parent_frame = quat_from_angle_axis(torch.tensor(0.3), torch.tensor((0.0, 0.0, 1.0)))
    child_frame = quat_from_angle_axis(torch.tensor(-0.2), torch.tensor((0.0, 1.0, 0.0)))
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    topology = SimpleNamespace(
        body_count=2,
        coordinate_count=10,
        body_joint=np.asarray((0, 1)),
        joint_parent=np.asarray((-1, 0)),
        joint_child=np.asarray((0, 1)),
        joint_transform_parent=np.asarray((identity, (0.0, 0.0, 0.0, *parent_frame.tolist()))),
        joint_transform_child=np.asarray((identity, (0.0, 0.0, 0.0, *child_frame.tolist()))),
    )
    target_coordinates = torch.tensor(((2.8, 2.5, 2.7),))
    root_rotation = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    child_rotation = quat_mul(
        quat_mul(parent_frame.expand(1, 4), ordered_hinge_rotation(target_coordinates, torch.eye(3))),
        quat_conjugate(child_frame).expand(1, 4),
    )
    target_rotation = torch.stack((root_rotation, child_rotation))
    joint_q = torch.zeros(1, 10)

    kinematic_seed_target_rotations(
        tree,
        topology,
        target_body_indices=(0, 1),
        target_parent_rows=(-1, 0),
        target_rotation_xyzw=target_rotation,
        joint_q=joint_q,
    )

    torch.testing.assert_close(joint_q[:, 7:10], target_coordinates, atol=1.0e-6, rtol=0.0)


def test_serial_semantic_path_seed_leaves_unsupported_joints_at_default() -> None:
    """A semantic edge spanning multiple target joints must defer to shared IK without partial seeding."""
    tree = KinematicTree(
        body_names=("root", "elbow", "hand"),
        parent_indices=(-1, 0, 1),
        joint_names=("shoulder", "wrist"),
        joint_child_body_indices=(1, 2),
        joint_coordinate_ranges=((0, 1), (1, 2)),
        coordinate_names=("shoulder", "wrist"),
        coordinate_axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        coordinate_q_indices=(7, 8),
        coordinate_qd_indices=(6, 7),
        coordinate_lower_limits_rad=(-1.0, -1.0),
        coordinate_upper_limits_rad=(1.0, 1.0),
    )
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    topology = SimpleNamespace(
        body_count=3,
        coordinate_count=9,
        body_joint=np.asarray((0, 1, 2)),
        joint_parent=np.asarray((-1, 0, 1)),
        joint_child=np.asarray((0, 1, 2)),
        joint_transform_parent=np.asarray((identity, identity, identity)),
        joint_transform_child=np.asarray((identity, identity, identity)),
    )
    target_rotation = torch.zeros(2, 1, 4)
    target_rotation[..., 3] = 1.0
    target_rotation[1] = quat_from_angle_axis(torch.tensor(0.7), torch.tensor((1.0, 0.0, 0.0)))
    joint_q = torch.tensor(((0.0,) * 7 + (0.2, -0.4),))
    expected = joint_q.clone()

    kinematic_seed_target_rotations(
        tree,
        topology,
        target_body_indices=(0, 2),
        target_parent_rows=(-1, 0),
        target_rotation_xyzw=target_rotation,
        joint_q=joint_q,
    )

    torch.testing.assert_close(joint_q, expected)


def test_direct_child_seed_uses_propagated_parent_when_semantic_parent_was_skipped() -> None:
    """A direct child must fit against its parent's reachable default rotation, not its desired rotation."""
    tree = KinematicTree(
        body_names=("root", "bridge", "parent", "child"),
        parent_indices=(-1, 0, 1, 2),
        joint_names=("bridge_joint", "parent_joint", "child_joint"),
        joint_child_body_indices=(1, 2, 3),
        joint_coordinate_ranges=((0, 1), (1, 2), (2, 5)),
        coordinate_names=("bridge_y", "parent_z", "child_x", "child_y", "child_z"),
        coordinate_axes=(
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        coordinate_q_indices=(7, 8, 9, 10, 11),
        coordinate_qd_indices=(6, 7, 8, 9, 10),
        coordinate_lower_limits_rad=(-1.0,) * 5,
        coordinate_upper_limits_rad=(1.0,) * 5,
    )
    identity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    topology = SimpleNamespace(
        body_count=4,
        coordinate_count=12,
        body_joint=np.asarray((0, 1, 2, 3)),
        joint_parent=np.asarray((-1, 0, 1, 2)),
        joint_child=np.asarray((0, 1, 2, 3)),
        joint_transform_parent=np.asarray((identity,) * 4),
        joint_transform_child=np.asarray((identity,) * 4),
    )
    default_parent_coordinates = torch.tensor(((0.2, -0.1),))
    desired_child_coordinates = torch.tensor(((0.4, -0.2, 0.1),))
    bridge_rotation = ordered_hinge_rotation(default_parent_coordinates[:, :1], torch.tensor(((0.0, 1.0, 0.0),)))
    parent_local = ordered_hinge_rotation(default_parent_coordinates[:, 1:], torch.tensor(((0.0, 0.0, 1.0),)))
    actual_parent = quat_mul(bridge_rotation, parent_local)
    target_child = quat_mul(actual_parent, ordered_hinge_rotation(desired_child_coordinates, torch.eye(3)))
    target_rotation = torch.stack(
        (
            torch.tensor(((0.0, 0.0, 0.0, 1.0),)),
            quat_from_angle_axis(torch.tensor(1.0), torch.tensor((0.0, 0.0, 1.0))).unsqueeze(0),
            target_child,
        )
    )
    joint_q = torch.zeros(1, 12)
    joint_q[:, 7:9].copy_(default_parent_coordinates)

    kinematic_seed_target_rotations(
        tree,
        topology,
        target_body_indices=(0, 2, 3),
        target_parent_rows=(-1, 0, 1),
        target_rotation_xyzw=target_rotation,
        joint_q=joint_q,
    )

    torch.testing.assert_close(joint_q[:, 7:9], default_parent_coordinates)
    torch.testing.assert_close(joint_q[:, 9:12], desired_child_coordinates, atol=1.0e-6, rtol=0.0)
