# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for MJCF-backed shared Newton kinematics."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.utils.math import convert_quat, quat_from_angle_axis

from isaaclab_tasks.core.multi_task.kinematics import (
    KinematicTree,
    NewtonKinematics,
    NewtonKinematicsCfg,
    kinematic_pose_forward,
    ordered_hinge_rotation,
    time_gradient,
)
from isaaclab_tasks.core.multi_task.motion.data.clip_index import MotionClipIndex
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    CmuHumEnvSmplClip,
    LafanG1Clip,
    cmu_humenv_smpl_skeleton,
    lafan_g1_29dof_skeleton,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
    MotionObjectiveMeasureCriterionCfg,
    MotionSemanticFamilyCfg,
    MotionSemanticSolveCfg,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
    _MotionCorpusCandidate,
    motion_solve_semantic_sequence,
)
from isaaclab_tasks.core.multi_task.motion.retarget import MotionSemanticProjection, MotionSemanticTargets
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import (
    G1_REFERENCE_MJCF_SHA256,
    G1FrameBuilder,
    g1_frame_builder,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import smpl_live_joint_mujoco_names
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
    SmplFrameBuilder,
    smpl_frame_builder,
    smpl_reference_kinematics,
)

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)
_G1_JOINT_NAMES = (
    "floating_base_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_G1_JOINT_Q = (
    0.10000000149011612,
    -0.20000000298023224,
    0.800000011920929,
    0.04989069700241089,
    -0.09978139400482178,
    0.024945348501205444,
    0.9934446811676025,
    -0.20000000298023224,
    -0.18571428954601288,
    -0.17142857611179352,
    -0.15714286267757416,
    -0.1428571492433548,
    -0.12857143580913544,
    -0.11428571492433548,
    -0.10000000149011612,
    -0.08571428805589676,
    -0.0714285746216774,
    -0.05714286118745804,
    -0.04285714402794838,
    -0.02857143059372902,
    -0.014285716228187084,
    1.862645149230957e-09,
    0.014285716228187084,
    0.02857143059372902,
    0.04285714402794838,
    0.05714286118745804,
    0.0714285746216774,
    0.08571428805589676,
    0.10000000149011612,
    0.11428571492433548,
    0.12857143580913544,
    0.1428571492433548,
    0.15714286267757416,
    0.17142857611179352,
    0.18571428954601288,
    0.20000000298023224,
)
_G1_BODY_INDICES = (0, 6, 12, 15, 22, 29)
_G1_BODY_POSE_XYZW = (
    (0.1000000015, -0.2000000030, 0.8000000119, 0.0498906970, -0.0997813940, 0.0249453485, 0.9934446216),
    (0.4151560366, -0.1001123711, 0.1341816336, -0.0767300501, -0.3377700150, -0.1141770035, 0.9311217070),
    (0.3518230319, -0.2880905271, 0.0754134655, -0.0051898076, -0.2187042981, -0.0405256934, 0.9749354720),
    (0.0855356827, -0.2053959668, 0.8518964648, 0.0442119017, -0.0991327763, 0.0100433892, 0.9940407872),
    (0.2710635364, -0.0419005491, 0.9295188785, 0.0877974331, -0.0254558884, 0.0922354460, 0.9915322065),
    (0.2568849921, -0.2980040312, 0.8668662906, 0.1879945844, 0.1055041552, 0.1894592345, 0.9579311609),
)

_G1_ROOT_QD = (0.3, -0.2, 0.1, 0.4, -0.5, 0.6)
_G1_BODY_QD = (
    (0.3000000119, -0.2000000030, 0.1000000015, 0.4000000060, -0.5, 0.6000000238),
    (0.5430480242, 0.2328508943, 0.2986770570, 0.4000000060, -0.5, 0.6000000238),
    (0.6860843897, 0.2222770005, 0.1945078969, 0.4000000060, -0.5, 0.6000000238),
    (0.1663342714, -0.3577778637, 0.0576289594, 0.4000000060, -0.5, 0.6000000238),
    (0.0973597467, -0.1486403793, 0.2778931558, 0.4000000060, -0.5, 0.6000000238),
    (0.2802806199, -0.1292574108, 0.1720984578, 0.4000000060, -0.5, 0.6000000238),
)


def _g1_mjcf_path() -> Path:
    """Return the exact external BFM MJCF when that reference checkout is available."""
    configured = os.environ.get("BFM_ZERO_G1_MJCF")
    candidates = (
        Path(configured).expanduser() if configured else None,
        Path.cwd().parents[1] / "BFM-Zero/humanoidverse/data/robots/g1/g1_29dof.xml",
    )
    path = next((candidate for candidate in candidates if candidate is not None and candidate.is_file()), None)
    if path is None:
        pytest.skip("The exact BFM-Zero G1 MJCF checkout is unavailable.")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == G1_REFERENCE_MJCF_SHA256
    return path


@pytest.fixture(scope="module")
def g1_kinematics() -> NewtonKinematics:
    """Build the exact released G1 reference model on CPU."""
    return NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(_g1_mjcf_path()), device="cpu"))


@pytest.fixture(scope="module")
def smpl_kinematics() -> NewtonKinematics:
    """Build the packaged exact target-SMPL reference model on CPU."""
    return smpl_reference_kinematics("", "cpu")


@pytest.fixture(scope="module")
def smpl_cross_builder(smpl_kinematics: NewtonKinematics) -> SmplFrameBuilder:
    """Build one source-generic G1-to-SMPL semantic retargeter."""
    source = lafan_g1_29dof_skeleton()
    return smpl_frame_builder(source, smpl_kinematics)


def _semantic_candidate(
    builder: SmplFrameBuilder | G1FrameBuilder,
    clip: CmuHumEnvSmplClip | LafanG1Clip,
    *,
    device: str = "cpu",
) -> _MotionCorpusCandidate:
    """Solve one clip through the production corpus-level semantic executor."""
    root_position, local_rotation = clip.semantic_local_pose(builder.source_skeleton, device=device)
    targets = builder.generate_semantic_targets(root_position, local_rotation)
    clip_index = MotionClipIndex(
        source_content_sha256="0" * 64,
        clips=(
            MotionClipIndex.Clip(
                clip_id="test",
                frame_count=clip.frame_count,
                source_fps=clip.source_fps,
                content_sha256="1" * 64,
            ),
        ),
    )
    candidate = _MotionCorpusCandidate(
        builder=builder,
        source=None,
        clip_index=clip_index,
        device=device,
        frames=None,
        pending=iter((targets,)),
    )
    return motion_solve_semantic_sequence(MotionSemanticSolveCfg(), candidate)


def _semantic_acceptance_bound(objective: str) -> float:
    """Return one root-visible semantic-family acceptance bound."""
    criteria = (
        criterion
        for criterion in MotionSemanticFamilyCfg().criteria
        if isinstance(criterion, MotionObjectiveMeasureCriterionCfg)
    )
    return next(criterion.upper for criterion in criteria if criterion.objective == objective)


def _semantic_output(
    builder: SmplFrameBuilder | G1FrameBuilder,
    clip: CmuHumEnvSmplClip | LafanG1Clip,
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return generalized coordinates and production semantic quality measures."""
    candidate = _semantic_candidate(builder, clip, device=device)
    if candidate.semantic_joint_q is None or candidate.semantic_quality is None:
        raise RuntimeError("The semantic executor did not materialize coordinates and quality.")
    generalized_position = torch.cat(
        (
            candidate.semantic_joint_q[:, :3],
            convert_quat(candidate.semantic_joint_q[:, 3:7], to="wxyz"),
            candidate.semantic_joint_q[:, 7:],
        ),
        dim=-1,
    )
    return generalized_position, candidate.semantic_quality[:, 0], candidate.semantic_quality[:, 1]


def test_mjcf_model_derives_exact_g1_dimensions_and_order(g1_kinematics: NewtonKinematics) -> None:
    """The parsed model, not a copied skeleton, must own every target label."""
    kin = g1_kinematics
    assert kin.mjcf_path == str(_g1_mjcf_path())
    assert kin.usd_path == ""
    assert kin.model.body_count == 30
    assert kin.model.joint_count == 30
    assert kin.model.joint_coord_count == 36
    assert kin.model.joint_dof_count == 35
    assert tuple(kin.body_names) == _G1_BODY_NAMES
    assert tuple(kin.joint_names) == _G1_JOINT_NAMES
    assert tuple(kin.joint_q_names[:7]) == (
        "floating_base_joint:position_x",
        "floating_base_joint:position_y",
        "floating_base_joint:position_z",
        "floating_base_joint:rotation_x",
        "floating_base_joint:rotation_y",
        "floating_base_joint:rotation_z",
        "floating_base_joint:rotation_w",
    )
    assert tuple(kin.joint_q_names[7:]) == _G1_JOINT_NAMES[1:]
    assert tuple(kin.joint_qd_names[:6]) == (
        "floating_base_joint:linear_velocity_x",
        "floating_base_joint:linear_velocity_y",
        "floating_base_joint:linear_velocity_z",
        "floating_base_joint:angular_velocity_x",
        "floating_base_joint:angular_velocity_y",
        "floating_base_joint:angular_velocity_z",
    )
    assert tuple(kin.joint_qd_names[6:]) == _G1_JOINT_NAMES[1:]
    tree = KinematicTree.from_newton(kin)
    assert tree.body_names == _G1_BODY_NAMES
    assert tree.joint_names == _G1_JOINT_NAMES[1:]
    assert tree.root_body_index == 0
    assert tree.joint_child_body_indices == tuple(range(1, 30))
    assert tree.num_bodies == 30 and tree.num_joints == 29


def test_mjcf_torch_batched_fk_matches_frozen_bfm_oracle_without_output_allocation(
    g1_kinematics: NewtonKinematics,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-owned Torch outputs must match released G1 reference kinematics."""
    kin = g1_kinematics
    joint_q = torch.tensor((_G1_JOINT_Q,), dtype=torch.float32)
    joint_qd = torch.zeros((1, 35), dtype=torch.float32)
    joint_qd[0, :6] = torch.tensor(_G1_ROOT_QD)
    body_q = torch.empty((1, 30, 7), dtype=torch.float32)
    body_qd = torch.empty((1, 30, 6), dtype=torch.float32)
    pointers = (body_q.data_ptr(), body_qd.data_ptr())

    returned = kin.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)
    assert returned == (body_q, body_qd)
    expected = torch.tensor(_G1_BODY_POSE_XYZW, dtype=torch.float32)
    body_indices = torch.tensor(_G1_BODY_INDICES)
    actual = body_q[0, body_indices]
    torch.testing.assert_close(actual[:, :3], expected[:, :3], rtol=0.0, atol=2.0e-6)
    sign = torch.where(torch.sum(actual[:, 3:] * expected[:, 3:], dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    torch.testing.assert_close(actual[:, 3:], sign * expected[:, 3:], rtol=0.0, atol=2.0e-6)
    expected_qd = torch.tensor(_G1_BODY_QD, dtype=torch.float32)
    torch.testing.assert_close(body_qd[0, body_indices], expected_qd, rtol=0.0, atol=2.0e-6)

    def allocation_forbidden(*_args, **_kwargs):
        raise AssertionError("Torch allocation occurred inside caller-owned batched FK.")

    monkeypatch.setattr(torch, "empty", allocation_forbidden)
    monkeypatch.setattr(torch, "zeros", allocation_forbidden)
    monkeypatch.setattr(torch, "empty_like", allocation_forbidden)
    monkeypatch.setattr(torch, "zeros_like", allocation_forbidden)
    kin.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)
    assert (body_q.data_ptr(), body_qd.data_ptr()) == pointers


def test_mjcf_torch_batched_fk_rejects_implicit_hot_path_copies(g1_kinematics: NewtonKinematics) -> None:
    """Wrong shapes, dtypes, and strides must fail instead of allocating repairs."""
    joint_q = torch.zeros((2, 36), dtype=torch.float32)
    joint_qd = torch.zeros((2, 35), dtype=torch.float32)
    body_q = torch.empty((2, 30, 7), dtype=torch.float32)
    body_qd = torch.empty((2, 30, 6), dtype=torch.float32)

    with pytest.raises(ValueError, match="contiguous"):
        g1_kinematics.eval_fk_batched_torch(joint_q[:, ::2], joint_qd, body_q, body_qd)
    with pytest.raises(ValueError, match="float32"):
        g1_kinematics.eval_fk_batched_torch(joint_q.to(torch.float64), joint_qd, body_q, body_qd)
    with pytest.raises(ValueError, match="shape"):
        g1_kinematics.eval_fk_batched_torch(joint_q, joint_qd, body_q[:, :29].contiguous(), body_qd)


def test_mjcf_and_usd_paths_are_mutually_exclusive() -> None:
    """One kinematic model cannot silently merge two target descriptions."""
    with pytest.raises(ValueError, match="exactly one"):
        NewtonKinematics(
            NewtonKinematicsCfg(
                usd_path="/tmp/robot.usd",
                mjcf_path="/tmp/robot.xml",
                device="cpu",
            )
        )


def test_usd_model_preserves_existing_public_labels() -> None:
    """Adding MJCF must not change the established USD body/joint label API."""
    usd_path = Path(ISAACLAB_ASSETS_DATA_DIR) / "Assets/Robots/Unitree/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
    kin = NewtonKinematics(NewtonKinematicsCfg(usd_path=str(usd_path), device="cpu"))

    assert kin.usd_path == str(usd_path)
    assert kin.mjcf_path == ""
    assert tuple(kin.body_names) == _G1_BODY_NAMES
    assert tuple(kin.joint_names) == ("", *_G1_JOINT_NAMES[1:])


def test_mjcf_kinematics_does_not_require_external_meshes(tmp_path: Path) -> None:
    """Reference FK must parse the articulated tree without loading visual meshes."""
    isolated_mjcf = tmp_path / "g1_29dof.xml"
    shutil.copyfile(_g1_mjcf_path(), isolated_mjcf)
    assert not (tmp_path / "meshes").exists()

    kin = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(isolated_mjcf), device="cpu"))

    assert kin.model.body_count == 30
    assert kin.model.joint_coord_count == 36


def _assert_grouped_coordinate_pose_matches_newton(
    kinematics: NewtonKinematics, skeleton, joint_q: torch.Tensor
) -> None:
    tree = KinematicTree.from_newton(kinematics)
    coordinate_indices = torch.tensor(tree.coordinate_q_indices, dtype=torch.int64)
    coordinates = joint_q.index_select(1, coordinate_indices)
    rest_translation = torch.tensor(skeleton.rest_translation_m, dtype=torch.float32)
    rest_rotation = convert_quat(torch.tensor(skeleton.rest_rotation_wxyz, dtype=torch.float32), to="xyzw")
    coordinate_axes = torch.tensor(tree.coordinate_axes, dtype=torch.float32)
    pose_delta = torch.zeros(joint_q.shape[0], tree.num_bodies, 4, dtype=torch.float32)
    pose_delta[..., 3] = 1.0
    for child, (start, stop) in zip(tree.joint_child_body_indices, tree.joint_coordinate_ranges):
        pose_delta[:, child] = ordered_hinge_rotation(coordinates[:, start:stop], coordinate_axes[start:stop])
    pose_delta[:, 0].copy_(joint_q[:, 3:7])
    pose_position, pose_rotation = kinematic_pose_forward(
        rest_translation, rest_rotation, pose_delta, joint_q[:, :3], tree.parent_indices
    )
    body_q = torch.empty(joint_q.shape[0], tree.num_bodies, 7, dtype=torch.float32)
    body_qd = torch.empty(joint_q.shape[0], tree.num_bodies, 6, dtype=torch.float32)
    kinematics.eval_fk_batched_torch(
        joint_q, torch.zeros(joint_q.shape[0], kinematics.model.joint_dof_count), body_q, body_qd
    )

    torch.testing.assert_close(pose_position, body_q[..., :3], atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        torch.abs(torch.sum(pose_rotation * body_q[..., 3:7], dim=-1)),
        torch.ones(joint_q.shape[0], tree.num_bodies),
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_g1_rest_times_pose_delta_matches_exact_newton_fk(g1_kinematics: NewtonKinematics) -> None:
    """G1 rest composition and scalar-axis frames match exact Newton FK for a multi-joint pose."""
    joint_q = torch.tensor((_G1_JOINT_Q,), dtype=torch.float32)
    _assert_grouped_coordinate_pose_matches_newton(g1_kinematics, lafan_g1_29dof_skeleton(), joint_q)


def test_smpl_grouped_xyz_and_distal_chain_match_exact_newton_fk(smpl_kinematics: NewtonKinematics) -> None:
    """SMPL grouped XYZ order and a nonzero distal chain match exact Newton FK."""
    kinematics = smpl_kinematics
    tree = KinematicTree.from_newton(kinematics)
    joint_q = torch.zeros(1, kinematics.model.joint_coord_count, dtype=torch.float32)
    joint_q[:, :3] = torch.tensor((0.1, -0.2, 0.9))
    joint_q[:, 3:7] = quat_from_angle_axis(torch.tensor([0.25]), torch.tensor([[0.2, -0.3, 0.9327379]]))
    coordinates = {name: index for index, name in enumerate(tree.coordinate_names)}
    values = {
        "L_Hip_x_L_Hip_y_L_Hip_z:coordinate_0": 0.2,
        "L_Hip_x_L_Hip_y_L_Hip_z:coordinate_1": -0.1,
        "L_Hip_x_L_Hip_y_L_Hip_z:coordinate_2": 0.15,
        "L_Knee_x_L_Knee_y_L_Knee_z:coordinate_0": 0.4,
        "L_Shoulder_x_L_Shoulder_y_L_Shoulder_z:coordinate_0": -0.25,
        "L_Shoulder_x_L_Shoulder_y_L_Shoulder_z:coordinate_1": 0.3,
        "L_Elbow_x_L_Elbow_y_L_Elbow_z:coordinate_1": -0.5,
        "L_Wrist_x_L_Wrist_y_L_Wrist_z:coordinate_2": 0.2,
    }
    for name, value in values.items():
        joint_q[:, tree.coordinate_q_indices[coordinates[name]]] = value
    _assert_grouped_coordinate_pose_matches_newton(kinematics, cmu_humenv_smpl_skeleton(), joint_q)


def test_smpl_local_pose_retarget_restores_root_and_projects_exact_limits(
    smpl_cross_builder: SmplFrameBuilder, smpl_kinematics: NewtonKinematics
) -> None:
    """The production Newton-IK edge publishes legal q and post-root-restore residuals."""
    source = lafan_g1_29dof_skeleton()
    kinematics = smpl_kinematics
    builder = smpl_cross_builder
    pose = np.zeros((3, source.num_bodies, 3), dtype=np.float32)
    source_by_name = {name: index for index, name in enumerate(source.body_names)}
    pose[1:, source_by_name["left_hip_pitch_link"], 1] = 0.25
    pose[1:, source_by_name["left_knee_link"], 1] = 0.4
    root_position = np.asarray(((0.0, 0.0, 0.8), (0.01, 0.0, 0.8), (0.02, 0.0, 0.8)), dtype=np.float32)

    generalized_position, marker_error_m, orientation_error_rad = _semantic_output(
        builder, LafanG1Clip(root_position, pose, 30.0)
    )
    tree = KinematicTree.from_newton(kinematics)

    source_root_xy = torch.from_numpy(root_position[:, :2])
    torch.testing.assert_close(generalized_position[:1, :2], source_root_xy[:1])
    source_displacement_x = source_root_xy[:, 0] - source_root_xy[0, 0]
    target_displacement_x = generalized_position[:, 0] - generalized_position[0, 0]
    root_motion_scale = target_displacement_x[-1] / source_displacement_x[-1]
    torch.testing.assert_close(target_displacement_x, root_motion_scale * source_displacement_x)
    torch.testing.assert_close(generalized_position[:, 1], source_root_xy[:, 1])
    assert torch.all(torch.isfinite(generalized_position))
    assert torch.all(torch.isfinite(marker_error_m))
    assert torch.all(torch.isfinite(orientation_error_rad))
    assert marker_error_m.amax() <= _semantic_acceptance_bound("landmark_position")
    assert torch.all(tree.coordinates_within_limits(generalized_position[:, 7:]))
    assert torch.count_nonzero(generalized_position[:, 7:]) > 0


def _lafan_semantic_clip(
    source,
    role: str,
    angles_rad: np.ndarray,
    *,
    use_position_body: bool = False,
) -> LafanG1Clip:
    landmarks = {landmark.name: landmark for landmark in source.landmarks}
    landmark = landmarks[role]
    body_name = landmark.position_body_name if use_position_body else landmark.rotation_body_name
    body = source.body_names.index(body_name)
    joint = source.joint_child_body_indices.index(body)
    axis = np.asarray(source.joint_axes[joint], dtype=np.float32)
    pose = np.zeros((angles_rad.shape[0], source.num_bodies, 3), dtype=np.float32)
    pose[:, body] = angles_rad[:, None] * axis
    root = np.zeros((angles_rad.shape[0], 3), dtype=np.float32)
    root[:, 2] = 0.8
    return LafanG1Clip(root, pose, 30.0)


def _target_coordinate_slice(tree: KinematicTree, body_name: str) -> slice:
    body = tree.body_names.index(body_name)
    joint = tree.joint_child_body_indices.index(body)
    start, stop = tree.joint_coordinate_ranges[joint]
    return slice(start, stop)


def _source_support_positions(source, clip) -> torch.Tensor:
    root_position, body_rotation = clip.semantic_local_pose(source, device="cpu")
    position, _ = kinematic_pose_forward(
        torch.tensor(source.rest_translation_m),
        convert_quat(torch.tensor(source.rest_rotation_wxyz), to="xyzw"),
        body_rotation,
        root_position,
        source.parent_indices,
    )
    landmarks = {landmark.name: landmark for landmark in source.landmarks}
    support_indices = torch.tensor(
        tuple(source.body_names.index(landmarks[role].position_body_name) for role in ("left_ankle", "right_ankle"))
    )
    return position.index_select(1, support_indices)


def _source_support_height(source, clip) -> torch.Tensor:
    return _source_support_positions(source, clip)[..., 2].amin(dim=1)


def test_smpl_semantic_retarget_preserves_axial_wrist_rotation(
    smpl_cross_builder: SmplFrameBuilder, smpl_kinematics: NewtonKinematics
) -> None:
    """A terminal source rotation must change target wrist coordinates, unlike position-only IK."""
    source = smpl_cross_builder.source_skeleton
    clip = _lafan_semantic_clip(source, "left_wrist", np.asarray((0.0, 1.0, -1.0), dtype=np.float32))
    generalized_position, position_error, orientation_error = _semantic_output(smpl_cross_builder, clip)
    wrist = generalized_position[:, 7:][
        :, _target_coordinate_slice(KinematicTree.from_newton(smpl_kinematics), "L_Wrist")
    ]

    response = torch.linalg.vector_norm(wrist[1:] - wrist[:1], dim=-1)
    assert torch.all(response > 0.1)
    assert position_error.amax() <= _semantic_acceptance_bound("landmark_position")


def test_smpl_semantic_retarget_stays_continuous_and_matches_support_height(
    smpl_cross_builder: SmplFrameBuilder, smpl_kinematics: NewtonKinematics
) -> None:
    """A smooth one-sided ramp cannot select a distant unrelated-limb solution."""
    source = smpl_cross_builder.source_skeleton
    clip = _lafan_semantic_clip(source, "left_hip", np.linspace(0.0, 0.56, 9, dtype=np.float32), use_position_body=True)
    generalized_position, marker_error_m, orientation_error_rad = _semantic_output(smpl_cross_builder, clip)
    tree = KinematicTree.from_newton(smpl_kinematics)
    coordinates = generalized_position[:, 7:]
    right_shoulder = coordinates[:, _target_coordinate_slice(tree, "R_Shoulder")]
    assert torch.max(torch.abs(torch.diff(right_shoulder, dim=0))) < 0.02
    assert torch.max(torch.abs(torch.diff(coordinates, dim=0))) < 0.1
    assert torch.max(torch.abs(torch.diff(coordinates, n=2, dim=0))) / (1.0 / clip.source_fps) ** 2 < 80.0

    joint_q = generalized_position.clone()
    joint_q[:, 3:7] = convert_quat(generalized_position[:, 3:7], to="xyzw")
    body_q = torch.empty(generalized_position.shape[0], tree.num_bodies, 7)
    body_qd = torch.empty(generalized_position.shape[0], tree.num_bodies, 6)
    smpl_kinematics.eval_fk_batched_torch(
        joint_q, torch.zeros(generalized_position.shape[0], smpl_kinematics.model.joint_dof_count), body_q, body_qd
    )
    target_supports = torch.tensor((tree.body_names.index("L_Ankle"), tree.body_names.index("R_Ankle")))
    target_height = body_q.index_select(1, target_supports)[..., 2].amin(dim=1)
    source_height = _source_support_height(source, clip)
    torch.testing.assert_close(target_height, source_height, atol=2.0e-6, rtol=0.0)
    assert torch.all(body_q.index_select(1, target_supports)[..., 2] >= source_height[:, None] - 2.0e-6)
    assert torch.all(torch.isfinite(marker_error_m))
    assert torch.all(torch.isfinite(orientation_error_rad))
    assert marker_error_m.amax() <= _semantic_acceptance_bound("landmark_position")


def test_semantic_solver_schedule_aligns_projection_and_convergence_checks() -> None:
    """Projected LM measures convergence only after a projection boundary."""
    cfg = MotionSemanticSolveCfg()

    assert cfg.max_iterations == 30
    assert cfg.projection_interval == cfg.convergence_check_interval == 5
    assert cfg.max_iterations % cfg.projection_interval == 0


def test_g1_and_smpl_builders_share_one_semantic_projection(
    g1_kinematics: NewtonKinematics,
    smpl_cross_builder: SmplFrameBuilder,
) -> None:
    """Both robot targets parameterize the same source-to-target projection implementation."""
    g1_cross_builder = g1_frame_builder(cmu_humenv_smpl_skeleton(), g1_kinematics)

    assert type(g1_cross_builder.semantic) is MotionSemanticProjection
    assert type(smpl_cross_builder.semantic) is MotionSemanticProjection


def test_g1_exact_profile_matches_frozen_bfm_fk_and_analytic_velocity(g1_kinematics: NewtonKinematics) -> None:
    """The production G1 builder must match a frozen external FK pose and analytic rigid translation."""
    source = lafan_g1_29dof_skeleton()
    frame_count = 4
    source_fps = 30.0
    root_velocity = np.asarray((0.3, -0.2, 0.1), dtype=np.float32)
    frame_time = np.arange(frame_count, dtype=np.float32)[:, None] / source_fps
    root_translation = np.asarray(_G1_JOINT_Q[:3], dtype=np.float32)[None] + frame_time * root_velocity
    pose_axis_angle = np.zeros((frame_count, source.num_bodies, 3), dtype=np.float32)
    pose_axis_angle[:, 0] = np.asarray((0.1, -0.2, 0.05), dtype=np.float32)
    pose_axis_angle[:, 1:] = (
        np.asarray(_G1_JOINT_Q[7:], dtype=np.float32)[None, :, None]
        * np.asarray(source.joint_axes, dtype=np.float32)[None]
    )
    clip = LafanG1Clip(root_translation, pose_axis_angle, source_fps)
    builder = g1_frame_builder(source, g1_kinematics)
    joint_q, joint_qd = clip.free_root_coordinates(source, device="cpu")

    frames = builder.build_exact_coordinates(joint_q, joint_qd, clip.source_fps)

    body_indices = torch.tensor(_G1_BODY_INDICES)
    frozen_pose = torch.tensor(_G1_BODY_POSE_XYZW, dtype=torch.float32)
    translation = torch.from_numpy(frame_time * root_velocity)
    expected_position = frozen_pose[None, :, :3] + translation[:, None]
    actual_pose = frames.body_position[:, body_indices]
    actual_rotation = frames.body_rotation[:, body_indices]
    torch.testing.assert_close(actual_pose, expected_position, rtol=0.0, atol=2.0e-6)
    expected_rotation = frozen_pose[None, :, 3:].expand_as(actual_rotation)
    sign = torch.where((actual_rotation * expected_rotation).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    torch.testing.assert_close(actual_rotation, sign * expected_rotation, rtol=0.0, atol=2.0e-6)
    torch.testing.assert_close(
        frames.body_linear_velocity,
        torch.from_numpy(root_velocity).expand_as(frames.body_linear_velocity),
        rtol=0.0,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        frames.body_angular_velocity, torch.zeros_like(frames.body_angular_velocity), atol=1.0e-5, rtol=0.0
    )
    torch.testing.assert_close(frames.joint_velocity, torch.zeros_like(frames.joint_velocity), atol=2.0e-6, rtol=0.0)


def test_smpl_exact_profile_matches_independent_tree_fk_and_analytic_velocity(
    smpl_kinematics: NewtonKinematics,
) -> None:
    """The native SMPL route must match source-tree FK and analytic rigid translation."""
    source = cmu_humenv_smpl_skeleton()
    frame_count = 4
    source_fps = 30.0
    root_velocity = torch.tensor((0.3, -0.2, 0.1), dtype=torch.float32)
    frame_time = torch.arange(frame_count, dtype=torch.float32)[:, None] / source_fps
    root_position = torch.tensor((0.1, -0.2, 0.9), dtype=torch.float32)[None] + frame_time * root_velocity
    source_coordinates = torch.linspace(-0.25, 0.25, source.num_joints, dtype=torch.float32)

    qpos = np.zeros((frame_count, 7 + source.num_joints), dtype=np.float32)
    qpos[:, :3] = root_position.numpy()
    qpos[:, 3] = 1.0
    qpos[:, 7:] = source_coordinates.numpy()
    qvel = np.zeros((frame_count, 6 + source.num_joints), dtype=np.float32)
    qvel[:, :3] = root_velocity.numpy()
    clip = CmuHumEnvSmplClip(qpos, qvel, source_fps)
    builder = smpl_frame_builder(source, smpl_kinematics)
    joint_q, joint_qd = clip.free_root_coordinates(source, device="cpu")

    frames = builder.build_exact_coordinates(joint_q, joint_qd, clip.source_fps)

    axes = torch.tensor(source.joint_axes, dtype=torch.float32).reshape(source.num_bodies - 1, 3, 3)
    coordinates = source_coordinates.reshape(source.num_bodies - 1, 3)
    local_rotation = ordered_hinge_rotation(coordinates, axes).expand(frame_count, -1, -1)
    root_rotation = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=torch.float32).expand(frame_count, 1, 4)
    body_position, body_rotation = kinematic_pose_forward(
        torch.tensor(source.rest_translation_m, dtype=torch.float32),
        convert_quat(torch.tensor(source.rest_rotation_wxyz, dtype=torch.float32), to="xyzw"),
        torch.cat((root_rotation, local_rotation), dim=1),
        root_position,
        source.parent_indices,
    )
    live_names = smpl_live_joint_mujoco_names(builder.joint_names)
    live_indices = torch.tensor([source.joint_names.index(name) for name in live_names])

    torch.testing.assert_close(frames.joint_position, source_coordinates[live_indices].expand(frame_count, -1))
    torch.testing.assert_close(frames.body_position, body_position, rtol=0.0, atol=2.0e-6)
    sign = torch.where((frames.body_rotation * body_rotation).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    torch.testing.assert_close(frames.body_rotation, sign * body_rotation, rtol=0.0, atol=2.0e-6)
    torch.testing.assert_close(frames.body_linear_velocity, root_velocity.expand_as(frames.body_linear_velocity))
    torch.testing.assert_close(frames.body_angular_velocity, torch.zeros_like(frames.body_angular_velocity))
    torch.testing.assert_close(frames.joint_velocity, torch.zeros_like(frames.joint_velocity))


def test_smpl_semantic_retarget_uses_declared_unsmoothed_derivative(
    smpl_cross_builder: SmplFrameBuilder,
) -> None:
    """Cross-source SMPL velocities use first-edge/central-interior gradients without smoothing."""
    source = smpl_cross_builder.source_skeleton
    clip = _lafan_semantic_clip(source, "left_knee", np.asarray((0.0, 0.1, 0.3, 0.6, 1.0), dtype=np.float32))
    candidate = _semantic_candidate(smpl_cross_builder, clip)
    clip_index = MotionClipIndex(
        source_content_sha256="0" * 64,
        clips=(
            MotionClipIndex.Clip(
                clip_id="test",
                frame_count=clip.frame_count,
                source_fps=clip.source_fps,
                content_sha256="1" * 64,
            ),
        ),
    )
    assert candidate.semantic_joint_q is not None
    frames = smpl_cross_builder.build_semantic_corpus(candidate.semantic_joint_q, clip_index)
    expected = time_gradient(frames.joint_position.unsqueeze(0), 1.0 / clip.source_fps).squeeze(0)
    torch.testing.assert_close(frames.joint_velocity, expected)


def test_g1_semantic_retarget_uses_source_roles_and_exact_support(
    g1_kinematics: NewtonKinematics,
) -> None:
    """CMU-to-G1 composition uses semantic world poses and corrects target morphology height."""
    source = cmu_humenv_smpl_skeleton()
    tree = KinematicTree.from_newton(g1_kinematics)
    builder = g1_frame_builder(source, g1_kinematics)
    frame_count = 9
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qpos[:, 2] = 0.9
    qpos[:, 3] = 1.0
    source_landmarks = {landmark.name: landmark for landmark in source.landmarks}
    hip_body = source.body_names.index(source_landmarks["left_hip"].position_body_name)
    qpos[:, 7 + 3 * (hip_body - 1) + 1] = np.linspace(0.0, 0.56, frame_count, dtype=np.float32)
    clip = CmuHumEnvSmplClip(qpos, np.zeros((frame_count, 75), dtype=np.float32), 30.0)
    candidate = _semantic_candidate(builder, clip)
    assert candidate.semantic_joint_q is not None and candidate.semantic_quality is not None
    joint_q = candidate.semantic_joint_q
    marker_error_m = candidate.semantic_quality[:, 0]
    orientation_error_rad = candidate.semantic_quality[:, 1]

    body_q = torch.empty(frame_count, tree.num_bodies, 7)
    body_qd = torch.empty(frame_count, tree.num_bodies, 6)
    g1_kinematics.eval_fk_batched_torch(
        joint_q, torch.zeros(frame_count, g1_kinematics.model.joint_dof_count), body_q, body_qd
    )
    target_supports = torch.tensor(
        (tree.body_names.index("left_ankle_roll_link"), tree.body_names.index("right_ankle_roll_link"))
    )
    target_feet = body_q.index_select(1, target_supports)[..., :3]
    target_height = target_feet[..., 2].amin(dim=1)
    source_feet = _source_support_positions(source, clip)
    source_height = source_feet[..., 2].amin(dim=1)
    torch.testing.assert_close(target_height, source_height, atol=2.0e-6, rtol=0.0)
    assert torch.all(target_feet[..., 2] >= source_height[:, None] - 2.0e-6)

    step_seconds = 1.0 / clip.source_fps
    coordinates = joint_q[:, 7:]
    assert torch.max(torch.abs(torch.diff(coordinates, dim=0))) < 0.1
    assert torch.max(torch.abs(torch.diff(coordinates, n=2, dim=0))) / step_seconds**2 < 2.0
    source_speed = torch.linalg.vector_norm(torch.diff(source_feet[..., :2], dim=0) / step_seconds, dim=-1)
    target_speed = torch.linalg.vector_norm(torch.diff(target_feet[..., :2], dim=0) / step_seconds, dim=-1)
    source_support = source_feet[..., 2] <= source_height[:, None] + 0.02
    source_planted = source_support[:-1] & source_support[1:] & (source_speed < 0.15)
    assert torch.any(source_planted)
    assert torch.max(target_speed[source_planted]) < 0.08
    assert torch.all(torch.isfinite(marker_error_m))
    assert torch.all(torch.isfinite(orientation_error_rad))
    assert marker_error_m.amax() <= _semantic_acceptance_bound("landmark_position")


def test_semantic_ik_position_objective_is_uniform_scale_invariant(tmp_path: Path) -> None:
    """Uniformly scaling target geometry must preserve q and coordinates and scale physical residuals."""
    mjcf = """<mujoco model="scaled_chain">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <freejoint name="root_joint"/>
      <geom type="sphere" size="{radius}" mass="1"/>
      <body name="elbow" pos="0 0 {scale}">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="{radius} {half_length}" mass="1"/>
        <body name="wrist" pos="0 0 {scale}">
          <joint name="elbow_joint" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="{radius}" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    results = []
    for scale in (0.5, 2.0):
        path = tmp_path / f"chain_{scale}.xml"
        path.write_text(mjcf.format(scale=scale, radius=0.05 * scale, half_length=0.5 * scale))
        kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(path), device="cpu"))
        tree = KinematicTree.from_newton(kinematics)
        desired_q = torch.tensor(kinematics.default_joint_q, dtype=torch.float32).unsqueeze(0)
        desired_q[:, tree.coordinate_q_indices[0]] = 0.7
        desired_q[:, tree.coordinate_q_indices[1]] = -0.4
        desired_body_q = torch.empty(1, 3, 7)
        desired_body_qd = torch.empty(1, 3, 6)
        kinematics.eval_fk_batched_torch(
            desired_q,
            torch.zeros(1, kinematics.model.joint_dof_count),
            desired_body_q,
            desired_body_qd,
        )
        body_indices = (0, 1, 2)
        body_index_tensor = torch.tensor(body_indices)
        target_position = desired_body_q[..., :3].transpose(0, 1).contiguous()
        target_rotation = torch.zeros(3, 1, 4)
        target_rotation[..., 3] = 1.0
        targets = MotionSemanticTargets(
            body_indices=body_indices,
            parent_rows=(-1, 0, 1),
            body_index_tensor=body_index_tensor,
            position_m=target_position,
            rotation_xyzw=target_rotation,
            segment_lengths_m=torch.full((3,), scale),
            segment_length_values_m=(scale, scale, scale),
            coordinate_indices=torch.tensor(tree.coordinate_q_indices),
            coordinate_lower_limits_rad=torch.tensor(tree.coordinate_lower_limits_rad),
            coordinate_upper_limits_rad=torch.tensor(tree.coordinate_upper_limits_rad),
            support_body_indices=torch.tensor((2,)),
            source_support_height_m=target_position[2, :, 2].clone(),
        )
        clip_index = MotionClipIndex(
            source_content_sha256="0" * 64,
            clips=(
                MotionClipIndex.Clip(
                    clip_id="test",
                    frame_count=1,
                    source_fps=30.0,
                    content_sha256="1" * 64,
                ),
            ),
        )
        builder = SimpleNamespace(semantic_reference_kinematics=kinematics, semantic_target_tree=tree)
        candidate = motion_solve_semantic_sequence(
            MotionSemanticSolveCfg(),
            _MotionCorpusCandidate(
                builder=builder,
                source=None,
                clip_index=clip_index,
                device="cpu",
                frames=None,
                pending=iter((targets,)),
            ),
        )
        assert candidate.semantic_joint_q is not None and candidate.semantic_quality is not None
        results.append(
            (
                candidate.semantic_joint_q[:, tree.coordinate_q_indices],
                candidate.semantic_quality[:, 0],
                candidate.semantic_quality[:, 1],
            )
        )

    small_q, small_error_m, small_orientation_error = results[0]
    large_q, large_error_m, large_orientation_error = results[1]
    torch.testing.assert_close(small_q, large_q, atol=3.0e-5, rtol=0.0)
    torch.testing.assert_close(4.0 * small_error_m, large_error_m, atol=3.0e-5, rtol=0.0)
    torch.testing.assert_close(small_orientation_error, large_orientation_error, atol=3.0e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU/GPU semantic-retarget parity.")
def test_semantic_retarget_cpu_gpu_has_clean_launch_and_fk_residual_parity(tmp_path: Path) -> None:
    """The 75-DOF/84-residual solve must launch cleanly and preserve FK diagnostics across devices."""
    script = r"""
import sys
import numpy as np
import torch
from isaaclab_tasks.core.multi_task.motion.data.clip_index import MotionClipIndex
from isaaclab_tasks.core.multi_task.motion.data.sources import LafanG1Clip, lafan_g1_29dof_skeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionSemanticSolveCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
    _MotionCorpusCandidate,
    motion_solve_semantic_sequence,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import smpl_frame_builder, smpl_reference_kinematics

device, output_path = sys.argv[1:]
source = lafan_g1_29dof_skeleton()
pose = np.zeros((9, source.num_bodies, 3), dtype=np.float32)
landmarks = {landmark.name: landmark for landmark in source.landmarks}
body = source.body_names.index(landmarks["left_hip"].position_body_name)
joint = source.joint_child_body_indices.index(body)
pose[:, body] = np.linspace(0.0, 0.56, 9, dtype=np.float32)[:, None] * np.asarray(
    source.joint_axes[joint], dtype=np.float32
)
root = np.zeros((9, 3), dtype=np.float32)
root[:, 2] = 0.8
reference = smpl_reference_kinematics("", device)
builder = smpl_frame_builder(source, reference)
clip = LafanG1Clip(root, pose, 30.0)
root_position, local_rotation = clip.semantic_local_pose(source, device=device)
targets = builder.generate_semantic_targets(root_position, local_rotation)
index = MotionClipIndex(
    source_content_sha256="0" * 64,
    clips=(
        MotionClipIndex.Clip(
            clip_id="test",
            frame_count=clip.frame_count,
            source_fps=clip.source_fps,
            content_sha256="1" * 64,
        ),
    ),
)
candidate = motion_solve_semantic_sequence(
    MotionSemanticSolveCfg(),
    _MotionCorpusCandidate(
        builder=builder,
        source=None,
        clip_index=index,
        device=device,
        frames=None,
        pending=iter((targets,)),
    ),
)
assert candidate.semantic_joint_q is not None
assert candidate.semantic_quality is not None
assert candidate.frame_finite is not None
output = (
    candidate.semantic_joint_q,
    candidate.semantic_quality,
    candidate.frame_finite,
)
if device != "cpu":
    torch.cuda.synchronize()
torch.save(tuple(value.cpu() for value in output), output_path)
"""
    outputs = []
    for device in ("cpu", "cuda:0"):
        output_path = tmp_path / f"semantic_{device.replace(':', '_')}.pt"
        result = subprocess.run(
            [sys.executable, "-c", script, device, str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "CUDA error" not in combined
        assert "Failed to configure kernel dynamic shared memory" not in combined
        outputs.append(torch.load(output_path, weights_only=True))

    cpu, gpu = outputs
    torch.testing.assert_close(cpu[0], gpu[0], atol=3.0e-4, rtol=3.0e-4)
    torch.testing.assert_close(cpu[1], gpu[1], atol=1.1e-3, rtol=3.0e-4)
    torch.testing.assert_close(cpu[2], gpu[2])
