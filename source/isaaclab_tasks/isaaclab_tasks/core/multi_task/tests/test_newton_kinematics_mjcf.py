# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for MJCF-backed shared Newton kinematics."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

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
    quat_from_angle_axis,
    quat_mul,
)

from isaaclab_tasks.core.multi_task.kinematics import (
    IKTrajectorySolver,
    KinematicTree,
    NewtonKinematics,
    NewtonKinematicsBuildCfg,
    NewtonKinematicsCfg,
    kinematic_pose_forward,
    kinematic_tree_forward,
    ordered_hinge_coordinate_velocity,
    ordered_hinge_rotation,
    time_quaternion_angular_velocity,
)
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import IKObjectiveMeshCollisionCfg
from isaaclab_tasks.core.multi_task.motion import retarget as motion_retarget
from isaaclab_tasks.core.multi_task.motion.data.clip_index import MotionClipIndex
from isaaclab_tasks.core.multi_task.motion.data.frames import (
    MotionSourceProjectionExact,
    MotionSourceProjectionTrajectory,
)
from isaaclab_tasks.core.multi_task.motion.data.sources import (
    CmuHumEnvSmplClip,
    LafanG1Clip,
    cmu_humenv_smpl_skeleton,
    lafan_g1_29dof_skeleton,
)
from isaaclab_tasks.core.multi_task.motion.data.sources.lafan_bvh import LafanBvhHierarchy, lafan_bvh_skeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
    MotionGroundPenetrationCriterionCfg,
    MotionSourceGlobalPositionObjectiveCfg,
    MotionSourceRotationObjectiveCfg,
    MotionTaskTableCfg,
    MotionTrajectorySolveCfg,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
    _QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_NAMES,
    _TRAJECTORY_METRIC_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
    _motion_task_view,
    _MotionTrajectorySolvedCandidate,
    _MotionTrajectoryTargetCandidate,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
    _motion_contact_rows_accepted,
    _motion_source_fidelity_accepted,
    motion_objective_source_global_position,
    motion_objective_source_rotation,
    motion_solve_trajectory,
)
from isaaclab_tasks.core.multi_task.motion.retarget import (
    MotionTrajectoryProjection,
    MotionTrajectoryTargets,
    motion_contact_probe_offsets,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
    G1_MOTION_ARTICULATION_CFG,
    G1_SIMULATOR_BODY_NAMES,
    G1_SIMULATOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import (
    G1_REFERENCE_MJCF_SHA256,
    _G1FrameTarget,
    g1_frame_target,
    g1_source_projection,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl import reference as smpl_reference_module
from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import (
    SMPL_MOTION_ARTICULATION_CFG,
    smpl_live_joint_mujoco_names,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
    _SmplFrameTarget,
    smpl_frame_target,
    smpl_reference_kinematics,
    smpl_source_projection,
)
from isaaclab_tasks.core.multi_task.motion.robots.target import write_velocity_canonical

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

_METRIC_SOURCE_REQUIRED_POSITION = _TRAJECTORY_METRIC_NAMES.index("source_required_position_max_m")
_METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION = _TRAJECTORY_METRIC_NAMES.index("source_required_distal_direction_max_rad")
_METRIC_SOURCE_ROOT_ROTATION = _TRAJECTORY_METRIC_NAMES.index("source_root_rotation_max_rad")
_METRIC_CONTACT_GAP = _TRAJECTORY_METRIC_NAMES.index("contact_gap_max_m")
_METRIC_CONTACT_TILT = _TRAJECTORY_METRIC_NAMES.index("contact_tilt_max_rad")
_METRIC_CONTACT_CUMULATIVE_DRIFT = _TRAJECTORY_METRIC_NAMES.index("contact_cumulative_drift_max_m")
_TARGET_GROUND_PENETRATION = _TARGET_COORDINATE_QUALITY_NAMES.index("ground_penetration_max_m")

_CONTACT_CHANNELS = (
    MotionTaskTableCfg.ContactChannelCfg(name="left_foot", source_probe_roles=("left_ankle", "left_toe")),
    MotionTaskTableCfg.ContactChannelCfg(name="right_foot", source_probe_roles=("right_ankle", "right_toe")),
)
_CONTACT_PROBE_ROLES = tuple(role for channel in _CONTACT_CHANNELS for role in channel.source_probe_roles)


def _contact_offsets(target, contact_channels=_CONTACT_CHANNELS) -> torch.Tensor:
    """Build target-device canonical source-probe offsets."""
    frame_target = getattr(target, "frame_target", target)
    return motion_contact_probe_offsets(contact_channels, frame_target.kinematics.device)


def _trajectory_projection(source, target, contact_channels=_CONTACT_CHANNELS) -> MotionTrajectoryProjection:
    """Build one projection with shared-layout semantics."""
    return MotionTrajectoryProjection(source, target, contact_channels, _contact_offsets(target, contact_channels))


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


_SMPL_CONTACT_PATCHES = _contact_patches("L_Ankle", "R_Ankle")
_G1_CONTACT_PATCHES = _contact_patches("left_ankle_roll_link", "right_ankle_roll_link")

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


@dataclass(frozen=True, slots=True)
class _LocalPoseClip:
    """Minimal test clip carrying world-root positions and root/local unit rotations."""

    root_position_m: torch.Tensor
    local_rotation_xyzw: torch.Tensor
    source_fps: float

    def __post_init__(self) -> None:
        """Require one finite float32 pose trajectory."""
        frame_count = self.root_position_m.shape[0] if self.root_position_m.ndim == 2 else 0
        body_count = self.local_rotation_xyzw.shape[1] if self.local_rotation_xyzw.ndim == 3 else 0
        if (
            frame_count < 1
            or body_count < 1
            or self.root_position_m.shape != (frame_count, 3)
            or self.local_rotation_xyzw.shape != (frame_count, body_count, 4)
            or self.root_position_m.dtype is not torch.float32
            or self.local_rotation_xyzw.dtype is not torch.float32
            or self.root_position_m.device != self.local_rotation_xyzw.device
            or not np.isfinite(self.source_fps)
            or self.source_fps <= 0.0
        ):
            raise ValueError("Local-pose test clips require nonempty same-device float32 poses and positive timing.")
        norm = torch.linalg.vector_norm(self.local_rotation_xyzw, dim=-1)
        if not bool(torch.all(torch.isfinite(self.root_position_m))) or not torch.allclose(
            norm, torch.ones_like(norm), atol=1.0e-6, rtol=1.0e-6
        ):
            raise ValueError("Local-pose test clips require finite positions and unit rotations.")

    @property
    def frame_count(self) -> int:
        """Number of source frames."""
        return self.root_position_m.shape[0]

    def local_pose(self, source_skeleton, *, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the stored pose after checking the source body count."""
        if source_skeleton.num_bodies != self.local_rotation_xyzw.shape[1]:
            raise ValueError("Local-pose test clip and source skeleton body counts differ.")
        return self.root_position_m.to(device=device), self.local_rotation_xyzw.to(device=device)


_MotionTestClip = CmuHumEnvSmplClip | LafanG1Clip | _LocalPoseClip


def test_from_articulation_builds_native_mjcf_scene_mechanics() -> None:
    """The selected SMPL scene MJCF is the exact kinematics owner."""
    kinematics = NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), SMPL_MOTION_ARTICULATION_CFG, "cpu"
    )

    assert kinematics.asset_path == SMPL_MOTION_ARTICULATION_CFG.spawn.asset_path
    assert kinematics.mjcf_path == SMPL_MOTION_ARTICULATION_CFG.spawn.asset_path
    assert kinematics.usd_path == ""


def test_from_articulation_builds_usd_scene_mechanics() -> None:
    """The selected G1 scene USD is the exact kinematics owner."""
    kinematics = NewtonKinematics.from_articulation(
        NewtonKinematicsBuildCfg(collapse_fixed_joints=False), G1_MOTION_ARTICULATION_CFG, "cpu"
    )

    assert kinematics.asset_path == G1_MOTION_ARTICULATION_CFG.spawn.usd_path
    assert kinematics.usd_path == G1_MOTION_ARTICULATION_CFG.spawn.usd_path
    assert kinematics.mjcf_path == ""


@pytest.fixture(scope="module")
def smpl_cross_projection(smpl_kinematics: NewtonKinematics) -> MotionSourceProjectionTrajectory:
    """Build one raw-LAFAN-to-SMPL source projection and its independent target."""
    body_names = (
        "Hips",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToe",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToe",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
    )
    hierarchy = LafanBvhHierarchy(
        body_names=body_names,
        parent_indices=(-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20),
        rest_translation_m=(
            (0.0, 0.0, 0.0),
            (-0.1, -0.1, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, -0.1, 0.15),
            (0.1, -0.1, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, -0.1, 0.15),
            (0.0, 0.2, 0.0),
            (0.0, 0.15, 0.0),
            (0.0, 0.15, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.2, 0.0),
            (-0.1, 0.1, 0.0),
            (-0.15, 0.0, 0.0),
            (-0.3, 0.0, 0.0),
            (-0.25, 0.0, 0.0),
            (0.1, 0.1, 0.0),
            (0.15, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (0.25, 0.0, 0.0),
        ),
        channels=(
            ("Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation"),
            *((("Zrotation", "Yrotation", "Xrotation"),) * (len(body_names) - 1)),
        ),
        end_sites=(),
        source_sha256="0" * 64,
    )
    source = lafan_bvh_skeleton(hierarchy)
    target = smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)
    return smpl_source_projection(source, target, object(), _CONTACT_CHANNELS, _contact_offsets(target))


def _clip_index(frame_count: int, source_fps: float) -> MotionClipIndex:
    """Return one deterministic test clip index."""
    return MotionClipIndex(
        source_content_sha256="0" * 64,
        skeleton_identity_sha256s=("2" * 64,),
        clips=(MotionClipIndex.Clip("test", frame_count, source_fps, "1" * 64, 0),),
    )


def _semantic_candidate(
    projection: MotionSourceProjectionTrajectory,
    clip: _MotionTestClip,
    *,
    device: str = "cpu",
) -> _MotionTrajectorySolvedCandidate:
    """Solve one clip through the production corpus-level semantic executor."""
    root_position, local_rotation = clip.local_pose(projection.source_skeleton, device=device)
    targets = projection.target_projection.generate_targets(root_position, local_rotation)
    clip_index = _clip_index(clip.frame_count, clip.source_fps)
    candidate = _MotionTrajectoryTargetCandidate(
        target=projection.target,
        clip_index=clip_index,
        pending=iter((targets,)),
        source_body_counts=(projection.source_skeleton.num_bodies,),
        device=device,
        inspection=False,
    )
    return motion_solve_trajectory(MotionTrajectorySolveCfg(), candidate)


def _source_required_position_bound() -> float:
    """Return the target-required source-position bound [m]."""
    return MotionTrajectorySolveCfg().acceptance.source.required_position_upper_m


def _source_root_rotation_bound() -> float:
    """Return the unified source-root rotation bound [rad]."""
    return MotionTrajectorySolveCfg().acceptance.source.root_rotation_upper_rad


def _contact_gap_bound() -> float:
    """Return the trajectory contact-gap acceptance bound [m]."""
    return MotionTrajectorySolveCfg().acceptance.contact.gap_upper_m


def _contact_tilt_bound() -> float:
    """Return the trajectory contact-tilt acceptance bound [rad]."""
    return MotionTrajectorySolveCfg().acceptance.contact.tilt_upper_rad


def _semantic_ground_penetration_bound() -> float:
    """Return the route-independent target ground-penetration bound [m]."""
    return MotionGroundPenetrationCriterionCfg().upper_m


def _assert_exact_velocity_respects_model_limits(
    target: _SmplFrameTarget | _G1FrameTarget,
    joint_q: torch.Tensor,
    clip: _MotionTestClip,
) -> None:
    """Require exact q-derived velocities to remain inside every finite target-model limit."""
    joint_qd = torch.empty(
        (joint_q.shape[0], target.kinematics.model.joint_dof_count),
        dtype=joint_q.dtype,
        device=joint_q.device,
    )
    write_velocity_canonical(
        target,
        joint_q,
        torch.tensor((0, clip.frame_count), dtype=torch.int32, device=joint_q.device),
        torch.tensor((1.0 / clip.source_fps,), dtype=torch.float32, device=joint_q.device),
        joint_qd,
    )
    velocity_lower = torch.tensor(target.kinematics.topology.joint_velocity_lower)
    velocity_upper = torch.tensor(target.kinematics.topology.joint_velocity_upper)
    torch.testing.assert_close(velocity_lower, -velocity_upper)
    joint_type = torch.tensor(target.kinematics.topology.joint_type)
    dof_joint = torch.tensor(target.kinematics.topology.dof_joint)
    free = joint_type[dof_joint] == int(newton.JointType.FREE)
    assert torch.count_nonzero(free) == 6
    assert torch.all(torch.isneginf(velocity_lower[free]))
    assert torch.all(torch.isposinf(velocity_upper[free]))
    finite = torch.isfinite(velocity_upper) & (velocity_upper > 0.0)
    assert torch.all(finite | free)
    utilization = joint_qd[:, finite].abs().amax(dim=0) / velocity_upper[finite]
    assert utilization.max() < 1.0e-4


def _semantic_output(
    projection: MotionSourceProjectionTrajectory,
    clip: _MotionTestClip,
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coordinates, internal refinement error, and source-fidelity quality measures."""
    candidate = _semantic_candidate(projection, clip, device=device)
    generalized_position = torch.cat(
        (
            candidate.coordinates.joint_q[:, :3],
            convert_quat(candidate.coordinates.joint_q[:, 3:7], to="wxyz"),
            candidate.coordinates.joint_q[:, 7:],
        ),
        dim=-1,
    )
    return (
        generalized_position,
        candidate.trajectory_quality[:, _METRIC_SOURCE_REQUIRED_POSITION],
        candidate.trajectory_quality[:, _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION],
        candidate.trajectory_quality[:, _METRIC_SOURCE_ROOT_ROTATION],
        candidate.trajectory_quality[:, _METRIC_CONTACT_CUMULATIVE_DRIFT],
        candidate.target_coordinate_evidence[:, _TARGET_GROUND_PENETRATION],
    )


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


def _assert_native_trajectory_evidence_obeys_target_morphology(source, target, clip) -> None:
    """Require source motion, target morphology, and rest-calibrated semantic rotations."""
    root_position, local_rotation = clip.local_pose(source, device="cpu")
    projection = _trajectory_projection(source, target.trajectory_target)
    targets = projection.generate_targets(root_position, local_rotation)
    rest_translation = torch.tensor(source.rest_translation_m, dtype=torch.float32)
    rest_rotation = convert_quat(torch.tensor(source.rest_rotation_wxyz, dtype=torch.float32), to="xyzw")
    source_position, source_world_rotation = kinematic_pose_forward(
        rest_translation, rest_rotation, local_rotation, root_position, source.parent_indices
    )
    layout = target.trajectory_target
    source_landmarks = {landmark.name: landmark for landmark in source.landmarks}
    source_body_by_name = {name: index for index, name in enumerate(source.body_names)}
    source_indices = tuple(source_body_by_name[source_landmarks[role].position_body_name] for role in layout.roles)
    source_semantic_position = source_position[:, source_indices].transpose(0, 1)

    root_position_target = targets.source_landmark_position_m[0]
    root_rotation_target = targets.source_landmark_rotation_xyzw[0]
    for index, row in enumerate(layout.root_cluster_rows):
        relative_root = quat_apply(
            quat_conjugate(root_rotation_target),
            targets.source_landmark_position_m[row] - root_position_target,
        )
        torch.testing.assert_close(
            relative_root, layout.root_cluster_offset_m[index].expand_as(relative_root), atol=2.0e-6, rtol=0.0
        )

    root_cluster_rows = set(layout.root_cluster_rows)
    for row, parent in enumerate(layout.parent_rows[1:], start=1):
        if row in root_cluster_rows:
            continue
        target_edge = targets.source_landmark_position_m[row] - targets.source_landmark_position_m[parent]
        source_edge = source_semantic_position[row] - source_semantic_position[parent]
        target_direction = target_edge / torch.linalg.vector_norm(target_edge, dim=-1, keepdim=True)
        source_direction = source_edge / torch.linalg.vector_norm(source_edge, dim=-1, keepdim=True)
        torch.testing.assert_close(target_direction, source_direction, atol=2.0e-6, rtol=0.0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(target_edge, dim=-1),
            layout.segment_lengths_m[row].expand(clip.frame_count),
            atol=2.0e-6,
            rtol=0.0,
        )

    distal = targets.source_direction_point_position_m - targets.source_landmark_position_m.index_select(
        0, layout.direction_rows
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(distal, dim=-1),
        layout.direction_lengths_m[:, None].expand(-1, clip.frame_count),
        atol=2.0e-6,
        rtol=0.0,
    )

    rotation_source_indices = torch.tensor(
        tuple(source_body_by_name[source_landmarks[role].rotation_body_name] for role in layout.rotation_roles)
    )
    _, source_rest_world_rotation = kinematic_tree_forward(rest_translation, rest_rotation, source.parent_indices)
    source_rest_semantic_rotation = source_rest_world_rotation.index_select(0, rotation_source_indices)
    source_to_target_rest_rotation = quat_mul(
        quat_conjugate(source_rest_semantic_rotation), layout.rotation_rest_xyzw.to(rest_rotation)
    )
    expected_rotation = quat_mul(
        source_world_rotation.index_select(1, rotation_source_indices),
        source_to_target_rest_rotation.unsqueeze(0).expand(clip.frame_count, -1, -1),
    ).transpose(0, 1)
    expected_rotation /= torch.linalg.vector_norm(expected_rotation, dim=-1, keepdim=True)
    rotation_alignment = torch.abs(torch.sum(targets.source_landmark_rotation_xyzw * expected_rotation, dim=-1))
    torch.testing.assert_close(rotation_alignment, torch.ones_like(rotation_alignment), atol=2.0e-6, rtol=0.0)
    assert targets.rotation_body_indices == layout.rotation_body_indices
    assert targets.rotation_weights == layout.rotation_weights


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


def test_smpl_target_rejects_reordered_packaged_child_bodies(
    smpl_kinematics: NewtonKinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed assumes each packaged three-axis joint owns the next body."""
    tree = KinematicTree.from_newton(smpl_kinematics)
    child_bodies = list(tree.joint_child_body_indices)
    child_bodies[0], child_bodies[1] = child_bodies[1], child_bodies[0]
    reordered = replace(tree, joint_child_body_indices=tuple(child_bodies))
    monkeypatch.setattr(KinematicTree, "from_newton", classmethod(lambda cls, reference: reordered))

    with pytest.raises(ValueError, match="child bodies in order"):
        smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)


def test_smpl_target_rejects_non_xyz_coordinate_order(
    smpl_kinematics: NewtonKinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equivalent cardinal-axis sets are insufficient for the ordered fitter."""
    tree = KinematicTree.from_newton(smpl_kinematics)
    coordinate_axes = list(tree.coordinate_axes)
    coordinate_axes[0], coordinate_axes[1] = coordinate_axes[1], coordinate_axes[0]
    reordered = replace(tree, coordinate_axes=tuple(coordinate_axes))
    monkeypatch.setattr(KinematicTree, "from_newton", classmethod(lambda cls, reference: reordered))

    with pytest.raises(ValueError, match="ordered positive X, Y, and Z"):
        smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)


def test_smpl_target_rejects_nonzero_default_coordinates(
    smpl_kinematics: NewtonKinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target default body rotations represent rest only for zero coordinates."""
    default_joint_q = smpl_kinematics.default_joint_q.copy()
    default_joint_q[7] = 0.125
    monkeypatch.setattr(smpl_kinematics, "_default_joint_q", default_joint_q)

    with pytest.raises(ValueError, match="default non-root coordinates must be zero"):
        smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)


@pytest.mark.parametrize("mutation", ("missing_root", "duplicate", "hand_overlap", "child_before_parent"))
def test_smpl_target_rejects_incomplete_or_nontopological_seed_maps(
    smpl_kinematics: NewtonKinematics, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Every uninitialized global-rotation slot must have one proven writer."""
    rotations = list(smpl_reference_module._SMPL_RETARGET_ROTATIONS)
    expected_message = "uniquely cover"
    if mutation == "missing_root":
        rotations[0] = replace(rotations[0], body_name="L_Hand")
    elif mutation == "duplicate":
        rotations[-1] = rotations[-2]
    elif mutation == "hand_overlap":
        rotations[-1] = replace(rotations[-1], body_name="R_Hand")
    else:
        rotations[1], rotations[2] = rotations[2], rotations[1]
        expected_message = "parent-before-child topology"
    monkeypatch.setattr(smpl_reference_module, "_SMPL_RETARGET_ROTATIONS", tuple(rotations))

    with pytest.raises(ValueError, match=expected_message):
        smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)


def test_smpl_full_rotation_seed_preserves_root_limits_and_improves_default(
    smpl_kinematics: NewtonKinematics,
) -> None:
    """The real packaged target reconstructs a legal pose better than its default seed."""
    target = smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)
    tree = target.kinematic_tree
    frame_count = 3
    root_position = torch.tensor(((0.1, -0.2, 0.9), (0.2, -0.1, 1.0), (0.3, 0.0, 1.1)))
    root_rotation = quat_from_angle_axis(torch.tensor((0.1, 0.2, 0.3)), torch.tensor(((0.0, 0.0, 1.0),) * frame_count))
    reference_joint_q = torch.tensor(smpl_kinematics.default_joint_q, dtype=torch.float32)
    reference_joint_q = reference_joint_q.expand(frame_count, -1).clone()
    reference_joint_q[:, :3].copy_(root_position)
    reference_joint_q[:, 3:7].copy_(root_rotation)
    hip_y = target.reference_coordinate_names.index("L_Hip_y")
    reference_joint_q[:, tree.coordinate_q_indices[hip_y]] = torch.tensor((0.0, 0.3, -0.2))

    body_q = torch.empty((frame_count, tree.num_bodies, 7), dtype=torch.float32)
    body_qd = torch.empty((frame_count, tree.num_bodies, 6), dtype=torch.float32)
    zero_qd = torch.zeros((frame_count, smpl_kinematics.model.joint_dof_count), dtype=torch.float32)
    smpl_kinematics.eval_fk_batched_torch(reference_joint_q, zero_qd, body_q, body_qd)
    rotation_body_indices = target._trajectory_seed_rotation_body_indices
    rotation_index = torch.tensor(rotation_body_indices, dtype=torch.int64)
    landmark_rotation = body_q[:, :, 3:7].index_select(1, rotation_index).transpose(0, 1).contiguous()

    seed_joint_q = target.trajectory_seed_joint_q(
        root_position_m=root_position,
        rotation_body_indices=rotation_body_indices,
        landmark_rotation_xyzw=landmark_rotation,
    )
    torch.testing.assert_close(seed_joint_q[:, :3], root_position, rtol=0.0, atol=0.0)
    torch.testing.assert_close(seed_joint_q[:, 3:7], landmark_rotation[0], rtol=0.0, atol=0.0)
    assert torch.all(tree.coordinates_within_limits(seed_joint_q[:, 7:]))
    torch.testing.assert_close(
        target._reference_from_canonical_indices,
        torch.arange(tree.num_coordinates, dtype=torch.int64),
        rtol=0.0,
        atol=0.0,
    )

    default_joint_q = torch.tensor(smpl_kinematics.default_joint_q, dtype=torch.float32)
    default_joint_q = default_joint_q.expand(frame_count, -1).clone()
    default_joint_q[:, :3].copy_(root_position)
    default_joint_q[:, 3:7].copy_(landmark_rotation[0])

    def mapped_rotation_error(joint_q: torch.Tensor) -> torch.Tensor:
        actual_body_q = torch.empty_like(body_q)
        actual_body_qd = torch.empty_like(body_qd)
        smpl_kinematics.eval_fk_batched_torch(joint_q, zero_qd, actual_body_q, actual_body_qd)
        actual = actual_body_q[:, :, 3:7].index_select(1, rotation_index).transpose(0, 1)
        alignment = torch.abs(torch.sum(actual * landmark_rotation, dim=-1)).clamp(max=1.0)
        return 2.0 * torch.acos(alignment)

    seed_error = mapped_rotation_error(seed_joint_q)
    default_error = mapped_rotation_error(default_joint_q)
    assert seed_error.mean() < default_error.mean()


def test_g1_native_trajectory_evidence_obeys_target_morphology(g1_kinematics: NewtonKinematics) -> None:
    """Native G1 evidence combines current source directions with target-owned geometry."""
    source = lafan_g1_29dof_skeleton()
    frame_count = 4
    root_translation = np.asarray(_G1_JOINT_Q[:3], dtype=np.float32)[None].repeat(frame_count, axis=0)
    root_translation[:, 0] += np.linspace(0.0, 0.15, frame_count, dtype=np.float32)
    pose_axis_angle = np.zeros((frame_count, source.num_bodies, 3), dtype=np.float32)
    pose_axis_angle[:, 0] = np.asarray((0.1, -0.2, 0.05), dtype=np.float32)
    pose_axis_angle[:, 1:] = (
        np.asarray(_G1_JOINT_Q[7:], dtype=np.float32)[None, :, None]
        * np.asarray(source.joint_axes, dtype=np.float32)[None]
    )
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)

    _assert_native_trajectory_evidence_obeys_target_morphology(
        source, target, LafanG1Clip(root_translation, pose_axis_angle, 30.0)
    )


def test_semantic_rotation_and_endpoint_targets_are_world_equivariant(
    g1_kinematics: NewtonKinematics,
) -> None:
    """A source world rotation left-multiplies every target rotation and rotates every target point."""
    source = lafan_g1_29dof_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    local_rotation = torch.zeros((3, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    root_position = torch.zeros((3, 3), dtype=torch.float32)
    projection = _trajectory_projection(source, target.trajectory_target)
    base = projection.generate_targets(root_position, local_rotation)
    world_rotation = quat_from_angle_axis(torch.tensor([torch.pi]), torch.tensor([[0.0, 0.0, 1.0]]))[0]
    rotated_local_rotation = local_rotation.clone()
    rotated_local_rotation[:, 0] = quat_mul(world_rotation.expand(3, 4), rotated_local_rotation[:, 0])
    rotated = projection.generate_targets(root_position, rotated_local_rotation)

    expected_rotation = quat_mul(
        world_rotation.expand_as(base.source_landmark_rotation_xyzw), base.source_landmark_rotation_xyzw
    )
    alignment = torch.abs(torch.sum(rotated.source_landmark_rotation_xyzw * expected_rotation, dim=-1))
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=2.0e-6, rtol=0.0)
    for base_point, rotated_point in (
        (base.source_landmark_position_m, rotated.source_landmark_position_m),
        (base.source_direction_point_position_m, rotated.source_direction_point_position_m),
    ):
        expected_point = quat_apply(
            world_rotation.expand(base_point.numel() // 3, 4), base_point.reshape(-1, 3)
        ).view_as(base_point)
        torch.testing.assert_close(rotated_point, expected_point, atol=2.0e-6, rtol=0.0)


def test_smpl_native_trajectory_evidence_obeys_target_morphology(smpl_kinematics: NewtonKinematics) -> None:
    """Native SMPL evidence combines current source directions with target-owned geometry."""
    source = cmu_humenv_smpl_skeleton()
    frame_count = 4
    qpos = np.zeros((frame_count, 7 + source.num_joints), dtype=np.float32)
    qpos[:, :3] = np.asarray((0.1, -0.2, 0.9), dtype=np.float32)
    qpos[:, 0] += np.linspace(0.0, 0.15, frame_count, dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.linspace(-0.25, 0.25, source.num_joints, dtype=np.float32)
    qvel = np.zeros((frame_count, 6 + source.num_joints), dtype=np.float32)
    target = smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)

    _assert_native_trajectory_evidence_obeys_target_morphology(source, target, CmuHumEnvSmplClip(qpos, qvel, 30.0))


def test_smpl_local_pose_retarget_tracks_root_and_projects_exact_limits(
    smpl_cross_projection: MotionSourceProjectionTrajectory, smpl_kinematics: NewtonKinematics
) -> None:
    """The production Newton-IK edge tracks source root evidence and publishes legal coordinates."""
    kinematics = smpl_kinematics
    projection = smpl_cross_projection
    source = projection.source_skeleton
    clip = _semantic_test_clip(
        source, "left_hip", np.asarray((0.0, 0.25, 0.25), dtype=np.float32), use_position_body=True
    )
    root_position = torch.tensor(((0.0, 0.0, 1.0), (0.01, 0.0, 1.0), (0.02, 0.0, 1.0)))
    clip = replace(clip, root_position_m=root_position)
    source_root_position, source_local_rotation = clip.local_pose(source, device="cpu")
    expected_root_xy = projection.target_projection.generate_targets(
        source_root_position, source_local_rotation
    ).source_landmark_position_m[0, :, :2]

    generalized_position, source_required_position_error_m, _, source_rotation_error_rad, _, _ = _semantic_output(
        projection, clip
    )
    tree = KinematicTree.from_newton(kinematics)

    root_tracking_error = torch.linalg.vector_norm(generalized_position[:, :2] - expected_root_xy, dim=-1)
    assert root_tracking_error.amax() <= _source_required_position_bound()
    assert torch.all(torch.isfinite(generalized_position))
    assert torch.all(torch.isfinite(source_required_position_error_m))
    assert torch.all(torch.isfinite(source_rotation_error_rad))
    assert source_required_position_error_m.amax() <= _source_required_position_bound()
    assert torch.all(tree.coordinates_within_limits(generalized_position[:, 7:]))
    assert torch.count_nonzero(generalized_position[:, 7:]) > 0


def _semantic_test_clip(
    source,
    role: str,
    angles_rad: np.ndarray,
    *,
    use_position_body: bool = False,
) -> _LocalPoseClip:
    """Build one rest-rooted local-pose clip with a single rotating semantic body."""
    landmarks = {landmark.name: landmark for landmark in source.landmarks}
    landmark = landmarks[role]
    body_name = landmark.position_body_name if use_position_body else landmark.rotation_body_name
    body = source.body_names.index(body_name)
    joint = source.joint_child_body_indices.index(body)
    frame_count = angles_rad.shape[0]
    local_rotation = torch.zeros((frame_count, source.num_bodies, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0
    root_body = source.parent_indices.index(-1)
    root_rest_rotation = convert_quat(torch.tensor(source.rest_rotation_wxyz[root_body]), to="xyzw")
    local_rotation[:, root_body] = root_rest_rotation
    axis = torch.tensor(source.joint_axes[joint], dtype=torch.float32).expand(frame_count, 3)
    local_rotation[:, body] = quat_from_angle_axis(torch.as_tensor(angles_rad), axis)
    root_position = torch.zeros((frame_count, 3), dtype=torch.float32)
    root_position[:, 2] = 0.8
    return _LocalPoseClip(root_position, local_rotation, 30.0)


def _target_coordinate_slice(tree: KinematicTree, body_name: str) -> slice:
    body = tree.body_names.index(body_name)
    joint = tree.joint_child_body_indices.index(body)
    start, stop = tree.joint_coordinate_ranges[joint]
    return slice(start, stop)


def _source_support_positions(source, clip, roles: tuple[str, ...]) -> torch.Tensor:
    """Return the source bodies selected by the target's declared support roles [m]."""
    root_position, body_rotation = clip.local_pose(source, device="cpu")
    position, _ = kinematic_pose_forward(
        torch.tensor(source.rest_translation_m),
        convert_quat(torch.tensor(source.rest_rotation_wxyz), to="xyzw"),
        body_rotation,
        root_position,
        source.parent_indices,
    )
    landmarks = {landmark.name: landmark for landmark in source.landmarks}
    support_indices = torch.tensor(tuple(source.body_names.index(landmarks[role].position_body_name) for role in roles))
    return position.index_select(1, support_indices)


def _target_support_positions(
    target: _SmplFrameTarget | _G1FrameTarget, generalized_position: torch.Tensor
) -> torch.Tensor:
    """Materialize the target's declared body-local collider support points [m]."""
    kinematics = target.kinematics
    layout = target.trajectory_target
    frame_count = generalized_position.shape[0]
    body_q = torch.empty(frame_count, kinematics.model.body_count, 7)
    body_qd = torch.empty(frame_count, kinematics.model.body_count, 6)
    kinematics.eval_fk_batched_torch(
        generalized_position,
        torch.zeros(frame_count, kinematics.model.joint_dof_count),
        body_q,
        body_qd,
    )
    support_pose = body_q.index_select(1, layout.support_body_indices)
    support_offset = layout.support_point_body_m.unsqueeze(0).expand(frame_count, -1, -1)
    return support_pose[..., :3] + quat_apply(
        support_pose[..., 3:7].reshape(-1, 4), support_offset.reshape(-1, 3)
    ).view_as(support_offset)


def test_g1_foot_landmarks_and_direction_points_share_ankle_roll_frames(
    g1_kinematics: NewtonKinematics,
) -> None:
    """G1 foot position and direction evidence use the SOMA/Proto ankle-roll frames."""
    layout = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES).trajectory_target
    position_bodies = {landmark.role: landmark.position_body_name for landmark in layout.landmarks}
    direction_bodies = {point.name: point.body_name for point in layout.direction_points}

    assert position_bodies["left_ankle"] == direction_bodies["left_foot"] == "left_ankle_roll_link"
    assert position_bodies["right_ankle"] == direction_bodies["right_foot"] == "right_ankle_roll_link"


def test_target_owned_hand_endpoint_uses_anatomical_wrist_forward(g1_kinematics: NewtonKinematics) -> None:
    """Raw-anatomy hand evidence uses the target length and position-derived wrist-forward law."""
    source = lafan_g1_29dof_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = _trajectory_projection(source, target.trajectory_target)
    clip = _semantic_test_clip(source, "left_elbow", np.asarray((0.0, 1.0, -1.0), dtype=np.float32))
    root_position, local_rotation = clip.local_pose(source, device="cpu")
    targets = projection.generate_targets(root_position, local_rotation)
    layout = target.trajectory_target
    hand_row = layout.direction_roles.index("left_hand")
    wrist_position_row = layout.roles.index("left_wrist")
    elbow_position_row = layout.roles.index("left_elbow")
    hand_point = targets.source_direction_point_position_m[hand_row]
    endpoint_offset = hand_point - targets.source_landmark_position_m[wrist_position_row]
    forearm = (
        targets.source_landmark_position_m[wrist_position_row] - targets.source_landmark_position_m[elbow_position_row]
    )
    endpoint_length = torch.linalg.vector_norm(endpoint_offset, dim=-1)
    forearm_length = torch.linalg.vector_norm(forearm, dim=-1)
    cosine = torch.sum(endpoint_offset * forearm, dim=-1) / (endpoint_length * forearm_length)

    assert layout.direction_points[hand_row].source_direction_law == "wrist_forward"
    assert layout.direction_points[hand_row].point_body_m == (0.0, 0.0, 0.14)
    assert hand_row not in targets.required_direction_rows
    torch.testing.assert_close(endpoint_length, torch.full_like(endpoint_length, 0.14), atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(cosine, torch.zeros_like(cosine), atol=2.0e-6, rtol=0.0)
    assert torch.linalg.vector_norm(endpoint_offset[1:] - endpoint_offset[:1], dim=-1).amin() > 0.01


def test_smpl_semantic_retarget_stays_continuous_and_respects_support_contract(
    smpl_cross_projection: MotionSourceProjectionTrajectory, smpl_kinematics: NewtonKinematics
) -> None:
    """A smooth one-sided ramp cannot select a distant unrelated-limb solution."""
    source = smpl_cross_projection.source_skeleton
    clip = _semantic_test_clip(source, "left_hip", np.linspace(0.0, 0.56, 9, dtype=np.float32), use_position_body=True)
    (
        generalized_position,
        source_required_position_error_m,
        source_required_distal_direction_error_rad,
        source_rotation_error_rad,
        _contact_patch_drift_m,
        ground_penetration_m,
    ) = _semantic_output(smpl_cross_projection, clip)
    tree = KinematicTree.from_newton(smpl_kinematics)
    coordinates = generalized_position[:, 7:]
    right_shoulder = coordinates[:, _target_coordinate_slice(tree, "R_Shoulder")]
    assert torch.max(torch.abs(torch.diff(right_shoulder, dim=0))) < 0.02
    assert torch.max(torch.abs(torch.diff(coordinates, n=2, dim=0))) / (1.0 / clip.source_fps) ** 2 < 80.0

    joint_q = generalized_position.clone()
    joint_q[:, 3:7] = convert_quat(generalized_position[:, 3:7], to="xyzw")
    _assert_exact_velocity_respects_model_limits(smpl_cross_projection.target, joint_q, clip)
    assert ground_penetration_m[0] <= _semantic_ground_penetration_bound()
    assert tuple(patch.channel for patch in smpl_cross_projection.target.contact_patches) == tuple(
        channel.name for channel in _CONTACT_CHANNELS
    )
    assert torch.all(torch.isfinite(source_required_position_error_m))
    assert torch.all(torch.isfinite(source_required_distal_direction_error_rad))
    assert torch.all(torch.isfinite(source_rotation_error_rad))
    assert source_required_position_error_m.amax() <= _source_required_position_bound()
    assert source_rotation_error_rad.amax() <= _source_root_rotation_bound()


def test_smpl_semantic_ramp_completes_ground_constrained_refinement(
    smpl_cross_projection: MotionSourceProjectionTrajectory,
) -> None:
    """A ground-aligned frame seed must not fail only at nonlinear globalization."""
    source = smpl_cross_projection.source_skeleton
    clip = _semantic_test_clip(
        source,
        "left_hip",
        np.linspace(0.0, 0.56, 9, dtype=np.float32),
        use_position_body=True,
    )
    candidate = _semantic_candidate(smpl_cross_projection, clip)

    assert candidate.constraint_geometry_feasible.item()
    assert candidate.inner_solve_converged.item()
    assert candidate.nonlinear_phases_converged.item()


def test_smpl_extreme_raw_lafan_pose_is_clamped_to_declared_coordinate_limits(
    smpl_cross_projection: MotionSourceProjectionTrajectory,
) -> None:
    """An infeasible raw pose leaves target bounds unconstrained but is legal after projection."""
    source = smpl_cross_projection.source_skeleton
    clip = _semantic_test_clip(
        source,
        "left_hip",
        np.linspace(0.0, 2.5, 9, dtype=np.float32),
        use_position_body=True,
    )
    root_position, local_rotation = clip.local_pose(source, device="cpu")
    targets = smpl_cross_projection.target_projection.generate_targets(root_position, local_rotation)
    target = smpl_cross_projection.target
    model = target.kinematics.model
    objectives = motion_objective_source_global_position(MotionSourceGlobalPositionObjectiveCfg(), targets)
    objectives += motion_objective_source_rotation(MotionSourceRotationObjectiveCfg(), targets)
    optimizer = ik.IKOptimizerLM(model, clip.frame_count, objectives, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    unconstrained = torch.empty_like(targets.initial_joint_q)
    optimizer.step(wp.from_torch(targets.initial_joint_q), wp.from_torch(unconstrained), iterations=50)

    coordinates = unconstrained.index_select(1, targets.coordinate_indices)
    violation = torch.maximum(
        targets.coordinate_lower_limits_rad - coordinates,
        coordinates - targets.coordinate_upper_limits_rad,
    ).clamp_min_(0.0)
    assert violation.amax() > 0.1

    lower = torch.full((model.joint_coord_count,), -torch.inf)
    upper = torch.full_like(lower, torch.inf)
    lower.index_copy_(0, targets.coordinate_indices, targets.coordinate_lower_limits_rad)
    upper.index_copy_(0, targets.coordinate_indices, targets.coordinate_upper_limits_rad)

    def project(joint_q_wp) -> None:
        joint_q = wp.to_torch(joint_q_wp)
        torch.maximum(joint_q, lower, out=joint_q)
        torch.minimum(joint_q, upper, out=joint_q)

    projected = torch.empty_like(targets.initial_joint_q)
    optimizer = ik.IKOptimizerLM(model, clip.frame_count, objectives, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    optimizer.solve(
        wp.from_torch(targets.initial_joint_q),
        wp.from_torch(projected),
        max_iterations=50,
        convergence_tolerance=None,
        projection=project,
    )
    projected_coordinates = projected.index_select(1, targets.coordinate_indices)
    assert torch.all(projected_coordinates >= targets.coordinate_lower_limits_rad - 1.0e-6)
    assert torch.all(projected_coordinates <= targets.coordinate_upper_limits_rad + 1.0e-6)
    assert torch.max(torch.abs(projected_coordinates - coordinates)) > 0.1


def test_trajectory_solver_runs_each_phase_to_convergence_or_cap() -> None:
    """Each trajectory phase uses the shared convergence policy and bounded iteration cap."""
    cfg = MotionTrajectorySolveCfg()

    assert not hasattr(cfg, "phases")
    assert cfg.max_iterations == 200
    assert cfg.convergence_tolerance == 1.0e-6
    assert cfg.convergence_check_interval == 1
    assert cfg.krylov_max_iterations == 128
    assert cfg.krylov_relative_tolerance == 1.0e-4
    assert cfg.joint_default_position_weight == 0.0025
    assert cfg.joint_temporal_velocity_weight == 1.0e-4
    assert cfg.joint_temporal_acceleration_weight == 1.0e-8
    assert cfg.joint_temporal_jerk_weight == 1.0e-8
    collision_objectives = tuple(
        objective for objective in cfg.objectives if isinstance(objective, IKObjectiveMeshCollisionCfg)
    )
    assert len(collision_objectives) == 1
    assert collision_objectives[0].weight == 5.0


def test_g1_and_smpl_source_projections_share_one_trajectory_implementation(
    g1_kinematics: NewtonKinematics,
    smpl_cross_projection: MotionSourceProjectionTrajectory,
) -> None:
    """Both robot targets parameterize the same source-to-target projection implementation."""
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    g1_cross_projection = g1_source_projection(
        cmu_humenv_smpl_skeleton(), target, object(), _CONTACT_CHANNELS, _contact_offsets(target)
    )

    assert type(g1_cross_projection.target_projection) is MotionTrajectoryProjection
    assert type(smpl_cross_projection.target_projection) is MotionTrajectoryProjection


def test_trajectory_targets_keep_stationary_source_support_independent_of_seed_fk(
    g1_kinematics: NewtonKinematics, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stationary source foot must not inherit root translation from the target-coordinate seed."""
    source = lafan_g1_29dof_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = _trajectory_projection(source, target.trajectory_target)
    frame_count = 3
    root_position = torch.zeros(frame_count, 3)
    root_position[:, 0] = torch.linspace(0.0, 0.08, frame_count)
    root_position[:, 2] = 0.9
    local_rotation = torch.zeros(frame_count, source.num_bodies, 4)
    local_rotation[..., 3] = 1.0

    original_forward = motion_retarget.kinematic_pose_forward
    source_by_name = {name: index for index, name in enumerate(source.body_names)}
    landmarks = {landmark.name: landmark for landmark in source.landmarks}
    position_indices = {
        role: source_by_name[landmarks[role].position_body_name] for role in target.trajectory_target.roles
    }
    left_toe_index = source_by_name[landmarks["left_toe"].position_body_name]
    target_lengths = target.trajectory_target.segment_lengths_m
    source_pose, _ = original_forward(
        torch.tensor(source.rest_translation_m),
        convert_quat(torch.tensor(source.rest_rotation_wxyz), to="xyzw"),
        local_rotation,
        root_position,
        source.parent_indices,
    )
    desired = torch.empty(len(target.trajectory_target.roles), frame_count, 3)
    target_root = root_position
    desired[0].copy_(target_root)
    for row, parent in enumerate(target.trajectory_target.parent_rows[1:], start=1):
        source_edge = source_pose[:, position_indices[target.trajectory_target.roles[row]]]
        source_edge = source_edge - source_pose[:, position_indices[target.trajectory_target.roles[parent]]]
        source_edge = torch.nn.functional.normalize(source_edge)
        desired[row].copy_(desired[parent] + target_lengths[row] * source_edge)

    role_rows = {role: row for row, role in enumerate(target.trajectory_target.roles)}
    hip_row = role_rows["left_hip"]
    knee_row = role_rows["left_knee"]
    ankle_row = role_rows["left_ankle"]
    root_to_hip = desired[hip_row] - desired[0]
    fixed_ankle = desired[hip_row, 0].clone()
    fixed_ankle[2] -= 0.8 * (target_lengths[knee_row] + target_lengths[ankle_row])
    for frame in range(frame_count):
        hip = target_root[frame] + root_to_hip[0]
        hip_to_ankle = fixed_ankle - hip
        distance = torch.linalg.vector_norm(hip_to_ankle)
        direction = hip_to_ankle / distance
        perpendicular = torch.stack((-direction[2], direction.new_zeros(()), direction[0]))
        upper = target_lengths[knee_row]
        lower = target_lengths[ankle_row]
        along = (upper.square() - lower.square() + distance.square()) / (2.0 * distance)
        height = torch.sqrt(upper.square() - along.square())
        desired[hip_row, frame] = hip
        desired[knee_row, frame] = hip + along * direction + height * perpendicular
        desired[ankle_row, frame] = fixed_ankle

    def stationary_support_forward(*args, **kwargs):
        position, rotation = original_forward(*args, **kwargs)
        for row, role in enumerate(target.trajectory_target.roles):
            position[:, position_indices[role]].copy_(desired[row])
        position[:, left_toe_index].copy_(fixed_ankle + fixed_ankle.new_tensor((0.04, 0.0, -0.05)))
        return position, rotation

    monkeypatch.setattr(motion_retarget, "kinematic_pose_forward", stationary_support_forward)
    targets = projection.generate_targets(root_position, local_rotation)

    ground_alignment = targets.source_landmark_position_m[0] - desired[0]
    torch.testing.assert_close(targets.source_landmark_position_m, desired + ground_alignment, atol=2.0e-6, rtol=0.0)
    assert torch.all(torch.isfinite(targets.target_support_position_m))
    layout = target.trajectory_target
    for start, stop in zip(layout.support_patch_offsets[:-1], layout.support_patch_offsets[1:], strict=True):
        local_points = layout.support_point_body_m[start:stop]
        local_distances = torch.cdist(local_points, local_points)
        world_patch = targets.target_support_position_m[start:stop].transpose(0, 1)
        torch.testing.assert_close(
            torch.cdist(world_patch, world_patch),
            local_distances.expand(frame_count, -1, -1),
            atol=2.0e-6,
            rtol=0.0,
        )
    source_support_displacement = (
        targets.source_contact_probe_position_m[:1, :, :2] - targets.source_contact_probe_position_m[:1, :1, :2]
    )
    target_support_displacement = (
        targets.target_support_position_m[:1, :, :2] - targets.target_support_position_m[:1, :1, :2]
    )
    support_point_root = layout.support_point_root_m[:, None].expand(-1, frame_count, -1)
    root_rotation = targets.initial_joint_q[:, 3:7][None].expand(len(support_point_root), -1, -1)
    expected_target_support = targets.initial_joint_q[None, :, :3] + quat_apply(
        root_rotation.reshape(-1, 4), support_point_root.reshape(-1, 3)
    ).view_as(support_point_root)
    torch.testing.assert_close(targets.target_support_position_m, expected_target_support, atol=2.0e-6, rtol=0.0)
    assert torch.linalg.vector_norm(source_support_displacement, dim=-1).amax() < 1.0e-6
    assert torch.linalg.vector_norm(target_support_displacement, dim=-1).amax() > 0.04

    twisted_rotation = local_rotation.clone()
    angle = torch.linspace(-1.0, 1.0, frame_count)
    twisted_rotation[:, position_indices["left_hip"], 0] = torch.sin(0.5 * angle)
    twisted_rotation[:, position_indices["left_hip"], 3] = torch.cos(0.5 * angle)
    twisted_targets = projection.generate_targets(root_position, twisted_rotation)
    parent_rows = targets.parent_row_tensor[1:]
    source_edges = targets.source_landmark_position_m[1:] - targets.source_landmark_position_m.index_select(
        0, parent_rows
    )
    twisted_source_edges = twisted_targets.source_landmark_position_m[
        1:
    ] - twisted_targets.source_landmark_position_m.index_select(0, parent_rows)
    torch.testing.assert_close(twisted_source_edges, source_edges, atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(twisted_targets.initial_joint_q[:, 7:], targets.initial_joint_q[:, 7:])


def test_trajectory_targets_use_source_world_direction_and_target_length_when_bind_directions_differ(
    g1_kinematics: NewtonKinematics,
) -> None:
    """Canonical source direction and robot-owned length remain independent under adversarial bind geometry."""
    source = lafan_g1_29dof_skeleton()
    landmarks = {landmark.name: landmark for landmark in source.landmarks}
    body_by_name = {name: index for index, name in enumerate(source.body_names)}
    wrist_body = body_by_name[landmarks["left_wrist"].position_body_name]
    rest_translation = list(source.rest_translation_m)
    rest_translation[wrist_body] = tuple(-value for value in rest_translation[wrist_body])
    source = replace(source, rest_translation_m=tuple(rest_translation))

    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = _trajectory_projection(source, target.trajectory_target)
    root_position = torch.tensor(g1_kinematics.default_joint_q[:3], dtype=torch.float32).unsqueeze(0)
    local_rotation = torch.zeros(1, source.num_bodies, 4)
    local_rotation[..., 3] = 1.0
    local_rotation[:, 0] = convert_quat(torch.tensor(source.rest_rotation_wxyz[0]), to="xyzw")

    targets = projection.generate_targets(root_position, local_rotation)
    source_position, _ = kinematic_pose_forward(
        torch.tensor(source.rest_translation_m),
        convert_quat(torch.tensor(source.rest_rotation_wxyz), to="xyzw"),
        local_rotation,
        root_position,
        source.parent_indices,
    )
    roles = target.trajectory_target.roles
    wrist_row = roles.index("left_wrist")
    elbow_row = targets.parent_rows[wrist_row]
    source_edge = (
        source_position[:, wrist_body] - source_position[:, body_by_name[landmarks["left_elbow"].position_body_name]]
    )
    target_edge = targets.source_landmark_position_m[wrist_row] - targets.source_landmark_position_m[elbow_row]
    source_direction = torch.nn.functional.normalize(source_edge, dim=-1)
    target_direction = torch.nn.functional.normalize(target_edge, dim=-1)
    torch.testing.assert_close(target_direction, source_direction, atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(target_edge, dim=-1),
        target.trajectory_target.segment_lengths_m[wrist_row].reshape(1),
        atol=2.0e-6,
        rtol=0.0,
    )


def test_trajectory_initializer_is_limit_valid_but_does_not_own_support_targets(
    g1_kinematics: NewtonKinematics,
) -> None:
    """The soft coordinate prior is limit-valid but generated support geometry remains independent."""
    source = lafan_g1_29dof_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = _trajectory_projection(source, target.trajectory_target)
    root_position = torch.tensor(g1_kinematics.default_joint_q[:3], dtype=torch.float32).unsqueeze(0)
    local_rotation = torch.zeros(1, source.num_bodies, 4)
    local_rotation[..., 3] = 1.0
    local_rotation[:, 0].copy_(projection._source_rest_rotation_xyzw[0])

    targets = projection.generate_targets(root_position, local_rotation)
    reference_coordinates = targets.initial_joint_q.index_select(1, targets.coordinate_indices)
    assert torch.all(reference_coordinates >= targets.coordinate_lower_limits_rad)
    assert torch.all(reference_coordinates <= targets.coordinate_upper_limits_rad)

    support_point_root = target.trajectory_target.support_point_root_m[:, None]
    root_rotation = targets.initial_joint_q[:, 3:7][None].expand(len(support_point_root), -1, -1)
    expected_target_support = targets.initial_joint_q[None, :, :3] + quat_apply(
        root_rotation.reshape(-1, 4), support_point_root.reshape(-1, 3)
    ).view_as(support_point_root)
    torch.testing.assert_close(
        targets.target_support_position_m,
        expected_target_support,
        atol=2.0e-6,
        rtol=0.0,
    )


def test_smpl_source_refinement_freezes_authored_root_and_rolls_back_rejection(
    smpl_cross_projection: MotionSourceProjectionTrajectory,
) -> None:
    """Fixed-root SMPL source fitting cannot move the authored root toward rejected evidence."""
    source = smpl_cross_projection.source_skeleton
    clip = _semantic_test_clip(
        source,
        "left_hip",
        np.zeros(3, dtype=np.float32),
        use_position_body=True,
    )
    root_position, local_rotation = clip.local_pose(source, device="cpu")
    targets = smpl_cross_projection.target_projection.generate_targets(root_position, local_rotation)
    initial_root_x = targets.initial_joint_q[:, 0].clone()
    targets.source_landmark_position_m[..., 0].add_(0.2)
    targets.source_contact_probe_position_m[..., 0].add_(torch.arange(3, dtype=torch.float32))
    candidate = motion_solve_trajectory(
        MotionTrajectorySolveCfg(),
        _MotionTrajectoryTargetCandidate(
            target=smpl_cross_projection.target,
            clip_index=_clip_index(clip.frame_count, clip.source_fps),
            pending=iter((targets,)),
            source_body_counts=(source.num_bodies,),
            device="cpu",
            inspection=True,
        ),
    )

    assert candidate.view_evidence is not None
    stage_quality = candidate.view_evidence.stage_quality
    assert stage_quality.shape == (1, 5, len(_TRAJECTORY_METRIC_NAMES))
    assert torch.isfinite(stage_quality[:, (0, 1, 3), :4]).all()
    assert torch.isnan(stage_quality[:, 2]).all()
    assert torch.isnan(stage_quality[:, 4:]).all()
    assert not torch.equal(stage_quality[:, 0], stage_quality[:, 3])
    assert not torch.allclose(
        candidate.view_evidence.solved_robot_landmarks,
        candidate.view_evidence.target_landmarks,
    )
    solved_root_x = candidate.coordinates.joint_q[:, 0]
    target_root_x = targets.source_landmark_position_m[0, :, 0]
    torch.testing.assert_close(solved_root_x, initial_root_x, rtol=0.0, atol=0.0)
    assert torch.mean(torch.abs(solved_root_x - target_root_x)) >= 0.19
    assert candidate.nonlinear_refinement_required[0]
    assert not candidate.nonlinear_phases_converged[0]


def test_trajectory_phases_certify_clean_seed_and_rollback_rejected_clips(
    smpl_cross_projection: MotionSourceProjectionTrajectory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source and contact refinement commit only hard-accepted clips without mutating neighbors."""
    source = smpl_cross_projection.source_skeleton
    frame_count = 5
    source_fps = 30.0
    static_clip = _semantic_test_clip(
        source,
        "left_hip",
        np.zeros(frame_count, dtype=np.float32),
        use_position_body=True,
    )

    def targets_from_clip(clip: _LocalPoseClip) -> MotionTrajectoryTargets:
        root_position, local_rotation = clip.local_pose(source, device="cpu")
        return smpl_cross_projection.target_projection.generate_targets(root_position, local_rotation)

    clean_targets = targets_from_clip(static_clip)
    fast_probe_x = torch.arange(frame_count, dtype=torch.float32).unsqueeze(0)
    clean_targets.source_contact_probe_position_m[:, :, 0].add_(fast_probe_x)

    source_dirty_targets = targets_from_clip(static_clip)
    source_dirty_targets.source_contact_probe_position_m[:, :, 0].add_(fast_probe_x)
    source_dirty_targets.source_direction_point_position_m[0].copy_(
        source_dirty_targets.source_landmark_position_m[source_dirty_targets.direction_position_rows[0]]
    )

    moving_root = torch.zeros((frame_count, 3), dtype=torch.float32)
    moving_root[:, 0] = torch.linspace(0.0, 1.0, frame_count)
    moving_root[:, 2] = 0.8
    moving_clip = replace(static_clip, root_position_m=moving_root, source_fps=source_fps)
    contact_dirty_targets = targets_from_clip(moving_clip)
    planted_probe = contact_dirty_targets.source_contact_probe_position_m[:, :1].clone()
    contact_dirty_targets.source_contact_probe_position_m[:, 1:].copy_(planted_probe.expand(-1, frame_count - 1, -1))

    clip_index = MotionClipIndex(
        source_content_sha256="0" * 64,
        skeleton_identity_sha256s=("2" * 64,),
        clips=tuple(
            MotionClipIndex.Clip(
                clip_id=f"transaction_{index}",
                frame_count=frame_count,
                source_fps=source_fps,
                content_sha256=str(index + 3) * 64,
                skeleton_id=0,
            )
            for index in range(3)
        ),
    )
    cfg = MotionTrajectorySolveCfg(
        max_iterations=1,
        acceptance=MotionTrajectorySolveCfg.AcceptanceCfg(
            source=MotionTrajectorySolveCfg.AcceptanceCfg.SourceCfg(
                required_position_upper_m=10.0,
                required_distal_position_upper_m=10.0,
                required_distal_direction_upper_rad=3.0,
                root_rotation_upper_rad=3.0,
            ),
            contact=MotionTrajectorySolveCfg.AcceptanceCfg.ContactCfg(require_any_stable_contact=False),
        ),
    )
    solve_calls: list[torch.Tensor] = []

    def reject_active_segments(
        _solver,
        _joint_q,
        joint_q_out,
        segment_offsets,
        _step_seconds,
        _pose_weights,
        _temporal_weights,
        **kwargs,
    ) -> None:
        segment_active = kwargs["segment_active"]
        solve_calls.append(segment_active.clone())
        for segment in torch.nonzero(segment_active, as_tuple=False).flatten().tolist():
            start = int(segment_offsets[segment])
            stop = int(segment_offsets[segment + 1])
            joint_q_out[start:stop, 0].add_(100.0)
        kwargs["segment_feasible"].fill_(True)
        kwargs["segment_direction_valid"].fill_(True)
        kwargs["segment_globalization_succeeded"].fill_(True)
        residual_constraints_satisfied = kwargs["segment_residual_constraints_satisfied"]
        if residual_constraints_satisfied is not None:
            residual_constraints_satisfied.fill_(True)
        segment_active.zero_()

    monkeypatch.setattr(IKTrajectorySolver, "solve", reject_active_segments)
    candidate = motion_solve_trajectory(
        cfg,
        _MotionTrajectoryTargetCandidate(
            target=smpl_cross_projection.target,
            clip_index=clip_index,
            pending=iter((clean_targets, source_dirty_targets, contact_dirty_targets)),
            source_body_counts=(source.num_bodies,) * 3,
            device="cpu",
            inspection=True,
        ),
    )

    assert candidate.view_evidence is not None
    seed_quality = candidate.view_evidence.stage_quality[:, 0]
    source_attempt_quality = candidate.view_evidence.stage_quality[:, 1]
    physical_attempt_quality = candidate.view_evidence.stage_quality[:, 2]
    post_source_quality = candidate.view_evidence.stage_quality[:, 3]
    contact_attempt_quality = candidate.view_evidence.stage_quality[:, 4]
    source_accepted = _motion_source_fidelity_accepted(cfg.acceptance.source, seed_quality)
    contact_accepted = _motion_contact_rows_accepted(cfg.acceptance.contact, seed_quality)
    torch.testing.assert_close(source_accepted, torch.tensor((True, False, True)))
    torch.testing.assert_close(contact_accepted, torch.tensor((True, True, False)))
    assert len(solve_calls) == 3
    torch.testing.assert_close(solve_calls[0], torch.tensor((1, 1, 1), dtype=torch.int32))
    torch.testing.assert_close(solve_calls[1], torch.tensor((1, 0, 1), dtype=torch.int32))
    torch.testing.assert_close(solve_calls[2], torch.tensor((1, 0, 1), dtype=torch.int32))
    torch.testing.assert_close(candidate.nonlinear_refinement_required, torch.tensor((True, True, True)))
    torch.testing.assert_close(candidate.nonlinear_phases_converged, torch.tensor((True, False, False)))
    assert torch.isfinite(physical_attempt_quality[[0, 2], 0]).all()
    assert torch.isnan(physical_attempt_quality[1]).all()
    assert torch.isfinite(source_attempt_quality[:, 0]).all()
    assert torch.isnan(contact_attempt_quality[1]).all()
    torch.testing.assert_close(source_attempt_quality[1, 0], post_source_quality[1, 0])
    torch.testing.assert_close(contact_attempt_quality[2, 0], candidate.trajectory_quality[2, 0])
    contact_start = _TRAJECTORY_METRIC_NAMES.index("contact_gap_max_m")
    torch.testing.assert_close(
        post_source_quality[:, :contact_start], seed_quality[:, :contact_start], rtol=0.0, atol=1.0e-7, equal_nan=True
    )
    torch.testing.assert_close(
        _motion_source_fidelity_accepted(cfg.acceptance.source, candidate.trajectory_quality), source_accepted
    )
    torch.testing.assert_close(
        _motion_contact_rows_accepted(cfg.acceptance.contact, candidate.trajectory_quality), contact_accepted
    )


def test_later_phase_keeps_healthy_witness_and_retries_only_solver_failures(
    smpl_cross_projection: MotionSourceProjectionTrajectory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healthy intermediate q is retained while failed solves retry and final rejection rolls back."""
    source = smpl_cross_projection.source_skeleton
    frame_count = 5
    source_fps = 30.0
    clip = _semantic_test_clip(
        source,
        "left_hip",
        np.zeros(frame_count, dtype=np.float32),
        use_position_body=True,
    )

    def make_targets() -> MotionTrajectoryTargets:
        root_position, local_rotation = clip.local_pose(source, device="cpu")
        targets = smpl_cross_projection.target_projection.generate_targets(root_position, local_rotation)
        targets.source_contact_probe_position_m[:, :, 0].add_(torch.arange(frame_count, dtype=torch.float32))
        return targets

    clip_index = MotionClipIndex(
        source_content_sha256="0" * 64,
        skeleton_identity_sha256s=("7" * 64,),
        clips=tuple(
            MotionClipIndex.Clip(
                clip_id=f"adaptive_retry_{index}",
                frame_count=frame_count,
                source_fps=source_fps,
                content_sha256=str(index + 8) * 64,
                skeleton_id=0,
            )
            for index in range(2)
        ),
    )
    cfg = MotionTrajectorySolveCfg(
        max_iterations=4,
        acceptance=MotionTrajectorySolveCfg.AcceptanceCfg(
            source=MotionTrajectorySolveCfg.AcceptanceCfg.SourceCfg(
                required_position_upper_m=10.0,
                required_distal_position_upper_m=10.0,
                required_distal_direction_upper_rad=math.nextafter(math.pi, 0.0),
                root_rotation_upper_rad=math.nextafter(math.pi, 0.0),
            ),
            contact=MotionTrajectorySolveCfg.AcceptanceCfg.ContactCfg(require_any_stable_contact=False),
        ),
    )
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    physical_baseline: torch.Tensor | None = None

    def propose(
        _solver,
        _joint_q,
        joint_q_out,
        segment_offsets,
        _step_seconds,
        _pose_weights,
        _temporal_weights,
        **kwargs,
    ) -> None:
        nonlocal physical_baseline
        active = kwargs["segment_active"]
        damping = kwargs["segment_damping"]
        calls.append((active.clone(), damping.clone()))
        kwargs["segment_feasible"].fill_(True)
        kwargs["segment_direction_valid"].fill_(True)
        kwargs["segment_globalization_succeeded"].fill_(True)
        residual = kwargs["segment_residual_constraints_satisfied"]
        if residual is not None:
            residual.fill_(True)

        call = len(calls) - 1
        first_stop = int(segment_offsets[1])
        second_stop = int(segment_offsets[2])
        if call == 1:
            physical_baseline = joint_q_out.clone()
            joint_q_out[:second_stop, 7].fill_(float("nan"))
            active[1] = 0
        elif call in (2, 3, 4):
            assert physical_baseline is not None
            assert torch.isnan(joint_q_out[:first_stop, 7]).all()
            if call == 2:
                kwargs["segment_globalization_succeeded"][0] = False
            elif call == 3:
                kwargs["segment_direction_valid"][0] = False
            else:
                joint_q_out[:first_stop].copy_(physical_baseline[:first_stop])
                active[0] = 0

    monkeypatch.setattr(IKTrajectorySolver, "solve", propose)
    candidate = motion_solve_trajectory(
        cfg,
        _MotionTrajectoryTargetCandidate(
            target=smpl_cross_projection.target,
            clip_index=clip_index,
            pending=iter((make_targets(), make_targets())),
            source_body_counts=(source.num_bodies,) * 2,
            device="cpu",
            inspection=True,
        ),
    )

    assert len(calls) == 6
    torch.testing.assert_close(calls[0][0], torch.tensor((1, 1), dtype=torch.int32))
    torch.testing.assert_close(calls[1][0], torch.tensor((1, 1), dtype=torch.int32))
    for call in (2, 3, 4):
        torch.testing.assert_close(calls[call][0], torch.tensor((1, 0), dtype=torch.int32))
    torch.testing.assert_close(calls[0][1], torch.full((2,), cfg.damping))
    torch.testing.assert_close(calls[1][1], torch.full((2,), cfg.damping))
    torch.testing.assert_close(calls[2][1], torch.full((2,), cfg.damping))
    torch.testing.assert_close(calls[3][1], torch.tensor((2.0 * cfg.damping, cfg.damping)))
    torch.testing.assert_close(calls[4][1], torch.tensor((4.0 * cfg.damping, cfg.damping)))
    torch.testing.assert_close(calls[5][1], torch.full((2,), cfg.damping))
    torch.testing.assert_close(candidate.nonlinear_phases_converged, torch.tensor((True, False)))

    assert physical_baseline is not None
    torch.testing.assert_close(
        candidate.coordinates.joint_q[frame_count:, 7:],
        physical_baseline[frame_count:, 7:],
        rtol=0.0,
        atol=0.0,
    )
    assert candidate.view_evidence is not None
    physical_attempt = candidate.view_evidence.stage_quality[:, 2, _METRIC_SOURCE_REQUIRED_POSITION]
    assert not torch.isfinite(physical_attempt[1])
    assert torch.isfinite(candidate.trajectory_quality[1, _METRIC_SOURCE_REQUIRED_POSITION])


def test_g1_exact_profile_matches_frozen_bfm_fk_and_analytic_velocity(g1_kinematics: NewtonKinematics) -> None:
    """The G1 target and projection match frozen external FK and analytic rigid translation."""
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
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = g1_source_projection(source, target, object(), _CONTACT_CHANNELS, _contact_offsets(target))
    joint_q, joint_qd = clip.free_root_coordinates(source, device="cpu")

    assert isinstance(projection, MotionSourceProjectionExact)
    coordinates = projection.convert_coordinates(joint_q, joint_qd, clip.source_fps)
    frames = target.materialize_coordinates(coordinates, _clip_index(clip.frame_count, clip.source_fps))

    body_indices = torch.tensor([G1_SIMULATOR_BODY_NAMES.index(_G1_BODY_NAMES[index]) for index in _G1_BODY_INDICES])
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


def test_g1_builder_maps_reference_coordinates_to_the_live_articulation_axes(
    g1_kinematics: NewtonKinematics,
) -> None:
    """The table boundary must expose exact frames in non-identity simulator order."""
    source = lafan_g1_29dof_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = g1_source_projection(source, target, object(), _CONTACT_CHANNELS, _contact_offsets(target))
    reference_joint_names = tuple(g1_kinematics.joint_names[1:])
    reference_body_names = tuple(g1_kinematics.body_names)
    joint_permutation = torch.tensor([reference_joint_names.index(name) for name in G1_SIMULATOR_JOINT_NAMES])
    body_permutation = torch.tensor([reference_body_names.index(name) for name in G1_SIMULATOR_BODY_NAMES])

    assert tuple(joint_permutation.tolist()) != tuple(range(29))
    assert tuple(body_permutation.tolist()) != tuple(range(30))
    assert target.joint_names == G1_SIMULATOR_JOINT_NAMES
    assert target.reference_frame_names == (*G1_SIMULATOR_BODY_NAMES, G1_HEAD_FRAME_NAME)
    expected_joint_q_indices = tuple(7 + index for index in joint_permutation.tolist())
    assert target.joint_q_indices == expected_joint_q_indices
    assert target.joint_q_indices != tuple(range(7, 36))

    joint_q = torch.zeros(3, g1_kinematics.model.joint_coord_count)
    joint_q[:, 6] = 1.0
    joint_q[:, 7:] = torch.arange(29, dtype=torch.float32)
    assert isinstance(projection, MotionSourceProjectionExact)
    coordinates = projection.convert_coordinates(joint_q, None, 30.0)
    frames = target.materialize_coordinates(coordinates, _clip_index(3, 30.0))
    body_q = torch.empty(3, g1_kinematics.model.body_count, 7)
    body_qd = torch.empty(3, g1_kinematics.model.body_count, 6)
    g1_kinematics.eval_fk_batched_torch(
        joint_q,
        torch.zeros(3, g1_kinematics.model.joint_dof_count),
        body_q,
        body_qd,
    )

    torch.testing.assert_close(frames.joint_position, joint_q[:, 7:].index_select(1, joint_permutation))
    view = _motion_task_view(
        _clip_index(3, 30.0), frames, target, torch.ones((1, len(_QUALITY_NAMES)), dtype=torch.float32), None
    )
    torch.testing.assert_close(
        view.kinematic_view.joint_q_indices,
        torch.tensor(expected_joint_q_indices, dtype=torch.int64),
    )
    reconstructed = torch.tensor(g1_kinematics.default_joint_q, dtype=torch.float32)
    reconstructed.index_copy_(0, view.kinematic_view.joint_q_indices, frames.joint_position[0])
    torch.testing.assert_close(reconstructed[7:], joint_q[0, 7:])
    torch.testing.assert_close(frames.body_position[:, :-1], body_q[..., :3].index_select(1, body_permutation))
    torch.testing.assert_close(frames.body_rotation[:, :-1], body_q[..., 3:].index_select(1, body_permutation))


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
    target = smpl_frame_target(smpl_kinematics, _SMPL_CONTACT_PATCHES)
    projection = smpl_source_projection(source, target, object(), _CONTACT_CHANNELS, _contact_offsets(target))
    joint_q, joint_qd = clip.free_root_coordinates(source, device="cpu")

    assert isinstance(projection, MotionSourceProjectionExact)
    coordinates = projection.convert_coordinates(joint_q, joint_qd, clip.source_fps)
    frames = target.materialize_coordinates(coordinates, _clip_index(clip.frame_count, clip.source_fps))

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
    live_names = smpl_live_joint_mujoco_names(target.joint_names)
    live_indices = torch.tensor([source.joint_names.index(name) for name in live_names])

    torch.testing.assert_close(frames.joint_position, source_coordinates[live_indices].expand(frame_count, -1))
    torch.testing.assert_close(frames.body_position, body_position, rtol=0.0, atol=2.0e-6)
    sign = torch.where((frames.body_rotation * body_rotation).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    torch.testing.assert_close(frames.body_rotation, sign * body_rotation, rtol=0.0, atol=2.0e-6)
    torch.testing.assert_close(frames.body_linear_velocity, root_velocity.expand_as(frames.body_linear_velocity))
    torch.testing.assert_close(frames.body_angular_velocity, torch.zeros_like(frames.body_angular_velocity))
    torch.testing.assert_close(frames.joint_velocity, torch.zeros_like(frames.joint_velocity))


def test_smpl_semantic_retarget_uses_declared_d6_tangent_derivative(
    smpl_cross_projection: MotionSourceProjectionTrajectory,
) -> None:
    """Cross-source SMPL velocities follow ordered D6 angular-velocity coordinates."""
    source = smpl_cross_projection.source_skeleton
    clip = _semantic_test_clip(source, "left_knee", np.asarray((0.0, 0.1, 0.3, 0.6, 1.0), dtype=np.float32))
    candidate = _semantic_candidate(smpl_cross_projection, clip)
    clip_index = MotionClipIndex(
        source_content_sha256="0" * 64,
        skeleton_identity_sha256s=("2" * 64,),
        clips=(
            MotionClipIndex.Clip(
                clip_id="test",
                frame_count=clip.frame_count,
                source_fps=clip.source_fps,
                content_sha256="1" * 64,
                skeleton_id=0,
            ),
        ),
    )
    target = smpl_cross_projection.target
    frames = target.materialize_coordinates(candidate.coordinates, clip_index)
    coordinates = candidate.coordinates.joint_q[:, 7:].view(clip.frame_count, -1, 3).to(torch.float64)
    axes = torch.tensor(target.kinematic_tree.coordinate_axes, dtype=coordinates.dtype).view(-1, 3, 3)
    local_rotation = ordered_hinge_rotation(coordinates, axes)
    angular_velocity = time_quaternion_angular_velocity(local_rotation.unsqueeze(0), 1.0 / clip.source_fps).squeeze(0)
    expected = ordered_hinge_coordinate_velocity(coordinates, axes, angular_velocity).flatten(1)
    expected[-1].copy_(expected[-2])
    live_reference_names = smpl_live_joint_mujoco_names(target.joint_names)
    live_from_reference = torch.tensor(
        tuple(target.reference_coordinate_names.index(name) for name in live_reference_names), dtype=torch.int64
    )
    expected_live = expected.index_select(1, live_from_reference).to(frames.joint_velocity.dtype)
    torch.testing.assert_close(frames.joint_velocity, expected_live)


def test_g1_semantic_retarget_uses_source_roles_and_support_contract(
    g1_kinematics: NewtonKinematics,
) -> None:
    """CMU-to-G1 composition uses semantic world poses and corrects target morphology height."""
    source = cmu_humenv_smpl_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = g1_source_projection(source, target, object(), _CONTACT_CHANNELS, _contact_offsets(target))
    frame_count = 9
    qpos = np.zeros((frame_count, 76), dtype=np.float32)
    qpos[:, 2] = 0.9
    qpos[:, 3:5] = np.sqrt(0.5)
    source_landmarks = {landmark.name: landmark for landmark in source.landmarks}
    hip_body = source.body_names.index(source_landmarks["left_hip"].position_body_name)
    qpos[:, 7 + 3 * (hip_body - 1) + 1] = np.linspace(0.0, 0.56, frame_count, dtype=np.float32)
    clip = CmuHumEnvSmplClip(qpos, np.zeros((frame_count, 75), dtype=np.float32), 30.0)
    candidate = _semantic_candidate(projection, clip)
    joint_q = candidate.coordinates.joint_q
    _assert_exact_velocity_respects_model_limits(target, joint_q, clip)
    source_required_position_error_m = candidate.trajectory_quality[:, _METRIC_SOURCE_REQUIRED_POSITION]
    source_required_distal_direction_error_rad = candidate.trajectory_quality[
        :, _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION
    ]
    source_rotation_error_rad = candidate.trajectory_quality[:, _METRIC_SOURCE_ROOT_ROTATION]

    source_support = _source_support_positions(source, clip, _CONTACT_PROBE_ROLES)
    target_support = _target_support_positions(target, joint_q)
    source_height = source_support[..., 2].amin(dim=1)
    target_height = target_support[..., 2].amin(dim=1)
    assert torch.max(torch.abs(target_height - source_height)) <= _contact_gap_bound()
    assert tuple(patch.channel for patch in target.contact_patches) == tuple(
        channel.name for channel in _CONTACT_CHANNELS
    )

    assert torch.all(torch.isfinite(source_required_position_error_m))
    assert torch.all(torch.isfinite(source_required_distal_direction_error_rad))
    assert torch.all(torch.isfinite(source_rotation_error_rad))
    assert source_required_position_error_m.amax() <= _source_required_position_bound()
    assert source_rotation_error_rad.amax() <= _source_root_rotation_bound()
    assert candidate.target_coordinate_evidence[0, _TARGET_GROUND_PENETRATION] <= _semantic_ground_penetration_bound()


def test_g1_lie_down_pose_is_rejected_by_contact_geometry_contract(g1_kinematics: NewtonKinematics) -> None:
    """A lying CMU pose clears the ground but cannot claim geometrically valid toe contact."""
    source = cmu_humenv_smpl_skeleton()
    target = g1_frame_target(g1_kinematics, _G1_CONTACT_PATCHES)
    projection = g1_source_projection(source, target, object(), _CONTACT_CHANNELS, _contact_offsets(target))
    qpos = np.zeros((9, 76), dtype=np.float32)
    qpos[:, 2] = 0.9
    qpos[:, 3] = 1.0
    clip = CmuHumEnvSmplClip(qpos, np.zeros((9, 75), dtype=np.float32), 30.0)
    candidate = _semantic_candidate(projection, clip)

    assert candidate.constraint_geometry_feasible[0]
    assert candidate.inner_solve_converged[0]
    assert candidate.nonlinear_phases_converged[0]
    assert candidate.target_coordinate_evidence[0, _TARGET_GROUND_PENETRATION] <= _semantic_ground_penetration_bound()
    assert candidate.trajectory_quality[0, _METRIC_CONTACT_GAP] > _contact_gap_bound()
    assert candidate.trajectory_quality[0, _METRIC_CONTACT_TILT] > _contact_tilt_bound()


def test_landmark_position_objective_is_uniform_scale_invariant(tmp_path: Path) -> None:
    """Uniformly scaling target geometry must preserve the dimensionless objective weighting."""
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
        desired_q = torch.tensor(kinematics.default_joint_q, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
        desired_q[:, tree.coordinate_q_indices[0]] = 0.7
        desired_q[:, tree.coordinate_q_indices[1]] = -0.4
        desired_body_q = torch.empty(2, 3, 7)
        desired_body_qd = torch.empty(2, 3, 6)
        kinematics.eval_fk_batched_torch(
            desired_q,
            torch.zeros(2, kinematics.model.joint_dof_count),
            desired_body_q,
            desired_body_qd,
        )
        body_indices = (0, 1, 2)
        body_index_tensor = torch.tensor(body_indices)
        target_position = desired_body_q[..., :3].transpose(0, 1).contiguous()
        target_rotation = torch.zeros(2, 4)
        target_rotation[..., 3] = 1.0
        targets = MotionTrajectoryTargets(
            position_body_indices=body_indices,
            root_body_index=body_indices[0],
            source_root_policy="optimized",
            initializer_policy="direct",
            parent_rows=(-1, 0, 1),
            parent_row_tensor=torch.tensor((-1, 0, 1)),
            position_weights=(1.0, 1.0, 1.0),
            required_position_rows=(0, 2),
            required_position_row_tensor=torch.tensor((0, 2), dtype=torch.int64),
            position_normal_channel_slots=torch.full((len(body_indices),), -1, dtype=torch.int64),
            position_body_index_tensor=body_index_tensor,
            rotation_body_indices=(0,),
            rotation_weights=(1.0,),
            source_landmark_rotation_xyzw=target_rotation.unsqueeze(0),
            direction_body_indices=(),
            direction_position_rows=(),
            direction_weights=(),
            contact_direction_rows=(),
            contact_direction_row_tensor=torch.empty(0, dtype=torch.int64),
            direction_contact_channel_slots=torch.empty(0, dtype=torch.int64),
            required_direction_rows=(),
            required_direction_row_tensor=torch.empty(0, dtype=torch.int64),
            direction_body_index_tensor=torch.empty(0, dtype=torch.int64),
            direction_position_row_tensor=torch.empty(0, dtype=torch.int64),
            direction_point_body_m=torch.empty((0, 3)),
            source_landmark_position_m=target_position,
            source_direction_point_position_m=torch.empty((0, 2, 3)),
            direction_length_values_m=(),
            initial_joint_q=desired_q.clone(),
            segment_lengths_m=torch.full((3,), scale),
            segment_length_values_m=(scale, scale, scale),
            coordinate_indices=torch.tensor(tree.coordinate_q_indices),
            coordinate_lower_limits_rad=torch.tensor(tree.coordinate_lower_limits_rad),
            coordinate_upper_limits_rad=torch.tensor(tree.coordinate_upper_limits_rad),
            source_contact_probe_position_m=target_position[:1].clone(),
            contact_channel_probe_offsets=torch.tensor((0, 1), dtype=torch.int32),
            target_support_position_m=target_position[:1].clone(),
            contact_body_indices=torch.tensor((0,)),
            contact_normal_body=torch.tensor(((0.0, 0.0, 1.0),)),
            contact_forward_body=torch.tensor(((1.0, 0.0, 0.0),)),
            contact_distal_point_body_m=torch.zeros((1, 3)),
            leg_chain_body_indices=torch.zeros((1, 3), dtype=torch.int64),
            leg_chain_parent_body_indices=torch.zeros(1, dtype=torch.int64),
            leg_knee_hint_anatomy=torch.tensor(((1.0, 0.0, 0.0),)),
            leg_knee_hint_root=torch.tensor(((1.0, 0.0, 0.0),)),
            leg_segment_lengths_m=torch.ones((1, 2)),
            support_patch_offsets=(0, 1),
            support_body_indices=torch.tensor((0,)),
            support_point_body_m=torch.zeros((1, 3)),
            support_channel_slots=torch.tensor((0,)),
        )
        objectives = motion_objective_source_global_position(
            MotionSourceGlobalPositionObjectiveCfg(weight=1.0, root_weight=10.0), targets
        )
        results.append(scale * torch.tensor(tuple(objective.weight for objective in objectives)))

    torch.testing.assert_close(results[0], torch.tensor((1.0, 1.0, 10.0)))
    torch.testing.assert_close(results[0], results[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU/GPU semantic-retarget parity.")
def test_semantic_retarget_cpu_gpu_has_clean_launch_and_finite_outputs(tmp_path: Path) -> None:
    """The 75-DOF/126-residual solve must launch cleanly with finite outputs on both devices."""
    script = r"""
import sys
import numpy as np
import torch
from isaaclab_tasks.core.multi_task.motion.data.clip_index import MotionClipIndex
from isaaclab_tasks.core.multi_task.motion.data.sources import LafanG1Clip, lafan_g1_29dof_skeleton
from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import MotionTaskTableCfg, MotionTrajectorySolveCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
    _MotionTrajectoryTargetCandidate,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import motion_solve_trajectory
from isaaclab_tasks.core.multi_task.motion.retarget import motion_contact_probe_offsets
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import (
    smpl_frame_target,
    smpl_reference_kinematics,
    smpl_source_projection,
)

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
contact_patch_cfg = MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg
contact_patches = (
    contact_patch_cfg(channel="left_foot", body_name="L_Ankle", height_band_m=0.005),
    contact_patch_cfg(channel="right_foot", body_name="R_Ankle", height_band_m=0.005),
)
contact_channels = (
    MotionTaskTableCfg.ContactChannelCfg(name="left_foot", source_probe_roles=("left_ankle", "left_toe")),
    MotionTaskTableCfg.ContactChannelCfg(name="right_foot", source_probe_roles=("right_ankle", "right_toe")),
)
target = smpl_frame_target(reference, contact_patches)
projection = smpl_source_projection(
    source,
    target,
    object(),
    contact_channels,
    motion_contact_probe_offsets(contact_channels, reference.device),
)
clip = LafanG1Clip(root, pose, 30.0)
root_position, local_rotation = clip.local_pose(source, device=device)
targets = projection.target_projection.generate_targets(root_position, local_rotation)
index = MotionClipIndex(
    source_content_sha256="0" * 64,
    skeleton_identity_sha256s=("2" * 64,),
    clips=(
        MotionClipIndex.Clip(
            clip_id="test",
            frame_count=clip.frame_count,
            source_fps=clip.source_fps,
            content_sha256="1" * 64,
            skeleton_id=0,
        ),
    ),
)
candidate = motion_solve_trajectory(
    MotionTrajectorySolveCfg(),
    _MotionTrajectoryTargetCandidate(
        target=target,
        clip_index=index,
        pending=iter((targets,)),
        source_body_counts=(projection.source_skeleton.num_bodies,),
        device=device,
        inspection=False,
    ),
)
output = (
    candidate.coordinates.joint_q,
    candidate.trajectory_quality,
    candidate.target_coordinate_evidence,
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
    for output in outputs:
        assert all(torch.isfinite(value).all() for value in output)
    assert tuple(value.shape for value in cpu) == tuple(value.shape for value in gpu)
    torch.testing.assert_close(cpu[2], gpu[2])
