# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate schema carried by native CMU HumEnv SMPL rows."""

from __future__ import annotations

from isaaclab_assets.robots.smpl.smpl_constants import (
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    SMPL_HUMENV_MJCF_SHA256,
)

from ..skeleton import MotionSkeleton

_PARENT_INDICES = (
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    12,
    11,
    14,
    15,
    16,
    17,
    11,
    19,
    20,
    21,
    22,
)
_REST_TRANSLATION_M = (
    (-0.0018, -0.2233, 0.0282),
    (0.0695, -0.0914, -0.0068),
    (0.0343, -0.3752, -0.0045),
    (-0.0136, -0.3980, -0.0437),
    (0.0264, -0.0558, 0.1193),
    (-0.0677, -0.0905, -0.0043),
    (-0.0383, -0.3826, -0.0089),
    (0.0158, -0.3984, -0.0423),
    (-0.0254, -0.0481, 0.1233),
    (-0.0025, 0.1090, -0.0267),
    (0.0055, 0.1352, 0.0011),
    (0.0015, 0.0529, 0.0254),
    (-0.0028, 0.2139, -0.0429),
    (0.0052, 0.0650, 0.0513),
    (0.0788, 0.1217, -0.0341),
    (0.0910, 0.0305, -0.0089),
    (0.2596, -0.0128, -0.0275),
    (0.2492, 0.0090, -0.0012),
    (0.0840, -0.0082, -0.0149),
    (-0.0818, 0.1188, -0.0386),
    (-0.0960, 0.0326, -0.0091),
    (-0.2537, -0.0133, -0.0214),
    (-0.2553, 0.0078, -0.0056),
    (-0.0846, -0.0061, -0.0103),
)
_JOINT_CHILD_BODY_INDICES = tuple(body_index for body_index in range(1, 24) for _ in range(3))
_JOINT_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) * 23
_LANDMARKS = (
    MotionSkeleton.Landmark("pelvis", "Pelvis", "Pelvis"),
    MotionSkeleton.Landmark("left_hip", "L_Hip", "L_Hip"),
    MotionSkeleton.Landmark("left_knee", "L_Knee", "L_Knee"),
    MotionSkeleton.Landmark("left_ankle", "L_Ankle", "L_Ankle"),
    MotionSkeleton.Landmark("right_hip", "R_Hip", "R_Hip"),
    MotionSkeleton.Landmark("right_knee", "R_Knee", "R_Knee"),
    MotionSkeleton.Landmark("right_ankle", "R_Ankle", "R_Ankle"),
    MotionSkeleton.Landmark("torso", "Torso", "Torso"),
    MotionSkeleton.Landmark("left_shoulder", "L_Shoulder", "L_Shoulder"),
    MotionSkeleton.Landmark("left_elbow", "L_Elbow", "L_Elbow"),
    MotionSkeleton.Landmark("left_wrist", "L_Wrist", "L_Wrist"),
    MotionSkeleton.Landmark("right_shoulder", "R_Shoulder", "R_Shoulder"),
    MotionSkeleton.Landmark("right_elbow", "R_Elbow", "R_Elbow"),
    MotionSkeleton.Landmark("right_wrist", "R_Wrist", "R_Wrist"),
)


def cmu_humenv_smpl_skeleton() -> MotionSkeleton:
    """Return the coordinate system carried by native HumEnv SMPL rows."""
    return MotionSkeleton(
        identifier="smpl_humenv_24_body_69_joint_v1",
        content_sha256=SMPL_HUMENV_MJCF_SHA256,
        body_names=MUJOCO_BODY_NAMES,
        parent_indices=_PARENT_INDICES,
        rest_translation_m=_REST_TRANSLATION_M,
        rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0),) * 24,
        joint_names=MUJOCO_JOINT_NAMES,
        joint_child_body_indices=_JOINT_CHILD_BODY_INDICES,
        joint_axes=_JOINT_AXES,
        landmarks=_LANDMARKS,
        root_translation_frame="world",
        root_rotation_convention="wxyz",
    )
