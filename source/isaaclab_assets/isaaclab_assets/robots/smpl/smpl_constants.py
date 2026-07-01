# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native SMPL HumEnv asset paths, source ordering, and physical constants."""

from __future__ import annotations

import os

_SMPL_DIR = os.path.dirname(os.path.abspath(__file__))
SMPL_ROBOT_MJCF_PATH = os.path.join(_SMPL_DIR, "robot.xml")
SMPL_ROBOT_MJCF_SHA256 = "f91b6d512f9fcece8d781ccf49afbea2ba5dfd33be0be376e84d5b5579e79fe2"
SMPL_HUMENV_MJCF_PATH = os.path.join(_SMPL_DIR, "humenv.xml")
SMPL_HUMENV_MJCF_SHA256 = "a38c014814929152ddeb3e2a4f18a99c5eb861f0c7f6678416624c39dba9b024"
SMPL_NEWTON_VARIANT = "mujoco"
SMPL_ARTICULATION_ROOT_PRIM_PATH = "/Geometry/Pelvis"

NUM_JOINTS = 69
NUM_BODIES = 24

# Canonical qpos/qvel/body order in the packaged HumEnv MJCF.
MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "L_Hip_x",
    "L_Hip_y",
    "L_Hip_z",
    "L_Knee_x",
    "L_Knee_y",
    "L_Knee_z",
    "L_Ankle_x",
    "L_Ankle_y",
    "L_Ankle_z",
    "L_Toe_x",
    "L_Toe_y",
    "L_Toe_z",
    "R_Hip_x",
    "R_Hip_y",
    "R_Hip_z",
    "R_Knee_x",
    "R_Knee_y",
    "R_Knee_z",
    "R_Ankle_x",
    "R_Ankle_y",
    "R_Ankle_z",
    "R_Toe_x",
    "R_Toe_y",
    "R_Toe_z",
    "Torso_x",
    "Torso_y",
    "Torso_z",
    "Spine_x",
    "Spine_y",
    "Spine_z",
    "Chest_x",
    "Chest_y",
    "Chest_z",
    "Neck_x",
    "Neck_y",
    "Neck_z",
    "Head_x",
    "Head_y",
    "Head_z",
    "L_Thorax_x",
    "L_Thorax_y",
    "L_Thorax_z",
    "L_Shoulder_x",
    "L_Shoulder_y",
    "L_Shoulder_z",
    "L_Elbow_x",
    "L_Elbow_y",
    "L_Elbow_z",
    "L_Wrist_x",
    "L_Wrist_y",
    "L_Wrist_z",
    "L_Hand_x",
    "L_Hand_y",
    "L_Hand_z",
    "R_Thorax_x",
    "R_Thorax_y",
    "R_Thorax_z",
    "R_Shoulder_x",
    "R_Shoulder_y",
    "R_Shoulder_z",
    "R_Elbow_x",
    "R_Elbow_y",
    "R_Elbow_z",
    "R_Wrist_x",
    "R_Wrist_y",
    "R_Wrist_z",
    "R_Hand_x",
    "R_Hand_y",
    "R_Hand_z",
)

MUJOCO_BODY_NAMES: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)

# Exact source-body mass [kg] and inertia [kg m^2] in MUJOCO_BODY_NAMES order.
# Values come from MjModel.body_mass and the full body-frame inertia reconstructed
# from MjModel.body_inertia/body_iquat for the packaged robot.xml bytes above.
HUMENV_BODY_MASS: tuple[float, ...] = (
    16.22447251,  # Pelvis
    7.472438076,  # L_Hip
    3.547588285,  # L_Knee
    0.68436781,  # L_Ankle
    0.160539641,  # L_Toe
    7.336628871,  # R_Hip
    3.549124492,  # R_Knee
    0.6944991877,  # R_Ankle
    0.1662379751,  # R_Toe
    4.399722331,  # Torso
    3.894810565,  # Spine
    9.603862394,  # Chest
    0.6445350243,  # Neck
    4.32872112,  # Head
    1.085518848,  # L_Thorax
    1.895978601,  # L_Shoulder
    1.049660741,  # L_Elbow
    0.2980927806,  # L_Wrist
    0.1593473083,  # L_Hand
    1.059814912,  # R_Thorax
    1.982844157,  # R_Shoulder
    1.085693513,  # R_Elbow
    0.3164382033,  # R_Wrist
    0.1645415404,  # R_Hand
)

# MuJoCo-order full 3x3 inertia tensor (row-major 9-vec) about CoM in the BODY frame (see note above).
HUMENV_BODY_INERTIA: tuple[tuple[float, ...], ...] = (
    (0.06544865681, 0, 0, 0, 0.09905911182, 0, 0, 0, 0.08999417453),  # Pelvis
    (
        0.06691693971,
        0.004872847967,
        5.842224472e-05,
        0.004872847967,
        0.01383049835,
        -0.0006417897322,
        5.842224472e-05,
        -0.0006417897322,
        0.06735282138,
    ),  # L_Hip
    (
        0.03188604904,
        -0.0009137905897,
        -0.0001006394159,
        -0.0009137905897,
        0.005306062132,
        -0.002930816159,
        -0.0001006394159,
        -0.002930816159,
        0.03159464459,
    ),  # L_Knee
    (0.002139324649, 0, 0, 0, 0.002180370749, 0, 0, 0, 0.00102332378),  # L_Ankle
    (0.0001530563532, 0, 0, 0, 0.0002539201988, 0, 0, 0, 0.0001436744166),  # L_Toe
    (
        0.06668354441,
        -0.005374129858,
        -0.0001240543913,
        -0.005374129858,
        0.01333744753,
        -0.001243794247,
        -0.0001240543913,
        -0.001243794247,
        0.06719084173,
    ),  # R_Hip
    (
        0.03191508286,
        0.001048637285,
        0.0001110063736,
        0.001048637285,
        0.005294165587,
        -0.002822396096,
        0.0001110063736,
        -0.002822396096,
        0.03165755386,
    ),  # R_Knee
    (0.00226107869, 0, 0, 0, 0.002272202252, 0, 0, 0, 0.001069003245),  # R_Ankle
    (0.0001605332419, 0, 0, 0, 0.0002618192695, 0, 0, 0, 0.0001529926874),  # R_Toe
    (
        0.0116513314,
        -3.485484206e-05,
        -2.581840152e-07,
        -3.485484206e-05,
        0.01071154159,
        -6.970968412e-06,
        -2.581840152e-07,
        -6.970968412e-06,
        0.01165257068,
    ),  # Torso
    (
        0.009343649751,
        -5.172132842e-06,
        -2.537272715e-06,
        -5.172132842e-06,
        0.009069624298,
        -0.0001344754539,
        -2.537272715e-06,
        -0.0001344754539,
        0.009277778248,
    ),  # Spine
    (
        0.04264020721,
        1.901627458e-05,
        -4.911488136e-06,
        1.901627458e-05,
        0.03976887568,
        0.0007416347085,
        -4.911488136e-06,
        0.0007416347085,
        0.04244878511,
    ),  # Chest
    (
        0.0009629377535,
        -2.038057375e-05,
        -1.609542747e-05,
        -2.038057375e-05,
        0.0007081569813,
        -0.0002024908618,
        -1.609542747e-05,
        -0.0002024908618,
        0.0008046418877,
    ),  # Neck
    (0.02754965498, 0, 0, 0, 0.01363310516, 0, 0, 0, 0.02451429802),  # Head
    (
        0.001417470101,
        -0.0002890217952,
        8.370576582e-05,
        -0.0002890217952,
        0.002182927482,
        2.80552292e-05,
        8.370576582e-05,
        2.80552292e-05,
        0.002271672139,
    ),  # L_Thorax
    (
        0.002474936595,
        0.0003436729147,
        0.0007461319859,
        0.0003436729147,
        0.009503466815,
        -3.639668224e-05,
        0.0007461319859,
        -3.639668224e-05,
        0.009441212234,
    ),  # L_Shoulder
    (
        0.0008195982477,
        -0.0001195312618,
        1.549479319e-05,
        -0.0001195312618,
        0.004126742276,
        5.593040324e-07,
        1.549479319e-05,
        5.593040324e-07,
        0.004130984405,
    ),  # L_Elbow
    (
        0.0001437060849,
        1.553557477e-05,
        2.853472918e-05,
        1.553557477e-05,
        0.0003019901652,
        -2.774209781e-06,
        2.853472918e-05,
        -2.774209781e-06,
        0.000298405081,
    ),  # L_Wrist
    (0.0001670002284, 0, 0, 0, 0.0003355158496, 0, 0, 0, 0.0001950352626),  # L_Hand
    (
        0.001348250006,
        0.0003023382558,
        -8.527489265e-05,
        0.0003023382558,
        0.002138956603,
        2.886910428e-05,
        -8.527489265e-05,
        2.886910428e-05,
        0.002233168132,
    ),  # R_Thorax
    (
        0.002689349638,
        -0.0003783872973,
        -0.0006054196757,
        -0.0003783872973,
        0.009873021918,
        -3.180142748e-05,
        -0.0006054196757,
        -3.180142748e-05,
        0.009842015527,
    ),  # R_Shoulder
    (
        0.0008611615062,
        0.0001071146056,
        -7.684308663e-05,
        0.0001071146056,
        0.004422996674,
        2.308805999e-06,
        -7.684308663e-05,
        2.308805999e-06,
        0.004424558693,
    ),  # R_Elbow
    (
        0.0001562624517,
        -1.275227126e-05,
        -2.136867077e-05,
        -1.275227126e-05,
        0.0003304188825,
        -1.556379564e-06,
        -2.136867077e-05,
        -1.556379564e-06,
        0.0003287397022,
    ),  # R_Wrist
    (0.0001782599171, 0, 0, 0, 0.0003410819984, 0, 0, 0, 0.0001923254765),  # R_Hand
)

assert len(HUMENV_BODY_MASS) == NUM_BODIES
assert len(HUMENV_BODY_INERTIA) == NUM_BODIES and all(len(r) == 9 for r in HUMENV_BODY_INERTIA)


__all__ = [
    "HUMENV_BODY_INERTIA",
    "HUMENV_BODY_MASS",
    "MUJOCO_BODY_NAMES",
    "MUJOCO_JOINT_NAMES",
    "NUM_BODIES",
    "NUM_JOINTS",
    "SMPL_ARTICULATION_ROOT_PRIM_PATH",
    "SMPL_HUMENV_MJCF_PATH",
    "SMPL_HUMENV_MJCF_SHA256",
    "SMPL_NEWTON_VARIANT",
    "SMPL_ROBOT_MJCF_PATH",
    "SMPL_ROBOT_MJCF_SHA256",
]
