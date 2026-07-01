# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for MJCF-backed shared Newton kinematics."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

_G1_MJCF_SHA256 = "439c1ec0806583d73b492da9484b0cb9e9eae215e0d9506e3c2fa69016733532"
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
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _G1_MJCF_SHA256
    return path


@pytest.fixture(scope="module")
def g1_kinematics() -> NewtonKinematics:
    """Build the exact released G1 reference model on CPU."""
    return NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(_g1_mjcf_path()), device="cpu"))


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
