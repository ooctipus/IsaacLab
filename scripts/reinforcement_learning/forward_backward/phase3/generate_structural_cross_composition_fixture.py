# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the deterministic portable-human to G1 structural proof fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from inspect import getsourcefile
from pathlib import Path

import numpy as np
from motion_environment_identity import motion_environment_axes, motion_g1_live_axes

from isaaclab_tasks.core.multi_task.kinematics import KinematicTreeRotationProjection
from isaaclab_tasks.core.multi_task.motion.data import MotionSkeleton
from isaaclab_tasks.core.multi_task.motion.data.sources import lafan_g1_29dof_skeleton
from isaaclab_tasks.core.multi_task.motion.robots.g1.frames import G1_HEAD_FRAME_NAME
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1PoseFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

ROOT = Path(__file__).parent
DEFAULT_ARCHIVE = ROOT / "fixtures" / "human_motion_g1_structural_v1.npz"
DEFAULT_RECORD = ROOT / "fixtures" / "human_motion_g1_structural_v1.json"
_CONTENT_SHA256 = hashlib.sha256(b"analytical-human-30-body-structural-fixture-v1").hexdigest()


def source_skeleton() -> MotionSkeleton:
    """Return the declared human source skeleton used by the structural proof."""
    target = lafan_g1_29dof_skeleton()
    return MotionSkeleton(
        identifier="analytical_human_30_body_v1",
        content_sha256=_CONTENT_SHA256,
        body_names=tuple(f"human_body_{index:02d}" for index in range(target.num_bodies)),
        parent_indices=target.parent_indices,
        rest_translation_m=tuple(
            tuple(1.1 * component for component in translation) for translation in target.rest_translation_m
        ),
        rest_rotation_wxyz=target.rest_rotation_wxyz,
        joint_names=tuple(f"human_ball_joint_{index:02d}" for index in range(1, target.num_bodies)),
        joint_child_body_indices=tuple(range(1, target.num_bodies)),
        joint_axes=(None,) * (target.num_bodies - 1),
        root_translation_frame="right_handed_z_up_world",
        root_rotation_convention="local_wxyz_full_body_rotations",
    )


def arrays() -> dict[str, np.ndarray]:
    """Return ordered portable arrays with independent per-clip clocks."""
    target = lafan_g1_29dof_skeleton()
    source_fps = (24.0, 60.0)
    frame_counts = (12, 15)
    offsets = np.asarray((0, frame_counts[0], sum(frame_counts)), dtype=np.int64)
    joint_rates = np.linspace(0.01, 0.29, target.num_joints, dtype=np.float32)
    axes = np.asarray(target.joint_axes, dtype=np.float32)
    child_indices = np.asarray(target.joint_child_body_indices, dtype=np.int64)
    roots: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for clip_index, (fps, frame_count) in enumerate(zip(source_fps, frame_counts, strict=True)):
        time = np.arange(frame_count, dtype=np.float32) / np.float32(fps)
        root = np.zeros((frame_count, 3), dtype=np.float32)
        root[:, 0] = time * np.float32(0.1 + 0.02 * clip_index)
        root[:, 2] = 0.85
        roots.append(root)

        quaternion = np.zeros((frame_count, target.num_bodies, 4), dtype=np.float32)
        quaternion[..., 0] = 1.0
        root_angle = np.float32(0.2) * time
        quaternion[:, 0, 0] = np.cos(np.float32(0.5) * root_angle)
        quaternion[:, 0, 3] = np.sin(np.float32(0.5) * root_angle)
        joint_angle = time[:, None] * joint_rates[None]
        quaternion[:, child_indices, 0] = np.cos(np.float32(0.5) * joint_angle)
        quaternion[:, child_indices, 1:] = np.sin(np.float32(0.5) * joint_angle)[..., None] * axes[None]
        rotations.append(quaternion)

    return {
        "clip_ids": np.asarray(("walk_24hz", "turn_60hz"), dtype="<U16"),
        "frame_offsets": offsets,
        "source_fps": np.asarray(source_fps, dtype=np.float64),
        "root_translation": np.concatenate(roots, axis=0),
        "local_rotation_wxyz": np.concatenate(rotations, axis=0),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_sha256(value: object) -> str:
    path = getsourcefile(value)
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    return _sha256_file(Path(path))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def write_archive(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write an NPZ whose bytes do not depend on wall time or host metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in tensors.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(value))


def record(archive_path: Path, tensors: dict[str, np.ndarray]) -> dict[str, object]:
    """Return the structural claim, provenance, and separated error owners."""
    source = source_skeleton()
    target = lafan_g1_29dof_skeleton()
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_cmu"))
    table_cfg = cfg.commands.motion.task_table
    joint_names, body_names = motion_g1_live_axes(cfg)
    reference_frame_names = (*body_names, G1_HEAD_FRAME_NAME)
    axis_contract = {
        "joint_names": joint_names,
        "reference_frame_names": reference_frame_names,
        "scene_robot_asset": Path(cfg.scene.robot.spawn.usd_path).name,
        "frame_builder_factory": (
            f"{table_cfg.frame_builder_factory.__module__}:{table_cfg.frame_builder_factory.__qualname__}"
        ),
    }
    return {
        "schema": "forward_backward_phase3g_structural_cross_composition_v3",
        "claim": {
            "level": "structural_interface_proof",
            "canonicality": "portable_source_contract_with_declared_skeleton_not_a_canonical_dataset",
            "proves": [
                "source-level local rotations remain independent of the scene articulation",
                "per-clip source clocks survive source-to-trajectory construction",
                "one non-native source can feed the G1 task-table axes and learner views",
            ],
            "does_not_prove": [
                "learned human-to-robot retarget quality",
                "simulator tracking quality",
                "policy quality or convergence",
            ],
        },
        "source": {
            "semantic_level": "analytical_human_local_rotation_motion",
            "skeleton": asdict(source),
            "archive": {
                "file": archive_path.name,
                "sha256": _sha256_file(archive_path),
                "ordered_fields": list(tensors),
                "tensors": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": _sha256_bytes(np.ascontiguousarray(value).tobytes()),
                    }
                    for name, value in tensors.items()
                },
                "clips": [
                    {"id": "walk_24hz", "frames": 12, "source_fps": 24.0},
                    {"id": "turn_60hz", "frames": 15, "source_fps": 60.0},
                ],
                "license": "synthetic_fixture_bsd_3_clause",
            },
        },
        "target": {
            "scene_robot": "g1_29dof",
            "scene_robot_asset": axis_contract["scene_robot_asset"],
            "reference_skeleton_identity_sha256": target.identity_sha256,
            "joint_names": list(joint_names),
            "reference_frame_names": list(reference_frame_names),
            "axis_contract_sha256": _canonical_sha256(axis_contract),
            "target_joint_source_body_indices": list(range(1, source.num_bodies)),
            "projection": "globally_closest_sign_invariant_geodesic_fit_per_ordered_target_hinge_chain",
            "trajectory_builder_factory": axis_contract["frame_builder_factory"],
            "trajectory_builder_contract": "g1_human_ordered_hinge_fit_structural_v3",
        },
        "composition": {
            "identifier": "g1_analytical_human_50hz",
            "selected_preset": "g1_cmu",
            "scene_robot": "g1_29dof",
            "source_identifier": "analytical_human_motion",
            "source_template": table_cfg.source.identifier,
            "physics_dt_seconds": cfg.sim.dt,
            "control_decimation": cfg.decimation,
            "control_dt_seconds": cfg.sim.dt * cfg.decimation,
        },
        "code_identity": {
            "generator_sha256": _sha256_file(Path(__file__)),
            "target_builder_sha256": _source_sha256(G1PoseFrameBuilder),
            "projection_sha256": _source_sha256(KinematicTreeRotationProjection),
            "environment_cfg_sha256": _source_sha256(MotionImitationEnvCfg),
        },
        "error_ownership": {
            "retarget": {
                "owner": "source_to_trajectory_builder",
                "fixture_metric": "analytic_g1_joint_position_and_velocity_max_abs_error",
                "position_tolerance_rad": 2.0e-6,
                "velocity_tolerance_rad_s": 5.0e-5,
                "status": "measured_by_structural_test",
            },
            "simulator_tracking": {
                "owner": "controller_and_simulator",
                "metric": "robot_state_minus_materialized_reference_at_equal_time",
                "status": "not_claimed_by_pure_data_fixture",
            },
            "policy": {
                "owner": "learner_and_observation_action_routes",
                "metric": "evaluation_emd_broad_reward_safety_after_tracking_error_is_reported",
                "status": "not_claimed_by_pure_data_fixture",
            },
        },
    }


def main() -> None:
    """Write the structural archive and its exact evidence record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()
    tensors = arrays()
    write_archive(args.archive, tensors)
    args.record.write_text(json.dumps(record(args.archive, tensors), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
