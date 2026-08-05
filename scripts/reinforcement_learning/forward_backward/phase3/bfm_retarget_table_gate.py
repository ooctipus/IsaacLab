# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 gate of bfm-converter-20260805: table-build inspection receipt for one v5-dump split.

Builds the production motion task table from a retarget-dump source and receipts:
resolved-route assertion (family must be EXACT — risk-4), clip/frame counts vs the
registration pins and the campaign dump MANIFEST (fail-closed exclusions), joint-order
check against the live articulation axes, root-quaternion norm statistics, and a
one-clip FK spot check against the payload's own solved_robot_landmarks (verifies the
xyzw root convention and the joint remap numerically, ground-gauge corrected).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.motion.data.sources.retarget_dump_v5 import RetargetDumpV5Clips
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import build_motion_task_table
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

_ROUTE_AXES = {
    "cmu_smpl": ("smpl", "cmu_retarget", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows"),
    "lafan_g1": (
        "g1",
        "lafan_retarget",
        "physx",
        "timing_sim200_control50_horizon501",
        "sampling_clip_time",
        "evidence_physical_auxiliary",
        "randomization_physics_observation_pose_push",
    ),
}
_ROUTE_SETS = {
    ("cmu_smpl", "train"): "cmu_smpl_train",
    ("cmu_smpl", "evaluation"): "cmu_smpl_test",
    ("lafan_g1", "train"): "lafan_g1_train",
    ("lafan_g1", "evaluation"): "lafan_g1_eval",
}
# Body-origin landmark roles shared by the producer payloads and the trainer targets.
_G1_ROLE_BODIES = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_pitch_link",
    "left_knee": "left_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_hip": "right_hip_pitch_link",
    "right_knee": "right_knee_link",
    "right_ankle": "right_ankle_roll_link",
    "torso": "torso_link",
    "left_shoulder": "left_shoulder_pitch_link",
    "left_elbow": "left_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_shoulder": "right_shoulder_pitch_link",
    "right_elbow": "right_elbow_link",
    "right_wrist": "right_wrist_yaw_link",
}
_SMPL_ROLE_BODIES = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "left_knee": "L_Knee",
    "left_ankle": "L_Ankle",
    "left_toe": "L_Toe",
    "right_hip": "R_Hip",
    "right_knee": "R_Knee",
    "right_ankle": "R_Ankle",
    "right_toe": "R_Toe",
    "torso": "Torso",
    "spine": "Spine",
    "chest": "Chest",
    "neck": "Neck",
    "head": "Head",
    "left_thorax": "L_Thorax",
    "left_shoulder": "L_Shoulder",
    "left_elbow": "L_Elbow",
    "left_wrist": "L_Wrist",
    "right_thorax": "R_Thorax",
    "right_shoulder": "R_Shoulder",
    "right_elbow": "R_Elbow",
    "right_wrist": "R_Wrist",
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolved_cfg(route: str, split: str, args: argparse.Namespace) -> MotionImitationEnvCfg:
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=frozenset(_ROUTE_AXES[route]))
    cfg.seed = 0
    cfg.sim.device = args.device
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.target_artifact_root = str(args.target_artifact_root.expanduser().resolve())
    table_cfg.motion_split = split
    return cfg


def _fk_kinematics(route: str, cfg: MotionImitationEnvCfg, args: argparse.Namespace, device: str) -> NewtonKinematics:
    """Build the FK oracle for the spot check: the exact target model the table uses.

    G1 goes through the scene articulation (g1_29dof_rev_1_0 — also the model the
    producer pipeline solved on); SMPL uses the packaged scene MJCF, kinematically
    identical to the producer's humenv.xml (visual-only diff).
    """
    del args
    if route == "cmu_smpl":
        return NewtonKinematics(
            NewtonKinematicsCfg(mjcf_path=cfg.scene.robot.spawn.asset_path, device=device, collapse_fixed_joints=False)
        )
    table_cfg = cfg.commands.motion.task_table
    return NewtonKinematics.from_articulation(table_cfg.target_kinematics.kinematics, cfg.scene.robot, device)


def _model_coordinate_columns(kinematics: NewtonKinematics, source_names: tuple[str, ...], route: str) -> list[int]:
    """Map each source coordinate (skeleton order) to its FK-model joint_q column."""
    joint_q_start = kinematics.topology.joint_q_start
    model_columns: dict[str, int] = {}
    for joint_index in range(1, kinematics.topology.joint_count):
        start, stop = int(joint_q_start[joint_index]), int(joint_q_start[joint_index + 1])
        name = kinematics.joint_names[joint_index]
        for component in range(stop - start):
            label = f"{name}:{component}" if stop - start > 1 else name
            model_columns[label] = start + component
    if route == "cmu_smpl":
        from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import smpl_live_joint_mujoco_names

        labels = tuple(model_columns)
        resolved = dict(zip(smpl_live_joint_mujoco_names(labels), (model_columns[label] for label in labels)))
        model_columns = resolved
    return [model_columns[name] for name in source_names]


def _fk_spot_check(
    route: str,
    source: RetargetDumpV5Clips,
    entry: dict[str, object],
    clip_index_position: int,
    kinematics: NewtonKinematics,
    source_root: Path,
    device: str,
) -> dict[str, object]:
    """Compare trainer-side FK of converted coordinates against producer landmarks."""
    skeleton = source.skeleton(0)
    ((_, clip),) = tuple(source.clips((clip_index_position,)))
    joint_q_source, _ = clip.free_root_coordinates(skeleton, device=device)
    columns = _model_coordinate_columns(kinematics, skeleton.joint_names, route)
    frame_count = joint_q_source.shape[0]
    joint_q = torch.empty((frame_count, kinematics.model.joint_coord_count), dtype=torch.float32, device=device)
    joint_q[:, :7] = joint_q_source[:, :7]
    for model_column, source_column in zip(columns, range(7, joint_q_source.shape[1])):
        joint_q[:, model_column] = joint_q_source[:, source_column]
    joint_qd = torch.zeros((frame_count, kinematics.model.joint_dof_count), dtype=torch.float32, device=device)
    body_q = torch.empty((frame_count, kinematics.model.body_count, 7), dtype=torch.float32, device=device)
    body_qd = torch.empty((frame_count, kinematics.model.body_count, 6), dtype=torch.float32, device=device)
    kinematics.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)

    payload = torch.load(source_root / str(entry["file"]), map_location="cpu", weights_only=False)
    landmark_names = tuple(payload["landmark_names"])
    stride = int(entry["stride"])
    solved = payload["solved_robot_landmarks"][::stride].to(device)
    ground_shift = float(entry["ground_shift_m"])
    role_bodies = _SMPL_ROLE_BODIES if route == "cmu_smpl" else _G1_ROLE_BODIES
    body_index = {name: index for index, name in enumerate(kinematics.body_names)}
    per_role: dict[str, float] = {}
    per_role_shifted: dict[str, float] = {}
    for role, body_name in role_bodies.items():
        if role not in landmark_names or body_name not in body_index:
            continue
        fk_positions = body_q[:, body_index[body_name], :3]
        reference = solved[:, landmark_names.index(role), :]
        per_role[role] = float((fk_positions - reference).norm(dim=-1).max())
        corrected = fk_positions.clone()
        corrected[:, 2] -= ground_shift
        per_role_shifted[role] = float((corrected - reference).norm(dim=-1).max())
    raw_max = max(per_role.values())
    shifted_max = max(per_role_shifted.values())
    landmark_frame = "post_gauge" if raw_max <= shifted_max else "pre_gauge"
    return {
        "clip_id": str(entry["clip_id"]),
        "frames_compared": frame_count,
        "roles_compared": len(per_role),
        "ground_shift_m": ground_shift,
        "landmark_frame": landmark_frame,
        "max_deviation_m_raw": raw_max,
        "max_deviation_m_gauge_corrected": shifted_max,
        "max_deviation_m": min(raw_max, shifted_max),
        "per_role_m": dict(sorted((per_role if landmark_frame == "post_gauge" else per_role_shifted).items())),
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    device = args.device
    route, split = args.route, args.split
    set_name = _ROUTE_SETS[(route, split)]
    cfg = _resolved_cfg(route, split, args)
    table_cfg = cfg.commands.motion.task_table
    split_cfg = table_cfg.source.train if split == "train" else table_cfg.source.evaluation
    source_root = Path(table_cfg.source_artifact_root)
    index_path = source_root / split_cfg.artifact
    index_payload = json.loads(index_path.read_text())
    manifest_path = source_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_record = manifest["sets"][set_name]

    # Fail-closed reconciliation: registration pins == manifest accepted subset.
    accepted_manifest = sorted(
        clip_id for clip_id, value in manifest_record["clips"].items() if value["accepted_full"]
    )
    index_clip_ids = [clip["clip_id"] for clip in index_payload["clips"]]
    if index_clip_ids != accepted_manifest:
        raise RuntimeError("Split index accepted clips differ from the manifest accepted_full subset.")
    if len(index_clip_ids) != split_cfg.clip_count:
        raise RuntimeError("Registration clip pin differs from the manifest accepted subset.")
    excluded = sorted(set(manifest_record["clips"]) - set(accepted_manifest))

    table = build_motion_task_table(cfg.commands.motion, cfg.scene, device)
    if table.family_name != "exact":
        raise RuntimeError(
            f"RISK-4 VIOLATION: resolved coordinate family is {table.family_name!r}, not 'exact' — "
            "the build would measure the wrong solver."
        )
    if len(table.clip_index.clips) != split_cfg.clip_count or table.clip_index.total_frames != split_cfg.frame_count:
        raise RuntimeError("Built table counts differ from the registration pins.")
    if tuple(table.clip_ids) != tuple(index_clip_ids):
        raise RuntimeError("Built table clip order differs from the frozen index.")

    if route == "cmu_smpl":
        from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_JOINT_NAMES
        from isaaclab_tasks.core.multi_task.motion.robots.smpl.articulation import smpl_live_joint_mujoco_names

        joint_order_matches = smpl_live_joint_mujoco_names(tuple(table.joint_names)) == tuple(MUJOCO_JOINT_NAMES)
    else:
        from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import G1_SIMULATOR_JOINT_NAMES

        joint_order_matches = tuple(table.joint_names) == tuple(G1_SIMULATOR_JOINT_NAMES)

    # Root-quaternion norm statistics recomputed at the decode boundary.
    source = table_cfg.source.open_split(source_root, split_cfg)
    skeleton = source.skeleton(0)
    sample_positions = sorted({0, len(index_clip_ids) // 2, len(index_clip_ids) - 1})
    norm_deviation = 0.0
    for _, clip in source.clips(tuple(sample_positions)):
        joint_q, _ = clip.free_root_coordinates(skeleton, device=device)
        norms = torch.linalg.vector_norm(joint_q[:, 3:7], dim=-1)
        norm_deviation = max(norm_deviation, float((norms - 1.0).abs().max()))
    source.close()

    fk_source = table_cfg.source.open_split(source_root, split_cfg)
    kinematics = _fk_kinematics(route, cfg, args, device)
    fk = _fk_spot_check(
        route,
        fk_source,
        index_payload["clips"][args.fk_clip_position],
        args.fk_clip_position,
        kinematics,
        source_root,
        device,
    )
    fk_source.close()
    fk_pass = fk["max_deviation_m"] <= args.fk_tolerance_m
    if not fk_pass:
        raise RuntimeError(f"FK spot check exceeded {args.fk_tolerance_m} m: {fk}")
    if not joint_order_matches:
        raise RuntimeError(f"Table joint order differs from the live articulation axes: {table.joint_names}")

    return {
        "schema": "bfm_converter_20260805_table_gate_receipt_v1",
        "status": "passed",
        "route": route,
        "split": split,
        "set": set_name,
        "resolved_route_family": table.family_name,
        "family_identity_sha256": table.family_identity_sha256,
        "table_identity_sha256": table.cache_identity,
        "construction_version": table.construction_version,
        "counts": {
            "clip_count": len(table.clip_index.clips),
            "frame_count": table.clip_index.total_frames,
            "pinned_clip_count": split_cfg.clip_count,
            "pinned_frame_count": split_cfg.frame_count,
            "manifest_total_clips": len(manifest_record["clips"]),
            "excluded_fail_closed": excluded,
        },
        "identity": {
            "split_index_artifact": split_cfg.artifact,
            "split_index_sha256": _file_sha256(index_path),
            "pinned_artifact_sha256": split_cfg.artifact_sha256,
            "manifest_sha256": _file_sha256(manifest_path),
            "index_manifest_sha256": index_payload["manifest_sha256"],
            "engine_commit": index_payload["engine_commit"],
            "run_json_sha256": index_payload["run_json_sha256"],
            "source_skeleton_coordinate_identity_sha256": skeleton.coordinate_identity_sha256,
        },
        "joint_order": {
            "matches_live_articulation": joint_order_matches,
            "table_joint_names_sha256": hashlib.sha256(json.dumps(list(table.joint_names)).encode()).hexdigest(),
        },
        "root_quaternion_max_norm_deviation": norm_deviation,
        "fk_spot_check": {**fk, "tolerance_m": args.fk_tolerance_m, "passed": fk_pass},
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("cmu_smpl", "lafan_g1"), required=True)
    parser.add_argument("--split", choices=("train", "evaluation"), required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--target_artifact_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fk_clip_position", type=int, default=0)
    parser.add_argument("--fk_tolerance_m", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
