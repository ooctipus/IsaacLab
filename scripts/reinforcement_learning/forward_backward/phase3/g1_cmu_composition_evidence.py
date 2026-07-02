# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure SMPL-to-G1 fit residuals and post-unwrapping velocities on real CMU rows."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from motion_environment_identity import (
    motion_composition_dependency_identity,
    motion_g1_live_axes,
)

from isaaclab.utils.math import convert_quat

from isaaclab_tasks.core.multi_task.motion.data.importers import HumEnvHdf5Clips
from isaaclab_tasks.core.multi_task.motion.trajectory.g1_smpl import (
    G1SmplHumEnvFrameBuilder,
    fit_ordered_hinge_coordinates,
    smpl_humenv_local_rotation_wxyz,
)
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets


def _source_sha256(value: object) -> str:
    """Hash the file defining one measured source-to-trajectory boundary."""
    path = inspect.getsourcefile(value)
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _statistics(values: torch.Tensor) -> dict[str, float | int]:
    """Return exact finite scalar statistics for one measured tensor."""
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(values)
    if not torch.all(finite):
        raise ValueError("Composition evidence contains non-finite measurements.")
    quantiles = torch.quantile(values, torch.tensor((0.5, 0.95, 0.99, 0.999), dtype=torch.float64))
    return {
        "count": values.numel(),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "q50": float(quantiles[0]),
        "q95": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "q999": float(quantiles[3]),
        "max": float(values.max()),
    }


def _unmeasured_error_layers() -> dict[str, dict[str, object]]:
    """Declare evidence owned by later simulator and policy probes."""
    return {
        "reference_controller_simulator": {
            "status": "not_measured_by_source_composition_probe",
            "required_metric": "simulated_robot_state_minus_materialized_reference_at_equal_time",
        },
        "policy": {
            "status": "not_measured_by_source_composition_probe",
            "required_metrics": ["evaluation_emd", "broad_reward", "safety_violations"],
        },
    }


def _resolved_builder(
    source_artifact_root: Path,
    reference_artifact_root: Path,
    motion_split: str,
    device: torch.device,
) -> tuple[MotionImitationEnvCfg, G1SmplHumEnvFrameBuilder]:
    """Build the direct G1-CMU trajectory policy without creating a simulator."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={"g1_cmu"})
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(source_artifact_root)
    table_cfg.reference_artifact_root = str(reference_artifact_root)
    table_cfg.motion_split = motion_split
    joint_names, body_names = motion_g1_live_axes(cfg)
    robot = SimpleNamespace(
        joint_names=joint_names,
        body_names=body_names,
    )
    env = SimpleNamespace(cfg=cfg, device=str(device), scene={"robot": robot})
    builder = cfg.commands.motion.task_table.frame_builder_factory(env)
    if not isinstance(builder, G1SmplHumEnvFrameBuilder):
        raise TypeError("The resolved g1_cmu command axis must build G1SmplHumEnvFrameBuilder.")
    return cfg, builder


def _projection_groups(mapping: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    """Return the contiguous source-body groups encoded by the target hinge map."""
    groups: list[tuple[int, int, int]] = []
    start = 0
    while start < len(mapping):
        source_body = mapping[start]
        stop = start + 1
        while stop < len(mapping) and mapping[stop] == source_body:
            stop += 1
        groups.append((start, stop, source_body))
        start = stop
    return tuple(groups)


def _measure(
    source_artifact_root: Path,
    reference_artifact_root: Path,
    split_name: str,
    device: torch.device,
    max_clips: int | None,
) -> dict[str, object]:
    """Measure only the source-to-robot layer; do not imply simulator or policy quality."""
    cfg, builder = _resolved_builder(source_artifact_root, reference_artifact_root, split_name, device)
    source_cfg = cfg.commands.motion.task_table.source
    split = source_cfg.train if split_name == "train" else source_cfg.evaluation
    source = source_cfg.open_split(source_artifact_root, split)
    index = source.inspect()
    if (
        index.source_content_sha256 != split.source_content_sha256
        or len(index.clips) != split.clip_count
        or index.total_frames != split.frame_count
    ):
        raise ValueError(
            "Motion source identity/counts differ from the selected split: "
            f"hash={index.source_content_sha256}, clips={len(index.clips)}, frames={index.total_frames}."
        )
    foot_indices = torch.tensor(
        tuple(index for index, name in enumerate(builder.reference_frame_names) if "ankle" in name),
        dtype=torch.int64,
        device=device,
    )
    if foot_indices.numel() != 4:
        raise ValueError("G1 body-origin clearance requires all four ankle-link bodies.")

    residual_chunks: list[torch.Tensor] = []
    velocity_chunks: list[torch.Tensor] = []
    step_chunks: list[torch.Tensor] = []
    body_height_chunks: list[torch.Tensor] = []
    foot_height_chunks: list[torch.Tensor] = []
    selected_clip_ids: list[str] = []
    groups = _projection_groups(builder.projection.target_joint_source_body_indices)
    group_indices = tuple(source_body for _, _, source_body in groups)
    target_axes = torch.tensor(
        builder.projection.target_builder.source_skeleton.joint_axes,
        dtype=torch.float32,
        device=device,
    )
    for clip_position, (clip_id, fields) in enumerate(source.clips()):
        if max_clips is not None and clip_position >= max_clips:
            break
        qpos_array = fields["qpos"]
        if not hasattr(qpos_array, "dtype"):
            raise TypeError("HumEnv qpos must be a NumPy array.")
        qpos = torch.as_tensor(qpos_array, device=device)
        local_wxyz = smpl_humenv_local_rotation_wxyz(qpos, builder.source_skeleton)
        local_xyzw = convert_quat(local_wxyz, to="xyzw")
        residual = torch.stack(
            tuple(
                fit_ordered_hinge_coordinates(
                    local_xyzw[:, source_body],
                    target_axes[start:stop],
                )[1]
                for start, stop, source_body in groups
            ),
            dim=-1,
        )
        pose_axis_angle = builder.projection.project_local_rotations(local_wxyz)
        facts = builder.projection.target_builder.build_pose_frames(
            pose_axis_angle,
            qpos[:, :3],
            builder.source_fps,
        )

        joint_position = facts.joint_position
        body_height = facts.body_position[..., 2]
        residual_chunks.append(residual.cpu())
        velocity_chunks.append(facts.joint_velocity.abs().cpu())
        step_chunks.append(torch.diff(joint_position, dim=0).abs().cpu())
        body_height_chunks.append(body_height.cpu())
        foot_height_chunks.append(body_height.index_select(1, foot_indices).cpu())
        selected_clip_ids.append(clip_id)

    if not selected_clip_ids:
        raise ValueError("The requested CMU evidence selection contains no clips.")

    residual = torch.cat(residual_chunks)
    absolute_velocity = torch.cat(velocity_chunks)
    absolute_step = torch.cat(step_chunks)
    body_height = torch.cat(body_height_chunks)
    foot_height = torch.cat(foot_height_chunks)
    group_names = tuple(builder.source_skeleton.body_names[index] for index in group_indices)
    group_statistics = {name: _statistics(residual[:, group]) for group, name in enumerate(group_names)}
    return {
        "schema": "forward_backward_phase3g_g1_cmu_composition_evidence_v3",
        "code_identity": {
            "probe_sha256": _source_sha256(_measure),
            "composition_dependency_identity": motion_composition_dependency_identity(
                preset="g1_cmu",
                cfg=cfg,
                importer_type=HumEnvHdf5Clips,
                frame_builder_type=G1SmplHumEnvFrameBuilder,
                frame_builder_identity_sha256=builder.construction_identity_sha256,
                reference_artifact_root=reference_artifact_root,
            ),
        },
        "composition": {
            "selected": "g1_cmu",
            "source": source_cfg.identifier,
            "scene_robot": "g1_29dof",
            "frame_builder_type": f"{type(builder).__module__}:{type(builder).__qualname__}",
            "frame_builder_version": builder.version,
            "frame_builder_identity_sha256": builder.construction_identity_sha256,
            "joint_names": list(builder.joint_names),
            "reference_frame_names": list(builder.reference_frame_names),
        },
        "source": {
            "artifact_root": str(source_artifact_root.expanduser().resolve()),
            "split": split.name,
            "declared_source_content_sha256": split.source_content_sha256,
            "inspected_content_identity_sha256": index.content_identity_sha256,
            "selected_clip_count": len(selected_clip_ids),
            "selected_frame_count": int(residual.shape[0]),
            "selected_clip_ids": selected_clip_ids,
            "complete_split": max_clips is None or len(selected_clip_ids) == len(index.clips),
        },
        "error_layers": {
            "retarget_fit": {
                "status": "measured",
                "metric": "sign_invariant_local_rotation_geodesic_error_rad",
                "group_order": list(group_names),
                "all_groups_rad": _statistics(residual),
                "groups_rad": group_statistics,
                "absolute_joint_velocity_rad_s": _statistics(absolute_velocity),
                "absolute_joint_step_rad": _statistics(absolute_step),
                "reference_ground_feasibility": {
                    "metric": "materialized_body_origin_height_m_against_z_zero",
                    "all_body_origin_height_m": _statistics(body_height),
                    "ankle_link_origin_height_m": _statistics(foot_height),
                    "all_body_origin_penetration_m": _statistics((-body_height).clamp_min(0.0)),
                    "ankle_link_origin_penetration_m": _statistics((-foot_height).clamp_min(0.0)),
                    "geometry_scope": (
                        "body origins only; collision-geometry sole clearance is owned by the simulator probe"
                    ),
                },
            },
            **_unmeasured_error_layers(),
        },
    }


def main() -> None:
    """Measure a declared real source split and atomically write its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path, required=True)
    parser.add_argument("--motion_split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_clips", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("max_clips must be positive when provided.")

    report = _measure(
        args.source_artifact_root.expanduser().resolve(),
        args.reference_artifact_root.expanduser().resolve(),
        args.motion_split,
        torch.device(args.device),
        args.max_clips,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
