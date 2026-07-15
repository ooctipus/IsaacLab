# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare intrinsic source-to-robot retargeting quality across the four motion cells."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch
from benchmark_nonlinear_iterations import count_trajectory_nonlinear_iterations

from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
    _TRAJECTORY_INSPECTION_QUALITY_PREFIX,
    _TRAJECTORY_INSPECTION_STAGE_NAMES,
    _TRAJECTORY_METRIC_NAMES,
)
from isaaclab_tasks.utils import resolve_presets

_QUALITY_ROUTE = "trajectory_route"
_QUALITY_ACCEPTED = "accepted"


def _statistics(values: torch.Tensor) -> dict[str, float | int | None]:
    """Summarize one finite scalar distribution."""
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Benchmark statistics require finite values.")
    quantiles = torch.quantile(values, torch.tensor((0.5, 0.95), dtype=torch.float64))
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "max": float(values.max()),
    }


def _ordered(values: torch.Tensor, sequences: object) -> torch.Tensor:
    """Return reset-state rows in flattened sequence order."""
    return values if sequences.state_indices is None else values.index_select(0, sequences.state_indices)


def _quality_summary(
    quality: object, clip_ids: tuple[str, ...]
) -> tuple[dict[str, object], dict[str, object] | None, torch.Tensor, torch.Tensor]:
    """Summarize the builder-owned quality columns without a parallel name registry."""
    if quality is None or quality.scope != "sequence":
        raise ValueError("Motion inspection requires sequence-scoped quality.")
    names = tuple(quality.names)
    values = quality.values.detach().to(device="cpu", dtype=torch.float64)
    if values.shape != (len(clip_ids), len(names)) or len(set(names)) != len(names):
        raise ValueError("Motion quality must contain one uniquely named row per source clip.")
    if _QUALITY_ROUTE not in names or _QUALITY_ACCEPTED not in names:
        raise ValueError("Motion quality must publish route and acceptance.")

    route = values[:, names.index(_QUALITY_ROUTE)]
    accepted_value = values[:, names.index(_QUALITY_ACCEPTED)]
    semantic = torch.stack((route, accepted_value), dim=1)
    if not bool(torch.isfinite(semantic).all() and torch.all((semantic == 0.0) | (semantic == 1.0))):
        raise ValueError("Motion route and acceptance must be finite binary values.")
    trajectory = route > 0.5
    accepted = accepted_value > 0.5

    def metric_summary(name: str) -> dict[str, float | int | None]:
        metric = values[:, names.index(name)]
        finite = metric[torch.isfinite(metric)]
        return {
            **_statistics(finite),
            "missing_clips": metric.numel() - finite.numel(),
        }

    expected_stage_names = tuple(
        f"{_TRAJECTORY_INSPECTION_QUALITY_PREFIX}{stage}/{metric}"
        for stage in _TRAJECTORY_INSPECTION_STAGE_NAMES
        for metric in _TRAJECTORY_METRIC_NAMES
    )
    actual_stage_names = tuple(name for name in names if name.startswith(_TRAJECTORY_INSPECTION_QUALITY_PREFIX))
    if actual_stage_names and actual_stage_names != expected_stage_names:
        raise ValueError("Motion inspection stage quality is partial, duplicated, or out of order.")
    stage_quality = None
    if actual_stage_names:
        stage_quality = {
            stage: {
                "metrics": {
                    metric: metric_summary(f"{_TRAJECTORY_INSPECTION_QUALITY_PREFIX}{stage}/{metric}")
                    for metric in _TRAJECTORY_METRIC_NAMES
                }
            }
            for stage in _TRAJECTORY_INSPECTION_STAGE_NAMES
        }
    metrics = {}
    for column, name in enumerate(names):
        if name in (_QUALITY_ROUTE, _QUALITY_ACCEPTED) or name.startswith(_TRAJECTORY_INSPECTION_QUALITY_PREFIX):
            continue
        metrics[name] = metric_summary(name)
    return (
        {
            "accepted_clips": int(accepted.sum()),
            "rejected_clips": int((~accepted).sum()),
            "stored_coordinate_clips": int((~trajectory).sum()),
            "trajectory_clips": int(trajectory.sum()),
            "metrics": metrics,
        },
        stage_quality,
        accepted,
        trajectory,
    )


def _root_motion(
    view: object, clip_ids: tuple[str, ...]
) -> tuple[dict[str, dict[str, float | int | None]], list[float]]:
    """Measure root motion and rates without differencing across clip boundaries."""
    sequences = view.sequences
    if sequences.frame_dt is None or sequences.sequence_count != len(clip_ids):
        raise ValueError("Root-motion reporting requires one physical clock per clip.")
    root_pose = _ordered(view.state_bank.root_pose, sequences).detach().to(device="cpu", dtype=torch.float64)
    root_velocity = _ordered(view.state_bank.root_velocity, sequences).detach().to(device="cpu", dtype=torch.float64)
    joint_velocity = _ordered(view.state_bank.joint_velocity, sequences).detach().to(device="cpu", dtype=torch.float64)
    if root_pose.shape[1:] != (1, 7) or root_velocity.shape[1:] != (1, 6):
        raise ValueError("Motion benchmarking requires exactly one free-root entity.")
    root_pose = root_pose[:, 0]
    root_velocity = root_velocity[:, 0]
    offsets = sequences.offsets.detach().cpu()
    frame_dt = sequences.frame_dt.detach().to(device="cpu", dtype=torch.float64)
    samples: dict[str, list[torch.Tensor]] = {
        "horizontal_displacement_m": [],
        "horizontal_path_m": [],
        "net_yaw_rad": [],
        "yaw_path_rad": [],
        "root_linear_speed_mps": [],
        "root_angular_speed_radps": [],
        "derived_root_speed_mps": [],
        "derived_yaw_rate_radps": [],
        "root_linear_acceleration_mps2": [],
        "root_angular_acceleration_radps2": [],
        "joint_acceleration_abs_radps2": [],
    }
    mean_speeds = []
    for clip_index in range(len(clip_ids)):
        start = int(offsets[clip_index])
        stop = int(offsets[clip_index + 1])
        step = float(frame_dt[clip_index])
        pose = root_pose[start:stop]
        velocity = root_velocity[start:stop]
        joint_rate = joint_velocity[start:stop]
        if pose.shape[0] == 0 or step <= 0.0:
            raise ValueError("Motion clips must contain at least one frame and a positive clock.")

        x, y, z, w = pose[:, 3:7].unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        position_delta = pose[1:, :3] - pose[:-1, :3]
        yaw_delta = torch.atan2(torch.sin(yaw[1:] - yaw[:-1]), torch.cos(yaw[1:] - yaw[:-1]))
        displacement = torch.linalg.vector_norm(pose[-1, :2] - pose[0, :2])
        path = torch.linalg.vector_norm(position_delta[:, :2], dim=-1).sum()
        duration = (pose.shape[0] - 1) * step
        mean_speed = 0.0 if duration == 0.0 else float(path / duration)
        mean_speeds.append(mean_speed)
        samples["horizontal_displacement_m"].append(displacement.reshape(1))
        samples["horizontal_path_m"].append(path.reshape(1))
        samples["net_yaw_rad"].append(yaw_delta.sum().reshape(1))
        samples["yaw_path_rad"].append(yaw_delta.abs().sum().reshape(1))
        samples["root_linear_speed_mps"].append(torch.linalg.vector_norm(velocity[:, :3], dim=-1))
        samples["root_angular_speed_radps"].append(torch.linalg.vector_norm(velocity[:, 3:], dim=-1))
        if position_delta.numel():
            samples["derived_root_speed_mps"].append(torch.linalg.vector_norm(position_delta / step, dim=-1))
            samples["derived_yaw_rate_radps"].append((yaw_delta / step).abs())
            acceleration = (velocity[1:] - velocity[:-1]) / step
            samples["root_linear_acceleration_mps2"].append(torch.linalg.vector_norm(acceleration[:, :3], dim=-1))
            samples["root_angular_acceleration_radps2"].append(torch.linalg.vector_norm(acceleration[:, 3:], dim=-1))
            samples["joint_acceleration_abs_radps2"].append(((joint_rate[1:] - joint_rate[:-1]) / step).abs())
    summary = {
        name: _statistics(torch.cat(values) if values else torch.empty(0, dtype=torch.float64))
        for name, values in samples.items()
    }
    return summary, mean_speeds


def _landmark_summary(view: object) -> dict[str, dict[str, float | int | None]] | None:
    """Compare trajectory landmarks in world, root-relative, and heading-quotient geometry."""
    target_item = next((item for item in view.points if item.name == "target_landmarks"), None)
    solved_item = next((item for item in view.points if item.name == "solved_robot_landmarks"), None)
    if target_item is None and solved_item is None:
        return None
    if target_item is None or solved_item is None:
        raise ValueError("Target and solved landmark evidence must be published together.")
    target = _ordered(target_item.points, view.sequences).detach().to(device="cpu", dtype=torch.float64)
    solved = _ordered(solved_item.points, view.sequences).detach().to(device="cpu", dtype=torch.float64)
    if target.shape != solved.shape:
        raise ValueError("Target and solved landmark evidence must share one shape.")

    world_error = torch.linalg.vector_norm(solved - target, dim=-1)
    target_relative = target - target[:, :1]
    solved_relative = solved - solved[:, :1]
    root_error = torch.linalg.vector_norm(solved_relative - target_relative, dim=-1)
    cosine = torch.sum(
        solved_relative[..., 0] * target_relative[..., 0] + solved_relative[..., 1] * target_relative[..., 1],
        dim=1,
    )
    sine = torch.sum(
        solved_relative[..., 0] * target_relative[..., 1] - solved_relative[..., 1] * target_relative[..., 0],
        dim=1,
    )
    yaw = torch.atan2(sine, cosine)
    aligned = solved_relative.clone()
    aligned[..., 0] = (
        torch.cos(yaw)[:, None] * solved_relative[..., 0] - torch.sin(yaw)[:, None] * solved_relative[..., 1]
    )
    aligned[..., 1] = (
        torch.sin(yaw)[:, None] * solved_relative[..., 0] + torch.cos(yaw)[:, None] * solved_relative[..., 1]
    )
    return {
        "world_error_m": _statistics(world_error),
        "root_relative_error_m": _statistics(root_error),
        "heading_aligned_error_m": _statistics(torch.linalg.vector_norm(aligned - target_relative, dim=-1)),
    }


def _contact_summary(
    view: object, contact_patches: tuple[object, ...], clip_ids: tuple[str, ...]
) -> tuple[dict[str, object] | None, list[str]]:
    """Measure per-channel contact coverage and intervals without joining adjacent clips."""
    contact_points = next((item for item in view.points if item.name == "contact_points"), None)
    if contact_points is None:
        return None, ["not_applicable"] * len(clip_ids)
    if contact_points.valid is None:
        raise ValueError("Contact points must publish source-owned activity.")
    valid = _ordered(contact_points.valid, view.sequences).detach().cpu()
    point_counts = tuple(int(patch.points_per_body) for patch in contact_patches)
    names = tuple(patch.channel for patch in contact_patches)
    if valid.shape[1] != sum(point_counts):
        raise ValueError("Contact activity width differs from configured contact patches.")
    active = []
    cursor = 0
    for count in point_counts:
        patch_valid = valid[:, cursor : cursor + count]
        channel_valid = patch_valid[:, :1]
        if not torch.equal(patch_valid, channel_valid.expand_as(patch_valid)):
            raise ValueError("Every contact patch must repeat one source-owned activity mask across its points.")
        active.append(channel_valid[:, 0])
        cursor += count
    active = torch.stack(active, dim=1)

    offsets = view.sequences.offsets.detach().cpu()
    frame_dt = view.sequences.frame_dt.detach().to(device="cpu", dtype=torch.float64)
    coverage: list[list[torch.Tensor]] = [[] for _ in names]
    durations: list[list[torch.Tensor]] = [[] for _ in names]
    interval_counts = [0] * len(names)
    patterns = []
    for clip_index in range(len(clip_ids)):
        start = int(offsets[clip_index])
        stop = int(offsets[clip_index + 1])
        step = float(frame_dt[clip_index])
        clip_active = active[start:stop]
        used = torch.any(clip_active, dim=0)
        used_indices = torch.nonzero(used, as_tuple=False).flatten()
        if used_indices.numel() == 0:
            patterns.append("none")
        elif used_indices.numel() == 1:
            patterns.append(names[int(used_indices[0])])
        elif bool(torch.any(clip_active[:, used_indices].sum(dim=1) > 1)):
            patterns.append("simultaneous")
        else:
            patterns.append("alternating")
        for channel in range(len(names)):
            mask = clip_active[:, channel]
            coverage[channel].append(mask.to(torch.float64).mean().reshape(1))
            starts = mask & ~torch.cat((mask.new_zeros(1), mask[:-1]))
            stops = mask & ~torch.cat((mask[1:], mask.new_zeros(1)))
            lengths = torch.nonzero(stops, as_tuple=False).flatten() + 1
            lengths -= torch.nonzero(starts, as_tuple=False).flatten()
            interval_counts[channel] += lengths.numel()
            if lengths.numel():
                durations[channel].append(lengths.to(torch.float64) * step)
    channels = {}
    for channel, name in enumerate(names):
        channel_duration = torch.cat(durations[channel]) if durations[channel] else torch.empty(0)
        channels[name] = {
            "clip_coverage_fraction": _statistics(torch.cat(coverage[channel])),
            "interval_count": interval_counts[channel],
            "interval_duration_s": _statistics(channel_duration),
        }
    return {"channels": channels}, patterns


def _rejection_summary(
    clips: tuple[object, ...],
    accepted: torch.Tensor,
    trajectory: torch.Tensor,
    mean_speeds: list[float],
    contact_patterns: list[str],
) -> dict[str, object]:
    """Stratify acceptance by route, skeleton, speed, and source contact pattern."""
    speed_bands = ["stationary" if speed < 0.1 else "ordinary" if speed < 1.0 else "fast" for speed in mean_speeds]
    labels = {
        "route": ["trajectory" if bool(value) else "stored_coordinates" for value in trajectory],
        "skeleton": [str(clip.skeleton_id) for clip in clips],
        "speed": speed_bands,
        "contact": contact_patterns,
    }

    def group(values: list[str]) -> dict[str, dict[str, float | int]]:
        counts: dict[str, list[int]] = {}
        for value, is_accepted in zip(values, accepted.tolist(), strict=True):
            row = counts.setdefault(value, [0, 0])
            row[0] += 1
            row[1] += int(is_accepted)
        return {
            value: {
                "clips": count,
                "accepted": accepted_count,
                "accepted_fraction": accepted_count / count,
            }
            for value, (count, accepted_count) in sorted(counts.items())
        }

    rejected = [
        {
            "clip_id": clip.clip_id,
            "route": labels["route"][clip_index],
            "speed": speed_bands[clip_index],
            "contact": contact_patterns[clip_index],
        }
        for clip_index, clip in enumerate(clips)
        if not bool(accepted[clip_index])
    ]
    return {
        "speed_bands_mps": {"stationary_upper": 0.1, "ordinary_upper": 1.0},
        "by": {name: group(values) for name, values in labels.items()},
        "rejected_clips": rejected,
    }


def _rss_bytes() -> int:
    """Return current process resident host memory [bytes] on Linux."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _build_report(
    robot: str,
    source: str,
    source_artifact_root: Path,
    target_artifact_root: Path | None,
    motion_split: str,
    device: torch.device,
    inspection_limit: int | None = None,
) -> dict[str, object]:
    """Build and analyze one complete acceptance run or limited inspection."""
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg

    if inspection_limit is not None and (type(inspection_limit) is not int or inspection_limit < 1):
        raise ValueError("inspection_limit must be a positive integer when provided.")
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={robot, source})
    table_cfg = cfg.commands.motion.task_table
    source_artifact_root = source_artifact_root.expanduser().resolve()
    table_cfg.source_artifact_root = str(source_artifact_root)
    if target_artifact_root is None:
        if table_cfg.target_kinematics.calibration is not None:
            raise ValueError("A calibrated target requires --target_artifact_root.")
    else:
        table_cfg.target_artifact_root = str(target_artifact_root.expanduser().resolve())
    table_cfg.motion_split = motion_split
    split = table_cfg.source.train if motion_split == "train" else table_cfg.source.evaluation
    clip_source = table_cfg.source.open_split(source_artifact_root, split)
    try:
        index = clip_source.inspect()
    finally:
        clip_source.close()
    selected_count = len(index.clips) if inspection_limit is None else min(inspection_limit, len(index.clips))
    selected_clips = index.clips[:selected_count]
    clip_ids = index.clip_ids[:selected_count]
    input_frames = index.offsets[selected_count]

    resident_before = _rss_bytes()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        allocated_before = None
    with count_trajectory_nonlinear_iterations() as nonlinear_iterations:
        started = time.perf_counter()
        view = table_cfg.build_inspection_view(
            cfg.commands.motion, cfg.scene, str(device), sequence_limit=selected_count
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        construction_seconds = time.perf_counter() - started
    retarget_stages = nonlinear_iterations.stage_report()

    analysis_started = time.perf_counter()
    quality, stage_quality, accepted, trajectory = _quality_summary(view.quality, clip_ids)
    motion, mean_speeds = _root_motion(view, clip_ids)
    landmarks = _landmark_summary(view)
    contacts, contact_patterns = _contact_summary(view, tuple(table_cfg.target_kinematics.contact_patches), clip_ids)
    rejection = _rejection_summary(selected_clips, accepted, trajectory, mean_speeds, contact_patterns)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    analysis_seconds = time.perf_counter() - analysis_started
    resident_after = _rss_bytes()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss = int(peak_rss if os.uname().sysname == "Darwin" else peak_rss * 1024)
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    output_frames = view.sequences.frame_count

    return {
        "composition": {"robot": robot, "source": source},
        "run": {
            "scope": "full_acceptance" if inspection_limit is None else "limited_inspection",
            "accepting": (
                inspection_limit is None and quality["rejected_clips"] == 0 and output_frames == input_frames
            ),
        },
        "corpus": {
            "source_identifier": table_cfg.source.identifier,
            "split": split.name,
            "declared_clips": len(index.clips),
            "declared_frames": index.offsets[-1],
            "inspection_limit": inspection_limit,
            "clips": selected_count,
            "input_frames": input_frames,
            "output_frames": output_frames,
        },
        "quality": quality,
        "stage_quality": stage_quality,
        "motion": motion,
        "landmarks": landmarks,
        "contacts": contacts,
        "rejection": rejection,
        "performance": {
            "device": str(device),
            "construction_seconds": construction_seconds,
            "analysis_seconds": analysis_seconds,
            "input_frames_per_construction_second": input_frames / construction_seconds,
            "output_frames_per_construction_second": output_frames / construction_seconds,
            "host_resident_incremental_bytes": max(0, resident_after - resident_before),
            "host_peak_resident_bytes": peak_rss,
            "cuda_peak_incremental_allocated_bytes": (
                None if allocated_before is None else peak_allocated - allocated_before
            ),
            "cuda_peak_reserved_bytes": peak_reserved,
            "trajectory_nonlinear_iterations": nonlinear_iterations.report(),
            "retarget_stages": retarget_stages,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one independent source-target benchmark request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("smpl", "g1"), required=True)
    parser.add_argument("--source", choices=("cmu", "lafan"), required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--target_artifact_root", type=Path)
    parser.add_argument("--motion_split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--inspection_limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.inspection_limit is not None and args.inspection_limit < 1:
        parser.error("--inspection_limit must be positive")
    return args


def main() -> None:
    """Run one full intrinsic retargeting benchmark and atomically persist its report."""
    args = _parse_args()
    report = _build_report(
        args.robot,
        args.source,
        args.source_artifact_root,
        args.target_artifact_root,
        args.motion_split,
        torch.device(args.device),
        inspection_limit=args.inspection_limit,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "quality": {
                    "accepted_clips": report["quality"]["accepted_clips"],
                    "rejected_clips": report["quality"]["rejected_clips"],
                },
                "performance": report["performance"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
