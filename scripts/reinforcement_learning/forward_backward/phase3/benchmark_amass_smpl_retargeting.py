# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark raw AMASS-SMPL conversion against released HumEnv trajectories.

This report keeps pose parity separate from the intentional velocity-policy
difference: raw conversion derives branch-safe analytic rates, while the exact
prepared route preserves released native generalized velocities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from isaaclab_tasks.core.multi_task.kinematics import time_quaternion_angular_velocity_segmented
from isaaclab_tasks.core.multi_task.motion.mdp.commands import build_motion_task_table
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
    build_motion_task_table_oracle,
)
from isaaclab_tasks.core.multi_task.motion.robots.smpl.observations import (
    smpl_humenv_observation,
    smpl_humenv_tracking_pose,
)
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

_METAMOTIVO_FPS = 30.0


def _config(
    dataset: str,
    source_artifact_root: Path,
    target_artifact_root: Path,
    motion_split: str,
) -> MotionImitationEnvCfg:
    """Resolve one source route against explicit source and target artifact roots."""
    selected = {"smpl", dataset, "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows"}
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=selected)
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(source_artifact_root)
    table_cfg.target_artifact_root = str(target_artifact_root)
    table_cfg.motion_split = motion_split
    return cfg


def _build(cfg: MotionImitationEnvCfg, label: str, device: torch.device, *, oracle: bool = False):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    builder = build_motion_task_table_oracle if oracle else build_motion_task_table
    table = builder(cfg.commands.motion, cfg.scene, str(device))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    summary = (
        f"{label}: clips={len(table.clip_index.clips)} frames={table.clip_index.total_frames} "
        f"time={elapsed:.3f}s peak_delta={(peak - before) / 2**20:.3f}MiB"
    )
    print(summary, flush=True)
    return table, elapsed, peak - before


def _aligned_delta(name: str, actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    if name.endswith("rotation"):
        sign = torch.where(torch.sum(actual * expected, dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
        actual = actual * sign
    return actual - expected


def _aggregate(table, oracle, fields):
    rows = {}
    device = table.field(fields[0]).device
    total_abs = torch.zeros((), dtype=torch.float64, device=device)
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    total_count = 0
    maximum = 0.0
    for name in fields:
        actual = table.field(name)
        expected = oracle.field(name)
        if actual.shape != expected.shape:
            raise RuntimeError(f"{name} shape mismatch: {actual.shape} != {expected.shape}")
        delta = _aligned_delta(name, actual, expected).to(torch.float64)
        absolute = delta.abs()
        count = absolute.numel()
        sum_abs = absolute.sum()
        sum_sq = torch.sum(delta * delta)
        field_max = absolute.amax()
        rows[name] = {
            "mean_absolute_error": float((sum_abs / count).cpu()),
            "root_mean_square_error": float(torch.sqrt(sum_sq / count).cpu()),
            "maximum_absolute_error": float(field_max.cpu()),
            "elements": count,
        }
        total_abs += sum_abs
        total_sq += sum_sq
        total_count += count
        maximum = max(maximum, rows[name]["maximum_absolute_error"])
        del delta, absolute
    return {
        "mean_absolute_error": float((total_abs / total_count).cpu()),
        "root_mean_square_error": float(torch.sqrt(total_sq / total_count).cpu()),
        "maximum_absolute_error": maximum,
        "elements": total_count,
        "fields": rows,
    }


def _tensor_profile(values: torch.Tensor, thresholds: tuple[float, ...]) -> dict:
    """Summarize element magnitudes and per-row maxima."""
    absolute = values.abs().to(torch.float64)
    row_maximum = absolute.flatten(1).amax(dim=1)
    return {
        "maximum_absolute": float(absolute.amax().cpu()),
        "mean_absolute": float(absolute.mean().cpu()),
        "root_mean_square": float(torch.sqrt(torch.mean(absolute.square())).cpu()),
        "rows": values.shape[0],
        "elements": values.numel(),
        "row_counts_over": {str(value): int((row_maximum > value).sum().cpu()) for value in thresholds},
    }


def _branch_profile(table, offsets: torch.Tensor) -> dict:
    """Summarize coordinate-branch jumps and their stored generalized velocities."""
    device = table.field("joint_position").device
    edge_rows = torch.arange(table.clip_index.total_frames, dtype=torch.int64, device=device)
    valid = torch.ones_like(edge_rows, dtype=torch.bool)
    valid[offsets[1:] - 1] = False
    edge_rows = edge_rows[valid]
    coordinate_jump = table.field("joint_position").index_select(0, edge_rows + 1) - table.field(
        "joint_position"
    ).index_select(0, edge_rows)
    jump_maximum = coordinate_jump.abs().amax(dim=1)
    branch = jump_maximum > math.pi
    velocity_maximum = table.field("joint_velocity").index_select(0, edge_rows).abs().amax(dim=1)
    branch_velocity = velocity_maximum[branch]
    return {
        "valid_edges": edge_rows.shape[0],
        "branch_jump_edges_over_pi": int(branch.sum().cpu()),
        "maximum_coordinate_jump_rad": float(jump_maximum.amax().cpu()),
        "maximum_joint_velocity_rad_per_s": float(velocity_maximum.amax().cpu()),
        "maximum_joint_velocity_on_branch_edges_rad_per_s": (
            float(branch_velocity.amax().cpu()) if branch_velocity.numel() else 0.0
        ),
        "branch_edges_with_joint_velocity_over_100_rad_per_s": int((branch_velocity > 100.0).sum().cpu()),
    }


def _normalized_raw(line: str) -> str:
    return Path(line).with_suffix("").as_posix().replace("/", "_").removesuffix("_poses")


def _normalized_prepared(line: str) -> str:
    return Path(line).stem.removeprefix("0-CMU_").removesuffix("_poses")


@dataclass(frozen=True, slots=True)
class _PhaseZeroPairing:
    """Exact MetaMotivo target-rate rows selected from the current output."""

    raw_indices: torch.Tensor
    strides: tuple[int, ...]
    full_decimated_counts: tuple[int, ...]
    truncated_clip_indices: tuple[int, ...]


def _phase_zero_pairing(raw_index: object, oracle_index: object, device: torch.device) -> _PhaseZeroPairing:
    """Map each 30 Hz oracle row to PHC's phase-zero source row."""
    if len(raw_index.clips) != len(oracle_index.clips):
        raise RuntimeError("Current and MetaMotivo clip counts differ.")
    indices = []
    strides = []
    full_decimated_counts = []
    truncated_clip_indices = []
    for clip_index, (raw_clip, oracle_clip) in enumerate(zip(raw_index.clips, oracle_index.clips, strict=True)):
        if not math.isclose(float(oracle_clip.source_fps), _METAMOTIVO_FPS, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(f"MetaMotivo clip {clip_index} is not sampled at 30 Hz.")
        sample_ratio = float(raw_clip.source_fps) / _METAMOTIVO_FPS
        stride = round(sample_ratio)
        if stride < 1 or not math.isclose(sample_ratio, stride, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(f"Current clip {clip_index} does not have an integer 30 Hz sampling stride.")
        full_count = (int(raw_clip.frame_count) + stride - 1) // stride
        oracle_count = int(oracle_clip.frame_count)
        if oracle_count > full_count:
            raise RuntimeError(
                f"MetaMotivo clip {clip_index} has {oracle_count} rows, exceeding its "
                f"{full_count}-row phase-zero decimated span."
            )
        if oracle_count < full_count:
            truncated_clip_indices.append(clip_index)
        indices.append(
            int(raw_index.offsets[clip_index]) + torch.arange(oracle_count, dtype=torch.int64, device=device) * stride
        )
        strides.append(stride)
        full_decimated_counts.append(full_count)
    return _PhaseZeroPairing(
        raw_indices=torch.cat(tuple(indices)),
        strides=tuple(strides),
        full_decimated_counts=tuple(full_decimated_counts),
        truncated_clip_indices=tuple(truncated_clip_indices),
    )


def _manifest_pairing(
    raw_artifact_root: Path, oracle_artifact_root: Path, motion_split: str
) -> tuple[tuple[str, ...], int]:
    """Pair production raw rows with released prepared rows by normalized clip name."""
    split_name = "train" if motion_split == "train" else "test"
    raw_manifest = raw_artifact_root / f"data_preparation/test_train_split/0-CMU_{split_name}_raw.csv"
    with raw_manifest.open(newline="") as stream:
        raw_rows = tuple(csv.DictReader(stream))
    prepared_manifest = oracle_artifact_root / f"data_preparation/test_train_split/0-CMU_{split_name}_0.1.txt"
    prepared_rows = tuple(line for line in prepared_manifest.read_text().splitlines() if line.strip())
    raw_names = tuple(_normalized_raw(row["relative_path"]) for row in raw_rows)
    prepared_names = tuple(_normalized_prepared(line) for line in prepared_rows)
    if len(set(raw_names)) != len(raw_names) or len(set(prepared_names)) != len(prepared_names):
        raise RuntimeError("Raw/prepared split pairing requires unique normalized clip names.")
    if raw_names != prepared_names:
        mismatch = next(
            (index for index, pair in enumerate(zip(raw_names, prepared_names, strict=False)) if pair[0] != pair[1]),
            min(len(raw_names), len(prepared_names)),
        )
        raise RuntimeError(f"Raw/prepared split pairing differs at row {mismatch}.")
    return raw_names, sum(int(row["source_frame_count"]) for row in raw_rows)


def main(
    raw_artifact_root: Path,
    oracle_artifact_root: Path,
    motion_split: str,
    output: Path,
    device: torch.device,
) -> None:
    """Build both production routes and write the layered comparison report."""
    if device.type != "cuda":
        raise ValueError("The AMASS benchmark requires CUDA for production conversion and memory timing.")
    torch.cuda.set_device(device)
    raw_artifact_root = raw_artifact_root.expanduser().resolve()
    oracle_artifact_root = oracle_artifact_root.expanduser().resolve()
    raw_cfg = _config("cmu", raw_artifact_root, raw_artifact_root, motion_split)
    prepared_cfg = _config("humenv_cmu", oracle_artifact_root, raw_artifact_root, motion_split)
    raw_source_cfg = raw_cfg.commands.motion.task_table.source
    prepared_source_cfg = prepared_cfg.commands.motion.task_table.source
    raw_split_cfg = raw_source_cfg.train if motion_split == "train" else raw_source_cfg.evaluation
    prepared_split_cfg = prepared_source_cfg.train if motion_split == "train" else prepared_source_cfg.evaluation
    raw_names, raw_input_frames = _manifest_pairing(raw_artifact_root, oracle_artifact_root, motion_split)
    clip_count = len(raw_names)
    if (
        raw_split_cfg is None
        or prepared_split_cfg is None
        or clip_count != raw_split_cfg.clip_count
        or clip_count != prepared_split_cfg.clip_count
        or raw_input_frames != raw_split_cfg.frame_count
    ):
        raise RuntimeError("Resolved source declarations do not match the paired split manifests.")
    print(f"split pairing: {clip_count}/{clip_count} names agree", flush=True)

    raw, raw_seconds, raw_peak = _build(raw_cfg, "raw-smpl", device)
    raw_peak_bytes = int(raw_peak)
    oracle, oracle_seconds, _ = _build(prepared_cfg, "prepared-smpl", device, oracle=True)
    if (
        len(raw.clip_index.clips) != clip_count
        or len(oracle.clip_index.clips) != clip_count
        or oracle.clip_index.total_frames != prepared_split_cfg.frame_count
    ):
        raise RuntimeError("Built tables do not match their declared evaluation-corpus sizes.")
    raw_counts = torch.tensor([clip.frame_count for clip in raw.clip_index.clips], device=device)
    oracle_counts = torch.tensor([clip.frame_count for clip in oracle.clip_index.clips], device=device)
    pairing = _phase_zero_pairing(raw.clip_index, oracle.clip_index, device)
    full_decimated_counts = torch.tensor(pairing.full_decimated_counts, device=device)
    full_decimated_span = oracle_counts == full_decimated_counts
    paired = SimpleNamespace(field=lambda name: raw.field(name).index_select(0, pairing.raw_indices))

    coordinate_error = {
        "pose_rows": _aggregate(paired, oracle, ("root_position", "root_rotation", "joint_position")),
    }
    fk_error = {
        "pose_rows": _aggregate(paired, oracle, ("body_position", "body_rotation")),
    }

    joint_delta = paired.field("joint_position") - oracle.field("joint_position")
    wrapped_joint_delta = torch.atan2(torch.sin(joint_delta), torch.cos(joint_delta)).to(torch.float64)
    modular_joint_position_error = {
        "mean_absolute_error": float(wrapped_joint_delta.abs().mean().cpu()),
        "root_mean_square_error": float(torch.sqrt(torch.mean(wrapped_joint_delta.square())).cpu()),
        "maximum_absolute_error": float(wrapped_joint_delta.abs().amax().cpu()),
        "elements": wrapped_joint_delta.numel(),
    }
    joint_velocity_row_error = (paired.field("joint_velocity") - oracle.field("joint_velocity")).abs().amax(dim=1)
    frame_clip = torch.repeat_interleave(torch.arange(clip_count, device=device), oracle_counts)
    oracle_offsets = torch.tensor(oracle.clip_index.offsets, dtype=torch.int64, device=device)
    truncated_clip_indices = torch.tensor(pairing.truncated_clip_indices, dtype=torch.int64, device=device)
    truncated_terminal_rows = oracle_offsets.index_select(0, truncated_clip_indices + 1) - 1
    shared_edge_rows = torch.ones(oracle.clip_index.total_frames, dtype=torch.bool, device=device)
    shared_edge_rows[oracle_offsets[1:] - 1] = False
    valid_paired = SimpleNamespace(field=lambda name: paired.field(name)[shared_edge_rows])
    valid_oracle = SimpleNamespace(field=lambda name: oracle.field(name)[shared_edge_rows])
    velocity_fields = ("root_linear_velocity", "root_angular_velocity", "joint_velocity")
    fk_velocity_fields = ("body_linear_velocity", "body_angular_velocity")
    coordinate_error["velocity_rows_with_shared_next_frame"] = _aggregate(valid_paired, valid_oracle, velocity_fields)
    fk_error["velocity_rows_with_shared_next_frame"] = _aggregate(valid_paired, valid_oracle, fk_velocity_fields)
    truncated_velocity_error = None
    if truncated_terminal_rows.numel():
        terminal_paired = SimpleNamespace(
            field=lambda name: paired.field(name).index_select(0, truncated_terminal_rows)
        )
        terminal_oracle = SimpleNamespace(
            field=lambda name: oracle.field(name).index_select(0, truncated_terminal_rows)
        )
        truncated_velocity_error = _aggregate(terminal_paired, terminal_oracle, (*velocity_fields, *fk_velocity_fields))
    truncated_terminal_error = {
        "clip_indices": list(pairing.truncated_clip_indices),
        "clip_names": [raw_names[index] for index in pairing.truncated_clip_indices],
        "reason": "MetaMotivo prefix ends before the full phase-zero decimated span",
        "velocity_error": truncated_velocity_error,
    }
    valid_joint_velocity_row_error = joint_velocity_row_error[shared_edge_rows]
    valid_frame_clip = frame_clip[shared_edge_rows]
    clip_velocity_max = torch.zeros(clip_count, device=device)
    clip_velocity_max.scatter_reduce_(
        0, valid_frame_clip, valid_joint_velocity_row_error, reduce="amax", include_self=True
    )
    top_values, top_indices = torch.topk(clip_velocity_max, 10)
    velocity_outliers = {
        "rows_over_1e-3": int((valid_joint_velocity_row_error > 1.0e-3).sum().cpu()),
        "rows_over_1_rad_per_s": int((valid_joint_velocity_row_error > 1.0).sum().cpu()),
        "row_fraction_over_1_rad_per_s": float((valid_joint_velocity_row_error > 1.0).float().mean().cpu()),
        "clips_over_1_rad_per_s": int((clip_velocity_max > 1.0).sum().cpu()),
        "top_clips": [
            {"clip_index": int(index), "clip_name": raw_names[int(index)], "maximum_error": float(value)}
            for value, index in zip(top_values.cpu(), top_indices.cpu(), strict=True)
        ],
    }

    raw_offsets = torch.tensor(raw.clip_index.offsets, dtype=torch.int64, device=device)
    raw_steps = torch.tensor(
        [1.0 / clip.source_fps for clip in raw.clip_index.clips], dtype=torch.float32, device=device
    )
    prepared_steps = torch.tensor(
        [1.0 / clip.source_fps for clip in oracle.clip_index.clips], dtype=torch.float32, device=device
    )
    raw_edge_rows = torch.ones(raw.clip_index.total_frames, dtype=torch.bool, device=device)
    raw_edge_rows[raw_offsets[1:] - 1] = False
    prepared_edge_rows = torch.ones(oracle.clip_index.total_frames, dtype=torch.bool, device=device)
    prepared_edge_rows[oracle_offsets[1:] - 1] = False
    raw_rotation_velocity = time_quaternion_angular_velocity_segmented(
        raw.field("body_rotation"), raw_offsets, raw_steps
    )
    prepared_rotation_velocity = time_quaternion_angular_velocity_segmented(
        oracle.field("body_rotation"), oracle_offsets, prepared_steps
    )
    velocity_profiles = {
        "current_native": {
            "joint_velocity": _tensor_profile(raw.field("joint_velocity"), (30.0, 100.0, 180.0)),
            "body_angular_velocity": _tensor_profile(raw.field("body_angular_velocity"), (10.0, 30.0, 100.0)),
            "body_linear_velocity": _tensor_profile(raw.field("body_linear_velocity"), (10.0, 30.0)),
            "coordinate_branches": _branch_profile(raw, raw_offsets),
        },
        "prepared_exact_native": {
            "joint_velocity": _tensor_profile(oracle.field("joint_velocity"), (30.0, 100.0, 180.0)),
            "body_angular_velocity": _tensor_profile(oracle.field("body_angular_velocity"), (10.0, 30.0, 100.0)),
            "body_linear_velocity": _tensor_profile(oracle.field("body_linear_velocity"), (10.0, 30.0)),
            "coordinate_branches": _branch_profile(oracle, oracle_offsets),
        },
    }
    physical_velocity_consistency = {
        "current_native_body_angular_vs_next_pose": _tensor_profile(
            raw.field("body_angular_velocity")[raw_edge_rows] - raw_rotation_velocity[raw_edge_rows],
            (0.1, 1.0, 10.0),
        ),
        "prepared_native_body_angular_vs_next_pose": _tensor_profile(
            oracle.field("body_angular_velocity")[prepared_edge_rows] - prepared_rotation_velocity[prepared_edge_rows],
            (0.1, 1.0, 10.0),
        ),
    }

    raw_observation = smpl_humenv_observation(
        paired.field("body_position"),
        paired.field("body_rotation"),
        paired.field("body_linear_velocity"),
        paired.field("body_angular_velocity"),
    )
    oracle_observation = smpl_humenv_observation(
        oracle.field("body_position"),
        oracle.field("body_rotation"),
        oracle.field("body_linear_velocity"),
        oracle.field("body_angular_velocity"),
    )
    observation_delta = (raw_observation[shared_edge_rows] - oracle_observation[shared_edge_rows]).to(torch.float64)
    observation_error = {
        "mean_absolute_error": float(observation_delta.abs().mean().cpu()),
        "root_mean_square_error": float(torch.sqrt(torch.mean(observation_delta * observation_delta)).cpu()),
        "maximum_absolute_error": float(observation_delta.abs().amax().cpu()),
        "elements": observation_delta.numel(),
    }
    tracking_delta = (smpl_humenv_tracking_pose(raw_observation) - smpl_humenv_tracking_pose(oracle_observation)).to(
        torch.float64
    )
    tracking_observation_error = {
        "mean_absolute_error": float(tracking_delta.abs().mean().cpu()),
        "root_mean_square_error": float(torch.sqrt(torch.mean(tracking_delta.square())).cpu()),
        "maximum_absolute_error": float(tracking_delta.abs().amax().cpu()),
        "elements": tracking_delta.numel(),
    }

    quality = raw.view.quality
    if quality is None or quality.scope != "sequence":
        raise RuntimeError("Raw table does not expose sequence quality.")
    accepted_index = quality.names.index("accepted")
    accepted = quality.values[:, accepted_index] > 0.5
    result = {
        "inputs": {
            "raw_artifact_root": str(raw_artifact_root),
            "oracle_artifact_root": str(oracle_artifact_root),
            "motion_split": motion_split,
            "oracle": "released_humenv_metamotivo_training_motion",
        },
        "source": {
            "raw_input_clips": clip_count,
            "raw_input_frames": raw_input_frames,
            "paired_oracle_clips": clip_count,
            "current_output_frames": raw.clip_index.total_frames,
            "oracle_output_frames": oracle.clip_index.total_frames,
            "all_pair_names_match": True,
            "pairing_clock_hz": _METAMOTIVO_FPS,
            "phase_zero_mapping": "current_offset + oracle_row * round(current_fps / 30)",
            "phase_zero_stride_clip_counts": {
                str(stride): pairing.strides.count(stride) for stride in sorted(set(pairing.strides))
            },
            "full_decimated_span_clips": int(full_decimated_span.sum().cpu()),
            "oracle_truncated_prefix_clips": len(pairing.truncated_clip_indices),
            "oracle_truncated_target_frames": int((full_decimated_counts - oracle_counts).sum().cpu()),
            "truncated_prefixes": [
                {
                    "clip_index": index,
                    "clip_name": raw_names[index],
                    "raw_frames": int(raw_counts[index].cpu()),
                    "raw_fps": float(raw.clip_index.clips[index].source_fps),
                    "phase_zero_stride": pairing.strides[index],
                    "full_decimated_frames": pairing.full_decimated_counts[index],
                    "oracle_frames": int(oracle_counts[index].cpu()),
                    "omitted_decimated_tail_frames": (
                        pairing.full_decimated_counts[index] - int(oracle_counts[index].cpu())
                    ),
                }
                for index in pairing.truncated_clip_indices
            ],
        },
        "coverage": {
            "accepted_clips": int(accepted.sum().cpu()),
            "declared_clips": clip_count,
            "fraction": float(accepted.float().mean().cpu()),
        },
        "performance": {
            "raw_build_seconds": raw_seconds,
            "raw_input_frames_per_second": raw_input_frames / raw_seconds,
            "output_frames_per_second": raw.clip_index.total_frames / raw_seconds,
            "peak_incremental_gpu_bytes": raw_peak_bytes,
            "peak_incremental_gpu_mib": raw_peak_bytes / 2**20,
            "prepared_oracle_build_seconds": oracle_seconds,
        },
        "report_semantics": {
            "pose_fields": "current phase-zero 30 Hz rows versus released HumEnv/MetaMotivo rows",
            "current_velocity": "native current-route velocity sampled at paired timestamps",
            "oracle_velocity": "released native generalized velocity preserved by the exact route",
            "velocity_difference": "intentional policy difference, not an equality oracle",
        },
        "pose_parity": {
            "robot_coordinates": coordinate_error["pose_rows"],
            "modulo_2pi_joint_position": modular_joint_position_error,
            "forward_kinematics": fk_error["pose_rows"],
            "tracking_pose_observation": tracking_observation_error,
        },
        "velocity_policy_difference_on_shared_valid_edges": {
            "shared_edge_rows": int(shared_edge_rows.sum().cpu()),
            "robot_coordinates": coordinate_error["velocity_rows_with_shared_next_frame"],
            "forward_kinematics": fk_error["velocity_rows_with_shared_next_frame"],
            "policy_observation": observation_error,
            "joint_velocity_outliers": velocity_outliers,
        },
        "velocity_profiles": velocity_profiles,
        "physical_angular_velocity_consistency": physical_velocity_consistency,
        "oracle_truncated_prefix_terminal_rows": truncated_terminal_error,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(serialized, end="", flush=True)


def _parse_args() -> argparse.Namespace:
    """Parse the licensed-corpus benchmark inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_artifact_root",
        type=Path,
        required=True,
        help="Root containing registered raw AMASS CMU artifacts and target SMPL calibration.",
    )
    parser.add_argument(
        "--oracle_artifact_root",
        type=Path,
        required=True,
        help="Root containing released HumEnv/MetaMotivo CMU artifacts.",
    )
    parser.add_argument("--motion_split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used for conversion and measurement.")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/amass_smpl_full_report.json"), help="Output JSON path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.raw_artifact_root,
        args.oracle_artifact_root,
        args.motion_split,
        args.output,
        torch.device(args.device),
    )
