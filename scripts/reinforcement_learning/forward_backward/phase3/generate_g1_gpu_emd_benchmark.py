# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the full-workload G1 exact-assignment boundary on one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.motion.tracking import _UniformEmdWorkspace

_CLIP_COUNT = 862
_FRAME_COUNT = 499
_ROLLOUT_CHUNK_SIZES = (380, 380, 102)
_FEATURE_WIDTHS = (29, 23)


def _sha256(path: Path) -> str:
    """Return the digest of one regular evidence input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _workspace_bytes(capacity: int, frames: int, feature_width: int) -> int:
    """Return fixed bytes allocated by :class:`_UniformEmdWorkspace`."""
    return (
        4 * capacity * frames * frames
        + 2 * 4 * capacity * frames
        + 2 * 4 * capacity * frames * feature_width
        + 3 * 8 * capacity * (frames + 1)
        + 3 * 4 * capacity * (frames + 1)
        + 3 * 8 * capacity
    )


def _device_uuid(properties: object) -> str:
    """Return Torch's physical UUID in NVIDIA's conventional representation."""
    value = str(getattr(properties, "uuid", ""))
    if len(value) != 36:
        raise RuntimeError("Torch did not expose a physical CUDA device UUID.")
    return f"GPU-{value}"


def _measure(
    workspace: _UniformEmdWorkspace,
    observed: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
) -> tuple[float, int]:
    """Measure one full assignment call and its incremental allocator peak."""
    device = observed.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    workspace.compute(observed, target, output)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    peak_increment = torch.cuda.max_memory_allocated(device) - baseline
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Exact G1 assignment produced a nonfinite cost.")
    return seconds, peak_increment


def main() -> None:
    """Run the fixed benchmark and atomically write its identity-bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The G1 EMD benchmark requires one CUDA device.")
    torch.cuda.set_device(device)
    wp.init()

    length_values = (_FRAME_COUNT,) * _CLIP_COUNT
    workspace = _UniformEmdWorkspace(lengths=length_values, device=device, feature_width=max(_FEATURE_WIDTHS))
    output = torch.empty(_CLIP_COUNT, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    inputs = {
        width: (
            torch.randn(_CLIP_COUNT, _FRAME_COUNT, width, generator=generator, device=device),
            torch.randn(_CLIP_COUNT, _FRAME_COUNT, width, generator=generator, device=device),
        )
        for width in _FEATURE_WIDTHS
    }

    warm_observed, warm_target = inputs[max(_FEATURE_WIDTHS)]
    workspace.compute(warm_observed, warm_target, output)
    torch.cuda.synchronize(device)

    calls = []
    repeated_call_peak_bytes = 0
    for width in _FEATURE_WIDTHS:
        observed, target = inputs[width]
        seconds, peak_increment = _measure(workspace, observed, target, output)
        repeated_call_peak_bytes = max(repeated_call_peak_bytes, peak_increment)
        calls.append(
            {
                "batch_size": _CLIP_COUNT,
                "feature_width": width,
                "seconds": seconds,
            }
        )

    root = Path(__file__).resolve().parents[4]
    tracking = root / "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/motion/tracking.py"
    kernel = root / "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/motion/impl/uniform_emd_warp.py"
    properties = torch.cuda.get_device_properties(device)
    total_seconds = sum(float(call["seconds"]) for call in calls)
    report = {
        "schema": "g1_lafan_gpu_emd_benchmark_v2",
        "recorded_at": datetime.now(UTC).isoformat(),
        "contract": {
            "assignment": "exact_shortest_augmenting_path",
            "cost": "released_float32_euclidean_distance",
            "tie_policy": "lowest_unused_column",
            "motion_assignment": "released_random_shuffle_first_env",
            "motion_assignment_rng": "evaluation_transaction_python_rng",
            "rollout_horizon": "maximum_unique_clip_length_per_chunk",
            "trajectory_device": "cuda",
            "host_trajectory_copies": 0,
            "per_step_host_synchronizations": 0,
            "workspace_bytes": _workspace_bytes(_CLIP_COUNT, _FRAME_COUNT, max(_FEATURE_WIDTHS)),
            "repeated_call_peak_increment_bytes": repeated_call_peak_bytes,
        },
        "workload": {
            "clip_count": _CLIP_COUNT,
            "frame_count_per_clip": _FRAME_COUNT,
            "rollout_chunk_sizes": list(_ROLLOUT_CHUNK_SIZES),
            "assignment_batch_size": _CLIP_COUNT,
            "feature_widths": list(_FEATURE_WIDTHS),
            "assignment_calls": len(_FEATURE_WIDTHS),
        },
        "measurements": {
            "device": {
                "name": properties.name,
                "uuid": _device_uuid(properties),
                "torch_version": torch.__version__,
                "warp_version": wp.__version__,
            },
            "full_two_call_seconds": total_seconds,
            "calls": calls,
        },
        "implementation_sha256": {
            "producer.py": _sha256(Path(__file__).resolve()),
            "tracking.py": _sha256(tracking),
            "uniform_emd_warp.py": _sha256(kernel),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
