# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-GPU ownership evidence for isolated Phase 3 measurements."""

from __future__ import annotations

import os
import subprocess


def _compute_rows(device: str) -> tuple[int, str, list[tuple[str, int]]]:
    """Return the owner PID, its one physical GPU UUID, and all host compute rows."""
    if not device.startswith("cuda"):
        raise RuntimeError("GPU-isolated evidence requires a CUDA device.")
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[str, int]] = []
    for line in completed.stdout.splitlines():
        fields = tuple(value.strip() for value in line.split(","))
        if len(fields) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi compute row: {line!r}.")
        rows.append((fields[0], int(fields[1])))

    owner_pid = os.getpid()
    owner_uuids = sorted({gpu_uuid for gpu_uuid, pid in rows if pid == owner_pid})
    if len(owner_uuids) != 1:
        raise RuntimeError(
            "Evidence process must own exactly one physical GPU context, "
            f"found UUIDs {owner_uuids} for PID {owner_pid}."
        )
    return owner_pid, owner_uuids[0], rows


def exclusive_physical_gpu_snapshot(device: str) -> dict[str, object]:
    """Return and enforce sole compute-process ownership of one physical GPU."""
    owner_pid, physical_gpu_uuid, rows = _compute_rows(device)
    compute_pids = sorted({pid for gpu_uuid, pid in rows if gpu_uuid == physical_gpu_uuid})
    competing_compute_pids = [pid for pid in compute_pids if pid != owner_pid]
    if competing_compute_pids:
        raise RuntimeError(f"Physical GPU {physical_gpu_uuid} is shared with compute PIDs {competing_compute_pids}.")
    return {
        "physical_gpu_uuid": physical_gpu_uuid,
        "owner_pid": owner_pid,
        "compute_pids": compute_pids,
        "competing_compute_pids": competing_compute_pids,
        "exclusive": True,
    }


def validate_same_exclusive_gpu(*snapshots: dict[str, object]) -> str:
    """Require multiple ownership snapshots from one PID and physical UUID."""
    if not snapshots:
        raise ValueError("At least one GPU ownership snapshot is required.")
    first = snapshots[0]
    uuid = first.get("physical_gpu_uuid")
    pid = first.get("owner_pid")
    for snapshot in snapshots:
        if (
            snapshot.get("physical_gpu_uuid") != uuid
            or snapshot.get("owner_pid") != pid
            or snapshot.get("compute_pids") != [pid]
            or snapshot.get("competing_compute_pids") != []
            or snapshot.get("exclusive") is not True
        ):
            raise RuntimeError("Evidence did not retain sole ownership of one physical GPU.")
    if not isinstance(uuid, str):
        raise RuntimeError("Physical GPU UUID is absent.")
    return uuid


__all__ = ["exclusive_physical_gpu_snapshot", "validate_same_exclusive_gpu"]
