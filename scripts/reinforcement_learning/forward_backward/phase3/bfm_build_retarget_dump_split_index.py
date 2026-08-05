# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Freeze one retarget dump split index from the campaign dump MANIFEST (bfm-converter-20260805).

Reads MANIFEST.json, keeps ``accepted_full`` clips only (fail-closed), verifies
every payload file hash and clip identity against the manifest, computes the
converter clock (integer stride to 30 Hz), and writes the immutable
``<set>.split_index_v1.json`` artifact whose sha256 becomes the trainer-side
SplitCfg pin. Pure CPU; run once per dump set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from isaaclab_tasks.core.multi_task.motion.data.sources.retarget_dump_v5 import (
    INDEX_SCHEMA,
    PAYLOAD_SCHEMA,
    resolve_stride,
)

_SET_ROUTES = {
    "cmu_smpl_train": "cmu_smpl",
    "cmu_smpl_test": "cmu_smpl",
    "lafan_g1_train": "lafan_g1",
    "lafan_g1_eval": "lafan_g1",
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_index(dumps_root: Path, set_name: str) -> tuple[Path, dict[str, object]]:
    """Build and write one frozen split index; returns its path and summary pins."""
    route = _SET_ROUTES[set_name]
    manifest_path = dumps_root / "MANIFEST.json"
    manifest_sha256 = _file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    record = manifest["sets"][set_name]
    dump_dir = Path(record["dump_dir"])
    if dump_dir != dumps_root / set_name:
        raise ValueError(f"Manifest dump_dir {dump_dir} differs from the expected layout.")

    clips = []
    excluded = []
    raw_frames_accepted = 0
    output_frames = 0
    for clip_id in sorted(record["clips"]):
        entry = record["clips"][clip_id]
        if not entry["accepted_full"]:
            excluded.append(clip_id)
            continue
        file_name = f"{set_name}/{clip_id}.pt"
        path = dumps_root / file_name
        actual_sha = _file_sha256(path)
        if actual_sha != entry["payload_sha256"]:
            raise ValueError(f"Payload hash differs from the manifest for {clip_id!r}: {actual_sha}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_clip_id = payload["clip_ids"][0]
        frame_dt_s = float(payload["frame_dt_s"][0])
        if (
            payload["schema"] != PAYLOAD_SCHEMA
            or len(payload["clip_ids"]) != 1
            or payload_clip_id.replace("/", "_") != clip_id
            or int(payload["frame_count"]) != int(entry["frames"])
            or not bool(payload["accepted"][0])
            or abs(frame_dt_s - float(entry["frame_dt_s"])) > 1.0e-9
        ):
            raise ValueError(f"Payload identity differs from the manifest for {clip_id!r}.")
        stride = resolve_stride(frame_dt_s, route)
        output_frame_count = len(range(0, int(entry["frames"]), stride))
        if output_frame_count < 3:
            raise ValueError(f"Accepted clip {clip_id!r} is too short at the converter clock.")
        raw_frames_accepted += int(entry["frames"])
        output_frames += output_frame_count
        clips.append(
            {
                "clip_id": clip_id,
                "payload_clip_id": payload_clip_id,
                "file": file_name,
                "payload_sha256": entry["payload_sha256"],
                "q_sha256": entry["q_sha256"],
                "frame_count": int(entry["frames"]),
                "frame_dt_s": frame_dt_s,
                "ground_shift_m": float(entry["ground_shift_m"]),
                "stride": stride,
                "output_frame_count": output_frame_count,
            }
        )

    index = {
        "schema": INDEX_SCHEMA,
        "payload_schema": PAYLOAD_SCHEMA,
        "set": set_name,
        "route": route,
        "manifest_sha256": manifest_sha256,
        "run_json_sha256": record["run_json_sha256"],
        "engine_commit": record["engine_commit"],
        "target_fps": 30.0,
        "excluded_fail_closed": excluded,
        "totals": {
            "accepted_clips": len(clips),
            "accepted_raw_frames": raw_frames_accepted,
            "accepted_output_frames": output_frames,
            "excluded_clips": len(excluded),
        },
        "clips": clips,
    }
    output_path = dumps_root / f"{set_name}.split_index_v1.json"
    output_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    pins = {
        "set": set_name,
        "artifact": output_path.name,
        "artifact_sha256": _file_sha256(output_path),
        "clip_count": len(clips),
        "frame_count": output_frames,
        "raw_frame_count": raw_frames_accepted,
        "excluded": len(excluded),
    }
    return output_path, pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps_root", type=Path, required=True)
    parser.add_argument("--set", dest="set_name", choices=tuple(_SET_ROUTES), action="append", required=True)
    args = parser.parse_args()
    for set_name in args.set_name:
        _path, pins = build_index(args.dumps_root.expanduser().resolve(), set_name)
        print(json.dumps(pins, indent=1))


if __name__ == "__main__":
    main()
