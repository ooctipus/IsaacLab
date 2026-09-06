# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate deterministic ground-motion clip rows from the official LAFAN1 zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from isaaclab_tasks.core.multi_task.motion.data.sources.lafan_bvh import (
    LafanClipRow,
    decode_lafan_bvh,
    lafan_clip_rows_content_sha256,
    write_lafan_clip_rows,
)

_LAFAN_ZIP_SHA256 = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"
_GROUND_PREFIXES = ("dance", "fallAndGetUp", "fight", "fightAndSports", "jumps", "run", "sprint", "walk")
_WINDOW_FRAMES = 300


def _rows(zip_path: Path) -> tuple[tuple[LafanClipRow, ...], tuple[LafanClipRow, ...]]:
    """Return nonoverlapping 300-frame training windows and full evaluation clips."""
    training = []
    evaluation = []
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        selected = tuple(name for name in names if Path(name).stem.startswith(_GROUND_PREFIXES))
        if len(selected) != 40 or any(Path(name).name != name for name in selected):
            raise ValueError("Official LAFAN ground categories must resolve to 40 root-level BVH members.")
        for member in selected:
            source_bytes = archive.read(member)
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            clip = decode_lafan_bvh(source_bytes)
            stem = Path(member).stem
            evaluation.append(
                LafanClipRow(
                    stem,
                    member,
                    source_sha256,
                    clip.frame_count,
                    clip.source_fps,
                    0,
                    clip.frame_count,
                )
            )
            for window in range(clip.frame_count // _WINDOW_FRAMES):
                start = window * _WINDOW_FRAMES
                training.append(
                    LafanClipRow(
                        f"{stem}_clip{window}",
                        member,
                        source_sha256,
                        clip.frame_count,
                        clip.source_fps,
                        start,
                        start + _WINDOW_FRAMES,
                    )
                )
    return tuple(training), tuple(evaluation)


def main() -> None:
    """Write both row files and print their immutable identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Official lafan1.zip path.")
    parser.add_argument("--output_dir", type=Path, default=Path.cwd(), help="Destination directory.")
    args = parser.parse_args()
    with args.zip_path.open("rb") as zip_file:
        zip_sha256 = hashlib.file_digest(zip_file, "sha256").hexdigest()
    if zip_sha256 != _LAFAN_ZIP_SHA256:
        raise ValueError(f"Official LAFAN zip digest differs: expected {_LAFAN_ZIP_SHA256}, got {zip_sha256}.")
    training, evaluation = _rows(args.zip_path)
    if len(training) != 862 or sum(row.frame_count for row in training) != 258_600:
        raise ValueError("LAFAN training-window law must produce 862 clips and 258,600 frames.")
    if len(evaluation) != 40 or sum(row.frame_count for row in evaluation) != 264_705:
        raise ValueError("LAFAN evaluation rows must produce 40 full clips and 264,705 frames.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in (("lafan_ground_train.csv", training), ("lafan_ground_evaluation.csv", evaluation)):
        path = args.output_dir / name
        write_lafan_clip_rows(path, rows)
        outputs[name] = {
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_content_sha256": lafan_clip_rows_content_sha256(zip_sha256, rows),
            "clip_count": len(rows),
            "frame_count": sum(row.frame_count for row in rows),
        }
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
