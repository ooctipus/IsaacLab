# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare compact SMPL mechanics and deterministic concrete AMASS clip rows."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from isaaclab_tasks.core.multi_task.motion.data.sources.amass_smplh import AmassClipRow, write_amass_clip_rows


def prepare_model(source: Path, output: Path, gender: str) -> None:
    """Convert one licensed SMPL pickle into the compact runtime archive."""
    with source.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
        source_file.seek(0)
        model = pickle.load(source_file, encoding="latin1")
    required = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table")
    if any(name not in model for name in required):
        raise ValueError(f"SMPL pickle is missing one of the required fields: {required}.")

    kintree = np.asarray(model["kintree_table"])
    if kintree.shape != (2, 24):
        raise ValueError("SMPL kinematic tree must have shape [2, 24].")
    body_ids = [int(value) for value in kintree[1]]
    id_to_index = {body_id: index for index, body_id in enumerate(body_ids)}
    parents = [-1]
    parents.extend(id_to_index[int(parent_id)] for parent_id in kintree[0, 1:])

    joint_regressor = model["J_regressor"]
    if hasattr(joint_regressor, "toarray"):
        joint_regressor = joint_regressor.toarray()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.array("smpl_lbs_v1"),
        gender=np.array(gender),
        source_sha256=np.array(source_sha256),
        vertex_template_m=np.ascontiguousarray(model["v_template"], dtype=np.float32),
        shape_blend_directions_m=np.ascontiguousarray(model["shapedirs"][..., :10], dtype=np.float32),
        pose_blend_directions_m=np.ascontiguousarray(model["posedirs"], dtype=np.float32),
        joint_regressor=np.ascontiguousarray(joint_regressor, dtype=np.float32),
        skinning_weights=np.ascontiguousarray(model["weights"], dtype=np.float32),
        parent_indices=np.ascontiguousarray(parents, dtype=np.int64),
    )


def prepare_split(prepared_selection: Path, raw_root: Path, output: Path) -> tuple[int, str]:
    """Translate one frozen prepared selection into ordered concrete NPZ rows."""
    selected = tuple(line.strip() for line in prepared_selection.read_text(encoding="utf-8").splitlines())
    if not selected or any(not name for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("Prepared AMASS selection must contain unique nonempty clip names.")

    raw_by_prepared_name: dict[str, Path] = {}
    for raw_path in sorted(raw_root.rglob("*_poses.npz")):
        relative = raw_path.relative_to(raw_root)
        prepared_name = f"0-CMU_{relative.parent.name}_{raw_path.stem}.hdf5"
        if prepared_name in raw_by_prepared_name:
            raise ValueError(f"Prepared clip name maps to multiple raw files: {prepared_name}")
        raw_by_prepared_name[prepared_name] = relative
    missing = tuple(name for name in selected if name not in raw_by_prepared_name)
    if missing:
        raise FileNotFoundError(f"Prepared clips have no raw AMASS file: {missing[:3]}")

    relative_paths = tuple(raw_by_prepared_name[name] for name in selected)
    rows = tuple(
        AmassClipRow.from_file(relative_path.as_posix(), raw_root / relative_path) for relative_path in relative_paths
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_amass_clip_rows(output, rows)
    return len(rows), hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    """Run one explicit AMASS preparation operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    model = subparsers.add_parser("model", help="Convert one licensed SMPL model.")
    model.add_argument("source", type=Path, help="Licensed SMPL model pickle.")
    model.add_argument("output", type=Path, help="Destination compact NPZ archive.")
    model.add_argument("--gender", required=True, choices=("female", "male", "neutral"))

    split = subparsers.add_parser("split", help="Create concrete clip rows from a prepared selection.")
    split.add_argument("prepared_selection", type=Path, help="Frozen ordered prepared clip selection.")
    split.add_argument("raw_root", type=Path, help="Raw CMU directory containing *_poses.npz clips.")
    split.add_argument("output", type=Path, help="Destination concrete clip-row CSV file.")

    args = parser.parse_args()
    if args.operation == "model":
        prepare_model(args.source, args.output, args.gender)
    else:
        count, digest = prepare_split(args.prepared_selection, args.raw_root, args.output)
        print(f"{args.output}: {count} clips, sha256={digest}")


if __name__ == "__main__":
    main()
