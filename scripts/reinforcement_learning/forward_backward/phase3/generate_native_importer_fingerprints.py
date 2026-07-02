# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fingerprint the native HumEnv HDF5 and BFM G1 joblib motion inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

HUMENV_ORIGINAL_NAME = "0-CMU_91_91_19_poses.hdf5"
HUMENV_SHA256 = "7b1580a40b2a9134dd25977b256747fbe08cdb9ef50ac501f0d911ba385d17e1"
BFM_ORIGINAL_NAME = "lafan_29dof_10s-clipped.pkl"
BFM_SHA256 = "7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010"


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: np.ndarray) -> str:
    """Hash one scalar or frame in C byte order without dtype conversion."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _field_fingerprint(value: object) -> dict[str, object]:
    """Describe one native field and sample its temporal boundary bytes."""
    if type(value) is int:
        return {
            "scalar_type": "python:int",
            "scalar_encoding": "UTF-8 decimal",
            "shape": [],
            "scalar_bytes_sha256": hashlib.sha256(str(value).encode()).hexdigest(),
        }
    if type(value) is str:
        return {
            "scalar_type": "python:str",
            "scalar_encoding": "UTF-8",
            "shape": [],
            "scalar_bytes_sha256": hashlib.sha256(value.encode()).hexdigest(),
        }
    array = np.asarray(value)
    result: dict[str, object] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    if array.ndim == 0:
        result["scalar_bytes_sha256"] = _bytes_sha256(array)
        return result
    if array.shape[0] < 1:
        raise ValueError("Native motion arrays must contain at least one frame.")
    indices = (0, array.shape[0] // 2, array.shape[0] - 1)
    result["frame_bytes"] = [
        {"position": position, "index": index, "sha256": _bytes_sha256(array[index])}
        for position, index in zip(("first", "middle", "last"), indices, strict=True)
    ]
    return result


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    """Hash a string sequence without discarding or normalizing its order."""
    encoded = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_file_identity(path: Path, expected_sha256: str) -> None:
    """Reject source bytes that differ from the frozen scientific input."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Unexpected SHA256 for {path.name}: {actual_sha256}.")


def capture_humenv_hdf5(path: Path) -> dict[str, object]:
    """Capture the exact field order, shapes, dtypes, and sampled HDF5 bytes."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("HumEnv HDF5 fingerprinting requires h5py.") from error

    _require_file_identity(path, HUMENV_SHA256)
    with h5py.File(path, "r") as source:
        episode_ids = list(source.keys())
        if episode_ids != ["ep_0"]:
            raise ValueError(f"Expected one HumEnv episode named ep_0, got {episode_ids}.")
        episode = source[episode_ids[0]]
        field_order = list(episode.keys())
        fields = {name: _field_fingerprint(episode[name][...]) for name in field_order}

    clip_id = Path(HUMENV_ORIGINAL_NAME).stem
    return {
        "source_file": {"name": HUMENV_ORIGINAL_NAME, "sha256": HUMENV_SHA256},
        "ordered_clip_ids": [clip_id],
        "ordered_clip_ids_sha256": _ordered_ids_sha256([clip_id]),
        "clips": [
            {
                "clip_id": clip_id,
                "native_episode_id": episode_ids[0],
                "field_order": field_order,
                "fields": fields,
            }
        ],
    }


def _ordered_mapping(value: object, description: str) -> Mapping[str, object]:
    """Require a string-keyed mapping while retaining its insertion order."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping, got {type(value).__name__}.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{description} keys must all be strings.")
    return value


def _capture_bfm_clip(clip_id: str, clip: object, position: int) -> dict[str, object]:
    """Fingerprint one selected BFM clip without reordering its native fields."""
    fields_by_name = _ordered_mapping(clip, f"BFM clip {clip_id}")
    field_order = list(fields_by_name)
    return {
        "position": position,
        "clip_id": clip_id,
        "field_order": field_order,
        "fields": {name: _field_fingerprint(fields_by_name[name]) for name in field_order},
    }


def capture_bfm_g1_joblib(path: Path) -> dict[str, object]:
    """Capture BFM insertion order and sampled clip bytes from the monolithic joblib."""
    try:
        import joblib
    except ImportError as error:
        raise ImportError("BFM G1 joblib fingerprinting requires joblib.") from error

    _require_file_identity(path, BFM_SHA256)
    clips = _ordered_mapping(joblib.load(path, mmap_mode="r"), "BFM G1 joblib root")
    ordered_clip_ids = list(clips)
    if not ordered_clip_ids:
        raise ValueError("The BFM G1 joblib contains no clips.")
    sampled_positions = (0, len(ordered_clip_ids) // 2, len(ordered_clip_ids) - 1)
    sampled_clips = [
        _capture_bfm_clip(ordered_clip_ids[position], clips[ordered_clip_ids[position]], position)
        for position in sampled_positions
    ]
    return {
        "source_file": {"name": BFM_ORIGINAL_NAME, "sha256": BFM_SHA256},
        "ordered_clip_ids": ordered_clip_ids,
        "ordered_clip_ids_sha256": _ordered_ids_sha256(ordered_clip_ids),
        "sampled_clips": sampled_clips,
    }


def capture(humenv_hdf5: Path, bfm_g1_joblib: Path) -> dict[str, object]:
    """Capture the complete immutable native-importer fingerprint document."""
    return {
        "format": "forward_backward_phase3c_native_importer_fingerprints_v1",
        "sampling": {
            "sampled_clip_positions": ["first", "middle=floor(clip_count/2)", "last"],
            "sampled_frame_positions": ["first", "middle=floor(frame_count/2)", "last"],
            "hash_input": "one declared-encoding scalar or one C-contiguous native-dtype frame",
        },
        "humenv_hdf5": capture_humenv_hdf5(humenv_hdf5),
        "bfm_g1_joblib": capture_bfm_g1_joblib(bfm_g1_joblib),
    }


def write_fingerprints(fingerprints: Mapping[str, object], output: Path) -> None:
    """Atomically write deterministic, human-readable JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(fingerprints, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humenv_hdf5", type=Path, required=True)
    parser.add_argument("--bfm_g1_joblib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Regenerate the frozen fingerprint fixture from exact native source bytes."""
    args = _parser().parse_args(argv)
    write_fingerprints(capture(args.humenv_hdf5, args.bfm_g1_joblib), args.output)


if __name__ == "__main__":
    main()
