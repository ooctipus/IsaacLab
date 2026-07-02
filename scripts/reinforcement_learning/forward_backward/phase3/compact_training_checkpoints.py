# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retain one full training checkpoint and compact evaluation milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

import torch

_CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt")
_MANIFEST_SCHEMA = "forward_backward_evaluation_checkpoint_v1"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoint_iteration(path: Path) -> int:
    """Return the numeric iteration encoded by a checkpoint filename."""
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Checkpoint name must match model_<iteration>.pt: {path}.")
    return int(match.group(1))


def _write_json_atomic(path: Path, value: object) -> None:
    """Write one JSON object through an atomic same-directory replacement."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    """Flush directory-entry updates for one local output directory."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_existing_milestone(checkpoint: Path, manifest_path: Path) -> dict[str, object]:
    """Validate one previously published compact checkpoint and its manifest."""
    if checkpoint.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Compact checkpoints and manifests must not be symbolic links.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError(f"Invalid compact checkpoint manifest: {manifest_path}.")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"Compact checkpoint manifest has no output record: {manifest_path}.")
    if output.get("bytes") != checkpoint.stat().st_size or output.get("sha256") != _sha256(checkpoint):
        raise ValueError(f"Compact checkpoint bytes differ from its manifest: {checkpoint}.")
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {"model_state_dict"}:
        raise ValueError(f"Compact checkpoint must contain only model_state_dict: {checkpoint}.")
    model = saved["model_state_dict"]
    if (
        not isinstance(model, Mapping)
        or not model
        or not all(isinstance(value, torch.Tensor) for value in model.values())
    ):
        raise ValueError(f"Compact checkpoint model_state_dict is invalid: {checkpoint}.")
    return manifest


def compact_checkpoint(source: Path, output_dir: Path) -> dict[str, object]:
    """Publish and verify one evaluation-only checkpoint from a trusted full checkpoint."""
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Full checkpoint must be a regular non-symbolic file: {source}.")
    iteration = _checkpoint_iteration(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / source.name
    manifest_path = output.with_suffix(".json")
    if output.is_file() and manifest_path.is_file():
        return _validate_existing_milestone(output, manifest_path)

    source_before = source.stat()
    saved = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(saved, Mapping):
        raise ValueError(f"Full checkpoint root must be a mapping: {source}.")
    model = saved.get("model_state_dict")
    if (
        not isinstance(model, Mapping)
        or not model
        or not all(isinstance(value, torch.Tensor) for value in model.values())
    ):
        raise ValueError(f"Full checkpoint has no tensor-only model_state_dict: {source}.")
    saved_iteration = saved.get("iter")
    if saved_iteration != iteration:
        raise ValueError(f"Checkpoint iteration {saved_iteration!r} differs from filename iteration {iteration}.")
    collected_transitions = saved.get("collected_transitions")
    if (
        not isinstance(collected_transitions, int)
        or isinstance(collected_transitions, bool)
        or collected_transitions < 0
    ):
        raise ValueError(f"Checkpoint collected_transitions is invalid: {source}.")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save({"model_state_dict": model}, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    compact = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(compact, dict) or set(compact) != {"model_state_dict"}:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Temporary compact checkpoint failed exact-key validation: {temporary}.")
    compact_model = compact["model_state_dict"]
    if not isinstance(compact_model, Mapping) or tuple(compact_model) != tuple(model):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Temporary compact checkpoint changed model state keys: {temporary}.")
    for name, value in model.items():
        restored = compact_model[name]
        if restored.shape != value.shape or restored.dtype != value.dtype:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Temporary compact checkpoint changed tensor {name!r}: {temporary}.")

    source_after = source.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Full checkpoint changed while it was being compacted: {source}.")
    source_sha256 = _sha256(source)
    source_final = source.stat()
    if (
        source_before.st_size,
        source_before.st_mtime_ns,
    ) != (source_final.st_size, source_final.st_mtime_ns):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Full checkpoint changed while it was being hashed: {source}.")
    output_sha256 = _sha256(temporary)
    model_scalar_count = sum(value.numel() for value in model.values())
    manifest: dict[str, object] = {
        "schema": _MANIFEST_SCHEMA,
        "iteration": iteration,
        "collected_transitions": collected_transitions,
        "model_tensor_count": len(model),
        "model_scalar_count": model_scalar_count,
        "source": {
            "filename": source.name,
            "bytes": source_after.st_size,
            "sha256": source_sha256,
        },
        "output": {
            "filename": output.name,
            "bytes": temporary.stat().st_size,
            "sha256": output_sha256,
        },
    }
    os.replace(temporary, output)
    _write_json_atomic(manifest_path, manifest)
    _fsync_directory(output_dir)
    _validate_existing_milestone(output, manifest_path)
    return manifest


def discover_stable_checkpoints(
    run_dir: Path,
    observations: dict[Path, tuple[int, int, float]],
    *,
    now: float,
    stable_seconds: float,
) -> list[Path]:
    """Return checkpoints whose size and modification time remained unchanged."""
    current: dict[Path, tuple[int, int, float]] = {}
    ready: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        if path.is_symlink():
            raise ValueError(f"Training checkpoint must not be a symbolic link: {path}.")
        if not path.is_file():
            continue
        iteration = _checkpoint_iteration(path)
        stat = path.stat()
        previous = observations.get(path)
        unchanged_since = (
            previous[2] if previous is not None and previous[:2] == (stat.st_size, stat.st_mtime_ns) else now
        )
        current[path] = (stat.st_size, stat.st_mtime_ns, unchanged_since)
        if stat.st_size > 0 and now - unchanged_since >= stable_seconds:
            ready.append((iteration, path))
    observations.clear()
    observations.update(current)
    return [path for _iteration, path in sorted(ready)]


def retain_latest_full_checkpoints(processed: set[Path], *, keep_full: int) -> list[Path]:
    """Delete processed full checkpoints older than the newest retained checkpoints."""
    if keep_full < 1:
        raise ValueError("keep_full must be positive.")
    checkpoints = sorted(
        ((_checkpoint_iteration(path), path) for path in processed if path.is_file()),
        key=lambda item: item[0],
    )
    removed: list[Path] = []
    for _iteration, path in checkpoints[:-keep_full]:
        path.unlink()
        processed.remove(path)
        removed.append(path)
    return removed


def watch(
    run_dir: Path,
    *,
    poll_seconds: float,
    stable_seconds: float,
    keep_full: int,
    final_iteration: int | None,
) -> None:
    """Watch a run directory until stopped or until its final milestone is compacted."""
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"Run directory must be a regular directory: {run_dir}.")
    if poll_seconds <= 0.0 or stable_seconds < 0.0:
        raise ValueError("poll_seconds must be positive and stable_seconds must be non-negative.")
    output_dir = run_dir / "evaluation_milestones"
    observations: dict[Path, tuple[int, int, float]] = {}
    processed: set[Path] = set()
    while True:
        ready = discover_stable_checkpoints(
            run_dir,
            observations,
            now=time.monotonic(),
            stable_seconds=stable_seconds,
        )
        for source in ready:
            if source in processed:
                continue
            manifest = compact_checkpoint(source, output_dir)
            processed.add(source)
            print(
                f"compacted iteration={manifest['iteration']} transitions={manifest['collected_transitions']} "
                f"source={source.name}",
                flush=True,
            )
        for removed in retain_latest_full_checkpoints(processed, keep_full=keep_full):
            print(f"removed superseded full checkpoint {removed.name}", flush=True)
        if final_iteration is not None and any(
            manifest.name == f"model_{final_iteration}.json" for manifest in output_dir.glob("model_*.json")
        ):
            print(f"final iteration {final_iteration} compacted; watcher complete", flush=True)
            return
        time.sleep(poll_seconds)


def main() -> None:
    """Parse command-line arguments and run the checkpoint watcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument("--stable_seconds", type=float, default=60.0)
    parser.add_argument("--keep_full", type=int, default=1)
    parser.add_argument("--final_iteration", type=int)
    args = parser.parse_args()
    watch(
        args.run_dir.expanduser().resolve(),
        poll_seconds=args.poll_seconds,
        stable_seconds=args.stable_seconds,
        keep_full=args.keep_full,
        final_iteration=args.final_iteration,
    )


if __name__ == "__main__":
    main()
