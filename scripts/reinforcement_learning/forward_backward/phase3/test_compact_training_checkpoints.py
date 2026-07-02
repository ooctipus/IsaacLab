# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for durable evaluation-checkpoint compaction and full-checkpoint retention."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).with_name("compact_training_checkpoints.py")


@pytest.fixture(scope="module")
def module():
    """Load the standalone checkpoint-compaction script."""
    spec = importlib.util.spec_from_file_location("compact_training_checkpoints", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _checkpoint(path: Path, iteration: int) -> None:
    torch.save(
        {
            "model_state_dict": {
                "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + iteration,
                "bias": torch.tensor([iteration], dtype=torch.float32),
            },
            "optimizer_state_dicts": {"actor": {"state": {}}},
            "iter": iteration,
            "collected_transitions": iteration * 1024,
        },
        path,
    )


def test_compact_checkpoint_publishes_exact_model_and_verified_manifest(tmp_path: Path, module) -> None:
    source = tmp_path / "model_12.pt"
    output_dir = tmp_path / "evaluation_milestones"
    _checkpoint(source, 12)

    manifest = module.compact_checkpoint(source, output_dir)

    output = output_dir / source.name
    saved = torch.load(output, map_location="cpu", mmap=True, weights_only=True)
    assert set(saved) == {"model_state_dict"}
    assert set(saved["model_state_dict"]) == {"weight", "bias"}
    assert manifest["iteration"] == 12
    assert manifest["collected_transitions"] == 12 * 1024
    assert manifest["model_tensor_count"] == 2
    assert manifest["model_scalar_count"] == 7
    assert manifest["output"]["bytes"] == output.stat().st_size
    assert module.compact_checkpoint(source, output_dir) == manifest


def test_discovery_waits_for_unchanged_checkpoint_bytes(tmp_path: Path, module) -> None:
    source = tmp_path / "model_1.pt"
    _checkpoint(source, 1)
    observations: dict[Path, tuple[int, int, float]] = {}

    assert module.discover_stable_checkpoints(tmp_path, observations, now=0.0, stable_seconds=10.0) == []
    assert module.discover_stable_checkpoints(tmp_path, observations, now=9.0, stable_seconds=10.0) == []
    source.write_bytes(source.read_bytes() + b"changed")
    assert module.discover_stable_checkpoints(tmp_path, observations, now=11.0, stable_seconds=10.0) == []
    assert module.discover_stable_checkpoints(tmp_path, observations, now=21.0, stable_seconds=10.0) == [source]


def test_retention_deletes_only_processed_superseded_full_checkpoints(tmp_path: Path, module) -> None:
    first = tmp_path / "model_10.pt"
    second = tmp_path / "model_20.pt"
    unprocessed = tmp_path / "model_30.pt"
    for iteration, path in ((10, first), (20, second), (30, unprocessed)):
        _checkpoint(path, iteration)

    processed = {first, second}
    removed = module.retain_latest_full_checkpoints(processed, keep_full=1)

    assert removed == [first]
    assert not first.exists()
    assert second.exists()
    assert unprocessed.exists()
    assert processed == {second}
