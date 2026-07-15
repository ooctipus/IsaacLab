# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the raw-AMASS versus prepared-HumEnv benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).with_name("benchmark_amass_smpl_retargeting.py")


@pytest.fixture(scope="module")
def benchmark():
    """Load the benchmark module without invoking its licensed-corpus entry point."""
    spec = importlib.util.spec_from_file_location("benchmark_amass_smpl_retargeting_tested", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifests(raw_root: Path, oracle_root: Path, prepared_rows: tuple[str, ...], split_name: str) -> None:
    raw_manifest_root = raw_root / "data_preparation" / "test_train_split"
    oracle_manifest_root = oracle_root / "data_preparation" / "test_train_split"
    raw_manifest_root.mkdir(parents=True)
    oracle_manifest_root.mkdir(parents=True)
    (raw_manifest_root / f"0-CMU_{split_name}_raw.csv").write_text(
        "relative_path,source_frame_count\n01/walk_poses.npz,12\n02/run_poses.npz,34\n"
    )
    (oracle_manifest_root / f"0-CMU_{split_name}_0.1.txt").write_text("\n".join(prepared_rows) + "\n")


def _index(clips: tuple[tuple[int, float], ...]) -> SimpleNamespace:
    """Build one minimal clip index for target-rate pairing tests."""
    offsets = [0]
    values = []
    for frame_count, source_fps in clips:
        values.append(SimpleNamespace(frame_count=frame_count, source_fps=source_fps))
        offsets.append(offsets[-1] + frame_count)
    return SimpleNamespace(clips=tuple(values), offsets=tuple(offsets))


def test_manifest_pairing_uses_separate_roots_and_selected_split(benchmark, tmp_path: Path) -> None:
    """Pairing must preserve normalized train order across independent artifact roots."""
    ordered = ("0-CMU_01_walk_poses.hdf5", "0-CMU_02_run_poses.hdf5")
    raw_root = tmp_path / "raw"
    oracle_root = tmp_path / "oracle"
    _write_manifests(raw_root, oracle_root, ordered, "train")

    assert benchmark._manifest_pairing(raw_root, oracle_root, "train") == (("01_walk", "02_run"), 46)

    prepared = oracle_root / "data_preparation" / "test_train_split" / "0-CMU_train_0.1.txt"
    prepared.write_text("\n".join(reversed(ordered)) + "\n")
    with pytest.raises(RuntimeError, match="differs at row 0"):
        benchmark._manifest_pairing(raw_root, oracle_root, "train")


def test_phase_zero_pairing_matches_phc_decimation_at_60_and_120_hz(benchmark) -> None:
    """Every oracle row must select the same phase-zero native row as PHC."""
    raw = _index(((402, 60.0), (316, 120.0)))
    oracle = _index(((201, 30.0), (79, 30.0)))

    pairing = benchmark._phase_zero_pairing(raw, oracle, torch.device("cpu"))

    assert pairing.strides == (2, 4)
    assert pairing.full_decimated_counts == (201, 79)
    assert pairing.truncated_clip_indices == ()
    assert torch.equal(pairing.raw_indices[:201], torch.arange(0, 402, 2))
    assert torch.equal(pairing.raw_indices[201:], 402 + torch.arange(0, 316, 4))


def test_phase_zero_pairing_reports_oracle_truncated_prefix(benchmark) -> None:
    """A shorter oracle remains an explicit prefix of the exact decimated clock."""
    pairing = benchmark._phase_zero_pairing(_index(((12, 60.0),)), _index(((4, 30.0),)), torch.device("cpu"))

    assert torch.equal(pairing.raw_indices, torch.tensor((0, 2, 4, 6)))
    assert pairing.full_decimated_counts == (6,)
    assert pairing.truncated_clip_indices == (0,)


@pytest.mark.parametrize(
    ("raw_clips", "oracle_clips", "message"),
    (
        ((((12, 50.0),)), (((6, 30.0),)), "integer 30 Hz sampling stride"),
        ((((12, 60.0),)), (((7, 30.0),)), "exceeding its 6-row"),
        ((((12, 60.0),)), (((6, 60.0),)), "not sampled at 30 Hz"),
    ),
)
def test_phase_zero_pairing_rejects_incompatible_clocks_and_counts(
    benchmark, raw_clips: tuple[tuple[int, float], ...], oracle_clips: tuple[tuple[int, float], ...], message: str
) -> None:
    """Impossible target-rate pairings must fail instead of silently realigning."""
    with pytest.raises(RuntimeError, match=message):
        benchmark._phase_zero_pairing(_index(raw_clips), _index(oracle_clips), torch.device("cpu"))


def test_profiles_count_only_in_clip_branch_edges_and_row_thresholds(benchmark) -> None:
    """Branch counts must exclude clip boundaries and thresholds must count per-row maxima."""
    fields = {
        "joint_position": torch.tensor(((0.0, 0.0), (0.1, 0.2), (0.2, 6.0), (-8.0, 0.0), (-7.9, 0.1))),
        "joint_velocity": torch.tensor(((1.0, 2.0), (101.0, 0.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0))),
    }
    table = SimpleNamespace(
        clip_index=SimpleNamespace(total_frames=5),
        field=lambda name: fields[name],
    )

    branch = benchmark._branch_profile(table, torch.tensor((0, 3, 5), dtype=torch.int64))
    magnitude = benchmark._tensor_profile(torch.tensor(((0.5, -1.0), (4.0, 0.0))), (0.75, 3.0))

    assert branch["valid_edges"] == 3
    assert branch["branch_jump_edges_over_pi"] == 1
    assert branch["branch_edges_with_joint_velocity_over_100_rad_per_s"] == 1
    assert branch["maximum_coordinate_jump_rad"] == pytest.approx(5.8)
    assert magnitude["row_counts_over"] == {"0.75": 2, "3.0": 1}


def test_cli_requires_separate_artifact_roots_and_accepts_train_split(benchmark, monkeypatch, tmp_path: Path) -> None:
    """The tracked producer must expose independent raw and oracle roots."""
    raw_root = tmp_path / "raw"
    oracle_root = tmp_path / "oracle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--raw_artifact_root",
            str(raw_root),
            "--oracle_artifact_root",
            str(oracle_root),
            "--motion_split",
            "train",
        ],
    )

    args = benchmark._parse_args()

    assert args.raw_artifact_root == raw_root
    assert args.oracle_artifact_root == oracle_root
    assert args.motion_split == "train"
    assert args.device == "cuda:0"
    assert args.output == Path("/tmp/amass_smpl_full_report.json")
