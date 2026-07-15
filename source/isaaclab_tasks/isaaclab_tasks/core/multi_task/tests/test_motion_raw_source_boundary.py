# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for immutable artifacts shared by native motion-source splits."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_tasks.core.multi_task.motion.data import MotionSourceCfg
from isaaclab_tasks.core.multi_task.motion.data.sources.amass_smplh import (
    amass_clip_rows_content_sha256,
    read_amass_clip_rows,
)
from isaaclab_tasks.core.multi_task.motion.identity import canonical_sha256
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
    build_motion_task_table,
    build_motion_task_table_inspection,
)
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionSourcesCfg

_DIGEST = hashlib.sha256(b"dependency").hexdigest()
_OTHER_DIGEST = hashlib.sha256(b"other dependency").hexdigest()
_RAW_SOURCE_ROOT = Path(__file__).parents[1] / "motion" / "data" / "sources"
_PREPARE_AMASS = Path(__file__).parents[1] / "scripts" / "prepare_amass.py"


def _split(name: str) -> MotionSourceCfg.SplitCfg:
    return MotionSourceCfg.SplitCfg(
        name=name,
        artifact=f"{name}.txt",
        artifact_sha256=_DIGEST,
        source_content_sha256=_OTHER_DIGEST,
        clip_count=1,
        frame_count=1,
    )


def _source(*dependencies: MotionSourceCfg.DependencyCfg, purpose: str = "production") -> MotionSourceCfg:
    return MotionSourceCfg(
        identifier="native_motion",
        open_source=lambda *_args: None,
        format="native",
        semantic_level="body_motion",
        decoder_version="native_v1",
        source_fps=30.0,
        license="test",
        clip_directory=None,
        dependencies=dependencies,
        train=_split("train"),
        evaluation=_split("evaluation"),
        purpose=purpose,
    )


def test_motion_source_purpose_is_explicit_and_validated() -> None:
    """Native sources default to production while unknown purposes fail at configuration time."""
    assert _source().purpose == "production"
    assert _source(purpose="oracle").purpose == "oracle"

    with pytest.raises(ValueError, match="purpose must be"):
        _source(purpose="comparison")


def test_released_sources_are_oracles_and_raw_cmu_is_the_production_default() -> None:
    """Released retargets cannot become the implicit runtime corpus."""
    sources = MotionSourcesCfg()

    assert sources.default.identifier == sources.cmu.identifier == "amass_cmu_smplh"
    assert sources.cmu.purpose == sources.lafan.purpose == "production"
    assert sources.humenv_cmu.purpose == sources.bfm_lafan.purpose == "oracle"


def test_oracle_source_fails_before_opening_in_production_but_inspection_can_open() -> None:
    """Runtime rejects released evidence deterministically while inspection reaches its decoder."""
    opened: list[tuple[str, object]] = []
    split = object()

    def open_split(root: str, selected_split: object) -> None:
        opened.append((root, selected_split))
        raise RuntimeError("inspection opened oracle")

    source = SimpleNamespace(
        identifier="released_retarget",
        purpose="oracle",
        train=split,
        evaluation=split,
        open_split=open_split,
    )
    command = SimpleNamespace(
        task_table=SimpleNamespace(
            source=source,
            source_artifact_root="/supplied/oracle",
            motion_split="train",
        )
    )

    with pytest.raises(ValueError, match="oracle-only"):
        build_motion_task_table(command, object(), "cpu")
    assert opened == []

    with pytest.raises(RuntimeError, match="inspection opened oracle"):
        build_motion_task_table_inspection(command, object(), "cpu", sequence_limit=1)
    assert opened == [("/supplied/oracle", split)]


def test_motion_source_rejects_duplicate_dependency_names() -> None:
    """A decoder cannot bind two different artifacts through one dependency name."""
    dependency = MotionSourceCfg.DependencyCfg(name="body_model", artifact="model.npz", artifact_sha256=_DIGEST)

    with pytest.raises(ValueError, match="must be unique"):
        _source(dependency, dependency)


def test_motion_source_resolves_verified_dependencies(tmp_path) -> None:
    """Resolution returns source-root paths only after every artifact hash matches."""
    artifact = tmp_path / "model.npz"
    artifact.write_bytes(b"dependency")
    source = _source(MotionSourceCfg.DependencyCfg(name="body_model", artifact=artifact.name, artifact_sha256=_DIGEST))

    assert source.resolve_dependencies(tmp_path) == {"body_model": artifact}


def test_motion_source_rejects_missing_dependency(tmp_path) -> None:
    """A missing declared model fails at the source boundary before native decoding."""
    source = _source(MotionSourceCfg.DependencyCfg(name="body_model", artifact="missing.npz", artifact_sha256=_DIGEST))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        source.resolve_dependencies(tmp_path)


def test_motion_source_rejects_dependency_hash_mismatch(tmp_path) -> None:
    """A changed model cannot silently alter conversion under the same source config."""
    artifact = tmp_path / "model.npz"
    artifact.write_bytes(b"other dependency")
    source = _source(MotionSourceCfg.DependencyCfg(name="body_model", artifact=artifact.name, artifact_sha256=_DIGEST))

    with pytest.raises(ValueError, match="hash differs"):
        source.resolve_dependencies(tmp_path)


@pytest.mark.parametrize(
    ("filename", "forbidden"),
    (("amass_smplh.py", ("humenv", "g1")), ("lafan_bvh.py", ("smpl", "g1"))),
)
def test_raw_motion_decoder_does_not_name_a_target_robot(filename: str, forbidden: tuple[str, ...]) -> None:
    """Raw decoders stop at source mechanics and cannot contain a source-to-robot product route."""
    source = (_RAW_SOURCE_ROOT / filename).read_text(encoding="utf-8").casefold()

    for name in forbidden:
        assert name not in source


def test_prepare_amass_split_preserves_selected_order_and_digest(tmp_path) -> None:
    """The one-time split tool freezes concrete metadata in prepared selection order."""
    raw_root = tmp_path / "CMU"
    first = raw_root / "01" / "walk_poses.npz"
    second = raw_root / "02" / "run_poses.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)

    def write_clip(path: Path, frame_count: int) -> None:
        translation = np.zeros((frame_count, 3), dtype=np.float64)
        pose = np.zeros((frame_count, 156), dtype=np.float64)
        np.savez(
            path,
            trans=translation,
            gender=np.array("male"),
            mocap_framerate=np.array(120.0, dtype=np.float64),
            betas=np.linspace(-0.1, 0.1, 16, dtype=np.float64),
            dmpls=np.zeros((frame_count, 8), dtype=np.float64),
            poses=pose,
        )

    write_clip(first, 5)
    write_clip(second, 7)
    selection = tmp_path / "prepared.txt"
    selection.write_text("0-CMU_02_run_poses.hdf5\n0-CMU_01_walk_poses.hdf5\n", encoding="utf-8")
    output = tmp_path / "raw.csv"
    prepare_split = runpy.run_path(_PREPARE_AMASS)["prepare_split"]

    count, digest = prepare_split(selection, raw_root, output)
    rows = read_amass_clip_rows(output)

    assert count == 2
    assert tuple(row.relative_path for row in rows) == ("02/run_poses.npz", "01/walk_poses.npz")
    assert tuple(row.source_frame_count for row in rows) == (7, 5)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert amass_clip_rows_content_sha256(rows) == canonical_sha256(
        tuple((row.relative_path, row.source_sha256) for row in rows)
    )
