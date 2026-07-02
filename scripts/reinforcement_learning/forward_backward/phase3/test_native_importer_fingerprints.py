# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate Phase 3C importer fingerprints with or without licensed source data."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
FIXTURE = ROOT / "fixtures" / "native_importer_fingerprints_v1.json"
GENERATOR = ROOT / "generate_native_importer_fingerprints.py"
HUMENV_SOURCE = Path("/tmp/phase3_humenv_clip.hdf5")
FINGERPRINT_SHA256 = "a7284316431a2719be130c063b0554713770afea572f243d00cf84846129db05"
BFM_SOURCE = ROOT.parents[5] / "BFM-Zero" / "humanoidverse" / "data" / "lafan_29dof_10s-clipped.pkl"


def _generator_module():
    spec = importlib.util.spec_from_file_location("generate_native_importer_fingerprints", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _assert_field_fingerprint(field: dict[str, object]) -> None:
    assert isinstance(field["shape"], list)
    if field["shape"]:
        assert isinstance(field["dtype"], str)
        frames = field["frame_bytes"]
        assert [frame["position"] for frame in frames] == ["first", "middle", "last"]
        assert [frame["index"] for frame in frames] == [0, field["shape"][0] // 2, field["shape"][0] - 1]
        assert all(len(frame["sha256"]) == 64 for frame in frames)
    else:
        assert field["scalar_type"] in {"python:int", "python:str"}
        assert isinstance(field["scalar_encoding"], str)
        assert len(field["scalar_bytes_sha256"]) == 64


def _assert_clip_fingerprint(clip: dict[str, object]) -> None:
    assert list(clip["fields"]) == clip["field_order"]
    for field in clip["fields"].values():
        _assert_field_fingerprint(field)


def test_frozen_fingerprints_preserve_native_order_and_exact_byte_contracts() -> None:
    """The portable fixture must retain identities and native ordering without raw motion data."""
    assert _sha256(FIXTURE) == FINGERPRINT_SHA256
    frozen = _load_fixture()
    assert frozen["format"] == "forward_backward_phase3c_native_importer_fingerprints_v1"

    humenv = frozen["humenv_hdf5"]
    assert humenv["source_file"] == {
        "name": "0-CMU_91_91_19_poses.hdf5",
        "sha256": "7b1580a40b2a9134dd25977b256747fbe08cdb9ef50ac501f0d911ba385d17e1",
    }
    assert humenv["ordered_clip_ids"] == ["0-CMU_91_91_19_poses"]
    assert humenv["ordered_clip_ids_sha256"] == "56739a94d4dfde3a7d085552ed26d3d184396b40a9542136c24eecdd758e80a0"
    assert [clip["clip_id"] for clip in humenv["clips"]] == humenv["ordered_clip_ids"]
    assert humenv["clips"][0]["field_order"] == [
        "motion_id",
        "observation",
        "qpos",
        "qvel",
        "terminated",
        "truncated",
    ]
    _assert_clip_fingerprint(humenv["clips"][0])

    bfm = frozen["bfm_g1_joblib"]
    assert bfm["source_file"] == {
        "name": "lafan_29dof_10s-clipped.pkl",
        "sha256": "7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
    }
    assert len(bfm["ordered_clip_ids"]) == 862
    assert len(set(bfm["ordered_clip_ids"])) == 862
    assert bfm["ordered_clip_ids_sha256"] == "7e34cb01112bf9807006c75f287bafba33eb86dc66f188a4e80ad03500eaf895"
    positions = [0, len(bfm["ordered_clip_ids"]) // 2, len(bfm["ordered_clip_ids"]) - 1]
    assert [clip["position"] for clip in bfm["sampled_clips"]] == positions
    assert [clip["clip_id"] for clip in bfm["sampled_clips"]] == [bfm["ordered_clip_ids"][index] for index in positions]
    for clip in bfm["sampled_clips"]:
        assert clip["field_order"] == [
            "root_trans_offset",
            "pose_aa",
            "dof",
            "root_rot",
            "smpl_joints",
            "fps",
            "motion_name",
        ]
        assert clip["fields"]["fps"]["scalar_type"] == "python:int"
        assert clip["fields"]["motion_name"]["scalar_type"] == "python:str"
        _assert_clip_fingerprint(clip)

    assert not any(value.startswith("/") for value in _strings(frozen))


def test_humenv_fingerprint_regenerates_from_exact_source_when_available() -> None:
    """Local exact HumEnv bytes must regenerate the frozen HDF5 contract."""
    if not HUMENV_SOURCE.is_file():
        pytest.skip("Exact local HumEnv HDF5 source is unavailable.")
    module = _generator_module()
    assert module.capture_humenv_hdf5(HUMENV_SOURCE) == _load_fixture()["humenv_hdf5"]


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")
def test_bfm_fingerprint_regenerates_from_exact_source_when_available() -> None:
    """Local exact BFM bytes must regenerate insertion order and sampled clip contracts."""
    if not BFM_SOURCE.is_file():
        pytest.skip("Exact local BFM G1 joblib source is unavailable.")
    module = _generator_module()
    assert module.capture_bfm_g1_joblib(BFM_SOURCE) == _load_fixture()["bfm_g1_joblib"]
