# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict and tamper-evident Newton debug archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from isaaclab_newton.physics import _debug_archive as archive

_CREATED_AT = "2026-06-27T12:34:56+00:00"


def _write_archive(path: Path, arrays: dict[str, np.ndarray], **kwargs):
    return archive.write_archive(
        path,
        arrays,
        dependency_names=(),
        created_at=_CREATED_AT,
        **kwargs,
    )


def _read_members(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.array(source[key], copy=True) for key in source.files}


def _rewrite_members(path: Path, members: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **members)


def test_archive_round_trip_preserves_values_dtype_shape_and_manifest(tmp_path):
    """A complete archive round-trips exact arrays and its validated manifest."""
    path = tmp_path / "nested" / "incident.npz"
    arrays = {
        "state": np.arange(12, dtype=np.float32).reshape(3, 4),
        "world_ids": np.array([2, 9], dtype=np.int32),
    }

    written = _write_archive(
        path,
        arrays,
        required_keys={"world_ids", "state"},
        metadata={"trigger": "nonfinite"},
    )
    loaded, manifest = archive.load_archive(path, required_keys={"state"})

    assert path.is_file()
    assert manifest == written
    assert manifest["status"] == "complete"
    assert manifest["required_keys"] == ["state", "world_ids"]
    assert manifest["metadata"] == {"trigger": "nonfinite"}
    for key, expected in arrays.items():
        assert loaded[key].dtype == expected.dtype
        assert loaded[key].shape == expected.shape
        np.testing.assert_array_equal(loaded[key], expected)
        assert loaded[key].flags.c_contiguous


def test_archive_round_trip_preserves_empty_multidimensional_arrays(tmp_path):
    """Checksum inventory supports arrays with zero-length dimensions."""
    path = tmp_path / "empty.npz"
    expected = np.empty((1, 0, 7), dtype=np.float32)

    _write_archive(path, {"empty": expected})
    loaded, manifest = archive.load_archive(path)

    assert loaded["empty"].shape == expected.shape
    assert loaded["empty"].dtype == expected.dtype
    assert manifest["arrays"]["empty"]["sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_failed_atomic_replacement_preserves_previous_archive(monkeypatch, tmp_path):
    """A failed replacement leaves the prior valid archive intact and removes its temporary file."""
    path = tmp_path / "incident.npz"
    _write_archive(path, {"value": np.array([1], dtype=np.int32)})

    def fail_replace(source, destination):
        raise OSError("replace refused")

    monkeypatch.setattr(archive.os, "replace", fail_replace)
    with pytest.raises(archive.DebugArchiveWriteError, match="atomically install.*replace refused"):
        _write_archive(path, {"value": np.array([2], dtype=np.int32)})

    loaded, _ = archive.load_archive(path)
    np.testing.assert_array_equal(loaded["value"], np.array([1], dtype=np.int32))
    assert list(tmp_path.glob(".incident.npz.*.tmp")) == []


def test_partial_archive_is_rejected_by_default_but_can_be_explicitly_accepted(tmp_path):
    """Consumers must opt in before using a known-incomplete incident archive."""
    path = tmp_path / "partial.npz"
    _write_archive(path, {"state": np.arange(2)}, status="partial")

    with pytest.raises(archive.DebugArchiveValidationError, match="status 'partial' is not accepted"):
        archive.load_archive(path)
    with pytest.raises(archive.DebugArchiveValidationError, match="status 'partial' is not accepted"):
        archive.validate_archive(path)

    loaded, manifest = archive.load_archive(path, allowed_statuses={"partial"})
    assert manifest["status"] == "partial"
    assert archive.validate_archive(path, allowed_statuses={"partial"}) == manifest
    np.testing.assert_array_equal(loaded["state"], np.arange(2))


def test_archive_rejects_dtype_tampering(tmp_path):
    """Changing a member dtype without updating its manifest is detected."""
    path = tmp_path / "dtype.npz"
    _write_archive(path, {"state": np.arange(4, dtype=np.float32)})
    members = _read_members(path)
    members["state"] = members["state"].astype(np.float64)
    _rewrite_members(path, members)

    with pytest.raises(archive.DebugArchiveValidationError, match=r"arrays\.state\.dtype"):
        archive.load_archive(path)


def test_archive_rejects_shape_tampering(tmp_path):
    """Changing only member dimensions is detected even when bytes remain identical."""
    path = tmp_path / "shape.npz"
    _write_archive(path, {"state": np.arange(6, dtype=np.int32).reshape(2, 3)})
    members = _read_members(path)
    members["state"] = members["state"].reshape(3, 2)
    _rewrite_members(path, members)

    with pytest.raises(archive.DebugArchiveValidationError, match=r"arrays\.state\.shape"):
        archive.load_archive(path)


def test_archive_rejects_checksum_tampering(tmp_path):
    """Changing member values with the same schema is detected by SHA-256."""
    path = tmp_path / "checksum.npz"
    _write_archive(path, {"state": np.arange(4, dtype=np.float32)})
    members = _read_members(path)
    members["state"][0] = 123.0
    _rewrite_members(path, members)

    with pytest.raises(archive.DebugArchiveValidationError, match=r"arrays\.state\.sha256.*checksum mismatch"):
        archive.load_archive(path)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_archive_rejects_member_inventory_tampering(tmp_path, change):
    """The manifest and physical NPZ member set must agree exactly."""
    path = tmp_path / f"members_{change}.npz"
    _write_archive(path, {"state": np.arange(4, dtype=np.float32)})
    members = _read_members(path)
    if change == "missing":
        del members["state"]
        expected = "arrays declared.*are missing"
    else:
        members["unexpected"] = np.ones(1, dtype=np.float32)
        expected = "arrays are present but absent"
    _rewrite_members(path, members)

    with pytest.raises(archive.DebugArchiveValidationError, match=expected):
        archive.load_archive(path)


def test_archive_writer_rejects_object_arrays(tmp_path):
    """Object arrays are rejected before writing because they require pickle."""
    path = tmp_path / "object.npz"

    with pytest.raises(archive.DebugArchiveValidationError, match="object dtype.*requires pickle"):
        _write_archive(path, {"unsafe": np.array([object()], dtype=object)})
    assert not path.exists()


def test_archive_loader_always_disables_pickle(monkeypatch, tmp_path):
    """Loading a valid archive always passes ``allow_pickle=False`` to NumPy."""
    path = tmp_path / "safe.npz"
    _write_archive(path, {"safe": np.arange(2, dtype=np.int32)})
    load_calls: list[object] = []
    numpy_load = np.load

    def recording_load(*args, **kwargs):
        load_calls.append(kwargs.get("allow_pickle"))
        return numpy_load(*args, **kwargs)

    monkeypatch.setattr(archive.np, "load", recording_load)
    archive.load_archive(path)

    assert load_calls == [False]


def test_archive_loader_rejects_a_manually_injected_pickle_member(tmp_path):
    """A hostile object member cannot be loaded through the strict archive reader."""
    path = tmp_path / "hostile.npz"
    safe = np.arange(2, dtype=np.int32)
    manifest = archive.create_manifest(
        {"safe": safe},
        dependency_names=(),
        created_at=_CREATED_AT,
    )
    members = {
        archive.MANIFEST_KEY: np.asarray(json.dumps(manifest, sort_keys=True, separators=(",", ":"))),
        "safe": safe,
        "hostile": np.array([object()], dtype=object),
    }
    _rewrite_members(path, members)

    with pytest.raises(archive.DebugArchiveValidationError, match="allow_pickle=False"):
        archive.load_archive(path)
