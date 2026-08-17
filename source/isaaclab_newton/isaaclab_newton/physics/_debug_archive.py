# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict, versioned NPZ archives for Newton debug incidents."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import sys
import tempfile
from collections.abc import Collection, Iterable, Mapping
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Final

import numpy as np


FORMAT_NAME: Final[str] = "isaaclab_newton.debug_incident"
"""Stable name of the debug incident archive format."""

FORMAT_VERSION: Final[int] = 1
"""Current debug incident archive format version."""

MANIFEST_KEY: Final[str] = "__manifest__"
"""Reserved NPZ member containing the JSON archive manifest."""

ARCHIVE_STATUSES: Final[frozenset[str]] = frozenset({"complete", "partial"})
"""Valid archive completion statuses."""

DEFAULT_ALLOWED_STATUSES: Final[frozenset[str]] = frozenset({"complete"})
"""Archive statuses accepted by strict readers unless explicitly overridden."""

DEFAULT_DEPENDENCIES: Final[tuple[str, ...]] = (
    "isaaclab",
    "isaaclab_newton",
    "newton",
    "mujoco-warp",
    "warp-lang",
    "torch",
    "numpy",
)
"""Distribution names queried for archive provenance by default."""

_RESERVED_ARRAY_KEYS: Final[frozenset[str]] = frozenset({MANIFEST_KEY, "file"})
_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "format_name",
        "format_version",
        "status",
        "created_at",
        "required_keys",
        "arrays",
        "dependencies",
        "runtime",
        "metadata",
    }
)
_INVENTORY_FIELDS: Final[frozenset[str]] = frozenset(
    {"dtype", "dtype_descr", "shape", "nbytes", "sha256"}
)
_PROVENANCE_STATUSES: Final[frozenset[str]] = frozenset({"available", "unavailable", "error"})


class DebugArchiveError(RuntimeError):
    """Base exception for debug archive creation and validation failures."""


class DebugArchiveWriteError(DebugArchiveError):
    """Raised when a debug archive cannot be written atomically."""


class DebugArchiveValidationError(DebugArchiveError):
    """Raised when an archive or manifest violates the format contract."""


def collect_dependency_provenance(
    distribution_names: Iterable[str] = DEFAULT_DEPENDENCIES,
) -> dict[str, dict[str, Any]]:
    """Collect installed-distribution provenance without importing dependencies.

    Missing distributions and absent or malformed ``direct_url.json`` metadata are
    represented explicitly. They are not errors because provenance is diagnostic
    context rather than incident payload.

    Args:
        distribution_names: Distribution names understood by
            :mod:`importlib.metadata`.

    Returns:
        A mapping from each requested distribution name to its availability,
        version, and PEP 610 direct-URL metadata status.

    Raises:
        DebugArchiveValidationError: If a distribution name is invalid or appears
            more than once.
    """
    names = _validate_distribution_names(distribution_names)
    provenance: dict[str, dict[str, Any]] = {}
    for name in names:
        try:
            distribution = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            provenance[name] = {
                "status": "unavailable",
                "reason": f"distribution {name!r} is not installed",
                "version": None,
                "direct_url": {
                    "status": "unavailable",
                    "reason": "distribution metadata is unavailable",
                },
            }
            continue

        direct_url = _read_direct_url(distribution, name)
        provenance[name] = {
            "status": "available",
            "distribution_name": distribution.metadata.get("Name", name),
            "version": distribution.version,
            "direct_url": direct_url,
        }
    return provenance


def create_array_inventory(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Create a dtype, shape, size, and checksum inventory for arrays.

    Values are converted to independent, C-contiguous NumPy arrays before their
    inventory is computed. Object arrays, including structured dtypes containing
    object fields, are rejected because they require pickle when loaded.

    Args:
        arrays: Array names and array-like values to inventory.

    Returns:
        A JSON-compatible inventory sorted by array name.

    Raises:
        DebugArchiveValidationError: If a key is unsafe or a value cannot be
            represented as a non-object NumPy array.
    """
    normalized = _normalize_arrays(arrays, "arrays")
    return _create_array_inventory(normalized)


def create_manifest(
    arrays: Mapping[str, Any],
    *,
    status: str = "complete",
    required_keys: Collection[str] = (),
    metadata: Mapping[str, Any] | None = None,
    dependency_names: Iterable[str] = DEFAULT_DEPENDENCIES,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create and validate a manifest for a set of incident arrays.

    Args:
        arrays: Array names and array-like values represented by the manifest.
        status: Capture status, either ``"complete"`` or ``"partial"``.
        required_keys: Array keys that consumers must always find in the archive.
        metadata: Additional JSON-compatible incident metadata.
        dependency_names: Distribution names to include in provenance.
        created_at: Optional timezone-aware ISO 8601 creation timestamp. The
            current UTC time is used when omitted.

    Returns:
        A validated, JSON-compatible manifest.

    Raises:
        DebugArchiveValidationError: If arrays or manifest values violate the
            archive contract.
    """
    normalized = _normalize_arrays(arrays, "arrays")
    return _create_manifest(
        normalized,
        status=status,
        required_keys=required_keys,
        metadata=metadata,
        dependency_names=dependency_names,
        created_at=created_at,
    )


def write_archive(
    path: str | os.PathLike[str],
    arrays: Mapping[str, Any],
    *,
    status: str = "complete",
    required_keys: Collection[str] = (),
    metadata: Mapping[str, Any] | None = None,
    dependency_names: Iterable[str] = DEFAULT_DEPENDENCIES,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write an incident archive atomically as compressed NPZ.

    The archive is first written and synchronized in the destination directory,
    then installed with :func:`os.replace`. Existing archives are replaced only
    after the new archive is complete.

    Args:
        path: Destination path ending in ``.npz``.
        arrays: Array names and array-like values to save.
        status: Capture status, either ``"complete"`` or ``"partial"``.
        required_keys: Array keys that consumers must always find in the archive.
        metadata: Additional JSON-compatible incident metadata.
        dependency_names: Distribution names to include in provenance.
        created_at: Optional timezone-aware ISO 8601 creation timestamp.

    Returns:
        The manifest embedded in the archive.

    Raises:
        DebugArchiveValidationError: If the path, arrays, or manifest are invalid.
        DebugArchiveWriteError: If the temporary archive cannot be written,
            synchronized, or installed at the destination.
    """
    archive_path = _validate_archive_path(path)
    normalized = _normalize_arrays(arrays, f"{archive_path}::arrays")
    manifest = _create_manifest(
        normalized,
        status=status,
        required_keys=required_keys,
        metadata=metadata,
        dependency_names=dependency_names,
        created_at=created_at,
    )
    manifest_json = _encode_manifest(manifest, f"{archive_path}::{MANIFEST_KEY}")
    payload = dict(normalized)
    payload[MANIFEST_KEY] = np.asarray(manifest_json)

    temporary_path: Path | None = None
    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{archive_path.name}.",
            suffix=".tmp",
            dir=archive_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, archive_path)
    except Exception as exc:
        cleanup_error: OSError | None = None
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as unlink_exc:
                cleanup_error = unlink_exc
        error = DebugArchiveWriteError(
            f"{archive_path}: failed to write and atomically install the debug archive: {exc}"
        )
        if cleanup_error is not None:
            error.add_note(f"Temporary archive cleanup also failed at {temporary_path}: {cleanup_error}")
        raise error from exc
    return manifest


def load_archive(
    path: str | os.PathLike[str],
    *,
    required_keys: Collection[str] = (),
    allowed_statuses: Collection[str] = DEFAULT_ALLOWED_STATUSES,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and fully validate a debug incident archive without pickle.

    Args:
        path: Existing ``.npz`` debug archive.
        required_keys: Additional array keys required by the caller.
        allowed_statuses: Capture statuses accepted by the caller. Defaults to
            complete archives only; partial archives require explicit opt-in.

    Returns:
        A tuple containing independent NumPy arrays and the decoded manifest.

    Raises:
        DebugArchiveValidationError: If the file is missing, malformed,
            incompatible, incomplete, or fails an integrity check.
    """
    archive_path = _validate_archive_path(path)
    if not archive_path.is_file():
        raise DebugArchiveValidationError(f"{archive_path}: debug archive file does not exist")

    caller_required = _validate_required_keys(required_keys, f"{archive_path}::required_keys")
    accepted_statuses = _validate_allowed_statuses(allowed_statuses, archive_path)
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            member_names = list(archive.files)
            if len(member_names) != len(set(member_names)):
                raise DebugArchiveValidationError(
                    f"{archive_path}: NPZ contains duplicate array member names"
                )
            if MANIFEST_KEY not in member_names:
                raise DebugArchiveValidationError(
                    f"{archive_path}::{MANIFEST_KEY}: required JSON manifest is missing"
                )

            manifest = _decode_manifest(archive[MANIFEST_KEY], archive_path)
            arrays: dict[str, np.ndarray] = {}
            for key in member_names:
                if key == MANIFEST_KEY:
                    continue
                value = np.asarray(archive[key])
                if value.dtype.hasobject:
                    raise DebugArchiveValidationError(
                        f"{archive_path}::{key}: object dtype {value.dtype!s} is forbidden; "
                        "archives must load with allow_pickle=False"
                    )
                arrays[key] = _normalize_array(value)
    except DebugArchiveValidationError:
        raise
    except Exception as exc:
        raise DebugArchiveValidationError(
            f"{archive_path}: failed to read a strict NPZ debug archive with allow_pickle=False: {exc}"
        ) from exc

    _validate_loaded_archive(
        archive_path,
        arrays,
        manifest,
        caller_required=caller_required,
        allowed_statuses=accepted_statuses,
    )
    return arrays, manifest


def validate_archive(
    path: str | os.PathLike[str],
    *,
    required_keys: Collection[str] = (),
    allowed_statuses: Collection[str] = DEFAULT_ALLOWED_STATUSES,
) -> dict[str, Any]:
    """Validate an archive and return its manifest.

    Args:
        path: Existing ``.npz`` debug archive.
        required_keys: Additional array keys required by the caller.
        allowed_statuses: Capture statuses accepted by the caller.

    Returns:
        The validated archive manifest.

    Raises:
        DebugArchiveValidationError: If any archive validation fails.
    """
    _, manifest = load_archive(path, required_keys=required_keys, allowed_statuses=allowed_statuses)
    return manifest


def _create_manifest(
    arrays: Mapping[str, np.ndarray],
    *,
    status: str,
    required_keys: Collection[str],
    metadata: Mapping[str, Any] | None,
    dependency_names: Iterable[str],
    created_at: str | None,
) -> dict[str, Any]:
    _validate_status(status, "manifest.status")
    required = _validate_required_keys(required_keys, "manifest.required_keys")
    missing = sorted(set(required).difference(arrays))
    if missing:
        raise DebugArchiveValidationError(
            f"manifest.required_keys: required arrays are absent: {missing}"
        )

    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    _validate_timestamp(timestamp, "manifest.created_at")
    if not isinstance(metadata, Mapping) and metadata is not None:
        raise DebugArchiveValidationError(
            f"manifest.metadata: expected a mapping, got {type(metadata).__name__}"
        )
    manifest_metadata = {} if metadata is None else dict(metadata)

    manifest: dict[str, Any] = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "status": status,
        "created_at": timestamp,
        "required_keys": required,
        "arrays": _create_array_inventory(arrays),
        "dependencies": collect_dependency_provenance(dependency_names),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "byte_order": sys.byteorder,
        },
        "metadata": manifest_metadata,
    }
    _encode_manifest(manifest, MANIFEST_KEY)
    _validate_manifest_structure(manifest, MANIFEST_KEY)
    return manifest


def _normalize_arrays(arrays: Mapping[str, Any], context: str) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise DebugArchiveValidationError(
            f"{context}: expected a mapping of array names to values, got {type(arrays).__name__}"
        )
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        _validate_array_key(key, context)
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise DebugArchiveValidationError(
                f"{context}.{key}: value cannot be converted to a NumPy array: {exc}"
            ) from exc
        if array.dtype.hasobject:
            raise DebugArchiveValidationError(
                f"{context}.{key}: object dtype {array.dtype!s} is forbidden because it requires pickle"
            )
        normalized[key] = _normalize_array(array)
    return normalized


def _normalize_array(array: np.ndarray) -> np.ndarray:
    normalized = np.array(array, copy=True, order="C", subok=False)
    if normalized.dtype.names is None:
        return normalized

    canonical = np.zeros(normalized.shape, dtype=normalized.dtype, order="C")
    _copy_structured_fields(canonical, normalized)
    return canonical


def _copy_structured_fields(target: np.ndarray, source: np.ndarray) -> None:
    assert target.dtype.names is not None
    for name in target.dtype.names:
        target_field = target[name]
        source_field = source[name]
        if target_field.dtype.names is None:
            target_field[...] = source_field
        else:
            _copy_structured_fields(target_field, source_field)


def _validate_array_key(key: object, context: str) -> None:
    if not isinstance(key, str) or not key:
        raise DebugArchiveValidationError(
            f"{context}: every array key must be a non-empty string, got {key!r}"
        )
    if key in _RESERVED_ARRAY_KEYS:
        raise DebugArchiveValidationError(
            f"{context}.{key}: array key is reserved by the NPZ archive format"
        )
    if "\x00" in key or "/" in key or "\\" in key:
        raise DebugArchiveValidationError(
            f"{context}.{key}: array keys cannot contain NUL or path separator characters"
        )


def _create_array_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for key in sorted(arrays):
        array = arrays[key]
        inventory[key] = {
            "dtype": array.dtype.str,
            "dtype_descr": _dtype_descriptor(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "sha256": _array_sha256(array),
        }
    return inventory


def _array_sha256(array: np.ndarray) -> str:
    contiguous = _normalize_array(array) if array.dtype.names is not None else np.ascontiguousarray(array)
    # Flatten before exposing the byte view because ``memoryview.cast`` rejects
    # multidimensional arrays containing a zero-length dimension.
    byte_view = memoryview(contiguous.reshape(-1).view(np.uint8))
    return hashlib.sha256(byte_view).hexdigest()


def _dtype_descriptor(dtype: np.dtype) -> Any:
    descriptor = np.lib.format.dtype_to_descr(dtype)

    def normalize(value: Any) -> Any:
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(descriptor)


def _read_direct_url(
    distribution: importlib_metadata.Distribution,
    distribution_name: str,
) -> dict[str, Any]:
    try:
        direct_url_text = distribution.read_text("direct_url.json")
    except (OSError, UnicodeError) as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": f"could not read direct_url.json for {distribution_name!r}: {exc}",
        }
    if direct_url_text is None:
        return {
            "status": "unavailable",
            "reason": f"distribution {distribution_name!r} does not provide direct_url.json",
        }
    try:
        direct_url = _strict_json_loads(direct_url_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": f"direct_url.json for {distribution_name!r} is invalid: {exc}",
        }
    if not isinstance(direct_url, dict):
        return {
            "status": "error",
            "error_type": "TypeError",
            "message": f"direct_url.json for {distribution_name!r} must contain a JSON object",
        }
    return {"status": "available", "value": direct_url}


def _validate_distribution_names(distribution_names: Iterable[str]) -> list[str]:
    try:
        names = list(distribution_names)
    except TypeError as exc:
        raise DebugArchiveValidationError(
            "dependency_names: expected an iterable of distribution names"
        ) from exc
    seen: set[str] = set()
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise DebugArchiveValidationError(
                f"dependency_names[{index}]: expected a non-empty string, got {name!r}"
            )
        normalized = name.casefold().replace("_", "-")
        if normalized in seen:
            raise DebugArchiveValidationError(
                f"dependency_names[{index}]: duplicate distribution name {name!r}"
            )
        seen.add(normalized)
    return names


def _validate_archive_path(path: str | os.PathLike[str]) -> Path:
    try:
        archive_path = Path(path)
    except TypeError as exc:
        raise DebugArchiveValidationError(
            f"archive path: expected a string or path-like value, got {path!r}"
        ) from exc
    if archive_path.suffix.lower() != ".npz":
        raise DebugArchiveValidationError(
            f"{archive_path}: debug archive path must end in '.npz'"
        )
    return archive_path


def _validate_required_keys(required_keys: Collection[str], context: str) -> list[str]:
    if isinstance(required_keys, str):
        raise DebugArchiveValidationError(
            f"{context}: expected a collection of keys, not a single string"
        )
    try:
        keys = list(required_keys)
    except TypeError as exc:
        raise DebugArchiveValidationError(f"{context}: expected a collection of array keys") from exc
    seen: set[str] = set()
    for index, key in enumerate(keys):
        if not isinstance(key, str) or not key:
            raise DebugArchiveValidationError(
                f"{context}[{index}]: expected a non-empty string, got {key!r}"
            )
        if key in seen:
            raise DebugArchiveValidationError(f"{context}[{index}]: duplicate required key {key!r}")
        if key == MANIFEST_KEY:
            raise DebugArchiveValidationError(
                f"{context}[{index}]: {MANIFEST_KEY!r} is implicit and cannot be an array requirement"
            )
        seen.add(key)
    return sorted(keys)


def _validate_allowed_statuses(statuses: Collection[str], archive_path: Path) -> frozenset[str]:
    if isinstance(statuses, str):
        raise DebugArchiveValidationError(
            f"{archive_path}::allowed_statuses: expected a collection, not a single string"
        )
    try:
        status_values = list(statuses)
    except TypeError as exc:
        raise DebugArchiveValidationError(
            f"{archive_path}::allowed_statuses: expected a collection of strings"
        ) from exc
    for index, status in enumerate(status_values):
        if not isinstance(status, str):
            raise DebugArchiveValidationError(
                f"{archive_path}::allowed_statuses[{index}]: expected a string, found {status!r}"
            )
    accepted = frozenset(status_values)
    if not accepted:
        raise DebugArchiveValidationError(
            f"{archive_path}::allowed_statuses: at least one status must be accepted"
        )
    invalid = sorted(status for status in accepted if status not in ARCHIVE_STATUSES)
    if invalid:
        raise DebugArchiveValidationError(
            f"{archive_path}::allowed_statuses: unknown statuses {invalid}; valid statuses are "
            f"{sorted(ARCHIVE_STATUSES)}"
        )
    return accepted


def _validate_loaded_archive(
    archive_path: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    caller_required: Collection[str],
    allowed_statuses: Collection[str],
) -> None:
    context = f"{archive_path}::{MANIFEST_KEY}"
    _validate_manifest_structure(manifest, context)

    status = manifest["status"]
    if status not in allowed_statuses:
        raise DebugArchiveValidationError(
            f"{context}.status: archive status {status!r} is not accepted; allowed statuses are "
            f"{sorted(allowed_statuses)}"
        )

    inventory = manifest["arrays"]
    archived_keys = set(arrays)
    inventoried_keys = set(inventory)
    missing_members = sorted(inventoried_keys.difference(archived_keys))
    extra_members = sorted(archived_keys.difference(inventoried_keys))
    if missing_members:
        raise DebugArchiveValidationError(
            f"{archive_path}: arrays declared by {MANIFEST_KEY}.arrays are missing: {missing_members}"
        )
    if extra_members:
        raise DebugArchiveValidationError(
            f"{archive_path}: arrays are present but absent from {MANIFEST_KEY}.arrays: {extra_members}"
        )

    all_required = set(manifest["required_keys"]).union(caller_required)
    missing_required = sorted(all_required.difference(archived_keys))
    if missing_required:
        raise DebugArchiveValidationError(
            f"{archive_path}: required incident arrays are missing: {missing_required}"
        )

    for key, array in arrays.items():
        expected = inventory[key]
        actual_descriptor = _dtype_descriptor(array.dtype)
        if array.dtype.str != expected["dtype"] or actual_descriptor != expected["dtype_descr"]:
            raise DebugArchiveValidationError(
                f"{context}.arrays.{key}.dtype: expected {expected['dtype']!r} with descriptor "
                f"{expected['dtype_descr']!r}, found {array.dtype.str!r} with descriptor "
                f"{actual_descriptor!r}"
            )
        actual_shape = list(array.shape)
        if actual_shape != expected["shape"]:
            raise DebugArchiveValidationError(
                f"{context}.arrays.{key}.shape: expected {expected['shape']}, found {actual_shape}"
            )
        if int(array.nbytes) != expected["nbytes"]:
            raise DebugArchiveValidationError(
                f"{context}.arrays.{key}.nbytes: expected {expected['nbytes']}, found {array.nbytes}"
            )
        actual_checksum = _array_sha256(array)
        if not hmac.compare_digest(actual_checksum, expected["sha256"]):
            raise DebugArchiveValidationError(
                f"{context}.arrays.{key}.sha256: checksum mismatch; expected "
                f"{expected['sha256']}, found {actual_checksum}"
            )


def _validate_manifest_structure(manifest: Mapping[str, Any], context: str) -> None:
    if not isinstance(manifest, Mapping):
        raise DebugArchiveValidationError(
            f"{context}: expected a JSON object, got {type(manifest).__name__}"
        )
    fields = set(manifest)
    missing = sorted(_MANIFEST_FIELDS.difference(fields))
    extra = sorted(fields.difference(_MANIFEST_FIELDS))
    if missing:
        raise DebugArchiveValidationError(f"{context}: required manifest fields are missing: {missing}")
    if extra:
        raise DebugArchiveValidationError(f"{context}: unknown manifest fields are present: {extra}")
    if manifest["format_name"] != FORMAT_NAME:
        raise DebugArchiveValidationError(
            f"{context}.format_name: expected {FORMAT_NAME!r}, found {manifest['format_name']!r}"
        )
    version = manifest["format_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != FORMAT_VERSION:
        raise DebugArchiveValidationError(
            f"{context}.format_version: expected {FORMAT_VERSION}, found {version!r}; "
            "use a compatible archive reader or migrate the archive"
        )
    _validate_status(manifest["status"], f"{context}.status")
    _validate_timestamp(manifest["created_at"], f"{context}.created_at")
    required = _validate_required_keys(manifest["required_keys"], f"{context}.required_keys")
    if required != manifest["required_keys"]:
        raise DebugArchiveValidationError(
            f"{context}.required_keys: keys must be unique and lexicographically sorted"
        )
    _validate_inventory(manifest["arrays"], f"{context}.arrays")
    missing_required = sorted(set(required).difference(manifest["arrays"]))
    if missing_required:
        raise DebugArchiveValidationError(
            f"{context}.required_keys: keys absent from the array inventory: {missing_required}"
        )
    _validate_dependencies(manifest["dependencies"], f"{context}.dependencies")
    if not isinstance(manifest["runtime"], Mapping):
        raise DebugArchiveValidationError(f"{context}.runtime: expected a JSON object")
    if not isinstance(manifest["metadata"], Mapping):
        raise DebugArchiveValidationError(f"{context}.metadata: expected a JSON object")


def _validate_status(status: object, context: str) -> None:
    if not isinstance(status, str) or status not in ARCHIVE_STATUSES:
        raise DebugArchiveValidationError(
            f"{context}: expected one of {sorted(ARCHIVE_STATUSES)}, found {status!r}"
        )


def _validate_timestamp(timestamp: object, context: str) -> None:
    if not isinstance(timestamp, str) or not timestamp:
        raise DebugArchiveValidationError(f"{context}: expected a timezone-aware ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DebugArchiveValidationError(
            f"{context}: invalid ISO 8601 timestamp {timestamp!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DebugArchiveValidationError(
            f"{context}: timestamp must include an explicit UTC offset, found {timestamp!r}"
        )


def _validate_inventory(inventory: object, context: str) -> None:
    if not isinstance(inventory, Mapping):
        raise DebugArchiveValidationError(f"{context}: expected a JSON object")
    if list(inventory) != sorted(inventory):
        raise DebugArchiveValidationError(f"{context}: array entries must be lexicographically sorted")
    for key, entry in inventory.items():
        _validate_array_key(key, context)
        entry_context = f"{context}.{key}"
        if not isinstance(entry, Mapping):
            raise DebugArchiveValidationError(f"{entry_context}: expected a JSON object")
        fields = set(entry)
        missing = sorted(_INVENTORY_FIELDS.difference(fields))
        extra = sorted(fields.difference(_INVENTORY_FIELDS))
        if missing or extra:
            raise DebugArchiveValidationError(
                f"{entry_context}: inventory fields mismatch; missing={missing}, extra={extra}"
            )
        if not isinstance(entry["dtype"], str) or not entry["dtype"]:
            raise DebugArchiveValidationError(f"{entry_context}.dtype: expected a non-empty string")
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
        ):
            raise DebugArchiveValidationError(
                f"{entry_context}.shape: expected a list of non-negative integer dimensions"
            )
        nbytes = entry["nbytes"]
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise DebugArchiveValidationError(
                f"{entry_context}.nbytes: expected a non-negative integer"
            )
        checksum = entry["sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise DebugArchiveValidationError(
                f"{entry_context}.sha256: expected a lowercase 64-character SHA-256 digest"
            )


def _validate_dependencies(dependencies: object, context: str) -> None:
    if not isinstance(dependencies, Mapping):
        raise DebugArchiveValidationError(f"{context}: expected a JSON object")
    for name, record in dependencies.items():
        if not isinstance(name, str) or not name:
            raise DebugArchiveValidationError(f"{context}: dependency names must be non-empty strings")
        record_context = f"{context}.{name}"
        if not isinstance(record, Mapping):
            raise DebugArchiveValidationError(f"{record_context}: expected a JSON object")
        status = record.get("status")
        if not isinstance(status, str) or status not in _PROVENANCE_STATUSES:
            raise DebugArchiveValidationError(
                f"{record_context}.status: expected one of {sorted(_PROVENANCE_STATUSES)}, found {status!r}"
            )
        if status == "available":
            if not isinstance(record.get("version"), str) or not record["version"]:
                raise DebugArchiveValidationError(
                    f"{record_context}.version: available dependencies require a non-empty version"
                )
        elif not isinstance(record.get("reason"), str) or not record["reason"]:
            raise DebugArchiveValidationError(
                f"{record_context}.reason: unavailable/error dependencies require a reason"
            )
        _validate_direct_url(record.get("direct_url"), f"{record_context}.direct_url")


def _validate_direct_url(direct_url: object, context: str) -> None:
    if not isinstance(direct_url, Mapping):
        raise DebugArchiveValidationError(f"{context}: expected a JSON object")
    status = direct_url.get("status")
    if not isinstance(status, str) or status not in _PROVENANCE_STATUSES:
        raise DebugArchiveValidationError(
            f"{context}.status: expected one of {sorted(_PROVENANCE_STATUSES)}, found {status!r}"
        )
    if status == "available":
        if not isinstance(direct_url.get("value"), Mapping):
            raise DebugArchiveValidationError(
                f"{context}.value: available direct-URL provenance requires a JSON object"
            )
    elif status == "unavailable":
        if not isinstance(direct_url.get("reason"), str) or not direct_url["reason"]:
            raise DebugArchiveValidationError(
                f"{context}.reason: unavailable direct-URL provenance requires a reason"
            )
    elif not isinstance(direct_url.get("message"), str) or not direct_url["message"]:
        raise DebugArchiveValidationError(
            f"{context}.message: direct-URL provenance errors require a message"
        )


def _encode_manifest(manifest: Mapping[str, Any], context: str) -> str:
    try:
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = _strict_json_loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DebugArchiveValidationError(
            f"{context}: manifest is not strict JSON-compatible: {exc}"
        ) from exc
    if decoded != manifest:
        raise DebugArchiveValidationError(
            f"{context}: manifest changes during a JSON round trip; use only string object keys, "
            "JSON arrays, and JSON scalar values"
        )
    return encoded


def _decode_manifest(encoded: np.ndarray, archive_path: Path) -> dict[str, Any]:
    context = f"{archive_path}::{MANIFEST_KEY}"
    if encoded.dtype.hasobject:
        raise DebugArchiveValidationError(
            f"{context}: object dtype is forbidden; the manifest must be a scalar UTF-8 string"
        )
    if encoded.shape != () or encoded.dtype.kind not in {"U", "S"}:
        raise DebugArchiveValidationError(
            f"{context}: expected a scalar string array, found shape={encoded.shape}, dtype={encoded.dtype}"
        )
    value = encoded.item()
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DebugArchiveValidationError(f"{context}: manifest is not valid UTF-8") from exc
    else:
        text = str(value)
    try:
        manifest = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DebugArchiveValidationError(f"{context}: invalid strict JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DebugArchiveValidationError(
            f"{context}: expected a JSON object, got {type(manifest).__name__}"
        )
    return manifest


def _strict_json_loads(text: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r}")

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
