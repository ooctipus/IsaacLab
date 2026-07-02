# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkout-independent identities for complete Python source packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular source file."""
    path = path.expanduser()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Python source identity requires a regular non-symbolic file: {path}.")
    path = path.resolve()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def python_package_identity(package_root: Path) -> dict[str, object]:
    """Return every package member and its checkout-independent bundle digest.

    Relative paths, content digests, and the complete file count are retained
    so a later bundle mismatch can identify its exact changed members.

    Args:
        package_root: Root directory of the imported Python package.

    Returns:
        Member manifest and canonical SHA-256 digest of all Python sources.

    Raises:
        ValueError: If the package root is invalid or contains no Python files.
    """
    package_root = package_root.expanduser()
    if not package_root.is_dir() or package_root.is_symlink():
        raise ValueError(f"Python package root must be a non-symbolic directory: {package_root}.")
    package_root = package_root.resolve()
    members: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*.py"), key=lambda value: value.relative_to(package_root).as_posix()):
        relative = path.relative_to(package_root).as_posix()
        members.append({"path": relative, "sha256": _file_sha256(path)})
    if not members:
        raise ValueError(f"Python package root contains no Python sources: {package_root}.")
    payload = {"python_file_count": len(members), "python_files": members}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**payload, "bundle_sha256": hashlib.sha256(encoded).hexdigest()}


def python_package_bundle_sha256(package_root: Path) -> str:
    """Return the canonical digest from :func:`python_package_identity`.

    Args:
        package_root: Root directory of the imported Python package.

    Returns:
        Canonical SHA-256 digest of all Python source members.
    """
    digest = python_package_identity(package_root)["bundle_sha256"]
    if not isinstance(digest, str):
        raise TypeError("Python package identity returned a non-string bundle digest.")
    return digest


__all__ = ["python_package_bundle_sha256", "python_package_identity"]
