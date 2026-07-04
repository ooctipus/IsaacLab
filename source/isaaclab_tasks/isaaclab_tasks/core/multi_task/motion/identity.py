# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical identity helpers for immutable motion metadata."""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path

_HEX = frozenset(string.hexdigits.lower())


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value with deterministic byte encoding."""
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one file without retaining its bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_sha256(name: str, value: str) -> None:
    """Reject values that are not lowercase hexadecimal SHA-256 digests."""
    if len(value) != 64 or set(value) - _HEX or value != value.lower():
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")


def validate_nonempty(name: str, value: str) -> None:
    """Reject empty identity and convention strings."""
    if not value:
        raise ValueError(f"{name} must not be empty.")
