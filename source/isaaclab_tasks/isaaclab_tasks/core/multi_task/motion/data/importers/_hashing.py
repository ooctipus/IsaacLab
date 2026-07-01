# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming hashes for native motion sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

NativeField = np.ndarray | int | str


def file_sha256(path: Path) -> str:
    """Hash one file without retaining its bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_sources_sha256(sources: Sequence[tuple[str, str]]) -> str:
    """Hash ordered source identifiers and their file digests."""
    payload = json.dumps(sources, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def clip_sha256(fields: Mapping[str, NativeField], field_order: Sequence[str]) -> str:
    """Hash exact native arrays and honestly encoded Python scalars."""
    digest = hashlib.sha256()
    for name in field_order:
        value = fields[name]
        if isinstance(value, np.ndarray):
            if not value.flags.c_contiguous:
                raise ValueError(f"Native field {name!r} must be C-contiguous.")
            declaration = {"name": name, "dtype": value.dtype.str, "shape": value.shape}
            raw = memoryview(value).cast("B")
        elif type(value) is int:
            declaration = {"name": name, "scalar_type": "python:int", "scalar_encoding": "UTF-8 decimal"}
            raw = str(value).encode()
        elif type(value) is str:
            declaration = {"name": name, "scalar_type": "python:str", "scalar_encoding": "UTF-8"}
            raw = value.encode()
        else:
            raise TypeError(f"Native field {name!r} has unsupported type {type(value).__name__}.")
        metadata = json.dumps(declaration, separators=(",", ":"), sort_keys=True).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()
