# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public declarations and process-wide registry for physics replay adapters."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

REPLAY_STAGES: frozenset[str] = frozenset({"state", "transition", "solver", "operation"})
"""Replay stages accepted by strict physics debug archives."""


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayCapability:
    """Replay capability declared by an incident manifest.

    Args:
        capability_id: Stable capability identifier within the archive.
        stage: Replay stage selected by the capability.
        status: Completeness status declared by the recorder.
        provider: Provider identifier required by the adapter.
        fields: Exact archive fields declared by the capability.
        adapter: Adapter identifier required to execute the capability.
        reason: Explanation for a non-complete capability.
    """

    capability_id: str
    stage: str
    status: str
    provider: str | None
    fields: tuple[str, ...] | None
    adapter: str | None
    reason: str | None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the capability.

        Returns:
            Capability fields represented with JSON-compatible containers.
        """
        return {
            "capability_id": self.capability_id,
            "stage": self.stage,
            "status": self.status,
            "provider": self.provider,
            "fields": list(self.fields) if self.fields is not None else None,
            "adapter": self.adapter,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Validated archive inputs passed to a replay adapter.

    Args:
        archive_path: Validated archive path.
        arrays: Validated archive arrays keyed by exact field name.
        manifest: Validated strict archive manifest.
        capability: Selected complete replay capability.
    """

    archive_path: Path
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, object]
    capability: ReplayCapability


ReplayCallback = Callable[[ReplayRequest], Mapping[str, object] | None]
"""Callback executed for one validated replay request."""


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayAdapter:
    """Explicit implementation for one or more replay providers.

    Args:
        adapter_id: Stable identifier matched against archive capabilities.
        stages: Supported replay stages.
        providers: Supported provider identifiers.
        required_fields: Exact archive fields required by the callback.
        callback: Replay implementation invoked after strict validation.
    """

    adapter_id: str
    stages: frozenset[str]
    providers: frozenset[str]
    required_fields: tuple[str, ...]
    callback: ReplayCallback = dataclasses.field(repr=False)


_REPLAY_ADAPTERS: dict[str, ReplayAdapter] = {}


def register_replay_adapter(adapter: ReplayAdapter) -> None:
    """Register one trusted replay implementation in the process-wide registry.

    Registration is explicit and never follows an archive-controlled module
    name. Importing the same adapter declaration twice fails.

    Args:
        adapter: Adapter declaration and execution callback.

    Raises:
        TypeError: If the declaration has the wrong type or field container
            types.
        ValueError: If a declaration field is empty, invalid, duplicated, or
            already registered.
    """
    if not isinstance(adapter, ReplayAdapter):
        raise TypeError(
            "Replay adapters must be ReplayAdapter instances, "
            f"got {type(adapter).__module__}.{type(adapter).__qualname__}."
        )
    if not isinstance(adapter.adapter_id, str):
        raise TypeError("Replay adapter_id must be a string.")
    if not adapter.adapter_id or adapter.adapter_id != adapter.adapter_id.strip():
        raise ValueError("Replay adapter_id must be a non-empty string without surrounding whitespace.")
    if not isinstance(adapter.stages, frozenset) or any(not isinstance(stage, str) for stage in adapter.stages):
        raise TypeError("Replay adapter stages must be a frozenset of strings.")
    if not adapter.stages or not adapter.stages.issubset(REPLAY_STAGES):
        raise ValueError(
            f"Replay adapter {adapter.adapter_id!r} has invalid stages {sorted(adapter.stages)}; "
            f"valid stages are {sorted(REPLAY_STAGES)}."
        )
    if not isinstance(adapter.providers, frozenset) or any(
        not isinstance(provider, str) for provider in adapter.providers
    ):
        raise TypeError("Replay adapter providers must be a frozenset of strings.")
    if not adapter.providers or any(not provider or provider != provider.strip() for provider in adapter.providers):
        raise ValueError(f"Replay adapter {adapter.adapter_id!r} must declare non-empty providers.")
    if not isinstance(adapter.required_fields, tuple) or any(
        not isinstance(field, str) for field in adapter.required_fields
    ):
        raise TypeError("Replay adapter required_fields must be a tuple of strings.")
    if len(adapter.required_fields) != len(set(adapter.required_fields)) or any(
        not field or field != field.strip() for field in adapter.required_fields
    ):
        raise ValueError(f"Replay adapter {adapter.adapter_id!r} required_fields must be unique non-empty strings.")
    if not callable(adapter.callback):
        raise TypeError(f"Replay adapter {adapter.adapter_id!r} callback must be callable.")
    if adapter.adapter_id in _REPLAY_ADAPTERS:
        raise ValueError(f"Replay adapter {adapter.adapter_id!r} is already registered.")
    _REPLAY_ADAPTERS[adapter.adapter_id] = adapter


def get_replay_adapter(adapter_id: str) -> ReplayAdapter | None:
    """Return the registered adapter with an exact identifier.

    Args:
        adapter_id: Exact adapter identifier.

    Returns:
        Registered adapter, or None when the identifier is unknown.
    """
    return _REPLAY_ADAPTERS.get(adapter_id)


def get_replay_adapter_ids() -> tuple[str, ...]:
    """Return registered adapter identifiers in deterministic order.

    Returns:
        Sorted adapter identifiers.
    """
    return tuple(sorted(_REPLAY_ADAPTERS))
