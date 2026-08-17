# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict debug capture and incident replay for Newton physics."""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

import numpy as np
import torch
import warp as wp

from ._debug_archive import write_archive
from ._debug_capture import (
    DebugCaptureBinding,
    DebugCaptureError,
    DebugCaptureField,
    DebugCapturePlan,
    DebugMappingKey,
    DebugSchemaError,
    clone_debug_value,
    debug_value_to_numpy,
    register_debug_container_type,
)
from .newton_manager_cfg import NewtonDebugCaptureCfg, NewtonDebugReplayCfg

if TYPE_CHECKING:
    from newton import Model, State

    from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

_TRIGGER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DETECT_NONFINITE_PROVIDERS = frozenset(
    {"state", "model", "control", "contacts", "collision_pipeline", "solver", "context", "operations"}
)
_ATTRIBUTE_FREQUENCY_PROVIDERS = frozenset({"state", "model", "control", "contacts"})


class _OperationProviderCleanupError(DebugCaptureError):
    """Recorder initialization failed and provider cleanup remains pending."""

    def __init__(
        self,
        action: str,
        operation_error: Exception,
        cleanup_error: DebugCaptureError,
    ) -> None:
        self.action = action
        self.operation_error = operation_error
        self.cleanup_error = cleanup_error
        super().__init__(
            f"operation_provider.{action} failed: {operation_error}. "
            f"Cleanup also failed: {cleanup_error}. "
            "Provider ownership is retained for a later close() retry."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _IncidentScope:
    """One independently exported failure scope."""

    world_id: int | None
    is_global: bool

    @property
    def label(self) -> str:
        """Stable filename label for this scope."""
        return "global" if self.is_global else f"world{self.world_id:06d}"


@dataclasses.dataclass(frozen=True, slots=True)
class _NormalizedTriggerResult:
    """Validated incident scopes and reason from one trigger."""

    reason: str
    scopes: frozenset[_IncidentScope]


_CapturePath = tuple[str | int | DebugMappingKey, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _RowPartition:
    """Validated active rows and their owning worlds for one provider."""

    contract: str
    provider: str
    capacity: int
    count_path: _CapturePath
    container_path: _CapturePath
    row_paths: frozenset[_CapturePath]
    world0: np.ndarray
    world1: np.ndarray

    def indices(self, scope: _IncidentScope, *, failed_worlds_only: bool) -> np.ndarray:
        """Return active rows relevant to one incident scope."""
        active = np.arange(self.world0.size, dtype=np.int64)
        if not failed_worlds_only:
            return active
        if scope.is_global:
            return active[(self.world0 < 0) & (self.world1 < 0)]
        assert scope.world_id is not None
        return active[(self.world0 == scope.world_id) | (self.world1 == scope.world_id)]

    def applies(self, field: DebugCaptureField) -> bool:
        """Return whether this partition owns one captured field."""
        if field.path == self.count_path:
            return True
        return field.path in self.row_paths


@dataclasses.dataclass(slots=True)
class _IncidentSources:
    """Composite root used by one frozen incident plan."""

    state: object
    model: object | None
    control: object | None
    contacts: object | None
    collision_pipeline: object | None
    solver: object | None
    context: Mapping[str, object] | None
    operations: object | None


@dataclasses.dataclass(slots=True)
class _ReplaySources:
    """Composite root used by one frozen transition plan."""

    state: object | None
    control: object | None
    contacts: object | None
    collision_pipeline: object | None
    solver: object | None
    context: Mapping[str, object] | None
    operations: object | None


@dataclasses.dataclass(slots=True)
class _CaptureSlot:
    """Preallocated values for one frozen capture binding."""

    binding: DebugCaptureBinding
    values: list[object]

    @classmethod
    def allocate(cls, binding: DebugCaptureBinding, root: object) -> _CaptureSlot:
        """Allocate one reusable slot from a live root."""
        snapshot = binding.clone(root)
        return cls(binding=binding, values=[captured.value for captured in snapshot.values])

    def assign(self, root: object) -> None:
        """Copy a live root into this slot without changing its schema."""
        for index, field in enumerate(self.binding.fields):
            source = field.validate(root)
            self.values[index] = _copy_debug_value(self.values[index], source, field.display_path)

    def numpy(self, index: int) -> np.ndarray:
        """Return an independent NumPy copy of one stored field."""
        field = self.binding.fields[index]
        return debug_value_to_numpy(self.values[index], field.display_path)


class _WorldLayout:
    """Discovered Newton world partitions, including both global regions."""

    def __init__(self, model: Model, model_plan: DebugCapturePlan) -> None:
        world_count = object.__getattribute__(model, "world_count")
        if isinstance(world_count, bool) or not isinstance(world_count, Integral) or int(world_count) < 1:
            raise DebugSchemaError(
                "model.world_count must be a non-boolean integer greater than or equal to 1, "
                f"got {world_count!r}."
            )
        self.world_count = int(world_count)
        self._starts: dict[str, np.ndarray] = {}
        self._owners: dict[str, np.ndarray] = {}
        start_paths: dict[str, str] = {}
        suffix = "_world_start"
        for field in model_plan.fields:
            if (
                not field.path
                or any(not isinstance(step, str) for step in field.path)
                or not str(field.path[-1]).endswith(suffix)
            ):
                continue
            parts = [str(step) for step in field.path]
            parts[-1] = parts[-1][: -len(suffix)]
            frequency = ":".join(parts).lower()
            if frequency in self._starts:
                raise DebugSchemaError(
                    f"World frequency '{frequency}' has duplicate start anchors "
                    f"'{start_paths[frequency]}' and '{field.display_path}'."
                )
            starts = debug_value_to_numpy(field.get(model), field.display_path)
            if starts.ndim != 1:
                raise DebugSchemaError(f"World start array {field.display_path} must be one-dimensional.")
            if not np.issubdtype(starts.dtype, np.integer):
                raise DebugSchemaError(f"World start array {field.display_path} must have an integer dtype.")
            starts = starts.astype(np.int64, copy=False)
            if starts.size > 1 and np.any(starts[1:] < starts[:-1]):
                raise DebugSchemaError(f"World start array {field.display_path} must be monotonic.")
            if starts.size not in (self.world_count + 1, self.world_count + 2):
                raise DebugSchemaError(
                    f"World start array {field.display_path} has length {starts.size}; expected "
                    f"{self.world_count + 1} without global regions or "
                    f"{self.world_count + 2} including global regions."
                )
            self._starts[frequency] = starts
            start_paths[frequency] = field.display_path

        frequency_map = vars(model).get("attribute_frequency", {})
        if frequency_map is None:
            frequency_map = {}
        if not isinstance(frequency_map, Mapping):
            raise DebugSchemaError("model.attribute_frequency must be a mapping.")
        invalid_frequency_keys = [
            name for name in frequency_map if not isinstance(name, str) or not name
        ]
        if invalid_frequency_keys:
            raise DebugSchemaError(
                "model.attribute_frequency keys must be non-empty strings; "
                f"invalid keys: {invalid_frequency_keys!r}."
            )
        self._attribute_frequency = {
            name: _frequency_name(frequency) for name, frequency in frequency_map.items()
        }
        for field in model_plan.fields:
            if not field.path or any(not isinstance(step, str) for step in field.path):
                continue
            attribute_name = ":".join(str(step) for step in field.path)
            frequency = self._attribute_frequency.get(attribute_name)
            if frequency in self._starts or attribute_name.lower() != f"{frequency}_world":
                continue
            owners = debug_value_to_numpy(field.get(model), field.display_path)
            if owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer):
                raise DebugSchemaError(f"World owner array {field.display_path} must be a one-dimensional integer array.")
            owners = owners.astype(np.int64, copy=False)
            if np.any((owners < -1) | (owners >= self.world_count)):
                raise DebugSchemaError(
                    f"World owner array {field.display_path} contains an index outside "
                    f"[-1, {self.world_count - 1}]."
                )
            self._owners[frequency] = owners
        self.validate_fields(model_plan.fields, composite_root=False)

    @property
    def is_multi(self) -> bool:
        """Whether environment-local incident artifacts are meaningful."""
        return self.world_count > 1

    def frequency_for(self, field: DebugCaptureField, *, composite_root: bool = False) -> str | None:
        """Resolve and validate a field's explicit world ownership contract."""
        if field.shape and field.shape[0] == 0:
            return "once"
        symbolic = field.symbolic_shape
        if symbolic and symbolic[0] == "nworld":
            self._validate_owned_extent(
                field,
                "world",
                source="symbolic leading dimension 'nworld'",
            )
            return "world"

        path = field.path
        use_attribute_frequency = True
        if composite_root:
            if not path:
                return None
            provider = path[0]
            use_attribute_frequency = (
                isinstance(provider, str) and provider in _ATTRIBUTE_FREQUENCY_PROVIDERS
            )
            path = path[1:]
        if not path or any(not isinstance(step, str) for step in path):
            return None
        if str(path[-1]).endswith("_world_start"):
            return "once"

        attribute_name = ":".join(str(step) for step in path)
        frequency = (
            self._attribute_frequency.get(attribute_name) if use_attribute_frequency else None
        )
        if frequency is not None:
            self._validate_owned_extent(
                field,
                frequency,
                source=f"model.attribute_frequency[{attribute_name!r}]={frequency!r}",
            )
            return frequency

        if not field.shape:
            return "once"
        return None

    def validate_fields(
        self,
        fields: tuple[DebugCaptureField, ...],
        *,
        composite_root: bool,
    ) -> None:
        """Validate every explicit ownership contract in a frozen field set."""
        for field in fields:
            self.frequency_for(field, composite_root=composite_root)

    def _validate_owned_extent(
        self,
        field: DebugCaptureField,
        frequency: str,
        *,
        source: str,
    ) -> None:
        """Require a trusted ownership declaration to match its row extent."""
        if frequency == "once":
            return
        if not field.shape:
            raise DebugSchemaError(
                f"Explicit ownership {source} for '{field.display_path}' requires an array with a leading dimension."
            )
        if frequency == "world":
            expected = self.world_count
        else:
            starts = self._starts.get(frequency)
            owners = self._owners.get(frequency)
            if starts is None and owners is None:
                raise DebugSchemaError(
                    f"Explicit ownership {source} for '{field.display_path}' references frequency "
                    f"'{frequency}' without a matching discovered '*_world_start' or '*_world' ownership anchor."
                )
            expected = int(starts[-1]) if starts is not None else int(owners.size)
            if starts is not None and isinstance(field.path[-1], str) and field.path[-1].endswith("_start"):
                expected += 1
        actual = int(field.shape[0])
        if actual != expected:
            raise DebugSchemaError(
                f"Explicit ownership {source} for '{field.display_path}' has leading extent {actual}; "
                f"expected {expected}."
            )

    def local_indices(
        self,
        frequency: str,
        world_id: int,
        *,
        include_globals: bool,
    ) -> np.ndarray | None:
        """Return row indices for one world and optionally its global dependencies."""
        if frequency == "world":
            if 0 <= world_id < self.world_count:
                return np.asarray([world_id], dtype=np.int64)
            return None

        starts = self._starts.get(frequency)
        owners = self._owners.get(frequency)
        if not (0 <= world_id < self.world_count):
            return None
        if starts is None:
            if owners is None:
                return None
            selected = owners == world_id
            if include_globals:
                selected |= owners == -1
            return np.flatnonzero(selected)
        local = np.arange(int(starts[world_id]), int(starts[world_id + 1]), dtype=np.int64)
        if not include_globals or starts.size != self.world_count + 2:
            return local
        prefix = np.arange(0, int(starts[0]), dtype=np.int64)
        suffix = np.arange(int(starts[-2]), int(starts[-1]), dtype=np.int64)
        return np.concatenate((prefix, local, suffix))

    def global_indices(self, frequency: str) -> np.ndarray | None:
        """Return the prefix and tail indices occupied by global entities."""
        if frequency == "world":
            return np.empty(0, dtype=np.int64)
        starts = self._starts.get(frequency)
        if starts is None:
            owners = self._owners.get(frequency)
            return None if owners is None else np.flatnonzero(owners == -1)
        if starts.size != self.world_count + 2:
            return np.empty(0, dtype=np.int64)
        prefix = np.arange(0, int(starts[0]), dtype=np.int64)
        suffix = np.arange(int(starts[-2]), int(starts[-1]), dtype=np.int64)
        return np.concatenate((prefix, suffix))

    def validate_extent(self, frequency: str, first_dimension: int) -> bool:
        """Return whether a field extent agrees with its discovered frequency."""
        if frequency == "world":
            return first_dimension == self.world_count
        starts = self._starts.get(frequency)
        if starts is not None:
            return starts.size > 0 and int(starts[-1]) == first_dimension
        owners = self._owners.get(frequency)
        return owners is not None and owners.size == first_dimension


class PhysicsIncidentRecorder:
    """Rolling, schema-driven Newton physics incident recorder.

    Every allocated state array, including registered namespace arrays, is
    copied into a chronological GPU ring. Every floating field participates in
    non-finite detection. Incident and replay provider schemas are discovered,
    validated, and frozen during initialization.
    """

    def __init__(
        self,
        model: Model,
        cfg: NewtonDebugCaptureCfg,
        *,
        state: State,
        control: object | None,
        solver: object | None,
        contacts: object | None = None,
        collision_pipeline: object | None = None,
        scene_exporter: Callable[[str, list[int]], None] | None = None,
        context_provider: Callable[[], Mapping[str, object]] | None = None,
        operation_provider: NewtonManager.DebugOperationProvider | None = None,
        triggers: Mapping[
            str,
            Callable[[NewtonManager.DebugTriggerContext], NewtonManager.DebugTriggerResult | None],
        ]
        | None = None,
    ) -> None:
        """Initialize and bind every configured capture provider.

        Args:
            model: Finalized Newton model used for schemas and world partitions.
            cfg: Strict incident and transition recording configuration.
            state: Allocated live state used to freeze the required state schema.
            control: Stable control provider, required when incident or replay
                capture records control.
            solver: Stable solver provider when incident or replay capture requests it.
            contacts: Stable contact provider when incident or replay capture requests it.
            collision_pipeline: Stable external collision pipeline when incident or replay capture requests it.
            scene_exporter: Optional callable accepting a USD path and world IDs.
            context_provider: Optional callable returning workflow context as a mapping.
            operation_provider: Provider implementing ``bind(solver)``, ``snapshot()``,
                and ``close()`` when incident or replay operation recording is enabled.
            triggers: Frozen mapping of registered trigger names to callbacks.
                Callbacks are evaluated by the manager, never by this recorder.

        Raises:
            DebugCaptureError: If a required provider is absent, empty, or invalid.
            DebugSchemaError: If a provider schema or field selection is invalid.
        """
        if not isinstance(cfg, NewtonDebugCaptureCfg):
            raise TypeError("cfg must be a NewtonDebugCaptureCfg.")
        if not isinstance(cfg.replay, NewtonDebugReplayCfg):
            raise TypeError("cfg.replay must be a NewtonDebugReplayCfg.")
        for name in ("history_length", "max_incidents", "max_gpu_bytes"):
            value = object.__getattribute__(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "failed_worlds_only",
            "halt_on_incident",
            "fail_on_capture_error",
            "record_model",
            "record_control",
            "record_contacts",
            "record_collision_pipeline",
            "record_solver",
            "record_operations",
            "capture_per_substep",
            "include_private_fields",
            "readback_preflight",
        ):
            if not isinstance(object.__getattribute__(cfg, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string.")
        for name in (
            "enabled",
            "record_state",
            "record_control",
            "record_solver",
            "record_contacts",
            "record_collision_pipeline",
            "record_operations",
        ):
            if not isinstance(object.__getattribute__(cfg.replay, name), bool):
                raise TypeError(f"replay.{name} must be a bool.")
        if cfg.replay.record_operations and not cfg.replay.enabled:
            raise ValueError("replay.record_operations=True requires replay.enabled=True.")
        if cfg.replay.enabled and not any(
            (
                cfg.replay.record_state,
                cfg.replay.record_control,
                cfg.replay.record_solver,
                cfg.replay.record_contacts,
                cfg.replay.record_collision_pipeline,
                cfg.replay.record_operations,
            )
        ):
            raise ValueError("Replay must record at least one provider when enabled.")
        if state is None:
            raise DebugCaptureError("Physics incident capture requires a non-None state provider.")
        if scene_exporter is not None and not callable(scene_exporter):
            raise TypeError("scene_exporter must be callable or None.")
        if context_provider is not None and not callable(context_provider):
            raise TypeError("context_provider must be callable or None.")

        self._model = model
        self._state = state
        self._control = control
        self._solver = solver
        self._contacts = contacts
        self._collision_pipeline = collision_pipeline
        self._scene_exporter = scene_exporter
        self._context_provider = context_provider
        self._operation_provider = operation_provider
        self._trigger_callbacks = _validate_trigger_registry(triggers)
        self._active_trigger_scopes: dict[str, set[_IncidentScope]] = {
            name: set() for name in self._trigger_callbacks
        }

        self._history_length = cfg.history_length
        self._output_dir = cfg.output_dir
        self._failed_worlds_only = cfg.failed_worlds_only
        self._max_incidents = cfg.max_incidents
        self._halt_on_incident = cfg.halt_on_incident
        self._fail_on_capture_error = cfg.fail_on_capture_error
        self._record_model = cfg.record_model
        self._record_control = cfg.record_control
        self._record_contacts = cfg.record_contacts
        self._record_collision_pipeline = cfg.record_collision_pipeline
        self._record_solver = cfg.record_solver
        self._record_operations = cfg.record_operations
        self._detect_nonfinite_in = _validate_detect_nonfinite_in(cfg.detect_nonfinite_in)
        self._detect_nonfinite_include_fields = _validate_pattern_tuple(
            cfg.detect_nonfinite_include_fields,
            "detect_nonfinite_include_fields",
            allow_empty=False,
        )
        self._detect_nonfinite_exclude_fields = _validate_pattern_tuple(
            cfg.detect_nonfinite_exclude_fields,
            "detect_nonfinite_exclude_fields",
            allow_empty=True,
        )
        self._capture_per_substep = cfg.capture_per_substep
        self._include_private_fields = cfg.include_private_fields
        self._readback_preflight = cfg.readback_preflight
        self._include_fields = _validate_pattern_tuple(cfg.include_fields, "include_fields", allow_empty=False)
        self._exclude_fields = _validate_pattern_tuple(cfg.exclude_fields, "exclude_fields", allow_empty=True)
        self._max_gpu_bytes = cfg.max_gpu_bytes
        self._replay_cfg = cfg.replay
        self._replay_enabled = cfg.replay.enabled
        self._replay_include_fields = _validate_pattern_tuple(
            cfg.replay.include_fields, "replay.include_fields", allow_empty=False
        )
        self._replay_exclude_fields = _validate_pattern_tuple(
            cfg.replay.exclude_fields, "replay.exclude_fields", allow_empty=True
        )

        self._validate_required_provider_presence()

        _register_root_type(model)
        _register_newton_namespace_type(model)
        self._model_plan = DebugCapturePlan.build(
            model, root_name="model", include_private=self._include_private_fields
        )
        self._layout = _WorldLayout(model, self._model_plan)

        _register_root_type(state)
        self._state_plan = DebugCapturePlan.build(
            state, root_name="state", include_private=self._include_private_fields
        )
        if not self._state_plan.fields:
            raise DebugCaptureError("Required state provider has no capturable fields.")
        self._state_binding = self._state_plan.bind()
        self._layout.validate_fields(self._state_binding.fields, composite_root=False)
        _validate_archive_key_uniqueness(self._state_binding, "history__state__")
        bytes_per_state, state_report = _binding_nbytes(self._state_binding, state)
        pre_state_slots = 1 + int(self._capture_per_substep)
        required_state_bytes = bytes_per_state * (self._history_length + pre_state_slots)
        if required_state_bytes > self._max_gpu_bytes:
            details = ", ".join(f"{path}={size}" for path, size in state_report)
            raise DebugCaptureError(
                f"State history requires {required_state_bytes} GPU bytes for {self._history_length} ring slots "
                f"and {pre_state_slots} pre-state slot(s), exceeding max_gpu_bytes={self._max_gpu_bytes}. "
                f"Fields: {details}"
            )
        self._state_gpu_bytes = required_state_bytes
        self._ring_times = np.zeros(self._history_length, dtype=np.float64)
        self._ring = [_CaptureSlot.allocate(self._state_binding, state) for _ in range(self._history_length)]
        self._pending_pre = _CaptureSlot.allocate(self._state_binding, state)
        self._pending_pre_valid = False
        self._substep_pre = (
            _CaptureSlot.allocate(self._state_binding, state) if self._capture_per_substep else None
        )
        self._substep_pre_valid = False
        self._write_idx = 0
        self._valid_count = 0

        self._substep_counter = 0
        self._next_substep_idx: int | None = None
        self._last_finite_substep = -1
        self._substep_meta: tuple[int, int, int] | None = None

        if self._record_collision_pipeline or self._replay_cfg.record_collision_pipeline:
            _register_same_package_child_types(
                self._collision_pipeline,
                include_private=self._include_private_fields,
            )
        initial_context = self._snapshot_context() if context_provider is not None else None
        self._context = initial_context
        self._context_enabled = context_provider is not None
        self._operation_bound = False
        operation_recording = cfg.record_operations or cfg.replay.record_operations
        self._operations = self._bind_operation_provider() if operation_recording else None
        try:
            self._initialize_capture_plans(state)
        except Exception as initialization_error:
            try:
                self._close_operation_provider()
            except DebugCaptureError as close_error:
                raise _OperationProviderCleanupError(
                    "capture-plan initialization",
                    initialization_error,
                    close_error,
                ) from initialization_error
            raise

    def _initialize_capture_plans(self, state: State) -> None:
        """Bind incident and replay schemas after operation-provider ownership."""
        self._incident_count = 0
        self._halted = False
        self._active_failed_worlds: set[int] = set()
        self._active_global_incident = False

        self._incident_source = self._make_incident_source()
        self._incident_plan, self._incident_binding = self._bind_incident_plan(self._incident_source)
        if self._readback_preflight:
            self._preflight_readback("incident", self._incident_binding, self._incident_source)
        self._nonfinite_fields = self._bind_nonfinite_fields(self._incident_binding)
        self._build_live_partitions(self._incident_source, self._incident_binding)

        self._replay_plan: DebugCapturePlan | None = None
        self._replay_binding: DebugCaptureBinding | None = None
        self._replay_source: _ReplaySources | None = None
        self._replay_pre: list[_CaptureSlot] = []
        self._replay_post: list[_CaptureSlot] = []
        self._replay_meta: list[dict[str, float | int]] = [{} for _ in range(self._history_length)]
        self._replay_gpu_bytes = 0
        self._replay_partition_metadata: list[dict[str, object]] = []
        self._replay_write_idx = 0
        self._replay_valid_count = 0
        self._replay_current_idx: int | None = None
        if self._replay_enabled:
            self._bind_replay_plan(self._make_replay_source(state, refresh=False))
        self._warn_capture_inventory()

    def _preflight_readback(self, plan_name: str, binding: DebugCaptureBinding, source: object) -> None:
        """Read selected fields individually with a path breadcrumb before each copy."""
        field_count = len(binding.fields)
        for index, field in enumerate(binding.fields, start=1):
            logger.warning(
                "Physics debug readback preflight: plan=%s; field=%d/%d; path=%s",
                plan_name,
                index,
                field_count,
                field.display_path,
            )
            try:
                value = clone_debug_value(field.validate(source), field.display_path)
                debug_value_to_numpy(value, field.display_path)
            except DebugCaptureError as exc:
                raise DebugCaptureError(
                    f"Physics debug readback preflight failed at '{field.display_path}': {exc}"
                ) from exc
        logger.warning("Physics debug readback preflight completed: plan=%s; fields=%d", plan_name, field_count)

    def _warn_capture_inventory(self) -> None:
        """Warn once with every known ignored or unallocated resource."""
        plans = [
            ("model", self._model_plan),
            ("state", self._state_plan),
            ("incident", self._incident_plan),
        ]
        if self._replay_plan is not None:
            plans.append(("replay", self._replay_plan))

        lines: list[str] = []
        for plan_name, plan in plans:
            for category, entries in (("unallocated", plan.unallocated), ("ignored", plan.ignored)):
                for entry in entries:
                    lines.append(
                        f"- plan={plan_name}; category={category}; path={entry.display_path}; "
                        f"value_type={entry.value_type!r}; annotation={entry.annotation!r}; "
                        f"symbolic_shape={entry.symbolic_shape!r}; reason={entry.reason}"
                    )
        if lines:
            logger.warning(
                "Physics incident recorder initialized with known ignored or unallocated resources. "
                "Complete capture inventory follows:\n%s",
                "\n".join(lines),
            )

    @property
    def history_length(self) -> int:
        """Configured snapshot capacity."""
        return self._history_length

    @property
    def halted(self) -> bool:
        """Whether the caller should stop simulation after an incident."""
        return self._halted

    @property
    def capture_per_substep(self) -> bool:
        """Whether per-substep observation is enabled."""
        return self._capture_per_substep

    def rearm_reset_worlds(
        self,
        world_ids: tuple[int, ...],
        *,
        all_worlds: bool = False,
    ) -> None:
        """Rearm incident scopes whose worlds were explicitly reset.

        This must run before :meth:`capture_pre` for the next dispatch. World
        scopes not listed in :paramref:`world_ids` remain active. Global
        scopes are rearmed only when :paramref:`all_worlds` is true and the
        IDs exactly cover every allocated world.

        Args:
            world_ids: Unique reset world indices.
            all_worlds: Whether every allocated world was reset.

        Raises:
            TypeError: If the argument types are invalid.
            ValueError: If IDs are duplicated, out of range, or inconsistent
                with :paramref:`all_worlds`.
            DebugCaptureError: If the next dispatch already has a pre-state.
        """
        if self._pending_pre_valid:
            raise DebugCaptureError(
                "rearm_reset_worlds() must run before capture_pre() for the next dispatch."
            )
        if not isinstance(world_ids, tuple):
            raise TypeError("world_ids must be a tuple of integers.")
        if any(not isinstance(world_id, int) or isinstance(world_id, bool) for world_id in world_ids):
            raise TypeError("world_ids must contain only integers.")
        if len(set(world_ids)) != len(world_ids):
            raise ValueError("world_ids must not contain duplicates.")
        invalid = [
            world_id
            for world_id in world_ids
            if world_id < 0 or world_id >= self._layout.world_count
        ]
        if invalid:
            raise ValueError(
                f"world_ids contains IDs outside [0, {self._layout.world_count}): {invalid}."
            )
        if not isinstance(all_worlds, bool):
            raise TypeError("all_worlds must be a bool.")
        reset_worlds = set(world_ids)
        allocated_worlds = set(range(self._layout.world_count))
        if all_worlds and reset_worlds != allocated_worlds:
            raise ValueError(
                "all_worlds=True requires world_ids to contain exactly every allocated world."
            )

        self._active_failed_worlds.difference_update(reset_worlds)
        for active_scopes in self._active_trigger_scopes.values():
            reset_scopes = {
                scope
                for scope in active_scopes
                if scope.world_id is not None and scope.world_id in reset_worlds
            }
            active_scopes.difference_update(reset_scopes)
            if all_worlds:
                active_scopes.discard(_IncidentScope(world_id=None, is_global=True))
        if all_worlds:
            self._active_global_incident = False

    def capture_pre(self, pre_state: State) -> None:
        """Capture the real pre-physics state for the next dispatch.

        Args:
            pre_state: Newton state immediately before physics integration.
        """
        if self._halted or not self._ring:
            return
        if self._pending_pre_valid:
            raise DebugCaptureError("capture_pre() called before the previous dispatch consumed its pre-state.")
        self._state_plan.validate(pre_state)
        self._pending_pre.assign(pre_state)
        self._pending_pre_valid = True
        if self._substep_pre is not None:
            self._substep_pre.assign(pre_state)
            self._substep_pre_valid = True
        self._next_substep_idx = 0
        self._last_finite_substep = -1
        self._substep_meta = None

    def step(
        self,
        current_state: State,
        sim_time: float,
        *,
        trigger_results: Mapping[str, object] | None = None,
    ) -> None:
        """Record one post-physics state and process dispatch-post incidents.

        Args:
            current_state: Post-physics Newton state.
            sim_time: Simulation time [s].
            trigger_results: Results returned by registered dispatch-post triggers.
        """
        if self._halted or not self._ring:
            return
        if not self._pending_pre_valid:
            raise DebugCaptureError("step() requires capture_pre() for the current physics dispatch.")
        try:
            normalized = self._normalize_trigger_results(trigger_results)
            self._refresh_incident_source(current_state)
            self._record_state(current_state, sim_time)
            bad_worlds, global_bad, bad_paths = self._detect_nonfinite()
            trigger_reasons = self._update_active_trigger_scopes(normalized)
            self._handle_detection(
                float(sim_time),
                bad_worlds,
                global_bad,
                bad_paths,
                trigger_reasons,
            )
        finally:
            self._pending_pre_valid = False
            self._next_substep_idx = None

    def observe_substep(
        self,
        current_state: State,
        sim_time: float,
        *,
        substep_idx: int = 0,
        trigger_results: Mapping[str, object] | None = None,
    ) -> None:
        """Record and inspect one solver-post substep.

        Args:
            current_state: State after the observed substep.
            sim_time: Simulation time [s].
            substep_idx: Substep index in the current dispatch.
            trigger_results: Results returned by registered solver-post triggers.

        Raises:
            DebugCaptureError: If the dispatch has no fresh pre-state.
        """
        if self._halted or not self._ring:
            return
        if not self._capture_per_substep or self._substep_pre is None:
            raise DebugCaptureError("observe_substep() requires capture_per_substep=True.")
        if not isinstance(substep_idx, int) or isinstance(substep_idx, bool) or substep_idx < 0:
            raise TypeError("substep_idx must be a non-negative integer.")
        if self._next_substep_idx is None or substep_idx == 0 and not self._pending_pre_valid:
            raise DebugCaptureError(
                "observe_substep() requires capture_pre() for the current physics dispatch."
            )
        if substep_idx != self._next_substep_idx:
            raise DebugCaptureError(
                f"observe_substep() expected substep_idx={self._next_substep_idx}, got {substep_idx}."
            )
        try:
            normalized = self._normalize_trigger_results(trigger_results)
            self._refresh_incident_source(current_state)
            self._substep_counter += 1
            self._record_state(current_state, sim_time)
            bad_worlds, global_bad, bad_paths = self._detect_nonfinite()
            trigger_reasons = self._update_active_trigger_scopes(normalized)

            self._pending_pre.values = list(self._substep_pre.values)
            self._pending_pre_valid = self._substep_pre_valid
            if bad_worlds or global_bad or trigger_reasons:
                self._substep_meta = (
                    substep_idx,
                    int(self._last_finite_substep),
                    int(self._substep_counter),
                )
            self._handle_detection(
                float(sim_time),
                bad_worlds,
                global_bad,
                bad_paths,
                trigger_reasons,
            )
            if not bad_worlds and not global_bad:
                self._substep_pre.assign(current_state)
                self._substep_pre_valid = True
                self._last_finite_substep = substep_idx
        finally:
            self._pending_pre_valid = False
            self._next_substep_idx = substep_idx + 1

    def record_step_replay_pre(
        self,
        pre_state: State,
        *,
        sim_time: float = 0.0,
        substep_idx: int = 0,
    ) -> None:
        """Record one transition input using the frozen replay plan.

        Args:
            pre_state: State before the solver transition.
            sim_time: Simulation time [s].
            substep_idx: Substep index in the dispatch.

        Raises:
            DebugCaptureError: If a transition is already open or a provider
                no longer matches its initialization-time schema.
        """
        if not self._replay_enabled or self._halted:
            return
        if self._replay_current_idx is not None:
            raise DebugCaptureError("Replay pre called before the previous transition was completed.")

        source = self._make_replay_source(pre_state)
        self._validate_replay_source(source)
        index = self._replay_write_idx
        self._replay_pre[index].assign(source)
        self._replay_meta[index] = {"sim_time": float(sim_time), "substep_idx": int(substep_idx)}
        self._replay_source = source
        self._replay_current_idx = index

    def record_step_replay_post(self, post_state: State) -> None:
        """Complete the open transition and make it chronologically valid.

        Args:
            post_state: State after the solver transition.

        Raises:
            DebugCaptureError: If no replay pre-record is open or a provider
                no longer matches its initialization-time schema.
        """
        if not self._replay_enabled or self._halted:
            return
        if self._replay_current_idx is None:
            raise DebugCaptureError("Replay post called without a matching replay pre record.")

        source = self._make_replay_source(post_state)
        self._validate_replay_source(source)
        index = self._replay_current_idx
        self._replay_post[index].assign(source)
        self._replay_source = source
        self._replay_write_idx = (index + 1) % self._history_length
        self._replay_valid_count = min(self._replay_valid_count + 1, self._history_length)
        self._replay_current_idx = None

    def clear(self) -> None:
        """Release all recording resources and close the operation provider."""
        try:
            self._close_operation_provider()
        finally:
            self._ring.clear()
            self._replay_pre.clear()
            self._replay_post.clear()
            self._history_length = 0
            self._write_idx = 0
            self._valid_count = 0
            self._replay_write_idx = 0
            self._replay_valid_count = 0
            self._replay_current_idx = None

    def _record_state(self, state: State, sim_time: float) -> None:
        self._ring[self._write_idx].assign(state)
        self._ring_times[self._write_idx] = float(sim_time)
        self._write_idx = (self._write_idx + 1) % self._history_length
        self._valid_count = min(self._valid_count + 1, self._history_length)

    def _validate_required_provider_presence(self) -> None:
        missing: set[str] = set()
        if self._record_control and self._control is None:
            missing.add("control")
        if self._record_contacts and self._contacts is None:
            missing.add("contacts")
        if self._record_collision_pipeline and self._collision_pipeline is None:
            missing.add("collision_pipeline")
        if self._record_solver and self._solver is None:
            missing.add("solver")
        if self._record_operations or self._replay_cfg.record_operations:
            if self._operation_provider is None:
                missing.add("operation_provider")
            if self._solver is None:
                missing.add("solver")
        if self._replay_enabled:
            if self._replay_cfg.record_control and self._control is None:
                missing.add("control")
            if self._replay_cfg.record_contacts and self._contacts is None:
                missing.add("contacts")
            if self._replay_cfg.record_collision_pipeline and self._collision_pipeline is None:
                missing.add("collision_pipeline")
            if self._replay_cfg.record_solver and self._solver is None:
                missing.add("solver")
        if missing:
            raise DebugCaptureError(
                "Physics incident recorder requires provider(s): " + ", ".join(sorted(missing)) + "."
            )


    def _snapshot_context(self) -> Mapping[str, object]:
        if self._context_provider is None:
            raise DebugCaptureError("Context provider is not configured.")
        try:
            value = self._context_provider()
        except Exception as exc:
            raise DebugCaptureError(f"context_provider() failed: {exc}") from exc
        if not isinstance(value, Mapping):
            raise DebugCaptureError(
                f"context_provider() must return a mapping, got {type(value).__module__}.{type(value).__qualname__}."
            )
        return dict(value)

    def _bind_operation_provider(self) -> object:
        if self._operation_provider is None or self._solver is None:
            raise DebugCaptureError("Operation recording requires operation_provider and solver.")
        try:
            bind = object.__getattribute__(self._operation_provider, "bind")
            snapshot = object.__getattribute__(self._operation_provider, "snapshot")
            close = object.__getattribute__(self._operation_provider, "close")
        except AttributeError as exc:
            raise DebugCaptureError(
                "operation_provider must define callable bind(solver), snapshot(), and close() methods."
            ) from exc
        if not callable(bind) or not callable(snapshot) or not callable(close):
            raise DebugCaptureError(
                "operation_provider must define callable bind(solver), snapshot(), and close() methods."
            )
        self._operation_bound = True
        try:
            bind(self._solver)
        except Exception as exc:
            self._raise_operation_provider_failure("bind(solver)", exc)
        try:
            return self._snapshot_operations()
        except Exception as exc:
            self._raise_operation_provider_failure("initial snapshot()", exc)

    def _raise_operation_provider_failure(
        self,
        action: str,
        operation_error: Exception,
    ) -> Never:
        """Close after failed provider initialization and raise both failures."""
        try:
            self._close_operation_provider()
        except DebugCaptureError as cleanup_error:
            raise _OperationProviderCleanupError(
                action,
                operation_error,
                cleanup_error,
            ) from operation_error
        if isinstance(operation_error, DebugCaptureError):
            raise operation_error
        raise DebugCaptureError(f"operation_provider.{action} failed: {operation_error}") from operation_error

    def _close_operation_provider(self) -> None:
        """Close an owned provider, retaining ownership when teardown fails."""
        if self._operation_provider is None or not self._operation_bound:
            return
        try:
            close = object.__getattribute__(self._operation_provider, "close")
            close()
        except Exception as exc:
            raise DebugCaptureError(f"operation_provider.close() failed: {exc}") from exc
        self._operation_bound = False

    def _snapshot_operations(self) -> object:
        if self._operation_provider is None or not self._operation_bound:
            raise DebugCaptureError("Operation provider has not been bound to the active solver.")
        snapshot = object.__getattribute__(self._operation_provider, "snapshot")
        try:
            value = snapshot()
        except Exception as exc:
            raise DebugCaptureError(f"operation_provider.snapshot() failed: {exc}") from exc
        if value is None:
            raise DebugCaptureError("operation_provider.snapshot() returned None.")
        return value

    def _refresh_incident_source(self, state: State) -> None:
        self._state = state
        self._incident_source.state = state
        if self._context_enabled:
            self._context = self._snapshot_context()
            self._incident_source.context = self._context
        if self._record_operations:
            self._operations = self._snapshot_operations()
            self._incident_source.operations = self._operations

    def _handle_detection(
        self,
        sim_time: float,
        bad_worlds: set[int],
        global_bad: bool,
        bad_paths: dict[str, list[str]],
        trigger_reasons: Mapping[_IncidentScope, Mapping[str, str]],
    ) -> None:
        new_worlds = set(bad_worlds.difference(self._active_failed_worlds))
        new_global = global_bad and not self._active_global_incident
        self._update_active_failures(bad_worlds, global_bad)

        scopes = {_IncidentScope(world_id=world_id, is_global=False) for world_id in new_worlds}
        if new_global:
            scopes.add(_IncidentScope(world_id=None, is_global=True))
        scopes.update(trigger_reasons)
        if not scopes:
            return

        if self._halt_on_incident:
            self._halted = True

        ordered_scopes = sorted(
            scopes,
            key=lambda scope: (scope.is_global, -1 if scope.world_id is None else scope.world_id),
        )
        if self._incident_count < self._max_incidents:
            errors = self._write_incident_artifacts(
                sim_time,
                ordered_scopes,
                bad_paths,
                trigger_reasons,
            )
            self._incident_count += 1
            if errors and self._fail_on_capture_error:
                detail = "; ".join(f"{error['provider']}: {error['message']}" for error in errors)
                raise DebugCaptureError(f"Incident artifacts were saved as partial, then capture failed: {detail}")
        else:
            logger.error(
                "Physics incident was not retained because max_incidents=%d was reached.",
                self._max_incidents,
            )

    def _normalize_trigger_results(
        self,
        trigger_results: Mapping[str, object] | None,
    ) -> dict[str, _NormalizedTriggerResult]:
        if trigger_results is None:
            trigger_results = {}
        if not isinstance(trigger_results, Mapping):
            raise TypeError("trigger_results must be a mapping or None.")
        unknown = sorted(set(trigger_results).difference(self._trigger_callbacks))
        if unknown:
            raise DebugCaptureError(f"Trigger results contain unregistered names: {unknown}.")

        normalized: dict[str, _NormalizedTriggerResult] = {}
        for name in sorted(trigger_results):
            result = trigger_results[name]
            try:
                reason = object.__getattribute__(result, "reason")
                world_ids = object.__getattribute__(result, "world_ids")
                global_scope = object.__getattribute__(result, "global_scope")
            except AttributeError as exc:
                raise DebugCaptureError(
                    f"Trigger result {name!r} must define reason, world_ids, and global_scope."
                ) from exc
            if not isinstance(reason, str) or not reason.strip():
                raise DebugCaptureError(f"Trigger result {name!r}.reason must be a non-empty string.")
            if not isinstance(world_ids, tuple):
                raise DebugCaptureError(f"Trigger result {name!r}.world_ids must be a tuple of integers.")
            if any(not isinstance(world_id, int) or isinstance(world_id, bool) for world_id in world_ids):
                raise DebugCaptureError(f"Trigger result {name!r}.world_ids must contain only integers.")
            if len(set(world_ids)) != len(world_ids):
                raise DebugCaptureError(f"Trigger result {name!r}.world_ids must not contain duplicates.")
            invalid_worlds = [
                world_id
                for world_id in world_ids
                if world_id < 0 or world_id >= self._layout.world_count
            ]
            if invalid_worlds:
                raise DebugCaptureError(
                    f"Trigger result {name!r} contains world IDs outside [0, {self._layout.world_count}): "
                    f"{invalid_worlds}."
                )
            if not isinstance(global_scope, bool):
                raise DebugCaptureError(f"Trigger result {name!r}.global_scope must be a bool.")
            if not world_ids and not global_scope:
                raise DebugCaptureError(
                    f"Trigger result {name!r} must select at least one world or global_scope=True."
                )

            scopes = {
                _IncidentScope(world_id=world_id, is_global=False) for world_id in world_ids
            }
            if global_scope:
                scopes.add(_IncidentScope(world_id=None, is_global=True))
            normalized[name] = _NormalizedTriggerResult(
                reason=reason.strip(),
                scopes=frozenset(scopes),
            )
        return normalized

    def _update_active_trigger_scopes(
        self,
        results: Mapping[str, _NormalizedTriggerResult],
    ) -> dict[_IncidentScope, dict[str, str]]:
        new_reasons: dict[_IncidentScope, dict[str, str]] = {}
        for name in self._trigger_callbacks:
            result = results.get(name)
            current = set() if result is None else set(result.scopes)
            new_scopes = current.difference(self._active_trigger_scopes[name])
            if result is not None:
                for scope in new_scopes:
                    new_reasons.setdefault(scope, {})[name] = result.reason
            self._active_trigger_scopes[name] = current
        return new_reasons

    def _update_active_failures(self, bad_worlds: set[int], global_bad: bool) -> None:
        self._active_failed_worlds = set(bad_worlds)
        self._active_global_incident = bool(global_bad)

    def _detect_nonfinite(self) -> tuple[set[int], bool, dict[str, list[str]]]:
        if not self._nonfinite_fields:
            return set(), False, {}

        bad_worlds: set[int] = set()
        global_bad = False
        scope_paths: dict[str, list[str]] = {}
        partitions = self._build_live_partitions(self._incident_source, self._incident_binding)

        for field in self._nonfinite_fields:
            provider = field.path[0] if field.path and isinstance(field.path[0], str) else None
            if provider not in self._detect_nonfinite_in:
                continue
            value = field.validate(self._incident_source)
            mask = _nonfinite_mask(value)
            if mask is None or not _mask_any(mask):
                continue

            display_path = field.display_path.removeprefix("incident.")
            if not self._layout.is_multi or not field.shape:
                global_bad = True
                scope_paths.setdefault("global", []).append(display_path)
                continue

            partition = self._partition_for_field(field, partitions)
            if partition is not None:
                global_scope = _IncidentScope(world_id=None, is_global=True)
                global_indices = partition.indices(global_scope, failed_worlds_only=True)
                if global_indices.size and _mask_any(mask, global_indices):
                    global_bad = True
                    scope_paths.setdefault("global", []).append(display_path)
                for world_id in range(self._layout.world_count):
                    world_scope = _IncidentScope(world_id=world_id, is_global=False)
                    indices = partition.indices(world_scope, failed_worlds_only=True)
                    if indices.size and _mask_any(mask, indices):
                        bad_worlds.add(world_id)
                        scope_paths.setdefault(f"world:{world_id}", []).append(display_path)
                continue

            frequency = self._layout.frequency_for(field, composite_root=True)
            if frequency is None or frequency == "once":
                global_bad = True
                scope_paths.setdefault("global", []).append(display_path)
                continue
            if not self._layout.validate_extent(frequency, int(field.shape[0])):
                global_bad = True
                scope_paths.setdefault("global", []).append(display_path)
                continue

            global_indices = self._layout.global_indices(frequency)
            if global_indices is not None and global_indices.size and _mask_any(mask, global_indices):
                global_bad = True
                scope_paths.setdefault("global", []).append(display_path)

            for world_id in range(self._layout.world_count):
                indices = self._layout.local_indices(frequency, world_id, include_globals=False)
                if indices is not None and indices.size and _mask_any(mask, indices):
                    bad_worlds.add(world_id)
                    scope_paths.setdefault(f"world:{world_id}", []).append(display_path)

        return bad_worlds, global_bad, scope_paths

    def _write_incident_artifacts(
        self,
        sim_time: float,
        scopes: list[_IncidentScope],
        bad_paths: dict[str, list[str]],
        trigger_reasons: Mapping[_IncidentScope, Mapping[str, str]],
    ) -> list[dict[str, str]]:
        common_errors: list[dict[str, str]] = []
        auxiliary: dict[str, np.ndarray] = {}
        auxiliary_fields: dict[str, DebugCaptureField] = {}
        captured_by_path: dict[_CapturePath, np.ndarray] = {}
        row_partitions: tuple[_RowPartition, ...] = ()
        provider_metadata = _plan_metadata(self._incident_binding)
        provider_metadata["nonfinite_detection"] = {
            "providers": sorted(self._detect_nonfinite_in),
            "include_patterns": list(self._detect_nonfinite_include_fields),
            "exclude_patterns": list(self._detect_nonfinite_exclude_fields),
            "fields": [field.display_path for field in self._nonfinite_fields],
        }

        try:
            self._incident_plan.validate_schema(self._incident_source)
            self._incident_binding.validate(self._incident_source)
            for field in self._incident_binding.fields:
                value = debug_value_to_numpy(field.validate(self._incident_source), field.display_path)
                key = _field_key("incident", field, strip_root=True)
                auxiliary[key] = value
                auxiliary_fields[key] = field
                captured_by_path[field.path] = value

            row_partitions = self._build_partitions(
                self._incident_binding.fields,
                lambda field: captured_by_path[field.path],
                all_fields=self._incident_binding.plan.fields,
            )
            provider_metadata["partition_contracts"] = _partition_metadata(
                row_partitions,
                self._incident_binding,
                row_encoding="compact_rows",
            )
        except Exception as exc:
            common_errors.append(_capture_error("incident", exc))

        all_errors = list(common_errors)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        for scope in scopes:
            errors = list(common_errors)
            arrays = self._state_history_arrays(scope)
            arrays.update(self._pre_state_arrays(scope))
            for key, value in auxiliary.items():
                arrays[key] = self._slice_partitioned_array(
                    value,
                    auxiliary_fields[key],
                    scope,
                    row_partitions,
                )
            self._add_replay_arrays(arrays, scope)

            arrays["history_length"] = np.int64(self._history_length)
            arrays["history_valid_count"] = np.int64(self._valid_count)
            arrays["incident__sim_time"] = np.float64(sim_time)
            arrays["incident__is_global"] = np.int8(scope.is_global)
            if scope.world_id is not None and self._failed_worlds_only:
                arrays["failed_world_ids"] = np.asarray([scope.world_id], dtype=np.int32)
            if self._substep_meta is not None:
                arrays["incident__capture_per_substep"] = np.int8(1)
                arrays["incident__failed_substep_idx"] = np.int64(self._substep_meta[0])
                arrays["incident__last_finite_substep_idx"] = np.int64(self._substep_meta[1])
                arrays["incident__substep_counter"] = np.int64(self._substep_meta[2])

            scope_key = "global" if scope.is_global else f"world:{scope.world_id}"
            arrays["incident__nonfinite_paths"] = np.asarray(
                bad_paths.get(scope_key, ()),
                dtype=np.str_,
            )
            scope_trigger_reasons = dict(sorted(trigger_reasons.get(scope, {}).items()))
            arrays["incident__trigger_names"] = np.asarray(
                tuple(scope_trigger_reasons),
                dtype=np.str_,
            )

            stem = f"physics_incident_{timestamp}_{self._incident_count:04d}_{scope.label}"
            archive_path = Path(self._output_dir) / f"{stem}.npz"
            if self._scene_exporter is not None:
                usd_path = str(Path(self._output_dir) / f"{stem}.usd")
                world_ids = [] if scope.is_global or scope.world_id is None else [scope.world_id]
                try:
                    self._scene_exporter(usd_path, world_ids)
                except Exception as exc:
                    errors.append(_capture_error("scene", exc))
                    logger.exception("Scene export failed for incident scope %s.", scope.label)

            metadata = {
                "incident": {
                    "event_index": self._incident_count,
                    "scope": scope.label,
                    "world_id": scope.world_id,
                    "global": scope.is_global,
                    "sim_time_seconds": float(sim_time),
                    "trigger_reasons": scope_trigger_reasons,
                },
                "incomplete": bool(errors),
                "errors": errors,
                "providers": provider_metadata,
                "state_plan": _plan_metadata(self._state_binding),
                "replay_partition_contracts": self._replay_partition_metadata,
                "capabilities": _capabilities(arrays, errors),
            }
            write_archive(
                archive_path,
                arrays,
                status="partial" if errors else "complete",
                required_keys=("history_length", "history_valid_count", "incident__sim_time"),
                metadata=metadata,
            )
            logger.error(
                "Physics incident detected in %s; exported %d valid snapshots to %s.",
                scope.label,
                self._valid_count,
                archive_path,
            )
            for error in errors:
                if error not in all_errors:
                    all_errors.append(error)
        return all_errors

    def _make_incident_source(self) -> _IncidentSources:
        source = _IncidentSources(
            state=self._state,
            model=self._model if self._record_model else None,
            control=self._control if self._record_control else None,
            contacts=self._contacts if self._record_contacts else None,
            collision_pipeline=self._collision_pipeline if self._record_collision_pipeline else None,
            solver=self._solver if self._record_solver else None,
            context=self._context,
            operations=self._operations if self._record_operations else None,
        )
        for provider in (
            source.state,
            source.model,
            source.control,
            source.contacts,
            source.collision_pipeline,
            source.solver,
            source.context,
            source.operations,
        ):
            _register_provider_root_types(
                provider,
                include_private=self._include_private_fields,
            )
        return source

    def _bind_incident_plan(
        self,
        source: _IncidentSources,
    ) -> tuple[DebugCapturePlan, DebugCaptureBinding]:
        plan = DebugCapturePlan.build(
            source,
            root_name="incident",
            include_private=self._include_private_fields,
        )
        include = _root_patterns(self._include_fields, "incident")
        exclude = _root_patterns(self._exclude_fields, "incident")
        binding = plan.bind(include, exclude)
        self._layout.validate_fields(binding.fields, composite_root=True)
        _validate_archive_key_uniqueness(binding, "incident__")

        required = {
            "model": self._record_model,
            "control": self._record_control,
            "contacts": self._record_contacts,
            "collision_pipeline": self._record_collision_pipeline,
            "solver": self._record_solver,
            "context": bool(self._context),
            "operations": self._record_operations,
        }
        for provider, enabled in required.items():
            if enabled:
                _require_selected_provider(binding, provider)
        for provider in sorted(self._detect_nonfinite_in):
            _require_selected_provider(binding, provider)
        binding.validate(source)
        return plan, binding


    def _bind_nonfinite_fields(
        self,
        binding: DebugCaptureBinding,
    ) -> tuple[DebugCaptureField, ...]:
        """Freeze strict scan-only fields from the recorded incident binding."""
        if not self._detect_nonfinite_in:
            return ()
        candidates = tuple(
            field
            for field in binding.fields
            if field.path
            and isinstance(field.path[0], str)
            and field.path[0] in self._detect_nonfinite_in
        )
        include_patterns = _root_patterns(self._detect_nonfinite_include_fields, "incident")
        exclude_patterns = _root_patterns(self._detect_nonfinite_exclude_fields, "incident")
        candidate_paths = tuple(field.display_path for field in candidates)
        scannable = tuple(
            field
            for field in candidates
            if _nonfinite_mask(field.validate(self._incident_source)) is not None
        )
        scannable_paths = tuple(field.display_path for field in scannable)

        unmatched_includes = [
            pattern
            for pattern in include_patterns
            if not any(fnmatch.fnmatchcase(path, pattern) for path in candidate_paths)
        ]
        if unmatched_includes:
            raise DebugSchemaError(
                "Non-finite include pattern(s) matched no recorded scan candidates: "
                + ", ".join(repr(pattern) for pattern in unmatched_includes)
                + "."
            )
        non_scannable_includes = [
            pattern
            for pattern in include_patterns
            if not any(fnmatch.fnmatchcase(path, pattern) for path in scannable_paths)
        ]
        if non_scannable_includes:
            raise DebugSchemaError(
                "Non-finite include pattern(s) matched only non-scannable recorded fields: "
                + ", ".join(repr(pattern) for pattern in non_scannable_includes)
                + ". Non-finite detection supports only floating-point and complex values."
            )

        included = tuple(
            field
            for field in scannable
            if any(fnmatch.fnmatchcase(field.display_path, pattern) for pattern in include_patterns)
        )
        included_paths = tuple(field.display_path for field in included)
        unmatched_excludes = [
            pattern
            for pattern in exclude_patterns
            if not any(fnmatch.fnmatchcase(path, pattern) for path in included_paths)
        ]
        if unmatched_excludes:
            raise DebugSchemaError(
                "Non-finite exclude pattern(s) matched no included scan candidates: "
                + ", ".join(repr(pattern) for pattern in unmatched_excludes)
                + "."
            )

        selected = tuple(
            field
            for field in included
            if not any(fnmatch.fnmatchcase(field.display_path, pattern) for pattern in exclude_patterns)
        )
        for provider in sorted(self._detect_nonfinite_in):
            if not any(field.path[0] == provider for field in selected):
                raise DebugCaptureError(
                    "Non-finite field filters eliminated every recorded scan field "
                    f"for provider '{provider}'."
                )
        if not selected:
            raise DebugCaptureError("Non-finite field filters selected no recorded scan fields.")
        return selected

    def _state_history_arrays(self, scope: _IncidentScope) -> dict[str, np.ndarray]:
        indices = self._ordered_indices(self._write_idx, self._valid_count)
        arrays: dict[str, np.ndarray] = {
            "history_sim_time": self._ring_times[indices].copy(),
        }
        for field_index, field in enumerate(self._state_binding.fields):
            values = [
                self._slice_array(
                    self._ring[index].numpy(field_index),
                    field,
                    scope,
                    composite_root=False,
                )
                for index in indices
            ]
            suffix = _field_key("", field, strip_root=True)
            arrays[f"history__state__{suffix}"] = np.stack(values)
        return arrays

    def _pre_state_arrays(self, scope: _IncidentScope) -> dict[str, np.ndarray]:
        if not self._pending_pre_valid:
            return {}
        arrays: dict[str, np.ndarray] = {}
        for field_index, field in enumerate(self._state_binding.fields):
            value = self._slice_array(
                self._pending_pre.numpy(field_index),
                field,
                scope,
                composite_root=False,
            )
            arrays[f"pre__state__{_field_key('', field, strip_root=True)}"] = value
        return arrays

    def _build_live_partitions(
        self,
        source: object,
        binding: DebugCaptureBinding,
    ) -> tuple[_RowPartition, ...]:
        return self._build_partitions(
            binding.fields,
            lambda field: debug_value_to_numpy(field.validate(source), field.display_path),
            all_fields=binding.plan.fields,
        )

    def _build_slot_partitions(
        self,
        binding: DebugCaptureBinding,
        slot: _CaptureSlot,
    ) -> tuple[_RowPartition, ...]:
        field_indices = {field.path: index for index, field in enumerate(binding.fields)}
        return self._build_partitions(
            binding.fields,
            lambda field: slot.numpy(field_indices[field.path]),
            all_fields=binding.plan.fields,
        )

    def _build_partitions(
        self,
        fields: tuple[DebugCaptureField, ...],
        read: Callable[[DebugCaptureField], np.ndarray],
        *,
        all_fields: tuple[DebugCaptureField, ...] | None = None,
    ) -> tuple[_RowPartition, ...]:
        """Discover and validate semantic active-row ownership contracts."""
        if not self._failed_worlds_only or not self._layout.is_multi:
            return ()

        selected = {field.path: field for field in fields}
        complete = fields if all_fields is None else all_fields
        provider_names = {
            field.path[0]
            for field in fields
            if field.path and isinstance(field.path[0], str)
        }

        partitions: list[_RowPartition] = []
        explicit_contracts = {
            "contacts": (
                ("rigid_contact_count",),
                ("rigid_contact_shape0",),
                ("rigid_contact_shape1",),
                "newton_rigid_contacts",
            ),
            "collision_pipeline": (
                ("broad_phase_pair_count",),
                ("broad_phase_shape_pairs",),
                None,
                "newton_broad_phase_pairs",
            ),
        }
        partitioned = provider_names.intersection(explicit_contracts)
        if partitioned:
            shape_world_field = next(
                (field for field in self._model_plan.fields if field.path == ("shape_world",)),
                None,
            )
            if shape_world_field is None:
                raise DebugSchemaError(
                    "Pair partitioning requires the allocated model.shape_world anchor."
                )
            shape_world = debug_value_to_numpy(
                shape_world_field.validate(self._model),
                shape_world_field.display_path,
            ).reshape(-1)
            if not np.issubdtype(shape_world.dtype, np.integer):
                raise DebugSchemaError("model.shape_world must have an integer dtype.")
            shape_world = shape_world.astype(np.int64, copy=False)
            invalid_worlds = shape_world[
                (shape_world < -1) | (shape_world >= self._layout.world_count)
            ]
            if invalid_worlds.size:
                raise DebugSchemaError(
                    "model.shape_world contains IDs outside "
                    f"[-1, {self._layout.world_count}): {np.unique(invalid_worlds).tolist()}."
                )

            for provider in sorted(partitioned):
                count_suffix, shape0_suffix, shape1_suffix, contract = explicit_contracts[provider]
                count_path = (provider, *count_suffix)
                shape0_path = (provider, *shape0_suffix)
                required_paths = [count_path, shape0_path]
                if shape1_suffix is not None:
                    required_paths.append((provider, *shape1_suffix))
                missing = [path for path in required_paths if path not in selected]
                if missing:
                    labels = [_capture_path_label(path) for path in missing]
                    raise DebugSchemaError(
                        f"Provider '{provider}' selected active-row fields but is missing "
                        f"required partition anchors {labels} for contract '{contract}'."
                    )

                count_field = selected[count_path]
                shape0_field = selected[shape0_path]
                if shape1_suffix is None:
                    pairs = np.asarray(read(shape0_field))
                    if (
                        pairs.ndim != 2
                        or pairs.shape[1] != 2
                        or not np.issubdtype(pairs.dtype, np.integer)
                    ):
                        raise DebugSchemaError(
                            f"Broad-phase anchor '{shape0_field.display_path}' must be an integer "
                            "[capacity, 2] array."
                        )
                    shape0 = pairs[:, 0]
                    shape1 = pairs[:, 1]
                    capacity = int(pairs.shape[0])
                else:
                    shape1_path = (provider, *shape1_suffix)
                    shape1_field = selected[shape1_path]
                    shape0 = np.asarray(read(shape0_field)).reshape(-1)
                    shape1 = np.asarray(read(shape1_field)).reshape(-1)
                    if shape0.shape != shape1.shape:
                        raise DebugSchemaError(
                            f"Contact shape anchors for '{provider}' must have equal length."
                        )
                    capacity = int(shape0.size)

                count = _read_active_count(read(count_field), count_field, capacity)
                if not np.issubdtype(shape0.dtype, np.integer) or not np.issubdtype(
                    shape1.dtype, np.integer
                ):
                    raise DebugSchemaError(f"Pair shape anchors for '{provider}' must be integer arrays.")
                shape0 = shape0[:count].astype(np.int64, copy=False)
                shape1 = shape1[:count].astype(np.int64, copy=False)
                invalid = np.concatenate((shape0, shape1))
                invalid = invalid[(invalid < -1) | (invalid >= shape_world.size)]
                if invalid.size:
                    raise DebugSchemaError(
                        f"Pair provider '{provider}' contains shape ids outside model.shape_world: "
                        f"{np.unique(invalid).tolist()}."
                    )
                world0 = np.full(count, -1, dtype=np.int64)
                world1 = np.full(count, -1, dtype=np.int64)
                valid0 = shape0 >= 0
                valid1 = shape1 >= 0
                world0[valid0] = shape_world[shape0[valid0]]
                world1[valid1] = shape_world[shape1[valid1]]
                cross_world = (world0 >= 0) & (world1 >= 0) & (world0 != world1)
                if np.any(cross_world):
                    rows = np.flatnonzero(cross_world).tolist()
                    worlds = np.stack((world0[cross_world], world1[cross_world]), axis=1).tolist()
                    raise DebugSchemaError(
                        f"Pair provider '{provider}' contains cross-world rows {rows} "
                        f"with worlds {worlds}."
                    )
                if provider == "contacts":
                    row_paths = frozenset(
                        field.path
                        for field in fields
                        if len(field.path) == 2
                        and field.path[0] == provider
                        and isinstance(field.path[1], str)
                        and field.path[1].startswith("rigid_contact_")
                        and not field.path[1].startswith(("rigid_contact_new_", "rigid_contact_broken_"))
                        and not field.path[1].endswith("_count")
                        and bool(field.shape)
                        and field.shape[0] == capacity
                    )
                else:
                    row_paths = frozenset({shape0_path})
                partitions.append(
                    _RowPartition(
                        contract=contract,
                        provider=provider,
                        capacity=capacity,
                        count_path=count_path,
                        container_path=(provider,),
                        row_paths=row_paths,
                        world0=world0,
                        world1=world1,
                    )
                )

        symbolic_anchors = [
            field
            for field in complete
            if field.path
            and field.path[-1] == "worldid"
            and field.symbolic_shape
            and isinstance(field.symbolic_shape[0], str)
            and field.symbolic_shape[0] != "nworld"
        ]
        used_count_paths = {partition.count_path for partition in partitions}
        for anchor in symbolic_anchors:
            assert anchor.symbolic_shape is not None
            capacity_symbol = anchor.symbolic_shape[0]
            assert isinstance(capacity_symbol, str)
            container_path = anchor.path[:-1]
            complete_rows = [
                field
                for field in complete
                if field.path[:-1] == container_path
                and field.symbolic_shape
                and field.symbolic_shape[0] == capacity_symbol
            ]
            selected_rows = [field for field in complete_rows if field.path in selected]
            if not selected_rows:
                continue
            if anchor.path not in selected:
                raise DebugSchemaError(
                    f"Symbolic active-row group '{_capture_path_label(container_path)}' selected "
                    f"'{capacity_symbol}' rows without required world-id anchor "
                    f"'{anchor.display_path}'."
                )

            count_field = self._find_symbolic_active_count(anchor, capacity_symbol, complete)
            if count_field.path not in selected:
                raise DebugSchemaError(
                    f"Symbolic active-row group '{anchor.display_path}' requires selected count "
                    f"anchor '{count_field.display_path}'."
                )
            if count_field.path in used_count_paths:
                raise DebugSchemaError(
                    f"Active-row count '{count_field.display_path}' ambiguously owns more than "
                    "one row partition."
                )

            world_ids = np.asarray(read(selected[anchor.path]))
            if world_ids.ndim != 1 or not np.issubdtype(world_ids.dtype, np.integer):
                raise DebugSchemaError(
                    f"World-id anchor '{anchor.display_path}' must be a one-dimensional integer array."
                )
            capacity = int(world_ids.shape[0])
            count = _read_active_count(read(selected[count_field.path]), count_field, capacity)
            active_world_ids = world_ids[:count].astype(np.int64, copy=False)
            invalid = active_world_ids[
                (active_world_ids < -1) | (active_world_ids >= self._layout.world_count)
            ]
            if invalid.size:
                raise DebugSchemaError(
                    f"World-id anchor '{anchor.display_path}' contains IDs outside "
                    f"[-1, {self._layout.world_count}): {np.unique(invalid).tolist()}."
                )
            provider = anchor.path[0]
            assert isinstance(provider, str)
            partitions.append(
                _RowPartition(
                    contract=f"symbolic_world_rows:{capacity_symbol}",
                    provider=provider,
                    capacity=capacity,
                    count_path=count_field.path,
                    container_path=container_path,
                    row_paths=frozenset(field.path for field in selected_rows),
                    world0=active_world_ids,
                    world1=np.full(count, -1, dtype=np.int64),
                )
            )
            used_count_paths.add(count_field.path)

        return tuple(partitions)

    def _find_symbolic_active_count(
        self,
        anchor: DebugCaptureField,
        capacity_symbol: str,
        fields: tuple[DebugCaptureField, ...],
    ) -> DebugCaptureField:
        """Find one nearest active-count anchor for a symbolic row group."""
        container_path = anchor.path[:-1]
        container_name = container_path[-1] if container_path else ""
        candidate_names: list[str] = []
        if capacity_symbol.endswith("max") and len(capacity_symbol) > 3:
            candidate_names.append(capacity_symbol[:-3])
        if isinstance(container_name, str):
            candidate_names.extend((f"{container_name}_count", f"n{container_name}"))
        candidate_names.append("count")
        candidate_names = list(dict.fromkeys(candidate_names))

        for depth in range(len(container_path), 0, -1):
            ancestor = container_path[:depth]
            matches = [
                field
                for field in fields
                if field.path[:-1] == ancestor
                and field.path
                and field.path[-1] in candidate_names
            ]
            if len(matches) > 1:
                raise DebugSchemaError(
                    f"World-id anchor '{anchor.display_path}' has ambiguous active-count "
                    f"candidates {[field.display_path for field in matches]}."
                )
            if matches:
                return matches[0]
        raise DebugSchemaError(
            f"World-id anchor '{anchor.display_path}' declares symbolic capacity "
            f"'{capacity_symbol}' but no sibling or ancestor active-count anchor named "
            f"one of {candidate_names} was discovered."
        )

    @staticmethod
    def _partition_for_field(
        field: DebugCaptureField,
        partitions: tuple[_RowPartition, ...],
    ) -> _RowPartition | None:
        matches = [partition for partition in partitions if partition.applies(field)]
        if len(matches) > 1:
            raise DebugSchemaError(
                f"Capture field '{field.display_path}' is ambiguously owned by row "
                f"partition contracts {[partition.contract for partition in matches]}."
            )
        return matches[0] if matches else None

    def _slice_partitioned_array(
        self,
        array: np.ndarray,
        field: DebugCaptureField,
        scope: _IncidentScope,
        partitions: tuple[_RowPartition, ...],
    ) -> np.ndarray:
        """Slice semantic active rows, then discovered world-frequency rows."""
        partition = self._partition_for_field(field, partitions)
        if partition is not None:
            indices = partition.indices(
                scope,
                failed_worlds_only=self._failed_worlds_only and not scope.is_global,
            )
            if field.path == partition.count_path:
                return np.asarray([indices.size], dtype=array.dtype)
            if array.ndim > 0 and array.shape[0] == partition.capacity:
                return np.take(array[: partition.world0.size], indices, axis=0)
        return self._slice_array(array, field, scope, composite_root=True)

    def _slice_array(
        self,
        array: np.ndarray,
        field: DebugCaptureField,
        scope: _IncidentScope,
        *,
        composite_root: bool,
    ) -> np.ndarray:
        if not self._failed_worlds_only or scope.is_global or array.ndim == 0:
            return array.copy()

        frequency = self._layout.frequency_for(field, composite_root=composite_root)
        if frequency is None or frequency == "once" or not array.shape:
            return array.copy()
        if not self._layout.validate_extent(frequency, int(array.shape[0])):
            return array.copy()

        if scope.is_global:
            indices = self._layout.global_indices(frequency)
        else:
            assert scope.world_id is not None
            indices = self._layout.local_indices(frequency, scope.world_id, include_globals=True)
        if indices is None:
            return array.copy()
        return np.take(array, indices, axis=0)

    def _make_replay_source(self, state: State, *, refresh: bool = True) -> _ReplaySources:
        cfg = self._replay_cfg
        context = self._snapshot_context() if refresh and self._context_enabled else self._context
        if refresh and cfg.record_operations:
            self._operations = self._snapshot_operations()
            if self._record_operations and hasattr(self, "_incident_source"):
                self._incident_source.operations = self._operations
        source = _ReplaySources(
            state=state if cfg.record_state else None,
            control=self._control if cfg.record_control else None,
            contacts=self._contacts if cfg.record_contacts else None,
            collision_pipeline=self._collision_pipeline if cfg.record_collision_pipeline else None,
            solver=self._solver if cfg.record_solver else None,
            context=context,
            operations=self._operations if cfg.record_operations else None,
        )
        for provider in (
            source.state,
            source.control,
            source.contacts,
            source.collision_pipeline,
            source.solver,
            source.context,
            source.operations,
        ):
            _register_provider_root_types(
                provider,
                include_private=self._include_private_fields,
            )
        return source

    def _bind_replay_plan(self, source: _ReplaySources) -> None:
        plan = DebugCapturePlan.build(
            source,
            root_name="replay",
            include_private=self._include_private_fields,
        )
        include = _root_patterns(self._replay_include_fields, "replay")
        exclude = _root_patterns(self._replay_exclude_fields, "replay")
        binding = plan.bind(include, exclude)
        self._layout.validate_fields(binding.fields, composite_root=True)
        _validate_archive_key_uniqueness(binding, "replay__pre__")

        required = {
            "state": self._replay_cfg.record_state,
            "control": self._replay_cfg.record_control,
            "contacts": self._replay_cfg.record_contacts,
            "collision_pipeline": self._replay_cfg.record_collision_pipeline,
            "solver": self._replay_cfg.record_solver,
            "context": bool(self._context),
            "operations": self._replay_cfg.record_operations,
        }
        for provider, enabled in required.items():
            if enabled:
                _require_selected_provider(binding, provider)
        binding.validate(source)
        if self._readback_preflight:
            self._preflight_readback("replay", binding, source)
        partitions = self._build_live_partitions(source, binding)
        _validate_packed_replay_keys(binding, partitions)
        self._replay_partition_metadata = _partition_metadata(
            partitions,
            binding,
            row_encoding="concatenated_rows_with_slot_offsets",
        )

        bytes_per_slot, report = _binding_nbytes(binding, source)
        replay_bytes = bytes_per_slot * self._history_length * 2
        remaining_bytes = self._max_gpu_bytes - self._state_gpu_bytes
        if replay_bytes > remaining_bytes:
            details = ", ".join(f"{path}={size}" for path, size in report)
            raise DebugCaptureError(
                f"Replay requires {replay_bytes} GPU bytes for pre/post history, but only "
                f"remaining budget is {remaining_bytes} bytes within max_gpu_bytes={self._max_gpu_bytes}. Fields: {details}"
            )

        self._replay_plan = plan
        self._replay_binding = binding
        self._replay_source = source
        self._replay_pre = [_CaptureSlot.allocate(binding, source) for _ in range(self._history_length)]
        self._replay_post = [_CaptureSlot.allocate(binding, source) for _ in range(self._history_length)]
        self._replay_gpu_bytes = replay_bytes

    def _validate_replay_source(self, source: _ReplaySources) -> None:
        if self._replay_plan is None or self._replay_binding is None:
            raise DebugCaptureError("Replay plan was not bound during recorder initialization.")
        self._replay_plan.validate_schema(source)
        self._replay_binding.validate(source)

    def _add_replay_arrays(
        self,
        arrays: dict[str, np.ndarray],
        scope: _IncidentScope,
    ) -> None:
        if self._replay_valid_count == 0 or self._replay_binding is None:
            return
        indices = self._ordered_indices(self._replay_write_idx, self._replay_valid_count)
        arrays["replay__valid_count"] = np.int64(self._replay_valid_count)
        arrays["replay__sim_time"] = np.asarray(
            [self._replay_meta[index]["sim_time"] for index in indices],
            dtype=np.float64,
        )
        arrays["replay__substep_idx"] = np.asarray(
            [self._replay_meta[index]["substep_idx"] for index in indices],
            dtype=np.int64,
        )
        pre_partitions = {
            index: self._build_slot_partitions(self._replay_binding, self._replay_pre[index])
            for index in indices
        }
        post_partitions = {
            index: self._build_slot_partitions(self._replay_binding, self._replay_post[index])
            for index in indices
        }
        for field_index, field in enumerate(self._replay_binding.fields):
            pre_values = [
                self._slice_partitioned_array(
                    self._replay_pre[index].numpy(field_index),
                    field,
                    scope,
                    pre_partitions[index],
                )
                for index in indices
            ]
            post_values = [
                self._slice_partitioned_array(
                    self._replay_post[index].numpy(field_index),
                    field,
                    scope,
                    post_partitions[index],
                )
                for index in indices
            ]
            suffix = _field_key("", field, strip_root=True)
            pre_key = f"replay__pre__{suffix}"
            post_key = f"replay__post__{suffix}"
            pre_partition = self._partition_for_field(field, pre_partitions[indices[0]])
            post_partition = self._partition_for_field(field, post_partitions[indices[0]])
            if (pre_partition is None) != (post_partition is None):
                raise DebugSchemaError(
                    f"Replay row ownership changed between pre/post for '{field.display_path}'."
                )
            is_packed_rows = (
                pre_partition is not None and field.path != pre_partition.count_path
            )
            if is_packed_rows:
                arrays[pre_key] = np.concatenate(pre_values, axis=0)
                arrays[post_key] = np.concatenate(post_values, axis=0)
                arrays[f"{pre_key}__slot_offsets"] = _replay_row_offsets(pre_values)
                arrays[f"{post_key}__slot_offsets"] = _replay_row_offsets(post_values)
            else:
                arrays[pre_key] = np.stack(pre_values)
                arrays[post_key] = np.stack(post_values)

    def _ordered_indices(self, write_index: int, valid_count: int) -> list[int]:
        start = (write_index - valid_count) % self._history_length
        return [(start + offset) % self._history_length for offset in range(valid_count)]


def _copy_debug_value(destination: object, source: object, path: str) -> object:
    if isinstance(destination, wp.array) and isinstance(source, wp.array):
        wp.copy(destination, source)
        return destination
    if isinstance(destination, torch.Tensor) and isinstance(source, torch.Tensor):
        with torch.no_grad():
            destination.copy_(source)
        return destination
    if isinstance(destination, np.ndarray) and isinstance(source, np.ndarray):
        np.copyto(destination, source)
        return destination
    return clone_debug_value(source, path)


def _nonfinite_mask(value: object) -> torch.Tensor | np.ndarray | None:
    if isinstance(value, wp.array):
        tensor = wp.to_torch(value)
        if tensor.is_floating_point() or tensor.is_complex():
            return ~torch.isfinite(tensor)
        return None
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return ~torch.isfinite(value)
        return None
    array = debug_value_to_numpy(value)
    if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
        return ~np.isfinite(array)
    return None


def _mask_any(mask: torch.Tensor | np.ndarray, indices: np.ndarray | None = None) -> bool:
    if indices is not None:
        if isinstance(mask, torch.Tensor):
            device_indices = torch.as_tensor(indices, dtype=torch.long, device=mask.device)
            mask = torch.index_select(mask, 0, device_indices)
        else:
            mask = np.take(mask, indices, axis=0)
    if isinstance(mask, torch.Tensor):
        return bool(mask.any().item())
    return bool(np.any(mask))


def _frequency_name(frequency: object) -> str:
    name = getattr(frequency, "name", frequency)
    return str(name).lower()


def _register_root_type(value: object) -> None:
    register_debug_container_type(type(value))


def _register_newton_namespace_type(model: object) -> None:
    for base in type(model).__mro__:
        namespace_type = vars(base).get("AttributeNamespace")
        if isinstance(namespace_type, type):
            register_debug_container_type(namespace_type)
            return


def _validate_trigger_registry(
    triggers: Mapping[str, Callable[..., object | None]] | None,
) -> dict[str, Callable[..., object | None]]:
    """Validate and freeze registered incident trigger names."""
    if triggers is None:
        return {}
    if not isinstance(triggers, Mapping):
        raise TypeError("triggers must be a mapping or None.")
    invalid_types = [name for name in triggers if not isinstance(name, str)]
    if invalid_types:
        raise TypeError("Trigger names must be strings.")
    output: dict[str, Callable[..., object | None]] = {}
    for name in sorted(triggers):
        if _TRIGGER_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"Trigger name {name!r} must be lower snake case and start with a letter."
            )
        callback = triggers[name]
        if not callable(callback):
            raise TypeError(f"Trigger {name!r} must be callable.")
        output[name] = callback
    return output


def _validate_detect_nonfinite_in(providers: tuple[str, ...]) -> frozenset[str]:
    """Validate provider roots selected for non-finite detection."""
    if not isinstance(providers, tuple):
        raise TypeError("detect_nonfinite_in must be a tuple of provider names.")
    if any(not isinstance(provider, str) or not provider for provider in providers):
        raise TypeError("detect_nonfinite_in must contain non-empty strings.")
    if len(set(providers)) != len(providers):
        raise ValueError("detect_nonfinite_in must not contain duplicates.")
    unknown = sorted(set(providers).difference(_DETECT_NONFINITE_PROVIDERS))
    if unknown:
        raise ValueError(f"detect_nonfinite_in contains unsupported providers: {unknown}.")
    return frozenset(providers)


def _register_same_package_child_types(root: object | None, *, include_private: bool) -> None:
    """Register direct physics child containers while preserving opaque runtimes."""
    if root is None:
        return
    root_package = type(root).__module__.partition(".")[0]
    try:
        members = vars(root)
    except TypeError:
        return
    for name, value in members.items():
        if name.startswith("_") and not include_private:
            continue
        if value is None or isinstance(value, type) or callable(value):
            continue
        if type(value).__module__.partition(".")[0] != root_package:
            continue
        if dataclasses.is_dataclass(value) or hasattr(value, "__dict__") or hasattr(type(value), "__slots__"):
            register_debug_container_type(type(value))


def _register_provider_root_types(
    root: object | None,
    *,
    include_private: bool,
    seen: set[int] | None = None,
) -> None:
    """Register provider roots, including every value in nested mappings."""
    if root is None:
        return
    if seen is None:
        seen = set()
    identity = id(root)
    if identity in seen:
        return
    seen.add(identity)

    _register_root_type(root)
    if isinstance(root, Mapping):
        for value in root.values():
            _register_provider_root_types(
                value,
                include_private=include_private,
                seen=seen,
            )
        return
    _register_same_package_child_types(root, include_private=include_private)


def _validate_archive_key_uniqueness(binding: DebugCaptureBinding, prefix: str) -> None:
    """Fail when two selected source paths normalize to one archive key."""
    sources_by_key: dict[str, str] = {}
    for field in binding.fields:
        key = prefix + _field_key("", field, strip_root=True)
        previous = sources_by_key.get(key)
        if previous is not None:
            raise DebugSchemaError(
                f"Archive key collision for '{key}' between source paths "
                f"'{previous}' and '{field.display_path}'. Adjust field names or selection patterns."
            )
        sources_by_key[key] = field.display_path


def _require_selected_provider(binding: DebugCaptureBinding, provider: str) -> None:
    """Require one configured provider to contribute selected fields."""
    available = [
        field for field in binding.plan.fields if field.path and field.path[0] == provider
    ]
    selected = [
        field for field in binding.fields if field.path and field.path[0] == provider
    ]
    if not available:
        raise DebugCaptureError(f"Required provider '{provider}' has no capturable fields.")
    if not selected:
        raise DebugCaptureError(
            f"Required provider '{provider}' has no selected capturable fields; "
            "adjust its record flag or field patterns."
        )



def _read_active_count(
    value: np.ndarray,
    field: DebugCaptureField,
    capacity: int,
) -> int:
    """Read and validate a scalar active-row count."""
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise DebugSchemaError(
            f"Active-row count anchor '{field.display_path}' must contain one integer."
        )
    count = int(array[0])
    if not 0 <= count <= capacity:
        raise DebugSchemaError(
            f"Active-row count {count} at '{field.display_path}' is outside capacity {capacity}."
        )
    return count


def _capture_path_label(path: _CapturePath) -> str:
    """Format one relative capture path for contract diagnostics."""
    return ".".join(_safe_key_component(step) for step in path)


def _partition_metadata(
    partitions: tuple[_RowPartition, ...],
    binding: DebugCaptureBinding,
    *,
    row_encoding: str,
) -> list[dict[str, object]]:
    """Describe every active-row contract embedded in an artifact."""
    display_paths = {field.path: field.display_path for field in binding.fields}
    output: list[dict[str, object]] = []
    for partition in partitions:
        row_paths = [
            display_paths[path]
            for path in sorted(partition.row_paths, key=lambda item: tuple(map(str, item)))
        ]
        output.append(
            {
                "contract": partition.contract,
                "provider": partition.provider,
                "capacity": partition.capacity,
                "count_path": display_paths[partition.count_path],
                "container_path": _capture_path_label(partition.container_path),
                "row_paths": row_paths,
                "row_encoding": row_encoding,
            }
        )
    return output



def _validate_packed_replay_keys(
    binding: DebugCaptureBinding,
    partitions: tuple[_RowPartition, ...],
) -> None:
    """Reject source keys that collide with packed-row offset metadata."""
    base_keys = {
        f"replay__{phase}__{_field_key('', field, strip_root=True)}"
        for phase in ("pre", "post")
        for field in binding.fields
    }
    offset_keys: set[str] = set()
    for field in binding.fields:
        owners = [partition for partition in partitions if partition.applies(field)]
        if len(owners) > 1:
            raise DebugSchemaError(
                f"Replay field '{field.display_path}' has ambiguous row partition ownership."
            )
        if not owners or field.path == owners[0].count_path:
            continue
        suffix = _field_key("", field, strip_root=True)
        for phase in ("pre", "post"):
            key = f"replay__{phase}__{suffix}__slot_offsets"
            if key in base_keys or key in offset_keys:
                raise DebugSchemaError(
                    f"Packed replay offset key '{key}' collides with another archive key."
                )
            offset_keys.add(key)


def _replay_row_offsets(values: list[np.ndarray]) -> np.ndarray:
    """Return prefix offsets for chronologically concatenated replay rows."""
    lengths = np.asarray([value.shape[0] for value in values], dtype=np.int64)
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))


def _binding_nbytes(
    binding: DebugCaptureBinding,
    root: object,
) -> tuple[int, list[tuple[str, int]]]:
    report: list[tuple[str, int]] = []
    total = 0
    for field in binding.fields:
        size = int(debug_value_to_numpy(field.validate(root), field.display_path).nbytes)
        report.append((field.display_path, size))
        total += size
    return total, report


def _root_patterns(patterns: tuple[str, ...], root_name: str) -> tuple[str, ...]:
    output: list[str] = []
    prefix = f"{root_name}."
    for pattern in patterns:
        if pattern == "*" or pattern == root_name or pattern.startswith(prefix):
            output.append(pattern)
        else:
            output.append(f"{prefix}{pattern}")
    return tuple(output)


def _validate_pattern_tuple(
    patterns: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(patterns, tuple):
        raise DebugSchemaError(f"{name} must be a tuple of glob strings.")
    if not patterns and not allow_empty:
        raise DebugSchemaError(f"{name} must not be empty.")
    if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        raise DebugSchemaError(f"{name} must contain only non-empty strings.")
    if len(set(patterns)) != len(patterns):
        raise DebugSchemaError(f"{name} must not contain duplicate patterns.")
    return patterns


def _field_key(prefix: str, field: DebugCaptureField, *, strip_root: bool) -> str:
    path = field.path
    components = [_safe_key_component(step) for step in path]
    body = "__".join(components) or "value"
    return f"{prefix}__{body}" if prefix else body


def _safe_key_component(step: str | int) -> str:
    if isinstance(step, DebugMappingKey):
        text = f"key_{step.key}"
    else:
        text = f"item_{step}" if isinstance(step, int) else step
    return "".join(character if character.isalnum() or character == "_" else "_" for character in text)


def _plan_metadata(binding: DebugCaptureBinding) -> dict[str, Any]:
    plan = binding.plan
    return {
        "root_type": plan.root_type,
        "plan_schema_fingerprint": plan.schema_fingerprint,
        "selected_schema_fingerprint": binding.schema_fingerprint,
        "include_patterns": list(binding.include_patterns),
        "exclude_patterns": list(binding.exclude_patterns),
        "selected_fields": [field.display_path for field in binding.fields],
        "unallocated": [_inventory_entry(entry) for entry in plan.unallocated],
        "ignored": [_inventory_entry(entry) for entry in plan.ignored],
    }


def _inventory_entry(entry) -> dict[str, Any]:
    return {
        "path": [str(step) for step in entry.path],
        "display_path": entry.display_path,
        "value_type": entry.value_type,
        "annotation": entry.annotation,
        "symbolic_shape": None if entry.symbolic_shape is None else list(entry.symbolic_shape),
        "reason": entry.reason,
    }


def _capture_error(provider: str, error: Exception) -> dict[str, str]:
    return {
        "provider": provider,
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def _capabilities(
    arrays: Mapping[str, np.ndarray],
    errors: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    failed = {error["provider"] for error in errors}
    state_fields = sorted(
        key for key in arrays if key == "history_sim_time" or key.startswith("history__") or key.startswith("pre__")
    )
    capabilities: dict[str, dict[str, Any]] = {
        "state_history": {
            "stage": "state",
            "status": "complete",
            "provider": "isaaclab_newton.physics_incident_recorder",
            "fields": state_fields,
            "adapter": "isaaclab_newton.state_history.v1",
        }
    }
    incident_fields = sorted(key for key in arrays if key.startswith("incident__"))
    if incident_fields:
        capabilities["incident_snapshot"] = {
            "stage": "solver",
            "status": "partial" if failed else "complete",
            "provider": "isaaclab_newton.schema_capture",
            "fields": incident_fields,
            "adapter": "isaaclab_newton.incident_snapshot.v1",
            **({"reason": "; ".join(sorted(failed))} if failed else {}),
        }
    replay_fields = sorted(key for key in arrays if key.startswith("replay__"))
    if replay_fields:
        capabilities["transition_history"] = {
            "stage": "transition",
            "status": "complete",
            "provider": "isaaclab_newton.physics_incident_recorder",
            "fields": replay_fields,
            "adapter": "isaaclab_newton.transition_history.v1",
        }
    return capabilities
