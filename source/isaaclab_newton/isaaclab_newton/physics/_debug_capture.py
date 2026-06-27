# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict schema discovery and value capture for Newton debug utilities.

This module intentionally contains no solver-specific field lists. A capture plan
is built once by recursively inspecting declared dataclass fields, instance
dictionaries, slots, and list/tuple elements. The resulting plan contains only
precompiled path getters and immutable schema descriptors, so repeated captures
do not need to rediscover the object graph.
"""

from __future__ import annotations

import dataclasses
import enum
import fnmatch
import functools
import hashlib
import inspect
import json
import types
from collections.abc import Callable, Mapping

import numpy as np
import torch
import warp as wp

@dataclasses.dataclass(frozen=True, slots=True)
class DebugMappingKey:
    """Explicit deterministic key step in a captured mapping path."""

    key: str | int


type DebugPathStep = str | int | DebugMappingKey
type DebugPath = tuple[DebugPathStep, ...]
type SymbolicShape = tuple[str | int, ...]


_REGISTERED_CONTAINER_TYPES: set[type] = set()


def register_debug_container_type(container_type: type) -> None:
    """Register an instance type whose public fields discovery may traverse.

    Root objects, dataclasses, and declarative slot types are traversed without
    registration. Framework objects with a normal ``__dict__`` must be
    registered explicitly so opaque resources such as Warp devices, meshes, and
    acceleration structures remain inventory leaves.

    Args:
        container_type: Concrete container class to register.

    Raises:
        TypeError: If ``container_type`` is not a class.
    """
    if not isinstance(container_type, type):
        raise TypeError("Debug capture container_type must be a class.")
    _REGISTERED_CONTAINER_TYPES.add(container_type)


class DebugCaptureError(RuntimeError):
    """Raised when a value cannot be cloned or converted for debug capture."""


class DebugSchemaError(DebugCaptureError):
    """Raised when a live object no longer matches its frozen capture plan."""


@dataclasses.dataclass(frozen=True, slots=True)
class DebugInventoryEntry:
    """Description of an object path that is not part of the captured values.

    Attributes:
        path: Relative attribute/index path from the plan root.
        display_path: Human-readable path including the root name.
        value_type: Fully qualified runtime type name, when a value exists.
        annotation: Stable description of the declared annotation, when known.
        symbolic_shape: Symbolic dimensions extracted from ``annotation.shape``.
        reason: Explanation of why the path is unallocated or ignored.
    """

    path: DebugPath
    display_path: str
    value_type: str | None
    annotation: str | None
    symbolic_shape: SymbolicShape | None
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class _CompiledPathGetter:
    """Callable attribute/index traversal compiled into a capture plan."""

    path: DebugPath

    def __call__(self, root: object) -> object:
        value = root
        for step in self.path:
            if isinstance(step, DebugMappingKey):
                value = value[step.key]  # type: ignore[index]
            elif isinstance(step, int):
                value = value[step]  # type: ignore[index]
            else:
                value = object.__getattribute__(value, step)
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class _ValueSchema:
    """Normalized schema of one capturable value."""

    kind: str
    dtype: str
    shape: tuple[int, ...]
    device: str | None
    sequence_signature: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DebugCaptureField:
    """Frozen descriptor for one value in a debug capture plan.

    Attributes:
        path: Relative attribute/index path from the plan root.
        display_path: Human-readable path including the root name.
        kind: Value category: Warp, Torch, NumPy, scalar, enum, or sequence.
        dtype: Stable logical dtype identifier.
        shape: Exact runtime shape expected for the value.
        device: Exact device expected for device-backed arrays.
        annotation: Stable description of the declared annotation, when known.
        symbolic_shape: Symbolic dimensions extracted from ``annotation.shape``.
        sequence_signature: Exact list/tuple and scalar-type topology for a sequence.
    """

    path: DebugPath
    display_path: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    device: str | None
    annotation: str | None
    symbolic_shape: SymbolicShape | None
    sequence_signature: str | None
    _getter: Callable[[object], object] = dataclasses.field(repr=False, compare=False)

    def get(self, root: object) -> object:
        """Read this field from ``root`` using its precompiled getter.

        Args:
            root: Live root object that the plan describes.

        Returns:
            The current value at :attr:`path`.

        Raises:
            DebugSchemaError: If the path is no longer readable.
        """
        try:
            return self._getter(root)
        except Exception as exc:
            raise DebugSchemaError(f"Debug capture path '{self.display_path}' is no longer readable: {exc}") from exc

    def validate(self, root: object) -> object:
        """Read and exactly validate this field against its frozen schema.

        Args:
            root: Live root object that the plan describes.

        Returns:
            The validated live value.

        Raises:
            DebugSchemaError: If kind, dtype, shape, device, or sequence topology changed.
        """
        value = self.get(root)
        actual = _describe_value(value, self.display_path)
        expected = _ValueSchema(
            kind=self.kind,
            dtype=self.dtype,
            shape=self.shape,
            device=self.device,
            sequence_signature=self.sequence_signature,
        )
        if actual != expected:
            raise DebugSchemaError(
                f"Debug capture schema changed at '{self.display_path}': expected {_format_schema(expected)}, "
                f"got {_format_schema(actual)}. Rebuild the capture plan for the new runtime schema."
            )
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class DebugCapturedValue:
    """A cloned value paired with the field that described it."""

    field: DebugCaptureField
    value: object = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True, slots=True)
class DebugCaptureSnapshot:
    """Immutable index of values cloned from one capture plan invocation."""

    schema_fingerprint: str
    values: tuple[DebugCapturedValue, ...]

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert all cloned values to independent non-object NumPy arrays.

        Returns:
            Mapping from human-readable field path to an independent NumPy array.

        Raises:
            DebugCaptureError: If a value cannot be converted without using an object dtype.
        """
        return {
            captured.field.display_path: debug_value_to_numpy(captured.value, captured.field.display_path)
            for captured in self.values
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DebugCapturePlan:
    """Frozen recursive plan for strict repeated capture of one object schema.

    Use :meth:`build` after all debug sources have been allocated. Call
    :meth:`validate_schema` when binding or when an upstream package may have
    changed, then use :meth:`clone` or :meth:`to_numpy` for repeated captures.

    Attributes:
        root_name: Display name prepended to every captured path.
        root_type: Fully qualified runtime type of the planned root.
        fields: Capturable values with precompiled path getters.
        unallocated: Declared fields whose value was absent or ``None``.
        ignored: Private or unsupported fields deliberately excluded from capture.
        schema_fingerprint: SHA-256 digest of the complete field and inventory schema.
        include_private: Whether private attributes were included during discovery.
    """

    root_name: str
    root_type: str
    fields: tuple[DebugCaptureField, ...]
    unallocated: tuple[DebugInventoryEntry, ...]
    ignored: tuple[DebugInventoryEntry, ...]
    schema_fingerprint: str
    include_private: bool

    @classmethod
    def build(
        cls,
        root: object,
        *,
        root_name: str = "root",
        include_private: bool = False,
    ) -> DebugCapturePlan:
        """Discover and freeze the capturable schema below ``root``.

        Discovery reads dataclass fields, ``vars()`` mappings, declared slots,
        and list/tuple elements. It never calls :func:`dir`, so dynamic and
        deprecated properties are not probed as a side effect.

        Args:
            root: Root object to inspect.
            root_name: Human-readable name used in paths and error messages.
            include_private: Whether attributes beginning with ``_`` are traversed.

        Returns:
            A frozen capture plan containing immutable descriptors and inventory.

        Raises:
            DebugSchemaError: If the root name is invalid or a discovered NumPy array uses object dtype.
        """
        if not isinstance(root_name, str) or not root_name:
            raise DebugSchemaError("Debug capture root_name must be a non-empty string.")

        builder = _PlanBuilder(root_name=root_name, include_private=include_private)
        builder.walk(root, (), None, is_root=True)
        fields = tuple(sorted(builder.fields, key=lambda field: _path_sort_key(field.path)))
        unallocated = tuple(sorted(builder.unallocated, key=lambda entry: _path_sort_key(entry.path)))
        ignored = tuple(sorted(builder.ignored, key=lambda entry: _path_sort_key(entry.path)))
        root_type = _qualified_type_name(type(root))
        fingerprint = _schema_fingerprint(root_type, fields, unallocated, ignored)
        return cls(
            root_name=root_name,
            root_type=root_type,
            fields=fields,
            unallocated=unallocated,
            ignored=ignored,
            schema_fingerprint=fingerprint,
            include_private=include_private,
        )

    def validate(self, root: object) -> None:
        """Validate every planned path without rediscovering the object graph.

        This is the inexpensive validation suitable for repeated capture. It
        detects missing or changed planned values but intentionally does not scan
        for newly added fields; use :meth:`validate_schema` for that check.

        Args:
            root: Live root object to validate.

        Raises:
            DebugSchemaError: If the root type or a planned value changed.
        """
        actual_root_type = _qualified_type_name(type(root))
        errors: list[str] = []
        if actual_root_type != self.root_type:
            errors.append(f"{self.root_name}: expected root type {self.root_type}, got {actual_root_type}")
        for field in self.fields:
            try:
                field.validate(root)
            except DebugSchemaError as exc:
                errors.append(str(exc))
        if errors:
            details = "\n  - ".join(errors)
            raise DebugSchemaError(f"Debug capture plan validation failed:\n  - {details}")

    def validate_schema(self, root: object) -> None:
        """Rediscover ``root`` and compare its complete schema fingerprint.

        Args:
            root: Live root object to rediscover.

        Raises:
            DebugSchemaError: If captured, unallocated, or ignored inventory changed.
        """
        actual = type(self).build(root, root_name=self.root_name, include_private=self.include_private)
        if actual.schema_fingerprint == self.schema_fingerprint:
            return
        differences = _schema_differences(self, actual)
        detail = "\n  - ".join(differences or ["schema fingerprint changed for an unknown reason"])
        raise DebugSchemaError(
            "Debug capture object graph no longer matches its frozen plan "
            f"({self.schema_fingerprint} != {actual.schema_fingerprint}):\n  - {detail}"
        )

    def bind(
        self,
        include_patterns: tuple[str, ...] = ("*",),
        exclude_patterns: tuple[str, ...] = (),
    ) -> DebugCaptureBinding:
        """Bind strict field-selection patterns to this frozen plan.

        Patterns use :func:`fnmatch.fnmatchcase` against complete
        :attr:`DebugCaptureField.display_path` values. Every pattern must match
        at least one applicable field, so an upstream rename cannot silently
        degrade a recording.

        Args:
            include_patterns: Patterns selecting fields from the full plan.
            exclude_patterns: Patterns removing fields from the included set.

        Returns:
            An immutable binding with its own selected-schema fingerprint.

        Raises:
            DebugSchemaError: If a pattern is invalid or unmatched, or if no
                fields remain selected.
        """
        includes = _validate_patterns(include_patterns, "include")
        excludes = _validate_patterns(exclude_patterns, "exclude", allow_empty=True)
        paths = tuple(field.display_path for field in self.fields)

        unmatched_includes = [
            pattern for pattern in includes if not any(fnmatch.fnmatchcase(path, pattern) for path in paths)
        ]
        if unmatched_includes:
            raise DebugSchemaError(
                f"Debug capture include pattern(s) matched no fields in plan '{self.root_name}': "
                f"{', '.join(repr(pattern) for pattern in unmatched_includes)}."
            )

        included = tuple(
            field
            for field in self.fields
            if any(fnmatch.fnmatchcase(field.display_path, pattern) for pattern in includes)
        )
        included_paths = tuple(field.display_path for field in included)
        unmatched_excludes = [
            pattern for pattern in excludes if not any(fnmatch.fnmatchcase(path, pattern) for path in included_paths)
        ]
        if unmatched_excludes:
            raise DebugSchemaError(
                f"Debug capture exclude pattern(s) matched no included fields in plan '{self.root_name}': "
                f"{', '.join(repr(pattern) for pattern in unmatched_excludes)}."
            )

        selected = tuple(
            field
            for field in included
            if not any(fnmatch.fnmatchcase(field.display_path, pattern) for pattern in excludes)
        )
        if not selected:
            raise DebugSchemaError(f"Debug capture selection for plan '{self.root_name}' contains no fields.")

        return DebugCaptureBinding(
            plan=self,
            fields=selected,
            include_patterns=includes,
            exclude_patterns=excludes,
            schema_fingerprint=_schema_fingerprint(self.root_type, selected, (), ()),
        )

    def clone(self, root: object, *, validate: bool = True) -> DebugCaptureSnapshot:
        """Clone every planned value without truncation.

        Args:
            root: Live root object to capture.
            validate: Whether to validate each planned value before cloning.

        Returns:
            Snapshot containing one independent clone for every planned field.

        Raises:
            DebugCaptureError: If any path cannot be read, validated, or cloned.
        """
        captured: list[DebugCapturedValue] = []
        for field in self.fields:
            value = field.validate(root) if validate else field.get(root)
            clone = clone_debug_value(value, field.display_path)
            captured.append(DebugCapturedValue(field=field, value=clone))
        return DebugCaptureSnapshot(schema_fingerprint=self.schema_fingerprint, values=tuple(captured))

    def to_numpy(self, root: object, *, validate: bool = True) -> dict[str, np.ndarray]:
        """Copy all live planned values directly to non-object NumPy arrays.

        Args:
            root: Live root object to capture.
            validate: Whether to validate each planned value before conversion.

        Returns:
            Mapping from human-readable field path to an independent NumPy array.

        Raises:
            DebugCaptureError: If any path cannot be read, validated, or converted.
        """
        output: dict[str, np.ndarray] = {}
        for field in self.fields:
            value = field.validate(root) if validate else field.get(root)
            output[field.display_path] = debug_value_to_numpy(value, field.display_path)
        return output


@dataclasses.dataclass(frozen=True, slots=True)
class DebugCaptureBinding:
    """Immutable selection of fields from a complete capture plan.

    The parent :attr:`plan` retains the full captured, unallocated, and ignored
    inventory. :attr:`schema_fingerprint` identifies only the selected field
    schemas, so archives can record both provider provenance and payload schema.

    Attributes:
        plan: Complete frozen discovery plan.
        fields: Selected fields in stable path order.
        include_patterns: Validated include patterns.
        exclude_patterns: Validated exclude patterns.
        schema_fingerprint: SHA-256 digest of selected field schemas.
    """

    plan: DebugCapturePlan
    fields: tuple[DebugCaptureField, ...]
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    schema_fingerprint: str

    def validate(self, root: object) -> None:
        """Validate selected fields against a live root.

        Args:
            root: Live root object described by :attr:`plan`.

        Raises:
            DebugSchemaError: If the root type or a selected field changed.
        """
        actual_root_type = _qualified_type_name(type(root))
        errors: list[str] = []
        if actual_root_type != self.plan.root_type:
            errors.append(
                f"{self.plan.root_name}: expected root type {self.plan.root_type}, got {actual_root_type}"
            )
        for field in self.fields:
            try:
                field.validate(root)
            except DebugSchemaError as exc:
                errors.append(str(exc))
        if errors:
            details = "\n  - ".join(errors)
            raise DebugSchemaError(f"Debug capture binding validation failed:\n  - {details}")

    def clone(self, root: object, *, validate: bool = True) -> DebugCaptureSnapshot:
        """Clone every selected value without truncation.

        Args:
            root: Live root object described by :attr:`plan`.
            validate: Whether to validate selected values before cloning.

        Returns:
            Snapshot containing one clone per selected field.

        Raises:
            DebugCaptureError: If a value cannot be read, validated, or cloned.
        """
        captured: list[DebugCapturedValue] = []
        for field in self.fields:
            value = field.validate(root) if validate else field.get(root)
            captured.append(
                DebugCapturedValue(field=field, value=clone_debug_value(value, field.display_path))
            )
        return DebugCaptureSnapshot(schema_fingerprint=self.schema_fingerprint, values=tuple(captured))

    def to_numpy(self, root: object, *, validate: bool = True) -> dict[str, np.ndarray]:
        """Copy every selected live value to a non-object NumPy array.

        Args:
            root: Live root object described by :attr:`plan`.
            validate: Whether to validate selected values before conversion.

        Returns:
            Mapping from stable display path to independent NumPy array.

        Raises:
            DebugCaptureError: If a value cannot be converted exactly.
        """
        output: dict[str, np.ndarray] = {}
        for field in self.fields:
            value = field.validate(root) if validate else field.get(root)
            output[field.display_path] = debug_value_to_numpy(value, field.display_path)
        return output


class _PlanBuilder:
    """Mutable implementation detail used only while constructing a frozen plan."""

    def __init__(self, root_name: str, include_private: bool) -> None:
        self.root_name = root_name
        self.include_private = include_private
        self.fields: list[DebugCaptureField] = []
        self.unallocated: list[DebugInventoryEntry] = []
        self.ignored: list[DebugInventoryEntry] = []
        self._seen_containers: dict[int, DebugPath] = {}
        self._preferred_container_paths: dict[int, DebugPath] = {}

    def walk(
        self,
        value: object,
        path: DebugPath,
        annotation: object | None,
        *,
        is_root: bool = False,
    ) -> None:
        """Recursively discover ``value`` at ``path``."""
        display_path = _format_path(self.root_name, path)
        symbolic_shape = extract_symbolic_shape(annotation)

        if value is None:
            self.unallocated.append(
                DebugInventoryEntry(
                    path=path,
                    display_path=display_path,
                    value_type=None,
                    annotation=_annotation_name(annotation),
                    symbolic_shape=symbolic_shape,
                    reason="value is unallocated (None)",
                )
            )
            return

        schema = _try_describe_value(value, display_path)
        if schema is not None:
            self.fields.append(
                DebugCaptureField(
                    path=path,
                    display_path=display_path,
                    kind=schema.kind,
                    dtype=schema.dtype,
                    shape=schema.shape,
                    device=schema.device,
                    annotation=_annotation_name(annotation),
                    symbolic_shape=symbolic_shape,
                    sequence_signature=schema.sequence_signature,
                    _getter=_CompiledPathGetter(path),
                )
            )
            return

        if isinstance(value, (types.ModuleType, type)) or callable(value):
            self._ignore(value, path, annotation, "callables, modules, and classes are not capturable")
            return
        if isinstance(value, Mapping):
            invalid_keys = [
                key
                for key in value
                if isinstance(key, bool) or not isinstance(key, (str, int))
            ]
            if invalid_keys:
                invalid_types = sorted({_qualified_type_name(type(key)) for key in invalid_keys})
                raise DebugSchemaError(
                    f"Debug capture mapping at '{display_path}' has unsupported key types "
                    f"{invalid_types}; only deterministic str and int keys are supported."
                )
            if not self._enter_container(value, path, annotation):
                return
            keys = sorted(value, key=lambda key: (0 if isinstance(key, str) else 1, str(key)))
            for key in keys:
                self.walk(value[key], (*path, DebugMappingKey(key)), None)
            return
        if isinstance(value, (set, frozenset, range, bytes, bytearray, memoryview)):
            self._ignore(value, path, annotation, "unordered or binary containers are not capturable")
            return

        if isinstance(value, (list, tuple)):
            if not self._enter_container(value, path, annotation):
                return
            for index, item in enumerate(value):
                self.walk(item, (*path, index), None)
            return

        if _is_opaque_runtime_resource(value):
            self._ignore(value, path, annotation, "opaque runtime resource")
            return
        if not is_root and not _is_traversable_container(value):
            self._ignore(value, path, annotation, "opaque unregistered object")
            return

        members = _object_members(value)
        if members is None:
            self._ignore(value, path, annotation, "unsupported leaf value")
            return
        if not members:
            self._ignore(value, path, annotation, "object has no discoverable instance fields")
            return
        if is_root and dataclasses.is_dataclass(value) and not isinstance(value, type):
            self._reserve_top_level_container_paths(members, path)
        if not self._enter_container(value, path, annotation):
            return

        for name, member, member_annotation, allocated in members:
            member_path = (*path, name)
            if name.startswith("_") and not self.include_private:
                self._ignore(member if allocated else None, member_path, member_annotation, "private attribute")
            elif not allocated:
                self.unallocated.append(
                    DebugInventoryEntry(
                        path=member_path,
                        display_path=_format_path(self.root_name, member_path),
                        value_type=None,
                        annotation=_annotation_name(member_annotation),
                        symbolic_shape=extract_symbolic_shape(member_annotation),
                        reason="declared attribute or slot is not initialized",
                    )
                )
            else:
                self.walk(member, member_path, member_annotation)

        represented_names = {name for name, *_ in members}
        for name, descriptor, descriptor_annotation in _property_declarations(type(value)):
            if name in represented_names:
                continue
            descriptor_path = (*path, name)
            reason = (
                "private attribute"
                if name.startswith("_") and not self.include_private
                else "property descriptor is inventoried but not invoked"
            )
            self._ignore(descriptor, descriptor_path, descriptor_annotation, reason)

    def _reserve_top_level_container_paths(
        self,
        members: list[tuple[str, object, object | None, bool]],
        root_path: DebugPath,
    ) -> None:
        """Reserve direct composite providers before sorted recursive traversal."""
        for name, member, _annotation, allocated in members:
            if not allocated or member is None:
                continue
            if _try_describe_value(member, _format_path(self.root_name, (*root_path, name))) is not None:
                continue
            if isinstance(member, (types.ModuleType, type)) or callable(member):
                continue
            self._preferred_container_paths.setdefault(id(member), (*root_path, name))

    def _enter_container(self, value: object, path: DebugPath, annotation: object | None) -> bool:
        preferred_path = self._preferred_container_paths.get(id(value))
        if preferred_path is not None and path != preferred_path:
            self._ignore(
                value,
                path,
                annotation,
                f"container identity is reserved for top-level provider path '{_format_path(self.root_name, preferred_path)}'",
            )
            return False
        first_path = self._seen_containers.get(id(value))
        if first_path is not None:
            self._ignore(
                value,
                path,
                annotation,
                f"container aliases or cycles previously visited path '{_format_path(self.root_name, first_path)}'",
            )
            return False
        self._seen_containers[id(value)] = path
        return True

    def _ignore(self, value: object, path: DebugPath, annotation: object | None, reason: str) -> None:
        self.ignored.append(
            DebugInventoryEntry(
                path=path,
                display_path=_format_path(self.root_name, path),
                value_type=None if value is None else _qualified_type_name(type(value)),
                annotation=_annotation_name(annotation),
                symbolic_shape=extract_symbolic_shape(annotation),
                reason=reason,
            )
        )


def extract_symbolic_shape(annotation: object | None) -> SymbolicShape | None:
    """Extract symbolic dimensions stored on a MuJoCo-Warp array annotation.

    MuJoCo-Warp's ``types.array`` wrapper stores dimensions such as ``"nworld"``
    and ``"nv"`` on the annotation object's ``shape`` attribute. This function
    intentionally does not resolve type hints or import private MJWarp modules.

    Args:
        annotation: Declared field annotation, if available.

    Returns:
        Symbolic dimensions, or ``None`` when the annotation has no usable metadata.
    """
    if annotation is None or isinstance(annotation, str):
        return None
    try:
        shape = object.__getattribute__(annotation, "shape")
    except (AttributeError, TypeError):
        return None
    if not isinstance(shape, (tuple, list)) or not shape:
        return None
    if not all(isinstance(dimension, (str, int)) and not isinstance(dimension, bool) for dimension in shape):
        return None
    return tuple(shape)


def clone_debug_value(value: object, path: str = "root") -> object:
    """Clone one supported debug value without truncation or dtype conversion.

    Args:
        value: Warp, Torch, or NumPy array, scalar, enum, or scalar sequence.
        path: Human-readable path used in an error message.

    Returns:
        An independent array clone or recursively cloned scalar sequence.

    Raises:
        DebugCaptureError: If ``value`` is unsupported or cloning fails.
    """
    try:
        if _is_warp_array(value):
            output = wp.empty_like(value)
            wp.copy(output, value)
            return output
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise DebugCaptureError(f"Debug capture path '{path}' uses a forbidden NumPy object dtype.")
            return value.copy(order="K")
        if isinstance(value, enum.Enum):
            return value
        if _is_scalar(value):
            return value
        if isinstance(value, list):
            return [_clone_sequence_item(item, path) for item in value]
        if isinstance(value, tuple):
            return tuple(_clone_sequence_item(item, path) for item in value)
    except DebugCaptureError:
        raise
    except Exception as exc:
        raise DebugCaptureError(f"Failed to clone debug capture path '{path}': {exc}") from exc
    raise DebugCaptureError(f"Debug capture path '{path}' has unsupported type {_qualified_type_name(type(value))}.")


def debug_value_to_numpy(value: object, path: str = "root") -> np.ndarray:
    """Convert one supported value to an independent non-object NumPy array.

    The conversion never slices, truncates, or silently changes an unsupported
    dtype. Torch and Warp device arrays are copied in full to the host.

    Args:
        value: Warp, Torch, or NumPy array, scalar, enum, or scalar sequence.
        path: Human-readable path used in an error message.

    Returns:
        Independent NumPy representation of the complete value.

    Raises:
        DebugCaptureError: If conversion fails or would produce an object array.
    """
    try:
        if _is_warp_array(value):
            output = np.array(value.numpy(), copy=True)
        elif isinstance(value, torch.Tensor):
            output = value.detach().cpu().numpy().copy()
        elif isinstance(value, np.ndarray):
            output = value.copy(order="K")
        elif isinstance(value, enum.Enum):
            output = np.asarray(value.value)
        elif _is_scalar(value) or isinstance(value, (list, tuple)):
            output = np.asarray(value)
        else:
            raise DebugCaptureError(
                f"Debug capture path '{path}' has unsupported type {_qualified_type_name(type(value))}."
            )
    except DebugCaptureError:
        raise
    except Exception as exc:
        raise DebugCaptureError(f"Failed to convert debug capture path '{path}' to NumPy: {exc}") from exc

    if output.dtype.hasobject:
        raise DebugCaptureError(
            f"Debug capture path '{path}' converted to forbidden object dtype; use numeric or string data only."
        )
    return output


def _try_describe_value(value: object, path: str) -> _ValueSchema | None:
    """Return a schema for a supported leaf, or ``None`` for a container/unsupported leaf."""
    if _is_warp_array(value):
        return _ValueSchema(
            kind="warp_array",
            dtype=_dtype_name(value.dtype),
            shape=_array_shape(value, path),
            device=str(value.device),
        )
    if isinstance(value, torch.Tensor):
        return _ValueSchema(
            kind="torch_tensor",
            dtype=str(value.dtype),
            shape=tuple(int(dimension) for dimension in value.shape),
            device=str(value.device),
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise DebugSchemaError(f"Debug capture path '{path}' uses a forbidden NumPy object dtype.")
        return _ValueSchema(
            kind="numpy_array",
            dtype=_numpy_dtype_key(value.dtype),
            shape=tuple(int(dimension) for dimension in value.shape),
            device=None,
        )
    if isinstance(value, enum.Enum):
        enum_value = np.asarray(value.value)
        if enum_value.dtype.hasobject:
            raise DebugSchemaError(f"Debug capture enum at '{path}' has a non-serializable value.")
        return _ValueSchema(
            kind="enum",
            dtype=_qualified_type_name(type(value)),
            shape=(),
            device=None,
        )
    if _is_scalar(value):
        return _describe_scalar(value, path)
    if isinstance(value, (list, tuple)) and _is_scalar_sequence(value):
        sequence = _sequence_to_numpy(value, path)
        logical_dtype = "str" if _sequence_is_string(value) else _numpy_dtype_key(sequence.dtype)
        return _ValueSchema(
            kind="list_sequence" if isinstance(value, list) else "tuple_sequence",
            dtype=logical_dtype,
            shape=tuple(int(dimension) for dimension in sequence.shape),
            device=None,
            sequence_signature=_sequence_signature(value),
        )
    return None


def _describe_value(value: object, path: str) -> _ValueSchema:
    schema = _try_describe_value(value, path)
    if schema is None:
        raise DebugSchemaError(
            f"Debug capture path '{path}' changed to unsupported type {_qualified_type_name(type(value))}."
        )
    return schema


def _describe_scalar(value: object, path: str) -> _ValueSchema:
    if isinstance(value, np.generic):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise DebugSchemaError(f"Debug capture scalar at '{path}' has a forbidden NumPy object dtype.")
        dtype = f"numpy_scalar:{_numpy_dtype_key(array.dtype)}"
    else:
        dtype = f"python_scalar:{_qualified_type_name(type(value))}"
    return _ValueSchema(kind="scalar", dtype=dtype, shape=(), device=None)


def _is_warp_array(value: object) -> bool:
    return isinstance(value, wp.array)


def _is_scalar(value: object) -> bool:
    return isinstance(value, (bool, int, float, complex, str, np.generic))


def _is_scalar_sequence(value: list | tuple, active: set[int] | None = None) -> bool:
    if active is None:
        active = set()
    if id(value) in active:
        return False
    active.add(id(value))
    try:
        for item in value:
            if isinstance(item, (list, tuple)):
                if not _is_scalar_sequence(item, active):
                    return False
            elif isinstance(item, enum.Enum) or not _is_scalar(item):
                return False
        return True
    finally:
        active.remove(id(value))


def _sequence_is_string(value: list | tuple) -> bool:
    leaves: list[object] = []

    def collect(sequence: list | tuple) -> None:
        for item in sequence:
            if isinstance(item, (list, tuple)):
                collect(item)
            else:
                leaves.append(item)

    collect(value)
    return bool(leaves) and all(isinstance(item, (str, np.str_)) for item in leaves)


def _sequence_to_numpy(value: list | tuple, path: str) -> np.ndarray:
    try:
        output = np.asarray(value)
    except Exception as exc:
        raise DebugSchemaError(f"Scalar sequence at '{path}' is ragged or cannot be represented exactly: {exc}") from exc
    if output.dtype.hasobject:
        raise DebugSchemaError(
            f"Scalar sequence at '{path}' would require an object dtype; use a rectangular numeric or string sequence."
        )
    return output


def _sequence_signature(value: list | tuple) -> str:
    container = "list" if isinstance(value, list) else "tuple"
    if not value:
        return f"{container}[]"
    child_signatures: set[str] = set()
    for item in value:
        if isinstance(item, (list, tuple)):
            child_signatures.add(_sequence_signature(item))
        elif isinstance(item, np.generic):
            child_signatures.add(f"numpy.{item.dtype}")
        else:
            child_signatures.add(_qualified_type_name(type(item)))
    return f"{container}[{'|'.join(sorted(child_signatures))}]"


def _clone_sequence_item(value: object, path: str) -> object:
    if isinstance(value, list):
        return [_clone_sequence_item(item, path) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_sequence_item(item, path) for item in value)
    if _is_scalar(value):
        return value
    raise DebugCaptureError(
        f"Debug capture scalar sequence at '{path}' contains unsupported type {_qualified_type_name(type(value))}."
    )


def _is_opaque_runtime_resource(value: object) -> bool:
    """Return whether a non-array value belongs to Warp's opaque runtime."""
    module = type(value).__module__
    return module == "warp" or module.startswith("warp.")


def _is_traversable_container(value: object) -> bool:
    """Return whether recursive discovery is explicitly allowed for a value."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return True
    if _REGISTERED_CONTAINER_TYPES and isinstance(value, tuple(_REGISTERED_CONTAINER_TYPES)):
        return True
    return any("__slots__" in vars(base) for base in type(value).__mro__)

def _object_members(value: object) -> list[tuple[str, object, object | None, bool]] | None:
    """Return instance fields without invoking dynamic attribute discovery."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        members: list[tuple[str, object, object | None, bool]] = []
        for field in sorted(dataclasses.fields(value), key=lambda item: item.name):
            try:
                member = object.__getattribute__(value, field.name)
            except AttributeError:
                members.append((field.name, None, field.type, False))
            else:
                members.append((field.name, member, field.type, True))
        return members

    annotations = _class_annotations(type(value))
    try:
        instance_vars = vars(value)
    except TypeError:
        instance_vars = {}

    discovered: dict[str, tuple[object, object | None, bool]] = {
        name: (member, annotations.get(name), True) for name, member in instance_vars.items()
    }
    has_instance_storage = bool(instance_vars) or hasattr(type(value), "__slots__") or hasattr(value, "__dict__")
    for name, annotation in _slot_declarations(type(value)):
        if name in discovered:
            continue
        try:
            member = object.__getattribute__(value, name)
        except AttributeError:
            discovered[name] = (None, annotation, False)
        else:
            discovered[name] = (member, annotation, True)

    for name, annotation in annotations.items():
        if name in discovered:
            continue
        try:
            member = inspect.getattr_static(value, name)
        except AttributeError:
            discovered[name] = (None, annotation, False)
            continue
        if isinstance(member, (property, functools.cached_property)):
            continue
        discovered[name] = (member, annotation, True)

    if not has_instance_storage:
        return None
    return [(name, *discovered[name]) for name in sorted(discovered)]


def _class_annotations(cls: type) -> dict[str, object]:
    annotations: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        raw = vars(base).get("__annotations__", {})
        if isinstance(raw, Mapping):
            annotations.update(raw)
    return annotations


def _property_declarations(cls: type) -> list[tuple[str, object, object | None]]:
    """Return safely inspectable property descriptors without invoking them."""
    declarations: dict[str, tuple[object, object | None]] = {}
    for base in reversed(cls.__mro__):
        for name, descriptor in vars(base).items():
            getter = None
            if isinstance(descriptor, property):
                getter = descriptor.fget
            elif isinstance(descriptor, functools.cached_property):
                getter = descriptor.func
            else:
                continue
            raw_annotations = getattr(getter, "__annotations__", {})
            try:
                getter_annotations = inspect.get_annotations(getter, eval_str=True)
            except (NameError, TypeError, ValueError):
                getter_annotations = raw_annotations
            annotation = getter_annotations.get("return") if isinstance(getter_annotations, Mapping) else None
            declarations[name] = (descriptor, annotation)
    return [(name, *declarations[name]) for name in sorted(declarations)]


def _slot_declarations(cls: type) -> list[tuple[str, object | None]]:
    declarations: dict[str, object | None] = {}
    for base in reversed(cls.__mro__):
        base_vars = vars(base)
        slots = base_vars.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        annotations = base_vars.get("__annotations__", {})
        for raw_name in slots:
            if raw_name in ("__dict__", "__weakref__"):
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                name = f"_{base.__name__.lstrip('_')}{raw_name}"
            annotation = annotations.get(raw_name) if isinstance(annotations, Mapping) else None
            declarations[name] = annotation
    return sorted(declarations.items())


def _array_shape(value: object, path: str) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in value.shape)  # type: ignore[union-attr]
    except Exception as exc:
        raise DebugSchemaError(f"Warp array at '{path}' has an unreadable shape: {exc}") from exc


def _numpy_dtype_key(dtype: np.dtype) -> str:
    if dtype.fields:
        return json.dumps(dtype.descr, separators=(",", ":"))
    return dtype.str


def _dtype_name(dtype: object) -> str:
    if isinstance(dtype, type):
        return _qualified_type_name(dtype)
    return str(dtype)


def _qualified_type_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _annotation_name(annotation: object | None) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return _qualified_type_name(annotation)
    return _qualified_type_name(type(annotation))


def _validate_patterns(
    patterns: tuple[str, ...],
    kind: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Validate immutable field-selection patterns."""
    if not isinstance(patterns, tuple):
        raise DebugSchemaError(f"Debug capture {kind} patterns must be a tuple of strings.")
    if not patterns and not allow_empty:
        raise DebugSchemaError(f"Debug capture {kind} patterns must not be empty.")
    invalid = [pattern for pattern in patterns if not isinstance(pattern, str) or not pattern]
    if invalid:
        raise DebugSchemaError(f"Debug capture {kind} patterns must contain only non-empty strings.")
    if len(set(patterns)) != len(patterns):
        raise DebugSchemaError(f"Debug capture {kind} patterns must not contain duplicates.")
    return patterns

def _format_path(root_name: str, path: DebugPath) -> str:
    output = root_name
    for step in path:
        if isinstance(step, DebugMappingKey):
            output += f"[{step.key!r}]"
        elif isinstance(step, int):
            output += f"[{step}]"
        else:
            output += f".{step}"
    return output


def _path_sort_key(path: DebugPath) -> tuple[str, ...]:
    output: list[str] = []
    for step in path:
        if isinstance(step, DebugMappingKey):
            key_type = "s" if isinstance(step.key, str) else "i"
            output.append(f"m:{key_type}:{step.key}")
        elif isinstance(step, int):
            output.append(f"i:{step:020d}")
        else:
            output.append(f"a:{step}")
    return tuple(output)


def _path_payload(path: DebugPath) -> list[object]:
    return [
        {"mapping_key": step.key} if isinstance(step, DebugMappingKey) else step
        for step in path
    ]


def _format_schema(schema: _ValueSchema) -> str:
    result = f"kind={schema.kind}, dtype={schema.dtype}, shape={schema.shape}"
    if schema.device is not None:
        result += f", device={schema.device}"
    if schema.sequence_signature is not None:
        result += f", sequence={schema.sequence_signature}"
    return result


def _schema_fingerprint(
    root_type: str,
    fields: tuple[DebugCaptureField, ...],
    unallocated: tuple[DebugInventoryEntry, ...],
    ignored: tuple[DebugInventoryEntry, ...],
) -> str:
    def field_payload(field: DebugCaptureField) -> dict[str, object]:
        return {
            "path": _path_payload(field.path),
            "kind": field.kind,
            "dtype": field.dtype,
            "shape": field.shape,
            "device": field.device,
            "annotation": field.annotation,
            "symbolic_shape": field.symbolic_shape,
            "sequence_signature": field.sequence_signature,
        }

    def inventory_payload(status: str, entry: DebugInventoryEntry) -> dict[str, object]:
        return {
            "status": status,
            "path": _path_payload(entry.path),
            "value_type": entry.value_type,
            "annotation": entry.annotation,
            "symbolic_shape": entry.symbolic_shape,
            "reason": entry.reason,
        }

    payload = {
        "root_type": root_type,
        "fields": [field_payload(field) for field in fields],
        "inventory": [inventory_payload("unallocated", entry) for entry in unallocated]
        + [inventory_payload("ignored", entry) for entry in ignored],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _schema_differences(expected: DebugCapturePlan, actual: DebugCapturePlan) -> list[str]:
    differences: list[str] = []
    if expected.root_type != actual.root_type:
        differences.append(f"root type changed from {expected.root_type} to {actual.root_type}")

    expected_fields = {field.path: field for field in expected.fields}
    actual_fields = {field.path: field for field in actual.fields}
    for path in sorted(expected_fields.keys() | actual_fields.keys(), key=_path_sort_key):
        before = expected_fields.get(path)
        after = actual_fields.get(path)
        display_path = _format_path(expected.root_name, path)
        if before is None:
            differences.append(f"new captured field '{display_path}'")
        elif after is None:
            differences.append(f"captured field '{display_path}' is missing or no longer capturable")
        else:
            before_schema = (before.kind, before.dtype, before.shape, before.device, before.sequence_signature)
            after_schema = (after.kind, after.dtype, after.shape, after.device, after.sequence_signature)
            if before_schema != after_schema:
                differences.append(
                    f"field '{display_path}' changed from {_format_schema(_field_value_schema(before))} "
                    f"to {_format_schema(_field_value_schema(after))}"
                )
            if before.symbolic_shape != after.symbolic_shape:
                differences.append(
                    f"field '{display_path}' symbolic shape changed from {before.symbolic_shape} "
                    f"to {after.symbolic_shape}"
                )

    expected_inventory = {
        entry.path: ("unallocated", entry.reason, entry.value_type) for entry in expected.unallocated
    } | {entry.path: ("ignored", entry.reason, entry.value_type) for entry in expected.ignored}
    actual_inventory = {entry.path: ("unallocated", entry.reason, entry.value_type) for entry in actual.unallocated} | {
        entry.path: ("ignored", entry.reason, entry.value_type) for entry in actual.ignored
    }
    for path in sorted(expected_inventory.keys() | actual_inventory.keys(), key=_path_sort_key):
        before = expected_inventory.get(path)
        after = actual_inventory.get(path)
        if before != after:
            differences.append(
                f"inventory at '{_format_path(expected.root_name, path)}' changed from {before} to {after}"
            )
    return differences


def _field_value_schema(field: DebugCaptureField) -> _ValueSchema:
    return _ValueSchema(
        kind=field.kind,
        dtype=field.dtype,
        shape=field.shape,
        device=field.device,
        sequence_signature=field.sequence_signature,
    )
