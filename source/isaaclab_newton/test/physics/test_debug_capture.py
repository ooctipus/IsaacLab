# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict automatic discovery and capture of debug values."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_newton.physics import _debug_capture as capture


@dataclasses.dataclass
class _NestedState:
    positions: np.ndarray
    labels: tuple[str, ...]
    optional: np.ndarray | None = None


@dataclasses.dataclass
class _DebugRoot:
    nested: _NestedState
    timestep: int
    callback: Callable[[], None]
    _private_cache: np.ndarray


class _SlottedState:
    __slots__ = ("allocated", "missing")
    __annotations__ = {"allocated": np.ndarray, "missing": np.ndarray}

    def __init__(self) -> None:
        self.allocated = np.arange(3, dtype=np.float32)


class _ExplosiveProperty:
    def __init__(self) -> None:
        self.values = np.arange(4, dtype=np.float32)

    @property
    def deprecated_dynamic_property(self):
        raise AssertionError("automatic discovery must not invoke properties")


class _ExplosiveCachedProperty:
    def __init__(self) -> None:
        self.values = np.arange(4, dtype=np.float32)
        self.calls = 0

    @functools.cached_property
    def lazy_values(self) -> np.ndarray:
        self.calls += 1
        raise AssertionError("automatic discovery must not invoke cached properties")


class _AnnotatedOrdinary:
    """Ordinary class with static, missing, and descriptor-backed annotations."""

    static_values: np.ndarray = np.arange(3, dtype=np.float32)
    missing_values: np.ndarray
    dynamic_values: np.ndarray

    def __init__(self) -> None:
        self.property_calls = 0

    @property
    def dynamic_values(self) -> np.ndarray:
        self.property_calls += 1
        raise AssertionError("static discovery must not invoke annotated descriptors")


class _OpaqueChild:
    def __init__(self) -> None:
        self.values = np.arange(4, dtype=np.float32)


class _RegisteredContainer:
    def __init__(self, values: np.ndarray, child: _RegisteredContainer | None = None) -> None:
        self.values = values
        self.child = child


class _FakeWarpResource:
    __module__ = "warp.fake_runtime"

    def __init__(self) -> None:
        self.values = np.arange(4, dtype=np.float32)


class _FakeWarpArray:
    """Small Warp-like array used to test cloning without a GPU allocation."""

    def __init__(self, values: np.ndarray, *, device: str = "cuda:7") -> None:
        self.values = np.array(values, copy=True)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.device = device

    def numpy(self) -> np.ndarray:
        return self.values.copy()


def _debug_root() -> _DebugRoot:
    return _DebugRoot(
        nested=_NestedState(
            positions=np.arange(12, dtype=np.float32).reshape(3, 4),
            labels=("left", "right"),
        ),
        timestep=7,
        callback=lambda: None,
        _private_cache=np.array([99], dtype=np.int64),
    )


def _install_fake_warp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "_is_warp_array", lambda value: isinstance(value, _FakeWarpArray))
    monkeypatch.setattr(
        capture.wp,
        "empty_like",
        lambda value: _FakeWarpArray(np.empty_like(value.values), device=value.device),
    )

    def copy(destination: _FakeWarpArray, source: _FakeWarpArray) -> None:
        destination.values[...] = source.values

    monkeypatch.setattr(capture.wp, "copy", copy)


def test_capture_plan_auto_discovers_supported_values_and_inventories_every_omission():
    """Discovery captures supported leaves and explicitly inventories everything else."""
    root = _debug_root()

    plan = capture.DebugCapturePlan.build(root, root_name="state")

    assert [field.display_path for field in plan.fields] == [
        "state.nested.labels",
        "state.nested.positions",
        "state.timestep",
    ]
    assert [entry.display_path for entry in plan.unallocated] == ["state.nested.optional"]
    ignored = {entry.display_path: entry.reason for entry in plan.ignored}
    assert "callables" in ignored["state.callback"]
    assert ignored["state._private_cache"] == "private attribute"
    assert len(plan.schema_fingerprint) == 64

    arrays = plan.to_numpy(root)
    np.testing.assert_array_equal(arrays["state.nested.positions"], root.nested.positions)
    np.testing.assert_array_equal(arrays["state.nested.labels"], np.asarray(root.nested.labels))
    assert arrays["state.timestep"].item() == 7
    assert all(not value.dtype.hasobject for value in arrays.values())


def test_capture_plan_include_private_discovers_private_arrays():
    """Private fields are captured only when the caller opts into them."""
    root = _debug_root()

    plan = capture.DebugCapturePlan.build(root, root_name="state", include_private=True)

    assert "state._private_cache" in {field.display_path for field in plan.fields}


def test_capture_binding_applies_full_path_include_and_exclude_patterns():
    """A binding captures exactly the fields selected by strict full-path globs."""
    root = _debug_root()
    plan = capture.DebugCapturePlan.build(root, root_name="state")

    binding = plan.bind(
        include_patterns=("state.nested.*", "state.timestep"),
        exclude_patterns=("*.labels",),
    )

    assert binding.plan is plan
    assert binding.include_patterns == ("state.nested.*", "state.timestep")
    assert binding.exclude_patterns == ("*.labels",)
    assert [field.display_path for field in binding.fields] == [
        "state.nested.positions",
        "state.timestep",
    ]
    assert binding.schema_fingerprint != plan.schema_fingerprint
    arrays = binding.to_numpy(root)
    assert list(arrays) == ["state.nested.positions", "state.timestep"]


@pytest.mark.parametrize(
    ("include_patterns", "exclude_patterns", "expected"),
    [
        (("nested.*",), (), "include pattern.*matched no fields"),
        (("state.nested.positions",), ("state.timestep",), "exclude pattern.*matched no included fields"),
    ],
)
def test_capture_binding_rejects_every_unmatched_pattern(include_patterns, exclude_patterns, expected):
    """A stale include or exclude cannot silently degrade the selected recording."""
    plan = capture.DebugCapturePlan.build(_debug_root(), root_name="state")

    with pytest.raises(capture.DebugSchemaError, match=expected):
        plan.bind(include_patterns=include_patterns, exclude_patterns=exclude_patterns)


def test_capture_binding_rejects_a_required_empty_selection():
    """Excluding every included field is an error rather than an empty successful capture."""
    plan = capture.DebugCapturePlan.build(_debug_root(), root_name="state")

    with pytest.raises(capture.DebugSchemaError, match="selection.*contains no fields"):
        plan.bind(
            include_patterns=("state.timestep",),
            exclude_patterns=("state.timestep",),
        )


def test_capture_binding_rejects_a_plan_with_no_capturable_fields():
    """The default required selection fails when automatic discovery found no values."""
    plan = capture.DebugCapturePlan.build(SimpleNamespace(callback=lambda: None), root_name="empty")

    with pytest.raises(capture.DebugSchemaError, match="include pattern.*matched no fields"):
        plan.bind()


def test_capture_plan_does_not_probe_dynamic_properties():
    """Properties are inventoried without invoking their getters or ``dir``."""
    plan = capture.DebugCapturePlan.build(_ExplosiveProperty(), root_name="source")

    assert [field.display_path for field in plan.fields] == ["source.values"]
    assert [(entry.display_path, entry.reason) for entry in plan.ignored] == [
        (
            "source.deprecated_dynamic_property",
            "property descriptor is inventoried but not invoked",
        )
    ]


def test_capture_plan_inventories_cached_properties_without_materializing_them():
    """Cached properties remain explicit omissions until application code resolves them."""
    root = _ExplosiveCachedProperty()

    plan = capture.DebugCapturePlan.build(root, root_name="source")

    assert root.calls == 0
    assert "lazy_values" not in vars(root)
    ignored = {entry.display_path: entry for entry in plan.ignored}
    assert ignored["source.lazy_values"].reason == "property descriptor is inventoried but not invoked"
    assert ignored["source.lazy_values"].annotation == "numpy.ndarray"


def test_ordinary_annotations_inventory_missing_and_static_members_without_descriptors():
    """Ordinary annotations are complete without invoking descriptor-backed names."""
    root = _AnnotatedOrdinary()

    plan = capture.DebugCapturePlan.build(root, root_name="source")

    assert root.property_calls == 0
    assert "dynamic_values" not in vars(root)
    assert {field.display_path for field in plan.fields} == {
        "source.property_calls",
        "source.static_values",
    }
    unallocated = {entry.display_path: entry for entry in plan.unallocated}
    assert unallocated["source.missing_values"].reason == "declared attribute or slot is not initialized"
    ignored = {entry.display_path: entry for entry in plan.ignored}
    assert ignored["source.dynamic_values"].reason == "property descriptor is inventoried but not invoked"


def test_mapping_discovery_is_deterministic_and_getters_preserve_key_types():
    """Mapping traversal is insertion-independent and distinguishes string and integer keys."""
    first = SimpleNamespace(
        values={
            2: np.array([2], dtype=np.int32),
            "zeta": np.array([3], dtype=np.int32),
            10: np.array([10], dtype=np.int32),
            "alpha": np.array([1], dtype=np.int32),
        }
    )
    second = SimpleNamespace(
        values={
            "alpha": np.array([11], dtype=np.int32),
            10: np.array([110], dtype=np.int32),
            "zeta": np.array([13], dtype=np.int32),
            2: np.array([12], dtype=np.int32),
        }
    )

    plan = capture.DebugCapturePlan.build(first, root_name="source")
    reordered_plan = capture.DebugCapturePlan.build(second, root_name="source")

    assert [field.display_path for field in plan.fields] == [
        "source.values[10]",
        "source.values[2]",
        "source.values['alpha']",
        "source.values['zeta']",
    ]
    assert plan.schema_fingerprint == reordered_plan.schema_fingerprint
    assert all(isinstance(field.path[-1], capture.DebugMappingKey) for field in plan.fields)
    arrays = plan.to_numpy(second)
    np.testing.assert_array_equal(arrays["source.values['alpha']"], [11])
    np.testing.assert_array_equal(arrays["source.values[10]"], [110])


def test_mapping_key_schema_drift_and_missing_getter_are_reported_exactly():
    """Renaming a mapping key invalidates both rediscovery and its compiled getter."""
    root = SimpleNamespace(values={"before": np.arange(2, dtype=np.float32)})
    plan = capture.DebugCapturePlan.build(root, root_name="source")
    field = plan.fields[0]
    root.values["after"] = root.values.pop("before")

    with pytest.raises(capture.DebugSchemaError, match=r"source\.values\['before'\].*no longer readable"):
        field.get(root)
    with pytest.raises(capture.DebugSchemaError) as exc_info:
        plan.validate_schema(root)
    message = str(exc_info.value)
    assert "captured field 'source.values['before']' is missing" in message
    assert "new captured field 'source.values['after']'" in message


@pytest.mark.parametrize("key", [True, 1.5, ("tuple",)])
def test_mapping_discovery_rejects_unsupported_key_types(key):
    """Mappings fail discovery when any key is not a deterministic string or integer."""
    with pytest.raises(
        capture.DebugSchemaError,
        match=r"mapping.*unsupported key types.*only deterministic str and int keys are supported",
    ):
        capture.DebugCapturePlan.build({key: np.ones(1, dtype=np.float32)}, root_name="mapping")


def test_registered_container_types_are_recursed_at_every_nested_level():
    """Explicit registration makes ordinary nested framework containers discoverable."""
    capture.register_debug_container_type(_RegisteredContainer)
    child = _RegisteredContainer(np.array([2], dtype=np.int32))
    root = SimpleNamespace(container=_RegisteredContainer(np.array([1], dtype=np.int32), child))

    plan = capture.DebugCapturePlan.build(root, root_name="source")

    assert [field.display_path for field in plan.fields] == [
        "source.container.child.values",
        "source.container.values",
    ]
    assert [entry.display_path for entry in plan.unallocated] == ["source.container.child.child"]


def test_unregistered_objects_remain_single_opaque_inventory_entries():
    """Discovery inventories an ordinary unregistered child without inspecting its internals."""
    root = SimpleNamespace(child=_OpaqueChild(), timestep=3)

    plan = capture.DebugCapturePlan.build(root, root_name="source")

    assert [field.display_path for field in plan.fields] == ["source.timestep"]
    assert [(entry.display_path, entry.reason) for entry in plan.ignored] == [
        ("source.child", "opaque unregistered object")
    ]


def test_warp_runtime_objects_remain_opaque_even_with_instance_storage():
    """Non-array ``warp.*`` resources are inventory boundaries, not recursive containers."""
    root = SimpleNamespace(resource=_FakeWarpResource(), timestep=3)

    plan = capture.DebugCapturePlan.build(root, root_name="source")

    assert [field.display_path for field in plan.fields] == ["source.timestep"]
    assert [(entry.display_path, entry.reason) for entry in plan.ignored] == [
        ("source.resource", "opaque runtime resource")
    ]


def test_validate_schema_detects_new_and_newly_allocated_fields():
    """Full validation catches both added fields and changes in allocation state."""
    root = SimpleNamespace(values=np.arange(2, dtype=np.float32), optional=None)
    plan = capture.DebugCapturePlan.build(root, root_name="source")
    root.added = np.ones(1, dtype=np.int32)

    with pytest.raises(capture.DebugSchemaError, match="new captured field 'source.added'"):
        plan.validate_schema(root)

    del root.added
    root.optional = np.ones(1, dtype=np.float32)
    with pytest.raises(capture.DebugSchemaError, match="inventory at 'source.optional' changed"):
        plan.validate_schema(root)


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        (np.arange(3, dtype=np.float32), "shape=\\(2,\\).*shape=\\(3,\\)"),
        (np.arange(2, dtype=np.float64), "dtype=<f4.*dtype=<f8"),
    ],
)
def test_capture_plan_rejects_shape_and_dtype_schema_drift(replacement, expected):
    """Repeated capture never accepts shape or dtype drift from a frozen plan."""
    root = SimpleNamespace(values=np.arange(2, dtype=np.float32))
    plan = capture.DebugCapturePlan.build(root, root_name="source")
    root.values = replacement

    with pytest.raises(capture.DebugSchemaError, match=expected):
        plan.clone(root)


def test_capture_plan_reports_a_missing_planned_allocation():
    """Deleting a planned value produces an actionable path error instead of a partial snapshot."""
    root = SimpleNamespace(values=np.arange(2, dtype=np.float32))
    plan = capture.DebugCapturePlan.build(root, root_name="source")
    del root.values

    with pytest.raises(capture.DebugSchemaError, match="source.values.*no longer readable"):
        plan.clone(root)


def test_capture_plan_inventories_uninitialized_slots_and_detects_late_allocation():
    """Declared but missing slots remain visible and cannot appear without rebinding."""
    root = _SlottedState()
    plan = capture.DebugCapturePlan.build(root, root_name="slots")

    assert [entry.display_path for entry in plan.unallocated] == ["slots.missing"]
    root.missing = np.ones(2, dtype=np.float32)
    with pytest.raises(capture.DebugSchemaError, match="inventory at 'slots.missing' changed"):
        plan.validate_schema(root)


def test_fake_warp_array_clone_and_numpy_conversion_preserve_every_element(monkeypatch):
    """Warp captures are independent, exact-shape copies with no capacity truncation."""
    _install_fake_warp(monkeypatch)
    values = np.arange(1028, dtype=np.float32).reshape(257, 4)
    live = _FakeWarpArray(values)
    root = SimpleNamespace(buffer=live)
    plan = capture.DebugCapturePlan.build(root, root_name="solver")

    snapshot = plan.clone(root)
    cloned = snapshot.values[0].value
    live.values.fill(-1)

    assert isinstance(cloned, _FakeWarpArray)
    assert cloned.shape == (257, 4)
    np.testing.assert_array_equal(cloned.values, values)
    converted = snapshot.to_numpy()["solver.buffer"]
    assert converted.shape == values.shape
    np.testing.assert_array_equal(converted, values)


def test_fake_warp_allocation_errors_include_the_capture_path(monkeypatch):
    """Allocation failures propagate as capture errors naming the exact field."""
    _install_fake_warp(monkeypatch)
    root = SimpleNamespace(buffer=_FakeWarpArray(np.arange(3, dtype=np.float32)))
    plan = capture.DebugCapturePlan.build(root, root_name="solver")

    def fail_allocation(value):
        raise MemoryError("allocation refused")

    monkeypatch.setattr(capture.wp, "empty_like", fail_allocation)
    with pytest.raises(capture.DebugCaptureError, match="solver.buffer.*allocation refused"):
        plan.clone(root)


def test_object_arrays_are_rejected_during_discovery_and_conversion():
    """Capture never emits NumPy object arrays that would require pickle."""
    root = SimpleNamespace(values=np.array([object()], dtype=object))

    with pytest.raises(capture.DebugSchemaError, match="forbidden NumPy object dtype"):
        capture.DebugCapturePlan.build(root, root_name="source")
    with pytest.raises(capture.DebugCaptureError, match="forbidden object dtype"):
        capture.debug_value_to_numpy([1, object()], "source.values")


def test_composite_provider_aliases_are_owned_by_their_top_level_path():
    """A nested model alias cannot steal fields from the top-level model provider."""

    @dataclasses.dataclass
    class Model:
        values: np.ndarray

    @dataclasses.dataclass
    class CollisionPipeline:
        model: Model
        broad_phase_pairs: np.ndarray

    @dataclasses.dataclass
    class Sources:
        collision_pipeline: CollisionPipeline
        model: Model

    model = Model(values=np.arange(4, dtype=np.float32))
    root = Sources(
        collision_pipeline=CollisionPipeline(
            model=model,
            broad_phase_pairs=np.asarray([[0, 1]], dtype=np.int32),
        ),
        model=model,
    )

    plan = capture.DebugCapturePlan.build(root, root_name="incident")

    assert [field.display_path for field in plan.fields] == [
        "incident.collision_pipeline.broad_phase_pairs",
        "incident.model.values",
    ]
    ignored = {entry.display_path: entry.reason for entry in plan.ignored}
    assert "reserved for top-level provider path 'incident.model'" in ignored["incident.collision_pipeline.model"]
