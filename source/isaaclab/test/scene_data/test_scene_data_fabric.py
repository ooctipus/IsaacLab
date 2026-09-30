# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for converting scene data into USD Fabric world matrices."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import warp as wp

from isaaclab.cloner.clone_plan import ClonePlan, PointCloudLayout
from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat, SceneDataPublication
from isaaclab.scene_data.scene_data_provider import SceneDataProvider

_SCALAR_CODES = {wp.float32: "f4", wp.float64: "f8"}


def _fabric_interface(data: wp.array, attrib: str, bucket_sizes: list[int]) -> dict:
    """Build a Fabric array interface over ``data`` without a live Kit stage.

    Fabric stores an attribute as a set of per-bucket pointers rather than one flat
    allocation, so the buckets are what make this a fair stand-in for a real selection.
    """
    shape = getattr(data.dtype, "_shape_", ())
    role = {1: "vector", 2: "matrix"}.get(len(shape), "")
    scalar_code = _SCALAR_CODES[getattr(data.dtype, "_wp_scalar_type_", data.dtype)]
    stride = wp.types.type_size_in_bytes(data.dtype)

    pointer = int(data.ptr)
    pointers = []
    for size in bucket_sizes:
        pointers.append(pointer)
        pointer += size * stride

    return {
        "version": 1,
        "device": str(data.device),
        "attribs": {
            attrib: {
                "type": (True, scalar_code, data.dtype._length_, 0, role),
                "access": 2,  # ReadWrite
                "pointers": pointers,
                "counts": list(bucket_sizes),
            }
        },
        "_ref": data,
    }


class _TransformBackend(SceneDataBackend):
    """Backend publishing packed ``wp.transformf`` state."""

    def __init__(self, positions: np.ndarray, device: str):
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1))
        packed = np.concatenate([positions, quaternions], axis=1).astype(np.float32)
        self._data = SceneDataFormat.Transform()
        self._data.transforms = wp.array(packed, dtype=wp.transformf, device=device)
        self._publication = SceneDataPublication(self._data, dirty=True)
        self.transform_reads = 0

    @property
    def transform_publication(self) -> SceneDataPublication:
        self.transform_reads += 1
        return self._publication


class _Vec3QuatBackend(_TransformBackend):
    """Backend publishing the same poses as separate position and quaternion arrays."""

    def __init__(self, positions: np.ndarray, device: str):
        super().__init__(positions, device)
        self._split = SceneDataFormat.Vec3_Quat()
        self._split.positions = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device=device)
        self._split.orientations = wp.array(
            np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1)).astype(np.float32), dtype=wp.quatf, device=device
        )
        self._publication.data = self._split


class _TransposedMatrixBackend(_TransformBackend):
    """Backend publishing the matrix layout consumed by USD xform attributes."""

    def __init__(self, positions: np.ndarray, device: str):
        super().__init__(positions, device)
        matrices = np.tile(np.eye(4), (len(positions), 1, 1))
        matrices[:, 3, :3] = positions
        self._data = SceneDataFormat.TransposedMatrix44d()
        self._data.matrices = wp.array(matrices, dtype=wp.mat44d, device=device)
        self._publication.data = self._data


class _IndexedTransformBackend(_TransformBackend):
    """Backend publishing native transforms plus clone-plan gather indices."""

    def __init__(self, positions: np.ndarray, source_indices: np.ndarray, device: str):
        super().__init__(positions, device)
        self._data = SceneDataFormat.IndexedTransform()
        self._data.transforms = self._publication.data.transforms
        self._data.source_indices = wp.array(source_indices.astype(np.int32), dtype=wp.int32, device=device)
        self._publication.data = self._data


def _fabric_output(
    source_order: np.ndarray, device: str, slots: np.ndarray | None = None
) -> tuple[SceneDataFormat.FabricMatrix44, wp.array, wp.array]:
    """Build the Fabric destination and its backing matrix arrays."""
    count = len(source_order)
    if slots is None:
        slots = np.arange(count, dtype=np.int32)
    base_count = int(slots.max()) + 1
    identity = np.tile(np.eye(4), (base_count, 1, 1))
    world_matrices = wp.array(identity, dtype=wp.mat44d, device=device)
    local_matrices = wp.array(identity, dtype=wp.mat44d, device=device)
    buckets = [base_count] if base_count < 4 else [2, base_count - 2]
    world = wp.fabricarray(data=_fabric_interface(world_matrices, "w", buckets), attrib="w")
    local = wp.fabricarray(data=_fabric_interface(local_matrices, "l", buckets), attrib="l")
    slot_array = wp.array(slots.astype(np.int32), dtype=wp.int32, device=device)
    output = SceneDataFormat.FabricMatrix44(
        matrices=wp.indexedfabricarray(fa=world, indices=slot_array),
        local_matrices=wp.indexedfabricarray(fa=local, indices=slot_array),
        source_indices=wp.array(source_order.astype(np.int32), dtype=wp.int32, device=device),
    )
    return output, world_matrices, local_matrices


def _write(backend: SceneDataBackend, source_order: np.ndarray, slots: np.ndarray, device: str) -> np.ndarray:
    """Run the provider's Fabric conversion and return the written local matrices."""
    output, _, local_matrices = _fabric_output(source_order, device, slots)

    SceneDataProvider(backend)._convert_transforms(backend.transform_publication.data, output, len(source_order))
    wp.synchronize_device(device)
    return local_matrices.numpy()


def test_convert_transforms_gathers_fabric_through_source_indices():
    """Each prim must take the backend transform its index names, in Fabric's layout."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    # reversed so a kernel that ignored source_indices would still be caught
    order = np.array([5, 4, 3, 2, 1, 0])
    slots = np.array([8, 2, 6, 1, 7, 4], dtype=np.int32)

    written = _write(_TransformBackend(positions, device), order, slots, device)

    # With identity parent frames, the converted local matrices equal the requested world poses.
    np.testing.assert_allclose(written[slots, 3, :3], positions[order], atol=1e-6)
    np.testing.assert_allclose(written[slots, 3, 3], 1.0, atol=1e-6)
    np.testing.assert_array_equal(written[[0, 3, 5]], np.tile(np.eye(4), (3, 1, 1)))


def test_named_vec3_quat_converts_to_plan_ordered_fabric_once_per_generation(monkeypatch):
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(9, dtype=np.float64).reshape(3, 3)
    backend = _TransformBackend(np.zeros((1, 3)), device)
    cameras = _Vec3QuatBackend(positions, device)
    provider = SceneDataProvider(backend)
    provider.register_transforms("camera", cameras.transform_publication)
    slots = np.array([5, 1, 3], dtype=np.int32)
    output, _, local_matrices = _fabric_output(np.arange(len(slots)), device, slots)
    provider._bind_fabric_outputs(
        lambda output_format, name: output
        if output_format is SceneDataFormat.FabricMatrix44 and name == "camera"
        else None
    )
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)
    first = provider.request_transforms(SceneDataFormat.FabricMatrix44, name="camera")
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44, name="camera") is first
    assert first is output
    assert launches[0].endswith("convert_Vec3_Quat_to_FabricMatrix44")
    assert len(launches) == 1
    assert provider._fabric_generation == 1
    written = local_matrices.numpy()
    np.testing.assert_allclose(written[slots, 3, :3], positions, atol=1e-6)
    np.testing.assert_array_equal(written[[0, 2, 4]], np.tile(np.eye(4), (3, 1, 1)))

    cameras._publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44, name="camera") is first
    assert len(launches) == 2
    assert provider._fabric_generation == 2

    mismatched = SceneDataProvider(backend)
    mismatched.register_transforms("camera", cameras.transform_publication)
    short_slots = np.array([2, 0], dtype=np.int32)
    short_output, _, _ = _fabric_output(np.arange(len(short_slots)), device, short_slots)
    mismatched._bind_fabric_outputs(lambda _output_format, _name: short_output)
    with np.testing.assert_raises_regex(RuntimeError, "has 2 entries; expected 3"):
        mismatched.request_transforms(SceneDataFormat.FabricMatrix44, name="camera")

    missing = SceneDataProvider(backend)
    missing.register_transforms("camera", cameras.transform_publication)
    missing._bind_fabric_outputs(lambda _output_format, _name: None)
    with np.testing.assert_raises_regex(RuntimeError, "destinations for 'camera' were not prepared"):
        missing.request_transforms(SceneDataFormat.FabricMatrix44, name="camera")


def test_named_vec3_quat_preserves_planned_world_scale_under_scaled_parent():
    """Replacing a camera's world pose must not erase its planned scale."""
    device = "cpu"
    target_position = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    backend = _TransformBackend(np.zeros((1, 3)), device)
    camera = _Vec3QuatBackend(target_position, device)
    provider = SceneDataProvider(backend)
    provider.register_transforms("camera", camera.transform_publication)

    output, world_matrices, local_matrices = _fabric_output(
        np.array([0], dtype=np.int32), device, np.array([1], dtype=np.int32)
    )
    parent = np.diag([2.0, 3.0, 4.0, 1.0])
    local = np.diag([0.5, 2.0, 0.25, 1.0])
    current_world = parent @ local
    world_storage = np.tile(np.eye(4), (2, 1, 1))
    local_storage = np.tile(np.eye(4), (2, 1, 1))
    world_storage[1] = current_world.T
    local_storage[1] = local.T
    world_matrices.assign(world_storage)
    local_matrices.assign(local_storage)
    provider._bind_fabric_outputs(
        lambda output_format, name: output
        if output_format is SceneDataFormat.FabricMatrix44 and name == "camera"
        else None
    )

    provider.request_transforms(SceneDataFormat.FabricMatrix44, name="camera")
    wp.synchronize_device(device)

    expected_world = current_world.copy()
    expected_world[:3, 3] = target_position[0]
    expected_local = np.linalg.inv(parent) @ expected_world
    np.testing.assert_allclose(local_matrices.numpy()[1], expected_local.T, atol=1e-6)


def test_transform_requests_alias_native_and_convert_once_per_generation(monkeypatch):
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)

    native_backend = _TransformBackend(positions, device)
    native_provider = SceneDataProvider(native_backend)
    native_calls = []
    monkeypatch.setattr(native_provider, "_convert_transforms", lambda *args: native_calls.append(args))
    assert native_provider.request_transforms(SceneDataFormat.Transform) is native_backend._data
    assert native_provider.request_transforms(SceneDataFormat.Transform) is native_backend._data
    assert native_calls == []

    converted_backend = _Vec3QuatBackend(positions, device)
    converted_provider = SceneDataProvider(converted_backend)
    original = converted_provider._convert_transforms
    conversion_calls = []

    def convert(*args):
        conversion_calls.append(args)
        original(*args)

    monkeypatch.setattr(converted_provider, "_convert_transforms", convert)
    first = converted_provider.request_transforms(SceneDataFormat.Transform)
    second = converted_provider.request_transforms(SceneDataFormat.Transform)
    assert first is second
    assert len(conversion_calls) == 1

    converted_backend._publication.dirty = True
    third = converted_provider.request_transforms(SceneDataFormat.Transform)
    assert third is first
    assert len(conversion_calls) == 2


def test_indexed_request_wraps_canonical_pointer_without_conversion(monkeypatch):
    """A canonical publisher gains an identity index map without copying its transform pointer."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    backend = _TransformBackend(positions, device)
    provider = SceneDataProvider(backend)
    conversion_calls = []
    monkeypatch.setattr(provider, "_convert_transforms", lambda *args: conversion_calls.append(args))

    first = provider.request_transforms(SceneDataFormat.IndexedTransform)
    assert provider.request_transforms(SceneDataFormat.IndexedTransform) is first
    assert first.transforms is backend._data.transforms
    assert first.source_indices.numpy().tolist() == list(range(len(positions)))
    assert conversion_calls == []

    replacement = _TransformBackend(positions + 10.0, device)._data.transforms
    backend._data.transforms = replacement
    backend._publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.IndexedTransform) is first
    assert first.transforms is replacement
    assert conversion_calls == []


def test_indexed_transform_gathers_each_requested_layout_in_one_kernel(monkeypatch):
    """Non-canonical native order is gathered directly, once for each requested layout."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(15, dtype=np.float64).reshape(5, 3)
    source_indices = np.array([4, 1, 3], dtype=np.int32)
    backend = _IndexedTransformBackend(positions, source_indices, device)
    provider = SceneDataProvider(backend)
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)
    assert provider.request_transforms(SceneDataFormat.IndexedTransform) is backend._data
    assert launches == []

    requests = (
        (SceneDataFormat.Transform, lambda data: data.transforms.numpy()[:, :3]),
        (SceneDataFormat.Vec3_Quat, lambda data: data.positions.numpy()),
        (SceneDataFormat.Vec3_Matrix33, lambda data: data.positions.numpy()),
        (SceneDataFormat.TransposedMatrix44d, lambda data: data.matrices.numpy()[:, 3, :3]),
        (SceneDataFormat.HostTransposedMatrix44d, lambda data: data.matrices[:, 3, :3]),
    )
    outputs = {}
    for output_format, positions_from in requests:
        start = len(launches)
        output = outputs[output_format] = provider.request_transforms(output_format)
        assert provider.request_transforms(output_format) is output
        assert len(launches[start:]) == 1
        assert launches[-1].endswith(f"convert_IndexedTransform_to_{output_format.__name__.removeprefix('Host')}")
        np.testing.assert_allclose(positions_from(output), positions[source_indices], atol=1e-6)

    replacement_positions = positions + 10.0
    backend._data.transforms = _TransformBackend(replacement_positions, device)._data.transforms
    backend._publication.dirty = True
    start = len(launches)
    refreshed = provider.request_transforms(SceneDataFormat.Transform)
    assert refreshed is outputs[SceneDataFormat.Transform]
    assert len(launches[start:]) == 1
    assert launches[-1].endswith("convert_IndexedTransform_to_Transform")
    np.testing.assert_allclose(refreshed.transforms.numpy()[:, :3], replacement_positions[source_indices], atol=1e-6)


def test_indexed_transform_gathers_fabric_directly_and_caches(monkeypatch):
    """Fabric composes its plan indices with native indices in one cached conversion."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    canonical_to_native = np.array([5, 3, 1, 4], dtype=np.int32)
    fabric_to_canonical = np.array([3, 1, 0], dtype=np.int32)
    backend = _IndexedTransformBackend(positions, canonical_to_native, device)
    provider = SceneDataProvider(backend)
    output, _, local_matrices = _fabric_output(fabric_to_canonical, device)
    provider._bind_fabric_outputs(
        lambda output_format, name: output if output_format is SceneDataFormat.FabricMatrix44 and name is None else None
    )
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is output
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is output
    assert len(launches) == 1
    assert launches[0].endswith("convert_IndexedTransform_to_FabricMatrix44")
    np.testing.assert_allclose(
        local_matrices.numpy()[:, 3, :3], positions[canonical_to_native[fabric_to_canonical]], atol=1e-6
    )


def test_transposed_matrix_request_preserves_pointer_and_generation_contract(monkeypatch):
    """The exact xform sink aliases natively or converts into one stable pointer per generation."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)

    native_backend = _TransposedMatrixBackend(positions, device)
    native_provider = SceneDataProvider(native_backend)
    native_calls = []
    monkeypatch.setattr(native_provider, "_convert_transforms", lambda *args: native_calls.append(args))
    native = native_provider.request_transforms(SceneDataFormat.TransposedMatrix44d)
    assert native is native_backend._data
    assert native_provider.request_transforms(SceneDataFormat.TransposedMatrix44d) is native
    assert native.matrices.ptr == native_backend._data.matrices.ptr
    assert native_calls == []

    round_trip = SceneDataProvider(_TransposedMatrixBackend(positions, device)).request_transforms(
        SceneDataFormat.Transform
    )
    wp.synchronize_device(device)
    np.testing.assert_allclose(round_trip.transforms.numpy()[:, :3], positions, atol=1e-6)

    backend = _TransformBackend(positions, device)
    provider = SceneDataProvider(backend)
    conversion_calls = []
    convert = provider._convert_transforms

    def record_conversion(*args):
        conversion_calls.append(args)
        convert(*args)

    monkeypatch.setattr(provider, "_convert_transforms", record_conversion)
    first = provider.request_transforms(SceneDataFormat.TransposedMatrix44d)
    assert provider.request_transforms(SceneDataFormat.TransposedMatrix44d) is first
    assert provider.transform_generation() == 1
    assert len(conversion_calls) == 1
    pointer = first.matrices.ptr
    wp.synchronize_device(device)
    np.testing.assert_allclose(first.matrices.numpy()[:, 3, :3], positions, atol=1e-6)

    backend._publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.TransposedMatrix44d) is first
    assert first.matrices.ptr == pointer
    assert provider.transform_generation() == 2
    assert len(conversion_calls) == 2


def test_host_matrix_request_moves_once_in_sdp_and_caches_the_result(monkeypatch):
    """A host-only sink requests its memory layout rather than copying renderer-side."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    backend = _TransformBackend(positions, device)
    provider = SceneDataProvider(backend)
    conversion_calls = []
    convert = provider._convert_transforms

    def record_conversion(*args):
        conversion_calls.append(args)
        convert(*args)

    monkeypatch.setattr(provider, "_convert_transforms", record_conversion)
    first = provider.request_transforms(SceneDataFormat.HostTransposedMatrix44d)
    assert provider.request_transforms(SceneDataFormat.HostTransposedMatrix44d) is first
    np.testing.assert_allclose(first.matrices[:, 3, :3], positions, atol=1e-6)
    assert len(conversion_calls) == 1

    backend._publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.HostTransposedMatrix44d) is first
    assert len(conversion_calls) == 2


def test_transform_consumers_refresh_independently_after_one_dirty_publication():
    """Clearing the publisher latch does not hide a generation from another cache key."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    for request_order in (("host", "split"), ("split", "host")):
        backend = _TransformBackend(np.zeros((2, 3)), device)
        provider = SceneDataProvider(backend)
        requests = {
            "host": lambda: provider.request_transforms(SceneDataFormat.HostTransposedMatrix44d),
            "split": lambda: provider.request_transforms(SceneDataFormat.Vec3_Quat),
        }
        for request in requests.values():
            request()

        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1))
        backend._data.transforms = wp.array(
            np.concatenate([positions, quaternions], axis=1), dtype=wp.transformf, device=device
        )
        backend._publication.dirty = True

        refreshed = {name: requests[name]() for name in request_order}
        np.testing.assert_allclose(refreshed["host"].matrices[:, 3, :3], positions)
        np.testing.assert_allclose(refreshed["split"].positions.numpy(), positions)


class _CountingBackend(SceneDataBackend):
    """Backend recording when a consumer requests native publications."""

    def __init__(self):
        self.transform_reads = 0
        points = SceneDataFormat.Points()
        points.points = wp.zeros(1, dtype=wp.vec3f)
        self._point_publication = SceneDataPublication(points, dirty=True)
        self._transform_publication = SceneDataPublication(SceneDataFormat.Transform(), dirty=True)

    @property
    def transform_publication(self) -> SceneDataPublication:
        self.transform_reads += 1
        return self._transform_publication

    @property
    def point_publications(self) -> dict[str, SceneDataPublication]:
        return {"points": self._point_publication}


def test_fabric_requests_fail_before_backend_reads_without_prepared_destinations():
    """A non-empty missing sink is an initialization bug, not a request-time fallback."""
    backend = _CountingBackend()
    provider = SceneDataProvider(backend)

    with np.testing.assert_raises_regex(RuntimeError, "point mapping.*not prepared"):
        provider.request_points(SceneDataFormat.FabricMeshPoints)
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is None

    transform_backend = _TransformBackend(np.zeros((1, 3)), "cpu")
    with np.testing.assert_raises_regex(RuntimeError, "transform destinations.*not prepared"):
        SceneDataProvider(transform_backend).request_transforms(SceneDataFormat.FabricMatrix44)


def test_prepared_fabric_transform_destination_prepares_once_per_generation(monkeypatch):
    backend = _TransformBackend(np.arange(6, dtype=np.float64).reshape(2, 3), "cpu")
    provider = SceneDataProvider(backend)
    outputs = [SimpleNamespace(source_indices=SimpleNamespace(device="cpu")) for _ in range(2)]
    prepared = iter(outputs)
    events = []

    def prepare(output_format, name):
        events.append(("prepare", name))
        return next(prepared) if output_format is SceneDataFormat.FabricMatrix44 else None

    provider._bind_fabric_outputs(prepare)

    def convert(*args):
        events.append(("convert", None))

    monkeypatch.setattr(provider, "_convert_transforms", convert)
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is outputs[0]
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is outputs[0]
    assert events == [("prepare", None), ("convert", None)]

    backend._publication.dirty = True
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is outputs[1]
    assert events == [("prepare", None), ("convert", None), ("prepare", None), ("convert", None)]


def test_prepared_fabric_point_destination_prepares_once_per_generation(monkeypatch):
    backend = _CountingBackend()
    provider = SceneDataProvider(backend)
    events = []

    plan = ClonePlan(
        (),
        (),
        np.zeros((0, 0), dtype=np.bool_),
        np.empty(0, dtype=np.int64),
        is_complete=True,
        point_clouds=(PointCloudLayout("/World/Points", 1, 0, None),),
    )
    provider._bind_point_plan(plan)
    outputs = [SimpleNamespace(binding_slots=SimpleNamespace(device="cpu")) for _ in range(2)]
    prepared = iter(outputs)

    def prepare(output_format, name):
        events.append(("prepare", name))
        return next(prepared) if output_format is SceneDataFormat.FabricMeshPoints and name == "points" else None

    provider._bind_fabric_outputs(prepare)
    monkeypatch.setattr(provider, "_convert_points", lambda *args: events.append(("convert", "points")))

    assert provider.request_points(SceneDataFormat.FabricMeshPoints) is outputs[0]
    assert provider.request_points(SceneDataFormat.FabricMeshPoints) is outputs[0]
    assert events == [("prepare", "points"), ("convert", "points")]

    backend._point_publication.dirty = True
    assert provider.request_points(SceneDataFormat.FabricMeshPoints) is outputs[1]
    assert events == [("prepare", "points"), ("convert", "points"), ("prepare", "points"), ("convert", "points")]


def test_fabric_requests_do_not_discover_or_author_stage_data():
    """The dynamic SDP boundary only transfers into clone-owned format pointers."""
    source = inspect.getsource(SceneDataProvider)
    assert all(
        name not in source
        for name in ("SelectPrims", "GetPrimAtPath", "DefinePrim", "get_current_stage", "usdrt", "hierarchy")
    )
    assert not hasattr(SceneDataProvider, "usd_stage")
    assert not hasattr(SceneDataProvider, "usdrt_stage")


def test_fabric_conversions_synchronize_only_the_producer_stream():
    """SDP waits for the Warp producer without draining unrelated device work."""
    source = "".join(
        inspect.getsource(method)
        for method in (
            SceneDataProvider._convert_transforms,
            SceneDataProvider._convert_points,
            SceneDataProvider._convert_point_bundle,
        )
    )
    assert "synchronize_device" not in source
    assert source.count("synchronize_stream") == 4


def test_clone_bound_fabric_output_ignores_backend_only_body():
    class PlannedBackend(_TransformBackend):
        @property
        def point_publications(self) -> dict[str, SceneDataPublication]:
            return {}

    backend = PlannedBackend(np.zeros((1, 3)), "cpu")
    provider = SceneDataProvider(backend)
    provider._bind_fabric_outputs(lambda _output_format, _name: None)
    assert provider.request_transforms(SceneDataFormat.FabricMatrix44) is None
