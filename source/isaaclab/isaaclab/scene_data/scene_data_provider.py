# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from isaaclab.utils.warp.fabric import _decompose_transformation_matrix

from .geometry_points import (
    body_points_to_fabric_mesh_points_kernel,
    body_points_to_mesh_points_kernel,
    body_points_to_points_kernel,
    cable_points_to_fabric_mesh_points_kernel,
    cable_points_to_points_kernel,
)
from .scene_data_backend import SceneDataBackend, SceneDataFormat, SceneDataPublication

if TYPE_CHECKING:
    from isaaclab.cloner.clone_plan import ClonePlan

    IndexedFabricArrayMat44d = Any
else:
    IndexedFabricArrayMat44d = wp.indexedfabricarray(dtype=wp.mat44d)


def _data_arrays(data: Any) -> tuple[Any, ...]:
    """Return every non-null pointer field in a published Warp struct."""
    return tuple(getattr(data, name) for name in data._cls.vars if getattr(data, name) is not None)


def _data_count(data: Any) -> int:
    """Return the common leading dimension of a publication's pointers."""
    if getattr(data, "_cls", type(data)) is SceneDataFormat.IndexedTransform:
        if data.transforms is None or data.source_indices is None:
            return 0
        _data_device(data)
        return int(data.source_indices.shape[0])
    arrays = _data_arrays(data)
    if not arrays:
        return 0
    counts = {int(array.shape[0]) for array in arrays}
    if len(counts) != 1:
        raise RuntimeError("SDP publication fields must have one element count.")
    return counts.pop()


def _data_device(data: Any):
    devices = [array.device for array in _data_arrays(data)]
    if not devices or any(str(device) != str(devices[0]) for device in devices[1:]):
        raise RuntimeError("SDP publication fields must expose pointers on one device.")
    return devices[0]


def _point_device(data: Any):
    """Return the common device of one native point pointer bundle."""
    if isinstance(data, SceneDataFormat.BodyPoints):
        arrays = (*data.points, *data.binding_ids)
    elif isinstance(data, SceneDataFormat.CablePoints):
        arrays = tuple(
            getattr(data, name)
            for name in (
                "body_q",
                "shape_body",
                "shape_transform",
                "shape_scale",
                "shape_ids",
                "shape_offsets",
                "segment_counts",
                "binding_ids",
            )
        )
    else:
        return _data_device(data)
    if not arrays or any(array is None for array in arrays):
        raise RuntimeError("SDP native point publication has missing pointers.")
    devices = {str(array.device) for array in arrays}
    if len(devices) != 1:
        raise RuntimeError("SDP native point publication pointers must share one device.")
    return arrays[0].device


class SceneDataProvider:
    def __init__(self, backend: SceneDataBackend):
        """Initialize the scene data provider.

        Args:
            backend: The simulation backend that supplies raw transform data.
        """
        self._backend = backend
        self._generations: dict[tuple[str, str | None], int] = {("transforms", None): 0}
        self._fabric_generation = 0
        self._transform_publications: dict[str, SceneDataPublication] = {}
        self._cache: dict[tuple[str, str | None, Any], tuple[int, Any]] = {}
        self._prepare_fabric_output: Callable[[Any, str | None], Any | None] | None = None
        self._host_transform_staging: dict[str | None, SceneDataFormat.TransposedMatrix44d] = {}
        self._point_maps: dict[str, tuple[Any, ...]] = {}
        self._point_source_counts: dict[str, int] = {}
        self._point_device_maps: dict[tuple[str, str], tuple[wp.array, ...]] = {}
        self._host_point_staging: dict[tuple[str, Any], wp.array] = {}

    def _bind_point_plan(self, plan: ClonePlan) -> None:
        """Bind the final plan's static native-to-drawable point maps once."""
        for name in plan.point_stream_names:
            bindings = plan.point_bindings(name)
            self._point_source_counts[name] = sum(binding.source_count for binding in bindings)
            mapping_offsets = np.full(len(bindings), -1, dtype=np.int32)
            templates: dict[tuple[int, int], int] = {}
            index_templates = []
            weight_templates = []
            mapping_count = 0
            for binding_index, binding in enumerate(bindings):
                if binding.source_indices is None:
                    continue
                key = (id(binding.source_indices), id(binding.weights))
                mapping_offset = templates.get(key)
                if mapping_offset is None:
                    mapping_offset = templates[key] = mapping_count
                    index_templates.append(binding.source_indices)
                    weight_templates.append(binding.weights)
                    mapping_count += binding.output_count
                mapping_offsets[binding_index] = mapping_offset
            self._point_maps[name] = (
                np.repeat(np.arange(len(bindings), dtype=np.int32), [binding.output_count for binding in bindings]),
                np.asarray([binding.source_offset for binding in bindings], dtype=np.int32),
                np.asarray([binding.source_count for binding in bindings], dtype=np.int32),
                np.asarray([binding.output_offset for binding in bindings], dtype=np.int32),
                np.asarray([binding.output_count for binding in bindings], dtype=np.int32),
                mapping_offsets,
                np.concatenate(index_templates) if index_templates else np.empty((0, 4), dtype=np.int32),
                np.concatenate(weight_templates) if weight_templates else np.empty((0, 4), dtype=np.float32),
            )

    def register_transforms(self, name: str, publication: SceneDataPublication) -> None:
        """Register one named transform publication."""
        current = self._transform_publications.get(name)
        if current is not None and current is not publication:
            raise ValueError(f"Transform publication {name!r} is already registered.")
        self._transform_publications[name] = publication
        self._generations.setdefault(("transforms", name), 0)

    def unregister_transforms(self, name: str, publication: SceneDataPublication) -> None:
        """Remove a named transform publication when ``publication`` still owns it."""
        if self._transform_publications.get(name) is publication:
            del self._transform_publications[name]
            self._generations.pop(("transforms", name), None)
            self._host_transform_staging.pop(name, None)
            self._cache = {key: value for key, value in self._cache.items() if key[:2] != ("transforms", name)}

    def request_transforms(
        self,
        output_format: Any,
        name: str | None = None,
    ) -> Any | None:
        """Return transforms in ``output_format``, converting once per dirty generation.

        The provider owns converted buffers. Repeated requests for the same format return the same
        current pointer. A native-format request aliases the publisher and launches no conversion.

        Args:
            output_format: Requested :class:`SceneDataFormat` struct type.
            name: Named transform publication, or ``None`` for the physics scene.

        Returns:
            Provider-owned output, or ``None`` when the backend publishes no transforms.

        Raises:
            RuntimeError: If no conversion exists or a non-empty destination was not prepared.
        """
        publication = self._backend.transform_publication if name is None else self._transform_publications[name]
        count = _data_count(publication.data)
        if not count:
            return None

        generation = self._refresh_generation(("transforms", name), publication)
        key = ("transforms", name, output_format)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == generation:
            return cached[1]
        input = publication.data
        output = cached[1] if cached is not None else None
        if getattr(input, "_cls", type(input)) is not output_format and output_format is SceneDataFormat.FabricMatrix44:
            if self._prepare_fabric_output is None:
                raise RuntimeError("Fabric transform destinations were not prepared before initialization.")
            output = self._prepare_fabric_output(output_format, name)
            if output is None:
                if name is None:
                    return None
                raise RuntimeError(f"Fabric transform destinations for {name!r} were not prepared.")
        if getattr(input, "_cls", type(input)) is output_format:
            output = input
        else:
            if output is None:
                output = output_format()
            if input._cls is SceneDataFormat.Transform and output_format is SceneDataFormat.IndexedTransform:
                output.transforms = input.transforms
                if output.source_indices is None:
                    output.source_indices = wp.array(np.arange(count, dtype=np.int32), device=input.transforms.device)
            else:
                self._init_output(output, count, input)
                self._convert_transforms(input, output, count, name)
        self._cache[key] = (generation, output)
        if output_format is SceneDataFormat.FabricMatrix44:
            self._fabric_generation += 1
        return output

    def transform_generation(self, name: str | None = None) -> int:
        """Generation of a transform publication requested from this provider."""
        return self._generations.get(("transforms", name), 0)

    def point_generation(self, name: str = "points") -> int:
        """Generation of a named point publication requested from this provider."""
        return self._generations.get(("points", name), 0)

    def _refresh_generation(self, key: tuple[str, str | None], publication: SceneDataPublication) -> int:
        """Consume a dirty latch and return its provider-owned generation."""
        generation = self._generations.get(key, 0)
        if publication.dirty:
            if key[0] == "points" or key[1] is None:
                self._backend._materialize(publication)
            generation += 1
            self._generations[key] = generation
            publication.dirty = False
        return generation

    def _convert_transforms(
        self,
        input,
        output,
        count: int,
        name: str | None = None,
    ) -> None:
        if isinstance(output, SceneDataFormat.HostTransposedMatrix44d):
            if input._cls is SceneDataFormat.TransposedMatrix44d:
                matrices = input.matrices
            else:
                staging = self._host_transform_staging.get(name)
                if staging is None:
                    staging = self._host_transform_staging[name] = SceneDataFormat.TransposedMatrix44d()
                    self._init_output(staging, count, input)
                matrices = staging.matrices
                kernel_name = f"convert_{input._cls.__name__}_to_TransposedMatrix44d"
                kernel = getattr(ConversionKernels, kernel_name, None)
                if kernel is None:
                    raise RuntimeError(f"SceneDataProvider has no {kernel_name} conversion.")
                wp.launch(
                    kernel=kernel,
                    dim=count,
                    inputs=[input],
                    outputs=[staging],
                    device=staging.matrices.device,
                )
            output.matrices = matrices.numpy().reshape(-1, 4, 4)
            return
        if isinstance(output, SceneDataFormat.FabricMatrix44):
            kernel_name = f"convert_{input._cls.__name__}_to_FabricMatrix44"
            if input._cls is SceneDataFormat.Vec3_Quat:
                if output.source_indices.shape[0] != count:
                    raise RuntimeError(
                        f"Fabric transform destination has {output.source_indices.shape[0]} entries; expected {count}."
                    )
                wp.launch(
                    ConversionKernels.convert_Vec3_Quat_to_FabricMatrix44,
                    dim=count,
                    inputs=[input.positions, input.orientations, output.matrices, output.local_matrices],
                    device=output.source_indices.device,
                )
                wp.synchronize_stream(output.source_indices.device)
                return
            elif input._cls is SceneDataFormat.Transform:
                kernel = ConversionKernels.convert_Transform_to_FabricMatrix44
                inputs = [input.transforms, output.source_indices, output.matrices]
            elif input._cls is SceneDataFormat.IndexedTransform:
                kernel = ConversionKernels.convert_IndexedTransform_to_FabricMatrix44
                inputs = [input.transforms, input.source_indices, output.source_indices, output.matrices]
            else:
                raise RuntimeError(f"SceneDataProvider has no {kernel_name} conversion.")
            wp.launch(
                kernel,
                dim=output.source_indices.shape[0],
                inputs=inputs,
                outputs=[output.local_matrices],
                device=output.source_indices.device,
            )
            wp.synchronize_stream(output.source_indices.device)
            return
        kernel_name = f"convert_{input._cls.__name__}_to_{output._cls.__name__}"
        kernel = getattr(ConversionKernels, kernel_name, None)
        if kernel is None:
            raise RuntimeError(f"SceneDataProvider has no {kernel_name} conversion.")
        wp.launch(
            kernel=kernel,
            dim=count,
            inputs=[input],
            outputs=[output],
            device=_data_device(output),
        )

    def _bind_fabric_outputs(
        self,
        prepare: Callable[[Any, str | None], Any | None],
    ) -> None:
        """Bind format pointers materialized by the clone backend."""
        self._prepare_fabric_output = prepare

    def _init_output(self, output: Any, count: int, input: Any) -> None:
        """Allocate converted Warp fields on the publisher's device."""
        if not hasattr(output, "_cls"):
            return
        device = _data_device(input)
        for field_name, field_value in output._cls.vars.items():
            if getattr(output, field_name) is None:
                setattr(output, field_name, wp.empty(count, dtype=field_value.type.dtype, device=device))

    def request_points(
        self,
        output_format: Any,
        name: str = "points",
    ) -> Any | None:
        """Return geometry points in ``output_format`` once per dirty generation.

        Args:
            output_format: Requested :class:`SceneDataFormat` struct type.
            name: Published point-stream name.

        Returns:
            Provider-owned output, or ``None`` when no points or Fabric destination exist.

        Raises:
            RuntimeError: If no conversion exists or a non-empty destination was not prepared.
        """
        publication = self._backend.point_publications[name]
        input = publication.data
        input_format = getattr(input, "_cls", type(input))
        native_bundle = isinstance(input, (SceneDataFormat.BodyPoints, SceneDataFormat.CablePoints))
        if native_bundle:
            ready = bool(input.points) if isinstance(input, SceneDataFormat.BodyPoints) else input.body_q is not None
            if not ready:
                return None
            if name not in self._point_maps:
                raise RuntimeError(f"Plan-owned point mapping for {name!r} was not prepared before initialization.")
            count = self._point_source_counts[name]
            self._validate_point_bundle(input, name)
        else:
            count = _data_count(input)
            if not count:
                return None
        if output_format in (SceneDataFormat.FabricMeshPoints, SceneDataFormat.HostMeshPoints) or native_bundle:
            if name not in self._point_maps:
                raise RuntimeError(f"Plan-owned point mapping for {name!r} was not prepared before initialization.")
            if not native_bundle and count != self._point_source_counts[name]:
                raise RuntimeError(
                    f"Point publication {name!r} has {count} native points; the clone plan requires "
                    f"{self._point_source_counts[name]}."
                )

        generation = self._refresh_generation(("points", name), publication)
        key = ("points", name, output_format)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == generation:
            return cached[1]
        output = cached[1] if cached is not None else None
        if output_format is SceneDataFormat.FabricMeshPoints and input_format is not output_format:
            if self._prepare_fabric_output is None:
                raise RuntimeError(f"Fabric point destinations for {name!r} were not prepared before initialization.")
            output = self._prepare_fabric_output(output_format, name)
            if output is None:
                raise RuntimeError(f"Fabric point destinations for {name!r} were not prepared before initialization.")
        if input_format is output_format:
            output = input
        else:
            if output is None:
                output = output_format()
            self._convert_points(input, output, count, name)
        self._cache[key] = (generation, output)
        return output

    def _validate_point_bundle(self, input: Any, name: str) -> None:
        """Validate native pointer-bundle cardinality against the plan without moving data."""
        _point_device(input)
        binding_count = len(self._point_maps[name][1])
        if isinstance(input, SceneDataFormat.BodyPoints):
            if (
                len(input.points) != len(input.binding_ids)
                or sum(points.shape[0] for points in input.points) != binding_count
            ):
                raise RuntimeError(f"BodyPoints publication {name!r} does not cover its {binding_count} plan bindings.")
            if any(
                len(points.shape) != 2 or points.shape[0] != ids.shape[0]
                for points, ids in zip(input.points, input.binding_ids, strict=True)
            ):
                raise RuntimeError(f"BodyPoints publication {name!r} has incompatible pointer and binding shapes.")
        elif any(
            array.shape[0] != binding_count for array in (input.shape_offsets, input.segment_counts, input.binding_ids)
        ):
            raise RuntimeError(f"CablePoints publication {name!r} does not cover its {binding_count} plan bindings.")

    def _device_point_map(self, name: str, device: Any) -> tuple[wp.array, ...]:
        """Upload one plan-owned gather map once for a publication device."""
        key = (name, str(device))
        if key not in self._point_device_maps:
            (
                bindings,
                source_offsets,
                source_counts,
                output_offsets,
                output_counts,
                mapping_offsets,
                indices,
                weights,
            ) = self._point_maps[name]
            self._point_device_maps[key] = (
                wp.array(bindings, dtype=wp.int32, device=device),
                wp.array(source_offsets, dtype=wp.int32, device=device),
                wp.array(source_counts, dtype=wp.int32, device=device),
                wp.array(output_offsets, dtype=wp.int32, device=device),
                wp.array(output_counts, dtype=wp.int32, device=device),
                wp.array(mapping_offsets, dtype=wp.int32, device=device),
                wp.array(indices, dtype=wp.int32, device=device),
                wp.array(weights, dtype=wp.float32, device=device),
            )
        return self._point_device_maps[key]

    def _convert_points(self, input, output, count: int, name: str) -> None:
        if isinstance(input, (SceneDataFormat.BodyPoints, SceneDataFormat.CablePoints)):
            self._convert_point_bundle(input, output, count, name)
            return
        if isinstance(output, SceneDataFormat.HostPoints):
            if input._cls is not SceneDataFormat.Points:
                raise RuntimeError(f"SceneDataProvider has no convert_{input._cls.__name__}_to_HostPoints conversion.")
            output.points = input.points.numpy().reshape(-1, 3)
            return
        if isinstance(output, SceneDataFormat.HostMeshPoints):
            if input._cls is not SceneDataFormat.Points:
                raise RuntimeError(
                    f"SceneDataProvider has no convert_{input._cls.__name__}_to_HostMeshPoints conversion."
                )
            bindings, source_offsets, _, output_offsets, _, mapping_offsets, indices, weights = self._device_point_map(
                name, input.points.device
            )
            key = (name, SceneDataFormat.HostMeshPoints)
            staging = self._host_point_staging.get(key)
            if staging is None:
                staging = self._host_point_staging[key] = wp.empty(
                    bindings.shape[0], dtype=wp.vec3f, device=input.points.device
                )
            wp.launch(
                ConversionKernels.convert_Points_to_HostMeshPoints,
                dim=bindings.shape[0],
                inputs=[
                    input.points,
                    bindings,
                    source_offsets,
                    output_offsets,
                    mapping_offsets,
                    indices,
                    weights,
                ],
                outputs=[staging],
                device=input.points.device,
            )
            output.points = staging.numpy().reshape(-1, 3)
            return
        if isinstance(output, SceneDataFormat.FabricMeshPoints):
            kernel_name = f"convert_{input._cls.__name__}_to_FabricMeshPoints"
            if input._cls is not SceneDataFormat.Points:
                raise RuntimeError(f"SceneDataProvider has no {kernel_name} conversion.")
            bindings, source_offsets, _, output_offsets, _, mapping_offsets, indices, weights = self._device_point_map(
                name, input.points.device
            )
            wp.launch(
                ConversionKernels.convert_Points_to_FabricMeshPoints,
                dim=bindings.shape[0],
                inputs=[
                    input.points,
                    output.world_matrices,
                    output.binding_slots,
                    bindings,
                    source_offsets,
                    output_offsets,
                    mapping_offsets,
                    indices,
                    weights,
                ],
                outputs=[output.points],
                device=input.points.device,
            )
            wp.synchronize_stream(input.points.device)
            return
        kernel_name = f"convert_{input._cls.__name__}_to_{output._cls.__name__}"
        kernel = getattr(ConversionKernels, kernel_name, None)
        if kernel is None:
            raise RuntimeError(f"SceneDataProvider has no {kernel_name} conversion.")
        self._init_output(output, count, input)
        wp.launch(kernel, dim=count, inputs=[input], outputs=[output], device=_data_device(input))

    def _convert_point_bundle(self, input, output, count: int, name: str) -> None:
        """Convert a native body or cable pointer bundle directly into one requested destination."""
        device = _point_device(input)
        (
            bindings,
            source_offsets,
            source_counts,
            output_offsets,
            output_counts,
            mapping_offsets,
            indices,
            weights,
        ) = self._device_point_map(name, device)

        if getattr(output, "_cls", type(output)) is SceneDataFormat.Points:
            if output.points is None:
                output.points = wp.empty(count, dtype=wp.vec3f, device=device)
            self._convert_point_bundle_to_flat(input, output.points, source_offsets, source_counts)
            return
        if isinstance(output, (SceneDataFormat.HostPoints, SceneDataFormat.HostMeshPoints)):
            mesh = isinstance(output, SceneDataFormat.HostMeshPoints)
            size = bindings.shape[0] if mesh else count
            key = (name, type(output))
            staging = self._host_point_staging.get(key)
            if staging is None:
                staging = self._host_point_staging[key] = wp.empty(size, dtype=wp.vec3f, device=device)
            if mesh and isinstance(input, SceneDataFormat.BodyPoints):
                for points, binding_ids in zip(input.points, input.binding_ids, strict=True):
                    wp.launch(
                        body_points_to_mesh_points_kernel,
                        dim=points.shape[0],
                        inputs=[
                            points,
                            binding_ids,
                            output_offsets,
                            output_counts,
                            mapping_offsets,
                            indices,
                            weights,
                        ],
                        outputs=[staging],
                        device=device,
                    )
            else:
                offsets = output_offsets if mesh else source_offsets
                self._convert_point_bundle_to_flat(input, staging, offsets, source_counts)
            output.points = staging.numpy().reshape(-1, 3)
            return
        if isinstance(output, SceneDataFormat.FabricMeshPoints):
            if isinstance(input, SceneDataFormat.BodyPoints):
                for points, binding_ids in zip(input.points, input.binding_ids, strict=True):
                    wp.launch(
                        body_points_to_fabric_mesh_points_kernel,
                        dim=points.shape[0],
                        inputs=[
                            points,
                            binding_ids,
                            output.world_matrices,
                            output.binding_slots,
                            output_counts,
                            mapping_offsets,
                            indices,
                            weights,
                        ],
                        outputs=[output.points],
                        device=device,
                    )
            else:
                wp.launch(
                    cable_points_to_fabric_mesh_points_kernel,
                    dim=input.segment_counts.shape[0],
                    inputs=[
                        input.shape_offsets,
                        input.segment_counts,
                        input.binding_ids,
                        input.shape_ids,
                        input.shape_body,
                        input.body_q,
                        input.shape_transform,
                        input.shape_scale,
                        output.world_matrices,
                        output.binding_slots,
                    ],
                    outputs=[output.points],
                    device=device,
                )
            wp.synchronize_stream(device)
            return
        raise RuntimeError(
            f"SceneDataProvider has no convert_{type(input).__name__}_to_{type(output).__name__} conversion."
        )

    def _convert_point_bundle_to_flat(
        self,
        input: SceneDataFormat.BodyPoints | SceneDataFormat.CablePoints,
        output: wp.array,
        offsets: wp.array,
        source_counts: wp.array,
    ) -> None:
        """Gather one native bundle directly into a requested flat output."""
        device = _point_device(input)
        if isinstance(input, SceneDataFormat.BodyPoints):
            for points, binding_ids in zip(input.points, input.binding_ids, strict=True):
                wp.launch(
                    body_points_to_points_kernel,
                    dim=points.shape[0],
                    inputs=[points, binding_ids, offsets, source_counts],
                    outputs=[output],
                    device=device,
                )
        else:
            wp.launch(
                cable_points_to_points_kernel,
                dim=input.segment_counts.shape[0],
                inputs=[
                    input.shape_offsets,
                    input.segment_counts,
                    input.binding_ids,
                    offsets,
                    input.shape_ids,
                    input.shape_body,
                    input.body_q,
                    input.shape_transform,
                    input.shape_scale,
                ],
                outputs=[output],
                device=device,
            )


class ConversionKernels:
    @wp.kernel
    def convert_IndexedTransform_to_Vec3_Quat(
        input: SceneDataFormat.IndexedTransform, output: SceneDataFormat.Vec3_Quat
    ):
        """Gather native transforms into canonical Vec3/Quat arrays."""
        tid = wp.tid()
        transform = input.transforms[input.source_indices[tid]]
        output.positions[tid] = wp.transform_get_translation(transform)
        output.orientations[tid] = wp.transform_get_rotation(transform)

    @wp.kernel
    def convert_IndexedTransform_to_Vec3_Matrix33(
        input: SceneDataFormat.IndexedTransform, output: SceneDataFormat.Vec3_Matrix33
    ):
        """Gather native transforms into canonical Vec3/Matrix33 arrays."""
        tid = wp.tid()
        transform = input.transforms[input.source_indices[tid]]
        output.positions[tid] = wp.transform_get_translation(transform)
        output.orientations[tid] = wp.quat_to_matrix(wp.transform_get_rotation(transform))

    @wp.kernel
    def convert_IndexedTransform_to_Transform(
        input: SceneDataFormat.IndexedTransform, output: SceneDataFormat.Transform
    ):
        """Gather native transforms into canonical packed transforms."""
        tid = wp.tid()
        output.transforms[tid] = input.transforms[input.source_indices[tid]]

    @wp.kernel
    def convert_IndexedTransform_to_TransposedMatrix44d(
        input: SceneDataFormat.IndexedTransform, output: SceneDataFormat.TransposedMatrix44d
    ):
        """Gather native transforms into canonical transposed matrices."""
        tid = wp.tid()
        transform = input.transforms[input.source_indices[tid]]
        output.matrices[tid] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))

    @wp.kernel
    def convert_Vec3_Quat_to_Vec3_Matrix33(input: SceneDataFormat.Vec3_Quat, output: SceneDataFormat.Vec3_Matrix33):
        """Convert Vec3/Quat to Vec3/Matrix33"""
        tid = wp.tid()
        output.positions[tid] = input.positions[tid]
        output.orientations[tid] = wp.quat_to_matrix(input.orientations[tid])

    @wp.kernel
    def convert_Vec3_Quat_to_Transform(input: SceneDataFormat.Vec3_Quat, output: SceneDataFormat.Transform):
        """Convert Vec3/Quat to Transform"""
        tid = wp.tid()
        output.transforms[tid] = wp.transformf(input.positions[tid], input.orientations[tid])

    @wp.kernel
    def convert_Vec3_Quat_to_TransposedMatrix44d(
        input: SceneDataFormat.Vec3_Quat,
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        """Convert Vec3/Quat to transposed double-precision Matrix44."""
        tid = wp.tid()
        matrix = wp.transform_to_matrix(wp.transformf(input.positions[tid], input.orientations[tid]))
        output.matrices[tid] = wp.transpose(wp.mat44d(matrix))

    @wp.kernel
    def convert_Vec3_Matrix33_to_Vec3_Quat(input: SceneDataFormat.Vec3_Matrix33, output: SceneDataFormat.Vec3_Quat):
        """Convert Vec3/Matrix33 to Vec3/Quat"""
        tid = wp.tid()
        output.positions[tid] = input.positions[tid]
        output.orientations[tid] = wp.quat_from_matrix(input.orientations[tid])

    @wp.kernel
    def convert_Vec3_Matrix33_to_Transform(input: SceneDataFormat.Vec3_Matrix33, output: SceneDataFormat.Transform):
        """Convert Vec3/Matrix33 to Transform"""
        tid = wp.tid()
        output.transforms[tid] = wp.transformf(input.positions[tid], wp.quat_from_matrix(input.orientations[tid]))

    @wp.kernel
    def convert_Vec3_Matrix33_to_TransposedMatrix44d(
        input: SceneDataFormat.Vec3_Matrix33,
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        """Convert Vec3/Matrix33 to transposed double-precision Matrix44."""
        tid = wp.tid()
        transform = wp.transformf(input.positions[tid], wp.quat_from_matrix(input.orientations[tid]))
        output.matrices[tid] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))

    @wp.kernel
    def convert_Transform_to_Vec3_Quat(input: SceneDataFormat.Transform, output: SceneDataFormat.Vec3_Quat):
        """Convert Transform to Vec3/Quat"""
        tid = wp.tid()
        output.positions[tid] = wp.transform_get_translation(input.transforms[tid])
        output.orientations[tid] = wp.transform_get_rotation(input.transforms[tid])

    @wp.kernel
    def convert_Transform_to_Vec3_Matrix33(input: SceneDataFormat.Transform, output: SceneDataFormat.Vec3_Matrix33):
        """Convert Transform to Vec3/Matrix33"""
        tid = wp.tid()
        output.positions[tid] = wp.transform_get_translation(input.transforms[tid])
        output.orientations[tid] = wp.quat_to_matrix(wp.transform_get_rotation(input.transforms[tid]))

    @wp.kernel
    def convert_Transform_to_TransposedMatrix44d(
        input: SceneDataFormat.Transform,
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        """Convert Transform to transposed double-precision Matrix44."""
        tid = wp.tid()
        output.matrices[tid] = wp.transpose(wp.mat44d(wp.transform_to_matrix(input.transforms[tid])))

    @wp.kernel
    def convert_TransposedMatrix44d_to_Vec3_Quat(
        input: SceneDataFormat.TransposedMatrix44d,
        output: SceneDataFormat.Vec3_Quat,
    ):
        """Convert transposed double-precision Matrix44 to Vec3/Quat."""
        tid = wp.tid()
        transform = wp.transform_from_matrix(wp.mat44f(wp.transpose(input.matrices[tid])))
        output.positions[tid] = wp.transform_get_translation(transform)
        output.orientations[tid] = wp.transform_get_rotation(transform)

    @wp.kernel
    def convert_TransposedMatrix44d_to_Vec3_Matrix33(
        input: SceneDataFormat.TransposedMatrix44d,
        output: SceneDataFormat.Vec3_Matrix33,
    ):
        """Convert transposed double-precision Matrix44 to Vec3/Matrix33."""
        tid = wp.tid()
        transform = wp.transform_from_matrix(wp.mat44f(wp.transpose(input.matrices[tid])))
        output.positions[tid] = wp.transform_get_translation(transform)
        output.orientations[tid] = wp.quat_to_matrix(wp.transform_get_rotation(transform))

    @wp.kernel
    def convert_TransposedMatrix44d_to_Transform(
        input: SceneDataFormat.TransposedMatrix44d,
        output: SceneDataFormat.Transform,
    ):
        """Convert transposed double-precision Matrix44 to Transform."""
        tid = wp.tid()
        output.transforms[tid] = wp.transform_from_matrix(wp.mat44f(wp.transpose(input.matrices[tid])))

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Quat_to_FabricMatrix44(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        world_matrices: IndexedFabricArrayMat44d,
        local_matrices: IndexedFabricArrayMat44d,
    ):
        """Convert named world poses into plan-owned Fabric local matrices."""
        i = wp.tid()
        _position, _orientation, scale = _decompose_transformation_matrix(wp.mat44f(world_matrices[i]))
        world = wp.transpose(wp.mat44d(wp.transform_compose(positions[i], orientations[i], scale)))
        local_matrices[i] = world * wp.inverse(world_matrices[i]) * local_matrices[i]

    @wp.kernel(enable_backward=False)
    def convert_Transform_to_FabricMatrix44(
        transforms: wp.array(dtype=wp.transformf),
        source_indices: wp.array(dtype=wp.int32),
        world_matrices: IndexedFabricArrayMat44d,
        local_matrices: IndexedFabricArrayMat44d,
    ):
        """Convert world-space transforms to authoritative Fabric local matrices.

        Fabric stores matrices transposed and in double precision. The destination arrays are
        indexed into exact clone-plan order independently of Fabric selection order.
        """
        i = wp.tid()
        source_index = source_indices[i]
        world = wp.transpose(wp.mat44d(wp.transform_to_matrix(transforms[source_index])))
        local_matrices[i] = world * wp.inverse(world_matrices[i]) * local_matrices[i]

    @wp.kernel(enable_backward=False)
    def convert_IndexedTransform_to_FabricMatrix44(
        transforms: wp.array(dtype=wp.transformf),
        canonical_to_native: wp.array(dtype=wp.int32),
        source_indices: wp.array(dtype=wp.int32),
        world_matrices: IndexedFabricArrayMat44d,
        local_matrices: IndexedFabricArrayMat44d,
    ):
        """Gather native transforms directly into clone-plan-indexed Fabric matrices."""
        i = wp.tid()
        source_index = canonical_to_native[source_indices[i]]
        world = wp.transpose(wp.mat44d(wp.transform_to_matrix(transforms[source_index])))
        local_matrices[i] = world * wp.inverse(world_matrices[i]) * local_matrices[i]

    @wp.kernel(enable_backward=False)
    def convert_Points_to_HostMeshPoints(
        input_points: wp.array(dtype=wp.vec3f),
        output_bindings: wp.array(dtype=wp.int32),
        source_offsets: wp.array(dtype=wp.int32),
        output_offsets: wp.array(dtype=wp.int32),
        mapping_offsets: wp.array(dtype=wp.int32),
        source_indices: wp.array2d(dtype=wp.int32),
        weights: wp.array2d(dtype=wp.float32),
        output_points: wp.array(dtype=wp.vec3f),
    ):
        """Interpolate native simulation nodes into flattened drawable points."""
        i = wp.tid()
        binding = output_bindings[i]
        local = i - output_offsets[binding]
        mapping = mapping_offsets[binding]
        if mapping < 0:
            output_points[i] = input_points[source_offsets[binding] + local]
        else:
            row = mapping + local
            source = source_offsets[binding]
            output_points[i] = (
                weights[row, 0] * input_points[source + source_indices[row, 0]]
                + weights[row, 1] * input_points[source + source_indices[row, 1]]
                + weights[row, 2] * input_points[source + source_indices[row, 2]]
                + weights[row, 3] * input_points[source + source_indices[row, 3]]
            )

    @wp.kernel(enable_backward=False)
    def convert_Points_to_FabricMeshPoints(
        input_points: wp.array(dtype=wp.vec3f),
        world_matrices: wp.fabricarray(dtype=wp.mat44d),
        binding_slots: wp.array(dtype=wp.int32),
        output_bindings: wp.array(dtype=wp.int32),
        source_offsets: wp.array(dtype=wp.int32),
        output_offsets: wp.array(dtype=wp.int32),
        mapping_offsets: wp.array(dtype=wp.int32),
        source_indices: wp.array2d(dtype=wp.int32),
        weights: wp.array2d(dtype=wp.float32),
        output_points: wp.fabricarrayarray(dtype=wp.vec3f),
    ):
        """Interpolate native nodes directly into plan-bound Fabric mesh point arrays.

        Fabric stores points in local space, so interpolation and world-to-local conversion are
        fused into this single launch.
        """
        i = wp.tid()
        binding = output_bindings[i]
        local = i - output_offsets[binding]
        mapping = mapping_offsets[binding]
        source = source_offsets[binding]
        point = wp.vec3f()
        if mapping < 0:
            point = input_points[source + local]
        else:
            row = mapping + local
            point = (
                weights[row, 0] * input_points[source + source_indices[row, 0]]
                + weights[row, 1] * input_points[source + source_indices[row, 1]]
                + weights[row, 2] * input_points[source + source_indices[row, 2]]
                + weights[row, 3] * input_points[source + source_indices[row, 3]]
            )
        slot = binding_slots[binding]
        world_to_local = wp.inverse(wp.transpose(wp.mat44f(world_matrices[slot])))
        output_points[slot][local] = wp.transform_point(world_to_local, point)
