# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Request-driven geometry conversion kernels for SceneData."""

from __future__ import annotations

import warp as wp


@wp.func
def _body_point(
    points: wp.array2d(dtype=wp.vec3f),
    body: int,
    local: int,
    mapping: int,
    source_indices: wp.array2d(dtype=wp.int32),
    weights: wp.array2d(dtype=wp.float32),
) -> wp.vec3f:
    if mapping < 0:
        return points[body, local]
    row = mapping + local
    return (
        weights[row, 0] * points[body, source_indices[row, 0]]
        + weights[row, 1] * points[body, source_indices[row, 1]]
        + weights[row, 2] * points[body, source_indices[row, 2]]
        + weights[row, 3] * points[body, source_indices[row, 3]]
    )


@wp.kernel(enable_backward=False)
def body_points_to_points_kernel(
    input_points: wp.array2d(dtype=wp.vec3f),
    binding_ids: wp.array(dtype=wp.int32),
    source_offsets: wp.array(dtype=wp.int32),
    source_counts: wp.array(dtype=wp.int32),
    output_points: wp.array(dtype=wp.vec3f),
):
    """Gather one native padded body array into canonical plan order."""
    body = wp.tid()
    binding = binding_ids[body]
    offset = source_offsets[binding]
    for local in range(source_counts[binding]):
        output_points[offset + local] = input_points[body, local]


@wp.kernel(enable_backward=False)
def body_points_to_mesh_points_kernel(
    input_points: wp.array2d(dtype=wp.vec3f),
    binding_ids: wp.array(dtype=wp.int32),
    output_offsets: wp.array(dtype=wp.int32),
    output_counts: wp.array(dtype=wp.int32),
    mapping_offsets: wp.array(dtype=wp.int32),
    source_indices: wp.array2d(dtype=wp.int32),
    weights: wp.array2d(dtype=wp.float32),
    output_points: wp.array(dtype=wp.vec3f),
):
    """Gather and interpolate native padded bodies directly into a flat mesh output."""
    body = wp.tid()
    binding = binding_ids[body]
    output = output_offsets[binding]
    mapping = mapping_offsets[binding]
    for local in range(output_counts[binding]):
        output_points[output + local] = _body_point(input_points, body, local, mapping, source_indices, weights)


@wp.kernel(enable_backward=False)
def body_points_to_fabric_mesh_points_kernel(
    input_points: wp.array2d(dtype=wp.vec3f),
    binding_ids: wp.array(dtype=wp.int32),
    world_matrices: wp.fabricarray(dtype=wp.mat44d),
    binding_slots: wp.array(dtype=wp.int32),
    output_counts: wp.array(dtype=wp.int32),
    mapping_offsets: wp.array(dtype=wp.int32),
    source_indices: wp.array2d(dtype=wp.int32),
    weights: wp.array2d(dtype=wp.float32),
    output_points: wp.fabricarrayarray(dtype=wp.vec3f),
):
    """Gather, interpolate, and transform native bodies directly into Fabric mesh arrays."""
    body = wp.tid()
    binding = binding_ids[body]
    slot = binding_slots[binding]
    mapping = mapping_offsets[binding]
    world_to_local = wp.inverse(wp.transpose(wp.mat44f(world_matrices[slot])))
    for local in range(output_counts[binding]):
        point = _body_point(input_points, body, local, mapping, source_indices, weights)
        output_points[slot][local] = wp.transform_point(world_to_local, point)


@wp.func
def _cable_point(
    curve_shape_offsets: wp.array(dtype=wp.int32),
    curve_segment_counts: wp.array(dtype=wp.int32),
    shape_ids: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
    shape_transform: wp.array(dtype=wp.transformf),
    shape_scale: wp.array(dtype=wp.vec3f),
    curve: int,
    point: int,
) -> wp.vec3f:
    offset = curve_shape_offsets[curve]
    segment_count = curve_segment_counts[curve]
    if point == 0:
        shape = shape_ids[offset]
        shape_q = wp.transform_multiply(body_q[shape_body[shape]], shape_transform[shape])
        return wp.transform_point(shape_q, wp.vec3f(0.0, 0.0, -shape_scale[shape][1]))
    if point == segment_count:
        shape = shape_ids[offset + segment_count - 1]
        shape_q = wp.transform_multiply(body_q[shape_body[shape]], shape_transform[shape])
        return wp.transform_point(shape_q, wp.vec3f(0.0, 0.0, shape_scale[shape][1]))
    left_shape = shape_ids[offset + point - 1]
    left_q = wp.transform_multiply(body_q[shape_body[left_shape]], shape_transform[left_shape])
    left_w = wp.transform_point(left_q, wp.vec3f(0.0, 0.0, shape_scale[left_shape][1]))
    right_shape = shape_ids[offset + point]
    right_q = wp.transform_multiply(body_q[shape_body[right_shape]], shape_transform[right_shape])
    right_w = wp.transform_point(right_q, wp.vec3f(0.0, 0.0, -shape_scale[right_shape][1]))
    return 0.5 * (left_w + right_w)


@wp.kernel(enable_backward=False)
def cable_points_to_points_kernel(
    curve_shape_offsets: wp.array(dtype=wp.int32),
    curve_segment_counts: wp.array(dtype=wp.int32),
    binding_ids: wp.array(dtype=wp.int32),
    output_offsets: wp.array(dtype=wp.int32),
    shape_ids: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
    shape_transform: wp.array(dtype=wp.transformf),
    shape_scale: wp.array(dtype=wp.vec3f),
    output_points: wp.array(dtype=wp.vec3f),
):
    """Derive Newton cable endpoints directly into a requested flat output."""
    curve = wp.tid()
    output = output_offsets[binding_ids[curve]]
    for point in range(curve_segment_counts[curve] + 1):
        output_points[output + point] = _cable_point(
            curve_shape_offsets,
            curve_segment_counts,
            shape_ids,
            shape_body,
            body_q,
            shape_transform,
            shape_scale,
            curve,
            point,
        )


@wp.kernel(enable_backward=False)
def cable_points_to_fabric_mesh_points_kernel(
    curve_shape_offsets: wp.array(dtype=wp.int32),
    curve_segment_counts: wp.array(dtype=wp.int32),
    binding_ids: wp.array(dtype=wp.int32),
    shape_ids: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
    shape_transform: wp.array(dtype=wp.transformf),
    shape_scale: wp.array(dtype=wp.vec3f),
    world_matrices: wp.fabricarray(dtype=wp.mat44d),
    binding_slots: wp.array(dtype=wp.int32),
    output_points: wp.fabricarrayarray(dtype=wp.vec3f),
):
    """Derive and transform Newton cable endpoints directly into Fabric curve arrays."""
    curve = wp.tid()
    slot = binding_slots[binding_ids[curve]]
    world_to_local = wp.inverse(wp.transpose(wp.mat44f(world_matrices[slot])))
    for point in range(curve_segment_counts[curve] + 1):
        point_w = _cable_point(
            curve_shape_offsets,
            curve_segment_counts,
            shape_ids,
            shape_body,
            body_q,
            shape_transform,
            shape_scale,
            curve,
            point,
        )
        output_points[slot][point] = wp.transform_point(world_to_local, point_w)
