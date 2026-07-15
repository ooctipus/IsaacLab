# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure Warp kernels for SMPL HumEnv coordinate construction."""

from __future__ import annotations

import warp as wp


@wp.func
def _smpl_axis_rotation(axis: wp.vec3, angle: float) -> wp.quat:
    half_angle = 0.5 * angle
    scale = wp.sin(half_angle)
    return wp.quat(axis[0] * scale, axis[1] * scale, axis[2] * scale, wp.cos(half_angle))


@wp.func
def _smpl_joint_rotation(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_axes: wp.array3d(dtype=wp.float32),
    frame: int,
    joint: int,
) -> wp.quat:
    coordinate = 7 + 3 * joint
    axis_0 = wp.vec3(coordinate_axes[joint, 0, 0], coordinate_axes[joint, 0, 1], coordinate_axes[joint, 0, 2])
    axis_1 = wp.vec3(coordinate_axes[joint, 1, 0], coordinate_axes[joint, 1, 1], coordinate_axes[joint, 1, 2])
    axis_2 = wp.vec3(coordinate_axes[joint, 2, 0], coordinate_axes[joint, 2, 1], coordinate_axes[joint, 2, 2])
    rotation_0 = _smpl_axis_rotation(axis_0, joint_q[frame, coordinate])
    rotation_1 = _smpl_axis_rotation(axis_1, joint_q[frame, coordinate + 1])
    rotation_2 = _smpl_axis_rotation(axis_2, joint_q[frame, coordinate + 2])
    return wp.mul(wp.mul(rotation_0, rotation_1), rotation_2)


@wp.func
def _smpl_joint_velocity_edge(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_axes: wp.array3d(dtype=wp.float32),
    source: int,
    destination: int,
    joint: int,
    inverse_dt: float,
) -> wp.vec3:
    source_rotation = _smpl_joint_rotation(joint_q, coordinate_axes, source, joint)
    destination_rotation = _smpl_joint_rotation(joint_q, coordinate_axes, destination, joint)
    relative = wp.mul(destination_rotation, wp.quat_inverse(source_rotation))
    relative_norm = wp.sqrt(wp.dot(relative, relative))
    relative = relative * (1.0 / wp.max(relative_norm, 1.0e-9))
    if relative[3] < 0.0:
        relative = relative * -1.0
    magnitude = wp.sqrt(relative[0] * relative[0] + relative[1] * relative[1] + relative[2] * relative[2])
    angle = 2.0 * wp.atan2(magnitude, relative[3])
    scale = 2.0
    if wp.abs(angle) > 1.0e-6:
        scale = angle / magnitude
    else:
        scale = 1.0 / (0.5 - angle * angle / 48.0)
    angular_velocity = wp.vec3(relative[0], relative[1], relative[2]) * (scale * inverse_dt)

    coordinate = 7 + 3 * joint
    axis_0 = wp.vec3(coordinate_axes[joint, 0, 0], coordinate_axes[joint, 0, 1], coordinate_axes[joint, 0, 2])
    axis_1_local = wp.vec3(coordinate_axes[joint, 1, 0], coordinate_axes[joint, 1, 1], coordinate_axes[joint, 1, 2])
    axis_2_local = wp.vec3(coordinate_axes[joint, 2, 0], coordinate_axes[joint, 2, 1], coordinate_axes[joint, 2, 2])
    rotation_0 = _smpl_axis_rotation(axis_0, joint_q[source, coordinate])
    axis_1 = wp.quat_rotate(rotation_0, axis_1_local)
    rotation_1 = _smpl_axis_rotation(axis_1_local, joint_q[source, coordinate + 1])
    axis_2 = wp.quat_rotate(wp.mul(rotation_0, rotation_1), axis_2_local)
    cross_12 = wp.cross(axis_1, axis_2)
    cross_02 = wp.cross(axis_0, axis_2)
    cross_01 = wp.cross(axis_0, axis_1)
    return wp.vec3(
        wp.dot(angular_velocity, cross_12) / wp.dot(axis_0, cross_12),
        wp.dot(angular_velocity, cross_02) / wp.dot(axis_1, cross_02),
        wp.dot(angular_velocity, cross_01) / wp.dot(axis_2, cross_01),
    )


@wp.kernel
def smpl_joint_velocity_canonical_warp(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_axes: wp.array3d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    segment_count: int,
    frame_count: int,
    joint_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Write destination-indexed ordered-D6 velocity edges."""
    frame, joint = wp.tid()
    if frame >= frame_count or joint >= joint_count:
        return
    low = int(0)
    high = segment_count
    while low + 1 < high:
        middle = (low + high) // 2
        if frame < clip_offsets[middle]:
            high = middle
        else:
            low = middle
    start = clip_offsets[low]
    source = frame - 1
    destination = frame
    if frame == start:
        source = start
        destination = start + 1
    velocity = _smpl_joint_velocity_edge(joint_q, coordinate_axes, source, destination, joint, 1.0 / step_seconds[low])
    output[frame, 6 + 3 * joint] = velocity[0]
    output[frame, 7 + 3 * joint] = velocity[1]
    output[frame, 8 + 3 * joint] = velocity[2]


@wp.kernel
def smpl_joint_velocity_stored_warp(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_axes: wp.array3d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int64),
    step_seconds: wp.array(dtype=wp.float32),
    segment_count: int,
    frame_count: int,
    joint_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Write source-indexed ordered-D6 velocity edges with repeated tails."""
    frame, joint = wp.tid()
    if frame >= frame_count or joint >= joint_count:
        return
    low = int(0)
    high = segment_count
    while low + 1 < high:
        middle = (low + high) // 2
        if frame < clip_offsets[middle]:
            high = middle
        else:
            low = middle
    stop = int(clip_offsets[low + 1])
    source = frame
    destination = frame + 1
    if frame == stop - 1:
        source = stop - 2
        destination = stop - 1
    velocity = _smpl_joint_velocity_edge(joint_q, coordinate_axes, source, destination, joint, 1.0 / step_seconds[low])
    output[frame, 3 * joint] = velocity[0]
    output[frame, 3 * joint + 1] = velocity[1]
    output[frame, 3 * joint + 2] = velocity[2]


@wp.func
def _euler_angle_wrap_once(value: float) -> float:
    """Wrap one Euler angle by at most one full turn."""
    pi = 3.141592653589793
    two_pi = 6.283185307179586
    if value > pi:
        value -= two_pi
    if value < -pi:
        value += two_pi
    return value


@wp.kernel
def time_select_euler_xyz_branches_segmented_warp(
    values: wp.array3d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int64),
    joint_count: int,
):
    """Apply HumEnv's coupled XYZ Euler branch recurrence independently per clip."""
    clip = wp.tid()
    start = wp.int32(offsets[clip])
    stop = wp.int32(offsets[clip + 1])
    frame = start + 1
    while frame < stop - 1:
        pass_index = int(0)
        active = int(1)
        while pass_index < 2 and active == 1:
            maximum = float(0.0)
            for joint in range(joint_count):
                for axis in range(3):
                    difference = wp.abs(values[frame, joint, axis] - values[frame - 1, joint, axis])
                    maximum = wp.max(maximum, difference)
            if maximum < 3.0:
                active = 0
            else:
                for joint in range(joint_count):
                    difference_x = wp.abs(values[frame, joint, 0] - values[frame - 1, joint, 0])
                    difference_y = wp.abs(values[frame, joint, 1] - values[frame - 1, joint, 1])
                    difference_z = wp.abs(values[frame, joint, 2] - values[frame - 1, joint, 2])
                    if difference_x + difference_y + difference_z >= 3.0:
                        values[frame, joint, 0] = _euler_angle_wrap_once(3.141592653589793 + values[frame, joint, 0])
                        values[frame, joint, 1] = _euler_angle_wrap_once(3.141592653589793 - values[frame, joint, 1])
                        values[frame, joint, 2] = _euler_angle_wrap_once(3.141592653589793 + values[frame, joint, 2])
                pass_index += 1
        frame += 1
