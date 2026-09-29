# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset writes and selected end-effector kinematics on raw Newton arrays."""

from __future__ import annotations

from dataclasses import MISSING

import torch
import warp as wp
from newton import JointType, Model, ModelFlags, State

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from ..newton_selection import NewtonSelection, NewtonSelectorCfg


@configclass
class KeyboardResetIKCfg:
    """Selected scalar arm joints and the link-local fingertip offset [m]."""

    joints: NewtonSelectorCfg = MISSING
    dofs: NewtonSelectorCfg = MISSING
    body: NewtonSelectorCfg = MISSING
    tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@wp.kernel
def _select_reset_articulations(
    articulation_world: wp.array[int],
    logical_worlds: wp.array[wp.int64],
    requested: wp.array[bool],
    selected: wp.array[bool],
):
    articulation = wp.tid()
    world = articulation_world[articulation]
    selected[articulation] = False
    if world >= 0:
        selected[articulation] = requested[int(logical_worlds[world])]


def prepare_reset_kinematics(roots, env_ids: torch.Tensor) -> tuple[tuple[Model, State, wp.array], ...]:
    """Prepare solve-local FK masks for the requested logical worlds, excluding global articulations."""
    parts = roots.native_bindings
    requested = torch.zeros(sum(len(worlds) for _, worlds in parts), dtype=torch.bool, device=env_ids.device)
    requested[env_ids] = True
    bindings = []
    for part, worlds in parts:
        model, state = part.owner.model, part.owner.state
        mask = wp.empty(model.articulation_count, dtype=wp.bool, device=model.device)
        wp.launch(
            _select_reset_articulations,
            dim=model.articulation_count,
            inputs=[model.articulation_world, wp.from_torch(worlds), wp.from_torch(requested), mask],
            device=model.device,
        )
        bindings.append((model, state, mask))
    return tuple(bindings)


@wp.kernel
def _tip_jacobian(
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    joint_parent: wp.array[int],
    joint_type: wp.array[int],
    joint_axis: wp.array[wp.vec3],
    joints: wp.array2d[int],
    dofs: wp.array2d[int],
    bodies: wp.array2d[int],
    logical_worlds: wp.array[wp.int64],
    offset: wp.vec3,
    out: wp.array3d[float],
):
    world, column = wp.tid()
    joint = joints[world, column]
    frame = joint_X_p[joint]
    parent = joint_parent[joint]
    if parent >= 0:
        frame = body_q[parent] * frame
    axis = wp.transform_vector(frame, joint_axis[dofs[world, column]])
    tip = wp.transform_point(body_q[bodies[world, 0]], offset)
    linear = axis
    angular = wp.vec3(0.0)
    if joint_type[joint] == int(JointType.REVOLUTE):
        linear = wp.cross(axis, tip - wp.transform_get_translation(frame))
        angular = axis
    for row in range(3):
        out[logical_worlds[world], row, column] = linear[row]
        out[logical_worlds[world], row + 3, column] = angular[row]


def tip_jacobian(ik: KeyboardResetIKCfg, out: wp.array) -> torch.Tensor:
    """Compute only the selected fingertip Jacobian [m/rad, rad/rad] without a padded articulation view."""
    for (dofs, worlds), (body, _) in zip(ik.dofs.native_bindings, ik.body.native_bindings, strict=True):
        model, state = dofs.owner.model, dofs.owner.state
        shape = dofs.dense_ids().shape
        wp.launch(
            _tip_jacobian,
            dim=shape,
            inputs=[
                state.body_q,
                model.joint_X_p,
                model.joint_parent,
                model.joint_type,
                model.joint_axis,
                dofs.joint_ids.reshape(shape),
                dofs.ids.reshape(shape),
                body.ids.reshape((shape[0], 1)),
                wp.from_torch(worlds),
                wp.vec3(*ik.tip_offset),
            ],
            outputs=[out],
            device=model.device,
        )
    return wp.to_torch(out)


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _write_fixed_root_poses(
    requested: wp.array[wp.int64],
    worlds: wp.array[wp.int64],
    bodies: wp.array2d[int],
    active: wp.array2d[bool],
    root_joints: wp.array[wp.int64],
    child: wp.array[wp.transform],
    poses: wp.array3d[float],
    parent: wp.array[wp.transform],
):
    world, column = wp.tid()
    row = requested[worlds[world]]
    if row >= 0 and active[world, column]:
        joint = root_joints[bodies[world, column]]
        pose = wp.transform(
            wp.vec3(poses[row, column, 0], poses[row, column, 1], poses[row, column, 2]),
            wp.quat(poses[row, column, 3], poses[row, column, 4], poses[row, column, 5], poses[row, column, 6]),
        )
        # Reset IK amplifies quaternion reassociation: preserve the task's Torch
        # frame composition order, with contraction disabled only for this kernel.
        other = child[joint]
        q1, q2 = wp.transform_get_rotation(pose), wp.transform_get_rotation(other)
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        quat = wp.quat(
            qq - xx + (x1 + w1) * (x2 + w2),
            qq - yy + (w1 - x1) * (y2 + z2),
            qq - zz + (z1 + y1) * (w2 - x2),
            qq - ww + (z1 - y1) * (y2 - z2),
        )
        xyz = wp.vec3(x1, y1, z1)
        offset = wp.transform_get_translation(other)
        cross = wp.cross(xyz, offset) * 2.0
        rotated = offset + w1 * cross + wp.cross(xyz, cross)
        parent[joint] = wp.transform(wp.transform_get_translation(pose) + rotated, quat)


def write_fixed_root_poses(env, roots: NewtonSelection, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    """Write participating fixed root poses [m, xyzw], preserving the caller's stream ordering."""
    if len(env_ids) == 0:
        return
    requested = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    requested[env_ids] = torch.arange(len(env_ids), device=env.device)
    requested_wp, poses_wp = wp.from_torch(requested), wp.from_torch(poses)
    stream = wp.stream_from_torch(torch.cuda.current_stream(env.device)) if poses.is_cuda else None
    with wp.ScopedStream(stream, sync_exit=True):
        for part, worlds in roots.native_bindings:
            model = part.owner.model
            shape = part.dense_ids().shape
            wp.launch(
                _write_fixed_root_poses,
                dim=shape,
                inputs=[
                    requested_wp,
                    wp.from_torch(worlds),
                    part.ids.reshape(shape),
                    part.active.reshape(shape),
                    wp.from_torch(part.owner.root_joint_ids),
                    model.joint_X_c,
                    poses_wp,
                ],
                outputs=[model.joint_X_p],
                device=model.device,
            )
        env.notify_model_changed(ModelFlags.JOINT_PROPERTIES, env_ids)
        env.invalidate_fk(env_ids)


def reset_root_state_uniform(env, env_ids, roots, pose_range, velocity_range):
    """Randomize fixed root poses [m, rad] with the original task's random-draw ordering."""
    if any(any(v != 0 for v in bounds) for bounds in velocity_range.values()):
        raise ValueError("The fixed keyboard root cannot have nonzero reset velocity.")
    ids = env.all_env_ids[env_ids]
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in axes], device=env.device)
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(ids), 6), device=env.device)
    poses = roots.read_model("body_q")[ids].clone()
    poses[..., :3] += samples[:, None, :3]
    delta = quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    poses[..., 3:] = quat_mul(poses[..., 3:], delta[:, None, :].expand_as(poses[..., 3:]))
    # Preserve RNG consumption of the old zero-velocity sample.
    sample_uniform(0.0, 0.0, (len(ids), 6), device=env.device)
    write_fixed_root_poses(env, roots, ids, poses)


def capture_reset_state(env, env_ids, roots, coords, dofs) -> torch.Tensor:
    """Pack root poses relative to world origins, coordinates, and velocities for snapshot replay."""
    poses = roots.read_state("body_q")[env_ids].clone()
    poses[..., :3] -= env.env_origins[env_ids, None, :]
    return torch.cat(
        (poses.flatten(1), coords.read_state("joint_q")[env_ids], dofs.read_state("joint_qd")[env_ids]), dim=1
    )


def restore_reset_state(env, snapshot, env_ids, roots, coords, dofs) -> None:
    """Restore selected worlds from a snapshot and leave other worlds' state untouched."""
    width = roots.width * 7
    poses = snapshot[:, :width].reshape(-1, roots.width, 7).clone()
    poses[..., :3] += env.env_origins[env_ids, None, :]
    n = coords.width
    coords.write_state("joint_q", snapshot[:, width : width + n], env_ids)
    dofs.write_state("joint_qd", snapshot[:, width + n :], env_ids)
    write_fixed_root_poses(env, roots, env_ids, poses)
