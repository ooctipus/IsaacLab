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
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_mul, sample_uniform

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


def write_fixed_root_poses(env, roots: NewtonSelection, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    """Write fixed root link poses [m, xyzw] and notify only the changed worlds."""
    if len(env_ids) == 0:
        return
    requested = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    requested[env_ids] = torch.arange(len(env_ids), device=env.device)
    for part, worlds in roots.native_bindings:
        model = part.owner.model
        joint_ids = part.owner.root_joint_ids[part.dense_ids()]
        child_frames = wp.to_torch(model.joint_X_c)[joint_ids]
        local_poses = poses[requested[worlds].clamp(min=0), : part.width]
        pos, quat = combine_frame_transforms(
            local_poses[..., :3], local_poses[..., 3:], child_frames[..., :3], child_frames[..., 3:]
        )
        target = wp.to_torch(model.joint_X_p)
        active = (part.dense_active() & (requested[worlds] >= 0)[:, None]).unsqueeze(-1)
        target[joint_ids] = torch.where(active, torch.cat((pos, quat), dim=-1), target[joint_ids])
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
