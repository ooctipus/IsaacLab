# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset writes and selected end-effector kinematics on raw Newton arrays."""

from __future__ import annotations

from dataclasses import MISSING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import JointType, ModelFlags

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
def _tip_jacobian(
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    joint_parent: wp.array[int],
    joint_type: wp.array[int],
    joint_axis: wp.array[wp.vec3],
    joints: wp.array2d[int],
    dofs: wp.array2d[int],
    bodies: wp.array2d[int],
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
        out[world, row, column] = linear[row]
        out[world, row + 3, column] = angular[row]


def tip_jacobian(ik: KeyboardResetIKCfg, out: wp.array) -> torch.Tensor:
    """Compute only the selected fingertip Jacobian [m/rad, rad/rad] without a padded articulation view."""
    model, state = NewtonManager.get_model(), NewtonManager.get_state()
    shape = ik.dofs.dense_ids().shape
    wp.launch(
        _tip_jacobian,
        dim=shape,
        inputs=[
            state.body_q,
            model.joint_X_p,
            model.joint_parent,
            model.joint_type,
            model.joint_axis,
            ik.dofs.joint_ids.reshape(shape),
            ik.dofs.ids.reshape(shape),
            ik.body.ids.reshape((shape[0], 1)),
            wp.vec3(*ik.tip_offset),
        ],
        outputs=[out],
        device=model.device,
    )
    return wp.to_torch(out)


def write_fixed_root_poses(env, roots: NewtonSelection, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    """Write fixed root link poses [m, xyzw] and notify only the changed worlds."""
    model = NewtonManager.get_model()
    body_ids = roots.dense_ids()[env_ids]
    joint_ids = roots.owner.root_joint_ids[body_ids]
    child_frames = wp.to_torch(model.joint_X_c)[joint_ids]
    pos, quat = combine_frame_transforms(poses[..., :3], poses[..., 3:], child_frames[..., :3], child_frames[..., 3:])
    target = wp.to_torch(model.joint_X_p)
    active = roots.dense_active()[env_ids, :, None]
    target[joint_ids] = torch.where(active, torch.cat((pos, quat), dim=-1), target[joint_ids])
    env._property_world_mask.zero_()
    wp.to_torch(env._property_world_mask)[env_ids] = True
    NewtonManager.notify_model_changed(ModelFlags.JOINT_PROPERTIES, world_mask=env._property_world_mask)
    NewtonManager.invalidate_fk(env_ids=wp.from_torch(env_ids.to(torch.int32)))


def reset_root_state_uniform(env, env_ids, roots, pose_range, velocity_range):
    """Randomize fixed root poses [m, rad] with the original task's random-draw ordering."""
    if any(any(v != 0 for v in bounds) for bounds in velocity_range.values()):
        raise ValueError("The fixed keyboard root cannot have nonzero reset velocity.")
    ids = env.scene._ALL_INDICES[env_ids]
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in axes], device=env.device)
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(ids), 6), device=env.device)
    poses = wp.to_torch(NewtonManager.get_model().body_q)[roots.dense_ids()[ids]].clone()
    poses[..., :3] += samples[:, None, :3]
    delta = quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    poses[..., 3:] = quat_mul(poses[..., 3:], delta[:, None, :].expand_as(poses[..., 3:]))
    # Preserve RNG consumption of the old zero-velocity sample.
    sample_uniform(0.0, 0.0, (len(ids), 6), device=env.device)
    write_fixed_root_poses(env, roots, ids, poses)


def capture_reset_state(env, env_ids, roots, coords, dofs) -> torch.Tensor:
    """Pack root poses relative to world origins, coordinates, and velocities for snapshot replay."""
    state = NewtonManager.get_state()
    poses = roots.dense(state.body_q)[env_ids].clone()
    poses[..., :3] -= env.scene.env_origins[env_ids, None, :]
    return torch.cat(
        (poses.flatten(1), coords.dense(state.joint_q)[env_ids], dofs.dense(state.joint_qd)[env_ids]), dim=1
    )


def restore_reset_state(env, snapshot, env_ids, roots, coords, dofs) -> None:
    """Restore selected worlds from a snapshot and leave other worlds' state untouched."""
    width = roots.counts[0] * 7
    poses = snapshot[:, :width].reshape(-1, roots.counts[0], 7).clone()
    poses[..., :3] += env.scene.env_origins[env_ids, None, :]
    state = NewtonManager.get_state()
    n = coords.counts[0]
    q_ids, qd_ids = coords.dense_ids()[env_ids], dofs.dense_ids()[env_ids]
    q, qd = wp.to_torch(state.joint_q), wp.to_torch(state.joint_qd)
    q[q_ids] = torch.where(coords.dense_active()[env_ids], snapshot[:, width : width + n], q[q_ids])
    qd[qd_ids] = torch.where(dofs.dense_active()[env_ids], snapshot[:, width + n :], qd[qd_ids])
    write_fixed_root_poses(env, roots, env_ids, poses)
