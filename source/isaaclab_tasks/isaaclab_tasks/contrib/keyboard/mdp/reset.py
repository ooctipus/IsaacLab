# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset writes and selected end-effector kinematics on raw Newton arrays."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
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
def _reset_kinematics(
    q: wp.array2d[float],
    root: wp.array2d[float],
    parents: wp.array[int],
    kinds: wp.array[int],
    columns: wp.array[int],
    joint_parent: wp.array[wp.transform],
    joint_child: wp.array[wp.transform],
    axes: wp.array[wp.vec3],
    selected: wp.array[int],
    offset: wp.vec3,
    poses: wp.array2d[wp.transform],
    jacobian: wp.array3d[float],
):
    row = wp.tid()
    root_pose = wp.transform(
        wp.vec3(root[row, 0], root[row, 1], root[row, 2]),
        wp.quat(root[row, 3], root[row, 4], root[row, 5], root[row, 6]),
    )
    for joint in range(parents.shape[0]):
        anchor = root_pose
        parent = parents[joint]
        if parent >= 0:
            anchor = poses[row, parent] * joint_parent[joint]
        motion = wp.transform_identity()
        column = columns[joint]
        if kinds[joint] == int(JointType.REVOLUTE):
            motion = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axes[joint], q[row, column]))
        elif kinds[joint] == int(JointType.PRISMATIC):
            motion = wp.transform(axes[joint] * q[row, column], wp.quat_identity())
        poses[row, joint] = (anchor * motion) * wp.transform_inverse(joint_child[joint])
    tip = wp.transform_point(poses[row, parents.shape[0] - 1], offset)
    for column in range(selected.shape[0]):
        joint = selected[column]
        anchor = poses[row, parents[joint]] * joint_parent[joint]
        axis = wp.transform_vector(anchor, axes[joint])
        linear, angular = axis, wp.vec3()
        if kinds[joint] == int(JointType.REVOLUTE):
            linear = wp.cross(axis, tip - wp.transform_get_translation(anchor))
            angular = axis
        for component in range(3):
            jacobian[row, component, column] = linear[component]
            jacobian[row, component + 3, column] = angular[component]


class ResetKinematics:
    """Compact scalar-chain reset workspace, independent of live physics state.

    Immutable topology is copied from one prepared world. Mutable storage contains
    only ancestor poses [m, xyzw] and a fingertip Jacobian [m/rad, rad/rad] for a
    reset batch. Joint coordinates and root poses belong to the caller's payload.
    The fixed root's child frame must be identity, matching keyboard snapshots.
    """

    def __init__(self, model, robot_coord_ids, ik_dof_ids, tip_body_id, capacity, offset):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Reset kinematics requires positive batch capacity.")
        arrays = self._compile(model, robot_coord_ids, ik_dof_ids, tip_body_id)
        self._topology = arrays
        self.device, self.capacity = model.device, capacity
        self.width = len(robot_coord_ids)
        self._offset = wp.vec3(*offset)
        dtypes = (int, int, int, wp.transform, wp.transform, wp.vec3, int)
        self._arrays = tuple(wp.array(value, dtype=dtype, device=self.device) for value, dtype in zip(arrays, dtypes))
        self._poses = wp.empty((capacity, len(arrays[0])), dtype=wp.transform, device=self.device)
        self._jacobian = wp.empty((capacity, 6, len(ik_dof_ids)), dtype=float, device=self.device)

    @staticmethod
    def _compile(model, robot_coord_ids, ik_dof_ids, tip_body_id):
        if model.world_count != 1:
            raise ValueError("Reset kinematics topology must come from one prepared world.")
        coords, dofs = np.asarray(robot_coord_ids), np.asarray(ik_dof_ids)
        if (
            coords.ndim != 1
            or dofs.ndim != 1
            or len(coords) == 0
            or len(dofs) == 0
            or len(set(coords.tolist())) != len(coords)
            or len(set(dofs.tolist())) != len(dofs)
        ):
            raise ValueError("Reset kinematics requires unique coordinate and DOF selections.")
        if (
            not np.issubdtype(coords.dtype, np.integer)
            or not np.issubdtype(dofs.dtype, np.integer)
            or np.any(coords < 0)
            or np.any(coords >= model.joint_coord_count)
            or np.any(dofs < 0)
            or np.any(dofs >= model.joint_dof_count)
            or not 0 <= tip_body_id < model.body_count
        ):
            raise ValueError("Reset kinematics selections are outside the prepared model.")
        child, parent = model.joint_child.numpy(), model.joint_parent.numpy()
        kinds, qs, ds = model.joint_type.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        owner = {int(body): joint for joint, body in enumerate(child)}
        chain, body = [], int(tip_body_id)
        while body >= 0:
            if body not in owner or len(chain) >= model.joint_count:
                raise ValueError("Reset tip must have an acyclic articulated ancestor chain.")
            joint = owner[body]
            if kinds[joint] not in (JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC):
                raise ValueError("Reset kinematics supports only fixed and scalar ancestor joints.")
            chain.append(joint)
            body = int(parent[joint])
        chain.reverse()
        xc = model.joint_X_c.numpy()[chain]
        if kinds[chain[0]] != JointType.FIXED or not np.array_equal(xc[0], [0, 0, 0, 0, 0, 0, 1]):
            raise ValueError("Reset kinematics requires a fixed root with an identity child frame.")
        local = {int(child[joint]): i for i, joint in enumerate(chain)}
        coord_columns = {int(coord): i for i, coord in enumerate(coords)}
        dof_joints = {int(ds[j]): i for i, j in enumerate(chain) if kinds[j] != JointType.FIXED}
        if any(int(qs[j]) not in coord_columns for j in chain if kinds[j] != JointType.FIXED):
            raise ValueError("Robot coordinates must include every scalar ancestor joint.")
        if any(int(dof) not in dof_joints for dof in dofs):
            raise ValueError("IK DOFs must be scalar ancestors of the selected tip.")
        axes = model.joint_axis.numpy()
        return (
            np.array([local.get(int(parent[j]), -1) for j in chain], dtype=np.int32),
            kinds[chain].copy(),
            np.array([coord_columns[int(qs[j])] if kinds[j] != JointType.FIXED else -1 for j in chain], dtype=np.int32),
            model.joint_X_p.numpy()[chain],
            xc,
            np.array([axes[ds[j]] if kinds[j] != JointType.FIXED else [0, 0, 0] for j in chain], dtype=np.float32),
            np.array([dof_joints[int(dof)] for dof in dofs], dtype=np.int32),
        )

    def validate_model(self, model, robot_coord_ids, ik_dof_ids, tip_body_id):
        """Reject a prototype whose compiled robot topology differs from this workspace."""
        other = self._compile(model, robot_coord_ids, ik_dof_ids, tip_body_id)
        if any(not np.array_equal(a, b) for a, b in zip(self._topology, other, strict=True)):
            raise ValueError("Keyboard prototypes must share identical reset robot kinematics.")

    def evaluate(self, q: torch.Tensor, root_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate compact coordinates [rad or m] and root poses [m, xyzw], without live writes."""
        count = len(q)
        if q.shape != (count, self.width) or root_pose.shape != (count, 7) or count > self.capacity:
            raise ValueError("Reset coordinate/root batch differs from the prepared layout.")
        if q.dtype != torch.float32 or root_pose.dtype != torch.float32 or q.device != root_pose.device:
            raise ValueError("Reset coordinates and root poses must be float32 on the same device.")
        q_wp, root_wp = wp.from_torch(q), wp.from_torch(root_pose)
        if q_wp.device != self.device:
            raise ValueError("Reset payload device differs from the prepared workspace.")
        if count:
            wp.launch(
                _reset_kinematics,
                count,
                [q_wp, root_wp, *self._arrays, self._offset, self._poses, self._jacobian],
                device=self.device,
            )
        return wp.to_torch(self._poses)[:count, -1], wp.to_torch(self._jacobian)[:count]


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
    stream = None
    if poses.is_cuda:
        stream = wp.get_stream(env.device)
        producer = torch.cuda.current_stream(env.device)
        if (stream.cuda_stream or 0) != producer.cuda_stream:
            stream = wp.stream_from_torch(producer)
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
        env.notify_model_changed(ModelFlags.JOINT_PROPERTIES, env_ids, root_poses_only=True)
        env.invalidate_fk(env_ids)


def sample_root_poses(poses: torch.Tensor, pose_range, velocity_range) -> torch.Tensor:
    """Sample fixed root poses [m, xyzw], preserving both original six-axis RNG draws."""
    if any(any(v != 0 for v in bounds) for bounds in velocity_range.values()):
        raise ValueError("The fixed keyboard root cannot have nonzero reset velocity.")
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in axes], device=poses.device)
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(poses), 6), device=poses.device)
    poses = poses.clone()
    poses[..., :3] += samples[:, None, :3]
    delta = quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    poses[..., 3:] = quat_mul(poses[..., 3:], delta[:, None, :].expand_as(poses[..., 3:]))
    # Preserve RNG consumption of the old zero-velocity sample.
    sample_uniform(0.0, 0.0, (len(poses), 6), device=poses.device)
    return poses


def reset_root_state_uniform(env, env_ids, roots, pose_range, velocity_range, *, defer_to_typing=False):
    """Randomize fixed roots, or consume the same draws before a complete typing reset.

    ``defer_to_typing`` is explicit: the caller must guarantee no intervening event
    observes the sampled pose and typing IK/snapshot reset replaces every root.
    Pre-solve reset keeps the default immediate write.
    """
    if defer_to_typing and env.cfg.commands.typing.reset.ik is None:
        raise ValueError("Deferred keyboard root writes require typing reset IK or complete snapshot replay.")
    ids = env.all_env_ids[env_ids]
    poses = sample_root_poses(roots.read_model("body_q")[ids], pose_range, velocity_range)
    if not defer_to_typing:
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
