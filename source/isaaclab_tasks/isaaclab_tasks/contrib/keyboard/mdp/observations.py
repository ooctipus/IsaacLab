# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from ..newton_selection import NewtonSelection, pose_field_active, pose_field_read
from ..selection_contracts import require_count_per_world, require_same_world_domain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


@wp.kernel(enable_backward=False)
def _encode_slots(slots: wp.array2d[wp.int64], active: wp.array2d[bool], out: wp.array3d[float]):
    world, letter, key = wp.tid()
    out[world, letter, key] = float(slots[world, letter] == wp.int64(key) and active[world, key])


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _relative_key_positions(
    roots: Any,
    keys: Any,
    out: wp.array2d[wp.vec3],
):
    world, slot = wp.tid()
    position = wp.vec3(0.0)
    if pose_field_active(roots, world, 0) and pose_field_active(keys, world, slot):
        root = pose_field_read(roots, world, 0)
        q = wp.transform_get_rotation(root)
        q = wp.quat(-q[0], -q[1], -q[2], q[3]) / wp.max(wp.dot(q, q), 1.0e-9)
        relative = wp.transform_get_translation(pose_field_read(keys, world, slot)) - wp.transform_get_translation(root)
        imaginary = wp.vec3(q[0], q[1], q[2])
        cross = 2.0 * wp.cross(imaginary, relative)
        position = relative + q[3] * cross + wp.cross(imaginary, cross)
    out[world, slot] = position


def _slots_onehot(slots: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """One-hot a ``(num_envs, max_len)`` slot-id buffer (``-1`` marks empty) into ``(num_envs, max_len * num_keys)``.

    Empty/padding positions (slot ``-1``) become all-zero rows, so they contribute nothing.
    """
    onehot = torch.empty((*slots.shape, active.shape[1]), dtype=torch.float, device=slots.device)
    wp.launch(
        _encode_slots,
        onehot.shape,
        inputs=[wp.from_torch(slots), wp.from_torch(active)],
        outputs=[wp.from_torch(onehot)],
        device=str(slots.device),
    )
    return onehot.reshape(slots.shape[0], -1)


def target_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target word as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of that target key (all-zero past the word's length). Pair with
    the perception key-position map (same global slot order) so a gather/attention actor can recover where
    each target key is.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return _slots_onehot(command.target, command.key_joints.dense_active())


def typed_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Typed buffer as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of the key typed there so far (all-zero for not-yet-typed
    positions). Comparing it against :func:`target_keys_onehot` yields the typing progress and any
    mistake the agent must backspace.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return _slots_onehot(command.typed, command.key_joints.dense_active())


def joint_pos(env: ManagerBasedRLEnv, joints: NewtonSelection) -> torch.Tensor:
    """Selected joint coordinates [m or rad, depending on joint type]."""
    return joints.read_state("joint_q")


def joint_vel(env: ManagerBasedRLEnv, joints: NewtonSelection) -> torch.Tensor:
    """Selected joint velocities [m/s or rad/s, depending on joint type]."""
    return joints.read_state("joint_qd")


def key_positions_b(env: ManagerBasedRLEnv, keys: NewtonSelection, root: NewtonSelection) -> torch.Tensor:
    """Key positions [m] relative to exactly one robot root per world, in stable slot order."""
    require_same_world_domain(keys, root)
    require_count_per_world(root, 1)
    roots = root.pose_field("state", "body_q")
    key_poses = keys.pose_field("state", "body_q")
    positions = wp.empty(keys.dense_shape, dtype=wp.vec3, device=key_poses.sources.device)
    wp.launch(
        _relative_key_positions,
        keys.dense_shape,
        inputs=[roots, key_poses],
        outputs=[positions],
        device=positions.device,
    )
    return wp.to_torch(positions).flatten(1)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous policy actions with episode-excluded DOFs cleared."""
    active = env.action_manager.get_term("action").dofs.dense_active()
    return torch.where(active, env.action_manager.action, 0.0)


def robot_state(
    env: ManagerBasedRLEnv, roots: NewtonSelection, joints: NewtonSelection, dofs: NewtonSelection
) -> torch.Tensor:
    """One record per arm: world root pose (wxyz), q, qd and previous action.

    The policy gathers present arms and constructs each arm's perspective. The
    task stores each root and joint state once, including zeros for absent arms.
    """
    pose = roots.read_state("body_q")
    shape = (*pose.shape[:2], 6)
    return torch.cat(
        (
            pose[..., :3],
            pose[..., (6, 3, 4, 5)],
            joints.read_state("joint_q").reshape(shape),
            dofs.read_state("joint_qd").reshape(shape),
            last_action(env).reshape(shape),
        ),
        dim=-1,
    )


def selection_active(env: ManagerBasedRLEnv, selection: NewtonSelection) -> torch.Tensor:
    """Presence of each selected entity in the world's current prototype."""
    return selection.dense_active().float()


def key_positions_w(env: ManagerBasedRLEnv, keys: NewtonSelection) -> torch.Tensor:
    """World-space key positions, stored once regardless of the number of arms."""
    return keys.read_state("body_q")[..., :3]
