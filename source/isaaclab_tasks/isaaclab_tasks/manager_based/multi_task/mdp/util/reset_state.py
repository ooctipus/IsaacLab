# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter-based per-env reset-state read/write helpers.

The generic helpers operate on caller-supplied :class:`ResetStateAdapter`
instances. Adapters own their tensor slice and the corresponding simulator
read/write calls, so the buffer and generic set/get path do not branch on
asset or command type.

- :func:`temporary_seed` — context manager that sets the IsaacSim torch
  seed within a ``with`` block and restores torch / cuda / numpy / python
  RNG state on exit. Used to make build-time sampling deterministic
  without polluting the global RNG.

Used by :class:`~.event_combinators.reset_accumulator` (state buffer
fill / draw) and the factory :func:`~..manager_based.multi_task.factory.mdp.observations.get_state` observation.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import warp as wp


class ResetStateAdapter(Protocol):
    """Adapter that owns one slice of a reset-state tensor."""

    def state_dim(self, env) -> int:
        """Return this adapter's reset-state slice width."""
        ...

    def get_state(self, env, env_ids, *, is_relative: bool = False) -> torch.Tensor:
        """Read this adapter's reset-state slice for ``env_ids``."""
        ...

    def set_state(self, env, states: torch.Tensor, env_ids: torch.Tensor, *, is_relative: bool = False) -> None:
        """Write this adapter's reset-state slice for ``env_ids``."""
        ...


@dataclass(frozen=True)
class ArticulationResetStateAdapter:
    """Reset-state adapter for one articulation asset."""

    asset_name: str

    def _asset(self, env):
        return env.scene._articulations[self.asset_name]

    def state_dim(self, env) -> int:
        """Return ``root_state(13) + joint_pos + joint_vel`` width."""
        articulation = self._asset(env)
        return 13 + 2 * articulation.num_joints

    def get_state(self, env, env_ids, *, is_relative: bool = False) -> torch.Tensor:
        """Read root state, joint positions, and joint velocities."""
        articulation = self._asset(env)
        root_state = wp.to_torch(articulation.data.root_state_w)[env_ids]
        if is_relative:
            root_state = root_state.clone()
            root_state[:, :3] -= env.scene.env_origins[env_ids]
        return torch.cat(
            [
                root_state,
                wp.to_torch(articulation.data.joint_pos)[env_ids],
                wp.to_torch(articulation.data.joint_vel)[env_ids],
            ],
            dim=-1,
        )

    def set_state(self, env, states: torch.Tensor, env_ids: torch.Tensor, *, is_relative: bool = False) -> None:
        """Write root state, joint positions, and joint velocities."""
        articulation = self._asset(env)
        root_state = states[:, :13]
        if is_relative:
            root_state = root_state.clone()
            root_state[:, :3] += env.scene.env_origins[env_ids]
        articulation.write_root_state_to_sim(root_state, env_ids=env_ids)
        n_joints = articulation.num_joints
        joint_position = states[:, 13 : 13 + n_joints]
        joint_velocity = states[:, 13 + n_joints : 13 + 2 * n_joints]
        articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)


@dataclass(frozen=True)
class RigidObjectResetStateAdapter:
    """Reset-state adapter for one rigid-object asset."""

    asset_name: str

    def _asset(self, env):
        return env.scene._rigid_objects[self.asset_name]

    def state_dim(self, env) -> int:
        """Return ``root_state(13)`` width."""
        return 13

    def get_state(self, env, env_ids, *, is_relative: bool = False) -> torch.Tensor:
        """Read root state for this rigid object."""
        rigid_object = self._asset(env)
        root_state = wp.to_torch(rigid_object.data.root_state_w)[env_ids]
        if is_relative:
            root_state = root_state.clone()
            root_state[:, :3] -= env.scene.env_origins[env_ids]
        return root_state

    def set_state(self, env, states: torch.Tensor, env_ids: torch.Tensor, *, is_relative: bool = False) -> None:
        """Write root state for this rigid object."""
        rigid_object = self._asset(env)
        root_state = states[:, :13]
        if is_relative:
            root_state = root_state.clone()
            root_state[:, :3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_state_to_sim(root_state, env_ids)


@dataclass(frozen=True)
class CallableResetStateAdapter:
    """Reset-state adapter backed by caller-provided get/set functions."""

    dim: int
    getter: Callable[[object, object, bool], torch.Tensor]
    setter: Callable[[object, torch.Tensor, torch.Tensor, bool], None]

    def state_dim(self, env) -> int:
        """Return the fixed slice width."""
        return self.dim

    def get_state(self, env, env_ids, *, is_relative: bool = False) -> torch.Tensor:
        """Read state through the caller-provided getter."""
        return self.getter(env, env_ids, is_relative)

    def set_state(self, env, states: torch.Tensor, env_ids: torch.Tensor, *, is_relative: bool = False) -> None:
        """Write state through the caller-provided setter."""
        self.setter(env, states, env_ids, is_relative)


def make_reset_state_adapters(env, keys: Sequence[str]) -> list[ResetStateAdapter]:
    """Create scene-asset adapters for ``keys`` using scene iteration order."""
    key_set = set(keys)
    adapters: list[ResetStateAdapter] = []
    found: set[str] = set()
    for name in env.scene._articulations:
        if name in key_set:
            adapters.append(ArticulationResetStateAdapter(name))
            found.add(name)
    for name in env.scene._rigid_objects:
        if name in key_set:
            adapters.append(RigidObjectResetStateAdapter(name))
            found.add(name)
    missing = key_set - found
    if missing:
        raise ValueError(f"Reset-state assets not found in scene: {sorted(missing)}")
    return adapters


def get_reset_state(
    env,
    env_ids,
    adapters: Sequence[ResetStateAdapter],
    is_relative: bool = False,
) -> torch.Tensor:
    """Read and concatenate reset-state slices from ``adapters``."""
    states = [adapter.get_state(env, env_ids, is_relative=is_relative) for adapter in adapters]
    if states:
        return torch.cat(states, dim=-1)
    num_envs = env.scene.env_origins[env_ids].shape[0]
    return torch.zeros(num_envs, 0, device=env.device)


def set_reset_state(
    env,
    states: torch.Tensor,
    env_ids: torch.Tensor,
    adapters: Sequence[ResetStateAdapter],
    is_relative: bool = False,
) -> None:
    """Split ``states`` by adapter slice width and write each slice."""
    offset = 0
    for adapter in adapters:
        width = adapter.state_dim(env)
        adapter.set_state(env, states[:, offset : offset + width], env_ids, is_relative=is_relative)
        offset += width
    if offset != states.shape[-1]:
        raise ValueError(f"Reset state width mismatch: consumed {offset} columns from {states.shape[-1]}.")


def pack_articulation_reset_state(
    root_pose: torch.Tensor,
    joint_pos: torch.Tensor,
    root_vel: torch.Tensor | None = None,
    joint_vel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack articulation tensors into ``root_state(13) + joint_pos + joint_vel`` layout.

    Args:
        root_pose: Root pose ``(..., 7)`` or full root state ``(..., 13)``.
        joint_pos: Joint positions.
        root_vel: Root velocity ``(..., 6)``. Defaults to zeros.
        joint_vel: Joint velocities. Defaults to zeros.

    Returns:
        Packed reset-state tensor.
    """
    if root_pose.shape[-1] == 13:
        root_state = root_pose.clone()
        if root_vel is not None:
            root_state[..., 7:13] = root_vel
    elif root_pose.shape[-1] == 7:
        root_state = torch.zeros(*root_pose.shape[:-1], 13, device=root_pose.device, dtype=root_pose.dtype)
        root_state[..., :7] = root_pose
        if root_vel is not None:
            root_state[..., 7:13] = root_vel
    else:
        raise ValueError(f"Expected root_pose last dimension 7 or 13, got {root_pose.shape[-1]}.")

    if joint_vel is None:
        joint_vel = torch.zeros_like(joint_pos)

    return torch.cat([root_state, joint_pos, joint_vel], dim=-1)


@contextmanager
def temporary_seed(seed: int, restore_numpy: bool = True, restore_python: bool = True):
    """Set the IsaacSim torch seed within a ``with`` block; restore on exit.

    Restores torch (CPU + all CUDA), numpy, and python ``random`` RNG state
    on exit. ``isaacsim.core.utils.torch.set_seed`` prints to stdout/stderr;
    this captures and discards that output to keep build logs clean.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state() if restore_numpy else None
    py_state = random.getstate() if restore_python else None

    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            import isaacsim.core.utils.torch as torch_utils

            torch_utils.set_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if np_state is not None:
            np.random.set_state(np_state)
        if py_state is not None:
            random.setstate(py_state)
