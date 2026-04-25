# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env reset-state read/write helpers.

Two pure helpers and one context manager:

- :func:`get_reset_state` — extract the per-env state slab (root pose +
  joint pos/vel for articulations, root pose for rigid objects) from a
  scene, optionally as env-origin-relative coordinates.
- :func:`set_reset_state` — write the same slab back into the scene,
  reversing the env-origin offset on read.
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
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np
import torch
import warp as wp


def set_reset_state(env, states: torch.Tensor, env_ids: torch.Tensor, keys: list[str], is_relative: bool = False):
    """Write a per-env state slab back into the scene.

    ``states`` is laid out as the concatenation of (root_state[13] +
    joint_pos[J] + joint_vel[J]) for each named articulation followed by
    root_state[13] for each named rigid object. ``is_relative=True`` adds
    the env origin to the root position before writing.
    """
    idx = 0
    for name, articulation in env.scene._articulations.items():
        if name in keys:
            root_state = states[:, idx : idx + 13].clone()
            if is_relative:
                root_state[:, :3] += env.scene.env_origins[env_ids]
            articulation.write_root_state_to_sim(root_state, env_ids=env_ids)
            n_j = articulation.num_joints
            joint_position = states[:, idx + 13 : idx + 13 + n_j].clone()
            joint_velocity = states[:, idx + 13 + n_j : idx + 13 + 2 * n_j].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            idx += 13 + 2 * n_j
    for name, rigid_object in env.scene._rigid_objects.items():
        if name in keys:
            root_state = states[:, idx : idx + 13].clone()
            if is_relative:
                root_state[:, :3] += env.scene.env_origins[env_ids]
            rigid_object.write_root_state_to_sim(root_state, env_ids)
            idx += 13


def get_reset_state(env, env_id: torch.Tensor, keys: list[str], is_relative=False):
    """Extract the per-env state slab from the scene.

    See :func:`set_reset_state` for the layout. ``is_relative=True``
    subtracts the env origin from the root position so the returned slab
    is suitable for cross-env replay (the buffer-pattern used by
    :class:`~.event_combinators.reset_accumulator`).
    """
    states = []
    for name, articulation in env.scene._articulations.items():
        if name in keys:
            state = wp.to_torch(articulation.data.root_state_w)[env_id].clone()
            if is_relative:
                state[:, :3] -= env.scene.env_origins[env_id]
            states.append(state)
            states.append(wp.to_torch(articulation.data.joint_pos)[env_id].clone())
            states.append(wp.to_torch(articulation.data.joint_vel)[env_id].clone())
    for name, rigid_object in env.scene._rigid_objects.items():
        if name in keys:
            state = wp.to_torch(rigid_object.data.root_state_w)[env_id].clone()
            if is_relative:
                state[:, :3] -= env.scene.env_origins[env_id]
            states.append(state)
    return torch.cat(states, dim=-1)


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
