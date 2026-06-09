# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-resident flat-tensor ring buffer for off-policy RL.

The buffer is pure storage with no knowledge of RL-specific structure
(observations, actions, rewards, goals). It stores flat tensors of shape
``[capacity, num_envs, data_dim]`` — the natural shape that vectorized envs
produce — and provides ``insert()`` for writing and direct ``data`` access
for reading. Consumers use index-based gathering to read what they need.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer on GPU.

    Stores transitions as flat tensors of shape ``[capacity, num_envs, data_dim]``.
    The caller is responsible for flattening structured data (obs, action, reward,
    done, extras) before inserting and unflattening after sampling.

    Args:
        capacity: Maximum number of timesteps stored per env before wrapping.
        num_envs: Number of parallel environments.
        data_dim: Dimensionality of each flattened transition.
        device: Torch device for the buffer tensor.
    """

    def __init__(self, capacity: int, num_envs: int, data_dim: int, device: str = "cpu") -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        self.data_dim = data_dim
        self.device = device

        self.data = torch.zeros(capacity, num_envs, data_dim, device=device)
        self._insert_pos = 0
        self._full = False

    @property
    def size(self) -> int:
        """Number of valid timesteps currently in the buffer (per env)."""
        return self.capacity if self._full else self._insert_pos

    def insert(self, chunk: torch.Tensor) -> None:
        """Insert a chunk of transitions.

        Args:
            chunk: Tensor of shape ``[T, num_envs, data_dim]`` where ``T`` is the
                number of timesteps to insert. ``T`` must be <= ``capacity``.
        """
        T = chunk.shape[0]
        assert self.capacity >= T, f"Chunk length {T} exceeds buffer capacity {self.capacity}"

        end = self._insert_pos + T
        if end <= self.capacity:
            self.data[self._insert_pos : end] = chunk
        else:
            # Wrap around: split into two slices.
            first = self.capacity - self._insert_pos
            self.data[self._insert_pos :] = chunk[:first]
            self.data[: T - first] = chunk[first:]
            self._full = True

        self._insert_pos = end % self.capacity
        if end >= self.capacity:
            self._full = True

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, num_envs={self.num_envs}, "
            f"data_dim={self.data_dim}, size={self.size}, device={self.device})"
        )
