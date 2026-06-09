# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Index-based Hindsight Experience Replay.

Implements goal relabeling (Andrychowicz et al. 2017) as an index-only
transform: instead of materializing relabeled trajectories, HER computes
``(t, e, future_t)`` index triples that point into the raw replay buffer.
Consumers gather the data they need at SGD time.

This eliminates the O(T^2) probability matrix from the old window-based
approach and enables training on the full buffer without intermediate copies.
"""

from __future__ import annotations

from typing import Any

import torch

from ..storage import ReplayBuffer


class HindsightRelabeling:
    """Index-based Hindsight Experience Replay.

    For each sampled ``(t, e)`` transition, samples a future timestep
    ``future_t > t`` within the same episode using a geometric distribution
    with parameter ``gamma``. Episode boundaries are detected via the seed
    column in the replay buffer (equal seeds = same episode).

    The episode boundary scan runs once per :meth:`update_episode_boundaries`
    call and is cached until the next call.

    Args:
        gamma: Geometric discount for future-timestep sampling [unitless].
            Higher values favor more distant futures.
        goal_start_idx: Start index of the achieved-goal slice within the
            state portion of the observation.
        goal_end_idx: End index (exclusive) of the achieved-goal slice.
    """

    def __init__(
        self,
        gamma: float,
        goal_start_idx: int,
        goal_end_idx: int,
        obs_dim: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.goal_start_idx = goal_start_idx
        self.goal_end_idx = goal_end_idx
        self.obs_dim = obs_dim

        self._episode_end: torch.Tensor | None = None
        # Lazily allocated pre-allocated index buffers.
        self._flat_buf: torch.Tensor | None = None
        self._geom_buf: torch.Tensor | None = None
        self._last_oversample_n: int = 0

    def update_episode_boundaries(self, buffer: ReplayBuffer, seed_col: int) -> None:
        """Compute episode-end indices via reverse scan on the seed column.

        For each ``(t, e)``, ``episode_end[t, e]`` is the last timestep
        index (inclusive) in the buffer that shares the same seed value.
        This is computed in one reverse pass with no O(T^2) matrix.

        Args:
            buffer: The replay buffer.
            seed_col: Column index of the seed field within ``data_dim``.
        """
        size = buffer.size
        if size < 2:
            self._episode_end = None
            return

        seeds = buffer.data[:size, :, seed_col]

        # Detect where the seed changes between consecutive timesteps.
        # seed_changes[t, e] = True means timestep t is the LAST step of its
        # episode (seed differs at t+1, or t is the last valid position).
        seed_changes = torch.zeros(size, buffer.num_envs, dtype=torch.bool, device=buffer.device)
        seed_changes[:-1] = seeds[:-1] != seeds[1:]
        seed_changes[-1] = True

        # For each (t, e), episode_end[t, e] = smallest t' >= t where
        # seed_changes[t', e] is True.  Computed via reverse cummin on a
        # tensor where boundary positions hold their own index and
        # non-boundary positions hold a large sentinel.
        timesteps = torch.arange(size, device=buffer.device).unsqueeze(1)
        sentinel = size  # any value > all valid indices
        boundary_or_sentinel = torch.where(seed_changes, timesteps, sentinel)

        # Reverse cummin: flip → cummin → flip back.
        flipped = boundary_or_sentinel.flip(0)
        episode_end_flipped, _ = flipped.cummin(dim=0)
        self._episode_end = episode_end_flipped.flip(0).int()

    def sample_indices(
        self,
        buffer: ReplayBuffer,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``(t, e, future_t)`` index triples.

        Each triple points into the replay buffer: ``t`` is the current
        timestep, ``e`` is the env index, and ``future_t`` is a future
        timestep in the same episode sampled from a geometric distribution.

        No data is copied — only integer indices are returned.

        Args:
            buffer: The replay buffer (read-only).
            num_samples: Number of index triples to return (= batch_size).

        Returns:
            Tuple of ``(t, e, future_t)``, each of shape ``[num_samples]``.
        """
        assert self._episode_end is not None, (
            "Call update_episode_boundaries() before sample_indices()."
        )
        size = buffer.size
        device = buffer.device
        num_envs = buffer.num_envs

        # Lazily allocate (or resize) pre-allocated index buffers.
        n = num_samples * 2
        if self._flat_buf is None or self._last_oversample_n != n:
            self._flat_buf = torch.empty(n, dtype=torch.long, device=device)
            self._geom_buf = torch.empty(num_samples, device=device)
            self._last_oversample_n = n

        # In-place random generation (no allocation).
        self._flat_buf.random_(0, (size - 1) * num_envs)
        t_all = self._flat_buf // num_envs
        e_all = self._flat_buf % num_envs
        horizon_all = self._episode_end[t_all, e_all].long() - t_all

        valid = (horizon_all > 0).nonzero(as_tuple=True)[0][:num_samples]
        t = t_all[valid]
        e = e_all[valid]
        horizon = horizon_all[valid]

        # In-place geometric sampling (no allocation).
        actual = t.shape[0]
        self._geom_buf[:actual].geometric_(1.0 - self.gamma)
        k = self._geom_buf[:actual].long().clamp(min=1)
        k = torch.min(k, horizon)
        future_t = t + k

        return t.int(), e.int(), future_t.int()


def resolve_her_config(
    alg_cfg: dict,
    obs: Any,
    obs_groups: dict[str, list[str]],
) -> dict:
    """Resolve HER configuration by deriving goal slice indices from obs groups.

    Mirrors :func:`~rsl_rl.extensions.resolve_rnd_config` — called at algorithm
    construction time to fill in runtime-derived fields.

    The convention: observation groups are concatenated in **sorted key order**
    to form the flat obs vector. The ``target_state`` group (default
    ``"target_state"``) is the commanded-goal suffix. The ``current_state``
    group (default ``"current_state"``) is the achieved-state slice that HER
    relabels from.

    ``obs_dim`` = total flat dim MINUS the target_state group dim.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs: Observation TensorDict from the environment.
        obs_groups: Resolved observation groups (from :func:`resolve_obs_groups`).
            Keys are set names like ``"actor"`` / ``"critic"``; values are lists
            of obs-group names. We use the first set to determine the group list.

    Returns:
        The updated algorithm configuration dictionary.
    """
    her_cfg = alg_cfg.get("her_cfg")
    if her_cfg is None:
        return alg_cfg

    if obs_groups:
        first_set = next(iter(obs_groups.values()))
        all_groups = sorted(first_set)
    else:
        all_groups = sorted(obs.keys())

    goal_group = her_cfg.get("target_state", "target_state")
    achieved_group = her_cfg.get("current_state", "current_state")

    from math import prod

    offset = 0
    goal_start = None
    goal_end = None
    obs_dim = 0
    for group in all_groups:
        # Flat dim = product of all non-batch dims, so multi-dim obs (e.g. a
        # CNN-shaped height-scan ``(B, 1, 76, 126)``) get the right total size.
        group_dim = int(prod(obs[group].shape[1:])) if group in obs else 0
        if group == achieved_group:
            goal_start = offset
            goal_end = offset + group_dim
        if group != goal_group:
            obs_dim += group_dim
        offset += group_dim

    if goal_start is None:
        goal_start = obs_dim
        goal_end = offset

    her_cfg["obs_dim"] = obs_dim
    her_cfg["goal_start_idx"] = goal_start
    her_cfg["goal_end_idx"] = goal_end
    alg_cfg["her_cfg"] = her_cfg
    return alg_cfg
