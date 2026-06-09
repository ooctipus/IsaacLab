# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused Warp kernel for HER sampling + gathering.

Replaces the separate sample_indices + buffer gather steps with a single
GPU kernel launch. For each output element, the kernel:

1. Generates a random (t, e) pair.
2. Looks up the episode boundary.
3. Samples a geometric future offset, clamped to the episode horizon.
4. Gathers state, action, and goal columns directly from the buffer.

Falls back gracefully if Warp is not installed.
"""

from __future__ import annotations

import math

import torch

try:
    import warp as wp

    # ``wp.from_torch`` reads ``warp._src.context.runtime.cuda_devices`` directly and
    # crashes if the runtime hasn't been initialized. Inside Isaac Lab the simulator
    # init does this for you; bare ``pytest`` invocations or other entry points that
    # import this module without going through the simulator launch path won't, so
    # initialize here. ``wp.init()`` is idempotent.
    wp.init()

    @wp.kernel(enable_backward=False)
    def _sample_and_gather_kernel(
        buffer_flat: wp.array2d(dtype=wp.float32),
        episode_end: wp.array2d(dtype=wp.int32),
        num_envs: int,
        buf_size: int,
        obs_dim: int,
        act_start: int,
        act_end: int,
        goal_start: int,
        goal_end: int,
        log_gamma: float,
        base_seed: int,
        out_state: wp.array2d(dtype=wp.float32),
        out_act: wp.array2d(dtype=wp.float32),
        out_goal: wp.array2d(dtype=wp.float32),
    ):
        i = wp.tid()

        state = wp.rand_init(base_seed, i)

        max_flat = (buf_size - 1) * num_envs
        flat_val = wp.abs(wp.randi(state)) % max_flat
        t = flat_val // num_envs
        e = flat_val % num_envs

        ep_end = int(episode_end[t, e])
        horizon = ep_end - t
        if horizon < 1:
            # Re-sample: shift t back by 1 to guarantee a valid future.
            t = wp.max(t - 1, 0)
            ep_end = int(episode_end[t, e])
            horizon = wp.max(ep_end - t, 1)

        # Geometric sampling via inverse CDF: k = ceil(log(1-u) / log(gamma)).
        u = wp.randf(state)
        u = wp.clamp(u, 0.001, 0.999)
        k = int(wp.ceil(wp.log(1.0 - u) / log_gamma))
        k = wp.clamp(k, 1, horizon)
        ft = t + k

        # Flatten indices for the 2D buffer view [capacity * num_envs, data_dim].
        current_idx = t * num_envs + e
        future_idx = ft * num_envs + e

        # Gather state columns [0, obs_dim).
        for c in range(obs_dim):
            out_state[i, c] = buffer_flat[current_idx, c]

        # Gather action columns [act_start, act_end).
        act_dim = act_end - act_start
        for c in range(act_dim):
            out_act[i, c] = buffer_flat[current_idx, act_start + c]

        # Gather goal columns [goal_start, goal_end) from future timestep.
        goal_dim = goal_end - goal_start
        for c in range(goal_dim):
            out_goal[i, c] = buffer_flat[future_idx, goal_start + c]

    _WARP_AVAILABLE = True

except ImportError:
    _WARP_AVAILABLE = False


def warp_sample_and_gather(
    buffer,
    episode_end: torch.Tensor,
    seed_col: int,
    obs_dim: int,
    act_start: int,
    act_end: int,
    goal_start: int,
    goal_end: int,
    gamma: float,
    num_samples: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused sampling + gathering via a single Warp kernel launch.

    Args:
        buffer: :class:`~rsl_rl.storage.ReplayBuffer` instance.
        episode_end: ``[size, num_envs]`` int32 tensor of episode-end indices.
        seed_col: Column index of the seed field (unused by kernel, kept for API compat).
        obs_dim: Number of state columns to gather.
        act_start: Start column of the action slice.
        act_end: End column (exclusive) of the action slice.
        goal_start: Start column of the goal slice (gathered from future timestep).
        goal_end: End column (exclusive) of the goal slice.
        gamma: Geometric discount for future sampling.
        num_samples: Number of samples to produce.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of ``(out_state, out_act, out_goal)``, each contiguous on the same device.
    """
    if not _WARP_AVAILABLE:
        raise ImportError("Warp is required for warp_sample_and_gather")

    device = buffer.device
    act_dim = act_end - act_start
    goal_dim = goal_end - goal_start

    out_state = torch.zeros(num_samples, obs_dim, device=device)
    out_act = torch.zeros(num_samples, act_dim, device=device)
    out_goal = torch.zeros(num_samples, goal_dim, device=device)

    flat_buffer = buffer.data.reshape(-1, buffer.data_dim)

    wp.launch(
        _sample_and_gather_kernel,
        dim=num_samples,
        inputs=[
            wp.from_torch(flat_buffer),
            wp.from_torch(episode_end),
            buffer.num_envs,
            buffer.size,
            obs_dim,
            act_start,
            act_end,
            goal_start,
            goal_end,
            math.log(gamma),
            seed,
        ],
        outputs=[
            wp.from_torch(out_state),
            wp.from_torch(out_act),
            wp.from_torch(out_goal),
        ],
        device=device,
    )

    return out_state, out_act, out_goal
