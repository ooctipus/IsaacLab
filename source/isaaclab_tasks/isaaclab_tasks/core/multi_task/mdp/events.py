# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful event terms shared by multi-task environments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RootVelocityPushDiscrete(ManagerTermBase):
    """Add random root-velocity increments on independent integer-second schedules.

    Every environment samples its own low-inclusive, high-exclusive integer
    interval [s]. The schedule advances from one global one-second event and is
    intentionally preserved across episodic resets.
    """

    _AXES = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize fixed-capacity schedule and velocity buffers.

        Args:
            cfg: Event-term configuration.
            env: Owning manager-based environment.
        """
        super().__init__(cfg, env)
        params = cfg.params
        interval_seconds_range = params["interval_seconds_range"]
        velocity_range = params["velocity_range"]
        self._validate(interval_seconds_range, velocity_range, cfg.interval_range_s, cfg.is_global_time)

        self._asset = env.scene[params["asset_cfg"].name]
        root_velocity = self._asset.data.root_vel_w.torch
        if root_velocity.shape != (env.num_envs, 6) or not root_velocity.is_floating_point():
            raise ValueError("RootVelocityPushDiscrete requires floating root velocities shaped [num_envs, 6].")

        ranges = tuple(velocity_range.get(axis, (0.0, 0.0)) for axis in self._AXES)
        lower = torch.tensor(tuple(bounds[0] for bounds in ranges), dtype=root_velocity.dtype, device=env.device)
        upper = torch.tensor(tuple(bounds[1] for bounds in ranges), dtype=root_velocity.dtype, device=env.device)
        self._velocity_lower = lower
        self._velocity_width = upper - lower

        self._interval_second_choices = torch.arange(*interval_seconds_range, dtype=torch.int32, device=env.device)
        self._elapsed_seconds = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._interval_seconds = torch.empty_like(self._elapsed_seconds)
        self._sample_indices = torch.empty(env.num_envs, dtype=torch.int64, device=env.device)
        self._sample_seconds = torch.empty_like(self._elapsed_seconds)

        self._trigger_mask = torch.empty(env.num_envs, dtype=torch.bool, device=env.device)
        self._trigger_mask_warp = wp.from_torch(self._trigger_mask, dtype=wp.bool)
        self._root_velocity = torch.empty_like(root_velocity)
        self._velocity_increment = torch.empty_like(root_velocity)
        self._sample_intervals()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Preserve the simulation-time schedule across episodic resets.

        Args:
            env_ids: Ignored environment indices.
        """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        interval_seconds_range: tuple[int, int],
        velocity_range: Mapping[str, tuple[float, float]],
    ) -> None:
        """Advance schedules by one second and apply due velocity increments.

        Args:
            env: Owning manager-based environment.
            env_ids: Indices supplied by the event manager.
            asset_cfg: Scene entity naming the pushed asset.
            interval_seconds_range: Low-inclusive, high-exclusive interval range [s].
            velocity_range: Uniform root-velocity increment ranges. Translational
                axes are [m/s] and rotational axes are [rad/s].
        """
        del env, env_ids, asset_cfg, interval_seconds_range, velocity_range

        self._elapsed_seconds.add_(1)
        torch.eq(self._elapsed_seconds, self._interval_seconds, out=self._trigger_mask)
        self._elapsed_seconds.masked_fill_(self._trigger_mask, 0)
        self._sample_intervals(self._trigger_mask)

        self._velocity_increment.uniform_().mul_(self._velocity_width).add_(self._velocity_lower)
        self._velocity_increment.mul_(self._trigger_mask.unsqueeze(-1))
        torch.add(self._asset.data.root_vel_w.torch, self._velocity_increment, out=self._root_velocity)
        self._asset.write_root_velocity_to_sim_mask(root_velocity=self._root_velocity, env_mask=self._trigger_mask_warp)

    def _sample_intervals(self, trigger_mask: torch.Tensor | None = None) -> None:
        """Sample intervals into every row or only triggered rows."""
        self._sample_indices.random_(0, self._interval_second_choices.shape[0])
        torch.index_select(self._interval_second_choices, 0, self._sample_indices, out=self._sample_seconds)
        if trigger_mask is None:
            self._interval_seconds.copy_(self._sample_seconds)
        else:
            torch.where(trigger_mask, self._sample_seconds, self._interval_seconds, out=self._interval_seconds)

    @classmethod
    def _validate(
        cls,
        interval_seconds_range: tuple[int, int],
        velocity_range: Mapping[str, tuple[float, float]],
        event_interval_seconds: tuple[float, float] | None,
        is_global_time: bool,
    ) -> None:
        """Validate the discrete schedule and root-velocity ranges."""
        if (
            not isinstance(interval_seconds_range, tuple)
            or len(interval_seconds_range) != 2
            or any(type(value) is not int for value in interval_seconds_range)
        ):
            raise TypeError("Discrete push intervals must be one pair of integer seconds.")
        low, high = interval_seconds_range
        if low < 1 or high <= low:
            raise ValueError("Discrete push interval bounds must be positive and increasing [s].")
        if (
            not is_global_time
            or event_interval_seconds is None
            or any(not math.isclose(value, 1.0) for value in event_interval_seconds)
        ):
            raise ValueError("RootVelocityPushDiscrete must run as one global event exactly once per second.")
        if not isinstance(velocity_range, Mapping) or any(axis not in cls._AXES for axis in velocity_range):
            raise ValueError(f"Root velocity ranges may contain only {cls._AXES}.")
        for axis, bounds in velocity_range.items():
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(isinstance(value, int | float) and math.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"Root velocity range for {axis!r} must be one finite ordered pair.")
