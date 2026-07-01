# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for motion imitation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_assets.robots.smpl.smpl_constants import HUMENV_BODY_INERTIA, HUMENV_BODY_MASS, MUJOCO_BODY_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_smpl_body_mass_inertia(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg) -> None:
    """Write the HumEnv body masses [kg] and inertias [kg m^2] at startup.

    Body names define the mapping from source-order constants to the active
    articulation; no simulator-specific captured permutation is involved.

    Args:
        env: Owning manager-based environment.
        env_ids: Unused startup environment indices.
        asset_cfg: Scene entity naming the SMPL articulation.
    """
    del env_ids
    asset = env.scene[asset_cfg.name]
    body_indices, body_names = asset.find_bodies(MUJOCO_BODY_NAMES, preserve_order=True)
    if tuple(body_names) != MUJOCO_BODY_NAMES:
        raise RuntimeError(f"SMPL body layout differs from HumEnv: {body_names}")
    body_ids = torch.tensor(body_indices, dtype=torch.int32, device=asset.device)

    masses = torch.tensor(HUMENV_BODY_MASS, dtype=torch.float32, device=asset.device)
    inertias = torch.tensor(HUMENV_BODY_INERTIA, dtype=torch.float32, device=asset.device)
    masses_full = masses.expand(asset.num_instances, -1).contiguous()
    inertias_full = inertias.expand(asset.num_instances, -1, -1).contiguous()
    asset.set_masses_index(masses=masses_full, body_ids=body_ids)
    asset.set_inertias_index(inertias=inertias_full, body_ids=body_ids)


class MotionPushVelocity(ManagerTermBase):
    """Add BFM-style random root velocities on independent integer-second schedules.

    Each environment owns an integer elapsed-seconds counter and a sampled interval.
    Triggered counters alone are reset and resampled. The state intentionally
    survives episodic resets, matching the released BFM training environment.

    Configure this term as a global ``interval`` event that fires once per second.
    Evaluation presets should omit the term entirely.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize fixed-capacity schedule and velocity buffers.

        Args:
            cfg: Event-term configuration.
            env: Owning manager-based environment.
        """
        super().__init__(cfg, env)
        params = cfg.params
        interval_seconds = params["interval_seconds_integer_high_exclusive"]
        linear_range = params["linear_velocity_range_m_s"]
        angular_range = params["angular_velocity_range_rad_s"]
        self._validate(interval_seconds, linear_range, angular_range, cfg.interval_range_s, cfg.is_global_time)

        self._asset = env.scene[params["asset_cfg"].name]
        self._linear_range_m_s = linear_range
        self._angular_range_rad_s = angular_range
        self._linear_width_m_s = linear_range[1] - linear_range[0]
        self._angular_width_rad_s = angular_range[1] - angular_range[0]

        self._interval_second_choices = torch.arange(*interval_seconds, dtype=torch.int32, device=env.device)
        self._elapsed_seconds = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._interval_seconds = torch.empty_like(self._elapsed_seconds)

        # Every hot-path tensor has fixed environment-major capacity. The mask's
        # Warp view shares storage and is created once at the backend boundary.
        self._trigger_mask = torch.empty(env.num_envs, dtype=torch.bool, device=env.device)
        wp.init()
        self._trigger_mask_warp = wp.from_torch(self._trigger_mask, dtype=wp.bool)
        self._sample_indices = torch.empty(env.num_envs, dtype=torch.int64, device=env.device)
        self._sample_seconds = torch.empty_like(self._elapsed_seconds)
        self._root_velocity = torch.empty_like(self._asset.data.root_vel_w.torch)
        self._velocity_increment = torch.empty_like(self._root_velocity)
        self._sample_intervals()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Preserve the push schedule across episodic resets.

        Args:
            env_ids: Ignored environment indices. Push timing is independent of
                episode timing.
        """

    def evaluation_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the persistent per-environment push schedule."""
        return {
            "elapsed_seconds": self._elapsed_seconds.clone(),
            "interval_seconds": self._interval_seconds.clone(),
        }

    def load_evaluation_state_dict(self, state: Mapping[str, torch.Tensor | float]) -> None:
        """Restore one push schedule captured by :meth:`evaluation_state_dict`."""
        if set(state) != {"elapsed_seconds", "interval_seconds"}:
            raise ValueError("Motion push evaluation state has unexpected fields.")
        elapsed_seconds = state["elapsed_seconds"]
        interval_seconds = state["interval_seconds"]
        if not isinstance(elapsed_seconds, torch.Tensor) or not isinstance(interval_seconds, torch.Tensor):
            raise TypeError("Motion push evaluation state must contain tensors.")
        for name, value, destination in (
            ("elapsed_seconds", elapsed_seconds, self._elapsed_seconds),
            ("interval_seconds", interval_seconds, self._interval_seconds),
        ):
            if (
                value.shape != destination.shape
                or value.dtype != destination.dtype
                or value.device != destination.device
            ):
                raise ValueError(f"Motion push evaluation {name} differs from its live tensor contract.")
            destination.copy_(value)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        interval_seconds_integer_high_exclusive: tuple[int, int],
        linear_velocity_range_m_s: tuple[float, float],
        angular_velocity_range_rad_s: tuple[float, float],
    ) -> None:
        """Advance every schedule by one second and apply due pushes.

        Args:
            env: Owning manager-based environment.
            env_ids: Indices supplied by the event manager. A global one-second
                event supplies ``None``; schedule selection remains per environment.
            asset_cfg: Scene entity naming the pushed articulation.
            interval_seconds_integer_high_exclusive: Low-inclusive,
                high-exclusive integer interval range [s].
            linear_velocity_range_m_s: Uniform horizontal velocity increment
                range [m/s].
            angular_velocity_range_rad_s: Uniform angular velocity increment
                range [rad/s].
        """
        del env, env_ids, asset_cfg
        del interval_seconds_integer_high_exclusive, linear_velocity_range_m_s, angular_velocity_range_rad_s

        self._elapsed_seconds.add_(1)
        torch.eq(self._elapsed_seconds, self._interval_seconds, out=self._trigger_mask)
        self._elapsed_seconds.masked_fill_(self._trigger_mask, 0)
        self._sample_intervals(self._trigger_mask)

        self._velocity_increment.uniform_()
        self._velocity_increment[:, :2].mul_(self._linear_width_m_s).add_(self._linear_range_m_s[0])
        self._velocity_increment[:, 2].zero_()
        self._velocity_increment[:, 3:].mul_(self._angular_width_rad_s).add_(self._angular_range_rad_s[0])
        self._velocity_increment.mul_(self._trigger_mask.unsqueeze(-1))
        torch.add(self._asset.data.root_vel_w.torch, self._velocity_increment, out=self._root_velocity)
        self._asset.write_root_velocity_to_sim_mask(root_velocity=self._root_velocity, env_mask=self._trigger_mask_warp)

    def _sample_intervals(self, trigger_mask: torch.Tensor | None = None) -> None:
        """Sample discrete intervals into all rows or one fixed-capacity mask."""
        self._sample_indices.random_(0, self._interval_second_choices.shape[0])
        torch.index_select(self._interval_second_choices, 0, self._sample_indices, out=self._sample_seconds)
        if trigger_mask is None:
            self._interval_seconds.copy_(self._sample_seconds)
        else:
            torch.where(trigger_mask, self._sample_seconds, self._interval_seconds, out=self._interval_seconds)

    @staticmethod
    def _validate(
        interval_seconds: tuple[int, int],
        linear_range: tuple[float, float],
        angular_range: tuple[float, float],
        event_interval_seconds: tuple[float, float] | None,
        is_global_time: bool,
    ) -> None:
        """Validate constructor inputs at the public event boundary."""
        low, high = interval_seconds
        if low < 1 or high <= low:
            raise ValueError("Push interval bounds must be positive and increasing [s].")
        if (
            not is_global_time
            or event_interval_seconds is None
            or any(not math.isclose(value, 1.0) for value in event_interval_seconds)
        ):
            raise ValueError("MotionPushVelocity must run as one global event exactly once per second.")
        for name, value_range in (("linear", linear_range), ("angular", angular_range)):
            if not all(math.isfinite(value) for value in value_range) or value_range[0] > value_range[1]:
                raise ValueError(f"Push {name} velocity range must be finite and ordered.")
