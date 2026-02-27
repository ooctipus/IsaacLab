# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for runtime multi-task composition."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import MISSING, field
from enum import Enum
from typing import Any

from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class SimParamsSource(str, Enum):
    """Policy for deriving multitask simulation parameters from per-task cfgs."""

    FIRST = "first"
    """Use the first task's values (consistent with robot/actions)."""
    CONSERVATIVE = "conservative"
    """episode_length_s = max(tasks), sim.dt = min(tasks); decimation and render_interval from first task."""


# Default sim params values.
DEFAULT_DECIMATION = 3
DEFAULT_EPISODE_LENGTH_S = 10.0
DEFAULT_SIM_DT = 1 / 60
DEFAULT_RENDER_INTERVAL = 3


@configclass
class MultiTaskRegistryConfig:
    """Runtime task registry configuration for grouped multitask envs."""

    task_names_by_group: list[str] = MISSING
    group_size: int = 1
    device: str = "cuda"
    use_fabric: bool | None = None
    # cache task configs to avoid loading them multiple times
    _task_cfg_cache: dict[str, ManagerBasedRLEnvCfg] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not self.task_names_by_group:
            raise ValueError("task_names_by_group must contain at least one task name.")

    @property
    def total_groups(self) -> int:
        return len(self.task_names_by_group)

    def get_group_index(self, env_idx: int) -> int:
        return int(env_idx) // int(self.group_size)

    def get_task_name_for_group(self, group_idx: int) -> str:
        if group_idx >= len(self.task_names_by_group):
            raise IndexError(
                f"group_idx {group_idx} is out of range for task_names_by_group={len(self.task_names_by_group)}."
            )
        return self.task_names_by_group[group_idx]

    def get_task_name_for_env(self, env_idx: int) -> str:
        return self.get_task_name_for_group(self.get_group_index(env_idx))

    def get_task_cfg(self, task_name: str) -> ManagerBasedRLEnvCfg:
        """Load and cache task config from registry."""
        if task_name not in self._task_cfg_cache:
            cfg = parse_env_cfg(task_name, device=self.device, num_envs=None, use_fabric=self.use_fabric)
            if not isinstance(cfg, ManagerBasedRLEnvCfg) and not isinstance(cfg, ManagerBasedEnvCfg):
                raise TypeError(f"Task '{task_name}' does not return ManagerBasedRLEnvCfg or ManagerBasedEnvCfg.")
            self._task_cfg_cache[task_name] = cfg
        return self._task_cfg_cache[task_name]

    def env_indices_for_group(self, group_idx: int) -> tuple[int, ...]:
        """Get environment indices for a specific group."""
        start = group_idx * self.group_size
        end = min(start + self.group_size, self.group_size * self.total_groups)
        return tuple(range(start, end))

    def get_sim_params(self, source: SimParamsSource | str) -> tuple[int, float, float, int]:
        """Compute multitask sim params from task cfgs.

        Returns:
            (decimation, episode_length_s, sim_dt, render_interval)
        """
        source_enum = SimParamsSource(source)
        if self.total_groups == 0:
            return (
                DEFAULT_DECIMATION,
                DEFAULT_EPISODE_LENGTH_S,
                DEFAULT_SIM_DT,
                DEFAULT_RENDER_INTERVAL,
            )
        if source_enum == SimParamsSource.FIRST:
            return self._get_sim_params_from_first_task()
        return self._get_sim_params_conservative()

    def _get_sim_params_from_first_task(self) -> tuple[int, float, float, int]:
        """Use first task's decimation, episode_length_s, sim.dt, render_interval."""
        first_cfg = self.get_task_cfg(self.get_task_name_for_group(0))
        decimation = getattr(first_cfg, "decimation", DEFAULT_DECIMATION)
        episode_length_s = getattr(first_cfg, "episode_length_s", DEFAULT_EPISODE_LENGTH_S)
        dt = getattr(first_cfg.sim, "dt", DEFAULT_SIM_DT)
        render_interval = getattr(first_cfg.sim, "render_interval", DEFAULT_RENDER_INTERVAL)

        return (decimation, episode_length_s, dt, render_interval)

    def _get_sim_params_conservative(self) -> tuple[int, float, float, int]:
        """Use max episode_length_s and min dt; decimation/render_interval from default values."""
        episode_lengths: list[float] = []
        dts: list[float] = []
        for group_idx in range(self.total_groups):
            cfg = self.get_task_cfg(self.get_task_name_for_group(group_idx))
            episode_lengths.append(getattr(cfg, "episode_length_s", DEFAULT_EPISODE_LENGTH_S))
            dts.append(getattr(cfg.sim, "dt", DEFAULT_SIM_DT))

        return (
            DEFAULT_DECIMATION,
            max(episode_lengths),
            min(dts),
            DEFAULT_RENDER_INTERVAL,
        )

    """Helper methods for multi-task composition."""

    @staticmethod
    def iter_scene_cfg_items(scene_cfg: InteractiveSceneCfg) -> Iterator[tuple[str, object]]:
        """Iterate over all attributes in scene config, including those added in __post_init__."""
        # Use vars() to get all instance attributes, including those added dynamically in __post_init__
        # This works for both dataclass and regular class instances
        for name, value in vars(scene_cfg).items():
            # Skip private attributes and internal fields
            if name.startswith("_"):
                continue
            yield name, value

    @staticmethod
    def clone_cfg(cfg: Any, **kwargs: Any) -> Any:
        """Clone a configuration object with updated parameters."""
        if hasattr(cfg, "replace") and callable(getattr(cfg, "replace")):
            return cfg.replace(**kwargs)
        else:
            raise TypeError(f"Unsupported config type for cloning: {type(cfg)}")

    @staticmethod
    def group_prim_from_template(env_ids: Iterable[int], prim_path: str) -> str:
        """Convert a template prim path to a group-specific regex pattern.
        The result matches only the given env indices, so resolvers can scope to one group.
        Example:
            >>> group_prim_from_template((0, 1, 2), "/World/envs/env_.*/Robot")
            '/World/envs/env_(0|1|2)/Robot'
            >>> group_prim_from_template((5, 6), "{ENV_REGEX_NS}/Cube")
            '/World/envs/env_(5|6)/Cube'
        """
        suffix: str | None = None
        if "{ENV_REGEX_NS}" in prim_path:
            suffix = prim_path.replace("{ENV_REGEX_NS}", "")
        elif prim_path.startswith("/World/envs/env_.*"):
            suffix = prim_path.split("/World/envs/env_.*")[-1]
        if suffix is None:
            return prim_path
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"
        env_pattern = "|".join(str(idx) for idx in env_ids)
        return f"/World/envs/env_({env_pattern}){suffix}"

    @staticmethod
    def should_group_cfg(cfg) -> bool:
        """Check if a config should be grouped (has per-env prim path)."""
        prim_path = getattr(cfg, "prim_path", "")
        return "{ENV_REGEX_NS}" in str(prim_path) or str(prim_path).startswith("/World/envs/env_")
