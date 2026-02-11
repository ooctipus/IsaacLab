# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for runtime multi-task composition."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import MISSING, field, is_dataclass
from dataclasses import replace as dc_replace
from typing import Any

from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@configclass
class MultiTaskRegistryConfig:
    """Runtime task registry configuration for grouped multitask envs."""

    task_names_by_group: list[str] = MISSING
    group_size: int = 1
    device: str = "cuda"
    use_fabric: bool | None = None
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
        if hasattr(cfg, "replace") and callable(cfg.replace):
            return cfg.replace(**kwargs)
        if is_dataclass(cfg):
            return dc_replace(cfg, **kwargs)
        raise TypeError(f"Unsupported config type for cloning: {type(cfg)}")

    @staticmethod
    def group_prim_from_template(env_ids: Iterable[int], prim_path: str) -> str:
        """Convert a template prim path to a group-specific regex pattern."""
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
