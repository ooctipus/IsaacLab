# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.sensors import SensorBaseCfg


##
# Helper functions for per-env sensors, or assets
##
def _normalize_env_ids(env_ids: Any | None) -> list[int]:
    """Normalize environment indices to a list of integers."""
    if env_ids is None or isinstance(env_ids, slice):
        return None
    if hasattr(env_ids, "tolist"):
        return [int(val) for val in env_ids.tolist()]
    return [int(val) for val in env_ids]


def _extract_env_ids_from_prim_path(prim_path: Any) -> tuple[int, ...]:
    """Extract environment indices from a prim path."""
    if prim_path in (None, MISSING):
        return ()
    # handle list or tuple of prim paths
    if isinstance(prim_path, (list, tuple)):
        env_ids: set[int] = set()
        for entry in prim_path:
            env_ids.update(_extract_env_ids_from_prim_path(entry))
        return tuple(sorted(env_ids))

    if not isinstance(prim_path, str):
        return ()

    # match explicit env_<id>
    matches = re.findall(r"env_(\d+)", prim_path)
    env_ids: set[int] = {int(match) for match in matches}

    # match env_(a|b|c) format
    regex_matches = re.findall(r"env_\(([^)]+)\)", prim_path)
    for group in regex_matches:
        parts = re.split(r"[|,]", group)
        for part in parts:
            part = part.strip()
            if part.isdigit():
                env_ids.add(int(part))

    if not env_ids:
        return ()
    return tuple(sorted(env_ids))


def _resolve_assigned_envs(cfg: SensorBaseCfg | AssetBaseCfg | Any) -> tuple[int, ...]:
    """Resolve assigned environments from configuration with "prim_path" property.
    Args:
        cfg: SensorBaseCfg | AssetBaseCfg
    Returns:
        tuple[int, ...]: The assigned environments.
    """
    explicit = getattr(cfg, "assigned_envs", None)
    if explicit:
        return tuple(int(val) for val in explicit)
    env_ids = _extract_env_ids_from_prim_path(getattr(cfg, "prim_path", None))
    if env_ids:
        return env_ids
    spawn_cfg = getattr(cfg, "spawn", None)
    if spawn_cfg is not None:
        env_ids = _extract_env_ids_from_prim_path(getattr(spawn_cfg, "prim_path", None))
        if env_ids:
            return env_ids
    return ()


class PerEnvMixin:
    """Mixin that adds per-environment management utilities to sensor/asset/action classes.

    Classes inheriting from this mixin should provide a configuration dataclass containing an
    ``assigned_envs`` field with the global environment indices that the sensor/asset/action instance should manage.
    An empty tuple denotes that the sensor/asset/action should manage all available environments (default behaviour).
    """

    def __init__(self, cfg: SensorBaseCfg | AssetBaseCfg | ActionTermCfg | Any, *args, **kwargs):
        # _assigned_envs can only be resolved from AssetBaseCfg or SensorBaseCfg with "prim_path" property
        # for other types of cfg, user has to manually set "assigned_envs" property
        self._assigned_envs = _resolve_assigned_envs(cfg)
        self._assigned_env_to_local = {env_idx: idx for idx, env_idx in enumerate(self._assigned_envs)}

        super().__init__(cfg, *args, **kwargs)

    @property
    def assigned_envs(self) -> tuple[int, ...]:
        """Global environment indices handled by this sensor instance."""
        return self._assigned_envs

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _filter_env_ids(self, env_ids: Any | None) -> torch.Tensor:
        """Normalize and filter environment indices to the managed subset. Returns local indices as a long tensor."""
        env_ids_list = _normalize_env_ids(env_ids)

        if env_ids_list is None:
            local_ids = list(range(len(self._assigned_envs)))
            return torch.tensor(local_ids, dtype=torch.long, device=self.device)

        local_ids = [self._assigned_env_to_local[idx] for idx in env_ids_list if idx in self._assigned_env_to_local]
        return torch.tensor(local_ids, dtype=torch.long, device=self.device)
