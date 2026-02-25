# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.string import resolve_assigned_env_ids_from_cfg


class PerEnvMixin:
    """Mixin that adds per-environment management utilities to sensor/asset/action classes.

    Classes inheriting from this mixin should provide a configuration dataclass containing an
    ``assigned_envs`` field with the global environment indices that the sensor/asset/action instance should manage.
    An empty tuple denotes that the sensor/asset/action should manage all available environments (default behaviour).
    """

    def __init__(self, cfg: SensorBaseCfg | AssetBaseCfg | ActionTermCfg | Any, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._assigned_envs = resolve_assigned_env_ids_from_cfg(cfg)
        self._assigned_envs_to_local_indices = {
            global_idx: local_idx for local_idx, global_idx in enumerate(self._assigned_envs)
        }

    @property
    def assigned_envs(self) -> tuple[int, ...]:
        """Global environment indices handled by this instance."""
        return self._assigned_envs

    @property
    def assigned_envs_to_local_indices(self) -> dict[int, int]:
        """Map of global environment indices to local indices."""
        return self._assigned_envs_to_local_indices

    @property
    def is_heterogeneous(self) -> bool:
        """
        Check if the articulation is heterogeneous.
        Returns:
            bool: True if global_to_local filtering is needed, in case of heterogeneous environments.
            False if no filtering is needed, fallback to homogeneous environments.
        """
        return self._assigned_envs is not None and len(self._assigned_envs) > 0

    def _filter_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """Filter global env indices to the managed subset.
        Args:
            env_ids: global env indices (tensor or sequence of int).
            If None, returns all local indices (shape (len(assigned_envs),)).
        Returns: 1D long tensor (local indices)
        """
        if env_ids is None:
            return torch.arange(len(self._assigned_envs), dtype=torch.long, device=self.device)

        env_ids = env_ids.cpu().tolist()
        local_list = [
            self._assigned_envs_to_local_indices[env_id]
            for env_id in env_ids
            if env_id in self._assigned_envs_to_local_indices
        ]
        return torch.tensor(local_list, dtype=torch.long, device=self.device)
