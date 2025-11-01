# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase
from isaaclab.terrains import TerrainImporter

from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class terrain_levels_vel(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain: TerrainImporter = env.scene.terrain
        # read optional parameters
        self._demotion_fraction: float = float(cfg.params.get("demotion_fraction", 0.05))
        # cache terrain layout
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self._num_levels * self._num_types,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        # initialize mapping that relates terrain columns to sub-terrain keys
        self._init_type_mapping(terrain)
        # pre-allocate result dictionary to avoid per-step allocations
        self._result: dict[str, torch.Tensor] = {
            "all": torch.zeros((), dtype=torch.float, device=env.device),
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
        }
        for name in self._type_names:
            self._result[f"{name}_level"] = torch.zeros((), dtype=torch.float, device=env.device)
            self._result[f"{name}_success"] = torch.zeros((), dtype=torch.float, device=env.device)

    def _init_type_mapping(self, terrain: TerrainImporter) -> None:
        gen_cfg = terrain.cfg.terrain_generator
        self._type_names = list(gen_cfg.sub_terrains.keys())
        props = torch.tensor(
            [sub_cfg.proportion for sub_cfg in gen_cfg.sub_terrains.values()], dtype=torch.float, device=self.device
        )
        if props.numel() == 0 or not torch.isfinite(props).all() or props.sum() <= 0:
            props = torch.ones((len(self._type_names),), device=self.device)
        props = props / props.sum()
        cum = torch.cumsum(props, dim=0)
        num_types = int(terrain.terrain_origins.shape[1])
        cols = torch.arange(num_types, device=self.device, dtype=torch.float)
        pos = cols / float(num_types) + 1e-3
        self._col_to_type_idx = torch.searchsorted(cum, pos, right=True).to(torch.long)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], demotion_fraction: float = 0.05):
        """Distance-based terrain curriculum with logging aligned to success-rate term.

        Promotes terrains when the agent gets within 0.5 m of the goal ("success"),
        and demotes when the traveled fraction falls below the configured demotion fraction.

        Returns a logging dict with overall and per-type mean levels and success.
        """
        terrain: TerrainImporter = env.scene.terrain

        # compute current progress towards goal for the provided envs
        command = env.command_manager.get_command("goal_point")
        distance = command[env_ids, :2].norm(2, dim=1)
        terrain_sample = self._num_types * terrain.terrain_levels[env_ids] + terrain.terrain_types[env_ids]
        success = distance < 0.5
        self.success_monitor.success_update(terrain_sample, success)

        total_distance = (
            env.command_manager.get_term("goal_point").pos_command_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
        ).norm(2, dim=1)
        distance_traveled = 1 - distance / total_distance
        move_up = success
        move_down = distance_traveled < self._demotion_fraction

        # update terrain levels/types via importer helper
        terrain.update_env_origins(env_ids, move_up, move_down)

        # aggregate reporting: overall mean and per-type mean level
        self._result["all"].copy_(terrain.terrain_levels.float().mean())
        for i, name in enumerate(self._type_names):
            key_level = f"{name}_level"
            mask = self._col_to_type_idx[terrain.terrain_types] == i
            if torch.any(mask):
                self._result[key_level].copy_(terrain.terrain_levels[mask].float().mean())
            else:
                self._result[key_level].zero_()

        # compute instantaneous success across all envs for consistent logging keys
        success = self.success_monitor.get_success_rate()  # [num_levels * num_types]
        self._result["all_success"].copy_(success.mean())
        per_col_success = success.view(self._num_levels, self._num_types).mean(dim=0)  # [num_types]
        for i, name in enumerate(self._type_names):
            key_succ = f"{name}_success"
            col_mask = self._col_to_type_idx == i
            if torch.any(col_mask):
                self._result[key_succ].copy_(per_col_success[col_mask].mean())
            else:
                self._result[key_succ].zero_()

        return self._result


class terrain_success_rate_levels(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain: TerrainImporter = env.scene.terrain
        # number of levels (rows) and terrain types (cols)
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self._num_levels * self._num_types,
            device=env.device,
        )
        # store sampled (level,type) as a flattened index in [0, num_levels * num_types)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        # initialize mapping that relates terrain columns to sub-terrain keys
        self._init_type_mapping(terrain)
        # pre-allocate result dictionary to avoid per-step allocations
        self._result: dict[str, torch.Tensor] = {
            "all": torch.zeros((), dtype=torch.float, device=env.device),
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
        }
        for name in self._type_names:
            self._result[f"{name}_level"] = torch.zeros((), dtype=torch.float, device=env.device)
            self._result[f"{name}_success"] = torch.zeros((), dtype=torch.float, device=env.device)

    def _init_type_mapping(self, terrain: TerrainImporter) -> None:
        gen_cfg = terrain.cfg.terrain_generator
        self._type_names = list(gen_cfg.sub_terrains.keys())
        props = torch.tensor(
            [sub_cfg.proportion for sub_cfg in gen_cfg.sub_terrains.values()], dtype=torch.float, device=self.device
        )
        if props.numel() == 0 or not torch.isfinite(props).all() or props.sum() <= 0:
            props = torch.ones((len(self._type_names),), device=self.device)
        props = props / props.sum()
        cum = torch.cumsum(props, dim=0)
        num_types = int(terrain.terrain_origins.shape[1])
        cols = torch.arange(num_types, device=self.device, dtype=torch.float)
        pos = cols / float(num_types) + 1e-3
        self._col_to_type_idx = torch.searchsorted(cum, pos, right=True).to(torch.long)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]):
        terrain: TerrainImporter = env.scene.terrain
        command = env.command_manager.get_command("goal_point")
        distance = command[env_ids, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples[env_ids], distance < 0.5)
        # ensure a concrete length for sampling (slice(None) is not valid for len())
        choices, probs = self.success_monitor.sample_by_target_rate(env_ids, target=0.5, kappa=1, return_probs=True)
        self.term_samples[env_ids] = choices.to(self.term_samples.dtype)
        # map flattened index back to (level, type)
        num_types = int(terrain.terrain_origins.shape[1])
        chosen_level, chosen_type = self.term_samples[env_ids] // num_types, self.term_samples[env_ids] % num_types
        terrain.env_origins[env_ids] = terrain.terrain_origins[chosen_level, chosen_type]
        terrain.terrain_levels[env_ids] = chosen_level
        terrain.terrain_types[env_ids] = chosen_type

        # aggregate reporting: overall mean and per-type mean level/success (in-place updates)
        self._result["all"].copy_(terrain.terrain_levels.float().mean())
        for i, name in enumerate(self._type_names):
            key = f"{name}_level"
            mask = self._col_to_type_idx[terrain.terrain_types] == i
            if torch.any(mask):
                self._result[key].copy_(terrain.terrain_levels[mask].float().mean())
            else:
                self._result[key].zero_()

        # compute success rates from monitor and aggregate per type
        success = self.success_monitor.get_success_rate()  # [num_levels * num_types]
        self._result["all_success"].copy_(success.mean())
        per_col_success = success.view(self._num_levels, self._num_types).mean(dim=0)  # [num_types]
        for i, name in enumerate(self._type_names):
            key_s = f"{name}_success"
            col_mask = self._col_to_type_idx == i
            if torch.any(col_mask):
                self._result[key_s].copy_(per_col_success[col_mask].mean())
            else:
                self._result[key_s].zero_()
        return self._result


def modify_reward(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    reward_term = env.reward_manager.get_term_cfg(reward_term)
    if reward_term.weight == 0.0:
        return
    success_monitor = getattr(env.curriculum_manager.cfg, "terrain_levels").func.success_monitor
    success_rate = success_monitor.get_success_rate().mean()
    if success_rate > 0.4 and env.common_step_counter > 100:
        reward_term.weight = 0.0
