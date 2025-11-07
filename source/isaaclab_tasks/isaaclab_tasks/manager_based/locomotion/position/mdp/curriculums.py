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
        # cache terrain layout
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])

        # disable resampling in the command term; curriculum will set commands directly
        goal_term = env.command_manager.get_term("goal_point")
        # prevent the command term from changing the target
        goal_term._resample_command = lambda env_ids: None  # type: ignore[attr-defined]

        # number of valid flat patches per (level, type)
        self.num_patches = int(goal_term.valid_targets.shape[2])

        # success monitor tracks each (level, type, patch_id)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self._num_levels * self._num_types * self.num_patches,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        # store sampled (level, type, patch) as a flattened index in [0, L*T*P)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        # initialize mapping that relates terrain columns to sub-terrain keys (for logging)
        self._init_type_mapping(terrain)

        # pre-allocate result dictionary to avoid per-step allocations
        self._result: dict[str, torch.Tensor] = {
            "all": torch.zeros((), dtype=torch.float, device=env.device),
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
            # average world-distance from env origin to sampled goal
            "avg_goal_distance": torch.zeros((), dtype=torch.float, device=env.device),
        }
        for name in self._type_names:
            self._result[f"{name}_success"] = torch.zeros((), dtype=torch.float, device=env.device)
            self._result[f"{name}_goal_dist"] = torch.zeros((), dtype=torch.float, device=env.device)
        # per-level goal distance logs
        for lvl in range(self._num_levels):
            self._result[f"level_{lvl}_goal_dist"] = torch.zeros((), dtype=torch.float, device=env.device)

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
        goal_term = env.command_manager.get_term("goal_point")

        # 1) Log success for current assignments (distance to goal in base frame)
        command = env.command_manager.get_command("goal_point")
        distance = command[env_ids, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples[env_ids], distance < 0.5)

        # 2) Sample next (level, type, patch) assignments targeting balanced success
        choices = self.success_monitor.sample_by_target_rate(env_ids, target=0.5, kappa=1)
        self.term_samples[env_ids] = choices.to(self.term_samples.dtype)

        # 3) Decode flattened indices -> (level, type, patch)
        L, T, P = self._num_levels, self._num_types, self.num_patches
        flat = self.term_samples[env_ids]
        patch_id = flat % P
        rem = flat // P
        chosen_type = rem % T
        chosen_level = rem // T

        # 4) Update env origins and terrain indicators
        terrain.env_origins[env_ids] = terrain.terrain_origins[chosen_level, chosen_type]
        terrain.terrain_levels[env_ids] = chosen_level
        terrain.terrain_types[env_ids] = chosen_type

        # 5) Set command target directly from valid targets and adjust height
        goal_term.pos_command_w[env_ids] = goal_term.valid_targets[chosen_level, chosen_type, patch_id]
        goal_term.pos_command_w[env_ids, 2] += goal_term.robot.data.default_root_state[env_ids, 2]
        r = torch.empty(len(env_ids), device=self.device)
        goal_term.heading_command_w[env_ids] = r.uniform_(*goal_term.cfg.ranges.heading)

        # aggregate reporting: overall mean terrain level (kept for compatibility)
        self._result["all"].copy_(terrain.terrain_levels.float().mean())

        # success rates: mean across all, and per type (avg over levels and patches)
        success = self.success_monitor.get_success_rate()  # [L*T*P]
        self._result["all_success"].copy_(success.mean())
        per_col_success = success.view(L, T, P).mean(dim=(0, 2))  # [T]
        for i, name in enumerate(self._type_names):
            key_succ = f"{name}_success"
            col_mask = self._col_to_type_idx == i
            if torch.any(col_mask):
                self._result[key_succ].copy_(per_col_success[col_mask].mean())
            else:
                self._result[key_succ].zero_()

        # goal distance logs (world frame): avg overall, per-type, and per-level
        d_all = (goal_term.pos_command_w[:, :2] - env.scene.env_origins[:, :2]).norm(2, dim=1)
        self._result["avg_goal_distance"].copy_(d_all.mean())
        # per-type
        for i, name in enumerate(self._type_names):
            key_gd = f"{name}_goal_dist"
            mask = self._col_to_type_idx[terrain.terrain_types] == i
            if torch.any(mask):
                self._result[key_gd].copy_(d_all[mask].mean())
            else:
                self._result[key_gd].zero_()


class terrain_spawn_goal_success_rate_levels(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain: TerrainImporter = env.scene.terrain
        # cache terrain layout
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])

        # disable resampling in the command term; curriculum will set commands directly
        goal_term = env.command_manager.get_term("goal_point")
        # prevent the command term from changing the target
        goal_term._resample_command = lambda env_ids: None  # type: ignore[attr-defined]

        # number of valid flat patches per (level, type)
        self.num_patches = int(goal_term.valid_targets.shape[2])

        # success monitor tracks each (level, type, patch_id)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self._num_levels * self._num_types * self.num_patches,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        # store sampled (level, type, patch) as a flattened index in [0, L*T*P)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        # initialize mapping that relates terrain columns to sub-terrain keys (for logging)
        self._init_type_mapping(terrain)

        # pre-allocate result dictionary to avoid per-step allocations
        self._result: dict[str, torch.Tensor] = {
            "all": torch.zeros((), dtype=torch.float, device=env.device),
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
            # average world-distance from env origin to sampled goal
            "avg_goal_distance": torch.zeros((), dtype=torch.float, device=env.device),
        }
        for name in self._type_names:
            self._result[f"{name}_success"] = torch.zeros((), dtype=torch.float, device=env.device)
            self._result[f"{name}_goal_dist"] = torch.zeros((), dtype=torch.float, device=env.device)

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
        goal_term = env.command_manager.get_term("goal_point")

        # 1) Log success for current assignments (distance to goal in base frame)
        command = env.command_manager.get_command("goal_point")
        distance = command[env_ids, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples[env_ids], distance < 0.5)

        # 2) Sample next (level, type, patch) assignments targeting balanced success
        choices = self.success_monitor.sample_by_target_rate(env_ids, target=0.5, kappa=1)
        self.term_samples[env_ids] = choices.to(self.term_samples.dtype)

        # 3) Decode flattened indices -> (level, type, patch)
        L, T, P = self._num_levels, self._num_types, self.num_patches
        flat = self.term_samples[env_ids]
        patch_id = flat % P
        rem = flat // P
        chosen_type = rem % T
        chosen_level = rem // T

        # 4) Update env origins and terrain indicators
        terrain.env_origins[env_ids] = terrain.terrain_origins[chosen_level, chosen_type]
        terrain.terrain_levels[env_ids] = chosen_level
        terrain.terrain_types[env_ids] = chosen_type

        # 5) Set command target directly from valid targets and adjust height
        goal_term.pos_command_w[env_ids] = goal_term.valid_targets[chosen_level, chosen_type, patch_id]
        goal_term.pos_command_w[env_ids, 2] += goal_term.robot.data.default_root_state[env_ids, 2]
        r = torch.empty(len(env_ids), device=self.device)
        goal_term.heading_command_w[env_ids] = r.uniform_(*goal_term.cfg.ranges.heading)

        # aggregate reporting: overall mean terrain level (kept for compatibility)
        self._result["all"].copy_(terrain.terrain_levels.float().mean())

        # success rates: mean across all, and per type (avg over levels and patches)
        success = self.success_monitor.get_success_rate()  # [L*T*P]
        self._result["all_success"].copy_(success.mean())
        per_col_success = success.view(L, T, P).mean(dim=(0, 2))  # [T]
        for i, name in enumerate(self._type_names):
            key_succ = f"{name}_success"
            col_mask = self._col_to_type_idx == i
            if torch.any(col_mask):
                self._result[key_succ].copy_(per_col_success[col_mask].mean())
            else:
                self._result[key_succ].zero_()

        # goal distance logs (world frame): avg overall, per-type, and per-level
        d_all = (goal_term.pos_command_w[:, :2] - env.scene.env_origins[:, :2]).norm(2, dim=1)
        self._result["avg_goal_distance"].copy_(d_all.mean())
        # per-type
        for i, name in enumerate(self._type_names):
            key_gd = f"{name}_goal_dist"
            mask = self._col_to_type_idx[terrain.terrain_types] == i
            if torch.any(mask):
                self._result[key_gd].copy_(d_all[mask].mean())
            else:
                self._result[key_gd].zero_()

        return self._result


class terrain_spawn_goal_pair_success_rate_levels(ManagerTermBase):
    """Success-rate curriculum over spawn→target patch pairs per (level, type).

    Tracks success for every possible (spawn_patch, target_patch) connection at each
    terrain (level, type). The total monitored count is:
        num_levels * num_types * num_patches_spawn * num_patches_targets

    For each sampled assignment, it sets the env origin to the sampled spawn location
    (instead of the terrain tile origin) and sets the goal command to the sampled
    target location.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain: TerrainImporter = env.scene.terrain

        # disable resampling in the command term; curriculum will set commands directly
        goal_term = env.command_manager.get_term("goal_point")
        goal_term._resample_command = lambda env_ids: None  # type: ignore[attr-defined]

        # cache terrain layout
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])

        self.num_patches_spawn = int(goal_term.valid_spawn.shape[2])
        self.num_patches_target = int(goal_term.valid_targets.shape[2])

        # success monitor tracks each (level, type, spawn_id, target_id)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self._num_levels * self._num_types * self.num_patches_spawn * self.num_patches_target,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        # store sampled (level, type, spawn_id, target_id) as a flattened index in [0, L*T*Ps*Pt)
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        # initialize mapping that relates terrain columns to sub-terrain keys (for logging)
        self._init_type_mapping(terrain)

        # pre-allocate result dictionary to avoid per-step allocations
        self._result: dict[str, torch.Tensor] = {
            "all": torch.zeros((), dtype=torch.float, device=env.device),
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
            # average world-distance from spawn to goal
            "avg_goal_distance": torch.zeros((), dtype=torch.float, device=env.device),
        }
        for name in self._type_names:
            self._result[f"{name}_success"] = torch.zeros((), dtype=torch.float, device=env.device)
            self._result[f"{name}_goal_dist"] = torch.zeros((), dtype=torch.float, device=env.device)

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
        goal_term = env.command_manager.get_term("goal_point")

        # 1) Log success for current assignments (distance to goal in base frame)
        command = env.command_manager.get_command("goal_point")
        distance = command[env_ids, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples[env_ids], distance < 0.5)

        # 2) Sample next (level, type, spawn, target) aiming for balanced success
        choices = self.success_monitor.sample_by_target_rate(env_ids, target=0.33, kappa=2)
        self.term_samples[env_ids] = choices.to(self.term_samples.dtype)

        # 3) Decode flattened indices -> (level, type, spawn_id, target_id)
        L, T, Ps, Pt = self._num_levels, self._num_types, self.num_patches_spawn, self.num_patches_target
        flat = self.term_samples[env_ids]
        target_id = flat % Pt
        rem = flat // Pt
        spawn_id = rem % Ps
        rem = rem // Ps
        chosen_type = rem % T
        chosen_level = rem // T

        # 4) Update env origins (set to spawn location) and terrain indicators
        spawn_w = goal_term.valid_spawn[chosen_level, chosen_type, spawn_id]
        terrain.env_origins[env_ids] = spawn_w
        terrain.terrain_levels[env_ids] = chosen_level
        terrain.terrain_types[env_ids] = chosen_type

        # 5) Set goal target directly from valid targets and adjust height
        goal_term.pos_command_w[env_ids] = goal_term.valid_targets[chosen_level, chosen_type, target_id]
        goal_term.pos_command_w[env_ids, 2] += goal_term.robot.data.default_root_state[env_ids, 2]
        r = torch.empty(len(env_ids), device=self.device)
        goal_term.heading_command_w[env_ids] = r.uniform_(*goal_term.cfg.ranges.heading)

        # aggregate reporting: overall mean terrain level (kept for compatibility)
        self._result["all"].copy_(terrain.terrain_levels.float().mean())

        # success rates: mean across all, and per type (avg over levels and pairs)
        success = self.success_monitor.get_success_rate()  # [L*T*Ps*Pt]
        self._result["all_success"].copy_(success.mean())
        per_col_success = success.view(L, T, Ps, Pt).mean(dim=(0, 2, 3))  # [T]
        for i, name in enumerate(self._type_names):
            key_succ = f"{name}_success"
            col_mask = self._col_to_type_idx == i
            if torch.any(col_mask):
                self._result[key_succ].copy_(per_col_success[col_mask].mean())
            else:
                self._result[key_succ].zero_()

        # goal distance logs (world frame): distance from spawn to target
        d_all = (goal_term.pos_command_w[:, :2] - env.scene.env_origins[:, :2]).norm(2, dim=1)
        self._result["avg_goal_distance"].copy_(d_all.mean())
        # per-type
        for i, name in enumerate(self._type_names):
            key_gd = f"{name}_goal_dist"
            mask = self._col_to_type_idx[terrain.terrain_types] == i
            if torch.any(mask):
                self._result[key_gd].copy_(d_all[mask].mean())
            else:
                self._result[key_gd].zero_()

        return self._result


def modify_reward(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    reward_term = env.reward_manager.get_term_cfg(reward_term)
    if reward_term.weight == 0.0:
        return
    success_monitor = getattr(env.curriculum_manager.cfg, "terrain_levels").func.success_monitor
    success_rate = success_monitor.get_success_rate().mean()
    if success_rate > 0.4 and env.common_step_counter > 100:
        reward_term.weight = 0.0
