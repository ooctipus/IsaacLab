# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import isaaclab.sim as sim_utils

from isaaclab.managers import ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import normalize, quat_from_angle_axis
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
        self._type_counts = torch.bincount(self._col_to_type_idx, minlength=len(self._type_names))
        self._type_counts = torch.bincount(self._col_to_type_idx, minlength=len(self._type_names))
        # Precompute counts per aggregated terrain type (used for fast reductions)
        self._type_counts = torch.bincount(self._col_to_type_idx, minlength=len(self._type_names))

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
        # Convert indices to tensor on device once for all subsequent indexing ops
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        command = env.command_manager.get_command("goal_point")
        distance = command.index_select(0, env_ids_t)[:, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples.index_select(0, env_ids_t), distance < 0.5)

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
        # Convert indices to tensor on device once for all subsequent indexing ops
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        command = env.command_manager.get_command("goal_point")
        distance = command.index_select(0, env_ids_t)[:, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples.index_select(0, env_ids_t), distance < 0.5)

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
        self.goal_term = env.command_manager.get_term("goal_point")
        self.goal_term._resample_command = lambda env_ids: None  # type: ignore[attr-defined]

        # cache terrain layout
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])

        self.num_patches_spawn = int(self.goal_term.valid_spawn.shape[2])
        self.num_patches_target = int(self.goal_term.valid_targets.shape[2])

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
        }
        for name in self._type_names:
            self._result[f"{name}_success"] = torch.zeros((), dtype=torch.float, device=env.device)
            # per-type sampling probability mass (sum of probabilities over all (levels, spawns, targets) columns)
            self._result[f"{name}_sample_prob"] = torch.zeros((), dtype=torch.float, device=env.device)

        # Precompute flattened views for fast gathers
        L, T = self._num_levels, self._num_types
        Ps, Pt = self.num_patches_spawn, self.num_patches_target
        self._valid_spawn_flat = self.goal_term.valid_spawn.reshape(L * T * Ps, -1)
        self._valid_targets_flat = self.goal_term.valid_targets.reshape(L * T * Pt, -1)

        # Preallocate reusable buffers to avoid per-step allocations
        n_types = len(self._type_names)
        self._buf_type_sums = torch.zeros(n_types, device=env.device, dtype=torch.float)
        self._buf_type_prob = torch.zeros(n_types, device=env.device, dtype=torch.float)
        self._buf_type_means = torch.zeros(n_types, device=env.device, dtype=torch.float)
        # Random heading buffer reused each call
        self._rand_heading = torch.empty(env.num_envs, device=env.device)

        # Spawn all possible spawn→target paths upfront (L*T*Ps*Pt lines + unique spawns + unique targets)
        self._init_path_visuals()

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
        # Cache counts per terrain type for grouped reductions
        self._type_counts = torch.bincount(self._col_to_type_idx, minlength=len(self._type_names))

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        terrain: TerrainImporter = env.scene.terrain
        goal_term = self.goal_term

        command = env.command_manager.get_command("goal_point")
        distance = command.index_select(0, env_ids)[:, :3].norm(2, dim=1)
        self.success_monitor.success_update(self.term_samples.index_select(0, env_ids), distance < 0.5)

        # 2) Sample next (level, type, spawn, target) aiming for balanced success
        choices, prob = self.success_monitor.sample_by_target_rate(env_ids, target=0.33, kappa=2, return_probs=True)
        # In-place index copy to avoid temporary tensors
        self.term_samples.index_copy_(0, env_ids, choices.to(self.term_samples.dtype))

        # 3) Decode flattened indices -> (level, type, spawn_id, target_id)
        L, T, Ps, Pt = self._num_levels, self._num_types, self.num_patches_spawn, self.num_patches_target
        # Decode flattened indices -> level, type, spawn_id, target_id (vectorized)
        flat = choices.to(torch.long)
        rem, target_id = torch.div(flat, Pt, rounding_mode='floor'), torch.remainder(flat, Pt)
        rem, spawn_id = torch.div(rem, Ps, rounding_mode='floor'), torch.remainder(rem, Ps)
        chosen_level, chosen_type = torch.div(rem, T, rounding_mode='floor'), torch.remainder(rem, T)

        # 4) Update env origins (set to spawn location) and terrain indicators
        # Use flattened gather to reduce advanced-indexing overhead
        spawn_lin = (chosen_level * (T * Ps) + chosen_type * Ps + spawn_id).to(torch.long)
        spawn_w = self._valid_spawn_flat.index_select(0, spawn_lin)
        terrain.env_origins.index_copy_(0, env_ids, spawn_w)
        terrain.terrain_levels.index_copy_(0, env_ids, chosen_level)
        terrain.terrain_types.index_copy_(0, env_ids, chosen_type)

        # 5) Set goal target directly from valid targets and adjust height
        target_lin = (chosen_level * (T * Pt) + chosen_type * Pt + target_id).to(torch.long)
        pos_cmd = self._valid_targets_flat.index_select(0, target_lin)
        goal_term.pos_command_w.index_copy_(0, env_ids, pos_cmd)
        # Adjust height directly (add z offset)
        goal_term.pos_command_w[env_ids, 2] += goal_term.robot.data.default_root_state[env_ids, 2]
        # Sample heading for the selected envs
        r = torch.empty(env_ids.numel(), device=self.device)
        goal_term.heading_command_w.index_copy_(0, env_ids, r.uniform_(*goal_term.cfg.ranges.heading))

        # aggregate reporting: overall mean terrain level (kept for compatibility)
        self._result["all"].copy_(terrain.terrain_levels.float().mean())

        # success rates: mean across all, and per type (avg over levels and pairs)
        success = self.success_monitor.get_success_rate()  # [L*T*Ps*Pt]
        self._result["all_success"].copy_(success.mean())
        # Recolor lines per current success rate (no extra smoothing)
        self._recolor_lines(success)
        # Per-type success via grouped reduction (avoid Python looped masking in reduction)
        per_col_success = success.view(L, T, Ps, Pt).mean(dim=(0, 2, 3))  # [T]
        self._buf_type_sums.zero_()
        self._buf_type_sums.index_add_(0, self._col_to_type_idx, per_col_success)
        means = self._buf_type_sums / self._type_counts.clamp_min(1).to(self._buf_type_sums.dtype)
        for i, name in enumerate(self._type_names):
            self._result[f"{name}_success"].copy_(means[i])

        # sampling probability logs: mass per terrain type (sum over levels, spawn/target pairs)
        # prob is a distribution over all (L, T, Ps, Pt) partitions and sums to 1.
        per_col_prob_mass = prob.view(L, T, Ps, Pt).sum(dim=(0, 2, 3))  # [T] columns
        self._buf_type_prob.zero_()
        self._buf_type_prob.index_add_(0, self._col_to_type_idx, per_col_prob_mass)
        for i, name in enumerate(self._type_names):
            self._result[f"{name}_sample_prob"].copy_(self._buf_type_prob[i])

        return self._result

    def _get_connecting_lines(self, start_pos: torch.Tensor, end_pos: torch.Tensor):
        v = end_pos - start_pos
        l = v.norm(2, dim=-1).clamp_min(1e-12)
        p = (start_pos + end_pos) * 0.5
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(v)
        b = v / l.unsqueeze(-1)
        c = torch.cross(z, b, dim=-1)
        w = 1.0 + (z * b).sum(-1, keepdim=True)
        q = torch.cat([w, c], dim=-1)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return p, q, l

    def _init_path_visuals(self) -> None:
        # Visualization markers: target, spawn, and 10 line color bins (red->green)
        FRAME_MARKER_CFG = VisualizationMarkersCfg(
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "spawn": sim_utils.CuboidCfg(
                    size=(0.09, 0.09, 0.09),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                # line color bins (indices 2..11)
                "line_0": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                "line_1": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8889, 0.1111, 0.0))),
                "line_2": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7778, 0.2222, 0.0))),
                "line_3": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6667, 0.3333, 0.0))),
                "line_4": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5556, 0.4444, 0.0))),
                "line_5": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4444, 0.5556, 0.0))),
                "line_6": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3333, 0.6667, 0.0))),
                "line_7": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2222, 0.7778, 0.0))),
                "line_8": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1111, 0.8889, 0.0))),
                "line_9": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
            }
        )
        self.frame_visualizer = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/CurriculumPaths"))

        L, T, Ps, Pt = self._num_levels, self._num_types, self.num_patches_spawn, self.num_patches_target
        G = L * T
        Sg = self.goal_term.valid_spawn.reshape(G, Ps, 3).clone(); Eg = self.goal_term.valid_targets.reshape(G, Pt, 3).clone()
        Sg[..., 2] += 0.2; Eg[..., 2] += 0.2
        start = Sg[:, :, None, :].expand(G, Ps, Pt, 3).reshape(-1, 3); end = Eg[:, None, :, :].expand(G, Ps, Pt, 3).reshape(-1, 3)
        Lp, Lq, Ll = self._get_connecting_lines(start, end)
        self._n_spawn, self._n_target, self._n_lines = G * Ps, G * Pt, Lp.size(0)
        Tr = torch.cat([Sg.reshape(-1, 3), Eg.reshape(-1, 3), Lp], 0)
        Or = torch.cat([torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self._n_spawn + self._n_target, 1), Lq], 0)
        Sc = torch.ones(self._n_spawn + self._n_target + self._n_lines, 3, device=self.device); Sc[-self._n_lines:, 2] = Ll
        base = torch.empty(self._n_spawn + self._n_target + self._n_lines, dtype=torch.int32, device=self.device)
        base[:self._n_spawn] = 1; base[self._n_spawn:self._n_spawn + self._n_target] = 0; base[self._n_spawn + self._n_target:] = 2
        self._marker_indices_base = base
        self.frame_visualizer.visualize(translations=Tr, orientations=Or, scales=Sc, marker_indices=base)

    def _recolor_lines(self, success: torch.Tensor) -> None:
        if not hasattr(self, "frame_visualizer"):
            return
        bins = torch.clamp((success * 9.0).round().to(torch.int32), 0, 9)
        self._marker_indices_base[self._n_spawn + self._n_target :] = 2 + bins
        self.frame_visualizer.visualize(marker_indices=self._marker_indices_base)


def skip_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if term_cfg.weight == 0.0:
        return
    success_monitor = getattr(env.curriculum_manager.cfg, "terrain_levels").func.success_monitor
    success_rate = success_monitor.get_success_rate().mean()
    if (success_rate > 0.1 and env.common_step_counter > 100) or env.common_step_counter > 15000:
        # Set weight to zero so manager skips computing it
        term_cfg.weight = 0.0
        # Additionally, replace the callable with a zero-return stub
        if hasattr(term_cfg.func, "reset"):
            # keep simple lambda style, but make signatures flexible to avoid TypeErrors
            term_cfg.func.reset = lambda *args, **kwargs: None
            term_cfg.func.__call__ = lambda *args, **kwargs: torch.zeros(env.num_envs, device=env.device)
        else:
            term_cfg.func = lambda env, **kwargs: torch.zeros(env.num_envs, device=env.device)
