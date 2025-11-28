# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.sim as sim_utils

from isaaclab.managers import ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands import RelativeStateCommand


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
        debug_vis = cfg.params.get("debug_vis", True)

        self.env = env
        self.goal_term: RelativeStateCommand = env.command_manager.get_term("goal_point")

        # Disable term's own resampling: curriculum owns cmd_indices AND cmd_buf writes
        self.goal_term.resample_indices = lambda env_ids: None

        # Discrete command table size
        self.num_discrete_cmd = int(self.goal_term.spec.num_descretized_cmd)

        # Success monitor over discrete command indices [0..num_discrete_cmd)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=cfg.params.get("history_len", 100),
            num_monitored_data=self.num_discrete_cmd,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        # simple result dict
        self._result: dict[str, torch.Tensor] = {
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
            # fraction of discrete commands in each success bin
            "bin_0_20_frac": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_20_40_frac": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_40_60_frac": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_60_80_frac": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_80_100_frac": torch.zeros((), dtype=torch.float, device=env.device),
            # probability mass (under sampling distribution) in each bin
            "bin_0_20_prob": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_20_40_prob": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_40_60_prob": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_60_80_prob": torch.zeros((), dtype=torch.float, device=env.device),
            "bin_80_100_prob": torch.zeros((), dtype=torch.float, device=env.device),
        }
        self.prob_mass_per_bin = torch.zeros(5, dtype=torch.float32, device=self.env.device)

        # Visualization: we can still build spawn→target lines from descretized_cmd[0:terrain_count]
        if debug_vis:
            self._init_path_visuals_from_discrete()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        debug_vis: bool = False,
        kappa: float = 2.0,
    ):
        if env_ids.numel() == 0:
            return self._result

        # 1) SUCCESS UPDATE — use cmd_indices *before* overwriting them
        prev_idx = self.goal_term.cmd_indices[env_ids]
        success = self.goal_term.get_task_success()[env_ids]
        self.success_monitor.success_update(prev_idx, success)

        # 2) SAMPLE NEXT DISCRETE COMMANDS
        choices, probs = self.success_monitor.sample_by_target_rate(env_ids, target=0.33, kappa=kappa, return_probs=True)
        self.goal_term.cmd_indices[env_ids] = choices.to(self.goal_term.cmd_indices.dtype)

        # 3) APPLY DISCRETE COMMAND ROWS TO cmd_buf[0] AND cmd_mask
        rows = self.goal_term.spec.descretized_cmd[choices]        # [len(env_ids), 15]
        env.scene.terrain.env_origins.index_copy_(0, env_ids, rows[:, 0:3])

        # 4) LOGGING / VISUALIZATION
        success_rates = self.success_monitor.get_success_rate()  # [num_discrete_cmd]
        self._result["all_success"].copy_(success_rates.mean())

        # Vectorized bin stats
        bin_ids = (success_rates * 5.0).floor().to(torch.long).clamp_(min=0, max=4)
        counts = torch.bincount(bin_ids, minlength=5).to(torch.float32)
        frac_per_bin = counts / float(success_rates.numel())
        self.prob_mass_per_bin.zero_()
        self.prob_mass_per_bin.scatter_add_(0, bin_ids, probs)

        bin_names = ["bin_0_20", "bin_20_40", "bin_40_60", "bin_60_80", "bin_80_100",]
        for i, name in enumerate(bin_names):
            self._result[f"{name}_frac"].copy_(frac_per_bin[i])
            self._result[f"{name}_prob"].copy_(self.prob_mass_per_bin[i])

        if debug_vis and hasattr(self, "frame_visualizer"):
            self._recolor_lines(success_rates)

        return self._result

    def _get_connecting_lines(self, start_pos: torch.Tensor, end_pos: torch.Tensor):
        v = end_pos - start_pos                    # [N,3]
        l = v.norm(2, dim=-1).clamp_min(1e-12)     # [N]
        p = (start_pos + end_pos) * 0.5            # [N,3]
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(v)
        b = v / l.unsqueeze(-1)                    # normalized direction
        c = torch.cross(z, b, dim=-1)
        w = 1.0 + (z * b).sum(-1, keepdim=True)
        q = torch.cat([w, c], dim=-1)              # [N,4] (w, x, y, z)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return p, q, l

    def _init_path_visuals_from_discrete(self) -> None:
        rows = self.goal_term.spec.descretized_cmd           # [N,15]
        mask_pos = self.goal_term.spec.descretized_mask[:, 0:3].any(dim=-1)  # [N] bool

        # Only visualize position-like discrete commands
        line_indices = torch.arange(self.num_discrete_cmd, device=self.device)[mask_pos]  # [N_pos]
        rows_pos = rows[mask_pos]                                                     # [N_pos,15]

        start = rows_pos[:, 0:3].clone()  # spawn
        end = rows_pos[:, 3:6].clone()  # target

        Lp, Lq, Ll = self._get_connecting_lines(start, end)
        self._n_lines = Lp.size(0)

        # Remember which discrete command index corresponds to each line
        self._line_indices = line_indices  # [N_pos]

        MARKER_CFG = VisualizationMarkersCfg(
            markers={
                "line_0": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                "line_1": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.0))),
                "line_2": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.0))),
                "line_3": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.0))),
                "line_4": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.0))),
                "line_5": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0))),
                "line_6": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.0))),
                "line_7": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.7, 0.0))),
                "line_8": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.0))),
                "line_9": sim_utils.CylinderCfg(radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
            }
        )
        self.frame_visualizer = VisualizationMarkers(MARKER_CFG.replace(prim_path="/World/Visuals/CurriculumPaths"))

        Sc = torch.ones(self._n_lines, 3, device=self.device)
        Sc[:, 2] = Ll

        self._marker_idx = torch.full((self._n_lines,), 5, dtype=torch.int32, device=self.device)
        self.frame_visualizer.visualize(translations=Lp, orientations=Lq, scales=Sc, marker_indices=self._marker_idx)

    def _recolor_lines(self, success: torch.Tensor) -> None:
        """Recolor lines according to success per discrete command index."""
        if not hasattr(self, "frame_visualizer"):
            return
        # success is [num_discrete_cmd], but we only have lines for a subset:
        line_success = success[self._line_indices]  # [N_pos]

        bins = torch.clamp((line_success * 9.0).round().to(torch.int32), 0, 9)
        self._marker_idx[:] = bins
        self.frame_visualizer.visualize(marker_indices=self._marker_idx)


class terrain_spawn_goal_pair_success_rate_levels_old(ManagerTermBase):
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
        debug_vis = cfg.params.get("debug_vis", True)
        # disable resampling in the command term; curriculum will set commands directly
        self.goal_term = env.command_manager.get_term("goal_point")
        self.goal_term._resample_command = lambda env_ids: None  # type: ignore[attr-defined]

        # cache terrain layout
        self._num_levels = int(terrain.terrain_origins.shape[0])
        self._num_types = int(terrain.terrain_origins.shape[1])

        self.num_patches_spawn = int(env.scene.terrain.flat_patches["spawn"].shape[2])
        self.num_patches_target = int(env.scene.terrain.flat_patches["target"].shape[2])

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
        self._valid_spawn_flat = env.scene.terrain.flat_patches["spawn"].reshape(L * T * Ps, -1)
        self._valid_targets_flat = env.scene.terrain.flat_patches["target"].reshape(L * T * Pt, -1)

        # Preallocate reusable buffers to avoid per-step allocations
        n_types = len(self._type_names)
        self._buf_type_sums = torch.zeros(n_types, device=env.device, dtype=torch.float)
        self._buf_type_prob = torch.zeros(n_types, device=env.device, dtype=torch.float)
        self._buf_type_means = torch.zeros(n_types, device=env.device, dtype=torch.float)
        # Random heading buffer reused each call
        self._rand_heading = torch.empty(env.num_envs, device=env.device)

        # Spawn all possible spawn→target paths upfront (L*T*Ps*Pt lines + unique spawns + unique targets)
        if debug_vis:
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

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, debug_vis=False, kappa: float = 2.0, success_term: str = "success"):
        terrain: TerrainImporter = env.scene.terrain
        success_mask = env.termination_manager.get_term(success_term)[env_ids]
        self.success_monitor.success_update(self.term_samples.index_select(0, env_ids), success_mask)

        # 2) Sample next (level, type, spawn, target) aiming for balanced success
        choices, prob = self.success_monitor.sample_by_target_rate(env_ids, target=0.33, kappa=kappa, return_probs=True)
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
        self.goal_term.pos_command_w.index_copy_(0, env_ids, pos_cmd)
        # Adjust height directly (add z offset)
        self.goal_term.pos_command_w[env_ids, 2] += self.goal_term.robot.data.default_root_state[env_ids, 2]
        # Sample heading for the selected envs
        r = torch.empty(env_ids.numel(), device=self.device)
        self.goal_term.heading_command_w.index_copy_(0, env_ids, r.uniform_(*self.goal_term.cfg.ranges.heading))

        # aggregate reporting: overall mean terrain level (kept for compatibility)
        self._result["all"].copy_(terrain.terrain_levels.float().mean())

        # success rates: mean across all, and per type (avg over levels and pairs)
        success = self.success_monitor.get_success_rate()  # [L*T*Ps*Pt]
        self._result["all_success"].copy_(success.mean())
        # Recolor lines per current success rate (no extra smoothing)
        if debug_vis:
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
        Sg = self._env.scene.terrain.flat_patches["spawn"].reshape(G, Ps, 3).clone()
        Eg = self._env.scene.terrain.flat_patches["target"].reshape(G, Pt, 3).clone()
        Sg[..., 2] += 0.2
        Eg[..., 2] += 0.2
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
    if (success_rate > 0.2 and env.common_step_counter > 100) or env.common_step_counter > 20000:
        # Set weight to zero so manager skips computing it
        term_cfg.weight = 0.0
        # Additionally, replace the callable with a zero-return stub
        if hasattr(term_cfg.func, "reset"):
            # keep simple lambda style, but make signatures flexible to avoid TypeErrors
            term_cfg.func.reset = lambda *args, **kwargs: None
            term_cfg.func.__call__ = lambda *args, **kwargs: torch.zeros(env.num_envs, device=env.device)
        else:
            term_cfg.func = lambda env, **kwargs: torch.zeros(env.num_envs, device=env.device)


def stricten_success_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], term: str):
    term_cfg = env.termination_manager.get_term_cfg(term)
    success_monitor = getattr(env.curriculum_manager.cfg, "terrain_levels").func.success_monitor
    success_rate = success_monitor.get_success_rate().mean()
    if (success_rate > 0.1 and env.common_step_counter > 100):
        term_cfg.params["thresh"][2] = 0.5
        term_cfg.params["thresh"][3] = 0.5


def activate_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if env.common_step_counter < 5000 and term_cfg.weight != 0.0:
        term_cfg.weight = 0.0
    if env.common_step_counter > 5000 and term_cfg.weight == 0.0:
        term_cfg.weight = 250.0
