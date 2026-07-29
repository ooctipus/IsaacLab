# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab_tasks.core.multi_task.curriculum import SamplerCfg, StateLayoutCfg, SuccessMonitorCfg

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

    Sampling is delegated to a multi_task :class:`Sampler` built from a
    :class:`SamplerCfg`; plug in :class:`BetaSamplingStrategyCfg`,
    :class:`UniformSamplingStrategyCfg`, :class:`FrontierSamplingStrategyCfg`, or a
    weighted mix to change exploration behavior without modifying this function.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        debug_vis = cfg.params.get("debug_vis", False)

        self.env = env
        self.goal_term: RelativeStateCommand = env.command_manager.get_term("goal_point")

        # Disable term's own resampling: curriculum owns cmd_indices AND cmd_buf writes
        self.goal_term.resample_indices = lambda env_ids: None

        # Discrete command table size
        self.num_discrete_cmd = int(self.goal_term.spec.num_descretized_cmd)

        # Curriculum owns the success_rates tensor; goal_term and monitor both see it
        self.success_rates = torch.zeros(self.num_discrete_cmd, device=env.device)
        self.goal_term.success_rates = self.success_rates

        # Success monitor: writes computed rates in-place into self.success_rates
        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = self.num_discrete_cmd
        monitor_cfg.device = env.device
        monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.success_rates)

        # Sampler with pluggable strategies
        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs
        layout_cfg: StateLayoutCfg = cfg.params["layout"]
        self._sampler = self._sampling_cfg.class_type(
            self._sampling_cfg, layout_cfg.build(env), env=env, success_rates=self.success_rates
        )

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
        layout: StateLayoutCfg,
        sampling: SamplerCfg,
        success_monitor_cfg: SuccessMonitorCfg,
        success_term: str = "success",
        debug_vis: bool = False,
    ):
        if env_ids.numel() == 0:
            return self._result

        # 1) SUCCESS UPDATE — use cmd_indices *before* overwriting them
        prev_idx = self.goal_term.cmd_indices[env_ids]
        success = env.termination_manager.get_term(success_term)[env_ids]
        self.success_monitor.success_update(prev_idx, success)

        # 2) SAMPLE NEXT DISCRETE COMMANDS
        num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
        probs, choices = self._sampler.probabilities_and_sample(num_samples)
        choices = choices[: len(env_ids)]
        self.goal_term.cmd_indices[env_ids] = choices.to(self.goal_term.cmd_indices.dtype)

        # 3) UPDATE ENV ORIGINS — must happen before scene.reset() places the robot
        rows = self.goal_term.spec.descretized_cmd[choices]
        env.scene.terrain.env_origins.index_copy_(0, env_ids.long(), rows[:, 0:3])

        # 4) LOGGING / VISUALIZATION
        success_rates = self.success_rates
        self._result["all_success"].copy_(success_rates.mean())

        # Vectorized bin stats
        bin_ids = (success_rates * 5.0).floor().to(torch.long).clamp_(min=0, max=4)
        counts = torch.bincount(bin_ids, minlength=5).to(torch.float32)
        frac_per_bin = counts / float(success_rates.numel())
        self.prob_mass_per_bin.zero_()
        self.prob_mass_per_bin.scatter_add_(0, bin_ids, probs)

        bin_names = [
            "bin_0_20",
            "bin_20_40",
            "bin_40_60",
            "bin_60_80",
            "bin_80_100",
        ]
        for i, name in enumerate(bin_names):
            self._result[f"{name}_frac"].copy_(frac_per_bin[i])
            self._result[f"{name}_prob"].copy_(self.prob_mass_per_bin[i])

        if debug_vis and hasattr(self, "frame_visualizer"):
            self._recolor_lines(success_rates)

        # Periodic heatmap logging
        self._log_counter = getattr(self, "_log_counter", 0) + 1
        if self._log_counter % 1000 == 0:
            self._log_terrain_heatmap(success_rates)

        return self._result

    def _log_terrain_heatmap(self, success_rates: torch.Tensor):
        """Generate a terrain success-rate heatmap and pass it to the logger via extras."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        spec = self.goal_term.spec
        offsets = spec.descretized_cmd_offsets
        cmd_names = list(self.goal_term.cfg.commands.keys())
        num_cmds = len(cmd_names)
        terrain_cfg = self.env.scene.terrain.cfg.terrain_generator
        num_rows = terrain_cfg.num_rows
        num_cols = terrain_cfg.num_cols
        n_tiles = num_rows * num_cols
        sub_terrain_names = list(terrain_cfg.sub_terrains.keys())

        # Column-to-terrain-type mapping (mirrors TerrainGenerator._generate_curriculum_terrains)
        proportions = np.array([cfg.proportion for cfg in terrain_cfg.sub_terrains.values()])
        proportions /= proportions.sum()
        cum_props = np.cumsum(proportions)
        col_to_type = [int(np.min(np.where(c / num_cols + 0.001 < cum_props)[0])) for c in range(num_cols)]

        # Tick positions (center of each terrain-type column group) and boundary lines
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        boundaries: list[float] = []
        group_start = 0
        for col in range(1, num_cols + 1):
            if col == num_cols or col_to_type[col] != col_to_type[group_start]:
                tick_positions.append((group_start + col - 1) / 2.0)
                tick_labels.append(sub_terrain_names[col_to_type[group_start]])
                if col < num_cols:
                    boundaries.append(col - 0.5)
                group_start = col

        # Compute per-tile per-command-type success rate
        tile_rates = torch.zeros(num_cmds, num_rows, num_cols, device=self.device)
        for cmd_id in range(num_cmds):
            start = int(offsets[cmd_id])
            end = int(offsets[cmd_id + 1])
            block = success_rates[start:end]
            entries_per_tile = block.numel() // n_tiles if n_tiles > 0 else 1
            if entries_per_tile > 0 and block.numel() >= n_tiles:
                reshaped = block[: n_tiles * entries_per_tile].view(n_tiles, entries_per_tile)
                per_tile = reshaped.mean(dim=1).view(num_rows, num_cols)
                tile_rates[cmd_id] = per_tile

        agg = tile_rates.mean(dim=0)  # [num_rows, num_cols]

        fig, axes = plt.subplots(
            1,
            num_cmds + 1,
            figsize=(2.5 * (num_cmds + 1), 5),
            squeeze=False,
            layout="constrained",
        )
        axes = axes[0]

        def _style_axis(ax, data, title, show_ylabel=False):
            ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)
            for b in boundaries:
                ax.axvline(b, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
            if show_ylabel:
                ax.set_ylabel("Difficulty level")
            else:
                ax.set_yticks([])

        im = axes[0].imshow(agg.cpu().numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
        _style_axis(axes[0], agg.cpu().numpy(), "All", show_ylabel=True)

        for cmd_id in range(num_cmds):
            short_name = cmd_names[cmd_id].replace("_cmd", "")
            _style_axis(axes[cmd_id + 1], tile_rates[cmd_id].cpu().numpy(), short_name)

        fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Success Rate")
        fig.suptitle(f"Terrain Success (step {self.env.common_step_counter})", fontsize=10)

        # Render to numpy HWC array and pass through extras for synced logging
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)

        self.env.extras.setdefault("log_images", {})["Curriculum/terrain_heatmap"] = img

    def _get_connecting_lines(self, start_pos: torch.Tensor, end_pos: torch.Tensor):
        """Compute position, orientation (XYZW), and length for cylinder markers connecting start to end."""
        v = end_pos - start_pos  # [N,3]
        length = v.norm(2, dim=-1).clamp_min(1e-12)  # [N]
        p = (start_pos + end_pos) * 0.5  # [N,3]
        # Cylinder default axis is Z. Rotate Z-axis onto beam direction.
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(v)
        b = v / length.unsqueeze(-1)
        c = torch.cross(z_axis, b, dim=-1)
        w = 1.0 + (z_axis * b).sum(-1, keepdim=True)
        q_wxyz = torch.cat([w, c], dim=-1)
        q_wxyz = q_wxyz / q_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # Convert WXYZ → XYZW for Isaac Lab 3.0
        q = torch.cat([q_wxyz[:, 1:], q_wxyz[:, :1]], dim=-1)
        return p, q, length

    def _init_path_visuals_from_discrete(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        rows = self.goal_term.spec.descretized_cmd  # [N,20]
        mask_pos = self.goal_term.spec.descretized_mask[:, 0:3].any(dim=-1)  # [N] bool

        # Only visualize position-like discrete commands
        line_indices = torch.arange(self.num_discrete_cmd, device=self.device)[mask_pos]  # [N_pos]
        rows_pos = rows[mask_pos]  # [N_pos,20]

        start = rows_pos[:, 0:3].clone()  # spawn
        end = rows_pos[:, 3:6].clone()  # target

        Lp, Lq, Ll = self._get_connecting_lines(start, end)
        self._n_lines = Lp.size(0)

        # Remember which discrete command index corresponds to each line
        self._line_indices = line_indices  # [N_pos]

        MARKER_CFG = VisualizationMarkersCfg(
            markers={
                "line_0": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                ),
                "line_1": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.0))
                ),
                "line_2": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.0))
                ),
                "line_3": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.0))
                ),
                "line_4": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.0))
                ),
                "line_5": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0))
                ),
                "line_6": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.0))
                ),
                "line_7": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.7, 0.0))
                ),
                "line_8": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.0))
                ),
                "line_9": sim_utils.CylinderCfg(
                    radius=0.01, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                ),
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


def skip_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if term_cfg.weight == 0.0:
        return
    success_rate = env.command_manager.get_term("goal_point").success_rates.mean()
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
    success_rate = env.command_manager.get_term("goal_point").success_rates.mean()
    if success_rate > 0.1 and env.common_step_counter > 100:
        term_cfg.params["thresh"][2] = 0.5
        term_cfg.params["thresh"][3] = 0.5


def activate_reward_term(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term: str):
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    if env.common_step_counter < 5000 and term_cfg.weight != 0.0:
        term_cfg.weight = 0.0
    if env.common_step_counter > 5000 and term_cfg.weight == 0.0:
        term_cfg.weight = 250.0
