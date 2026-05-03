# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab_tasks.manager_based.multi_task.mdp.util import (
    BetaSamplingCfg,
    FrontierSamplingCfg,
    SuccessMonitorCfg,
    UniformSamplingCfg,
    beta_sampling_probs,
    build_knn_indices,
    frontier_sampling_probs,
    state_frontier_weights,
    uniform_sampling_probs,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import RelativeStateCommand


_BIN_NAMES = ("bin_0_20", "bin_20_40", "bin_40_60", "bin_60_80", "bin_80_100")


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
        debug_vis = cfg.params.get("debug_vis", False)

        self.env = env
        self.goal_term: RelativeStateCommand = env.command_manager.get_term("goal_point")

        # Curriculum owns selected task rows; the command term reads this
        # tensor directly and skips its internal random resampler while bound.
        self.command_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.command_indices.copy_(self.goal_term.cmd_indices)
        self.goal_term.bind_command_indices(self.command_indices)

        # Discrete command table size
        self.num_discrete_cmd = int(self.goal_term.table.num_tasks)

        # Sampling strategy + success-monitor cfg are both required preset params.
        self._sampling_cfg: UniformSamplingCfg | BetaSamplingCfg | FrontierSamplingCfg = cfg.params["sampling"]

        # Curriculum owns the success monitor + the rate tensor it writes into.
        # The decoupled-rate-tensor pattern (factory-style) lets us hand the same
        # tensor to both the monitor (writer) and the command term (reader)
        # without a copy.
        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = self.num_discrete_cmd
        monitor_cfg.device = env.device
        self.success_rate = torch.zeros(self.num_discrete_cmd, device=env.device)
        self.goal_term.success_rates = self.success_rate
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.success_rate)

        # Optional per-bin diagnostics. ``log_bin_frac`` reports the fraction of
        # discrete commands whose success rate lands in each bin; ``log_bin_prob``
        # reports the probability mass the sampler places in each bin.
        self._log_bin_frac: bool = cfg.params.get("log_bin_frac", False)
        self._log_bin_prob: bool = cfg.params.get("log_bin_prob", False)

        self._result: dict[str, torch.Tensor] = {
            "all_success": torch.zeros((), dtype=torch.float, device=env.device),
        }
        if self._log_bin_frac:
            for name in _BIN_NAMES:
                self._result[f"{name}_frac"] = torch.zeros((), dtype=torch.float, device=env.device)
        if self._log_bin_prob:
            for name in _BIN_NAMES:
                self._result[f"{name}_prob"] = torch.zeros((), dtype=torch.float, device=env.device)
            self.prob_mass_per_bin = torch.zeros(len(_BIN_NAMES), dtype=torch.float32, device=env.device)

        # Frontier sampler needs a precomputed kNN graph over the *state*
        # buffer (the underlying pool of physically-valid xy positions),
        # not over tasks. The state pool is the natural spatial domain;
        # spawn/target are just two roles a state can play in a task. This
        # makes the algorithm topology-agnostic (one spawn / many targets,
        # many spawns / one target, many of both -- all work the same).
        self._state_knn_indices: torch.Tensor | None = None
        if isinstance(self._sampling_cfg, FrontierSamplingCfg):
            state_xy = self.goal_term.table.spawn_states[:, :2]
            self._state_knn_indices = build_knn_indices(state_xy, k=self._sampling_cfg.k)

        # Visualization: build spawn→target lines from the task table
        if debug_vis:
            self._init_path_visuals_from_discrete()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        sampling: UniformSamplingCfg | BetaSamplingCfg | FrontierSamplingCfg = BetaSamplingCfg(),
        success_monitor_cfg: SuccessMonitorCfg | None = None,
        debug_vis: bool = False,
        success_term: str = "success",
    ):
        if env_ids.numel() == 0:
            return self._result

        # 1) SUCCESS UPDATE — use cmd_indices *before* overwriting them
        prev_idx = self.command_indices[env_ids]
        success = env.termination_manager.get_term(success_term)[env_ids]
        self.success_monitor.success_update(prev_idx, success)

        # 2) SAMPLE NEXT DISCRETE COMMANDS
        if isinstance(self._sampling_cfg, FrontierSamplingCfg):
            rates = eval(self._sampling_cfg.success_rate_bind)  # noqa: S307
            assert self._state_knn_indices is not None  # built in __init__
            table = self.goal_term.table
            probs = frontier_sampling_probs(
                rates,
                state_knn_indices=self._state_knn_indices,
                spawn_index=table.spawn_index,
                target_index=table.target_index,
                base=self._sampling_cfg.base,
                frontier_lambda=self._sampling_cfg.frontier_lambda,
                dilation_steps=self._sampling_cfg.dilation_steps,
                eps=self._sampling_cfg.eps,
            )
        elif isinstance(self._sampling_cfg, BetaSamplingCfg):
            rates = eval(self._sampling_cfg.success_rate_bind)  # noqa: S307
            probs = beta_sampling_probs(
                rates,
                target=self._sampling_cfg.target,
                kappa=self._sampling_cfg.kappa,
            )
        else:
            probs = uniform_sampling_probs(self.success_monitor.success_rate)
        choices = torch.multinomial(probs, len(env_ids), replacement=True)
        self.command_indices[env_ids] = choices

        # 3) LOGGING / VISUALIZATION
        success_rates = self.success_monitor.success_rate.clone()  # [num_discrete_cmd]
        self._result["all_success"].copy_(success_rates.mean())
        self._update_bin_stats(success_rates, probs)

        if debug_vis and hasattr(self, "frame_visualizer"):
            self._recolor_lines(success_rates)

        # Periodic heatmap logging
        self._log_counter = getattr(self, "_log_counter", 0) + 1
        if self._log_counter % 1000 == 0:
            self._log_terrain_heatmap(success_rates)
            self._log_spawn_scatter(success_rates, probs)

        # Frontier-bin diagnostic: bucket per-task probs by per-task
        # state-frontier weight to verify the algorithm is differentiating
        # frontier-unlearned tasks from deep-unlearned tasks at the
        # per-task level (independent of dashboard aggregation choices).
        if isinstance(self._sampling_cfg, FrontierSamplingCfg) and self._log_counter % 50 == 0:
            self._log_frontier_bins(success_rates, probs)

        return self._result

    def _log_frontier_bins(self, success_rates: torch.Tensor, probs: torch.Tensor) -> None:
        """Bucket per-task probs by per-task frontier weight and log to extras + stdout.

        Uses :func:`state_frontier_weights` (the same helper the sampler
        itself calls) to compute per-state frontier, then aggregates per
        task via the same above-mean-deviation rule the sampler uses.
        Reports (a) count, (b) mean per-task probability, and (c) total
        probability mass for each bin. When the algorithm is functional
        the mean prob increases with bin index, and the high-frontier
        bin captures a meaningful fraction of total mass.
        """
        assert self._state_knn_indices is not None
        table = self.goal_term.table
        state_s, state_frontier = state_frontier_weights(
            success_rates,
            state_knn_indices=self._state_knn_indices,
            spawn_index=table.spawn_index,
            target_index=table.target_index,
            dilation_steps=self._sampling_cfg.dilation_steps,
        )
        spawn_f = state_frontier[table.spawn_index]
        target_f = state_frontier[table.target_index]
        task_frontier = (spawn_f - spawn_f.mean()).clamp_min(0.0) + (target_f - target_f.mean()).clamp_min(0.0)
        # Also bucket by self success rate so we can split "frontier-unlearned"
        # from "borderline" within the high-frontier bin.
        task_self_s = success_rates

        bins = [
            ("ftr<0.01", 0.0, 0.01),
            ("0.01-0.05", 0.01, 0.05),
            ("0.05-0.20", 0.05, 0.20),
            ("0.20-0.50", 0.20, 0.50),
            ("0.50+", 0.50, 1.001),
        ]
        log_dict = self.env.extras.setdefault("log", {})
        rows = []
        total_p = float(probs.sum())
        for label, lo, hi in bins:
            mask = (task_frontier >= lo) & (task_frontier < hi)
            n = int(mask.sum())
            if n == 0:
                rows.append((label, 0, 0.0, 0.0, 0.0, 0.0))
                continue
            mean_p = float(probs[mask].mean())
            mass = float(probs[mask].sum())
            mean_self_s = float(task_self_s[mask].mean())
            mean_state_frontier_min = float(task_frontier[mask].min())
            rows.append((label, n, mean_p, mass, mean_self_s, mean_state_frontier_min))
            log_dict[f"Frontier/bin_{label}_count"] = float(n)
            log_dict[f"Frontier/bin_{label}_mean_prob"] = mean_p
            log_dict[f"Frontier/bin_{label}_mass"] = mass
            log_dict[f"Frontier/bin_{label}_mean_self_s"] = mean_self_s

        print(
            "[FRONTIER DIAG] step="
            f"{self.env.common_step_counter}  total_p={total_p:.3f}  "
            f"state_frontier max={float(state_frontier.max()):.3f}  "
            f"state_s p10/50/90="
            f"{float(state_s.quantile(0.1)):.3f}/{float(state_s.quantile(0.5)):.3f}/{float(state_s.quantile(0.9)):.3f}",
            flush=True,
        )
        header = f"  {'bin':12s} {'count':>6s} {'mean_p':>10s} {'mass':>8s} {'mean_self_s':>11s} {'min_ftr':>8s}"
        print(header, flush=True)
        for label, n, mean_p, mass, mean_self_s, min_ftr in rows:
            print(
                f"  {label:12s} {n:6d} {mean_p:10.3e} {mass:8.3f} {mean_self_s:11.3f} {min_ftr:8.3f}",
                flush=True,
            )

    def _update_bin_stats(self, success_rates: torch.Tensor, probs: torch.Tensor) -> None:
        """Write per-bin frac/prob into ``self._result`` if their flags are enabled."""
        if not (self._log_bin_frac or self._log_bin_prob):
            return
        bin_ids = (success_rates * len(_BIN_NAMES)).floor().to(torch.long).clamp_(min=0, max=len(_BIN_NAMES) - 1)
        if self._log_bin_frac:
            counts = torch.bincount(bin_ids, minlength=len(_BIN_NAMES)).to(torch.float32)
            frac_per_bin = counts / float(success_rates.numel())
            for i, name in enumerate(_BIN_NAMES):
                self._result[f"{name}_frac"].copy_(frac_per_bin[i])
        if self._log_bin_prob:
            self.prob_mass_per_bin.zero_()
            self.prob_mass_per_bin.scatter_add_(0, bin_ids, probs)
            for i, name in enumerate(_BIN_NAMES):
                self._result[f"{name}_prob"].copy_(self.prob_mass_per_bin[i])

    def _log_terrain_heatmap(self, success_rates: torch.Tensor):
        """Generate a terrain success-rate heatmap and pass it to the logger via extras."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        table = self.goal_term.table
        offsets = table.offsets
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

        # Compute per-tile per-command-type success rate via scatter_mean over tile_index.
        # Tile occupancy is ragged (cells have varying state counts), so we can't reshape.
        tile_rates = torch.zeros(num_cmds, num_rows, num_cols, device=self.device)
        tile_index = table.tile_index
        for cmd_id in range(num_cmds):
            start = int(offsets[cmd_id])
            end = int(offsets[cmd_id + 1])
            if end <= start:
                continue
            block = success_rates[start:end]
            tiles = tile_index[start:end]
            sums = torch.zeros(n_tiles, device=self.device)
            counts = torch.zeros(n_tiles, device=self.device)
            sums.scatter_add_(0, tiles, block)
            counts.scatter_add_(0, tiles, torch.ones_like(block))
            per_tile = (sums / counts.clamp_min(1.0)).view(num_rows, num_cols)
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

    def _log_spawn_scatter(self, success_rates: torch.Tensor, probs: torch.Tensor) -> None:
        """Per-patch diagnostics scatter — four panels over the terrain.

        Each panel shows the same per-patch geometry (one dot per spawn/target
        patch in world xy) but colors the dots by a different per-patch quantity:

        1. **Success rate** — current rolling success of every command that uses
           this patch as either spawn or target.
        2. **Sampling probability** — total probability mass the curriculum
           sampler currently places on commands that touch this patch.
        3. **State frontier** *(when frontier sampling is active)* — the
           per-state ``state_frontier`` value the sampler uses, surfaced
           directly from :func:`state_frontier_weights`. Falls back to
           "Sampling prob (spawn only)" for non-frontier samplers.
        4. **Δ success since last log** — change in (1) versus the previous
           call to this method, so improvement vs. regression is visible
           directly.

        Aggregation is two GPU ``scatter_add_`` passes per metric (microseconds);
        rendering goes through the shared :class:`ScatterDashboard2D` utility.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        from isaaclab_tasks.manager_based.multi_task.utils.visualization import (
            PanelSpec,
            ScatterDashboard2D,
            aggregate_endpoints,
        )

        table = self.goal_term.table
        n_patches = int(table.spawn_states.shape[0])
        endpoints = (table.spawn_index, table.target_index)

        # Lazy: GPU buffers, dashboard geometry, prev-rate cache, terrain bg.
        if not hasattr(self, "_patch_sums"):
            self._patch_sums = torch.zeros(n_patches, device=self.device)
            self._patch_counts = torch.zeros(n_patches, device=self.device)
            self._patch_probs = torch.zeros(n_patches, device=self.device)
            self._patch_probs_counts = torch.zeros(n_patches, device=self.device)
            # Target-only buffers for FrontierSamplingCfg-style signals that
            # live in target space; spawn-side aggregation dilutes them.
            self._patch_probs_target = torch.zeros(n_patches, device=self.device)
            self._patch_probs_target_counts = torch.zeros(n_patches, device=self.device)
            self._prev_patch_rates = torch.zeros(n_patches, device=self.device)
            self._first_scatter_log = True
        if not hasattr(self, "_dashboard"):
            bg_image, bg_extent = self._render_terrain_background()
            self._dashboard = ScatterDashboard2D(
                positions=table.spawn_states[:, :2].detach().cpu().numpy(),
                background_image=bg_image,
                background_extent=bg_extent,
            )

        # GPU per-patch aggregation: success and probability, both with spawn
        # and target endpoint passes.
        sums, counts = aggregate_endpoints(
            success_rates,
            endpoints,
            n_patches,
            sums_buf=self._patch_sums,
            counts_buf=self._patch_counts,
        )
        prob_sums, prob_counts = aggregate_endpoints(
            probs,
            endpoints,
            n_patches,
            sums_buf=self._patch_probs,
            counts_buf=self._patch_probs_counts,
        )
        # State-pool diagnostic: when frontier sampling is active, show the
        # per-state frontier weight directly via the shared helper that the
        # sampler itself uses, so the panel can never silently diverge from
        # the algorithm. ``state_frontier`` lives natively on the patch
        # grid (one value per ``spawn_states`` row), so no aggregation is
        # needed -- this surfaces the spatial signal without the
        # spawn/target endpoint averaging that dilutes the "Sampling
        # probability" panel above.
        if isinstance(self._sampling_cfg, FrontierSamplingCfg) and self._state_knn_indices is not None:
            _, state_frontier_t = state_frontier_weights(
                success_rates,
                state_knn_indices=self._state_knn_indices,
                spawn_index=table.spawn_index,
                target_index=table.target_index,
                dilation_steps=self._sampling_cfg.dilation_steps,
            )
            # Track which states have any task touching them, for valid_mask.
            self._patch_probs_target_counts.zero_()
            ones = torch.ones_like(success_rates)
            self._patch_probs_target_counts.scatter_add_(0, table.spawn_index, ones)
            self._patch_probs_target_counts.scatter_add_(0, table.target_index, ones)
            spatial_panel_title = "State frontier (s_dil − s_state)"
            spatial_panel_values = state_frontier_t.cpu().numpy()
            spatial_valid = self._patch_probs_target_counts.cpu().numpy() > 0
        else:
            # Fall back to spawn-aggregated probability for non-frontier
            # samplers (Beta / Uniform have no spatial signal to surface).
            prob_spawn_sums, prob_spawn_counts = aggregate_endpoints(
                probs,
                (table.spawn_index,),
                n_patches,
                sums_buf=self._patch_probs_target,
                counts_buf=self._patch_probs_target_counts,
            )
            prob_spawn_sums = prob_spawn_sums / prob_spawn_counts.clamp_min(1.0)
            spatial_panel_title = "Sampling prob (spawn only)"
            spatial_panel_values = prob_spawn_sums.cpu().numpy()
            spatial_valid = prob_spawn_counts.cpu().numpy() > 0
        # Mean-aggregate (matching the success-rate panel) so a patch touched
        # by ``n_c`` tasks doesn't visually dominate one touched by 1 task —
        # the "I'm a convergence point" amplification under sum aggregation
        # crushes everything else into purple at the start of training and
        # makes the panel hard to read across topologies.
        prob_sums = prob_sums / prob_counts.clamp_min(1.0)
        rates_t = sums / counts.clamp_min(1.0)
        delta_t = rates_t - self._prev_patch_rates
        if self._first_scatter_log:
            delta_t = torch.zeros_like(rates_t)
            self._first_scatter_log = False
        self._prev_patch_rates.copy_(rates_t)

        rates = rates_t.cpu().numpy()
        delta = delta_t.cpu().numpy()
        prob_per_patch = prob_sums.cpu().numpy()
        valid = counts.cpu().numpy() > 0

        n_total = int(valid.sum())
        mean_rate = float(rates[valid].mean()) if n_total else 0.0
        mean_delta = float(delta[valid].mean()) if n_total else 0.0
        prob_max = max(float(prob_per_patch[valid].max()) if n_total else 1.0, 1e-9)
        spatial_max = max(float(spatial_panel_values[spatial_valid].max()) if int(spatial_valid.sum()) else 1.0, 1e-9)
        delta_range = max(float(np.abs(delta[valid]).max()) if n_total else 0.0, 0.05)

        rdylgn = plt.get_cmap("RdYlGn")
        viridis = plt.get_cmap("viridis")

        panels = [
            PanelSpec(
                values=rates,
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
                title=f"Success rate (step {self.env.common_step_counter})",
                legend_entries=[("1.0", rdylgn(1.0)), ("0.5", rdylgn(0.5)), ("0.0", rdylgn(0.0))],
                stats_text=f"N={n_total}\nmean={mean_rate:.3f}",
            ),
            PanelSpec(
                values=prob_per_patch,
                cmap="viridis",
                vmin=0.0,
                vmax=prob_max,
                title="Sampling probability",
                legend_entries=[
                    (f"{prob_max:.2e}", viridis(1.0)),
                    (f"{prob_max / 2:.2e}", viridis(0.5)),
                    ("0", viridis(0.0)),
                ],
                stats_text=f"N={n_total}\nsum={float(prob_per_patch[valid].sum()):.3f}",
            ),
            PanelSpec(
                values=spatial_panel_values,
                cmap="viridis",
                vmin=0.0,
                vmax=spatial_max,
                title=spatial_panel_title,
                legend_entries=[
                    (f"{spatial_max:.2e}", viridis(1.0)),
                    (f"{spatial_max / 2:.2e}", viridis(0.5)),
                    ("0", viridis(0.0)),
                ],
                stats_text=f"N={int(spatial_valid.sum())}\nmax={spatial_max:.3e}",
            ),
            PanelSpec(
                values=delta,
                cmap="RdYlGn",
                vmin=-delta_range,
                vmax=delta_range,
                title="Δ success rate vs last log",
                legend_entries=[
                    (f"+{delta_range:.2f}", rdylgn(1.0)),
                    ("0", rdylgn(0.5)),
                    (f"-{delta_range:.2f}", rdylgn(0.0)),
                ],
                stats_text=f"N={n_total}\nmean Δ={mean_delta:+.4f}",
            ),
        ]
        img = self._dashboard.render(panels, valid_mask=valid)
        self.env.extras.setdefault("log_images", {})["Curriculum/spawn_scatter"] = img

    def _render_terrain_background(self):
        """One-time top-down raycast of the ground mesh into a 2D heightmap.

        Builds a warp mesh from ``env.scene.terrain.terrain_mesh`` and casts a
        1024×1024 grid of downward rays to recover the topmost surface, so
        overhanging features (beams, floating islands) shadow correctly
        without needing per-pixel max-z bookkeeping. Returns ``(image, extent)``
        for ``ax.imshow(..., extent=...)``; ``(None, None)`` if the terrain has
        no mesh (e.g. plane-only scenes).
        """
        import numpy as np

        from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

        terrain_mesh = getattr(self.env.scene.terrain, "terrain_mesh", None)
        if terrain_mesh is None or terrain_mesh.vertices.shape[0] == 0:
            return None, None

        verts = np.asarray(terrain_mesh.vertices, dtype=np.float32)
        faces = np.asarray(terrain_mesh.faces, dtype=np.int32)
        xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
        ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
        zmax = float(verts[:, 2].max())

        # Crop the flat ``border_width`` perimeter — no patches live there and
        # it just compresses the active tile grid into a smaller central region.
        border = float(getattr(self.env.scene.terrain.cfg.terrain_generator, "border_width", 0.0))
        if border > 0.0:
            xmin += border
            xmax -= border
            ymin += border
            ymax -= border

        H, W = 1024, 1024
        xs = np.linspace(xmin, xmax, W, dtype=np.float32)
        ys = np.linspace(ymin, ymax, H, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        starts = np.stack(
            [
                grid_x.ravel(),
                grid_y.ravel(),
                np.full(H * W, zmax + 1.0, dtype=np.float32),
            ],
            axis=-1,
        )
        dirs = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (H * W, 1))
        starts_t = torch.from_numpy(starts).to(self.device)
        dirs_t = torch.from_numpy(dirs).to(self.device)

        wp_mesh = convert_to_warp_mesh(verts, faces, device=str(self.device))
        hits, _, _, _ = raycast_mesh(starts_t, dirs_t, wp_mesh, max_dist=zmax + 100.0)
        z = hits[:, 2].view(H, W).cpu().numpy()
        # Misses come back as +inf; convert to NaN so the colormap shows them transparent.
        z = np.where(np.isfinite(z), z, np.nan)

        return z, (xmin, xmax, ymin, ymax)

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

        table = self.goal_term.table
        mask_pos = table.task_mask[:, 0:3].any(dim=-1)

        line_indices = torch.arange(self.num_discrete_cmd, device=self.device)[mask_pos]

        spawn_idx = table.spawn_index[mask_pos]
        target_idx = table.target_index[mask_pos]
        start = table.spawn_states[spawn_idx, :3].clone()
        end = table.spawn_states[target_idx, :3].clone()

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
        self.frame_visualizer = VisualizationMarkers(MARKER_CFG.replace(prim_path="/Visuals/Command/curriculum_paths"))

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
