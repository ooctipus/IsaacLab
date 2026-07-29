# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def wandb_log_image(tag: str, image: object) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log({tag: wandb.Image(image)}, step=int(getattr(wandb.run, "step", 0)))


def _wandb_is_active() -> bool:
    try:
        import wandb
    except ImportError:
        return False
    return wandb.run is not None


def terrain_success_heatmap_image(env, goal_term, success_rates: torch.Tensor):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    table = goal_term.table
    offsets = table.offsets
    cmd_names = list(goal_term.cfg.commands.keys())
    terrain_cfg = env.scene.terrain.cfg.terrain_generator
    num_rows = terrain_cfg.num_rows
    num_cols = terrain_cfg.num_cols
    n_tiles = num_rows * num_cols
    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())

    proportions = np.array([cfg.proportion for cfg in terrain_cfg.sub_terrains.values()])
    proportions /= proportions.sum()
    cum_props = np.cumsum(proportions)
    col_to_type = [int(np.min(np.where(c / num_cols + 0.001 < cum_props)[0])) for c in range(num_cols)]

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

    num_cmds = len(cmd_names)
    tile_rates = torch.zeros(num_cmds, num_rows, num_cols, device=success_rates.device)
    tile_index = table.tile_index
    for cmd_id in range(num_cmds):
        start = int(offsets[cmd_id])
        end = int(offsets[cmd_id + 1])
        if end <= start:
            continue
        block = success_rates[start:end]
        tiles = tile_index[start:end]
        sums = torch.zeros(n_tiles, device=success_rates.device)
        counts = torch.zeros(n_tiles, device=success_rates.device)
        sums.scatter_add_(0, tiles, block)
        counts.scatter_add_(0, tiles, torch.ones_like(block))
        tile_rates[cmd_id] = (sums / counts.clamp_min(1.0)).view(num_rows, num_cols)

    agg = tile_rates.mean(dim=0)

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
        _style_axis(axes[cmd_id + 1], tile_rates[cmd_id].cpu().numpy(), cmd_names[cmd_id].replace("_cmd", ""))

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Success Rate")
    fig.suptitle(f"Terrain Success (step {env.common_step_counter})", fontsize=10)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()  # pyright: ignore[reportAttributeAccessIssue]
    plt.close(fig)
    return img


def spawn_goal_scatter_image(env, goal_term, sampler, success_rates: torch.Tensor, probs: torch.Tensor, state: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    from isaaclab_tasks.core.multi_task.viz import PanelSpec, ScatterDashboard2D, aggregate_endpoints

    from .terrain_background import render_terrain_background

    table = goal_term.table
    n_patches = int(table.spawn_states.shape[0])
    endpoints = (table.spawn_index, table.target_index)
    device = success_rates.device

    if "patch_sums" not in state:
        state["patch_sums"] = torch.zeros(n_patches, device=device)
        state["patch_counts"] = torch.zeros(n_patches, device=device)
        state["patch_probs"] = torch.zeros(n_patches, device=device)
        state["patch_probs_counts"] = torch.zeros(n_patches, device=device)
        state["prev_patch_rates"] = torch.zeros(n_patches, device=device)
        state["target_mask"] = torch.zeros(n_patches, device=device, dtype=torch.bool)
        state["target_mask"][table.target_index] = True
        state["first_scatter_log"] = True
    edge_ones = state.get("edge_ones")
    if edge_ones is None or edge_ones.shape != success_rates.shape or edge_ones.device != device:
        edge_ones = torch.ones_like(success_rates)
        state["edge_ones"] = edge_ones
    if "dashboard" not in state:
        bg_image, bg_extent = render_terrain_background(env.scene.terrain, device=env.device)
        state["dashboard"] = ScatterDashboard2D(
            positions=table.spawn_states[:, :2].detach().cpu().numpy(),
            background_image=bg_image,
            background_extent=bg_extent,
        )

    sums, counts = aggregate_endpoints(
        success_rates,
        endpoints,
        n_patches,
        sums_buf=state["patch_sums"],
        counts_buf=state["patch_counts"],
        ones_buf=edge_ones,
    )
    prob_sums, prob_counts = aggregate_endpoints(
        probs,
        endpoints,
        n_patches,
        sums_buf=state["patch_probs"],
        counts_buf=state["patch_probs_counts"],
        ones_buf=edge_ones,
    )

    rates_t = sums / counts.clamp_min(1.0)
    prob_t = prob_sums / prob_counts.clamp_min(1.0)
    delta_t = rates_t - state["prev_patch_rates"]
    if state["first_scatter_log"]:
        delta_t = torch.zeros_like(rates_t)
        state["first_scatter_log"] = False
    state["prev_patch_rates"].copy_(rates_t)

    rates = rates_t.cpu().numpy()
    delta = delta_t.cpu().numpy()
    prob_per_patch = prob_t.cpu().numpy()
    valid = counts.cpu().numpy() > 0
    target_mask = state["target_mask"].cpu().numpy()

    n_total = int(valid.sum())
    n_targets = int(target_mask.sum())
    n_combinations = int(table.spawn_index.numel())
    count_stats = f"N={n_total}\ntargets={n_targets}\ncombos={n_combinations}"
    mean_rate = float(rates[valid].mean()) if n_total else 0.0
    mean_delta = float(delta[valid].mean()) if n_total else 0.0
    prob_max = max(float(prob_per_patch[valid].max()) if n_total else 1.0, 1e-9)
    delta_range = max(float(np.abs(delta[valid]).max()) if n_total else 0.0, 0.05)

    rdylgn = plt.get_cmap("RdYlGn")
    viridis = plt.get_cmap("viridis")
    panels = [
        PanelSpec(
            values=rates,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            title=f"Success rate (step {env.common_step_counter})",
            legend_entries=[("1.0", rdylgn(1.0)), ("0.5", rdylgn(0.5)), ("0.0", rdylgn(0.0))],
            stats_text=f"{count_stats}\nmean={mean_rate:.3f}",
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
            stats_text=f"{count_stats}\nsum={float(prob_per_patch[valid].sum()):.3f}",
            outline_mask=target_mask,
        ),
    ]

    sampler_impl = getattr(sampler, "_impl", None)
    plot_strategy_indices = getattr(sampler_impl, "_plot_strategy_indices", [])
    strategy_weights = getattr(sampler_impl, "_weights", None)
    if plot_strategy_indices and strategy_weights is not None:
        scores = sampler.scores()
        weights = strategy_weights.to(device=success_rates.device, dtype=success_rates.dtype).clamp_min(0.0)
        weighted_scores = scores * weights.view(-1, 1)
        normalizer = weighted_scores.sum() + float(getattr(sampler_impl, "eps", 0.0)) * success_rates.numel()
        attribution = weighted_scores / normalizer.clamp_min(1.0e-12)
        for strategy_idx in plot_strategy_indices:
            attr_sums, attr_counts = aggregate_endpoints(
                attribution[strategy_idx], endpoints, n_patches, ones_buf=edge_ones
            )
            attr_t = attr_sums / attr_counts.clamp_min(1.0)
            attr_values = attr_t.cpu().numpy()
            attr_valid = attr_counts.cpu().numpy() > 0
            attr_values[~attr_valid] = np.nan
            attr_max = max(float(attr_values[attr_valid].max()) if int(attr_valid.sum()) else 1.0, 1e-9)
            panels.append(
                PanelSpec(
                    values=attr_values,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=attr_max,
                    title=f"{sampler.names[strategy_idx]} attribution",
                    legend_entries=[
                        (f"{attr_max:.2e}", viridis(1.0)),
                        (f"{attr_max / 2:.2e}", viridis(0.5)),
                        ("0", viridis(0.0)),
                    ],
                    stats_text=f"{count_stats}\nmax={attr_max:.3e}",
                )
            )

    panels.append(
        PanelSpec(
            values=delta,
            cmap="RdYlGn",
            vmin=-delta_range,
            vmax=delta_range,
            title="Delta success rate vs last log",
            legend_entries=[
                (f"+{delta_range:.2f}", rdylgn(1.0)),
                ("0", rdylgn(0.5)),
                (f"-{delta_range:.2f}", rdylgn(0.0)),
            ],
            stats_text=f"{count_stats}\nmean delta={mean_delta:+.4f}",
        ),
    )
    return state["dashboard"].render(panels, valid_mask=valid, figsize=(4.5 * len(panels), 5.0), dpi=120)


def log_spawn_goal_sampler_images(
    env,
    sampler,
    success_rates: torch.Tensor,
    probs: torch.Tensor,
    state: dict | None = None,
) -> None:
    if not _wandb_is_active():
        return
    if state is None:
        state = getattr(env, "_spawn_goal_sampler_image_state", None)
        if state is None:
            state = {}
            setattr(env, "_spawn_goal_sampler_image_state", state)
    goal_term = env.command_manager.get_term("goal_point")
    heatmap = terrain_success_heatmap_image(env, goal_term, success_rates)
    env.extras.setdefault("log_images", {})["Sampler/terrain_heatmap"] = heatmap
    wandb_log_image("Sampler/terrain_heatmap", heatmap)

    scatter = spawn_goal_scatter_image(env, goal_term, sampler, success_rates, probs, state)
    env.extras.setdefault("log_images", {})["Sampler/spawn_scatter"] = scatter
    wandb_log_image("Sampler/spawn_scatter", scatter)


class SpawnGoalSamplerImageLogger:
    """Callable logger that owns sampler image cache state."""

    def __init__(
        self,
        command_name: str = "goal_point",
        terrain_heatmap_tag: str | None = "Sampler/terrain_heatmap",
        spawn_scatter_tag: str | None = "Sampler/spawn_scatter",
        log_to_extras: bool = True,
        log_to_wandb: bool = True,
    ):
        self.command_name = command_name
        self.terrain_heatmap_tag = terrain_heatmap_tag
        self.spawn_scatter_tag = spawn_scatter_tag
        self.log_to_extras = log_to_extras
        self.log_to_wandb = log_to_wandb
        self.scatter_state: dict = {}

    def __call__(self, env, sampler, success_rates: torch.Tensor, probs: torch.Tensor) -> None:
        goal_term = env.command_manager.get_term(self.command_name)
        if self.terrain_heatmap_tag is not None:
            heatmap = terrain_success_heatmap_image(env, goal_term, success_rates)
            if self.log_to_extras:
                env.extras.setdefault("log_images", {})[self.terrain_heatmap_tag] = heatmap
            if self.log_to_wandb:
                wandb_log_image(self.terrain_heatmap_tag, heatmap)

        if self.spawn_scatter_tag is not None:
            scatter = spawn_goal_scatter_image(env, goal_term, sampler, success_rates, probs, self.scatter_state)
            if self.log_to_extras:
                env.extras.setdefault("log_images", {})[self.spawn_scatter_tag] = scatter
            if self.log_to_wandb:
                wandb_log_image(self.spawn_scatter_tag, scatter)
