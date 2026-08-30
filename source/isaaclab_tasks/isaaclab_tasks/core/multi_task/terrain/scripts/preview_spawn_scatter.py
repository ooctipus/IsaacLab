# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the spawn-scatter visualization once and save the image to a PNG.

Resolves the production ``Isaac-Position-v0`` config through the same Hydra
path used by training, builds the terrain mesh + command task table directly
without launching simulation, injects a synthetic distance-to-goal ->
success-rate distribution so the colormap shows a non-trivial pattern, calls
the sampler image logger, then writes the resulting numpy image (RGB, HxWx3)
to disk via PIL.

The script bypasses wandb entirely — its output is a standalone PNG you can
inspect to see exactly what the wandb upload will look like at log time.

Run::

    ./isaaclab.sh -p \
        source/isaaclab_tasks/isaaclab_tasks/core/multi_task/terrain/scripts/preview_spawn_scatter.py \
        --headless --num_envs 16 --output /tmp/spawn_scatter_preview.png
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import isaaclab_tasks  # noqa: F401  — registers ``Isaac-Position-v0``
from isaaclab_tasks.utils.hydra import hydra_task_config

_ROBOT_PRESETS = {"anymal_c", "go2", "spot", "h1", "b2", "mewtwo"}


def _ensure_default_robot_preset(args: list[str]) -> list[str]:
    """Keep the preview script's historical ANYmal-C default."""
    preset_idx = None
    selected_presets = set()
    for idx, arg in enumerate(args):
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        if key == "presets":
            preset_idx = idx
            selected_presets.update(v.strip() for v in val.split(",") if v.strip())

    if selected_presets & _ROBOT_PRESETS:
        return args

    args = list(args)
    if preset_idx is None:
        args.append("presets=anymal_c")
    else:
        args[preset_idx] = f"{args[preset_idx]},anymal_c"
    return args


parser = argparse.ArgumentParser(description="Preview the spawn-scatter visualization to a PNG.")
parser.add_argument("--task", type=str, default="Isaac-Position-v0", help="Gym task id.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device used for terrain sampling and rendering.")
parser.add_argument("--headless", action="store_true", help="Accepted for CLI compatibility; no simulator is launched.")
parser.add_argument(
    "--output",
    type=str,
    default="/tmp/spawn_scatter_preview.png",
    help="Where to write the rendered PNG.",
)
parser.add_argument("--seed", type=int, default=0, help="Torch seed for the synthetic noise.")
args_cli, remaining_args = parser.parse_known_args()
remaining_args = _ensure_default_robot_preset(remaining_args)
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, _agent_cfg):
    import numpy as np
    import torch
    from PIL import Image

    from isaaclab.terrains.terrain_generator import TerrainGenerator

    from isaaclab_tasks.core.multi_task.curriculum import StateLayout
    from isaaclab_tasks.core.multi_task.terrain.mdp.commands.task_table_builder import build_task_table
    from isaaclab_tasks.core.multi_task.terrain.scripts.validate_spawn_points import (
        _patch_kin_with_robot,  # pyright: ignore[reportPrivateUsage]
        _resolve_robot_usd,  # pyright: ignore[reportPrivateUsage]
    )
    from isaaclab_tasks.core.multi_task.terrain.viz.sampler_images import SpawnGoalSamplerImageLogger

    torch.manual_seed(args_cli.seed)

    device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    terrain_gen_cfg = env_cfg.scene.terrain.terrain_generator
    sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
    print(
        f"[preview] terrain={terrain_gen_cfg.num_rows}x{terrain_gen_cfg.num_cols} sub_terrains={sub_terrain_names}",
        flush=True,
    )

    terrain = TerrainGenerator(cfg=terrain_gen_cfg, device=device)
    terrain_origins = torch.tensor(terrain.terrain_origins, device=device, dtype=torch.float32)

    goal_cfg = env_cfg.commands.goal_point
    table_cfg = goal_cfg.task_table
    robot_usd = _resolve_robot_usd(env_cfg.scene.robot)
    pipeline_cfg = _patch_kin_with_robot(table_cfg.pipeline_cfg, env_cfg.scene.robot, robot_usd, device)

    table_data = build_task_table(
        terrain_mesh=terrain.terrain_mesh,
        terrain_origins=terrain_origins,
        cell_size=terrain_gen_cfg.size,
        pipeline_cfg=pipeline_cfg,
        env=None,
        commands=goal_cfg.commands,
        # Only used for command masks; scatter/heatmap rendering does not read joint mask columns.
        num_joints=0,
        device=device,
        pool_spacing=table_cfg.pool_spacing,
        pool_spacing_area_divisor=table_cfg.pool_spacing_area_divisor,
        pool_sampling_size=table_cfg.pool_sampling_size,
        robot_joint_names=None,
        exclude_self_pairs=table_cfg.exclude_self_pairs,
        max_spawns_per_cell=table_cfg.max_spawns_per_cell,
        num_targets_per_cell=table_cfg.num_targets_per_cell,
    )
    table_data["task_partition"] = torch.bucketize(
        torch.arange(int(table_data["num_tasks"]), device=device),
        table_data["offsets"][1:-1],
        right=True,
    )
    table = SimpleNamespace(**table_data)
    goal_term = SimpleNamespace(table=table, cfg=goal_cfg)
    _print_target_diagnostics(table, terrain_gen_cfg, goal_cfg)

    sampling_cfg = env_cfg.curriculum.terrain_levels.params["sampling"]
    if sampling_cfg.max_samples is None:
        sampling_cfg.max_samples = args_cli.num_envs
    layout = StateLayout(
        coords=table.spawn_states[:, :2],
        spawn_index=table.spawn_index,
        target_index=table.target_index,
        task_partition=table.task_partition,
    )
    success_rates = torch.zeros(int(table.spawn_index.shape[0]), device=device)
    sampler = sampling_cfg.class_type(sampling_cfg, layout, success_rates=success_rates)

    class _CommandManager:
        def get_term(self, name: str):
            if name != "goal_point":
                raise KeyError(name)
            return goal_term

    env = SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                cfg=env_cfg.scene.terrain,
                terrain_mesh=terrain.terrain_mesh,
            )
        ),
        command_manager=_CommandManager(),
        device=device,
        extras={},
        common_step_counter=0,
    )

    spawn_xy = table.spawn_states[table.spawn_index, :2]
    target_xy = table.spawn_states[table.target_index, :2]
    distance = (spawn_xy - target_xy).norm(dim=-1)
    # 0 m -> ~1.0 success, 5 m -> ~0.1 success, plus gaussian jitter.
    synth = (1.0 - distance.clamp_max(5.0) / 5.0) * 0.9 + 0.1
    synth = (synth + 0.15 * torch.randn_like(synth)).clamp(0.0, 1.0)
    success_rates.copy_(synth)
    synth_probs = sampler.probabilities().clone()

    image_logger = SpawnGoalSamplerImageLogger(log_to_wandb=False)
    # First call seeds the cached "previous rates" — diff panel shows zeros.
    image_logger(env, sampler, synth, synth_probs)
    # Second call so the Δ panel renders a meaningful diff vs. the first.
    synth2 = (synth + 0.05 * torch.randn_like(synth)).clamp(0.0, 1.0)
    success_rates.copy_(synth2)
    synth_probs2 = sampler.probabilities().clone()
    image_logger(env, sampler, synth2, synth_probs2)

    img = env.extras.get("log_images", {}).get("Sampler/spawn_scatter")
    if img is None:
        print("[preview] No image was written to extras['log_images'].", file=sys.stderr)
        sys.exit(1)
    Image.fromarray(np.asarray(img)).save(args_cli.output)
    num_states = table.spawn_states.shape[0]
    print(
        f"[preview] wrote {args_cli.output} (shape={img.shape}, tasks={table.num_tasks}, states={num_states})",
        flush=True,
    )


def _print_target_diagnostics(table, terrain_gen_cfg, goal_cfg) -> None:
    import torch

    requested = int(goal_cfg.task_table.num_targets_per_cell)
    if requested <= 0:
        return

    num_rows = int(terrain_gen_cfg.num_rows)
    num_cols = int(terrain_gen_cfg.num_cols)
    num_tiles = num_rows * num_cols
    sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
    offsets = table.offsets
    first_cmd_start = int(offsets[0])
    first_cmd_end = int(offsets[1])
    spawn_index = table.spawn_index[first_cmd_start:first_cmd_end]
    target_index = table.target_index[first_cmd_start:first_cmd_end]
    tile_index = table.tile_index[first_cmd_start:first_cmd_end]

    print(f"[preview] target diagnostics: requested={requested} per terrain cell", flush=True)
    for tile in range(num_tiles):
        mask = tile_index == tile
        if not bool(mask.any()):
            continue
        row, col = divmod(tile, num_cols)
        spawn_count = int(torch.unique(spawn_index[mask]).numel())
        target_ids = torch.unique(target_index[mask])
        target_count = int(target_ids.numel())
        # The plot is XY-only; distinct target states can overlap visually when
        # final FPS kept different yaw/joint states at nearly identical XY.
        target_xy = table.spawn_states[target_ids, :2]
        target_xy_count = int(torch.unique(torch.round(target_xy * 1000.0).to(torch.int64), dim=0).shape[0])
        if target_count < min(requested, spawn_count) or target_xy_count < target_count:
            terrain_name = sub_terrain_names[min(col, len(sub_terrain_names) - 1)]
            print(
                f"  tile=({row},{col}) {terrain_name}: spawns={spawn_count} "
                f"targets={target_count} visible_xy~={target_xy_count}",
                flush=True,
            )


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
