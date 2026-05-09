# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the spawn-scatter visualization once and save the image to a PNG.

Spins up ``Isaac-Position-v0`` headless with a small ``num_envs``, steps once
so the curriculum manager initializes its monitors, injects a synthetic
distance-to-goal -> success-rate distribution so the colormap shows a
non-trivial pattern, calls ``_log_spawn_scatter`` directly, then writes the
resulting numpy image (RGB, HxWx3) to disk via PIL.

The script bypasses wandb entirely — its output is a standalone PNG you can
inspect to see exactly what the wandb upload will look like at log time.

Run::

    ./isaaclab.sh -p \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/terrain/utils/tools/preview_spawn_scatter.py \
        --headless --num_envs 16 --output /tmp/spawn_scatter_preview.png
"""

from __future__ import annotations

import argparse
import sys


def _parse_args():
    parser = argparse.ArgumentParser(description="Preview the spawn-scatter visualization to a PNG.")
    parser.add_argument("--task", type=str, default="Isaac-Position-v0", help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs.")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/spawn_scatter_preview.png",
        help="Where to write the rendered PNG.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch seed for the synthetic noise.")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    return args_cli


def main():
    args_cli = _parse_args()
    args_cli.headless = True

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Deferred imports (post-Sim init).
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab_tasks  # noqa: F401  — registers ``Isaac-Position-v0``
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    torch.manual_seed(args_cli.seed)

    # ``parse_env_cfg`` already calls ``resolve_presets(cfg)`` with an empty
    # selected set, which collapses ``RobotArticulationCfg`` to its
    # ``MISSING`` default before we get a chance to pick a robot. Bypass it
    # and resolve manually with ``anymal_c`` selected so ``scene.robot`` lands
    # as the actual articulation cfg.
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    resolve_presets(env_cfg, selected={"anymal_c"})
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        print("[preview] env.reset done", flush=True)

        # One step so termination_manager has a "success" entry for the
        # curriculum's success_update on the next call.
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
        print("[preview] env.step done", flush=True)

        env_unwrapped = env.unwrapped
        # ``CurriculumManager`` doesn't expose a public getter; class-based
        # terms get instantiated and stashed back into ``term_cfg.func`` by
        # ``ManagerBase._resolve_common_term_cfg``.
        manager = env_unwrapped.curriculum_manager
        term_idx = manager._term_names.index("terrain_levels")
        term = manager._term_cfgs[term_idx].func
        print(f"[preview] curriculum term: {type(term).__name__}", flush=True)

        # Inject a synthetic distance -> success distribution so the dot
        # colormap exercises the full RdYlGn range. Real training fills this
        # tensor via the success monitor; we overwrite for preview only.
        table = term.goal_term.table
        spawn_xy = table.spawn_states[table.spawn_index, :2]
        target_xy = table.spawn_states[table.target_index, :2]
        distance = (spawn_xy - target_xy).norm(dim=-1)
        # 0 m -> ~1.0 success, 5 m -> ~0.1 success, plus gaussian jitter.
        synth = (1.0 - distance.clamp_max(5.0) / 5.0) * 0.9 + 0.1
        synth = (synth + 0.15 * torch.randn_like(synth)).clamp(0.0, 1.0)
        term.success_monitor.success_rate.copy_(synth)

        # Synthetic sampling probabilities (mirrors the Beta-target sampler:
        # mass concentrates on commands with success near the target). Just
        # something visually sensible for the preview panel.
        target = 0.66
        weights = torch.exp(-((synth - target) ** 2) / 0.05)
        synth_probs = weights / weights.sum()

        # First call seeds the cached "previous rates" — Δ panel shows zeros.
        term._log_spawn_scatter(synth, synth_probs)
        # Second call so the Δ panel renders a meaningful diff vs. the first.
        synth2 = (synth + 0.05 * torch.randn_like(synth)).clamp(0.0, 1.0)
        term.success_monitor.success_rate.copy_(synth2)
        term._log_spawn_scatter(synth2, synth_probs)

        img = env_unwrapped.extras.get("log_images", {}).get("Sampler/spawn_scatter")
        if img is None:
            print("[preview] No image was written to extras['log_images'].", file=sys.stderr)
            sys.exit(1)
        Image.fromarray(np.asarray(img)).save(args_cli.output)
        print(f"[preview] wrote {args_cli.output} (shape={img.shape})", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
