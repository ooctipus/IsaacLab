# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script for IsaacLab environments with NVTX instrumentation.

Run directly for a quick sanity check (no nsys needed)::

    ./isaaclab.sh -p scripts/octibenchmark/benchmark.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 --num_frames 50

Run under nsys for full profiling::

    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o /tmp/bench_shadow \\
        ./isaaclab.sh -p scripts/octibenchmark/benchmark.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 --num_frames 50

Then analyze::

    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep
"""

from __future__ import annotations

import argparse
import os
import sys

# Add scripts/ to path so octibenchmark is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

from octibenchmark.nvtx_hooks import install_nvtx_hooks

parser = argparse.ArgumentParser(description="Benchmark an IsaacLab environment with NVTX instrumentation.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--num_frames", type=int, default=100, help="Number of env steps to benchmark.")
parser.add_argument("--warmup_frames", type=int, default=10, help="Warmup steps (not profiled by nsys).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Benchmark without RL in the loop."""
    with launch_simulation(env_cfg, args_cli):
        # Override config
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.seed = args_cli.seed

        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped

        # Install NVTX hooks
        install_nvtx_hooks(unwrapped)

        env.reset()

        action_dim = unwrapped.single_action_space.shape[0]
        device = unwrapped.device

        # Warmup (outside nsys capture range)
        for _ in range(args_cli.warmup_frames):
            actions = 2.0 * torch.rand(unwrapped.num_envs, action_dim, device=device) - 1.0
            env.step(actions)

        # Signal nsys to start capture (only matters with --capture-range=cudaProfilerApi)
        torch.cuda.cudart().cudaProfilerStart()

        # Benchmark loop
        for _ in range(args_cli.num_frames):
            actions = 2.0 * torch.rand(unwrapped.num_envs, action_dim, device=device) - 1.0
            env.step(actions)

        # Signal nsys to stop capture
        torch.cuda.cudart().cudaProfilerStop()

        num_envs = unwrapped.num_envs
        env.close()

    print(
        f"[octibenchmark] Done: {args_cli.num_frames} frames, "
        f"{num_envs} envs, task={args_cli.task}"
    )


if __name__ == "__main__":
    main()
