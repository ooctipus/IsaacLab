# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script for IsaacLab environments with NVTX instrumentation.

Run directly for a quick sanity check (no nsys needed)::

    ./isaaclab.sh -p scripts/octibenchmark/benchmark.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 --num_frames 50

Run under nsys for full profiling (step phase)::

    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o /tmp/bench_shadow \\
        ./isaaclab.sh -p scripts/octibenchmark/benchmark.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 --num_frames 50

Profile startup time only::

    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o /tmp/bench_startup \\
        ./isaaclab.sh -p scripts/octibenchmark/benchmark.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 --phase startup

Then analyze::

    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add scripts/ to path so octibenchmark is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

from octibenchmark.nvtx_hooks import install_extra_nvtx_hooks, install_nvtx_hooks


def _query_gpu_memory() -> tuple[float, float]:
    """Query GPU memory usage in MB for the current CUDA device.

    Uses :func:`torch.cuda.mem_get_info` which reports memory across all
    CUDA contexts within this process (PyTorch, Warp, PhysX, Isaac Sim).

    Returns:
        ``(used_mb, total_mb)`` tuple. Returns ``(0.0, 0.0)`` on failure.
    """
    try:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return used / (1024 * 1024), total / (1024 * 1024)
    except Exception:
        return 0.0, 0.0


parser = argparse.ArgumentParser(description="Benchmark an IsaacLab environment with NVTX instrumentation.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--num_frames", type=int, default=100, help="Number of env steps to benchmark.")
parser.add_argument("--warmup_frames", type=int, default=10, help="Warmup steps (not profiled by nsys).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--phase",
    type=str,
    default="step",
    choices=["step", "startup"],
    help="What to profile: 'step' = stepping loop only, 'startup' = env creation + first reset only.",
)
parser.add_argument(
    "--extra_nvtx_hooks",
    type=str,
    default=None,
    help="JSON list of [attr_path, label] pairs for extra NVTX hooks.",
)
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

        if args_cli.phase == "startup":
            _run_startup(env_cfg)
        else:
            _run_step(env_cfg)


def _run_startup(env_cfg):
    """Profile env creation + first reset only."""
    # Start capture BEFORE gym.make — this is the expensive part
    torch.cuda.cudart().cudaProfilerStart()

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    torch.cuda.cudart().cudaProfilerStop()

    num_envs = env.unwrapped.num_envs
    env.close()

    print(f"[octibenchmark] Startup done: {num_envs} envs, task={args_cli.task}")


def _run_step(env_cfg):
    """Profile the stepping loop only."""
    # Create env and warmup WITHOUT nvtx hooks — no overhead during startup
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()

    action_dim = unwrapped.single_action_space.shape[0]
    device = unwrapped.device
    num_envs = unwrapped.num_envs

    # Warmup (no hooks, outside nsys capture)
    for _ in range(args_cli.warmup_frames):
        actions = 2.0 * torch.rand(num_envs, action_dim, device=device) - 1.0
        env.step(actions)

    # Install NVTX hooks AFTER warmup, right before capture
    install_nvtx_hooks(unwrapped)
    if args_cli.extra_nvtx_hooks:
        import json

        hooks = json.loads(args_cli.extra_nvtx_hooks)
        install_extra_nvtx_hooks(unwrapped, hooks)

    # Signal nsys to start capture
    torch.cuda.cudart().cudaProfilerStart()

    # Benchmark loop — track peak GPU memory and wall-clock time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    peak_used_mb, gpu_total_mb = 0.0, 0.0
    for _ in range(args_cli.num_frames):
        actions = 2.0 * torch.rand(num_envs, action_dim, device=device) - 1.0
        env.step(actions)
        used, total = _query_gpu_memory()
        if used > peak_used_mb:
            peak_used_mb, gpu_total_mb = used, total
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    # Signal nsys to stop capture
    torch.cuda.cudart().cudaProfilerStop()

    effective_fps = (args_cli.num_frames * num_envs) / elapsed_s if elapsed_s > 0 else 0.0
    step_ms = (elapsed_s / args_cli.num_frames) * 1000.0 if args_cli.num_frames > 0 else 0.0
    print(f"[octibenchmark] Step done: {args_cli.num_frames} frames, {num_envs} envs, task={args_cli.task}", flush=True)
    print(
        f'[octibenchmark:memory] {{"gpu_used_mb": {peak_used_mb:.1f}, "gpu_total_mb": {gpu_total_mb:.1f}}}',
        flush=True,
    )
    print(
        f'[octibenchmark:timing] {{"effective_fps": {effective_fps:.1f}, "step_ms": {step_ms:.3f}}}',
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
