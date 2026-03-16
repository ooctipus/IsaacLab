# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script for IsaacLab RL training with NVTX instrumentation.

Runs RSL-RL on-policy training with NVTX hooks on both the environment
and the RL runner's algorithm methods (act, process_env_step, update,
compute_returns).

Usage (step phase — profile training loop)::

    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o /tmp/bench_train \\
        python scripts/octibenchmark/benchmark_train.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 \\
        --max_iterations 5 --headless presets=newton,newton_renderer,rgb

Usage (startup phase — profile env + runner creation)::

    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o /tmp/bench_train_startup \\
        python scripts/octibenchmark/benchmark_train.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 --num_envs 64 \\
        --phase startup --headless presets=newton,newton_renderer,rgb
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

from octibenchmark.nvtx_hooks import _wrap_nvtx, install_extra_nvtx_hooks, install_nvtx_hooks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Benchmark RL training with NVTX instrumentation.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=10, help="RL training iterations.")
parser.add_argument("--warmup_frames", type=int, default=0, help="Warmup iterations before nsys capture.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument(
    "--phase",
    type=str,
    default="step",
    choices=["step", "startup"],
    help="What to profile: 'step' = training loop only, 'startup' = env + runner creation only.",
)
parser.add_argument(
    "--extra_nvtx_hooks",
    type=str,
    default=None,
    help="JSON list of [attr_path, label] pairs for extra NVTX hooks.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


class _LearningTimer:
    """Accumulates wall-clock time spent in the learning phase (compute_returns + update)."""

    def __init__(self):
        self.total_s: float = 0.0

    def wrap(self, obj, attr):
        """Replace *obj.attr* with a timed version that accumulates into *total_s*."""
        original = getattr(obj, attr)
        timer = self

        def _timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            timer.total_s += time.perf_counter() - t0
            return result

        setattr(obj, attr, _timed)


def _install_runner_nvtx(runner):
    """Install NVTX hooks on the RSL-RL runner's algorithm methods."""
    alg = getattr(runner, "alg", None)
    if alg is None:
        return
    if hasattr(alg, "act"):
        _wrap_nvtx(alg, "act", "runner.alg.act")
    if hasattr(alg, "process_env_step"):
        _wrap_nvtx(alg, "process_env_step", "runner.alg.process_env_step")
    if hasattr(alg, "update"):
        _wrap_nvtx(alg, "update", "runner.alg.update")
    if hasattr(alg, "compute_returns"):
        _wrap_nvtx(alg, "compute_returns", "runner.alg.compute_returns")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Benchmark RL training with NVTX instrumentation."""
    # Lazy import to avoid pulling rsl_rl at module level for non-training runs
    import importlib.metadata as metadata

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.seed = args_cli.seed

        if args_cli.phase == "startup":
            _run_startup(env_cfg, agent_cfg, OnPolicyRunner, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, metadata)
        else:
            _run_step(env_cfg, agent_cfg, OnPolicyRunner, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, metadata)


def _run_startup(env_cfg, agent_cfg, OnPolicyRunner, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, metadata):
    """Profile env creation + runner creation only."""
    torch.cuda.cudart().cudaProfilerStart()

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = os.path.join("/tmp", "octibench_train_logs")
    OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    torch.cuda.cudart().cudaProfilerStop()

    num_envs = env.unwrapped.num_envs
    env.close()

    print(f"[octibenchmark] Startup done: {num_envs} envs, task={args_cli.task}")


def _run_step(env_cfg, agent_cfg, OnPolicyRunner, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, metadata):
    """Profile the training loop only."""
    # Create env and runner WITHOUT hooks — no overhead during startup
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = os.path.join("/tmp", "octibench_train_logs")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    num_envs = unwrapped.num_envs

    # Warmup (no hooks, outside nsys capture)
    if args_cli.warmup_frames > 0:
        runner.learn(num_learning_iterations=args_cli.warmup_frames, init_at_random_ep_len=True)

    # Install NVTX hooks AFTER warmup, right before capture
    install_nvtx_hooks(unwrapped)
    if args_cli.extra_nvtx_hooks:
        import json

        hooks = json.loads(args_cli.extra_nvtx_hooks)
        install_extra_nvtx_hooks(unwrapped, hooks)
    _install_runner_nvtx(runner)

    # Wrap learning-phase methods to measure time spent outside collection
    learning_timer = _LearningTimer()
    alg = runner.alg
    if hasattr(alg, "compute_returns"):
        learning_timer.wrap(alg, "compute_returns")
    if hasattr(alg, "update"):
        learning_timer.wrap(alg, "update")

    # Signal nsys to start capture
    torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    # Signal nsys to stop capture
    torch.cuda.cudart().cudaProfilerStop()

    gpu_mem_used_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    gpu_mem_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    env.close()

    steps_per_iter = num_envs * agent_cfg.num_steps_per_env
    total_steps = args_cli.max_iterations * steps_per_iter
    collection_s = max(elapsed_s - learning_timer.total_s, 0.0)
    collection_fps = total_steps / collection_s if collection_s > 0 else 0.0
    iteration_fps = args_cli.max_iterations / elapsed_s if elapsed_s > 0 else 0.0
    step_ms = (elapsed_s / args_cli.max_iterations) * 1000.0 if args_cli.max_iterations > 0 else 0.0
    print(f"[octibenchmark] Training done: {args_cli.max_iterations} iterations, {num_envs} envs, task={args_cli.task}")
    print(
        f'[octibenchmark:memory] {{"gpu_used_mb": {gpu_mem_used_mb:.1f}, "gpu_total_mb": {gpu_mem_total_mb:.1f}}}',
        flush=True,
    )
    print(
        f'[octibenchmark:timing] {{"collection_fps": {collection_fps:.1f}, "iteration_fps": {iteration_fps:.1f}, "step_ms": {step_ms:.3f}}}',
        flush=True,
    )


if __name__ == "__main__":
    main()
