# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the native BFM Isaac Sim environment without constructing a learner."""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from pathlib import Path

import torch
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig


def _tensor_shapes(value) -> object:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {name: _tensor_shapes(tensor) for name, tensor in value.items()}
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe output: {args.output}")

    with args.config.open() as stream:
        resolved = json.load(stream)
    env_data = copy.deepcopy(resolved["env"])
    env_data["lafan_tail_path"] = str(args.data_path.resolve())
    env_data["device"] = "cuda:0"
    overrides = [
        value
        for value in env_data["hydra_overrides"]
        if not value.startswith("env.config.headless=") and not value.startswith("simulator=")
    ]
    env_data["hydra_overrides"] = overrides + ["env.config.headless=True", "simulator=isaacsim"]

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    build_started = time.perf_counter()
    env, _info = HumanoidVerseIsaacConfig(**env_data).build(num_envs=args.num_envs)
    observation, _info = env.reset(to_numpy=False)
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - build_started
    free_after_build, _total = torch.cuda.mem_get_info()

    actions = torch.zeros(args.num_envs, env.action_space.shape[-1], device="cuda:0")
    for _step in range(args.warmup_steps):
        observation, _reward, _terminated, _truncated, _info = env.step(actions, to_numpy=False)
    torch.cuda.synchronize()
    free_after_warmup, _total = torch.cuda.mem_get_info()
    step_started = time.perf_counter()
    for _step in range(args.measure_steps):
        observation, _reward, _terminated, _truncated, _info = env.step(actions, to_numpy=False)
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_started
    free_after_steps, _total = torch.cuda.mem_get_info()

    result = {
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "build_seconds": build_seconds,
        "step_seconds": step_seconds,
        "vector_steps_per_second": args.measure_steps / step_seconds,
        "edge_positions_per_second": args.num_envs * args.measure_steps / step_seconds,
        "device_total_bytes": total,
        "device_used_before_bytes": total - free_before,
        "device_used_after_build_bytes": total - free_after_build,
        "device_used_after_warmup_bytes": total - free_after_warmup,
        "device_used_after_steps_bytes": total - free_after_steps,
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "observation_shapes": _tensor_shapes(observation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
