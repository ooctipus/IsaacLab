# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the native MetaMotivo HumEnv vector environment without a learner."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers import TimeAwareObservation
from humenv import make_humenv


def _shapes(value) -> object:
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, dict):
        return {name: _shapes(array) for name, array in value.items()}
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--motions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=350)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe output: {args.output}")

    build_started = time.perf_counter()
    env, _motion_info = make_humenv(
        num_envs=args.num_envs,
        wrappers=[
            gymnasium.wrappers.FlattenObservation,
            lambda item: TimeAwareObservation(item, flatten=False),
        ],
        render_width=320,
        render_height=320,
        motions=str(args.motions),
        motion_base_path=str(args.motions_root),
        fall_prob=0.2,
        state_init="MoCapAndFall",
    )
    observation, info = env.reset()
    build_seconds = time.perf_counter() - build_started
    actions = np.zeros((args.num_envs, env.action_space.shape[-1]), dtype=np.float32)
    for _step in range(args.warmup_steps):
        observation, _reward, _terminated, _truncated, info = env.step(actions)

    terminated_count = 0
    truncated_count = 0
    step_started = time.perf_counter()
    for _step in range(args.measure_steps):
        observation, _reward, terminated, truncated, info = env.step(actions)
        terminated_count += int(np.asarray(terminated).sum())
        truncated_count += int(np.asarray(truncated).sum())
    step_seconds = time.perf_counter() - step_started
    result = {
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "build_seconds": build_seconds,
        "step_seconds": step_seconds,
        "vector_steps_per_second": args.measure_steps / step_seconds,
        "edge_positions_per_second": args.num_envs * args.measure_steps / step_seconds,
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "observation_shapes": _shapes(observation),
        "info_shapes": _shapes(info),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
    }
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
