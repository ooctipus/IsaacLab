# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Profile a single PPO ``update()`` call to find what dominates Learning time.

Runs a short rollout-collection + compute_returns at the configured ``num_envs``,
warms cuDNN autotune by doing a couple of throwaway updates, then wraps a single
``self.alg.update()`` call in :class:`torch.profiler.profile`. Prints two summary
tables — one grouped by operator name, one grouped by the high-level PPO phases
(actor forward, critic forward, loss, backward, optimizer step) — and writes a
Chrome trace for the chrome://tracing or perfetto UI.

Usage::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/core/multi_task/scripts/profile_ppo_update.py \\
        --task=Isaac-Position-v0 --num_envs=4096 --device=cuda:0 --warmup_iters=2 \\
        --output=/tmp/ppo_update.json presets=anymal_c,res02,cnn

The Chrome trace can be opened at https://ui.perfetto.dev/ or chrome://tracing.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from isaaclab.utils.string import list_intersection, string_to_callable

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Profile a single PPO update() to identify the time-dominant op.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--warmup_iters", type=int, default=2, help="Untimed full-iteration warmups before profiling.")
parser.add_argument("--top_ops", type=int, default=25, help="Number of operators to show in the summary table.")
parser.add_argument("--output", type=str, default=None, help="Optional Chrome trace output path.")
parser.add_argument(
    "--with_stack",
    action="store_true",
    default=False,
    help="Include Python call stacks in the profile (slower but pinpoints call sites).",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.task is None:
    parser.error("the following arguments are required: --task")


remaining_args_env_registration = None
if args_cli.external_callback:
    fn = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = fn()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402


def _instrument_alg(alg) -> None:
    """Wrap PPO's hot inner pieces with ``record_function`` so the profiler reports
    each high-level phase as its own row, alongside the low-level kernel ops.

    Targets the things we genuinely want to disambiguate: actor / critic forward
    (dominated by Linear ops), backward (autograd traversal), optimizer step (Adam
    state updates), and loss math (small but cumulative).
    """
    actor = alg.actor
    critic = alg.critic
    optimizer = alg.optimizer

    actor_forward = actor.forward

    def actor_forward_traced(*args, **kwargs):
        with record_function("ppo.actor_forward"):
            return actor_forward(*args, **kwargs)

    actor.forward = actor_forward_traced  # type: ignore[method-assign]

    critic_forward = critic.forward

    def critic_forward_traced(*args, **kwargs):
        with record_function("ppo.critic_forward"):
            return critic_forward(*args, **kwargs)

    critic.forward = critic_forward_traced  # type: ignore[method-assign]

    optimizer_step = optimizer.step

    def optimizer_step_traced(*args, **kwargs):
        with record_function("ppo.optimizer_step"):
            return optimizer_step(*args, **kwargs)

    optimizer.step = optimizer_step_traced  # type: ignore[method-assign]


def _profile_one_update(runner, output_path: Path | None, top_ops: int, with_stack: bool) -> None:
    """Profile exactly one ``alg.update()`` call inside a normal ``runner.learn(1)`` cycle.

    Monkey-patches ``alg.update`` to wrap the next call in :class:`torch.profiler.profile`,
    then triggers a single full iteration through ``runner.learn(1)`` so the rollout-collection
    path runs through the runner's own (unprofiled) plumbing — no inference-mode tensor
    issues from re-implementing the loop ourselves.
    """
    _instrument_alg(runner.alg)

    profile_results: dict = {}
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    original_update = runner.alg.update

    def profiled_update(*args, **kwargs):
        torch.cuda.synchronize()
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=with_stack,
            profile_memory=False,
        ) as prof:
            with record_function("ppo.update_total"):
                result = original_update(*args, **kwargs)
            torch.cuda.synchronize()
        profile_results["prof"] = prof
        # Restore so we don't profile any future calls (we only want one).
        runner.alg.update = original_update
        return result

    runner.alg.update = profiled_update  # type: ignore[method-assign]

    # Run a single full iteration: this drives rollout-collection through the runner's
    # normal (already-validated) path and triggers ``alg.update`` exactly once.
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    prof = profile_results.get("prof")
    if prof is None:
        print("[profile] WARNING: profiler context did not produce a result; was alg.update() called?")
        return

    print()
    print("=== Top operators by self CUDA time (whole update) ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=top_ops))

    print()
    print("=== Phase-level summary (look for ppo.* rows) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(output_path))
        print(f"\nWrote Chrome trace → {output_path}\n  open at https://ui.perfetto.dev/ or chrome://tracing")


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(agent_cfg, "seed", 42)

    with launch_simulation(env_cfg, args_cli):
        # Late imports to keep the orchestrator side light.
        import importlib.metadata as _metadata

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = agent_cfg.class_type(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

        # Warmup: cuDNN autotune + allocator warmup + any first-call JIT.
        if args_cli.warmup_iters > 0:
            print(f"\n[profile] running {args_cli.warmup_iters} warmup iter(s) (untimed)…")
            runner.learn(num_learning_iterations=args_cli.warmup_iters, init_at_random_ep_len=True)
            torch.cuda.synchronize()

        try:
            output_path = Path(args_cli.output) if args_cli.output else None
            _profile_one_update(runner, output_path, args_cli.top_ops, args_cli.with_stack)
        finally:
            env.close()


if __name__ == "__main__":
    main()
