# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Profile :class:`MultiTaskCommand` to surface dispatch bottlenecks.

Default mode (``--mode mock``) bypasses Isaac Sim entirely: builds the production
``MultiTaskCfg`` (8 tasks, mixed instant / tracking, multi-entity) on top of a
synthetic ``ManagerBasedRLEnv`` stub and feeds the readers pre-allocated random
tensors. The dispatch path under trace is byte-identical to production — only
the warp→torch reader handoff and PhysX step are missing, which are not what
the trace targets.

``--mode sim`` falls back to a real Isaac Sim env (``MultiTaskEnvCfg``) for
end-to-end timing including the reader path. Requires Isaac Sim to be properly
installed.

Phases recorded:

- ``MultiTaskCommand.update_command`` — full per-step path.
- ``MultiTaskCommand.dispatch`` — subclass-provided dispatch
  (``MultiTaskCommandTorch._dispatch`` or
  ``MultiTaskCommandWarp._dispatch``, whichever the cfg selected).

Output:

- Console: top-N ops by self CPU + self CUDA, plus per-phase totals.
- ``--output`` directory: ``trace_<timestamp>.json`` for ``chrome://tracing``
  / `Perfetto <https://ui.perfetto.dev>`_.

Usage::

    # Mock mode (no Isaac Sim required):
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/core/multi_task/mdp/commands/\\
multi_task_command/benchmark/trace_multi_task_command.py \\
        --backend mega_kernel --num_envs 4096 --num_steps 200

    # Sim mode (real env):
    ./isaaclab.sh -p ... --mode sim --backend mega_kernel --num_envs 1024 --num_steps 100
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import torch

if __package__:
    from .mock_command import build_mock_command
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mock_command import build_mock_command


# ---------------------------------------------------------------------------
# Profiler wiring.
# ---------------------------------------------------------------------------


def _wrap_phases(command_term):
    """Add ``record_function`` markers to the command term's hot path.

    Wraps :meth:`_update_command` and :meth:`_dispatch` on the instance
    (the subclass-specific dispatch). The trace labels stay the same
    whether the term is a Torch or Warp instance — the label reflects
    the phase, not the implementation.
    """
    from torch.profiler import record_function

    orig_update = command_term._update_command
    orig_dispatch = command_term._dispatch

    def wrapped_update():
        with record_function("MultiTaskCommand.update_command"):
            return orig_update()

    def wrapped_dispatch(valid_slots):
        with record_function("MultiTaskCommand.dispatch"):
            return orig_dispatch(valid_slots)

    command_term._update_command = wrapped_update
    command_term._dispatch = wrapped_dispatch


def _print_phase_summary(prof, phase_names: list[str]) -> None:
    events = prof.key_averages()
    by_key = {ev.key: ev for ev in events}
    print("\n[PHASES] (totals over recorded steps)")
    print(f"  {'phase':<42s}  {'count':>6s}  {'cpu(ms)':>10s}  {'cuda(ms)':>10s}")
    for name in phase_names:
        ev = by_key.get(name)
        if ev is None:
            print(f"  {name:<42s}  {'—':>6s}  {'—':>10s}  {'—':>10s}")
            continue
        cpu_ms = ev.self_cpu_time_total / 1e3
        cuda_attr = "self_device_time_total" if hasattr(ev, "self_device_time_total") else "self_cuda_time_total"
        cuda_ms = getattr(ev, cuda_attr) / 1e3
        print(f"  {name:<42s}  {ev.count:>6d}  {cpu_ms:>10.2f}  {cuda_ms:>10.2f}")


def _profile_loop(cmd, env, readers, mtc_mod, args_cli) -> None:
    """Run the trace loop with the profiler attached."""
    from unittest.mock import patch

    from torch.profiler import ProfilerActivity, profile, schedule

    _wrap_phases(cmd)

    out_dir = pathlib.Path(args_cli.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = out_dir / f"trace_{timestamp}.json"

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and "cuda" in str(env.device):
        activities.append(ProfilerActivity.CUDA)

    active_steps = max(1, args_cli.num_steps - args_cli.warmup_steps)
    prof_schedule = schedule(wait=0, warmup=args_cli.warmup_steps, active=active_steps, repeat=1)

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        with profile(
            activities=activities,
            schedule=prof_schedule,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _step in range(args_cli.num_steps):
                cmd._update_command()
                env.episode_length_buf += 1
                prof.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))

    sort_cpu = "self_cpu_time_total"
    sort_cuda = "self_cuda_time_total" if torch.cuda.is_available() else sort_cpu

    print("\n" + "=" * 88)
    print(f"[TOP {args_cli.top_n} BY SELF CPU TIME]")
    print("=" * 88)
    print(prof.key_averages().table(sort_by=sort_cpu, row_limit=args_cli.top_n))

    if torch.cuda.is_available() and "cuda" in str(env.device):
        print("=" * 88)
        print(f"[TOP {args_cli.top_n} BY SELF CUDA TIME]")
        print("=" * 88)
        print(prof.key_averages().table(sort_by=sort_cuda, row_limit=args_cli.top_n))

    _print_phase_summary(
        prof,
        [
            "MultiTaskCommand.update_command",
            "MultiTaskCommand.dispatch",
        ],
    )

    print(f"\n[TRACE] Wrote Chrome trace to: {trace_path}")
    print("       Open chrome://tracing or https://ui.perfetto.dev and load this file.")


# ---------------------------------------------------------------------------
# Sim-mode glue (production env path; only built when --mode sim).
# ---------------------------------------------------------------------------


def _run_sim_mode(args_cli) -> None:
    """End-to-end profile against a real ``ManagerBasedRLEnv``. Requires Isaac Sim."""
    sys.argv = [sys.argv[0]]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
        from isaaclab_tasks.utils import resolve_presets

        env_cfg = MultiTaskEnvCfg()
        resolve_presets(env_cfg)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.commands.goal_point.dispatch_backend = args_cli.backend
        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            env.reset()
            cmd = env.command_manager.get_term("goal_point")

            class _NullReaderTuple:
                pass

            _profile_loop(cmd, env, _NullReaderTuple(), None, args_cli)
        finally:
            env.close()
    finally:
        simulation_app.close()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MultiTaskCommand dispatch.")
    parser.add_argument(
        "--mode",
        choices=("mock", "sim"),
        default="mock",
        help="mock = synthetic env (no Isaac Sim); sim = real ManagerBasedRLEnv.",
    )
    parser.add_argument("--num_envs", type=int, default=4096, help="Parallel envs.")
    parser.add_argument("--num_steps", type=int, default=200, help="Total iterations (warmup + active).")
    parser.add_argument("--warmup_steps", type=int, default=40, help="Steps to skip before recording.")
    parser.add_argument("--top_n", type=int, default=25, help="Rows in the CPU / CUDA self-time tables.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(pathlib.Path.cwd() / "multitask_trace"),
        help="Directory for the Chrome trace JSON.",
    )
    parser.add_argument(
        "--mock_device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for mock-mode tensors.",
    )
    parser.add_argument(
        "--backend",
        default="torch",
        help="Command backend selected through public MultiTaskCfg fields.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Optional mock-mode preset passed to the public command benchmark builder.",
    )
    # Only the sim path needs AppLauncher's CLI surface.
    if "--mode" in sys.argv and "sim" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)

    args_cli, _ = parser.parse_known_args()
    return args_cli


def main() -> None:
    args_cli = _parse_args()

    if args_cli.mode == "sim":
        _run_sim_mode(args_cli)
        return

    torch.manual_seed(0)
    cmd, env, readers, mtc_mod = build_mock_command(
        args_cli.num_envs,
        args_cli.mock_device,
        dispatch_backend=args_cli.backend,
        preset=args_cli.preset,
    )
    print(f"[TRACE] Backend: {args_cli.backend}, preset={args_cli.preset or 'default'}")
    _profile_loop(cmd, env, readers, mtc_mod, args_cli)


if __name__ == "__main__":
    main()
