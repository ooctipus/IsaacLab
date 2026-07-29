# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace multi-task environment startup and first environment steps."""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import sys
import time
from collections.abc import Callable

from isaaclab.utils.string import list_intersection, string_to_callable

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Trace environment startup and first steps.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--trace_steps", type=int, default=10, help="Number of environment steps to trace.")
parser.add_argument(
    "--action_mode",
    choices=("random", "zero"),
    default="random",
    help="Action source for traced environment steps.",
)
parser.add_argument("--output", type=str, default=None, help="Optional structured JSON output path.")
parser.add_argument("--chrome_trace", type=str, default=None, help="Optional Chrome trace JSON output path.")
parser.add_argument("--top_events", type=int, default=25, help="Number of aggregate events to print.")
parser.add_argument(
    "--no_synchronize",
    action="store_true",
    default=False,
    help="Do not synchronize CUDA/Warp around trace timestamps.",
)
parser.add_argument(
    "--no_manager_trace",
    action="store_true",
    default=False,
    help="Do not monkey-patch manager/sim methods for per-step sub-spans.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.task is None:
    parser.error("the following arguments are required: --task")


remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args


imports_time_begin = time.perf_counter_ns()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.timer import Timer, TimerError  # noqa: E402

from isaaclab_tasks.core.multi_task.utils.trace import (  # noqa: E402
    TraceRecorder,
    reset_trace_recorder,
    set_trace_recorder,
    trace_span,
)
from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402

imports_time_end = time.perf_counter_ns()


def _apply_cli_overrides(env_cfg, agent_cfg) -> None:
    """Apply trace-script CLI overrides to an environment config."""
    if args_cli.num_envs is not None and hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None and hasattr(env_cfg, "sim"):
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric and hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False

    agent_seed = getattr(agent_cfg, "seed", None)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_seed


def _wrap_method(obj, method_name: str, span_name: str) -> None:
    """Wrap an instance method with a trace span."""
    method = getattr(obj, method_name, None)
    if method is None or getattr(method, "_trace_env_wrapped", False):
        return

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        with trace_span(span_name):
            return method(*args, **kwargs)

    wrapped._trace_env_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(obj, method_name, wrapped)
    except (AttributeError, TypeError):
        return


def _wrap_manager(env, manager_name: str, methods: tuple[str, ...]) -> None:
    manager = getattr(env, manager_name, None)
    if manager is None:
        return
    label = manager_name.removesuffix("_manager")
    for method_name in methods:
        _wrap_method(manager, method_name, f"manager.{label}.{method_name}")


def _wrap_module_function(module_name: str, function_name: str, span_name: str) -> None:
    """Wrap a module-level function with a trace span."""
    with contextlib.suppress(ImportError):
        module = importlib.import_module(module_name)
        _wrap_method(module, function_name, span_name)


def _install_startup_tracing() -> None:
    """Install trace wrappers for scene-startup clone utilities."""
    clone_spans = (
        ("isaaclab.cloner", "usd_replicate", "cloner.usd_replicate"),
        ("isaaclab.cloner.cloner_utils", "usd_replicate", "cloner.usd_replicate"),
        ("isaaclab.cloner", "filter_collisions", "cloner.filter_collisions"),
        ("isaaclab.cloner.cloner_utils", "filter_collisions", "cloner.filter_collisions"),
        ("isaaclab.cloner", "clone_from_template", "cloner.clone_from_template"),
        ("isaaclab.cloner.cloner_utils", "clone_from_template", "cloner.clone_from_template"),
        ("isaaclab_physx.cloner", "physx_replicate", "cloner.physx_replicate"),
        ("isaaclab_physx.cloner.physx_replicate", "physx_replicate", "cloner.physx_replicate"),
        ("isaaclab_ovphysx.cloner", "ovphysx_replicate", "cloner.ovphysx_replicate"),
        ("isaaclab_ovphysx.cloner.ovphysx_replicate", "ovphysx_replicate", "cloner.ovphysx_replicate"),
        ("isaaclab_newton.cloner", "newton_physics_replicate", "cloner.newton_physics_replicate"),
        ("isaaclab_newton.cloner.newton_replicate", "newton_physics_replicate", "cloner.newton_physics_replicate"),
    )
    for module_name, function_name, span_name in clone_spans:
        _wrap_module_function(module_name, function_name, span_name)


def _install_runtime_tracing(env: gym.Env) -> None:
    """Install per-step runtime trace wrappers on the live environment."""
    unwrapped = env.unwrapped
    _wrap_method(unwrapped, "_reset_idx", "env.reset_idx")

    _wrap_manager(unwrapped, "action_manager", ("process_action", "apply_action", "reset"))
    _wrap_manager(unwrapped, "command_manager", ("compute", "reset"))
    _wrap_manager(unwrapped, "observation_manager", ("compute", "reset"))
    _wrap_manager(unwrapped, "reward_manager", ("compute", "reset"))
    _wrap_manager(unwrapped, "termination_manager", ("compute", "reset"))
    _wrap_manager(unwrapped, "curriculum_manager", ("compute", "reset"))
    _wrap_manager(unwrapped, "event_manager", ("apply", "reset"))
    _wrap_manager(unwrapped, "recorder_manager", ("reset",))

    _wrap_method(unwrapped.scene, "write_data_to_sim", "scene.write_data_to_sim")
    _wrap_method(unwrapped.scene, "update", "scene.update")
    _wrap_method(unwrapped.scene, "reset", "scene.reset")
    _wrap_method(unwrapped.sim, "step", "sim.step")
    _wrap_method(unwrapped.sim, "render", "sim.render")


def _make_actions(env: gym.Env):
    """Create actions for a traced step."""
    unwrapped = env.unwrapped
    if args_cli.action_mode == "zero":
        return torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
    return 2.0 * torch.rand(unwrapped.action_space.shape, device=unwrapped.device) - 1.0


def _record_timer_span(recorder: TraceRecorder, name: str, anchor_ns: int) -> None:
    """Record an IsaacLab global timer as an approximate span."""
    try:
        duration_s = Timer.get_timer_info(name)
    except TimerError:
        return
    duration_ns = int(duration_s * 1_000_000_000)
    recorder.record_duration_ns(
        f"timer.{name}",
        max(anchor_ns - duration_ns, 0),
        anchor_ns,
        source="isaaclab.utils.Timer",
    )


def _run_traced_env(env_cfg, recorder: TraceRecorder) -> None:
    """Create, reset, and step the environment while recording trace spans."""
    env = None
    with recorder.span("env_creation", task=args_cli.task):
        env = gym.make(args_cli.task, cfg=env_cfg)
    env_creation_end = recorder.timestamp_ns()
    _record_timer_span(recorder, "scene_creation", env_creation_end)
    _record_timer_span(recorder, "simulation_start", env_creation_end)

    try:
        if not args_cli.no_manager_trace:
            _install_runtime_tracing(env)

        with recorder.span("env_reset"):
            env.reset()

        for step_id in range(max(args_cli.trace_steps, 0)):
            actions = _make_actions(env)
            with recorder.span("env_step", step=step_id, action_mode=args_cli.action_mode):
                env.step(actions)
    finally:
        if env is not None:
            with recorder.span("env_close"):
                env.close()


def _print_outputs(recorder: TraceRecorder) -> None:
    """Print and export trace outputs."""
    print()
    for line in recorder.summary_lines(top_n=args_cli.top_events):
        print(line)

    if args_cli.output:
        recorder.export_json(args_cli.output)
        print(f"\n[TRACE] Wrote JSON trace to: {args_cli.output}")
    if args_cli.chrome_trace:
        recorder.export_chrome_trace(args_cli.chrome_trace)
        print(f"[TRACE] Wrote Chrome trace to: {args_cli.chrome_trace}")


def _timed_phase(recorder: TraceRecorder, name: str, fn: Callable, *args, **kwargs):
    """Run ``fn`` and record a synchronized phase span."""
    start_ns = recorder.timestamp_ns()
    result = fn(*args, **kwargs)
    end_ns = recorder.timestamp_ns()
    recorder.record_duration_ns(name, start_ns, end_ns)
    return result


def main() -> None:
    """Trace the configured environment."""
    recorder = TraceRecorder(
        metadata={
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "trace_steps": args_cli.trace_steps,
            "action_mode": args_cli.action_mode,
            "hydra_args": remaining_args,
        },
        synchronize=not args_cli.no_synchronize,
    )
    recorder.record_duration_ns("python_imports", imports_time_begin, imports_time_end)

    env_cfg, agent_cfg = _timed_phase(recorder, "task_config", resolve_task_config, args_cli.task, args_cli.agent)
    _apply_cli_overrides(env_cfg, agent_cfg)

    token = set_trace_recorder(recorder)
    try:
        app_launch_start = recorder.timestamp_ns()
        with launch_simulation(env_cfg, args_cli):
            app_launch_end = recorder.timestamp_ns()
            recorder.record_duration_ns("app_launch", app_launch_start, app_launch_end)
            _install_startup_tracing()
            _run_traced_env(env_cfg, recorder)
            _print_outputs(recorder)
    finally:
        reset_trace_recorder(token)


if __name__ == "__main__":
    main()
