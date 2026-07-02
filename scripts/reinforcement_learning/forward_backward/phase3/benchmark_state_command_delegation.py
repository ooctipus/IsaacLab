# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the final StateCommand direct-payload delegation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import linecache
import math
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_STATE_COMMAND_PATH = Path(
    "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/mdp/commands/state_command/state_command.py"
)
_STATE_COMMAND_CFG_PATH = Path(
    "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/mdp/commands/state_command/state_command_cfg.py"
)
_HISTORICAL_REVISION = "fe51e5646a335e05844a10fa93d9479da2749847"


@dataclass(frozen=True, slots=True)
class Profile:
    """One representative StateCommand-owned tensor shape."""

    name: str
    command_dim: int
    error_dim: int
    row_width: int


_PROFILES = (
    Profile("position_like", command_dim=12, error_dim=4, row_width=48),
    Profile("factory_like", command_dim=7, error_dim=2, row_width=30),
)


class _Registry:
    def add_debug_vis_callback(self, _term: object) -> None:
        return None

    def clear_debug_vis_callback(self, _term: object) -> None:
        return None


class _Table:
    def __init__(self, num_tasks: int, row_width: int, device: torch.device) -> None:
        self.num_tasks = num_tasks
        row = torch.arange(num_tasks, dtype=torch.float32, device=device).unsqueeze(1)
        column = torch.arange(row_width, dtype=torch.float32, device=device).unsqueeze(0)
        self.rows = torch.remainder(row + column, 97.0).mul_(1.0 / 97.0)

    def sample_rows(self, count: int) -> torch.Tensor:
        return torch.randint(0, self.num_tasks, (count,), device=self.rows.device)

    def gather(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected = self.rows.index_select(0, rows)
        return selected, selected


class _Payload:
    def __init__(
        self,
        cfg: SimpleNamespace,
        env: SimpleNamespace,
        table: _Table,
        profile_cfg: Profile,
    ) -> None:
        self.table = table
        self.command_dim = profile_cfg.command_dim
        self.error_dim = profile_cfg.error_dim
        self.error_names = tuple(f"error_{index}" for index in range(profile_cfg.error_dim))
        self.reset_assets: list[str] = []
        self.bound = torch.zeros(env.num_envs, profile_cfg.row_width, device=env.device)

    def bind(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        _spawn, target = self.table.gather(task_rows)
        self.resample(env_ids, task_rows, target, None)

    def resample(
        self,
        env_ids: torch.Tensor,
        _task_rows: torch.Tensor,
        target: torch.Tensor,
        _origin: torch.Tensor | None,
    ) -> None:
        self.bound.index_copy_(0, env_ids, target)

    def update(self, _step_dt: float, command: torch.Tensor, error: torch.Tensor) -> None:
        command.copy_(self.bound[:, : self.command_dim])
        error.copy_(self.bound[:, self.command_dim : self.command_dim + self.error_dim])

    def command_std(self) -> torch.Tensor:
        return torch.ones_like(self.bound[:, : self.error_dim])

    def get_task_done(self) -> torch.Tensor:
        return torch.zeros(self.bound.shape[0], dtype=torch.bool, device=self.bound.device)

    def get_task_reward(self) -> torch.Tensor:
        return torch.zeros(self.bound.shape[0], device=self.bound.device)

    def log_metrics(self, _env: object, _success_rates: torch.Tensor) -> None:
        return None

    def set_debug_vis(self, _enabled: bool) -> None:
        return None

    def debug_visualize(self, _env: object) -> None:
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _canonical_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _historical_source(path: Path | None = None) -> bytes:
    if path is not None:
        return path.read_bytes()
    return subprocess.check_output(
        ["git", "show", f"{_HISTORICAL_REVISION}:{_STATE_COMMAND_PATH.as_posix()}"],
        cwd=_REPOSITORY_ROOT,
    )


def _state_command_from_source(
    source: bytes,
    module_name: str,
    *,
    stub_curriculum: bool,
) -> tuple[type, str]:
    filename = f"<{module_name}>"
    text = source.decode()
    linecache.cache[filename] = (len(source), None, text.splitlines(keepends=True), filename)
    module = types.ModuleType(module_name)
    module.__file__ = filename
    module.__package__ = "isaaclab_tasks.core.multi_task.mdp.commands.state_command"
    sys.modules[module_name] = module
    curriculum_name = "isaaclab_tasks.core.multi_task.curriculum"
    previous_curriculum = sys.modules.get(curriculum_name)
    if stub_curriculum:
        curriculum = types.ModuleType(curriculum_name)
        curriculum.set_reset_state = lambda *_args, **_kwargs: None
        sys.modules[curriculum_name] = curriculum
    try:
        exec(compile(text, filename, "exec"), module.__dict__)
    finally:
        if stub_curriculum:
            if previous_curriculum is None:
                del sys.modules[curriculum_name]
            else:
                sys.modules[curriculum_name] = previous_curriculum
    return module.StateCommand, _sha256(source)


def _historical_state_command(path: Path | None = None) -> tuple[type, str]:
    return _state_command_from_source(
        _historical_source(path),
        "_phase3_historical_state_command",
        stub_curriculum=True,
    )


def _current_state_command() -> tuple[type, str]:
    return _state_command_from_source(
        (_REPOSITORY_ROOT / _STATE_COMMAND_PATH).read_bytes(),
        "_phase3_current_state_command",
        stub_curriculum=False,
    )


def _make_cfg(
    profile_cfg: Profile,
    num_tasks: int,
    *,
    randomize: bool,
) -> SimpleNamespace:
    def table_factory(_cfg: object, env: SimpleNamespace) -> _Table:
        return _Table(num_tasks, profile_cfg.row_width, torch.device(env.device))

    def payload_factory(cfg: SimpleNamespace, env: SimpleNamespace, table: _Table) -> _Payload:
        return _Payload(cfg, env, table, profile_cfg)

    return SimpleNamespace(
        task_table=SimpleNamespace(class_type=table_factory),
        payload=SimpleNamespace(class_type=payload_factory),
        commands={},
        randomize_command_indices=randomize,
        states_relative=False,
        resampling_time_range=(1.0, 1.0),
        debug_vis=False,
    )


def _make_env(num_envs: int, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        device=device,
        step_dt=0.02,
        sim=SimpleNamespace(vis_marker_registry=_Registry()),
    )


def _build(
    command_class: type,
    profile_cfg: Profile,
    num_envs: int,
    num_tasks: int,
    device: torch.device,
    *,
    randomize: bool,
) -> Any:
    return command_class(
        _make_cfg(profile_cfg, num_tasks, randomize=randomize),
        _make_env(num_envs, device),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timed(
    operation,
    *,
    rows_per_call: int,
    warmup: int,
    iterations: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int]:
    for _ in range(warmup):
        operation()
    _sync(device)
    samples_ns: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        _sync(device)
        samples_ns.append((time.perf_counter_ns() - started) / iterations)
    median_ns = statistics.median(samples_ns)
    return {
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "median_ns_per_call": median_ns,
        "p95_ns_per_call": _percentile(samples_ns, 0.95),
        "calls_per_second": 1.0e9 / median_ns,
        "env_rows_per_second": rows_per_call * 1.0e9 / median_ns,
    }


def _owned_tensors(term: Any) -> dict[str, torch.Tensor]:
    tensors = {
        "time_left": term.time_left,
        "command_counter": term.command_counter,
        "cmd_indices": term.cmd_indices,
        "command": term._command,
        "error": term._err,
        "payload.bound": term.payload.bound,
        "table.rows": term.table.rows,
    }
    tensors.update({f"metric.{name}": value for name, value in term.metrics.items()})
    success_rates = vars(term).get("success_rates")
    if isinstance(success_rates, torch.Tensor):
        tensors["success_rates"] = success_rates
    return tensors


def _storage_record(term: Any) -> dict[str, Any]:
    tensors = _owned_tensors(term)
    state_names = tuple(name for name in tensors if not name.startswith(("payload.", "table.")))
    return {
        "state_command_owned_bytes": sum(tensors[name].numel() * tensors[name].element_size() for name in state_names),
        "payload_owned_bytes": tensors["payload.bound"].numel() * tensors["payload.bound"].element_size(),
        "table_owned_bytes": tensors["table.rows"].numel() * tensors["table.rows"].element_size(),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
    }


def _allocation_record(operation, term: Any, iterations: int, device: torch.device) -> dict[str, Any]:
    before = {name: value.data_ptr() for name, value in _owned_tensors(term).items()}
    tracemalloc.start()
    if device.type == "cuda":
        _sync(device)
        base_cuda = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_cuda = 0

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, profile_memory=True) as profiler:
        for _ in range(iterations):
            operation()
        _sync(device)

    _current_python, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = {name: value.data_ptr() for name, value in _owned_tensors(term).items()}
    events = profiler.key_averages()
    positive = [event for event in events if event.self_cpu_memory_usage > 0]
    positive_device = [event for event in events if event.self_device_memory_usage > 0]
    device_operator_bytes = {event.key: event.self_device_memory_usage for event in positive_device}
    operator_bytes = {event.key: event.self_cpu_memory_usage for event in positive}
    if device.type == "cuda":
        cuda_peak = max(0, torch.cuda.max_memory_allocated(device) - base_cuda)
    else:
        cuda_peak = 0
    return {
        "profiled_calls": iterations,
        "python_peak_bytes": peak_python,
        "positive_self_cpu_memory_bytes": sum(operator_bytes.values()),
        "positive_self_cpu_memory_operator_calls": sum(event.count for event in positive),
        "positive_self_cpu_memory_by_operator": operator_bytes,
        "cuda_peak_additional_bytes": cuda_peak,
        "positive_self_device_memory_bytes": sum(device_operator_bytes.values()),
        "positive_self_device_memory_operator_calls": sum(event.count for event in positive_device),
        "positive_self_device_memory_by_operator": device_operator_bytes,
        "owned_storage_pointers_stable": before == after,
    }


def _construction_record(
    command_class: type,
    profile_cfg: Profile,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    samples: list[float] = []
    peak_python = 0
    term = None
    for iteration in range(args.init_warmup):
        torch.manual_seed(args.seed + iteration)
        warm = _build(
            command_class,
            profile_cfg,
            args.num_envs,
            args.num_tasks,
            device,
            randomize=True,
        )
        _sync(device)
        del warm

    for iteration in range(args.init_iterations):
        torch.manual_seed(args.seed + iteration)
        tracemalloc.start()
        _sync(device)
        started = time.perf_counter_ns()
        term = _build(
            command_class,
            profile_cfg,
            args.num_envs,
            args.num_tasks,
            device,
            randomize=True,
        )
        _sync(device)
        samples.append(float(time.perf_counter_ns() - started))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_python = max(peak_python, peak)
    assert term is not None
    return {
        "samples": args.init_iterations,
        "median_ns": statistics.median(samples),
        "p95_ns": _percentile(samples, 0.95),
        "python_peak_bytes": peak_python,
        "storage": _storage_record(term),
    }


def _semantic_equivalence(
    current_class: type,
    historical_class: type,
    profile_cfg: Profile,
    args: argparse.Namespace,
    device: torch.device,
) -> bool:
    torch.manual_seed(args.seed)
    current = _build(current_class, profile_cfg, args.num_envs, args.num_tasks, device, randomize=True)
    torch.manual_seed(args.seed)
    historical = _build(historical_class, profile_cfg, args.num_envs, args.num_tasks, device, randomize=True)
    _sync(device)
    equal = (
        torch.equal(current.cmd_indices, historical.cmd_indices)
        and torch.equal(current.command, historical.command)
        and torch.equal(current.error, historical.error)
        and torch.equal(current.payload.bound, historical.payload.bound)
    )
    del current, historical
    return bool(equal)


def _implementation_record(
    command_class: type,
    profile_cfg: Profile,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    randomized = _build(
        command_class,
        profile_cfg,
        args.num_envs,
        args.num_tasks,
        device,
        randomize=True,
    )
    torch.manual_seed(args.seed)
    pinned = _build(
        command_class,
        profile_cfg,
        args.num_envs,
        args.num_tasks,
        device,
        randomize=False,
    )
    env_ids = torch.arange(args.batch_size, device=device, dtype=torch.int64)

    def randomized_operation() -> None:
        randomized._resample_command(env_ids)

    def pinned_operation() -> None:
        pinned._resample_command(env_ids)

    update_operation = randomized._update_command
    return {
        "construction": _construction_record(command_class, profile_cfg, args, device),
        "randomized_resample": _timed(
            randomized_operation,
            rows_per_call=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            device=device,
        ),
        "pinned_resample": _timed(
            pinned_operation,
            rows_per_call=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            device=device,
        ),
        "update": _timed(
            update_operation,
            rows_per_call=args.num_envs,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            device=device,
        ),
        "randomized_resample_allocations": _allocation_record(
            randomized_operation,
            randomized,
            args.profile_iterations,
            device,
        ),
        "pinned_resample_allocations": _allocation_record(
            pinned_operation,
            pinned,
            args.profile_iterations,
            device,
        ),
        "update_allocations": _allocation_record(
            update_operation,
            randomized,
            args.profile_iterations,
            device,
        ),
    }


def _ratio(current: float, historical: float) -> float:
    return current / historical


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if args.batch_size > args.num_envs:
        raise ValueError("batch_size must not exceed num_envs.")
    torch.set_num_threads(args.torch_threads)
    historical_class, historical_sha256 = _historical_state_command(args.historical_source)
    current_class, current_sha256 = _current_state_command()
    configuration = {
        "device": str(device),
        "num_envs": args.num_envs,
        "num_tasks": args.num_tasks,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "init_iterations": args.init_iterations,
        "init_warmup": args.init_warmup,
        "profile_iterations": args.profile_iterations,
        "torch_threads": args.torch_threads,
        "seed": args.seed,
        "profiles": [asdict(value) for value in _PROFILES],
    }

    implementations = {"current": current_class, "historical_reference": historical_class}
    results: dict[str, dict[str, Any]] = {}
    equivalence: dict[str, bool] = {}
    for profile_cfg in _PROFILES:
        equivalence[profile_cfg.name] = _semantic_equivalence(
            current_class,
            historical_class,
            profile_cfg,
            args,
            device,
        )
        results[profile_cfg.name] = {
            name: _implementation_record(command_class, profile_cfg, args, device)
            for name, command_class in implementations.items()
        }

    comparisons = {}
    for profile_name, profile_result in results.items():
        current = profile_result["current"]
        historical = profile_result["historical_reference"]
        comparisons[profile_name] = {
            "construction_median_ratio": _ratio(
                current["construction"]["median_ns"],
                historical["construction"]["median_ns"],
            ),
            "randomized_resample_median_ratio": _ratio(
                current["randomized_resample"]["median_ns_per_call"],
                historical["randomized_resample"]["median_ns_per_call"],
            ),
            "pinned_resample_median_ratio": _ratio(
                current["pinned_resample"]["median_ns_per_call"],
                historical["pinned_resample"]["median_ns_per_call"],
            ),
            "update_median_ratio": _ratio(
                current["update"]["median_ns_per_call"],
                historical["update"]["median_ns_per_call"],
            ),
            "state_owned_byte_ratio": _ratio(
                current["construction"]["storage"]["state_command_owned_bytes"],
                historical["construction"]["storage"]["state_command_owned_bytes"],
            ),
        }

    ratio_keys = (
        "construction_median_ratio",
        "randomized_resample_median_ratio",
        "pinned_resample_median_ratio",
        "update_median_ratio",
    )
    threshold = 1.25
    semantic_passed = all(equivalence.values())
    pointer_stability_passed = all(
        record["owned_storage_pointers_stable"]
        for profile_result in results.values()
        for implementation in profile_result.values()
        for key, record in implementation.items()
        if key.endswith("_allocations")
    )
    reference_ratio_passed = all(
        comparison[key] <= threshold for comparison in comparisons.values() for key in ratio_keys
    )
    gate_passed = semantic_passed and pointer_stability_passed and reference_ratio_passed

    cfg_path = _REPOSITORY_ROOT / _STATE_COMMAND_CFG_PATH
    script_path = Path(__file__).resolve()
    return {
        "schema": "forward_backward_phase3_state_command_systems_measurement_v1",
        "status": "passed" if gate_passed else "failed",
        "scope": {
            "measured": (
                "StateCommand construction and direct tensor-payload delegation on the declared CPU or CUDA device."
            ),
            "not_measured": [
                "simulator reset writes",
                "Position sensor and robot-state kernels",
                "Factory symmetry and Warp kernels",
                "historical lazy observation-cache materialization",
            ],
            "interpretation": (
                "The historical reference isolates shell ownership and dispatch; it is not an end-to-end domain "
                "benchmark."
            ),
        },
        "baseline": {
            "hot_shell_reference": (
                "Exact StateCommand source recovered from the frozen migration revision and run with tensor-equivalent "
                "table/payload work."
            ),
            "full_domain_before_after": {
                "status": "waived_not_reconstructable_as_controlled_comparison",
                "reason": (
                    "The historical cache path teleported simulator assets and recomputed observations; its complete "
                    "dependency state and matching runtime artifacts were not frozen. Recreating it in the final tree "
                    "would change the system being measured."
                ),
            },
        },
        "identity": {
            "repository_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPOSITORY_ROOT,
                text=True,
            ).strip(),
            "historical_revision": _HISTORICAL_REVISION,
            "historical_state_command_sha256": historical_sha256,
            "current_state_command_sha256": current_sha256,
            "current_state_command_cfg_sha256": _file_sha256(cfg_path),
            "benchmark_script_sha256": _file_sha256(script_path),
            "configuration_sha256": _canonical_sha256(configuration),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
            "cuda": torch.version.cuda,
        },
        "configuration": configuration,
        "semantic_equivalence": equivalence,
        "results": results,
        "comparison": comparisons,
        "decision": {
            "gate_passed": gate_passed,
            "semantic_equivalence_passed": semantic_passed,
            "owned_storage_pointer_stability_passed": pointer_stability_passed,
            "reference_ratio_gate_passed": reference_ratio_passed,
            "absolute_metrics_only_for_domain_payloads": True,
            "no_material_regression_threshold_ratio": threshold,
            "reference_ratio_gate_applies_to": list(ratio_keys),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--historical_source", type=Path)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_tasks", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--init_iterations", type=int, default=20)
    parser.add_argument("--init_warmup", type=int, default=5)
    parser.add_argument("--profile_iterations", type=int, default=20)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
