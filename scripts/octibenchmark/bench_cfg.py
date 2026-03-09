# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative benchmark configuration.

Define benchmark matrices as Python dataclasses. Each :class:`BenchmarkRun`
describes a single nsys profiling run. :class:`BenchmarkMatrix` generates
all combinations from the user-specified axes.

All Hydra overrides (presets, resolution, physics backend, etc.) are passed
through :attr:`BenchmarkMatrix.hydra_sweeps` or :attr:`BenchmarkMatrix.hydra_args`.
Nothing is hardcoded to a specific task or config schema.

Example — define a custom benchmark suite::

    from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher, Phase

    # Profile the stepping loop
    matrix = BenchmarkMatrix(
        tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
        num_envs=[2048, 4096, 8192],
        hydra_sweeps={
            "preset": [
                "presets=newton,newton_renderer,rgb",
                "presets=newton,newton_renderer,depth",
            ],
            "resolution": [
                "env.tiled_camera.width=128 env.tiled_camera.height=128",
                "env.tiled_camera.width=256 env.tiled_camera.height=256",
            ],
        },
        launcher=Launcher.NON_RL,
        num_frames=100,
    )

    # Profile startup time only
    startup_matrix = BenchmarkMatrix(
        tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
        num_envs=[2048, 4096, 8192],
        hydra_sweeps={"preset": ["presets=newton,newton_renderer,rgb"]},
        phase=Phase.STARTUP,
    )

    # Iterate over all runs
    for run in matrix.runs():
        print(run.tag)

    # Or directly launch everything
    matrix.execute(wandb_project="my-benchmarks")
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_BENCHMARK_SCRIPT = _SCRIPT_DIR / "benchmark.py"
_BENCHMARK_TRAIN_SCRIPT = _SCRIPT_DIR / "benchmark_train.py"


class Launcher(enum.Enum):
    """Which benchmark script to use."""

    NON_RL = "non_rl"
    """Random actions, no network training or inference."""

    RSL_RL = "rsl_rl"
    """RSL-RL on-policy training loop with NVTX instrumentation."""


class Phase(enum.Enum):
    """What part of the benchmark to profile."""

    STEP = "step"
    """Profile the stepping/training loop only. NVTX hooks are installed
    after warmup, right before nsys capture starts. Environment creation
    and warmup run without any instrumentation overhead."""

    STARTUP = "startup"
    """Profile environment creation and first reset only. nsys capture
    covers ``gym.make()`` + ``env.reset()`` (and runner creation for
    RSL_RL). No stepping loop is executed."""


@dataclasses.dataclass
class BenchmarkRun:
    """A single benchmark invocation with all parameters resolved."""

    task: str
    num_envs: int
    hydra_overrides: dict[str, str]
    """Resolved Hydra overrides from sweep axes. Keys are axis names, values
    are the override strings chosen for this run."""
    launcher: Launcher
    phase: Phase
    num_frames: int
    warmup_frames: int
    max_iterations: int
    """Only used for RSL_RL launcher."""
    hydra_args: list[str] = dataclasses.field(default_factory=list)
    """Constant Hydra overrides applied to every run."""
    extra_nvtx_hooks: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    """Extra NVTX hooks as ``(attr_path, label)`` tuples. See
    :func:`~octibenchmark.nvtx_hooks.install_extra_nvtx_hooks`."""

    @property
    def tag(self) -> str:
        """Short unique tag for file naming and wandb grouping."""
        parts = [self.task, f"envs{self.num_envs}"]
        for axis_name, override_str in sorted(self.hydra_overrides.items()):
            # Produce a compact label from the override string:
            # "presets=newton,newton_renderer,rgb" → "newton_newton_renderer_rgb"
            # "env.tiled_camera.width=128 env.tiled_camera.height=128" → "128_128"
            values = []
            for token in override_str.split():
                if "=" in token:
                    values.append(token.split("=", 1)[1])
                else:
                    values.append(token)
            label = "_".join(v.replace(",", "_") for v in values)
            parts.append(label)
        parts.append(self.launcher.value)
        if self.phase != Phase.STEP:
            parts.append(self.phase.value)
        return "__".join(parts)

    @property
    def all_hydra_args(self) -> list[str]:
        """Build the full Hydra CLI arg list for this run."""
        args = []
        for override_str in self.hydra_overrides.values():
            args.extend(override_str.split())
        args.extend(self.hydra_args)
        return args

    @property
    def script_path(self) -> Path:
        if self.launcher == Launcher.RSL_RL:
            return _BENCHMARK_TRAIN_SCRIPT
        return _BENCHMARK_SCRIPT

    def nsys_command(self, output_path: str) -> list[str]:
        """Build the full nsys + python command for this run."""
        script = str(self.script_path)
        cmd = [
            "nsys", "profile",
            "-t", "cuda,nvtx",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "-o", output_path,
            "--force-overwrite=true",
            sys.executable, script,
            "--task", self.task,
            "--num_envs", str(self.num_envs),
            "--phase", self.phase.value,
            "--headless",
        ]
        if self.launcher == Launcher.NON_RL:
            cmd.extend(["--num_frames", str(self.num_frames)])
            cmd.extend(["--warmup_frames", str(self.warmup_frames)])
        elif self.launcher == Launcher.RSL_RL:
            cmd.extend(["--max_iterations", str(self.max_iterations)])
            cmd.extend(["--warmup_frames", str(self.warmup_frames)])
        if self.extra_nvtx_hooks:
            cmd.extend(["--extra_nvtx_hooks", json.dumps(self.extra_nvtx_hooks)])
        cmd.extend(self.all_hydra_args)
        return cmd


@dataclasses.dataclass
class BenchmarkMatrix:
    """Declarative benchmark matrix that generates all runs.

    Define the axes you want to sweep and call :meth:`runs` to get
    the list of :class:`BenchmarkRun` instances.

    Fixed axes:
        ``tasks`` and ``num_envs`` are always sweep axes.

    Hydra sweep axes:
        ``hydra_sweeps`` is a dict mapping axis names to lists of Hydra
        override strings. The matrix generates the cross-product of all
        axes. Each override string can contain multiple space-separated
        overrides that belong together (e.g. width + height).

    Constant overrides:
        ``hydra_args`` are appended to every run unchanged.
    """

    tasks: list[str]
    """Task names to benchmark."""

    num_envs: list[int]
    """Environment counts to sweep."""

    hydra_sweeps: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    """Named sweep axes of Hydra overrides.

    Keys are axis names (used in tags and wandb grouping). Values are lists
    of Hydra override strings to sweep over. Each string can contain
    multiple space-separated overrides that belong together.

    Example::

        hydra_sweeps={
            "preset": [
                "presets=newton,newton_renderer,rgb",
                "presets=newton,newton_renderer,depth",
            ],
            "resolution": [
                "env.tiled_camera.width=128 env.tiled_camera.height=128",
                "env.tiled_camera.width=256 env.tiled_camera.height=256",
            ],
        }
    """

    hydra_args: list[str] = dataclasses.field(default_factory=list)
    """Constant Hydra overrides applied to every run."""

    launcher: Launcher = Launcher.NON_RL
    """Benchmark launcher type."""

    phase: Phase = Phase.STEP
    """What to profile: stepping loop or startup."""

    num_frames: int = 100
    """Env steps per run (non-RL only)."""

    warmup_frames: int = 10
    """Warmup steps before nsys capture."""

    max_iterations: int = 10
    """Training iterations (RSL_RL only)."""

    extra_nvtx_hooks: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    """Extra NVTX hooks as ``(attr_path, label)`` tuples passed to every run.

    Use this to annotate task-specific methods without modifying
    ``nvtx_hooks.py``. Example::

        extra_nvtx_hooks=[
            ("_compute_image_observations", "vision.image_obs"),
            ("feature_extractor.step", "vision.feature_extractor"),
        ]
    """

    def runs(self) -> list[BenchmarkRun]:
        """Generate all benchmark runs from the matrix axes.

        Returns:
            List of :class:`BenchmarkRun` instances.
        """
        # Build the sweep axes: tasks × num_envs × hydra_sweep_1 × hydra_sweep_2 × ...
        axis_names = sorted(self.hydra_sweeps.keys())
        axis_values = [self.hydra_sweeps[name] for name in axis_names]

        # If no hydra sweeps, use a single empty combo
        if axis_values:
            hydra_combos = list(itertools.product(*axis_values))
        else:
            hydra_combos = [()]

        all_runs = []
        for task, n_envs, *_ in itertools.product(self.tasks, self.num_envs, [None]):
            for combo in hydra_combos:
                overrides = dict(zip(axis_names, combo))
                run = BenchmarkRun(
                    task=task,
                    num_envs=n_envs,
                    hydra_overrides=overrides,
                    launcher=self.launcher,
                    phase=self.phase,
                    num_frames=self.num_frames,
                    warmup_frames=self.warmup_frames,
                    max_iterations=self.max_iterations,
                    hydra_args=list(self.hydra_args),
                    extra_nvtx_hooks=list(self.extra_nvtx_hooks),
                )
                all_runs.append(run)

        return all_runs

    def _print_dry_run(
        self,
        runs: list[BenchmarkRun],
        output_dir: str,
        verbose: bool,
    ) -> None:
        """Print a dry run summary of the benchmark runs."""
        # Discover which hydra sweep axes are present
        axis_names = sorted({k for r in runs for k in r.hydra_overrides})

        # Shorten task names by stripping common prefix
        tasks = [r.task for r in runs]
        if len(set(tasks)) == 1:
            task_labels = {tasks[0]: tasks[0]}
        else:
            prefix = os.path.commonprefix(tasks)
            # Cut at the last separator to avoid partial words
            cut = max(prefix.rfind("-"), prefix.rfind("_"), 0)
            if cut > 0:
                task_labels = {t: t[cut + 1:] for t in set(tasks)}
            else:
                task_labels = {t: t for t in set(tasks)}

        # Compute column widths from data
        def _override_label(override: str) -> str:
            vals = []
            for token in override.split():
                if "=" in token:
                    vals.append(token.split("=", 1)[1])
                else:
                    vals.append(token)
            return ",".join(vals) if vals else "—"

        columns = ["#", "task", "num_envs"] + axis_names + ["launcher", "phase"]
        col_widths = {c: len(c) for c in columns}
        rows = []
        for i, run in enumerate(runs, 1):
            row = {
                "#": str(i),
                "task": task_labels[run.task],
                "num_envs": str(run.num_envs),
                "launcher": run.launcher.value,
                "phase": run.phase.value,
            }
            for axis in axis_names:
                row[axis] = _override_label(run.hydra_overrides.get(axis, ""))
            rows.append(row)
            for c in columns:
                col_widths[c] = max(col_widths[c], len(row[c]))

        # Print table
        header = "  ".join(
            f"{c:>{col_widths[c]}}" if c == "#" else f"{c:<{col_widths[c]}}"
            for c in columns
        )
        print(f"\n{header}")
        print("─" * len(header))
        for row in rows:
            line = "  ".join(
                f"{row[c]:>{col_widths[c]}}" if c == "#" else f"{row[c]:<{col_widths[c]}}"
                for c in columns
            )
            print(line)

        if verbose:
            print(f"\nFull commands:\n")
            for i, run in enumerate(runs, 1):
                out_path = os.path.join(output_dir, run.tag)
                cmd = run.nsys_command(out_path)
                print(f"  [{i}] {' '.join(cmd)}\n")

    def execute(
        self,
        output_dir: str | None = None,
        wandb_project: str = "isaaclab-benchmarks",
        wandb_entity: str | None = None,
        kernel_patterns: list[str] | None = None,
        no_wandb: bool = False,
        tag: str | None = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> dict[str, dict]:
        """Execute all runs and optionally log to wandb.

        Args:
            output_dir: Directory for nsys outputs. Auto-created if None.
            wandb_project: wandb project name.
            wandb_entity: wandb entity (team/user).
            kernel_patterns: Regex patterns for GPU kernel aggregation.
            no_wandb: Skip wandb logging.
            tag: Custom tag for wandb (e.g. GPU name like ``RTX4090``).
                Appended to wandb group and run names to distinguish
                the same experiment run on different hardware.
            dry_run: Print summary table without executing.
            verbose: Show full nsys commands in dry run.

        Returns:
            Dict mapping run tags to analysis results.
        """
        valid_runs = self.runs()
        if not valid_runs:
            print("[BenchmarkMatrix] No runs to execute.")
            return {}

        if dry_run:
            self._print_dry_run(valid_runs, output_dir or "<auto>", verbose)
            return {}

        # Only create the output directory when actually running
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="octibench_")
        os.makedirs(output_dir, exist_ok=True)

        print(f"[BenchmarkMatrix] {len(valid_runs)} runs to execute.")
        print(f"[BenchmarkMatrix] Output directory: {output_dir}")

        all_results = {}
        failed_runs = []
        for i, run in enumerate(valid_runs):
            out_path = os.path.join(output_dir, run.tag)
            cmd = run.nsys_command(out_path)

            print(f"  [{i+1}/{len(valid_runs)}] {run.tag} ... ", end="", flush=True)

            result = subprocess.run(cmd, capture_output=True, text=True)
            nsys_rep = out_path + ".nsys-rep"

            if result.returncode != 0 or not os.path.exists(nsys_rep):
                # Detect common failure reasons from stderr
                stderr = result.stderr or ""
                if "OutOfMemoryError" in stderr or "out of memory" in stderr.lower():
                    reason = "OOM"
                elif "HYDRA_FULL_ERROR" in stderr:
                    reason = "config error"
                else:
                    reason = f"exit code {result.returncode}"
                print(f"FAILED ({reason})")
                failed_runs.append((run.tag, reason))
                if verbose:
                    # Show last few lines of stderr for debugging
                    tail = stderr.strip().split("\n")[-5:]
                    for line in tail:
                        print(f"    {line}")
                continue

            from octibenchmark.analyze import analyze
            analysis = analyze(nsys_rep, kernel_patterns)

            # Attach run metadata to the analysis for wandb grouping
            analysis["_meta"] = dataclasses.asdict(run)
            all_results[run.tag] = analysis
            print("OK")

        if failed_runs:
            print(f"\n[BenchmarkMatrix] {len(failed_runs)} run(s) failed:")
            for tag, reason in failed_runs:
                print(f"  {tag}: {reason}")

        # Save raw results
        json_path = os.path.join(output_dir, "matrix_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n[BenchmarkMatrix] Results saved to: {json_path}")

        # Log to wandb
        if not no_wandb and all_results:
            _log_matrix_to_wandb(
                all_results, wandb_project, wandb_entity, kernel_patterns, tag,
            )

        return all_results


def _shorten_task(task: str) -> str:
    """Shorten an IsaacLab task name for wandb metric labels.

    Args:
        task: Full task name like ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0``.

    Returns:
        Shortened name like ``Shadow-Vision``.
    """
    s = task
    for prefix in ("Isaac-", "Repose-Cube-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ("-v0", "-v1", "-Direct"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def _shorten_override(value: str) -> str:
    """Shorten a hydra override value for wandb metric labels.

    Examples:
        ``presets=newton,newton_renderer,rgb`` → ``newton_rgb``
        ``env.tiled_camera.width=128 env.tiled_camera.height=128`` → ``128x128``
        ``presets=newton`` → ``newton``

    Args:
        value: Hydra override string.

    Returns:
        Compact label for display.
    """
    import re

    # Resolution pattern: width=X height=Y → XxY
    dims = re.findall(r"(?:width|height)=(\d+)", value)
    if len(dims) >= 2:
        return f"{dims[0]}x{dims[1]}"

    # Presets pattern: presets=a,b,c → a_c (skip middle "renderer" parts)
    if "presets=" in value:
        presets = value.split("presets=")[-1].split(",")
        # Keep first and last if they differ, skip *_renderer
        keep = [p for p in presets if "renderer" not in p]
        return "_".join(keep) if keep else "_".join(presets)

    # Fallback: extract values from key=value pairs
    vals = re.findall(r"=([^\s]+)", value)
    return "_".join(vals) if vals else value


def _log_matrix_to_wandb(
    all_results: dict[str, dict],
    wandb_project: str,
    wandb_entity: str | None,
    kernel_patterns: list[str] | None,
    tag: str | None = None,
):
    """Log benchmark matrix results to wandb.

    Creates one wandb run per config group (task + overrides + launcher),
    all within the same wandb group. This makes wandb overlay different
    configs as separate lines on the same chart panel.

    Each chart panel = one NVTX range (e.g. ``env.step``).
    Each line = one config (e.g. ``newton_rgb/128x128``).
    X-axis = ``num_envs``.

    Args:
        all_results: Dict mapping run tags to analysis results.
        wandb_project: wandb project name.
        wandb_entity: wandb entity (team/user).
        kernel_patterns: Kernel patterns used (for labeling).
        tag: Custom tag (e.g. GPU name) appended to wandb group and
            run names to distinguish the same experiment across hardware.
    """
    import wandb

    # Collect all unique tasks for naming
    tasks = sorted({r["_meta"]["task"] for r in all_results.values()})

    # Discover all hydra sweep axis names from the metadata
    sweep_axis_names = set()
    for result in all_results.values():
        sweep_axis_names.update(result["_meta"].get("hydra_overrides", {}).keys())
    sweep_axis_names = sorted(sweep_axis_names)

    # Group by (task, hydra_overrides, launcher) → sweep num_envs
    from collections import defaultdict
    groups = defaultdict(list)
    for run_tag, result in all_results.items():
        meta = result["_meta"]
        overrides = meta.get("hydra_overrides", {})
        override_key = tuple(overrides.get(a, "") for a in sweep_axis_names)
        group_key = (meta["task"], override_key, meta["launcher"])
        groups[group_key].append((meta["num_envs"], result))

    # wandb group name — shared across all runs so they overlay
    short_tasks = "_".join(_shorten_task(t) for t in tasks)
    wandb_group = f"matrix_{short_tasks}" if len(tasks) <= 3 else f"matrix_{len(tasks)}_tasks"
    if tag:
        wandb_group = f"{wandb_group}_{tag}"

    run_urls = []
    for group_idx, (group_key, env_results) in enumerate(groups.items()):
        task, override_key, launcher = group_key
        config_parts = [_shorten_override(v) for v in override_key if v]
        config_label = "/".join(config_parts) if config_parts else "default"
        short_task = _shorten_task(task)
        run_name = f"{short_task}/{config_label}/{launcher}"
        if tag:
            run_name = f"{run_name} [{tag}]"

        run_config = {
            "task": task,
            "launcher": launcher,
            "overrides": {k: v for k, v in zip(sweep_axis_names, override_key)},
        }
        if tag:
            run_config["tag"] = tag

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=run_name,
            tags=[tag] if tag else None,
            config=run_config,
            reinit=True,
        )

        # Set num_envs as x-axis for all metrics
        wandb.define_metric("num_envs")
        wandb.define_metric("*", step_metric="num_envs")

        # Log the summary table only in the first run
        if group_idx == 0:
            table_rows = []
            base_columns = ["task", "num_envs"] + sweep_axis_names + ["launcher", "phase"]
            metric_columns = ["nvtx_range", "calls", "total_ms", "avg_ms", "pct_of_step"]

            for run_tag, result in all_results.items():
                meta = result["_meta"]
                ovr = meta.get("hydra_overrides", {})
                for nvtx in result.get("nvtx_ranges", []):
                    step_total_ns = 0
                    for r in result.get("nvtx_ranges", []):
                        if r["name"] == "env.step":
                            step_total_ns = r["total_ns"]
                            break
                    pct = (nvtx["total_ns"] / step_total_ns * 100) if step_total_ns else 0
                    row = [meta["task"], meta["num_envs"]]
                    for axis_name in sweep_axis_names:
                        row.append(ovr.get(axis_name, ""))
                    row.extend([
                        meta["launcher"], meta.get("phase", "step"),
                        nvtx["name"], nvtx["count"],
                        nvtx["total_ns"] / 1e6, nvtx["avg_ns"] / 1e6, pct,
                    ])
                    table_rows.append(row)
            table = wandb.Table(columns=base_columns + metric_columns, data=table_rows)
            wandb.log({"benchmark_matrix": table})

        # Log line chart data — simple metric names so all runs share panels
        for num_envs, result in sorted(env_results, key=lambda x: x[0]):
            log_data = {"num_envs": num_envs}

            for nvtx in result.get("nvtx_ranges", []):
                log_data[f"{nvtx['name']} (ms)"] = nvtx["avg_ns"] / 1e6

                if nvtx["name"] == "env.step" and nvtx["avg_ns"] > 0:
                    log_data["effective_fps"] = num_envs / (nvtx["avg_ns"] / 1e9)

            if result.get("kernel_aggregations"):
                for pattern, stats in result["kernel_aggregations"].items():
                    log_data[f"kernel:{pattern} (ms)"] = stats["total_ms"]

            wandb.log(log_data)

        run_urls.append(run.url)
        run.finish()

    print(f"\n[BenchmarkMatrix] wandb group: {wandb_group}")
    print(f"[BenchmarkMatrix] {len(run_urls)} wandb runs created")
