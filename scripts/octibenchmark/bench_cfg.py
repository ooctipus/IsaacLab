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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_BENCHMARK_SCRIPT = _SCRIPT_DIR / "benchmark.py"
_BENCHMARK_TRAIN_SCRIPT = _SCRIPT_DIR / "benchmark_train.py"

_gpu_metrics_cache: bool | None = None


def _gpu_metrics_supported() -> bool:
    """Check whether ``nsys --gpu-metrics-devices=all`` is usable.

    Runs a lightweight ``nsys`` dry-run once and caches the result.
    Returns False when the installed GPUs lack privilege or driver
    support for hardware metric collection.
    """
    global _gpu_metrics_cache
    if _gpu_metrics_cache is not None:
        return _gpu_metrics_cache
    try:
        probe = subprocess.run(
            ["nsys", "profile", "--gpu-metrics-devices=all", "-o", "/dev/null", "--force-overwrite=true", "sleep", "0"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        _gpu_metrics_cache = probe.returncode == 0
    except Exception:
        _gpu_metrics_cache = False
    if not _gpu_metrics_cache:
        print("[INFO] GPU hardware metrics not available (privilege or driver). Skipping --gpu-metrics-devices.")
    return _gpu_metrics_cache


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


class ProfileLevel(enum.Enum):
    """How much detail nsys captures.

    Higher levels produce richer ``.nsys-rep`` files at the cost of
    increased overhead and larger output files.
    """

    PLAIN = "plain"
    """No nsys profiling.  Runs the benchmark script directly and captures
    only FPS and GPU memory from stdout.  Zero profiling overhead."""

    LIGHT = "light"
    """Minimal capture: CUDA + NVTX traces only (default).  Low overhead,
    suitable for quick scaling sweeps."""

    FULL = "full"
    """Rich capture: adds GPU hardware metrics, CUDA memory tracking,
    OS runtime traces, GPU context switches, CUDA graph detail, and
    (when Kit/IsaacSim is in use) Kit-internal NVTX annotations."""


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
    profile_level: ProfileLevel = ProfileLevel.LIGHT
    """Controls nsys capture richness. See :class:`ProfileLevel`."""

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

        trace_apis = "cuda,nvtx"
        if self.profile_level == ProfileLevel.FULL:
            trace_apis = "cuda,nvtx,osrt"

        cmd = [
            "nsys",
            "profile",
            "-t",
            trace_apis,
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "-o",
            output_path,
            "--force-overwrite=true",
        ]

        if self.profile_level == ProfileLevel.FULL:
            if _gpu_metrics_supported():
                cmd.append("--gpu-metrics-devices=all")
            cmd.extend(
                [
                    "--cuda-memory-usage=true",
                    "--gpuctxsw=true",
                    "--cuda-graph-trace=graph:host-and-device",
                ]
            )

        cmd.extend(
            [
                sys.executable,
                script,
                "--task",
                self.task,
                "--num_envs",
                str(self.num_envs),
                "--phase",
                self.phase.value,
                "--headless",
            ]
        )
        if self.launcher == Launcher.NON_RL:
            cmd.extend(["--num_frames", str(self.num_frames)])
            cmd.extend(["--warmup_frames", str(self.warmup_frames)])
        elif self.launcher == Launcher.RSL_RL:
            cmd.extend(["--max_iterations", str(self.max_iterations)])
            cmd.extend(["--warmup_frames", str(self.warmup_frames)])
        if self.extra_nvtx_hooks:
            cmd.extend(["--extra_nvtx_hooks", json.dumps(self.extra_nvtx_hooks)])

        if self.profile_level == ProfileLevel.FULL:
            cmd.extend(
                [
                    "--kit_args",
                    "--/app/profileFromStart=true"
                    " --/profiler/enabled=true"
                    " --/app/profilerBackend=nvtx"
                    " --/app/profilerMask=1"
                    " --/plugins/carb.profiler-tracy.plugin/fibersAsThreads=false"
                    " --/profiler/channels/carb.events/enabled=false"
                    " --/profiler/channels/carb.tasking/enabled=false",
                ]
            )

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

    profile_level: ProfileLevel = ProfileLevel.LIGHT
    """Controls nsys capture richness for all runs in this matrix.
    See :class:`ProfileLevel`."""

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
        for task, combo in itertools.product(self.tasks, hydra_combos):
            for n_envs in self.num_envs:
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
                    profile_level=self.profile_level,
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
                task_labels = {t: t[cut + 1 :] for t in set(tasks)}
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
        header = "  ".join(f"{c:>{col_widths[c]}}" if c == "#" else f"{c:<{col_widths[c]}}" for c in columns)
        print(f"\n{header}")
        print("─" * len(header))
        for row in rows:
            line = "  ".join(
                f"{row[c]:>{col_widths[c]}}" if c == "#" else f"{row[c]:<{col_widths[c]}}" for c in columns
            )
            print(line)

        if verbose:
            print("\nFull commands:\n")
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
        profile_level: ProfileLevel | None = None,
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
            profile_level: Override the matrix's :attr:`profile_level`
                for this execution.  When ``None``, uses the value set
                on the matrix itself.

        Returns:
            Dict mapping run tags to analysis results.
        """
        if profile_level is not None:
            self.profile_level = profile_level

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

        # Protect the parent process from the OOM killer — lower our OOM
        # score so Linux strongly prefers killing child subprocesses.
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("-500")
        except (PermissionError, FileNotFoundError):
            pass

        print(f"[BenchmarkMatrix] {len(valid_runs)} runs to execute.")
        print(f"[BenchmarkMatrix] Output directory: {output_dir}")

        use_wandb = not no_wandb
        if use_wandb:
            os.environ["WANDB_SILENT"] = "true"
            sweep_axis_names = sorted({k for r in valid_runs for k in r.hydra_overrides})

        # Group runs by config so we know when a group is complete.
        from collections import defaultdict

        pending_groups: dict[tuple, list] = defaultdict(list)

        all_results = {}
        failed_runs = []
        wandb_run_count = 0
        wandb_group: str | None = None
        prev_group_key: tuple | None = None
        skip_group: tuple | None = None
        for i, run in enumerate(valid_runs):
            # Compute group key for this run
            if use_wandb:
                overrides = run.hydra_overrides
                override_key = tuple(overrides.get(a, "") for a in sweep_axis_names)
                cur_group_key = (run.task, override_key, run.launcher.value)
            else:
                overrides = run.hydra_overrides
                cur_group_key = (run.task, tuple(sorted(overrides.items())), run.launcher.value)

            # Flush the previous group when we transition to a new one
            if prev_group_key is not None and cur_group_key != prev_group_key:
                if use_wandb and pending_groups[prev_group_key]:
                    wandb_group = _flush_wandb_group(
                        prev_group_key,
                        pending_groups.pop(prev_group_key),
                        sweep_axis_names,
                        wandb_project,
                        wandb_entity,
                        wandb_group,
                        tag,
                        output_dir,
                        valid_runs,
                    )
                    wandb_run_count += 1
                if skip_group == prev_group_key:
                    skip_group = None
            prev_group_key = cur_group_key

            # If a previous run in this group OOM'd/crashed, skip remaining
            # (larger num_envs will only make it worse)
            if skip_group == cur_group_key:
                reason = "skipped (earlier run in group failed)"
                if verbose:
                    print(f"  [{i + 1}/{len(valid_runs)}] {run.tag}")
                    print(f"    SKIPPED (earlier failure in group)")
                else:
                    print(f"  [{i + 1}/{len(valid_runs)}] {run.tag} ... SKIPPED")
                failed_runs.append((run.tag, reason))
                continue

            out_path = os.path.join(output_dir, run.tag)
            is_plain = run.profile_level == ProfileLevel.PLAIN
            if is_plain:
                script = str(run.script_path)
                cmd = [sys.executable, script, "--task", run.task, "--num_envs", str(run.num_envs),
                       "--phase", run.phase.value, "--headless"]
                if run.launcher == Launcher.NON_RL:
                    cmd.extend(["--num_frames", str(run.num_frames), "--warmup_frames", str(run.warmup_frames)])
                elif run.launcher == Launcher.RSL_RL:
                    cmd.extend(["--max_iterations", str(run.max_iterations), "--warmup_frames", str(run.warmup_frames)])
                cmd.extend(run.all_hydra_args)
            else:
                cmd = run.nsys_command(out_path)

            if verbose:
                print(f"  [{i + 1}/{len(valid_runs)}] {run.tag}")
                print(f"    $ {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                stderr = result.stderr or ""
                stdout = result.stdout or ""
                if stderr:
                    sys.stderr.write(stderr)
                if stdout:
                    sys.stdout.write(stdout)
            else:
                print(f"  [{i + 1}/{len(valid_runs)}] {run.tag} ... ", end="", flush=True)
                result = subprocess.run(cmd, capture_output=True, text=True)
                stderr = result.stderr or ""
                stdout = result.stdout or ""

            combined_out = stdout + "\n" + stderr

            if is_plain:
                # PLAIN: only check exit code, build analysis from stdout
                if result.returncode != 0:
                    is_oom = "OutOfMemoryError" in combined_out or "out of memory" in combined_out.lower()
                    is_killed = result.returncode in (-9, 137)
                    if is_oom or is_killed:
                        reason = "OOM" if is_oom else "killed (likely OOM)"
                        skip_group = cur_group_key
                    elif "HYDRA_FULL_ERROR" in combined_out:
                        reason = "config error"
                        skip_group = cur_group_key
                    else:
                        reason = f"exit code {result.returncode}"
                    log_path = out_path + ".error.log"
                    with open(log_path, "w") as f:
                        f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                        f.write(f"=== RETURN CODE ===\n{result.returncode}\n\n")
                        f.write(f"=== STDERR ===\n{stderr}\n\n")
                        f.write(f"=== STDOUT ===\n{stdout}\n")
                    if verbose:
                        print(f"    FAILED ({reason}) → {log_path}")
                    else:
                        print(f"FAILED ({reason}) → {log_path}")
                    failed_runs.append((run.tag, reason))
                    continue

                analysis: dict = {}
                mem_match = re.search(r"\[octibenchmark:memory\]\s*(\{.*\})", combined_out)
                if mem_match:
                    try:
                        analysis["memory"] = json.loads(mem_match.group(1))
                    except json.JSONDecodeError:
                        pass
                timing_match = re.search(r"\[octibenchmark:timing\]\s*(\{.*\})", combined_out)
                if timing_match:
                    try:
                        analysis["timing"] = json.loads(timing_match.group(1))
                    except json.JSONDecodeError:
                        pass
            else:
                # LIGHT / FULL: expect .nsys-rep
                nsys_rep = out_path + ".nsys-rep"

                if result.returncode != 0 or not os.path.exists(nsys_rep):
                    combined = (stderr + "\n" + stdout).strip()

                    is_oom = "OutOfMemoryError" in combined or "out of memory" in combined.lower()
                    is_killed = result.returncode in (-9, 137)
                    if is_oom or is_killed:
                        reason = "OOM" if is_oom else "killed (likely OOM)"
                        skip_group = cur_group_key
                    elif "HYDRA_FULL_ERROR" in combined:
                        reason = "config error"
                        skip_group = cur_group_key
                    else:
                        reason = f"exit code {result.returncode}"
                    log_path = out_path + ".error.log"
                    with open(log_path, "w") as f:
                        f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                        f.write(f"=== RETURN CODE ===\n{result.returncode}\n\n")
                        f.write(f"=== STDERR ===\n{stderr}\n\n")
                        f.write(f"=== STDOUT ===\n{stdout}\n")
                    if verbose:
                        print(f"    FAILED ({reason}) → {log_path}")
                    else:
                        print(f"FAILED ({reason}) → {log_path}")
                    failed_runs.append((run.tag, reason))

                    continue

                from octibenchmark.analyze import analyze

                analysis = analyze(nsys_rep, kernel_patterns)

                # Parse memory stats printed by benchmark.py
                mem_match = re.search(r"\[octibenchmark:memory\]\s*(\{.*\})", combined_out)
                if mem_match:
                    try:
                        analysis["memory"] = json.loads(mem_match.group(1))
                    except json.JSONDecodeError:
                        pass

            # Attach run metadata to the analysis for wandb grouping
            analysis["_meta"] = dataclasses.asdict(run)
            all_results[run.tag] = analysis
            print("OK")

            # Incrementally save results
            json_path = os.path.join(output_dir, "matrix_results.json")
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            # Accumulate results for the current group
            if use_wandb:
                overrides = run.hydra_overrides
                override_key = tuple(overrides.get(a, "") for a in sweep_axis_names)
                group_key = (run.task, override_key, run.launcher.value)
                pending_groups[group_key].append((run.num_envs, analysis, run.tag))

        if failed_runs:
            print(f"\n[BenchmarkMatrix] {len(failed_runs)} run(s) failed:")
            for failed_tag, reason in failed_runs:
                print(f"  {failed_tag}: {reason}")

        # Flush the last group (group transition only triggers for previous groups)
        if use_wandb:
            for group_key, pending in list(pending_groups.items()):
                if pending:
                    wandb_group = _flush_wandb_group(
                        group_key,
                        pending,
                        sweep_axis_names,
                        wandb_project,
                        wandb_entity,
                        wandb_group,
                        tag,
                        output_dir,
                        valid_runs,
                    )
                    wandb_run_count += 1
            if wandb_run_count:
                print(f"\n[BenchmarkMatrix] wandb group: {wandb_group}")
                print(f"[BenchmarkMatrix] {wandb_run_count} wandb runs created")

        json_path = os.path.join(output_dir, "matrix_results.json")
        if all_results:
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"[BenchmarkMatrix] Results saved to: {json_path}")
        return all_results


def _flush_wandb_group(
    group_key: tuple,
    env_results: list[tuple[int, dict, str]],
    sweep_axis_names: list[str],
    wandb_project: str,
    wandb_entity: str | None,
    wandb_group: str | None,
    tag: str | None,
    output_dir: str | None,
    all_runs: list,
) -> str:
    """Create one wandb run for a config group and log all its data points.

    Returns the wandb group name (computed on first call).
    """
    import wandb

    task, override_key, launcher = group_key

    if wandb_group is None:
        tasks = sorted({r.task for r in all_runs})
        short_tasks = "_".join(_shorten_task(t) for t in tasks)
        wandb_group = f"matrix_{short_tasks}" if len(tasks) <= 3 else f"matrix_{len(tasks)}_tasks"
        if tag:
            wandb_group = f"{wandb_group}_{tag}"

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

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=run_name,
        tags=[tag] if tag else None,
        config=run_config,
        reinit=True,
    )
    wandb.define_metric("num_envs", hidden=True)
    wandb.define_metric("*", step_metric="num_envs")

    for num_envs, analysis, _run_tag in sorted(env_results, key=lambda x: x[0]):
        _log_single_result(analysis, num_envs, output_dir, _run_tag)

    # Upload .nsys-rep profiles as artifact
    if output_dir:
        artifact = wandb.Artifact(
            name=re.sub(r"[^a-zA-Z0-9._-]", "_", f"nsys-profiles-{run_name}"),
            type="nsys-profile",
            description=f"Nsight Systems profiles for {run_name}",
        )
        files_added = 0
        for _num_envs, _analysis, run_tag in env_results:
            nsys_path = os.path.join(output_dir, f"{run_tag}.nsys-rep")
            if os.path.isfile(nsys_path):
                artifact.add_file(nsys_path, name=f"{run_tag}.nsys-rep")
                files_added += 1
        if files_added:
            wandb.log_artifact(artifact)

    wandb.finish()
    env_results.clear()
    return wandb_group


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
            s = s[len(prefix) :]
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


_TERM_PATTERN = re.compile(r"^(\w+)\.term(?:\[([^\]]+)\])?:(.+)$")


_TOP_RANGES = {
    "env.step",
    "sim.step",
    "action.process",
    "observation.compute",
    "reward.compute",
    "termination.compute",
    "event.compute",
    "scene.write_data_to_sim",
    "scene.update",
}


def _log_single_result(
    analysis: dict,
    num_envs: int,
    output_dir: str | None,
    run_tag: str,
):
    """Log one benchmark result to the currently active wandb run.

    Args:
        analysis: Analysis dict from :func:`octibenchmark.analyze.analyze`.
        num_envs: Number of environments for this run.
        output_dir: Directory containing ``.nsys-rep`` files.
        run_tag: Tag identifying this run (for artifact naming).
    """
    import wandb

    log_data: dict = {"num_envs": num_envs}
    gpu_timing = analysis.get("gpu_timing", {})
    has_gpu = bool(gpu_timing)

    # Line charts: log both GPU kernel time and CPU wall time for top ranges
    nvtx_by_name = {n["name"]: n for n in analysis.get("nvtx_ranges", [])}
    for name in _TOP_RANGES:
        nvtx_entry = nvtx_by_name.get(name)
        if nvtx_entry is None:
            continue
        log_data[f"wall/{name} (ms)"] = nvtx_entry["avg_ns"] / 1e6
        if has_gpu:
            gpu_entry = gpu_timing.get(name)
            if gpu_entry is not None:
                log_data[f"gpu/{name} (ms)"] = gpu_entry["gpu_ns"] / 1e6

    env_step = nvtx_by_name.get("env.step")
    if env_step and env_step["avg_ns"] > 0:
        log_data["effective_fps"] = num_envs / (env_step["avg_ns"] / 1e9)

    # Fall back to wall-clock timing (primary source in PLAIN mode)
    timing = analysis.get("timing")
    if timing:
        if "effective_fps" not in log_data and "effective_fps" in timing:
            log_data["effective_fps"] = timing["effective_fps"]
        if "step_ms" in timing:
            log_data["wall/step_ms"] = timing["step_ms"]

    mem = analysis.get("memory")
    if mem and "gpu_used_mb" in mem:
        log_data["gpu_memory_used_mb"] = mem["gpu_used_mb"]

    # Per-term breakdown: one bar chart for GPU time, one for CPU wall time
    gpu_bar_data = []
    wall_bar_data = []
    for nvtx in analysis.get("nvtx_ranges", []):
        m = _TERM_PATTERN.match(nvtx["name"])
        if not m:
            continue
        category, group, term = m.group(1).capitalize(), m.group(2), m.group(3)
        label = f"{category}/{group}/{term}" if group else f"{category}/{term}"
        wall_bar_data.append([label, nvtx["avg_ns"] / 1e6])
        if has_gpu:
            gpu_entry = gpu_timing.get(nvtx["name"])
            if gpu_entry is not None:
                gpu_bar_data.append([label, gpu_entry["gpu_ns"] / 1e6])

    if gpu_bar_data:
        gpu_bar_data.sort(key=lambda r: r[1], reverse=True)
        gpu_table = wandb.Table(data=gpu_bar_data, columns=["term", "gpu_ms"])
        log_data[f"gpu_breakdown/{num_envs}_envs"] = wandb.plot.bar(
            gpu_table,
            "term",
            "gpu_ms",
            title=f"Per-term GPU Kernel Time ({num_envs} envs)",
        )
    if wall_bar_data:
        wall_bar_data.sort(key=lambda r: r[1], reverse=True)
        wall_table = wandb.Table(data=wall_bar_data, columns=["term", "wall_ms"])
        log_data[f"wall_breakdown/{num_envs}_envs"] = wandb.plot.bar(
            wall_table,
            "term",
            "wall_ms",
            title=f"Per-term CPU Wall Time ({num_envs} envs)",
        )

    wandb.log(log_data)
