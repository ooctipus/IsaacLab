# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare Newton revisions concurrently on separate GPUs using the handoff recipes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HARNESS = Path(__file__).resolve().parent
RECIPES = {
    "anymald": (
        "Isaac-Velocity-Flat-AnymalD",
        48,
        {"FEATHER_PGS_MF_EXACT_ROWSUM": "1", "FEATHER_PGS_WORLD_ROWS": "1"},
    ),
    "allegro": ("Isaac-Reorient-Cube-Allegro", 128, {"FEATHER_PGS_TIER_BLOCKS": "16384"}),
}
COMMON_ENVIRONMENT = {
    "FEATHER_PGS_GROUP_LANES": "16",
    "FEATHER_PGS_ROWS_MASKED": "1",
    "FEATHER_PGS_INK": "1",
    "NEWTON_NARROW_PHASE_THREADS_X": "4",
    "UV_NO_SYNC": "1",
    "PYTHONUNBUFFERED": "1",
    "REPEATS": "1",
}


def _git(path: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(path), *args])


def _source(path: Path) -> dict:
    """Record a commit and digest of staged, unstaged, and untracked changes."""
    root = Path(_git(path, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if root != path:
        raise ValueError(f"Expected checkout root {root}, got {path}")
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
    digest = hashlib.sha256(_git(path, "diff", "--binary", "HEAD"))
    for name in sorted(_git(path, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if name:
            file = path / os.fsdecode(name)
            content = os.fsencode(os.readlink(file)) if file.is_symlink() else file.read_bytes()
            digest.update(name + b"\0" + hashlib.sha256(content).digest())
    return {
        "path": str(path),
        "sha": _git(path, "rev-parse", "HEAD").decode().strip(),
        "dirty": bool(status),
        "status": status.decode(),
        "source_diff_sha256": digest.hexdigest(),
    }


def _gpus(indices: list[int]) -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={','.join(map(str, indices))}",
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    found = {}
    for index, uuid, name, driver, pci_bus_id in csv.reader(output.splitlines(), skipinitialspace=True):
        found[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "driver": driver,
            "pci_bus_id": pci_bus_id,
        }
    return [found[index] for index in indices]


def _require_idle(gpus: list[dict]) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={','.join(gpu['uuid'] for gpu in gpus)}",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if output:
        raise RuntimeError(f"Selected GPUs have active compute processes:\n{output}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _software() -> dict:
    versions = {}
    for name in ("warp-lang", "torch", "newton"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "nsys": subprocess.check_output(["nsys", "--version"], text=True).strip(),
        "installed_distributions": versions,
    }


def _run_batch(runs: list[dict]) -> None:
    """Start one profiler per GPU, then finish the whole batch before returning."""
    processes = []
    with ExitStack() as stack:
        try:
            for run in runs:
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith(("FEATHER_PGS_", "NEWTON_NARROW_PHASE_"))
                }
                env.update(run["environment"])
                log = stack.enter_context(Path(run["driver_log"]).open("x"))
                process = subprocess.Popen(
                    run["command"],
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append((run, process))
            for run, process in processes:
                run["returncode"] = process.wait()
        finally:
            for _, process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    failures = [run for run in runs if run.get("returncode") != 0]
    if failures:
        logs = [f"{run['driver_log']} and {Path(run['output_dir']) / 'capture.log'}" for run in failures]
        raise RuntimeError("Profiler failed; inspect " + ", ".join(logs))


def _read_result(run: dict) -> dict:
    directory = Path(run["output_dir"])
    analysis = json.loads((directory / "capture_analysis.json").read_text())
    capture = json.loads((directory / "capture.json").read_text())
    graph_us = float(analysis["summary"]["graph_span_us_per_step"])
    if not math.isfinite(graph_us) or graph_us <= 0:
        raise RuntimeError(f"Invalid graph timing in {directory}")
    before, after = capture["model"], capture["model_after"]
    if not before["state_finite"] or not after["state_finite"]:
        raise RuntimeError(f"Nonfinite simulation state in {directory}")
    return {
        "graph_span_us_per_step": graph_us,
        "state_finite": True,
        "contacts_before": before["contacts_active"],
        "contacts_after": after["contacts_active"],
        "gjk_items_before": before.get("gjk_items"),
        "gjk_items_after": after.get("gjk_items"),
    }


def _summaries(runs: list[dict]) -> list[dict]:
    groups = {}
    for run in runs:
        if "result" in run:
            key = (run["gpu_index"], run["task"], run["newton"])
            groups.setdefault(key, []).append(run["result"])
    rows = []
    for (gpu, task, revision), results in groups.items():
        times = [result["graph_span_us_per_step"] for result in results]
        median = statistics.median(times)
        rows.append(
            {
                "gpu_index": gpu,
                "task": task,
                "newton": revision,
                "repeats": len(times),
                "graph_span_us_per_step": times,
                "median_us": median,
                "min_us": min(times),
                "max_us": max(times),
                "spread_percent": 100.0 * (max(times) - min(times)) / median,
                "state_finite": all(result["state_finite"] for result in results),
                "contacts_before": [result["contacts_before"] for result in results],
                "contacts_after": [result["contacts_after"] for result in results],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newton", action="append", required=True, metavar="LABEL=CHECKOUT")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--task", choices=RECIPES, action="append")
    parser.add_argument("--repeats", type=int, default=3, help="A/B rounds; reverse revision order every other round.")
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, help="New directory; must not already exist.")
    args = parser.parse_args()
    if any(getattr(args, name) <= 0 for name in ("repeats", "num_envs", "steps", "profile_steps")):
        parser.error("repeats, num-envs, steps, and profile-steps must be positive")
    if args.warmup_steps < 0 or any(gpu < 0 for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error("warmup-steps must be nonnegative and GPU indices must be distinct and nonnegative")
    revisions = {}
    for entry in args.newton:
        label, separator, checkout = entry.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label) or label in revisions:
            parser.error(f"Expected a unique LABEL=CHECKOUT, got {entry!r}")
        path = Path(checkout).expanduser().resolve()
        if not (path / "newton" / "__init__.py").is_file():
            parser.error(f"Missing newton/__init__.py in {path}")
        revisions[label] = _source(path)
    tasks = list(dict.fromkeys(args.task or RECIPES))
    for executable in ("uv", "nsys", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is missing: {executable}")
    gpus = _gpus(args.gpus)
    _require_idle(gpus)
    if args.output_dir:
        output = args.output_dir.expanduser().resolve()
        for checkout in [REPO, *(Path(revision["path"]) for revision in revisions.values())]:
            if output.is_relative_to(checkout):
                ignored = subprocess.run(
                    ["git", "-C", str(checkout), "check-ignore", "--quiet", "--", str(output / "manifest.json")]
                )
                if ignored.returncode != 0:
                    parser.error(f"Output inside {checkout} must be git-ignored so it cannot alter the source digest")
        output.mkdir(parents=True, exist_ok=False)
    else:
        parent = REPO / "outputs" / "fpgs_profile"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="compare_gpus_", dir=parent))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "software": _software(),
        "isaaclab": _source(REPO),
        "newton": revisions,
        "gpus": gpus,
        "settings": {key: value for key, value in vars(args).items() if key not in ("output_dir", "newton")},
        "output_dir": str(output),
        "runs": [],
        "status": "running",
    }
    print(f"Results: {output}", flush=True)
    try:
        for repeat in range(args.repeats):
            labels = list(revisions) if repeat % 2 == 0 else list(reversed(revisions))
            for task in tasks:
                task_name, rows, recipe_env = RECIPES[task]
                for label in labels:
                    _require_idle(gpus)
                    if (
                        _source(REPO) != manifest["isaaclab"]
                        or _source(Path(revisions[label]["path"])) != revisions[label]
                    ):
                        raise RuntimeError("Checkout changed during comparison; start again with frozen sources")
                    batch = []
                    for gpu in gpus:
                        directory = output / f"round_{repeat + 1:02d}_{task}_{label}_gpu{gpu['index']}"
                        directory.mkdir()
                        env = {
                            **COMMON_ENVIRONMENT,
                            **recipe_env,
                            "GPU": gpu["uuid"],
                            "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                            "PYTHONPATH": revisions[label]["path"],
                            "OUT_DIR": str(directory),
                            "STEPS": str(args.steps),
                            "PROFILE_STEPS": str(args.profile_steps),
                        }
                        command = [
                            "bash",
                            str(HARNESS / "nsys_run.sh"),
                            "capture",
                            "feather_pgs",
                            task_name,
                            "--num-envs",
                            str(args.num_envs),
                            "--seed",
                            str(args.seed),
                            "--warmup-steps",
                            str(args.warmup_steps),
                        ]
                        for attribute in (
                            "grouped_dynamics=True",
                            f"mf_gs_parallel_rows={rows}",
                            "mf_gs_parallel_matrix_free=True",
                            "lazy_kinematics=True",
                        ):
                            command.extend(["--solver-attr", attribute])
                        batch.append(
                            {
                                "round": repeat + 1,
                                "task": task,
                                "newton": label,
                                "gpu_index": gpu["index"],
                                "gpu_uuid": gpu["uuid"],
                                "command": command,
                                "environment": env,
                                "output_dir": str(directory),
                                "driver_log": str(directory / "driver.log"),
                            }
                        )
                    manifest["runs"].extend(batch)
                    _write_json(output / "manifest.json", manifest)
                    print(f"Round {repeat + 1}: {task} / {label} on GPUs {args.gpus}", flush=True)
                    _run_batch(batch)
                    for run in batch:
                        run["result"] = _read_result(run)
                    if (
                        _source(REPO) != manifest["isaaclab"]
                        or _source(Path(revisions[label]["path"])) != revisions[label]
                    ):
                        raise RuntimeError("Checkout changed during capture; these samples are invalid")
                    _write_json(output / "manifest.json", manifest)
                    _write_json(output / "summary.json", _summaries(manifest["runs"]))
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        _write_json(output / "manifest.json", manifest)
        _write_json(output / "summary.json", _summaries(manifest["runs"]))
    for row in _summaries(manifest["runs"]):
        print(
            f"GPU {row['gpu_index']} {row['task']} {row['newton']}: {row['median_us']:.1f} us/step "
            f"(range {row['min_us']:.1f}–{row['max_us']:.1f}, spread {row['spread_percent']:.2f}%, "
            f"finite={row['state_finite']}, contacts={row['contacts_after']})"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        print("Comparison interrupted; running profiler processes were stopped.", file=sys.stderr)
        raise SystemExit(130) from None
