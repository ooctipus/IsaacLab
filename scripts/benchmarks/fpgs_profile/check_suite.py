# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check all 21 scenes at 16,384 worlds without a profiler or repeated setup."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]
PRESETS = {
    "smoke": {"worlds": 16384, "warmup": 8, "steps": 64, "blocks": 1},
    "perf": {"worlds": 16384, "warmup": 200, "steps": 64, "blocks": 3},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root):
    """Include tracked edits and untracked Python sources without modifying the checkout."""
    names = (
        subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        )
        .decode()
        .split("\0")
    )
    files = {
        name: digest(root / name) for name in sorted(set(names)) if name.endswith(".py") and (root / name).is_file()
    }
    return {
        "head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "python_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


class GraphTimer:
    """Time only the current physics graph on its actual launch stream."""

    def __init__(self, warp, manager, device, capacity):
        self.wp, self.manager, self.device = warp, manager, device
        self.events = [
            (warp.Event(device, enable_timing=True), warp.Event(device, enable_timing=True)) for _ in range(capacity)
        ]
        self.records = []
        self.step = 0

    def __enter__(self):
        self.original = self.wp.capture_launch

        def launch(graph, stream=None):
            if graph is not self.manager._graph:
                return self.original(graph, stream=stream)
            index = len(self.records)
            if index >= len(self.events):
                raise RuntimeError("More physics graph launches than the configured decimation allows")
            start, end = self.events[index]
            actual_stream = stream if stream is not None else self.wp.get_stream(self.device)
            actual_stream.record_event(start)
            result = self.original(graph, stream=stream)
            actual_stream.record_event(end)
            self.records.append((self.step, start, end))
            return result

        self.wp.capture_launch = launch
        return self

    def __exit__(self, *exc):
        self.wp.capture_launch = self.original

    def milliseconds(self, steps, decimation):
        self.wp.synchronize_device(self.device)
        values = [[] for _ in range(steps)]
        for step, start, end in self.records:
            elapsed = self.wp.get_event_elapsed_time(start, end, synchronize=False)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Invalid CUDA-event timing")
            values[step].append(elapsed)
        if any(len(row) != decimation for row in values):
            raise RuntimeError("Missing physics graph launches: refusing an incomplete physics timing")
        return [sum(row) for row in values]


def physics_launches_per_step(env):
    """Read the environment's selected physics stepping schedule."""
    return 1 if env.unwrapped._physics_handles_decimation else int(env.unwrapped.cfg.decimation)


def collision_work_snapshot(narrow):
    """Read retained reducer populations outside the physics timing region."""
    reducer = narrow.global_contact_reducer
    if reducer is None:
        return None
    raw = int(reducer.contact_count.numpy()[0])
    active = int(reducer.hashtable.active_slots[-1:].numpy()[0])
    if not 0 <= raw <= reducer.capacity or not 0 <= active <= reducer.hashtable.capacity:
        raise ValueError("Reducer work exceeds the recorded capacity")
    return {
        "raw_contacts": raw,
        "active_keys": active,
        "slots_per_key": int(reducer.values_per_key),
        "candidate_slots": active * int(reducer.values_per_key),
    }


def worker(job_path):
    """Run one isolated scene, retaining the task's physics settings and checking a reset."""
    started = time.perf_counter()
    job = json.loads(job_path.read_text())
    result = {"complete": False, "scene": job["scene"]["id"], "phases_seconds": {}}
    destination = job_path.with_name("result.json")
    try:
        import ast
        import contextlib

        # The harness disables Warp backward compilation before importing simulation packages.
        import run_profiled as harness  # isort: skip

        import newton
        import numpy as np
        import torch
        import warp as wp
        from isaaclab_newton.physics import NewtonFeatherPGSManager, NewtonManager
        from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg

        if not Path(newton.__file__).resolve().is_relative_to(Path(job["newton"])):
            raise RuntimeError("Wrong Newton checkout imported")
        if not Path(sys.modules["isaaclab"].__file__).resolve().is_relative_to(LAB):
            raise RuntimeError("Wrong Isaac Lab checkout imported")
        result["phases_seconds"]["imports"] = time.perf_counter() - started
        setup = time.perf_counter()
        scene, protocol = job["scene"], job["protocol"]
        if protocol["worlds"] != 16384:
            raise ValueError("The 21-scene check requires 16,384 worlds")
        torch.manual_seed(0)
        cfg = harness.parse_env_cfg(
            scene["task"], device="cuda:0", num_envs=protocol["worlds"], overrides=["physics=feather_pgs"]
        )
        cfg.seed = 0
        for key, value in scene["solver_attrs"].items():
            with contextlib.suppress(ValueError, SyntaxError):
                value = ast.literal_eval(value)
            setattr(cfg.sim.physics.solver_cfg, key, value)
        if scene["storage_at_16384_worlds"]:
            if cfg.sim.physics.collision_cfg is None:
                cfg.sim.physics.collision_cfg = NewtonCollisionPipelineCfg()
            for key, value in scene["storage_at_16384_worlds"].items():
                setattr(cfg.sim.physics.collision_cfg, key, value)
        cfg.validate()
        launch_args = {"device": "cuda:0", "headless": True, "visualizer": None, "visualizer_explicit": True}

        def check(label, env):
            state = NewtonManager.get_state_0()
            arrays = {name: getattr(state, name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError(f"Nonfinite state at {label}")
            solver = NewtonManager._solver
            narrow = NewtonManager._collision_pipeline.narrow_phase
            if not narrow.verify_buffers:
                raise RuntimeError("Collision buffer checking is disabled")
            solver.check_constraint_capacity()
            narrow.check_buffer_capacity()
            flags = {"solver": solver.constraint_capacity_status(), "collision": narrow.buffer_capacity_status()}
            if any(any(group.values()) for group in flags.values()):
                raise RuntimeError(f"Capacity overflow at {label}: {flags}")
            if NewtonManager._graph is None:
                raise RuntimeError("Missing captured physics graph")
            return {
                "label": label,
                "finite": True,
                "capacity_flags": flags,
                "collision_work": collision_work_snapshot(narrow),
                "selection": NewtonFeatherPGSManager.get_path_selection(),
                "state_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()},
                "body_velocity_rms": float(np.sqrt(np.mean(arrays["body_qd"].astype(float) ** 2))),
                "sim_dt": float(env.unwrapped.cfg.sim.dt),
                "decimation": int(env.unwrapped.cfg.decimation),
                "solver_dt": float(NewtonManager._solver_dt),
                "substeps": int(NewtonManager._num_substeps),
                "pgs_iterations": int(solver.pgs_iterations),
            }

        with harness.launch_simulation(cfg, launch_args):
            with contextlib.closing(harness.gym.make(scene["task"], cfg=cfg)) as env, torch.inference_mode():
                env.reset()
                model = NewtonManager.get_model()
                result["model"] = {
                    name: int(getattr(model, name))
                    for name in ("world_count", "body_count", "joint_count", "joint_dof_count", "shape_count")
                }
                if model.world_count != protocol["worlds"]:
                    raise RuntimeError("Scene did not honor the requested world count")
                result["phases_seconds"]["setup"] = time.perf_counter() - setup
                result["checks"] = [check("reset", env)]
                phase = time.perf_counter()
                for _ in range(protocol["warmup"]):
                    env.step(harness.sample_random_actions(env))
                wp.synchronize_device("cuda:0")
                result["phases_seconds"]["warmup"] = time.perf_counter() - phase
                result["checks"].append(check("warm", env))
                phase = time.perf_counter()
                steps = protocol["steps"] * protocol["blocks"]
                launches_per_step = physics_launches_per_step(env)
                action_hash = hashlib.sha256()
                timer = (
                    GraphTimer(wp, NewtonManager, "cuda:0", steps * launches_per_step)
                    if job["mode"] == "perf"
                    else None
                )
                with timer if timer else contextlib.nullcontext():
                    for step in range(steps):
                        if timer:
                            timer.step = step
                        actions = harness.sample_random_actions(env)
                        # Copying each action to CPU would synchronize every environment step.
                        env.step(actions)
                    action_hash.update(actions.cpu().numpy().tobytes())
                wp.synchronize_device("cuda:0")
                result["phases_seconds"]["run"] = time.perf_counter() - phase
                if timer:
                    timings = timer.milliseconds(steps, launches_per_step)
                    blocks = [
                        statistics.mean(timings[i : i + protocol["steps"]]) for i in range(0, steps, protocol["steps"])
                    ]
                    result["timing"] = {
                        "scope": "CUDA events around physics graphs, summed per environment step",
                        "mean_ms": statistics.mean(timings),
                        "blocks_ms": blocks,
                        "per_step_ms": timings,
                        "graph_launches_per_step": launches_per_step,
                        "last_action_sha256": action_hash.hexdigest(),
                    }
                result["checks"].append(check("after_run", env))
                phase = time.perf_counter()
                env.reset()
                for _ in range(8):
                    env.step(harness.sample_random_actions(env))
                result["checks"].append(check("after_reset", env))
                result["phases_seconds"]["reset_check"] = time.perf_counter() - phase
        result["complete"] = True
    except Exception as exc:
        import traceback

        result["error"] = repr(exc)
        traceback.print_exc()
    finally:
        result["total_seconds"] = time.perf_counter() - started
        write_json(destination, result)
    return 0 if result["complete"] else 1


def comparison(baseline, current):
    """Historical event measurements are a screen, never a paired speedup claim."""
    if baseline["comparison_key"] != current["comparison_key"]:
        raise ValueError("Baseline has different hardware, recipes, Lab sources or timing protocol")
    old = {row["scene"]: row for row in baseline["results"] if row.get("complete")}
    rows = []
    for row in current["results"]:
        if not row.get("complete") or row["scene"] not in old:
            rows.append({"scene": row["scene"], "status": "missing or failed"})
            continue
        before, after = old[row["scene"]]["timing"], row["timing"]
        reference = old[row["scene"]]
        initial_keys = ("state_sha256", "sim_dt", "decimation", "solver_dt", "substeps", "pgs_iterations")
        initial_changed = any(reference["checks"][0][key] != row["checks"][0][key] for key in initial_keys)
        if reference["model"] != row["model"] or initial_changed:
            rows.append({"scene": row["scene"], "status": "investigate: initial state or configuration changed"})
            continue
        change = 100 * (after["mean_ms"] / before["mean_ms"] - 1)
        spread = max((max(t["blocks_ms"]) - min(t["blocks_ms"])) / t["mean_ms"] * 100 for t in (before, after))
        rows.append(
            {
                "scene": row["scene"],
                "change_percent": change,
                "spread_percent": spread,
                "status": "investigate" if abs(change) >= 3 or spread >= 3 else "within screen tolerance",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=PRESETS, default="perf", help="Both modes retain 16,384 worlds; default: perf"
    )
    parser.add_argument("--task", action="append", help="Scene ID; omit to check all 21")
    parser.add_argument("--gpu", default="0", help="GPU index or UUID")
    parser.add_argument("--baseline", type=Path, help="Prior perf summary.json with the identical protocol")
    parser.add_argument("--timeout", type=int, default=900, help="Per-scene timeout in seconds (default: 900)")
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/fpgs-suite")
    args = parser.parse_args()
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.baseline and args.mode != "perf":
        parser.error("--baseline requires --mode perf")
    args.newton = args.newton.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    scenes = json.loads((HERE / "suite21.json").read_text())["scenes"]
    if args.task:
        if set(args.task) - {s["id"] for s in scenes}:
            parser.error("Unknown scene ID")
        scenes = [s for s in scenes if s["id"] in args.task]
    gpu = subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    uuid = gpu.split(",")[0]
    identity = {"newton": source_identity(args.newton), "lab": source_identity(LAB)}
    cache = args.cache_root / identity["newton"]["python_sha256"][:16]
    cache.mkdir(parents=True, exist_ok=True)
    key = {
        "hardware": gpu,
        "protocol": PRESETS[args.mode],
        "mode": args.mode,
        "lab": identity["lab"]["python_sha256"],
        "recipes": digest(HERE / "suite21.json"),
        "runner": digest(Path(__file__)),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("warp-lang", "torch", "numpy")},
    }
    summary = {
        "complete": False,
        "mode": args.mode,
        "sources": identity,
        "comparison_key": key,
        "scope": "16,384-world functional screen without timing"
        if args.mode == "smoke"
        else "Full-world CUDA-event performance screen; confirm changes with fresh paired measurements",
        "results": [],
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline and (not baseline["complete"] or baseline["comparison_key"] != key):
        raise ValueError("Incompatible or unfinished baseline; refusing to spend GPU time")
    started = time.perf_counter()
    for scene in scenes:
        running = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
        )
        if uuid in running:
            raise RuntimeError("Selected GPU is occupied; refusing to contaminate measurements")
        folder = args.output / scene["id"]
        folder.mkdir()
        job = {"scene": scene, "protocol": PRESETS[args.mode], "mode": args.mode, "newton": str(args.newton)}
        write_json(folder / "job.json", job)
        environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("FEATHER_", "NEWTON_", "FPGS_"))
            and k not in ("PYTHONPATH", "WARP_CACHE_PATH", "CUDA_CACHE_PATH")
        }
        environment.update(scene["environment"])
        environment.update(
            CUDA_VISIBLE_DEVICES=uuid,
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            WARP_CACHE_PATH=str(cache / "warp"),
            CUDA_CACHE_PATH=str(cache / "cuda"),
            PYTHONPATH=os.pathsep.join(
                [
                    str(args.newton),
                    *[str(p) for p in sorted((LAB / "source").iterdir()) if p.is_dir() and (p / p.name).is_dir()],
                ]
            ),
        )
        scene_started = time.perf_counter()
        with (folder / "run.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "--worker", str(folder / "job.json")],
                env=environment,
                cwd=LAB,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        result_path = folder / "result.json"
        result = (
            json.loads(result_path.read_text()) if result_path.exists() else {"scene": scene["id"], "complete": False}
        )
        result["returncode"] = code
        result["wall_seconds"] = time.perf_counter() - scene_started
        if code == 124:
            result["error"] = "Per-scene timeout exceeded"
        result["complete"] = result["complete"] and code == 0
        summary["results"].append(result)
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", summary)
        print(f"{scene['id']}: {'PASS' if result['complete'] else 'FAIL'} ({result['wall_seconds']:.1f}s)", flush=True)
    if identity != {"newton": source_identity(args.newton), "lab": source_identity(LAB)}:
        raise RuntimeError("Sources changed during the suite")
    summary["complete"] = all(row["complete"] for row in summary["results"])
    if baseline:
        summary["comparison"] = comparison(baseline, summary)
    write_json(args.output / "summary.json", summary)
    print(
        f"{sum(r['complete'] for r in summary['results'])}/{len(scenes)} passed "
        f"in {summary['elapsed_seconds'] / 60:.1f} minutes"
    )
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        raise SystemExit(worker(Path(sys.argv[2])))
    raise SystemExit(main())
