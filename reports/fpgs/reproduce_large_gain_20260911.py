# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify or replay the private, rejected FPGS certificate experiment on both GPUs.

This is an evidence launcher, not an Isaac Lab runtime or benchmark modification.
It requires the retained host layout and pinned environment; it installs nothing.
Component timings are not whole-physics gains or accuracy acceptance.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

ARCHIVE = Path("/home/octi/Projects/fpgs-large-gain-evidence-20260911-ri5D08/closed-core")
MANIFEST_SHA = "91b244a974bf7803b9f9c83847ba53ae9601587d73cfdcaaa837c183f5e0bfa5"
LAB = Path("/home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910")
NEWTON = Path("/home/octi/Projects/newton-fpgs-certified-simple-20260911")
UUIDS = (
    "GPU-883586b6-3100-0610-81e5-3b4c26f45639",
    "GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4",
)
LOADED_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCRIPTS = {
    "edges": Path("/tmp/fpgs-fp32-proof-PqrfRI/check_carrier_edges_strict.py"),
    "native512": Path("/tmp/fpgs-fp32-strict-certificate-L5WQkL/check_native_strict.py"),
    "cost16k": Path("/tmp/fpgs-fp32-strict-certificate-L5WQkL/check_cost_strict.py"),
}


def digest(path: Path) -> str:
    """Hash a file without loading a large reference capture into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_sources() -> int:
    """Verify retained sources and capture/report manifests against the archive."""
    manifest = ARCHIVE / "MANIFEST.json"
    if digest(manifest) != MANIFEST_SHA:
        raise RuntimeError("The closed evidence manifest changed")
    value = json.loads(manifest.read_text())
    if not value["complete"]:
        raise RuntimeError("The archive is incomplete")
    checked = 0
    for record in value["records"]:
        source = Path(record["source"])
        if source.suffix in (".py", ".json"):
            if digest(source) != record["sha256"]:
                raise RuntimeError(f"Retained source changed: {source}")
            checked += 1
    head = subprocess.check_output(["git", "-C", str(LAB), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    dirty = subprocess.check_output(["git", "-C", str(LAB), "status", "--porcelain=v1"], text=True, timeout=30)
    if head != "1d8feb82d17dbfab8f0772de56f84deae2cb7974" or dirty:
        raise RuntimeError("Original Isaac Lab runtime must retain its clean exact pin")
    return checked


def idle_devices() -> None:
    """Require the recorded hardware mapping and no competing compute process."""
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=30
    )
    devices = {int(index): uuid.strip() for index, uuid in (line.split(",") for line in value.splitlines())}
    if tuple(devices.get(index) for index in range(2)) != UUIDS:
        raise RuntimeError("GPU indices no longer match the recorded RTX/GB UUIDs")
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True, timeout=30
    )
    if busy.strip():
        raise RuntimeError("The GPU compute-process list is not empty")


def make_run(mode: str, gpu: int, output: Path) -> dict:
    """Build an isolated pinned command without starting a process."""
    removed = {
        "PYTHONPATH",
        "PYTHONHOME",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "UV_WORKING_DIR",
        "UV_WORKING_DIRECTORY",
        "UV_NO_PROJECT",
        "UV_ISOLATED",
        "UV_ACTIVE",
        "UV_PYTHON",
        "VIRTUAL_ENV",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_DEVICE_MAX_CONNECTIONS",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FEATHER_PGS_", "NEWTON_")) and key not in removed
    }
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(NEWTON),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        UV_NO_SYNC="1",
    )
    result_path = output / f"gpu{gpu}.json"
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        str(LAB / ".venv/bin/python"),
        "python",
        str(SCRIPTS[mode]),
        "--gpu",
        str(gpu),
        "--output",
        str(result_path),
    ]
    if mode == "native512":
        command += ["--worlds", "512", "--fixture-dir", f"/tmp/fpgs-kuka-live-512-20260911-02/gpu{gpu}"]
    return {
        "gpu": gpu,
        "uuid": UUIDS[gpu],
        "complete": False,
        "command": command,
        "result": str(result_path),
        "log": str(output / f"gpu{gpu}.log"),
        "environment": environment,
    }


# These three ownership/cleanup primitives are unchanged from the reviewed
# ce870340174b9b843a44226cc81d0e7dda76520a5e4cc2694327260c22a1849d
# compare_backends.py. The private tests compare their ASTs against that source.
def signal_group(pgid: int, signum: int) -> bool:
    """Signal only a recorded owned group; treat absence, not denial, as completion."""
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def wait_group_gone(pgid: int, timeout: float) -> bool:
    """Wait a bounded grace period for all group members, not just their leader."""
    deadline = time.monotonic() + timeout
    while signal_group(pgid, 0):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def stop_children(processes: list[tuple[dict, subprocess.Popen]], *, grace_seconds: float = 10.0) -> None:
    """Terminate owned groups and reap direct children, including exited leaders."""
    failures = []
    for run, process in processes:
        try:
            # start_new_session makes this exact recorded PID the owned PGID.
            # An exited leader does not prove that profiler descendants exited.
            deadline = time.monotonic() + grace_seconds
            signaled = signal_group(process.pid, signal.SIGTERM)
            run["cleanup_signals"] = ["SIGTERM"] if signaled else []
            timed_out = False
            if process.poll() is None:
                try:
                    process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
            if timed_out or (signaled and not wait_group_gone(process.pid, max(0.0, deadline - time.monotonic()))):
                if signal_group(process.pid, signal.SIGKILL):
                    run["cleanup_signals"].append("SIGKILL")
            if process.poll() is None:
                process.wait(timeout=grace_seconds)
            run["returncode"] = process.returncode
        except BaseException as error:
            failures.append(f"PID {process.pid}: {error!r}")
    if failures:
        raise RuntimeError("Could not finish child cleanup: " + "; ".join(failures))


def record_error(errors: list[dict], stage: str, error: BaseException) -> None:
    """Preserve ordered first-failure evidence without hiding later cleanup errors."""
    errors.append({"stage": stage, "type": type(error).__name__, "error": str(error)})


def run_pair(mode: str, output: Path, *, timeout_seconds: float, results: list[dict], errors: list[dict]) -> None:
    """Launch both owned groups before waits; bound, stop and reap every child."""
    processes = []
    deadline = time.monotonic() + timeout_seconds
    with ExitStack() as logs:
        try:
            for gpu in (0, 1):
                run = make_run(mode, gpu, output)
                results.append(run)
                log = logs.enter_context(Path(run["log"]).open("x"))
                process = subprocess.Popen(
                    run["command"],
                    env=run.pop("environment"),
                    cwd=LAB,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append((run, process))
                run.update(pid=process.pid, pgid=process.pid)
                print(json.dumps({"started_gpu": gpu, "pid": process.pid, "pgid": process.pid}), flush=True)
            for run, process in processes:
                run["returncode"] = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if run["returncode"] != 0:
                    record_error(errors, f"gpu{run['gpu']}_exit", RuntimeError(f"Child exited {run['returncode']}"))
        except BaseException as error:
            record_error(errors, "launch_or_wait", error)
        finally:
            try:
                stop_children(processes)
            except BaseException as error:
                record_error(errors, "cleanup", error)
            for run, process in processes:
                run["reaped"] = process.poll() is not None
                try:
                    run["owned_group_gone"] = wait_group_gone(process.pid, 10.0)
                except BaseException as error:
                    run["owned_group_gone"] = False
                    record_error(errors, "group_postcheck", error)
                if not run["reaped"] or not run["owned_group_gone"]:
                    record_error(errors, "group_postcheck", RuntimeError(f"Owned PID/PGID {process.pid} remains"))
                # Environment may contain unrelated private values; do not persist it.
                run.pop("environment", None)
    for run in results:
        run.pop("environment", None)
        try:
            report = json.loads(Path(run["result"]).read_text())
            run["complete"] = (
                report.get("complete") is True
                and report.get("source_guard_pass") is True
                and report.get("hardware", {}).get("uuid") == run["uuid"]
            )
            if not run["complete"]:
                raise RuntimeError(f"Child gate/source/UUID failed: {report.get('error', 'inspect child report')}")
            if report.get("hardware", {}).get("warp") != "1.17.0":
                raise RuntimeError("Pinned Warp runtime version changed")
            runtime = report.get("hardware", {}).get("runtime")
            if runtime is not None and Path(runtime).resolve() != (LAB / ".venv").resolve():
                raise RuntimeError("Child used a different Python environment")
            run["result_sha256"] = digest(Path(run["result"]))
        except BaseException as error:
            run["complete"] = False
            record_error(errors, f"gpu{run['gpu']}_report", error)


def interrupt_run(signum: int, _frame: object) -> None:
    """Convert termination into the same owned-group cleanup path as Ctrl-C."""
    raise KeyboardInterrupt(f"Received signal {signum}")


def main() -> None:
    """Verify by default; run a paired component gate only when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(SCRIPTS))
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite a prior run")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Maximum paired child runtime in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if digest(Path(__file__)) != LOADED_SOURCE_SHA:
        raise RuntimeError("The loaded reproducer source changed")
    checked = verify_sources()
    if args.run is None:
        print(json.dumps({"sources_verified": checked, "gpu_work_started": False, "manifest_sha256": MANIFEST_SHA}))
        return
    if args.output is None:
        parser.error("--output is required with --run")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "mode": args.run,
        "sources_verified_before": checked,
        "manifest_sha256": MANIFEST_SHA,
        "launcher_sha256": LOADED_SOURCE_SHA,
        "timeout_seconds": args.timeout,
        "results": [],
        "errors": [],
        "complete": False,
        "final_idle_pass": False,
        "whole_physics_speedup_claimed": False,
        "physical_quality_accepted": False,
    }
    previous_term = signal.signal(signal.SIGTERM, interrupt_run)
    try:
        idle_devices()
        summary["initial_idle_pass"] = True
        run_pair(
            args.run, args.output, timeout_seconds=args.timeout, results=summary["results"], errors=summary["errors"]
        )
    except BaseException as error:
        record_error(summary["errors"], "preflight_or_run", error)
    finally:
        try:
            summary["sources_verified_after"] = verify_sources()
            if digest(Path(__file__)) != LOADED_SOURCE_SHA:
                raise RuntimeError("The reproducer changed while running")
            summary["final_source_guard_pass"] = True
        except BaseException as error:
            summary["final_source_guard_pass"] = False
            record_error(summary["errors"], "source_after", error)
        try:
            idle_devices()
            summary["final_idle_pass"] = True
        except BaseException as error:
            record_error(summary["errors"], "idle_after", error)
        signal.signal(signal.SIGTERM, previous_term)
        summary["complete"] = (
            not summary["errors"]
            and len(summary["results"]) == 2
            and all(
                result.get("returncode") == 0
                and result.get("complete")
                and result.get("reaped")
                and result.get("owned_group_gone")
                for result in summary["results"]
            )
        )
        with (args.output / "paired.json").open("x") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
    print(json.dumps(summary))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
