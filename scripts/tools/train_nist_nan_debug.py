# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the current NIST task and upload a bounded Newton NaN incident artifact."""

import json
import os
import signal
import sys
import threading
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab_newton.physics import NewtonCfg, NewtonDebugCaptureCfg  # noqa: E402
from isaaclab_rl.entrypoints.backends import train_rsl_rl  # noqa: E402

TASK = "IsaacContrib-Factory-Variant-Franka"
WORKFLOW_ID = os.environ.get("ISAACLAB_WORKFLOW_ID", "local")
OUTPUT_DIR = Path("logs") / "nist_nan_debug" / WORKFLOW_ID
MAX_INCIDENTS = 5


def resolve_debug_task(task: str, agent: str):
    if task != TASK:
        raise ValueError(f"This diagnostic entrypoint only supports {TASK}.")
    env_cfg, agent_cfg = original_resolve_task_config(task, agent)
    physics = env_cfg.sim.physics
    if not isinstance(physics, NewtonCfg):
        raise TypeError("NIST NaN capture requires the Newton physics preset.")
    physics.debug_capture = NewtonDebugCaptureCfg(
        history_length=16,
        output_dir=str(OUTPUT_DIR),
        max_incidents=MAX_INCIDENTS,
        halt_on_incident=False,
        failed_worlds_only=True,
        record_scene=False,
        record_model=True,
        record_control=True,
        record_solver=True,
        include_private_fields=False,
        detect_nonfinite_in=("state",),
    )
    return env_cfg, agent_cfg


def upload_artifact(error: str | None) -> None:
    import wandb

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUT_DIR.glob("physics_incident_*.npz"))[:MAX_INCIDENTS]
    summary = OUTPUT_DIR / "summary.json"
    summary.write_text(json.dumps({"workflow_id": WORKFLOW_ID, "incident_files": [p.name for p in files], "error": error}, indent=2))
    if wandb.run is None:
        wandb.init(project="factory_manager12", entity="nvidia", name=f"nist-nan-debug-{WORKFLOW_ID}", job_type="physics-debug")
    artifact = wandb.Artifact(f"nist-nan-debug-{WORKFLOW_ID}", type="physics-debug")
    artifact.add_file(str(summary))
    for path in files:
        artifact.add_file(str(path))
    wandb.run.log_artifact(artifact).wait()
    print(f"Uploaded {len(files)} Newton incident files to W&B artifact {artifact.name}", flush=True)
    wandb.finish()


def main() -> int:
    global original_resolve_task_config
    original_resolve_task_config = train_rsl_rl.resolve_task_config
    train_rsl_rl.resolve_task_config = resolve_debug_task
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done = threading.Event()

    def stop_when_done() -> None:
        while not done.wait(10):
            if len(list(OUTPUT_DIR.glob("physics_incident_*.npz"))) >= MAX_INCIDENTS:
                os.kill(os.getpid(), signal.SIGINT)
                return

    watcher = threading.Thread(target=stop_when_done, daemon=True)
    watcher.start()
    error = None
    try:
        train_rsl_rl.run(sys.argv[1:])
    except KeyboardInterrupt:
        print("Stopped NIST diagnostic training after capturing five incidents.", flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        done.set()
        upload_artifact(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
