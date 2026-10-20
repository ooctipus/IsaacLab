# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the full ANYmal-C dataset and train NeRD in one two-GPU OSMO workflow.

Submit this script with ``./submit.sh`` and ``num_gpu=2``. Torchrun starts one copy per GPU; each copy records
four independent datasets, waits for all eight, then enters the existing data-parallel trainer. Completed
recordings and training checkpoints remain in the shared output directory so a restarted workflow can resume.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

HERE = Path(__file__).resolve().parent


def record_batch(seeds: list[int], output: Path, *, device: str, envs: int, steps: int) -> None:
    """Record several independent shards concurrently on one allocated GPU."""
    running: list[tuple[int, subprocess.Popen, TextIO]] = []
    try:
        for seed in seeds:
            dataset = output / "data" / f"seed-{seed}"
            complete = output / "data" / f"seed-{seed}.complete"
            if complete.is_file():
                print(f"Skipping completed recording seed {seed}", flush=True)
                continue
            log_path = output / "logs" / f"record-seed-{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
            command = [
                sys.executable,
                str(HERE / "record.py"),
                "--output",
                str(dataset),
                "--num-envs",
                str(envs),
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--device",
                device,
                "--shard-episodes",
                "4000",
                "--replace",
            ]
            running.append((seed, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"Recording seed {seed} on {device}; log: {log_path}", flush=True)
        while running:
            for seed, process, log in tuple(running):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                running.remove((seed, process, log))
                if code:
                    raise RuntimeError(f"Recording seed {seed} failed with exit code {code}.")
                (output / "data" / f"seed-{seed}.complete").write_text("ok\n")
                print(f"Finished recording seed {seed}", flush=True)
            if running:
                time.sleep(2)
    finally:
        for _, process, log in running:
            process.terminate()
            process.wait()
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true", help="Accepted from the OSMO torchrun launcher.")
    parser.add_argument("--output", type=Path, required=True, help="Shared, persistent output directory.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recordings", type=int, default=8)
    parser.add_argument("--processes-per-gpu", type=int, default=4)
    parser.add_argument("--record-envs", type=int, default=2048)
    parser.add_argument("--record-steps", type=int, default=618)
    parser.add_argument("--max-updates", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--normalizer-windows", type=int, default=2_000_000)
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--wandb-project", default="isaaclab-physicsnemo")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    if args.recordings < 2 or args.processes_per_gpu < 1:
        parser.error("recordings must be at least 2 and processes-per-gpu must be positive")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    workflow_id = re.sub(r"[^A-Za-z0-9_-]", "-", os.environ.get("ISAACLAB_WORKFLOW_ID", "local"))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sync = output / ".workflow" / workflow_id
    sync.mkdir(parents=True, exist_ok=True)
    failed = sync / f"failed-rank-{rank}"
    failed.unlink(missing_ok=True)
    seeds = list(range(args.seed, args.seed + args.recordings))
    try:
        assigned = seeds[rank::world_size]
        for start in range(0, len(assigned), args.processes_per_gpu):
            record_batch(
                assigned[start : start + args.processes_per_gpu],
                output,
                device=f"cuda:{local_rank}",
                envs=args.record_envs,
                steps=args.record_steps,
            )
        deadline = time.monotonic() + 3600
        while not all((output / "data" / f"seed-{seed}.complete").is_file() for seed in seeds):
            if any(sync.glob("failed-rank-*")):
                raise RuntimeError("Another recording rank failed; inspect its log.")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for all ANYmal-C recordings.")
            time.sleep(5)

        command = [
            sys.executable,
            str(HERE / "train.py"),
            "--dataset",
            *(str(output / "data" / f"seed-{seed}") for seed in seeds),
            "--output",
            str(output / "train"),
            "--max-updates",
            str(args.max_updates),
            "--batch-size",
            str(args.batch_size),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--normalizer-windows",
            str(args.normalizer_windows),
            "--read-workers",
            str(args.read_workers),
            "--device",
            f"cuda:{local_rank}",
        ]
        if args.wandb_project:
            command.extend(("--wandb-project", args.wandb_project))
            command.extend(("--wandb-name", args.wandb_name or f"anymal-c-nerd-{workflow_id}"))
        log_path = output / "logs" / f"train-rank-{rank}.log"
        print(f"Rank {rank} training {len(seeds)} datasets; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if rank == 0:
            print(f"ANYmal-C artifact: {output / 'train' / 'artifact'}", flush=True)
    except Exception as exc:
        failed.write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
