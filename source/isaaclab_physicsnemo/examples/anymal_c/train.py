# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the FlashNeRD ANYmal-C locomotion model on a dataset recorded by record.py.

The model sees the state in the base heading frame and is conditioned on the joint-position targets,
base-frame gravity, and the four fixed foot probes. FlashNeRD trains it for 60,000 updates with a 13-frame
receptive field and a 5 % trajectory validation split; this run uses resident windows, FP16 autocast,
global batch 4,096, and the ParallelNeRD geometry of the study's G1 model. Launch it under ``torchrun``
for several data-parallel ranks.
"""

import argparse
import logging
import os
from pathlib import Path

from anymal_c_env_cfg import (
    ANYMAL_MODEL_CONFIG,
    AnymalCTargetControlEnvCfg,
    anymal_dynamics_features,
    anymal_foot_contacts,
)
from isaaclab_physicsnemo import HeadingJointStateCodec, TrainingSettings, train_dynamics_artifact

from isaaclab.envs import ManagerBasedRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", type=Path, nargs="+", required=True, help="Datasets written by record.py, trained on together."
    )
    parser.add_argument("--output", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--max-updates", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=4096, help="Global batch over all ranks.")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--read-workers", type=int, default=8, help="Processes that read the recorded trajectories.")
    parser.add_argument(
        "--normalizer-windows", type=int, default=None, help="Fit the normalization on this many windows."
    )
    parser.add_argument(
        "--cache-dataset", action="store_true", help="Keep the loaded trajectories for later runs in --output."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default=None, help="Log training metrics and the final artifact to W&B.")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = f"cuda:{os.environ['LOCAL_RANK']}" if "LOCAL_RANK" in os.environ else args.device
    rank = int(os.environ.get("RANK", "0"))
    wandb_run = None
    if rank == 0 and args.wandb_project:
        import wandb

        args.output.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=str(args.output),
            config={
                "dataset": [str(path) for path in args.dataset],
                "max_updates": args.max_updates,
                "batch_size": args.batch_size,
                "history": 13,
                "seed": args.seed,
                "architecture": "parallel_nerd",
            },
        )
        wandb_run.define_metric("update")
        wandb_run.define_metric("train/*", step_metric="update")
        wandb_run.define_metric("validation/*", step_metric="update")

    def upload_checkpoint(updates: int, path: Path) -> None:
        if updates == args.max_updates:
            return  # The validated final artifact is uploaded below.
        artifact = wandb.Artifact("anymal-c-nerd", type="model", metadata={"updates": updates})
        artifact.add_dir(str(path))
        logged = wandb_run.log_artifact(artifact, aliases=[f"step-{updates}", "latest"])
        if wandb_run.settings.mode != "offline":
            logged.wait()  # Keep the staged files intact until the upload completes.

    try:
        # The codec and probes are built from the deployment scene, whose Newton model equals the teacher's.
        cfg = AnymalCTargetControlEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = device
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            model = env.sim.physics_manager.get_model()
            codec = HeadingJointStateCodec(model)
            summary = train_dynamics_artifact(
                args.dataset,
                model,
                args.output,
                codec=codec,
                features=anymal_dynamics_features(codec, anymal_foot_contacts(model)),
                settings=TrainingSettings(
                    history=13,
                    max_updates=args.max_updates,
                    batch_size=args.batch_size,
                    validation_fraction=0.05,
                    log_every=1000,
                    seed=args.seed,
                    architecture="parallel_nerd",
                    model_kwargs=ANYMAL_MODEL_CONFIG,
                    mixed_precision=True,
                    compile=True,
                    checkpoint_every=args.checkpoint_every,
                    validate_every=args.checkpoint_every,
                    read_workers=args.read_workers,
                    normalizer_windows=args.normalizer_windows,
                    cache_dataset=args.cache_dataset,
                ),
                device=device,
                on_metrics=(
                    (lambda update, metrics: wandb_run.log({"update": update, **metrics})) if wandb_run else None
                ),
                on_checkpoint=upload_checkpoint if wandb_run else None,
            )
        finally:
            env.close()
        if wandb_run is not None:
            artifact = wandb.Artifact("anymal-c-nerd", type="model", metadata={"updates": summary.parameter_updates})
            artifact.add_dir(str(summary.artifact_path))
            wandb_run.log_artifact(artifact, aliases=["latest", "final"])
            wandb_run.summary.update({f"final/{name}": value for name, value in summary.validation.items()})
            print(f"W&B: {wandb_run.url}")
        print(
            f"Trained {summary.parameter_updates} updates on {len(summary.train_trajectory_ids)} trajectories,"
            f" validated on {len(summary.validation_trajectory_ids)}"
        )
        for name, value in summary.validation.items():
            print(f"  validation {name}: {value:.6g}")
        print(f"Artifact: {summary.artifact_path}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
