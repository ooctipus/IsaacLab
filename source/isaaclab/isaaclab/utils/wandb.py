# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import contextlib
import os

# Suppress import error if wandb is not installed
with contextlib.suppress(ImportError):
    import wandb


def get_model_checkpoint(
    run_id: str, project="isaaclab", checkpoint: int = -1, wandb_username=None, tmp_folder_dir: str = "models_tmp"
) -> str:
    """
    Downloads a model checkpoint from Weights & Biases (wandb).
    Args:
        run_id: The ID of the wandb run.
        project: The name of the wandb project.
        checkpoint: The specific checkpoint iteration to download. If -1, downloads the latest.
        wandb_username: The username for wandb. If None, uses the environment variable WANDB_USERNAME.
        tmp_folder_dir: Directory to save the downloaded model checkpoint.
    Returns:
        The path to the downloaded model checkpoint.

    Example:
        model_path = get_model_checkpoint(
            run_id="my_run_id", project="my_project", checkpoint=100, wandb_username="my_username"
        )
        This will download the model checkpoint from https://wandb.ai/my_username/my_project/runs/my_run_id and save it
        to models_tmp/my_project/my_run_id/model_100.pt
    """

    api = wandb.Api()
    if wandb_username is None:
        wandb_username = os.environ.get("WANDB_USERNAME")

    print("Downloading model from wandb...", f"{wandb_username}/{project}/{run_id}")

    wdb_run = api.run(f"{wandb_username}/{project}/{run_id}")

    # Filter server-side: runs that log media (e.g. Object3D panels) accumulate tens of
    # thousands of files, and paginating all of them at the 50-per-page default costs
    # minutes. ``pattern`` is MySQL LIKE syntax, so this mirrors "contains model, ends in .pt".
    models = list(wdb_run.files(pattern="%model%.pt", per_page=1000))
    if not models:
        raise ValueError(f"No model checkpoints found in run {run_id}. Exiting...")

    # sort models
    models = sorted(models, key=lambda m: int(m.name.split("_")[-1].split(".")[0]))

    model = None
    if checkpoint == -1 or checkpoint is None:
        model = models[-1]
    else:
        for remote_model in models:
            if int(remote_model.name.split("_")[-1].split(".")[0]) == int(checkpoint):
                model = remote_model
                break
    if model is None:
        raise ValueError(f"Model with iteration {checkpoint} not found in run {run_id}. Exiting...")

    target_folder = os.path.join(tmp_folder_dir, project, run_id)
    # download models
    os.makedirs(target_folder, exist_ok=True)
    model.download(root=target_folder, replace=True)
    model_path = os.path.join(target_folder, model.name)
    return model_path
