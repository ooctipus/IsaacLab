# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import hashlib
import json
import os

# Suppress import error if wandb is not installed
with contextlib.suppress(ImportError):
    import wandb


@contextlib.contextmanager
def _exclusive_lock(path: str):
    """Serialize downloads sharing one local checkpoint cache."""
    if os.name != "posix":
        yield
        return

    import fcntl

    with open(path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _resolve_wandb_run(api, *, entity: str, project: str, run_id: str | None, run_name: str | None):
    if (run_id is None) == (run_name is None):
        raise ValueError("Exactly one of run_id or run_name must be provided.")
    if run_id is not None:
        return api.run(f"{entity}/{project}/{run_id}")

    project_path = f"{entity}/{project}"
    matches = [run for run in api.runs(project_path, filters={"display_name": run_name}) if run.name == run_name]
    if len(matches) != 1:
        raise ValueError(
            f"W&B display name '{run_name}' matched {len(matches)} runs in {project_path}; expected exactly one."
        )
    print(f"Resolved W&B run name '{run_name}' to ID '{matches[0].id}'.")
    return matches[0]


def _model_iteration(name: str) -> int:
    return int(name.rsplit("_", 1)[-1].removesuffix(".pt"))


def get_model_checkpoint(
    run_id: str | None = None,
    project="isaaclab",
    checkpoint: int = -1,
    wandb_username=None,
    tmp_folder_dir: str = "models_tmp",
    run_name: str | None = None,
) -> str:
    """Download a model checkpoint from Weights & Biases (W&B).

    Args:
        run_id: Internal ID of the W&B run. Mutually exclusive with :attr:`run_name`.
        project: Name of the W&B project.
        checkpoint: Specific checkpoint iteration to download. If -1, downloads the latest.
        wandb_username: W&B entity. If None, uses the environment variable ``WANDB_USERNAME``.
        tmp_folder_dir: Directory to save the downloaded model checkpoint.
        run_name: Exact W&B display name. It must identify exactly one run in the project.

    Returns:
        The path to the downloaded model checkpoint.
    """
    if wandb_username is None:
        wandb_username = os.environ.get("WANDB_USERNAME")
    if not wandb_username:
        raise ValueError("A W&B entity is required through wandb_username or WANDB_USERNAME.")

    cache_input = json.dumps(
        {
            "entity": wandb_username,
            "project": project,
            "run_id": run_id,
            "run_name": run_name,
            "checkpoint": checkpoint,
        },
        sort_keys=True,
    )
    cache_key = hashlib.sha256(cache_input.encode()).hexdigest()
    cache_root = os.path.join(tmp_folder_dir, ".wandb_checkpoint_cache")
    os.makedirs(cache_root, exist_ok=True)
    lock_path = os.path.join(cache_root, f"{cache_key}.lock")
    record_path = os.path.join(cache_root, f"{cache_key}.json")

    with _exclusive_lock(lock_path):
        if os.path.isfile(record_path):
            with contextlib.suppress(OSError, ValueError, KeyError):
                with open(record_path, encoding="utf-8") as record_file:
                    cached_path = json.load(record_file)["path"]
                if os.path.isfile(cached_path):
                    print(f"Using cached W&B checkpoint: {cached_path}")
                    return cached_path

        api = wandb.Api()
        wdb_run = _resolve_wandb_run(api, entity=wandb_username, project=project, run_id=run_id, run_name=run_name)
        print("Downloading model from wandb...", f"{wandb_username}/{project}/{wdb_run.id}")

        models = []
        for remote_file in wdb_run.files():
            if "model" not in remote_file.name or not remote_file.name.endswith(".pt"):
                continue
            with contextlib.suppress(ValueError):
                _model_iteration(remote_file.name)
                models.append(remote_file)
        models.sort(key=lambda model: _model_iteration(model.name))
        if not models:
            raise ValueError(f"No model checkpoints were found in W&B run '{wdb_run.id}'.")

        model = models[-1] if checkpoint == -1 or checkpoint is None else None
        if model is None:
            model = next((item for item in models if _model_iteration(item.name) == int(checkpoint)), None)
        if model is None:
            raise ValueError(f"Model with iteration {checkpoint} not found in run {wdb_run.id}.")

        target_folder = os.path.join(tmp_folder_dir, project, wdb_run.id)
        os.makedirs(target_folder, exist_ok=True)
        model.download(root=target_folder, replace=True)
        model_path = os.path.join(target_folder, model.name)
        record_tmp = f"{record_path}.{os.getpid()}.tmp"
        with open(record_tmp, "w", encoding="utf-8") as record_file:
            json.dump({"path": model_path}, record_file)
        os.replace(record_tmp, record_path)
        return model_path
