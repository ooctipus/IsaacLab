# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import io
import os

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter:
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, **kwargs):
        self.wandb_created_here = False
        self.log_dir = log_dir
        self.kwargs = kwargs
        if wandb.run is None:
            if kwargs["project"] is None:
                raise KeyError("project name must be provided when wandb is not initialized")
            if kwargs["name"] is None:
                raise KeyError("wand name must be provided when wandb is not initialized")
            try:
                entity = os.environ["WANDB_USERNAME"]
            except KeyError:
                raise KeyError(
                    "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
                )

            wandb.init(
                project=kwargs.get("project"), entity=entity, group=kwargs.get("group"), resume=kwargs.get("resume")
            )
            wandb.run.name = kwargs.get("name") + wandb.run.name.split("-")[-1]  # type: ignore
            self.wandb_created_here = True

        wandb.log({"log_dir": self.wandb_name})
        self.dump_logger("wandb_logger.yaml")

    @property
    def wandb_project(self):
        return self.kwargs.get("project")

    @property
    def wandb_name(self):
        return wandb.run.name  # type: ignore

    @property
    def wandb_id(self):
        return wandb.run.id  # type: ignore

    @property
    def group(self):
        return self.kwargs.get("group")

    def dump_logger(self, file_name: str) -> None:
        pass
        # logger_detail = {
        #     "project": self.wandb_project,
        #     "name": self.wandb_name,
        #     "id": self.wandb_id,
        # }
        # dump_yaml(self.storage_mgr, os.path.join(self.log_dir, file_name), logger_detail)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        wandb.log({tag: scalar_value}, step=global_step)

    def dump_history(self, prefix, file_name):
        import pandas as pd

        api = wandb.Api()
        run = api.run(f"{os.environ['WANDB_USERNAME']}/{self.wandb_project}/{self.wandb_id}")
        history = run.history()
        df = pd.DataFrame(history)
        eval_columns = [col for col in df.columns if col.startswith(prefix)]
        df_evaluation = df[eval_columns]

        csv_buffer = io.BytesIO()
        df_evaluation.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        # Define the remote path where the CSV will be stored
        # remote_path = os.path.join(self.log_dir, file_name)
        # # Write the CSV to remote storage using the storage handler
        # self.storage_mgr.write_file(remote_path, csv_buffer.read())

    def stop(self):
        if self.wandb_created_here:
            wandb.finish()

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def add_video_from_file(self, tag, video_path, frames_per_sec, global_step=None):
        wandb.log({tag: wandb.Video(video_path, caption="video", fps=frames_per_sec)}, step=global_step)

    def add_image(self, tag, image_array, caption="none", global_step=None):
        images = wandb.Image(image_array, caption=caption)
        wandb.log({tag: images}, step=global_step)

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
