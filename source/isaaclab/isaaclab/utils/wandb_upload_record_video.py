from __future__ import annotations

import os
import logging
from contextlib import contextmanager, nullcontext
from gymnasium.wrappers import RecordVideo

from .wandb_summary_writer import WandbSummaryWriter
import gc

class RecordVideoWandbUploadPatcher:

    def __init__(self, wandb_project: str = "", wandb_runid: str = ""):
        self.setup_writer: bool = False
        self.writer: WandbSummaryWriter = None  # type: ignore
        self.round_counter = 0
        self.wandb_project = wandb_project
        self.wandb_runid = wandb_runid

        self.original_step = RecordVideo.step
        self.original_stop_recording = RecordVideo.stop_recording
    
    def apply_patch(self):
        logging.debug("WandB video upload RecordVideo patcher: Patching.... ")
        
        def wandb_upload_record_video_step(record_video_self, action):
            # Initialize WandB writer on first use
            if not self.setup_writer:
                wandb_args = {"project": self.wandb_project, "group": "monitor", "name": self.wandb_runid}
                self.writer = WandbSummaryWriter(record_video_self.video_folder, 10, **wandb_args)
                self.setup_writer = True

            # Delegate to original step
            return self.original_step(record_video_self, action)

        def wandb_upload_stop_recording(record_video_self):
            # Finish recording
            if not record_video_self.recording:
                return
            video_folder = record_video_self.video_folder
            video_name = record_video_self._video_name
            self.original_stop_recording(record_video_self)
            video_path = os.path.join(video_folder, f"{video_name}.mp4")
            self.writer.add_video_from_file(f"Monitor/video-{self.round_counter}", video_path, frames_per_sec=30)
            self.round_counter += 1
            gc.collect()

        # Patch methods on RecordVideo
        RecordVideo.step = wandb_upload_record_video_step
        RecordVideo.stop_recording = wandb_upload_stop_recording

        logging.debug("WandB video upload RecordVideo patcher: Patch Done!")

    def remove_patch(self):
        RecordVideo.step = self.original_step
        RecordVideo.stop_recording = self.original_stop_recording
        logging.debug("WandB video upload RecordVideo patcher: Removed")


@contextmanager
def patch_record_video_with_wandb_upload(enable: bool, wandb_project: str, wandb_runid: str):
    if not enable:
        with nullcontext():
            yield
        return

    patcher = RecordVideoWandbUploadPatcher(wandb_project=wandb_project, wandb_runid=wandb_runid)
    patcher.apply_patch()
    try:
        yield
    finally:
        patcher.remove_patch()
