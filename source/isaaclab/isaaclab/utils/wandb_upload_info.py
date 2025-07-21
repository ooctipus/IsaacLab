# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
import torch
from contextlib import contextmanager, nullcontext

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

# from ...summary_writer import WandbSummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class InfoWandbUploadPatcher:

    def __init__(
        self,
        wandb_project: str = "",
        wandb_group: str = "",
        wandb_runid: str = "",
        info_interval: int = 50,
        pad: int = 35,
    ):
        self.round_counter = 0
        self.wandb_project = wandb_project
        self.wandb_runid = wandb_runid
        self.info_interval = info_interval
        self.pad = pad

        self.original_manager_based_rl_env_step = ManagerBasedRLEnv.step

    def apply_patch(self):
        logging.debug("WandB info upload ManagerBasedRLEnv patcher: Patching.... ")

        def wandb_upload_info_step(info_self, action):
            # Initialize WandB on first use
            if wandb.run is None:
                wandb_args = {"project": self.wandb_project, "group": "monitor", "name": self.wandb_runid}
                try:
                    entity = os.environ["WANDB_USERNAME"]
                except KeyError:
                    raise KeyError(
                        "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
                    )

                wandb.init(project=wandb_args.get("project"), entity=entity, group=wandb_args.get("group"))
                wandb.run.name = wandb_args.get("name") + wandb.run.name.split("-")[-1]  # type: ignore

            # Original step
            original_step = self.original_manager_based_rl_env_step

            obs_buf, reward_buf, reset_terminated, sreset_time_outs, extras = original_step(info_self, action)

            # Upload to wandb
            ep_infos = []
            if "episode" in extras:
                ep_infos.append(extras["episode"])
            elif "log" in extras:
                ep_infos.append(extras["log"])

            # -- Episode info
            ep_string = ""
            if ep_infos:
                for key in ep_infos[0]:
                    infotensor = torch.tensor([], device=info_self.device)
                    for ep_info in ep_infos:
                        # handle scalar and zero dimensional tensor infos
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(info_self.device)))
                    value = torch.mean(infotensor)

                    # log to logger and terminal
                    info_interval = self.info_interval
                    pad = self.pad

                    if info_self.common_step_counter % info_interval == 0:
                        if "/" in key:
                            wandb.log({key: value}, step=info_self.common_step_counter)
                            ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                        else:
                            wandb.log({"Episode/" + key: value}, step=info_self.common_step_counter)
                            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            print(ep_string)
            return obs_buf, reward_buf, reset_terminated, sreset_time_outs, extras

        # Patch methods on ManagerBasedEnv
        ManagerBasedRLEnv.step = wandb_upload_info_step

        logging.debug("WandB video upload ManagerBasedRLEnv patcher: Patch Done!")

    def remove_patch(self):
        ManagerBasedRLEnv.step = self.original_manager_based_rl_env_step
        logging.debug("WandB video upload ManagerBasedRLEnv patcher: Removed")


@contextmanager
def patch_info_with_wandb_upload(enable: bool, wandb_project: str, wandb_group: str, wandb_runid: str):
    if not enable:
        with nullcontext():
            yield
        return

    patcher = InfoWandbUploadPatcher(wandb_project=wandb_project, wandb_group=wandb_group, wandb_runid=wandb_runid)
    patcher.apply_patch()
    try:
        yield
    finally:
        patcher.remove_patch()
