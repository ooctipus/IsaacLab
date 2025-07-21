# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--logger", type=str, default=None, help="logging device.")

# add argparse arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.utils.wandb_upload_info import InfoWandbUploadPatcher

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    # exp_mgr.update_experiment_cfg(args_cli, "state_machine", "state_machine_entry_point")
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    sm_cfg = load_cfg_from_registry(task_name, "state_machine_entry_point")

    sim_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    sm = sm_cfg.class_type(sm_cfg, sim_env.unwrapped)

    if args_cli.logger == "wandb":
        info_upload_patcher = InfoWandbUploadPatcher(
            wandb_project=task_name, wandb_group="info", wandb_runid=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        info_upload_patcher.apply_patch()

    # reset environment at start
    sim_env.reset()
    sm.reset()
    timestep = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = sm.compute().to(sim_env.unwrapped.device)
            # step environment
            _, _, reset_terminated, reset_time_outs, _ = sim_env.step(actions)
            done = reset_terminated | reset_time_outs
            # reset state machine
            if done.any():
                sm.reset_idx(done.nonzero(as_tuple=False).squeeze(-1))
            # if args_cli.video:
            #     timestep += 1
            #     # Exit the play loop after recording one video
            #     if timestep == args_cli.video_length:
            #         break

    # close the environment
    if args_cli.logger == "wandb":
        info_upload_patcher.remove_patch()
    sim_env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
