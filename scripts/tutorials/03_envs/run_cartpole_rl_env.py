# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    uv run python scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

Trailing ``key=value`` arguments (e.g. ``physics=isaacsim_physx``) are forwarded as Hydra-style
overrides to the task configuration; see :func:`~isaaclab_tasks.utils.resolve_task_config`.

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import torch

from isaaclab.envs import ManagerBasedRLEnv


def main():
    """Main function."""
    # create environment configuration
    env_cfg, _ = resolve_task_config("Isaac-Cartpole", None, overrides=config_overrides)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        env = ManagerBasedRLEnv(cfg=env_cfg)

        count = 0
        while env.sim.is_headless_or_exist_active_visualizer():
            with torch.inference_mode():
                if count % 300 == 0:
                    count = 0
                    env.reset()
                    print("-" * 80)
                    print("[INFO]: Resetting environment...")
                joint_efforts = torch.randn_like(env.action_manager.action)
                obs, rew, terminated, truncated, info = env.step(joint_efforts)
                print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
                count += 1

        env.close()


if __name__ == "__main__":
    main()
