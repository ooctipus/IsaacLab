# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the visualizer tiled camera panel.

.. code-block:: bash

    # Kit visualizer tiled camera panel
    uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --task Isaac-Velocity-Rough-AnymalD --num_envs 256 visualizer=kit

    # Newton visualizer tiled camera panel
    uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --task Isaac-Velocity-Rough-AnymalD --num_envs 256 visualizer=newton_gl

"""

from __future__ import annotations

import argparse
import contextlib

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

DEFAULT_TASK = "Isaac-Velocity-Rough-AnymalD"


# add argparse arguments
parser = argparse.ArgumentParser(description="Showcase the Kit/Newton visualizer tiled camera panel.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)


def main():
    """Run a random-action environment with a tiled camera visualizer."""
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.streaming_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/StreamingCamera",
        offset=CameraCfg.OffsetCfg(pos=(3.0, 3.0, 3.0), rot=(0.1759, 0.4247, 0.8205, 0.3399), convention="world"),
        width=320,
        height=240,
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        renderer_cfg=MultiBackendRendererCfg(),
    )
    for visualizer_cfg in (
        env_cfg.sim.visualizer_cfgs.kit,
        env_cfg.sim.visualizer_cfgs.newton_gl,
        env_cfg.sim.visualizer_cfgs.rerun,
        env_cfg.sim.visualizer_cfgs.viser,
    ):
        visualizer_cfg.streaming_view = True
        visualizer_cfg.streaming_camera = "{ENV_REGEX_NS}/StreamingCamera"
        visualizer_cfg.streaming_envs = 12
    env_cfg.sim.visualizer_cfgs.kit.streaming_envs = 36
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg = resolve_config(env_cfg, config_overrides)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()

        sim = env.unwrapped.sim
        while sim.is_headless_or_exist_active_visualizer():
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                env.step(actions)

        env.close()


if __name__ == "__main__":
    main()
