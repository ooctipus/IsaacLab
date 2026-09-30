# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the H1 rough terrain environment.

.. code-block:: bash

    # This interactive demo supports only Isaac Sim PhysX and the Kit visualizer.
    uv run --extra isaacsim python scripts/demos/h1_locomotion.py physics=isaacsim_physx visualizer=kit

"""

import argparse
from importlib import metadata

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # isort: skip

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

TASK = "Isaac-Velocity-Rough-H1"
RL_LIBRARY = "rsl_rl"

parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser, agent_library=RL_LIBRARY)

import torch
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.core.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg


@configclass
class DemoSceneCfg(MySceneCfg):
    """H1 scene with one plan-owned third-person camera per robot."""

    third_person_camera = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/third_person_camera",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-2.5, 0.0, 0.8),
            rot=(0.479649603, -0.479649603, -0.519553959, 0.519553900),
        ),
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.5),
    )


@configclass
class DemoCfg(H1RoughEnvCfg):
    """Kit-only H1 interactive-demo configuration."""

    scene: DemoSceneCfg = DemoSceneCfg(num_envs=4096, env_spacing=2.5)
    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        device=args_cli.device,
        physics=preset(
            default=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
            isaacsim_physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
        ),
    )


class H1RoughDemo:
    """This class provides an interactive demo for the H1 rough terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-H1 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self, env_cfg: DemoCfg):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK, *backend_names)
        if checkpoint is None:
            raise FileNotFoundError("No published checkpoint is available for the H1 locomotion demo.")
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        plan = self.env.unwrapped.sim.get_clone_plan()
        if plan is None:
            raise RuntimeError("H1 third-person cameras require an active clone plan.")
        self._camera_paths = cloner.query.destination_paths(plan, env_cfg.scene.third_person_camera.prim_path)
        self.perspective_path = env_cfg.sim.visualizer_cfgs.prim_path
        self._third_person_view = False
        self.setup_viewport()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self._selected_id = None
        self._previous_selected_id = None
        self.set_up_keyboard()

    def setup_viewport(self):
        """Bind the viewport used to switch between the two planned cameras."""
        from omni.kit.viewport.utility import get_viewport_from_window_name

        self.viewport = get_viewport_from_window_name("Viewport")

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        import carb
        import omni

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._prim_selection = omni.usd.get_context().get_selection()
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        import carb

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    self._third_person_view = not self._third_person_view
                    self.viewport.set_active_camera(
                        self._camera_paths[self._selected_id] if self._third_person_view else self.perspective_path
                    )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid H1 robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self._third_person_view = False
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self._third_person_view = True
                    self.viewport.set_active_camera(self._camera_paths[self._selected_id])
            else:
                print("The selected prim was not a H1 robot")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")


def main():
    """Main function."""
    env_cfg = resolve_config(DemoCfg(), ["presets=play", *config_overrides])
    env_cfg.scene.num_envs = 25
    env_cfg.episode_length_s = 1000000
    env_cfg.curriculum = None
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
    with launch_simulation(env_cfg, args_cli):
        demo_h1 = H1RoughDemo(env_cfg)
        obs, _ = demo_h1.env.reset()
        while demo_h1.env.unwrapped.sim.is_headless_or_exist_active_visualizer():
            demo_h1.update_selected_object()
            with torch.inference_mode():
                action = demo_h1.policy(obs)
                obs, _, _, _ = demo_h1.env.step(action)
                obs[:, 9:13] = demo_h1.commands
        demo_h1.env.close()


if __name__ == "__main__":
    main()
