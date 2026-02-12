# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal repro for fixed-base write_root_link_pose_to_sim bug.

Usage:
    ./isaaclab.sh -p scripts/demos/repro_fixed_base_pose_bug.py --num_envs 4 --visualize
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Repro fixed-base pose bug")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--visualize", action="store_true", help="Enable visualizer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from newton import ModelBuilder
from newton.solvers import SolverNotifyFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage

from isaaclab_assets import ALLEGRO_HAND_CFG


def main():
    num_envs = args_cli.num_envs
    device = args_cli.device if hasattr(args_cli, "device") and args_cli.device else "cuda:0"

    sim_cfg = SimulationCfg(dt=0.01, device=device, gravity=(0.0, 0.0, -9.81))

    visualizer = None
    if args_cli.visualize:
        from isaaclab.visualizers import NewtonVisualizerCfg
        from isaaclab.visualizers.newton_visualizer import NewtonVisualizer

        visualizer = NewtonVisualizer(NewtonVisualizerCfg(
            window_width=1280, window_height=720, update_frequency=1,
            show_joints=True, show_contacts=True,
            camera_position=(1.0, 1.0, 1.0), camera_target=(0.0, 0.0, 0.3),
        ))

    with build_simulation_context(auto_add_lighting=True, add_ground_plane=True, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None

        # Spawn articulations at heterogeneous positions
        translations = torch.zeros(num_envs, 3, device=device)
        translations[:, 0] = torch.arange(num_envs, device=device) * 0.5
        translations[:, 2] = torch.arange(num_envs, device=device) * 0.1

        for i in range(num_envs):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i].tolist())

        articulation_cfg = ALLEGRO_HAND_CFG.copy()
        articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_.*/Robot"))

        def set_builder():
            stage = get_current_stage()
            builder = ModelBuilder()
            for i in range(num_envs):
                proto = ModelBuilder()
                proto.add_usd(stage, root_path=f"/World/Env_{i}", load_visual_shapes=False)
                builder.add_world(proto)
            NewtonManager.set_builder(builder)
            NewtonManager._num_envs = num_envs

        NewtonManager.add_on_init_callback(set_builder)

        sim.reset()
        if visualizer:
            visualizer.initialize()

        dt = sim.cfg.dt

        # Run a few steps
        for _ in range(5):
            sim.step(render=False)
            articulation.update(dt)
            if visualizer:
                visualizer.step(dt, state=NewtonManager._state_0)

        # Reset with new poses - only move in z-axis
        new_poses = torch.zeros((num_envs, 7), device=device)
        # new_poses[:, 0] = translations[:, 0]  # keep original x
        # new_poses[:, 1] = translations[:, 1]  # keep original y
        new_poses[:, 2] += 0.3  # move up in z
        new_poses[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity quaternion

        # print(f"Resetting {num_envs} fixed-base articulations to new poses...")
        articulation.write_root_link_pose_to_sim(new_poses)

        # Step to apply
        sim.step(render=False)
        articulation.update(dt)

        # Now trigger the buggy code path
        print("Running buggy set_root_transforms...")
        pose_wp = articulation._data._sim_bind_root_link_pose_w  # (num_instances,) of wp.transformf

        env_mask = articulation._data.ALL_ENV_MASK

        # THE BUG: reshape to (num_instances, 1, 1)
        pose_reshaped = pose_wp.reshape((articulation.num_instances, 1, 1))

        articulation._root_view.set_root_transforms(
            NewtonManager.get_model(), pose_reshaped, mask=env_mask
        )
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_PROPERTIES)

        print("Buggy call completed - simulating...")

        # Continue simulation
        for _ in range(100):
            sim.step(render=False)
            articulation.update(dt)
            if visualizer:
                visualizer.step(dt, state=NewtonManager._state_0)

        if visualizer:
            print("Close window to exit...")
            while visualizer.is_running():
                visualizer.step(dt, state=NewtonManager._state_0)
            visualizer.close()

        print("Done")


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
