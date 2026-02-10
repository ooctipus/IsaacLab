#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug script for visualizing finger collision tests with Newton visualizer.

Usage:
    python scripts/demos/debug_finger_collision.py --finger index --shape sphere --pipeline newton
    python scripts/demos/debug_finger_collision.py --finger thumb --shape box --pipeline mujoco

Controls:
    - SPACE: Pause/resume rendering
    - P: Pause/resume training (physics)
    - Mouse: Rotate camera
    - Scroll: Zoom
"""

import argparse
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.visualizers import NewtonVisualizerCfg
from isaaclab.visualizers.newton_visualizer import NewtonVisualizer

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG


# Configuration
FINGER_NAMES = ["index", "middle", "ring", "thumb"]

ALLEGRO_FINGERTIP_OFFSETS = {
    "index": (-0.052, -0.252, 0.052),
    "middle": (-0.001, -0.252, 0.052),
    "ring": (0.054, -0.252, 0.052),
    "thumb": (-0.168, -0.039, 0.080),
}

ALLEGRO_FINGER_JOINTS = {
    "index": ["index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"],
    "middle": ["middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"],
    "ring": ["ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"],
    "thumb": ["thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"],
}


def make_sim_cfg(use_mujoco_contacts: bool, device: str) -> SimulationCfg:
    """Create simulation config."""
    solver_cfg = MJWarpSolverCfg(
        njmax=300,
        nconmax=300,
        ls_iterations=20,
        cone="elliptic",
        impratio=100,
        ls_parallel=True,
        integrator="euler",
        use_mujoco_contacts=use_mujoco_contacts,
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )

    return SimulationCfg(
        newton_cfg=newton_cfg,
        dt=1.0 / 240.0,
        gravity=(0.0, 0.0, 0.0),  # Zero gravity
        device=device,
    )


def run_finger_collision_test(target_finger: str, drop_shape: str, use_mujoco_contacts: bool, device: str = "cuda:0"):
    """Run the finger collision test with visualization."""

    print(f"\n{'='*60}")
    print(f"Finger Collision Debug")
    print(f"  Target finger: {target_finger}")
    print(f"  Drop shape: {drop_shape}")
    print(f"  Pipeline: {'mujoco_contacts' if use_mujoco_contacts else 'newton_contacts'}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    SIM_DT = 1.0 / 240.0
    drop_steps = 480
    settle_steps = 30
    hand_pos = (0.0, 0.0, 0.5)
    drop_height = 0.10

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device)

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create hand
        hand_cfg = ALLEGRO_HAND_CFG.copy()
        hand_cfg.prim_path = "/World/Hand"
        hand_cfg.init_state.pos = hand_pos

        # Create ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        hand = Articulation(hand_cfg)

        # Get fingertip offset for target finger
        fingertip_offset = ALLEGRO_FINGERTIP_OFFSETS[target_finger]
        drop_pos = (
            hand_pos[0] + fingertip_offset[0],
            hand_pos[1] + fingertip_offset[1],
            hand_pos[2] + fingertip_offset[2] + drop_height,
        )

        # Create drop object
        drop_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0
        )
        drop_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        drop_mass_props = sim_utils.MassPropertiesCfg(mass=0.2)
        drop_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))

        if drop_shape == "sphere":
            drop_spawn = sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
            )
        else:  # box
            drop_spawn = sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
            )

        drop_obj_cfg = RigidObjectCfg(
            prim_path="/World/DropObject",
            spawn=drop_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=drop_pos),
        )
        drop_object = RigidObject(drop_obj_cfg)

        # Reset simulation
        sim.reset()
        hand.reset()
        drop_object.reset()

        # Create and initialize Newton visualizer AFTER sim.reset() so NewtonManager._model exists
        print("Initializing Newton visualizer...")
        viz_cfg = NewtonVisualizerCfg(
            window_width=1280,
            window_height=720,
            update_frequency=1,
            show_joints=True,
            show_contacts=True,  # Show contacts for debugging
            show_com=False,
            camera_position=(0.5, -0.8, 0.8),
            camera_target=(0.0, -0.1, 0.5),
        )
        visualizer = NewtonVisualizer(viz_cfg)
        visualizer.initialize()
        print("Visualizer initialized!")

        print("Settling hand...")
        for step in range(settle_steps):
            hand.write_data_to_sim()
            sim.step(render=False)
            # Update visualizer with current state
            visualizer.step(SIM_DT, state=NewtonManager._state_0)
            hand.update(SIM_DT)

        # Reset drop object and give initial velocity
        drop_object.reset()
        initial_velocity = torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]], device=device)
        drop_object.write_root_velocity_to_sim(initial_velocity)

        # Record initial joint positions
        initial_joint_pos = wp.to_torch(hand.data.joint_pos).clone()
        joint_names = hand.data.joint_names

        # Track peak deflection per finger
        peak_deflection = {finger: 0.0 for finger in FINGER_NAMES}

        print(f"\nDropping {drop_shape} on {target_finger} finger...")
        print("Controls: SPACE=pause rendering, P=pause physics, Mouse=rotate, Scroll=zoom")
        print("-" * 60)

        # Run simulation with visualization
        for step in range(drop_steps):
            hand.write_data_to_sim()
            drop_object.write_data_to_sim()
            sim.step(render=False)
            # Update visualizer
            visualizer.step(SIM_DT, state=NewtonManager._state_0)
            hand.update(SIM_DT)
            drop_object.update(SIM_DT)

            # Track peak deflection for each finger
            current_joint_pos = wp.to_torch(hand.data.joint_pos)[0]
            for finger_name in FINGER_NAMES:
                finger_deflection = 0.0
                for joint_name in ALLEGRO_FINGER_JOINTS[finger_name]:
                    if joint_name in joint_names:
                        idx = joint_names.index(joint_name)
                        finger_deflection += abs(current_joint_pos[idx].item() - initial_joint_pos[0, idx].item())
                peak_deflection[finger_name] = max(peak_deflection[finger_name], finger_deflection)

            # Print progress every 60 steps (0.25 seconds)
            if step % 60 == 0:
                print(f"Step {step:4d}: ", end="")
                for finger in FINGER_NAMES:
                    marker = ">>>" if finger == target_finger else "   "
                    print(f"{marker}{finger}: {peak_deflection[finger]:.4f}  ", end="")
                print()

        # Final results
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        target_peak = peak_deflection[target_finger]
        for finger in FINGER_NAMES:
            marker = ">>>" if finger == target_finger else "   "
            status = "PASS" if finger == target_finger or peak_deflection[finger] <= target_peak else "FAIL"
            print(f"{marker} {finger:8s}: {peak_deflection[finger]:.6f}  [{status}]")

        # Check if test would pass
        test_pass = True
        if target_peak <= 0.01:
            print(f"\nFAIL: Target finger '{target_finger}' deflection too small ({target_peak:.6f})")
            test_pass = False

        for finger in FINGER_NAMES:
            if finger != target_finger and peak_deflection[finger] > target_peak:
                print(f"\nFAIL: '{finger}' ({peak_deflection[finger]:.4f}) > target ({target_peak:.4f})")
                test_pass = False

        if test_pass:
            print("\n✓ TEST WOULD PASS")
        else:
            print("\n✗ TEST WOULD FAIL")

        print("\nClose the visualizer window to exit (or Ctrl+C).")

        # Keep the visualizer running until closed
        try:
            while visualizer.is_running():
                sim.step(render=False)
                visualizer.step(SIM_DT, state=NewtonManager._state_0)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            visualizer.close()


def main():
    parser = argparse.ArgumentParser(description="Debug finger collision tests with Newton visualizer")
    parser.add_argument(
        "--finger",
        type=str,
        choices=FINGER_NAMES,
        default="ring",
        help="Target finger to test (default: ring)",
    )
    parser.add_argument(
        "--shape",
        type=str,
        choices=["sphere", "box"],
        default="sphere",
        help="Drop shape type (default: sphere)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["newton", "mujoco"],
        default="mujoco",
        help="Collision pipeline (default: mujoco)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device (default: cuda:0)",
    )

    args = parser.parse_args()

    run_finger_collision_test(
        target_finger=args.finger,
        drop_shape=args.shape,
        use_mujoco_contacts=(args.pipeline == "mujoco"),
        device=args.device,
    )


if __name__ == "__main__":
    main()
