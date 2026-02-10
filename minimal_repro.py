#!/usr/bin/env python3
"""
MINIMAL REPRODUCTION: Newton solver NaN with heterogeneous objects.

Saves state history and replays the last 50 steps when crash happens.
Only the crashing environment moves during replay, others stay fixed.
"""

import argparse
import numpy as np
import torch
import warp as wp
import os
from collections import deque

os.environ["HYDRA_FULL_ERROR"] = "1"

import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab.sim._impl.newton_manager import NewtonManager


class StateSnapshot:
    """Snapshot of simulation state for replay."""
    def __init__(self, step, niter):
        self.step = step
        self.niter = niter.copy()
        state = NewtonManager._state_0
        self.body_q = state.body_q.numpy().copy()
        self.body_qd = state.body_qd.numpy().copy()


def main():
    parser = argparse.ArgumentParser(description="Minimal reproduction for Newton NaN with heterogeneous objects")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    args = parser.parse_args()
    
    from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
        DexsuiteKukaAllegroLiftEnvCfg,
    )
    
    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg.scene.num_envs = 16
    env_cfg.sim.newton_cfg.use_cuda_graph = False
    
    env = gym.make("Isaac-Dexsuite-Kuka-Allegro-Lift-v0", cfg=env_cfg, render_mode=None)
    env.reset()
    device = env.unwrapped.device
    
    # Initialize visualizer only if requested
    visualizer = None
    if args.visualize:
        from isaaclab.visualizers import NewtonVisualizerCfg
        from isaaclab.visualizers.newton_visualizer import NewtonVisualizer
        
        print("Initializing Newton visualizer...")
        viz_cfg = NewtonVisualizerCfg(
            window_width=1280,
            window_height=720,
            update_frequency=1,
            show_joints=True,
            show_contacts=True,
            show_com=False,
            camera_position=(2.0, 2.0, 2.0),
            camera_target=(0.0, 0.0, 0.5),
        )
        visualizer = NewtonVisualizer(viz_cfg)
        visualizer.initialize()
        print("Visualizer initialized!")
    
    print("\n=== Minimal repro with state history ===")
    print("When crash happens, replays last 50 steps")
    print("Only crashing env moves, others stay fixed")
    if args.visualize:
        print("Controls: Mouse=rotate, Scroll=zoom\n")
    
    dt = env.unwrapped.step_dt
    
    # Get object asset for velocity tracking
    object_asset = env.unwrapped.scene["object"]
    
    HISTORY_SIZE = 200  # More history
    state_history = deque(maxlen=HISTORY_SIZE)
    
    try:
        for step in range(2000):
            action = torch.rand(env.action_space.shape, device=device) * 2 - 1
            
            wp.synchronize()
            niter = NewtonManager._solver.mjw_data.solver_niter.numpy()
            state_history.append(StateSnapshot(step, niter))
            
            try:
                obs, _, _, _, _ = env.step(action)
                has_nan = torch.isnan(obs['policy']).any()
            except Exception:
                has_nan = True
            
            wp.synchronize()
            niter = NewtonManager._solver.mjw_data.solver_niter.numpy()
            max_niter = niter.max()
            
            # Get object velocities (in body frame)
            obj_lin_vel = wp.to_torch(object_asset.data.root_lin_vel_b)  # (num_envs, 3)
            obj_ang_vel = wp.to_torch(object_asset.data.root_ang_vel_b)  # (num_envs, 3)
            obj_lin_speed = torch.norm(obj_lin_vel, dim=1)  # (num_envs,)
            obj_ang_speed = torch.norm(obj_ang_vel, dim=1)  # (num_envs,)
            max_lin_speed = obj_lin_speed.max().item()
            max_ang_speed = obj_ang_speed.max().item()
            
            if visualizer:
                visualizer.step(dt, state=NewtonManager._state_0)
            
            # Print if high speed or high niter or periodically
            if max_lin_speed > 5 or max_ang_speed > 20 or max_niter > 10 or step % 200 == 0:
                lin_sp = np.array2string(obj_lin_speed.cpu().numpy(), precision=1, suppress_small=True, max_line_width=200)
                ang_sp = np.array2string(obj_ang_speed.cpu().numpy(), precision=1, suppress_small=True, max_line_width=200)
                print(f"Step {step}: niter={niter}, lin_sp={lin_sp}, ang_sp={ang_sp}")
            
            if has_nan:
                # Find which env crashed
                crashed_env = np.argmax(niter)
                print(f"\n!!! CRASH at step {step}, niter={niter} !!!")
                print(f"Environment {crashed_env} crashed!")
                print(f"Replaying last {len(state_history)} steps (only env {crashed_env} moves)...")
                
                # Get body count per env
                num_envs = env_cfg.scene.num_envs
                total_bodies = state_history[0].body_q.shape[0]
                bodies_per_env = total_bodies // num_envs
                
                # Get fixed state for non-crashing envs (from first snapshot)
                fixed_body_q = state_history[0].body_q.copy()
                fixed_body_qd = state_history[0].body_qd.copy()
                
                # Replay in loop if visualizer is active
                if visualizer:
                    replay_state = NewtonManager._state_0
                    wp_device = wp.get_device(str(device))
                    loop_count = 0
                    
                    while visualizer.is_running():
                        loop_count += 1
                        print(f"\n--- Replay loop {loop_count} (env {crashed_env} only) ---")
                        
                        for i, snapshot in enumerate(state_history):
                            # Create composite state: fixed envs + moving crashed env
                            composite_body_q = fixed_body_q.copy()
                            composite_body_qd = fixed_body_qd.copy()
                            
                            # Copy only crashed env's state
                            start = crashed_env * bodies_per_env
                            end = (crashed_env + 1) * bodies_per_env
                            composite_body_q[start:end] = snapshot.body_q[start:end]
                            composite_body_qd[start:end] = snapshot.body_qd[start:end]
                            
                            with wp.ScopedDevice(wp_device):
                                replay_state.body_q.assign(wp.from_numpy(composite_body_q, dtype=wp.transform))
                                replay_state.body_qd.assign(wp.from_numpy(composite_body_qd, dtype=wp.spatial_vector))
                            
                            for _ in range(3):  # Faster playback
                                visualizer.step(dt, state=replay_state)
                                if not visualizer.is_running():
                                    break
                            
                            if i % 50 == 0:
                                print(f"  Step {snapshot.step}: niter={snapshot.niter}")
                            
                            if not visualizer.is_running():
                                break
                        
                        for _ in range(60):  # Pause between loops
                            visualizer.step(dt, state=replay_state)
                            if not visualizer.is_running():
                                break
                
                break
            
            if visualizer and not visualizer.is_running():
                print("Visualizer closed by user")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if visualizer:
            visualizer.close()
        env.close()
        print("Done.")


if __name__ == "__main__":
    wp.init()
    main()
