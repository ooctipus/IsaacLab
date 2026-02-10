#!/usr/bin/env python3
"""Capture state/actions before NaN crash for replay."""

import torch
import warp as wp
import numpy as np
import pickle
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab.sim._impl.newton_manager import NewtonManager


def capture_state():
    """Capture current simulation state."""
    state = {}
    
    wp.synchronize()
    
    # Get Newton state
    newton_state = NewtonManager._state_0
    if newton_state is not None:
        if newton_state.body_q is not None:
            state['body_q'] = newton_state.body_q.numpy().copy()
        if newton_state.body_qd is not None:
            state['body_qd'] = newton_state.body_qd.numpy().copy()
        if newton_state.joint_q is not None:
            state['joint_q'] = newton_state.joint_q.numpy().copy()
        if newton_state.joint_qd is not None:
            state['joint_qd'] = newton_state.joint_qd.numpy().copy()
    
    # Get solver iterations
    if NewtonManager._solver is not None:
        state['solver_niter'] = NewtonManager._solver.mjw_data.solver_niter.numpy().copy()
    
    return state


def repro():
    from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
        DexsuiteKukaAllegroLiftEnvCfg,
    )
    
    env_cfg = DexsuiteKukaAllegroLiftEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.newton_cfg.use_cuda_graph = False
    
    env = gym.make("Isaac-Dexsuite-Kuka-Allegro-Lift-v0", cfg=env_cfg, render_mode=None)
    
    print(f"Num envs: {env.unwrapped.num_envs}")
    
    obs, _ = env.reset()
    device = env.unwrapped.device
    
    # Ring buffer to store last N steps
    HISTORY_SIZE = 20
    history = []
    
    for step in range(2000):
        action = torch.rand(env.action_space.shape, device=device) * 2 - 1
        
        # Capture state BEFORE step
        pre_state = capture_state()
        pre_state['action'] = action.cpu().numpy().copy()
        pre_state['step'] = step
        
        # Keep last N states
        history.append(pre_state)
        if len(history) > HISTORY_SIZE:
            history.pop(0)
        
        # Step - catch any exception
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            has_nan = torch.isnan(obs['policy']).any()
        except Exception as e:
            print(f"\n!!! Exception at step {step}: {e}")
            has_nan = True
        
        if has_nan:
            print(f"\n!!! NaN/crash at step {step} !!!")
            
            # Capture post-crash state
            post_state = capture_state()
            
            # Save everything
            crash_data = {
                'history': history,
                'post_crash_state': post_state,
                'crash_step': step,
            }
            
            with open('crash_data.pkl', 'wb') as f:
                pickle.dump(crash_data, f)
            
            print(f"Saved crash data to crash_data.pkl")
            print(f"History contains {len(history)} states before crash")
            
            # Print solver iterations for last few steps
            print("\nSolver iterations before crash:")
            for i, h in enumerate(history[-5:]):
                niter = h.get('solver_niter', [])
                print(f"  Step {h['step']}: niter={niter.max() if len(niter) > 0 else 'N/A'}")
            
            if 'solver_niter' in post_state:
                print(f"  At crash: niter={post_state['solver_niter']}")
            
            break
        
        # Log every 50 steps
        if step % 50 == 0:
            conv = NewtonManager.get_solver_convergence_steps()
            print(f"Step {step}: niter={conv['max']}")
    
    env.close()


if __name__ == "__main__":
    repro()
