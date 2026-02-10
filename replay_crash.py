#!/usr/bin/env python3
"""Replay captured crash state to reproduce NaN issue."""

import torch
import warp as wp
import numpy as np
import pickle
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


def analyze_crash_data():
    """Analyze the captured crash data - compare crashing vs non-crashing envs."""
    
    with open('crash_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    pre_crash = data['history'][-1]
    bq = pre_crash['body_q']
    bodies_per_env = bq.shape[0] // 4
    
    print(f"Bodies per env: {bodies_per_env}")
    print(f"Crash env: 1, Non-crash envs: 0, 2, 3")
    
    # Compare object positions (last body in each env)
    print("\n=== Object (last body) positions ===")
    for env in range(4):
        start = env * bodies_per_env
        obj_body = bq[start + bodies_per_env - 1]
        pos = obj_body[:3]
        quat = obj_body[3:7]
        crashed = "CRASHED" if env == 1 else ""
        print(f"Env {env}: pos=({pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}) {crashed}")
    
    # Compare hand tip positions (a few bodies before object)
    print("\n=== Some fingertip body positions ===")
    for env in range(4):
        start = env * bodies_per_env
        # Bodies 14, 19, 24, 29 are likely fingertips
        fingertips = [14, 19, 24, 29]
        crashed = "CRASHED" if env == 1 else ""
        print(f"Env {env} {crashed}:")
        for ft in fingertips:
            body = bq[start + ft]
            pos = body[:3]
            print(f"  Body {ft}: ({pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f})")
    
    # Check distances between fingertips and object
    print("\n=== Fingertip to object distances ===")
    for env in range(4):
        start = env * bodies_per_env
        obj_pos = bq[start + bodies_per_env - 1][:3]
        crashed = "CRASHED" if env == 1 else ""
        print(f"Env {env} {crashed}:")
        for ft in [14, 19, 24, 29]:
            ft_pos = bq[start + ft][:3]
            dist = np.linalg.norm(ft_pos - obj_pos)
            print(f"  Body {ft} to obj: {dist:.4f}")
    
    # Check post-crash solver iterations
    post = data['post_crash_state']
    print(f"\n=== Post-crash solver_niter: {post['solver_niter']} ===")


def extract_both_states():
    """Extract crashing (env 1) and non-crashing (env 0) states for comparison."""
    
    with open('crash_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    pre_crash = data['history'][-1]
    bq = pre_crash['body_q']
    jq = pre_crash['joint_q']
    action = pre_crash['action']
    
    bodies_per_env = bq.shape[0] // 4
    joints_per_env = jq.shape[0] // 4
    action_per_env = action.shape[1] if len(action.shape) > 1 else action.shape[0] // 4
    
    # Extract env 0 (non-crashing) and env 1 (crashing)
    for env, label in [(0, "env0_ok"), (1, "env1_crash")]:
        env_data = {
            'body_q': bq[env*bodies_per_env:(env+1)*bodies_per_env],
            'joint_q': jq[env*joints_per_env:(env+1)*joints_per_env],
            'action': action[:, env*action_per_env:(env+1)*action_per_env] if len(action.shape) > 1 else action[env*action_per_env:(env+1)*action_per_env],
        }
        
        with open(f'{label}_state.pkl', 'wb') as f:
            pickle.dump(env_data, f)
        
        print(f"Saved {label}_state.pkl")


def print_state_diff():
    """Print the difference between crashing and non-crashing env states."""
    
    with open('crash_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    pre_crash = data['history'][-1]
    bq = pre_crash['body_q']
    jq = pre_crash['joint_q']
    
    bodies_per_env = bq.shape[0] // 4
    joints_per_env = jq.shape[0] // 4
    
    # Compare env 0 (ok) vs env 1 (crashed)
    env0_bq = bq[0:bodies_per_env]
    env1_bq = bq[bodies_per_env:2*bodies_per_env]
    
    env0_jq = jq[0:joints_per_env]
    env1_jq = jq[joints_per_env:2*joints_per_env]
    
    print("=== Body position differences (env1 - env0) ===")
    # Subtract env offset (env 1 is at y=1.5, env 0 at y=-1.5)
    # Actually let's check the env offsets
    print(f"Env 0 body 0 pos: {env0_bq[0][:3]}")
    print(f"Env 1 body 0 pos: {env1_bq[0][:3]}")
    
    # The offset is (0, 3, 0) from env 0 to env 1
    env_offset = env1_bq[0][:3] - env0_bq[0][:3]
    print(f"Env offset: {env_offset}")
    
    print("\n=== Body positions relative to env origin ===")
    for i in range(bodies_per_env):
        env0_rel = env0_bq[i][:3] - env0_bq[0][:3]
        env1_rel = env1_bq[i][:3] - env1_bq[0][:3]
        diff = np.linalg.norm(env1_rel - env0_rel)
        if diff > 0.01:  # Only show significant differences
            print(f"Body {i}: env0=({env0_rel[0]:7.4f}, {env0_rel[1]:7.4f}, {env0_rel[2]:7.4f}) "
                  f"env1=({env1_rel[0]:7.4f}, {env1_rel[1]:7.4f}, {env1_rel[2]:7.4f}) diff={diff:.4f}")


if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    
    if mode == "analyze":
        analyze_crash_data()
    elif mode == "extract":
        extract_both_states()
    elif mode == "diff":
        print_state_diff()
    else:
        print(f"Usage: python {sys.argv[0]} [analyze|extract|diff]")
