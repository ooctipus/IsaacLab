# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step-5: sim round-trip -- the factory Newton-IK <-> live Isaac kinematic check.

Boots the live Factory env and validates that the offline Newton model used by
the IK builder is kinematically IDENTICAL to the live Isaac articulation: it
reads the env's live robot joint config, reorders it Isaac->Newton, runs Newton
FK, and asserts the Newton-FK end-effector pose matches the live Isaac EE pose
(both in the robot base frame). If they agree, every Newton-IK solution (Step 3,
solved to a target in the Newton model) lands the live Isaac EE on that same
target -- i.e. the offline IK transfers faithfully to the sim. This is the gate
before swapping _precollect_state_table (Step 6).

Read-only (no write/step), so it is robust to the write-then-refresh sim API.
NOTE: warp/newton/the IK prototypes are imported INSIDE the function, after
launch_simulation has started the Kit app -- importing them before the app boots
crashes the interpreter (matches the live Factory inspection script's deferred imports).

Run:
  SCRIPT=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/roundtrip_newton_ik.py
  ./isaaclab.sh -p $SCRIPT --headless --num_envs 32 presets=franka,nut_thread_m4
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Factory Newton-IK <-> Isaac kinematic round-trip.")
parser.add_argument("--task", type=str, default="Isaac-Factory-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

# Newton joint-coordinate order for the fixed-base Franka model (from the gate):
# arm coords 0..6 then the two prismatic fingers (coords 7,8).
NEWTON_COORD_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped
        try:
            _roundtrip(env)
        finally:
            env.close()


def _roundtrip(env) -> None:
    # deferred imports: warp/newton must init under the running Kit app
    import warp as wp
    from newton._src.sim.ik.ik_common import eval_fk_batched
    from prototype_factory_grasp_sampled_ik import _resolve_bodies
    from prototype_factory_ik_solve import EE_BODY, compose_model

    import isaaclab.utils.math as math_utils

    device = env.device
    robot = env.scene["robot"]
    isaac_joint_names = list(robot.joint_names)
    ee_isaac = robot.body_names.index(EE_BODY)
    n_env = env.num_envs
    print(f"[rt] env: {n_env} envs; EE '{EE_BODY}' isaac body idx={ee_isaac}")
    print(f"[rt] isaac joints = {isaac_joint_names}")

    model, held_body = compose_model()
    ee_newton, _, _ = _resolve_bodies(model)
    print(f"[rt] newton EE body idx={ee_newton}, coords={model.joint_coord_count}")

    # map each Newton coord -> live Isaac joint index, then gather the live config
    n2i = [isaac_joint_names.index(nm) for nm in NEWTON_COORD_NAMES]
    assert len(n2i) == model.joint_coord_count, (len(n2i), model.joint_coord_count)
    isaac_q = wp.to_torch(robot.data.joint_pos)  # [n_env, n_joints] live config
    newton_jq = isaac_q[:, n2i].contiguous()  # [n_env, 9] in Newton coord order

    # Newton FK of the live config -> EE pose in the Newton base frame (base at origin)
    body_q = wp.zeros((n_env, model.body_count), dtype=wp.transformf, device=device)
    eval_fk_batched(
        model,
        wp.from_torch(newton_jq),
        wp.zeros((n_env, model.joint_dof_count), dtype=wp.float32, device=device),
        body_q,
        wp.zeros((n_env, model.body_count), dtype=wp.spatial_vectorf, device=device),
    )
    bq = wp.to_torch(body_q).view(n_env, model.body_count, 7)
    newton_ee_pos, newton_ee_quat = bq[:, ee_newton, :3], bq[:, ee_newton, 3:7]

    # live Isaac EE pose expressed in the robot base frame
    base_pos = wp.to_torch(robot.data.root_pos_w)
    base_quat = wp.to_torch(robot.data.root_quat_w)
    ee_pos_w = wp.to_torch(robot.data.body_link_pos_w)[:, ee_isaac]
    ee_quat_w = wp.to_torch(robot.data.body_link_quat_w)[:, ee_isaac]
    isaac_ee_pos, isaac_ee_quat = math_utils.subtract_frame_transforms(base_pos, base_quat, ee_pos_w, ee_quat_w)

    # compare (both in base frame)
    pos_err = (isaac_ee_pos - newton_ee_pos).norm(dim=-1)
    rot_err = 2.0 * torch.acos((isaac_ee_quat * newton_ee_quat).sum(-1).abs().clamp(max=1.0))
    print("\n=== Newton-FK EE  vs  live Isaac EE (robot base frame) ===")
    print(f"  pos err [mm]:  mean={pos_err.mean() * 1e3:.3f}  max={pos_err.max() * 1e3:.3f}")
    rmean, rmax = np.degrees(rot_err.mean().item()), np.degrees(rot_err.max().item())
    print(f"  rot err [deg]: mean={rmean:.3f}  max={rmax:.3f}")

    ok = pos_err.max() < 1.5e-3 and rot_err.max() < np.radians(1.5)
    print(
        "\n[rt] "
        + (
            "PASS: Newton model is kinematically identical to the live Isaac articulation "
            "(joint reorder correct) -> Newton-IK solutions transfer to the sim."
            if ok
            else "FAIL: Newton/Isaac EE mismatch -- check joint reorder or model fidelity."
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
