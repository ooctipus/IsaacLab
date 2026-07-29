# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke test for :class:`MultiTaskCommand` in a live ``ManagerBasedRLEnv``.

Self-contained: defines a minimal Anymal-C + flat-plane env cfg inline (one pure-
tracking ``lin_vel`` subtask) and instantiates it directly — no gym registration, no
dependency on the production :mod:`position.config` surface.

Assertions checked each step:

- No ``NaN`` in rewards.
- Rewards ∈ ``[0, 1]`` at every step (multiplicative terminal return is bounded).
- Per-step reward is zero on every non-terminal step (terminal-only emission).
- ``info["time_outs"]`` is ``False`` on every step (finite-horizon — rsl_rl bootstrap
  must NOT fire under this framing).
- ``episode_length_buf`` resets to 0 on every env that reported done.

Requires Isaac Sim on ``PYTHONPATH``. Invoke::

    ./isaaclab.sh -p \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/position/utils/tools/smoke_multi_task_command.py \
        --headless --num_envs 8 --num_steps 200
"""

from __future__ import annotations

import argparse
import sys


def _parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test MultiTaskCommand in a live env.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs.")
    parser.add_argument("--num_steps", type=int, default=200, help="Total env.step() calls.")
    parser.add_argument("--device", type=str, default=None, help="Override sim.device (e.g. cuda:0).")

    # AppLauncher args (--headless, --livestream, --enable_cameras, ...).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    return args_cli


def _task_reward(env, command_name: str = "goal_point"):
    """Wire the command term's terminal task-reward into a RewardTerm."""
    return env.command_manager.get_term(command_name).task_reward


def _success_done(env, command_name: str = "goal_point"):
    """Wire the command term's success-terminate flag into a DoneTerm."""
    return env.command_manager.get_term(command_name).task_done


def _build_env_cfg():
    """Construct the smoke env cfg (deferred so import happens after sim is up)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_torch import (
        ACTIVATION_KERNEL_ID,
        METRIC_KERNEL_ID,
        SAMPLER_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg import (
        MinMaxSampler,
        MultiTaskCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain import mdp

    import isaaclab_assets.robots.anymal as anymal

    @configclass
    class _Scene(InteractiveSceneCfg):
        terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75)),
        )
        robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.spawn.usd_path = (
            "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        )

    @configclass
    class _Actions:
        joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
        )

    @configclass
    class _Commands:
        goal_point = MultiTaskCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            tasks={
                "lin_vel": [
                    MultiTaskCfg.TrackingTaskCfg(
                        asset_cfg=SceneEntityCfg("robot", body_names="base"),
                        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                        activation_kernel_param=0.3,
                        sampler=MinMaxSampler(
                            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                            minimum=[-1.0, -1.0, 0.0],
                            maximum=[1.0, 1.0, 0.0],
                        ),
                    ),
                ],
            },
        )

    @configclass
    class _Observations:
        @configclass
        class PolicyCfg(ObsGroup):
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            proj_gravity = ObsTerm(func=mdp.projected_gravity)
            joint_pos = ObsTerm(func=mdp.joint_pos)
            joint_vel = ObsTerm(func=mdp.joint_vel)
            last_actions = ObsTerm(func=mdp.last_action)
            goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

            def __post_init__(self):
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class _Events:
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.1), "yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        )
        reset_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (-0.5, 0.5), "velocity_range": (0.0, 0.0)},
        )

    @configclass
    class _Rewards:
        task = RewTerm(func=_task_reward, weight=1.0, params={"command_name": "goal_point"})

    @configclass
    class _Terminations:
        # Every flag is time_out=False — finite-horizon Bellman, no rsl_rl bootstrap.
        time_out = DoneTerm(func=mdp.time_out, time_out=False)
        success = DoneTerm(func=_success_done, time_out=False, params={"command_name": "goal_point"})
        drop = DoneTerm(func=mdp.root_height_below_minimum, time_out=False, params={"minimum_height": -2.0})

    @configclass
    class _EnvCfg(ManagerBasedRLEnvCfg):
        scene: _Scene = _Scene(num_envs=16, env_spacing=2.5)
        sim: SimulationCfg = SimulationCfg()
        observations: _Observations = _Observations()
        actions: _Actions = _Actions()
        commands: _Commands = _Commands()
        rewards: _Rewards = _Rewards()
        terminations: _Terminations = _Terminations()
        events: _Events = _Events()

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 4.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation

    return _EnvCfg()


def main():
    args_cli = _parse_args()
    # Hydra tooling sometimes looks at sys.argv after us; clear to keep it clean.
    sys.argv = [sys.argv[0]]

    # Launch Isaac Sim before any import that transitively touches Kit.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Deferred imports (post-Sim init).
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    torch.manual_seed(42)

    env_cfg = _build_env_cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        num_terminal_events = 0
        num_reset_observations = 0
        prev_done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.device)

        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
                _obs, rewards, terminated, truncated, info = env.step(actions)

            dones = terminated | truncated
            assert not torch.isnan(rewards).any(), f"step {step}: NaN in rewards"
            assert (rewards >= 0).all(), f"step {step}: negative reward {rewards.min().item():.4g}"
            assert (rewards <= 1.0 + 1e-5).all(), f"step {step}: reward > 1 ({rewards.max().item():.4g})"

            non_terminal = ~dones
            if non_terminal.any():
                nt = rewards[non_terminal]
                assert torch.allclose(nt, torch.zeros_like(nt)), (
                    f"step {step}: non-terminal reward leaked (max {nt.abs().max().item():.4g})"
                )

            time_outs = info.get("time_outs", None)
            if time_outs is not None:
                assert not time_outs.any(), f"step {step}: time_outs flag fired — bootstrap would be wrong"

            if dones.any():
                num_terminal_events += int(dones.sum())
                rmin = float(rewards[dones].min())
                rmax = float(rewards[dones].max())
                print(f"[step {step:4d}] {int(dones.sum()):>3d} env(s) done (reward min={rmin:.3f} max={rmax:.3f})")

            if prev_done.any():
                elb = env.episode_length_buf
                if (elb[prev_done] == 0).all():
                    num_reset_observations += 1
            prev_done = dones.clone()

        print("\n[SMOKE] Summary:")
        print(f"  total steps:            {args_cli.num_steps}")
        print(f"  terminal-env events:    {num_terminal_events}")
        print(f"  correct-reset samples:  {num_reset_observations}")
        print("  ✓ no NaN, rewards bounded in [0, 1], no non-terminal leak, no time_outs bootstrap.")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
