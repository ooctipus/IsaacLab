# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight sim-integration regression test for :class:`MultiTaskCommand`.

Skipped when Isaac Sim isn't available on ``PYTHONPATH`` (e.g. CI without sim).
Target budget: **under 20 seconds wallclock**, including Isaac Sim launch.

Covers what the mock tests can't: that the engine actually drives a live
:class:`ManagerBasedRLEnv` end-to-end with real Articulation data reads, and that the
reward/done signal behaves correctly over real physics steps.

Intentionally minimal to stay within budget:

- 4 envs (not 4096).
- Flat plane terrain (no generator, no retarget).
- Single pure-tracking ``lin_vel`` task.
- 10 physics steps with random actions.
- Only sanity invariants are asserted (no-NaN, reward ∈ [0, 1], terminal-only emission,
  ``time_outs = False``).

If budget is exceeded on the user's sim machine, this test should be deleted and the
non-sim regression suite relied on exclusively — the exact-reward-trace test in
``test_multi_task_command_mock.py`` catches everything this test does except real
Articulation wiring, which is mostly a "does the port compile against the live IsaacLab
API" check.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

# Skip the entire module if Isaac Sim isn't importable. The launch itself would fail
# with a clearer message than pytest would produce at collection time.
if importlib.util.find_spec("isaacsim") is None:
    pytest.skip("Isaac Sim not available on PYTHONPATH — skipping sim-integration test.", allow_module_level=True)


@pytest.fixture(scope="module")
def simulation_app():
    """Launch a single headless Isaac Sim app shared across all tests in this module.

    Module scope amortizes the 5-15s launch cost across every test below.
    """
    import argparse

    from isaaclab.app import AppLauncher

    # Minimal argparse Namespace — AppLauncher inspects attributes it knows.
    args = argparse.Namespace(
        headless=True,
        livestream=-1,
        enable_cameras=False,
        xr=False,
        device=None,
        cpu=False,
        verbose=False,
        info=False,
        experience="",
        kit_args="",
    )
    launcher = AppLauncher(args)
    app = launcher.app
    yield app
    app.close()


def _build_minimal_env_cfg():
    """Smallest possible env cfg exercising the MultiTaskCommand end-to-end."""
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

    from isaaclab_tasks.manager_based.locomotion.position import mdp
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.kernels import (
        ACTIVATION_KERNEL_ID,
        METRIC_KERNEL_ID,
        SAMPLER_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_cfg import (
        MinMaxSampler,
        MultiTaskCfg,
    )

    import isaaclab_assets.robots.anymal as anymal

    def _task_reward(env, command_name="goal_point"):
        return env.command_manager.get_term(command_name).task_reward

    def _success_done(env, command_name="goal_point"):
        return env.command_manager.get_term(command_name).task_done

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
            joint_pos = ObsTerm(func=mdp.joint_pos)
            goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

            def __post_init__(self):
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class _Events:
        reset_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (-0.3, 0.3), "velocity_range": (0.0, 0.0)},
        )

    @configclass
    class _Rewards:
        task = RewTerm(func=_task_reward, weight=1.0, params={"command_name": "goal_point"})

    @configclass
    class _Terminations:
        time_out = DoneTerm(func=mdp.time_out, time_out=False)
        success = DoneTerm(func=_success_done, time_out=False, params={"command_name": "goal_point"})

    @configclass
    class _EnvCfg(ManagerBasedRLEnvCfg):
        scene: _Scene = _Scene(num_envs=4, env_spacing=2.5)
        sim: SimulationCfg = SimulationCfg()
        observations: _Observations = _Observations()
        actions: _Actions = _Actions()
        commands: _Commands = _Commands()
        rewards: _Rewards = _Rewards()
        terminations: _Terminations = _Terminations()
        events: _Events = _Events()

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 2.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation

    return _EnvCfg()


def test_multi_task_command_runs_10_steps_in_live_env(simulation_app):
    """Drive the engine through 10 physics steps with random actions; assert core invariants.

    Asserted (each step):

    - No NaN in rewards.
    - Rewards ∈ ``[0, 1]`` (multiplicative terminal return is bounded).
    - Non-terminal steps produce exactly zero task reward (composer is terminal-only).
    - ``info["time_outs"]`` is False — Stage-3 finite-horizon framing, rsl_rl must not
      bootstrap.

    The weight="1.0" task reward term is the only reward. Random-action policy rarely
    matches the target velocity, so activations will be mostly low — but the invariants
    must still hold.
    """
    from isaaclab.envs import ManagerBasedRLEnv

    torch.manual_seed(0)
    env_cfg = _build_minimal_env_cfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        num_steps = 10
        for step in range(num_steps):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
                _obs, rewards, terminated, truncated, info = env.step(actions)

            assert not torch.isnan(rewards).any(), f"step {step}: NaN reward"
            assert (rewards >= 0).all(), f"step {step}: negative reward {rewards.min().item():.4g}"
            assert (rewards <= 1.0 + 1e-5).all(), f"step {step}: reward > 1 ({rewards.max().item():.4g})"

            dones = terminated | truncated
            non_terminal = ~dones
            if non_terminal.any():
                nt = rewards[non_terminal]
                assert torch.allclose(nt, torch.zeros_like(nt)), (
                    f"step {step}: non-terminal reward leaked (max abs {nt.abs().max().item():.4g})"
                )

            time_outs = info.get("time_outs", None)
            if time_outs is not None:
                assert not time_outs.any(), f"step {step}: time_outs flag set — rsl_rl bootstrap would fire"
    finally:
        env.close()
