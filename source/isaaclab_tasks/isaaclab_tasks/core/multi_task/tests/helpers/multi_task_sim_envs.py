# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small live-sim environment builders for multi-task command tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def pin_command_task_samples(cmd, task_names: list[str]) -> torch.Tensor:
    """Pin each environment to the corresponding command task name.

    Args:
        cmd: Command term with ``spec.task_names`` and ``task_samples``.
        task_names: Task name for each environment, ordered by environment id.

    Returns:
        Tensor of pinned task ids on the command device.
    """
    import torch

    pinned_ids = torch.tensor(
        [cmd.spec.task_names.index(task_name) for task_name in task_names],
        device=cmd.task_samples.device,
        dtype=cmd.task_samples.dtype,
    )

    def _pinned_resample(env_ids: torch.Tensor) -> None:
        cmd.task_samples[env_ids] = pinned_ids[env_ids]

    cmd.resample_indices = _pinned_resample
    return pinned_ids


def write_robot_standing_state(env, height: float = 0.55):
    """Write a nominal standing robot state at each environment origin.

    Args:
        env: Live manager-based environment.
        height: Base height above each environment origin [m].

    Returns:
        The robot articulation from ``env.scene``.
    """
    import torch
    import warp as wp

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    rest_pose = torch.zeros(env.num_envs, 7, device=env.device)
    rest_pose[:, :3] = env.scene.env_origins
    rest_pose[:, 2] += height
    rest_pose[:, 6] = 1.0

    robot.write_root_pose_to_sim_index(root_pose=rest_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(env.num_envs, 6, device=env.device),
        env_ids=env_ids,
    )
    default_joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
    default_joint_vel = wp.to_torch(robot.data.default_joint_vel)
    robot.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_joint_vel), env_ids=env_ids)
    return robot


def make_minimal_multi_task_env_cfg():
    """Build the smallest live env cfg that exercises :class:`MultiTaskCommand` end-to-end."""
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

    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.kernel_ids import (
        ACTIVATION_KERNEL_ID,
        METRIC_KERNEL_ID,
        SAMPLER_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.multi_task_cfg import (
        MinMaxSampler,
        MultiTaskCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain import mdp

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


def make_heterogeneous_multi_task_env_cfg(dispatch_backend: str = "torch"):
    """Build a four-env live cfg with one pinned command shape per environment."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.kernel_ids import (
        ACTIVATION_KERNEL_ID,
        METRIC_KERNEL_ID,
        SAMPLER_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.multi_task_cfg import (
        MinMaxSampler,
        MultiTaskCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain import mdp

    import isaaclab_assets.robots.anymal as anymal

    def _task_reward(env, command_name="goal_point"):
        return env.command_manager.get_term(command_name).task_reward

    def _success_done(env, command_name="goal_point"):
        return env.command_manager.get_term(command_name).task_done

    base_entity = SceneEntityCfg("robot", body_names="base")
    standing_z = 0.55

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
            resampling_time_range=(100.0, 100.0),
            debug_vis=False,
            dispatch_backend=dispatch_backend,
            tasks={
                "pure_track_zero_vel": [
                    MultiTaskCfg.TrackingTaskCfg(
                        asset_cfg=base_entity,
                        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                        activation_kernel_param=0.3,
                        sampler=MinMaxSampler(
                            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                            minimum=[0.0, 0.0, 0.0],
                            maximum=[0.0, 0.0, 0.0],
                        ),
                    ),
                ],
                "pure_instant_at_spawn": [
                    MultiTaskCfg.InstantaneousTaskCfg(
                        asset_cfg=base_entity,
                        state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                        activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                        activation_kernel_param=0.1,
                        sampler=MinMaxSampler(
                            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                            minimum=[0.0, 0.0, standing_z],
                            maximum=[0.0, 0.0, standing_z],
                        ),
                    ),
                ],
                "mixed_reach_and_hold": [
                    MultiTaskCfg.InstantaneousTaskCfg(
                        asset_cfg=base_entity,
                        state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                        activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                        activation_kernel_param=0.1,
                        sampler=MinMaxSampler(
                            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                            minimum=[0.0, 0.0, standing_z],
                            maximum=[0.0, 0.0, standing_z],
                        ),
                    ),
                    MultiTaskCfg.TrackingTaskCfg(
                        asset_cfg=base_entity,
                        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                        activation_kernel_param=0.3,
                        sampler=MinMaxSampler(
                            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                            minimum=[0.0, 0.0, 0.0],
                            maximum=[0.0, 0.0, 0.0],
                        ),
                    ),
                ],
                "pure_track_unreachable": [
                    MultiTaskCfg.TrackingTaskCfg(
                        asset_cfg=base_entity,
                        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                        activation_kernel_param=0.3,
                        sampler=MinMaxSampler(
                            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                            minimum=[10.0, 0.0, 0.0],
                            maximum=[10.0, 0.0, 0.0],
                        ),
                    ),
                ],
            },
        )

    @configclass
    class _Observations:
        @configclass
        class PolicyCfg(ObsGroup):
            goal_reach = ObsTerm(func=mdp.command_reach, params={"command_name": "goal_point"})
            goal_track = ObsTerm(func=mdp.command_track, params={"command_name": "goal_point"})
            goal_active = ObsTerm(func=mdp.command_active, params={"command_name": "goal_point"})
            goal_progress = ObsTerm(func=mdp.command_progress, params={"command_name": "goal_point"})

            def __post_init__(self):
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class _Events:
        pass

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
            self.episode_length_s = 0.5
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            self.rewards.task.weight = 1.0 / (self.sim.dt * self.decimation)

    return _EnvCfg()
