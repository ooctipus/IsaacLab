# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_physx.assets import SurfaceGripperCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.config.franka.franka_multitask_env_cfg import FrankaMultiTaskEnvCfg
from isaaclab_contrib.tasks.manipulation.multitask.mixin_utils import PerEnvArticulation, get_per_env_action_class
from isaaclab_contrib.tasks.manipulation.multitask.multitask_cfg import MultiTaskRegistryConfig


@configclass
class MultiRobotMultiTaskEnvCfg(FrankaMultiTaskEnvCfg):
    """Multi-robot multitask env that composes per-task configs at runtime."""

    def __post_init__(self):
        super().__post_init__()

    def _setup_group_robots(self):
        """Setup group robots if tasks have heterogeneous robot configs.
        Robot init_state is shifted by _group_ground_offset_z to align with shared ground plane.
        """
        for group_idx in range(self.tasks.total_groups):
            env_tuple = self.tasks.env_indices_for_group(group_idx)
            if not env_tuple:
                continue
            offset_z = self._group_ground_offset_z[group_idx]
            task_name = self.tasks.get_task_name_for_group(group_idx)
            task_cfg = self.tasks.get_task_cfg(task_name)
            robot_cfg = getattr(task_cfg.scene, "robot", None)
            if robot_cfg is None:
                continue
            if not isinstance(robot_cfg, ArticulationCfg):
                continue
            prim_path = self.tasks.group_prim_from_template(env_tuple, robot_cfg.prim_path)
            adj_init = self._apply_ground_offset_to_init_state(robot_cfg.init_state, offset_z)
            cloned = self.tasks.clone_cfg(
                robot_cfg,
                prim_path=prim_path,
                init_state=adj_init,
                class_type=PerEnvArticulation,
            )
            cloned.assigned_envs = env_tuple
            setattr(self.scene, f"robot_group_{group_idx}", cloned)

    def _setup_group_actions(self):
        """Setup per-group actions from each task's env cfg (any number of action terms with any names)."""
        for group_idx in range(self.tasks.total_groups):
            env_tuple = self.tasks.env_indices_for_group(group_idx)
            if not env_tuple:
                continue
            robot_asset_name = f"robot_group_{group_idx}"
            task_name = self.tasks.get_task_name_for_group(group_idx)
            task_cfg = self.tasks.get_task_cfg(task_name)
            task_actions = getattr(task_cfg, "actions", None)
            if task_actions is None:
                continue

            for action_name, action_src in vars(task_actions).items():
                if action_name.startswith("_") or action_src is None:
                    continue

                per_env_class = get_per_env_action_class(action_src)
                # SurfaceGripperBinaryAction uses the SurfaceGripper asset in the scene, not the robot
                if isinstance(action_src, mdp.SurfaceGripperBinaryActionCfg):
                    action_asset_name = None
                    for scene_name, cfg in self.tasks.iter_scene_cfg_items(task_cfg.scene):
                        if isinstance(cfg, SurfaceGripperCfg):
                            action_asset_name = f"{scene_name}_group_{group_idx}"
                            break
                    if action_asset_name is None:
                        raise ValueError(
                            f"SurfaceGripperBinaryActionCfg used but no SurfaceGripper in scene for task {task_name}"
                        )
                else:
                    action_asset_name = robot_asset_name

                kw = {"asset_name": action_asset_name}
                if per_env_class is not None:
                    kw["class_type"] = per_env_class
                cloned = self.tasks.clone_cfg(action_src, **kw)
                if per_env_class is not None:
                    cloned.assigned_envs = env_tuple
                setattr(self.actions, f"{action_name}_group_{group_idx}", cloned)

    def _setup_group_events(self):
        """Prepare per-task events for grouped envs."""
        self.events.reset_robot_init_state = EventTerm(func=mdp.reset_multitask_robot_init_state, mode="reset")

    def _setup_group_terminations(self):
        """Prepare per-task terminations for grouped envs."""
        # TODO: wire per-task events/terminations with env_ids filtered by group

        pass

    def _setup_group_observations(self):
        """Prepare per-task observations for grouped envs."""
        # TODO: wire per-task observations with env_ids filtered by group

        pass

    def _setup_group_rewards(self):
        """Prepare per-task rewards for grouped envs."""
        # TODO: wire per-task rewards with env_ids filtered by group

        pass


@configclass
class MultiRobotMultiTaskManipulationEnvCfg(MultiRobotMultiTaskEnvCfg):
    """Example multi-task config using Stack/Reach/Lift/Cabinet Franka tasks."""

    def __post_init__(self):
        self.tasks = MultiTaskRegistryConfig(
            task_names_by_group=[
                "Isaac-Stack-Cube-Franka-IK-Rel-v0",
                "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
                "Isaac-Lift-Cube-Franka-IK-Rel-v0",
                "Isaac-Open-Drawer-Franka-IK-Rel-v0",
            ],
            group_size=10,
            device=self.sim.device,
        )
        super().__post_init__()
