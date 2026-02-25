# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab_physx.assets import SurfaceGripperCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mixin_utils import (
    PerEnvArticulation,
    PerEnvRigidObject,
    PerEnvSurfaceGripper,
    get_per_env_action_class,
)
from isaaclab_contrib.tasks.manipulation.multitask.multitask_utils import MultiTaskRegistryConfig


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # Table is added per-group in _setup_group_assets() so tasks without a table (e.g. Open-Drawer) do not get one.
    # Shared ground plane: height is set from the first task in _setup_shared_ground_plane_and_offsets();
    # other tasks with different plane height have their assets' init_state shifted so they align with this plane.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ProxyObservationsCfg:
    """Minimal observations for debugging asset loading."""

    @configclass
    class PolicyCfg(ObsGroup):
        sim_time_s = ObsTerm(func=mdp.current_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ProxyActionsCfg:
    """Minimal actions for debugging asset loading."""

    arm_action: mdp.JointPositionActionCfg | None = None
    gripper_action: mdp.BinaryJointPositionActionCfg | None = None


@configclass
class ProxyRewardsCfg:
    """Minimal rewards for debugging asset loading."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class ProxyTerminationsCfg:
    """Minimal terminations for debugging asset loading."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ProxyEventsCfg:
    """Minimal events for debugging asset loading."""

    reset_scene_to_default = EventTerm(func=mdp.reset_multitask_scene_to_default, mode="reset")


@configclass
class SingleRobotMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Single robot multitask env that composes per-task configs at runtime.
    Multi-task shared the same robot and actions.
    """

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)
    tasks: MultiTaskRegistryConfig = MISSING
    actions: ProxyActionsCfg = ProxyActionsCfg()
    observations: ProxyObservationsCfg = ProxyObservationsCfg()
    rewards: ProxyRewardsCfg = ProxyRewardsCfg()
    terminations: ProxyTerminationsCfg = ProxyTerminationsCfg()
    events: ProxyEventsCfg = ProxyEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # setup the simulation parameters
        self.decimation = 3
        self.episode_length_s = 26.0
        self.sim.dt = 1 / 60
        self.sim.render_interval = 3

        # setup the tasks
        if self.tasks is MISSING:
            raise ValueError("tasks must be set to TaskRegistryMultiTaskConfig.")

        # Shared ground plane uses first task's height; record per-group z offset for other tasks
        self._setup_shared_ground_plane_and_offsets()

        # setup the shared robots and actions
        self._setup_group_robots()
        self._setup_group_actions()

        # setup the group assets, sensors, events, terminations, and observations
        self._setup_group_assets()

        self._setup_group_events()
        self._setup_group_terminations()
        self._setup_group_observations()
        self._setup_group_rewards()

    def _setup_shared_ground_plane_and_offsets(self):
        """Use the first task that has a plane to define shared plane pose and ref_plane_z.
        ref_robot_z from the first task that has a robot.
        Per-group offset_z: if task has plane, use ref_plane_z - task_plane_z; else use ref_robot_z - task_robot_z.
        """
        first_plane_cfg = None
        ref_plane_z: float = 0.0
        ref_robot_z: float = 0.0
        for group_idx in range(self.tasks.total_groups):
            task_name = self.tasks.get_task_name_for_group(group_idx)
            task_cfg = self.tasks.get_task_cfg(task_name)
            plane_cfg = getattr(task_cfg.scene, "plane", None)
            if plane_cfg is not None and isinstance(plane_cfg, AssetBaseCfg):
                first_plane_cfg = plane_cfg
                pos = getattr(plane_cfg.init_state, "pos", (0.0, 0.0, 0.0))
                ref_plane_z = float(pos[2]) if len(pos) >= 3 else 0.0
                break
        for group_idx in range(self.tasks.total_groups):
            task_name = self.tasks.get_task_name_for_group(group_idx)
            task_cfg = self.tasks.get_task_cfg(task_name)
            robot_cfg = getattr(task_cfg.scene, "robot", None)
            if robot_cfg is not None and isinstance(robot_cfg, ArticulationCfg):
                pos = getattr(robot_cfg.init_state, "pos", (0.0, 0.0, 0.0))
                ref_robot_z = float(pos[2]) if len(pos) >= 3 else 0.0
                break
        if first_plane_cfg is not None:
            self.scene.plane = self.tasks.clone_cfg(
                self.scene.plane,
                init_state=first_plane_cfg.init_state,
            )
        self._group_ground_offset_z: list[float] = []
        for group_idx in range(self.tasks.total_groups):
            task_name = self.tasks.get_task_name_for_group(group_idx)
            task_cfg = self.tasks.get_task_cfg(task_name)
            plane_cfg = getattr(task_cfg.scene, "plane", None)
            if plane_cfg is not None and isinstance(plane_cfg, AssetBaseCfg):
                pos = getattr(plane_cfg.init_state, "pos", (0.0, 0.0, 0.0))
                task_z = float(pos[2]) if len(pos) >= 3 else 0.0
                offset_z = ref_plane_z - task_z
            else:
                robot_cfg = getattr(task_cfg.scene, "robot", None)
                if robot_cfg is not None and isinstance(robot_cfg, ArticulationCfg):
                    pos = getattr(robot_cfg.init_state, "pos", (0.0, 0.0, 0.0))
                    task_z = float(pos[2]) if len(pos) >= 3 else 0.0
                else:
                    task_z = 0.0
                offset_z = ref_robot_z - task_z
            self._group_ground_offset_z.append(offset_z)

    def _apply_ground_offset_to_init_state(self, init_state, offset_z: float):
        """Return init_state with pos z shifted by offset_z to align with shared ground plane."""
        if init_state is None or offset_z == 0.0:
            return init_state
        pos = getattr(init_state, "pos", None)
        if pos is None:
            return init_state
        pos = (pos[0], pos[1], pos[2]) if len(pos) >= 3 else (0.0, 0.0, 0.0)
        new_pos = (pos[0], pos[1], pos[2] + offset_z)
        return self.tasks.clone_cfg(init_state, pos=new_pos)

    def _setup_group_assets(self):
        """Populate scene assets per group from registry task configs.

        Tables are added per-group; a single GroundPlane is shared, with assets shifted by _group_ground_offset_z.
        """
        for group_idx in range(self.tasks.total_groups):
            env_tuple = self.tasks.env_indices_for_group(group_idx)
            if not env_tuple:
                continue

            offset_z = self._group_ground_offset_z[group_idx]
            task_cfg = self.tasks.get_task_cfg(self.tasks.get_task_name_for_group(group_idx))

            for asset_name, asset_cfg in self.tasks.iter_scene_cfg_items(task_cfg.scene):
                if asset_name in {"robot", "ee_frame", "plane", "light"}:
                    continue
                if not isinstance(asset_cfg, (AssetBaseCfg, ArticulationCfg, RigidObjectCfg)):
                    continue
                if not self.tasks.should_group_cfg(asset_cfg):
                    continue

                prim_path = self.tasks.group_prim_from_template(env_tuple, asset_cfg.prim_path)
                adj_init = self._apply_ground_offset_to_init_state(asset_cfg.init_state, offset_z)

                # Per-env Articulations and Rigid Objects, Fixtures only revise path/init_state.
                if isinstance(asset_cfg, ArticulationCfg):
                    ClassType = PerEnvArticulation
                elif isinstance(asset_cfg, RigidObjectCfg):
                    ClassType = PerEnvRigidObject
                elif isinstance(asset_cfg, SurfaceGripperCfg):
                    ClassType = PerEnvSurfaceGripper
                elif isinstance(asset_cfg, AssetBaseCfg):
                    ClassType = None  # Fixtures don't change class type
                else:
                    raise ValueError(f"Unsupported asset type for PerEnvClass: {type(asset_cfg)}")

                cloned = self.tasks.clone_cfg(
                    asset_cfg,
                    prim_path=prim_path,
                    init_state=adj_init
                    if ClassType != PerEnvSurfaceGripper
                    else asset_cfg.init_state,  # SurfaceGripperCfg does not have init_state
                    class_type=ClassType,
                )

                cloned.assigned_envs = env_tuple
                setattr(self.scene, f"{asset_name}_group_{group_idx}", cloned)

    def _setup_group_robots(self):
        """Setup robots from the first representative task's env cfg.

        Robot init_state is shifted by _group_ground_offset_z to align with shared ground plane.
        """
        base_robot_cfg = None
        init_state_tensor = torch.zeros((self.scene.num_envs, 13), dtype=torch.float32)

        for group_idx in range(self.tasks.total_groups):
            env_tuple = self.tasks.env_indices_for_group(group_idx)
            if not env_tuple:
                continue

            offset_z = self._group_ground_offset_z[group_idx]
            task_cfg = self.tasks.get_task_cfg(self.tasks.get_task_name_for_group(group_idx))
            robot_cfg = getattr(task_cfg.scene, "robot", None)

            if not isinstance(robot_cfg, ArticulationCfg):
                continue

            if base_robot_cfg is None:
                base_robot_cfg = robot_cfg

            # Build per-env init_state tensor from original task cfg
            adj_init = self._apply_ground_offset_to_init_state(robot_cfg.init_state, offset_z)
            pos = getattr(adj_init, "pos", (0.0, 0.0, 0.0))
            rot = getattr(adj_init, "rot", (0.0, 0.0, 0.0, 1.0))  # XYZW quaternion format (IsaacLab 3.0)
            lin_vel = getattr(adj_init, "lin_vel", (0.0, 0.0, 0.0))
            ang_vel = getattr(adj_init, "ang_vel", (0.0, 0.0, 0.0))
            root_state = torch.tensor(tuple(pos) + tuple(rot) + tuple(lin_vel) + tuple(ang_vel), dtype=torch.float32)
            init_state_tensor[list(env_tuple)] = root_state.unsqueeze(0).repeat(len(env_tuple), 1)

        if base_robot_cfg is None:
            return

        setattr(self.scene, "robot", base_robot_cfg)
        self._robot_init_state_tensor = init_state_tensor

    def _setup_group_actions(self):
        """Setup actions from the first task's env cfg (arm_action, gripper_action)."""
        for group_idx in range(self.tasks.total_groups):
            task_cfg = self.tasks.get_task_cfg(self.tasks.get_task_name_for_group(group_idx))
            task_actions = getattr(task_cfg, "actions", None)
            if task_actions is not None:
                self.actions = task_actions
                break

        if self.actions is None:
            raise ValueError("No actions found in any task's env cfg.")

    def _setup_group_events(self):
        """Prepare per-task events for grouped envs."""
        # TODO: wire per-task events/terminations with env_ids filtered by group

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
class MultiRobotMultiTaskEnvCfg(SingleRobotMultiTaskEnvCfg):
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
