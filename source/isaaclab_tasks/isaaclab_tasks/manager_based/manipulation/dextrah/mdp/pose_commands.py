# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from .collision_analyzer_cfg import CollisionAnalyzerCfg
from .success_monitor_cfg import SuccessMonitorCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import pose_commands_cfg as dex_cmd_cfgs


class PoseAlignmentCommandChoice(CommandTerm):
    
    cfg: dex_cmd_cfgs.PoseAlignmentCommandChoiceCfg
    
    def __init__(self, cfg: dex_cmd_cfgs.PoseAlignmentCommandChoiceCfg, env: ManagerBasedEnv):
        self.terms: list[CommandTerm] = []
        self.term_names: list[str] = []
        self.sampling_strategy = cfg.sampling_strategy
        self.root_asset: RigidObject | Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.pos_only = True
        for term_name, term_cfg in cfg.terms.items():
            term_cfg.resampling_time_range = cfg.resampling_time_range
            self.terms.append(term_cfg.class_type(term_cfg, env))
            self.term_names.append(term_name)

        super().__init__(cfg, env)        
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.int, device=env.device)
        self.last_term_samples = torch.zeros((env.num_envs,), dtype=torch.int, device=env.device)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=len(self.terms),
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self._command = torch.zeros((env.num_envs, 7), device=self.device)
        self.success_recorder = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        
        for term_name in self.term_names:
            self.metrics[f"position_error_{term_name}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"orientation_error_{term_name}"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self):
        msg = "TERM CHOICES:"
        for term_cfg in self.terms:
            msg += term_cfg.__str__()


    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor):
        self.last_term_samples[env_ids] = self.term_samples[env_ids]
        if self.sampling_strategy == "uniform":
            self.term_samples[env_ids] = torch.randint(0, len(self.terms), (len(env_ids),), device=self.device, dtype=self.term_samples.dtype)
        else:
            self.term_samples[env_ids] = self.success_monitor.failure_rate_sampling(env_ids)

        for i, term in enumerate(self.terms):
            term_id_mask = self.term_samples[env_ids] == i
            term_ids = env_ids[term_id_mask]
            if term_ids.numel() > 0:
                term._resample_command(term_ids)


    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        if torch.any(reset_env):
            success_mask = torch.where(self.success_recorder[reset_env], True, False)
            reset_term_samples = self.last_term_samples[reset_env]
            self.success_monitor.success_update(reset_term_samples, success_mask)
        cmd_succ_rate = self.success_monitor.get_success_rate()
        log = {f"Metrics/task_success/{self.term_names[i]}": cmd_succ_rate[i].item() for i in range(len(self.terms))}
        for i, term in enumerate(self.terms):
            term._update_metrics()
            term_id_mask = self.term_samples == i
            name = self.term_names[i]
            term_pos_error = term.metrics["position_error"][term_id_mask]
            term_rot_error = term.metrics["orientation_error"][term_id_mask]
            self.metrics[f"position_error_{name}"][term_id_mask] = term_pos_error
            self.metrics[f"orientation_error_{name}"][term_id_mask] = term_rot_error
            if self.pos_only:
                self.success_recorder[term_id_mask] = term_pos_error < 0.05
            else:
                self.success_recorder[term_id_mask] = (term_pos_error < 0.05) & (term_rot_error < 0.4)
        self._env.extras['log'].update(log)

    def _update_command(self):
        for term in self.terms:
            term._update_command()

        for i, term in enumerate(self.terms):
            term_id_mask = self.term_samples == i
            self._command[term_id_mask] = term.command[term_id_mask]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.root_asset.is_initialized:
            return
        
        pos_command_w, quat_command_w = combine_frame_transforms(
            self.root_asset.data.root_pos_w,
            self.root_asset.data.root_quat_w,
            self._command[:, :3],
            self._command[:, 3:],
        )
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(pos_command_w, quat_command_w)
        # -- current body pose
        object_pose_w = self.object.data.root_pose_w
        self.current_pose_visualizer.visualize(object_pose_w[:, :3], object_pose_w[:, 3:7])


class ObjectUniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        root_state_w = self.object.data.root_state_w
        self.current_pose_visualizer.visualize(root_state_w[:, :3], root_state_w[:, 3:7])



class ObjectUniformTableTopCollisionFreePoseCommand(ObjectUniformPoseCommand):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: dex_cmd_cfgs.ObjectUniformTableTopCollisionFreePoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: dex_cmd_cfgs.ObjectUniformTableTopCollisionFreePoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.table: RigidObject = env.scene[cfg.table_name]
        self.precollecting_phase = True
        self.valid_samples, self.valid_sample_counts = self._collect_candidate_command_b()
        self.precollecting_phase = False

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        pos, quat = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )
        return torch.cat((pos, quat), dim=1)

    """
    Implementation specific functions.
    """
    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        if self.precollecting_phase:
            super()._resample_command(env_ids)
        else:
            # how many valid samples each env actually has (>=1 guaranteed)
            counts = self.valid_sample_counts[env_ids]  # shape (len(env_ids),)
            rand_f = torch.rand((len(env_ids),), device=self.device)
            rand_idx = torch.floor(rand_f * counts.to(dtype=rand_f.dtype)).to(torch.long)
            sampled_command_b = self.valid_samples[env_ids, rand_idx]
            self.pose_command_b[env_ids] = sampled_command_b
    
    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.table.data.root_pos_w,
            self.table.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


    def _collect_candidate_command_b(self):
        collision_analyzer_cfg = CollisionAnalyzerCfg(
            num_points=32,
            max_dist=0.5,
            min_dist=0.25,
            asset_cfg=SceneEntityCfg(self.cfg.object_name),
            obstacle_cfgs=[SceneEntityCfg(self.cfg.table_name)]
        )
        collision_analyzer = collision_analyzer_cfg.class_type(collision_analyzer_cfg, self._env)
        valid_tensor = torch.zeros((self._env.num_envs, self.cfg.num_samples, 7), device=self.device)
        env_arange = torch.arange(self._env.num_envs, dtype=torch.int , device=self.device)
        for i in range(self.cfg.num_samples):
            self._resample_command(env_arange)
            valid_tensor[:, i] = self.pose_command_b

        valid_sample_counts = torch.zeros((self._env.num_envs,), device=self.device, dtype=torch.int)
        valid_ptr = torch.zeros((self._env.num_envs,), device=self.device, dtype=torch.int)
        total_needed = int(self._env.num_envs * self.cfg.num_samples)
        pbar = tqdm(total=total_needed, desc="collect candidate commands", unit="sample")
        prev_have = 0
        # this commented loop option may take long long time
        while torch.any(valid_sample_counts < self.cfg.num_samples):
        # for i in range(2 * self.cfg.num_samples):
            if torch.all(valid_sample_counts >= self.cfg.num_samples):
                break
            left_env_ids = torch.arange(self._env.num_envs, device=self.device)[valid_sample_counts < self.cfg.num_samples]
            self._resample_command(left_env_ids)
            pos_command_w, quat_command_w = combine_frame_transforms(
                self.table.data.root_pos_w[left_env_ids],
                self.table.data.root_quat_w[left_env_ids],
                self.pose_command_b[left_env_ids, :3],
                self.pose_command_b[left_env_ids, 3:],
            )
            self.object.write_root_pose_to_sim(torch.cat((pos_command_w, quat_command_w), dim=1), env_ids=left_env_ids)

            collision_free_mask = collision_analyzer(self._env, left_env_ids)
            valid_env_ids = left_env_ids[collision_free_mask]
            idx = valid_ptr[valid_env_ids]
            pos_candidate_b, quat_candidate_b = self.pose_command_b[valid_env_ids, :3], self.pose_command_b[valid_env_ids, 3:]
            valid_tensor[valid_env_ids, idx] = torch.cat((pos_candidate_b, quat_candidate_b), dim=1)
            valid_ptr[valid_env_ids] = (idx + 1) % self.cfg.num_samples
            valid_sample_counts[valid_env_ids] = torch.clamp(valid_sample_counts[valid_env_ids] + 1, max=self.cfg.num_samples)

            # update progress bar
            current_have = int(valid_sample_counts.sum().item())
            delta = current_have - prev_have
            if delta > 0:
                pbar.update(delta)
                prev_have = current_have
        pbar.close()
        print("Min Sample Collected: ", torch.amin(valid_sample_counts).item(), "Max Sample Collected: ", torch.amax(valid_sample_counts).item())
        self._env.scene.reset()
        return valid_tensor, valid_sample_counts


class ObjectUniformTableTopRestPoseCommand(ObjectUniformTableTopCollisionFreePoseCommand):

    cfg: dex_cmd_cfgs.ObjectUniformTableTopRestPoseCommandCfg
    """Configuration for the command generator."""

    def _collect_candidate_command_b(self):
        collision_analyzer_cfg = CollisionAnalyzerCfg(
            num_points=32,
            max_dist=0.5,
            min_dist=0.1,
            asset_cfg=SceneEntityCfg(self.cfg.object_name),
            obstacle_cfgs=[SceneEntityCfg(self.cfg.table_name)]
        )
        collision_analyzer = collision_analyzer_cfg.class_type(collision_analyzer_cfg, self._env)
        valid_tensor = torch.zeros((self._env.num_envs, self.cfg.num_samples, 7), device=self.device)
        valid_sample_counts = torch.zeros((self._env.num_envs,), device=self.device, dtype=torch.int)
        valid_ptr = torch.zeros((self._env.num_envs,), device=self.device, dtype=torch.int)
        self._env.scene.reset()
        pos = self._env.scene["plane"].get_world_poses()[0]
        self._env.scene["plane"].set_world_poses(torch.tensor([[0.0, 0.0, -10.0]]))
        
        total_needed = int(self._env.num_envs * self.cfg.num_samples)
        pbar = tqdm(total=total_needed, desc="collect candidate commands", unit="sample")
        prev_have = 0
        # this commented loop option may take long long time
        # while torch.any(valid_sample_counts < self.cfg.num_samples):
        count = 0
        while (count <= 2 * self.cfg.num_samples) or torch.any(valid_sample_counts < 2):
            left_env_ids = torch.arange(self._env.num_envs, device=self.device)[valid_sample_counts < self.cfg.num_samples]
            self._resample_command(left_env_ids)
            pos_command_w, quat_command_w = combine_frame_transforms(
                self.table.data.root_pos_w[left_env_ids],
                self.table.data.root_quat_w[left_env_ids],
                self.pose_command_b[left_env_ids, :3],
                self.pose_command_b[left_env_ids, 3:],
            )
            self.object.write_root_pose_to_sim(torch.cat((pos_command_w, quat_command_w), dim=1), env_ids=left_env_ids)
            collision_free = collision_analyzer(self._env, left_env_ids)
            half_sec_steps = int(1 / self._env.sim.get_physics_dt())
            for i in range(half_sec_steps):
                self._env.sim.step(render=False)
                self._env.scene.update(dt=self._env.sim.get_physics_dt())
            rest = torch.sum(self.object.data.root_vel_w[left_env_ids], dim=1) < 1.0e-2
            didnotdrop = self.object.data.root_pos_w[left_env_ids, 2] > 0.0
            valid_env_ids = left_env_ids[rest & didnotdrop & collision_free]
            idx = valid_ptr[valid_env_ids]
            
            pos_candidate_b, quat_candidate_b = subtract_frame_transforms(
                self.table.data.root_pos_w[valid_env_ids],
                self.table.data.root_quat_w[valid_env_ids],
                self.object.data.root_pos_w[valid_env_ids],
                self.object.data.root_quat_w[valid_env_ids]
            )
            valid_tensor[valid_env_ids, idx] = torch.cat((pos_candidate_b, quat_candidate_b), dim=1)
            valid_ptr[valid_env_ids] = (idx + 1) % self.cfg.num_samples
            valid_sample_counts[valid_env_ids] = torch.clamp(valid_sample_counts[valid_env_ids] + 1, max=self.cfg.num_samples)
            count += 1
            # update progress bar
            current_have = int(valid_sample_counts.sum().item())
            delta = current_have - prev_have
            if delta > 0:
                pbar.update(delta)
                prev_have = current_have
        pbar.close()
        print("Min Sample Collected: ", torch.amin(valid_sample_counts).item(), "Max Sample Collected: ", torch.amax(valid_sample_counts).item())
        self._env.scene["plane"].set_world_poses(pos)
        self._env.scene.reset()
        return valid_tensor, valid_sample_counts