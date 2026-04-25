# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track the delta between a target robot state and the current state.

``cmd_buf[num_envs, 3, 12 + num_joints + 1]``:

    - row 0: target state
    - row 1: delta (target - current), zeroed on inactive columns
    - row 2: current state (from sim)

State columns::

    [0:3]           position
    [3:6]           orientation (roll, pitch, yaw)
    [6:9]           linear velocity
    [9:12]          angular velocity
    [12:12+nj]      joint positions
    [12+nj]         hold / success / remaining time

Error groups: ``pos, rot, lin_vel, ang_vel, joints`` -- each reduced
to a scalar norm.  Success = all active group errors below threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from .task_table_builder import build_task_table

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RelativeStateCommandCfg


class RelativeStateCommand(CommandTerm):
    """Track the delta between a target and current robot state.

    Computes per-group error norms for 5 groups:
    ``pos(3), rot(3), lin_vel(3), ang_vel(3), joints(num_joints)``.
    Inactive groups (per ``cmd_mask``) contribute zero delta.
    """

    cfg: RelativeStateCommandCfg

    @configclass
    class TaskTable:
        """Lean index-based task table. No state data copies."""

        num_tasks: int = 0
        spawn_index: torch.Tensor = MISSING
        """Index into ``spawn_states`` for each task's spawn point."""
        target_index: torch.Tensor = MISSING
        """Index into ``spawn_states`` for each task's target point."""
        tile_index: torch.Tensor = MISSING
        """Terrain tile (``row * num_cols + col``) each task belongs to ``[num_tasks]``."""
        params: torch.Tensor = MISSING
        """Per-task sampled parameters ``[num_tasks, 13]``:
        ``[0:3]`` pos offset, ``[3:6]`` rot, ``[6:9]`` lin_vel,
        ``[9:12]`` ang_vel, ``[12]`` hold_time."""
        task_mask: torch.Tensor = MISSING
        """Active DOF mask ``[num_tasks, 12 + num_joints]``."""
        offsets: torch.Tensor = MISSING
        """CSR offsets ``[num_cmd_types + 1]`` into the task table."""
        kind: torch.Tensor = MISSING
        """Command type tag ``[num_cmd_types]``: 0=pos, 1=pose, 2=vel."""
        spawn_states: torch.Tensor = MISSING
        """Zero-copy reference to the reset event's state buffer
        ``[num_states, 7 + num_joints]``."""

    def __init__(self, cfg: RelativeStateCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.num_joints = self.robot.num_joints
        # pos(3) + rot(3) + lin_vel(3) + ang_vel(3) + joint_q(num_joints) + hold(1)
        self.state_dim = 12 + self.num_joints + 1
        self.time_idx = 12 + self.num_joints  # column index for hold/success/remaining time

        # Build the task table: IK pipeline -> bin by cell -> Cartesian product
        terrain = env.scene.terrain
        if terrain.terrain_mesh is None:
            raise RuntimeError(
                "RelativeStateCommand requires a terrain with a mesh. "
                "Set terrain_type='generator' in TerrainImporterCfg."
            )
        terrain_gen = terrain.cfg.terrain_generator

        # Auto-fill kinematics fields from the robot ArticulationCfg so the
        # retarget pipeline stays consistent with the scene asset (single
        # source of truth). Command_presets.py leaves these blank on
        # purpose; overriding here would desync the pipeline's USD /
        # default stance from the one the physics sim actually loads.
        from isaaclab.utils.assets import check_file_path, retrieve_file_path

        robot_articulation_cfg = self.robot.cfg
        usd_path = robot_articulation_cfg.spawn.usd_path
        # Newton's USD loader is local-filesystem only; retrieve remote
        # nucleus/S3 URLs so newton.ModelBuilder().add_usd() can open them.
        if check_file_path(usd_path) == 2:
            usd_path = retrieve_file_path(usd_path, force_download=False)

        kin_cfg = cfg.pipeline_cfg.kin
        kin_cfg.usd_path = usd_path
        kin_cfg.default_pos = (0.0, 0.0, robot_articulation_cfg.init_state.pos[2])
        kin_cfg.default_joint_pos = robot_articulation_cfg.init_state.joint_pos
        kin_cfg.device = self.device

        table_data = build_task_table(
            terrain_mesh=terrain.terrain_mesh,
            terrain_origins=terrain.terrain_origins,
            cell_size=terrain_gen.size,
            pipeline_cfg=cfg.pipeline_cfg,
            commands=cfg.commands,
            num_joints=self.num_joints,
            pool_size=cfg.pool_size,
            device=self.device,
        )
        self.table = self.TaskTable(**table_data)

        self._command_names = list(cfg.commands.keys())
        self.success_rates: torch.Tensor | None = None
        self._success_per_cmd = torch.zeros(self.table.kind.shape[0], device=self.device)

        # [num_envs, {target=0, delta=1, current=2}, state_dim]
        self.cmd_buf = torch.zeros(self.num_envs, 3, self.state_dim, device=self.device).contiguous()
        self.cmd_buf[:, 1] = 1  # init delta to nonzero so nothing triggers success at t=0
        self.cmd_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)  # command type per env
        # which state columns are active: 12 root DOFs (pos/rot/lin_vel/ang_vel) + num_joints
        self.cmd_mask = torch.zeros(self.num_envs, 12 + self.num_joints, device=self.device, dtype=torch.bool)
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # task table row per env

        # success threshold per error group: pos, rot, lin_vel, ang_vel, joints
        self.num_error_groups = 5
        reward_scale = [cfg.pos_std, cfg.rot_std, cfg.lin_vel_std, cfg.ang_vel_std, cfg.joint_std]
        self._reward_scales = torch.tensor(reward_scale, device=self.device).view(1, self.num_error_groups)
        self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # per-group error norms: [num_envs, 5] for pos/rot/lin_vel/ang_vel/joints
        self._err = torch.empty(self.num_envs, self.num_error_groups, device=self.device)

        self._error_group_names = ["error_pos", "error_rot", "error_linvel", "error_angvel", "error_joints"]
        for name in self._error_group_names:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        self._warp_seed = 1

    def __str__(self) -> str:
        msg = "RelativeStateCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Delta state in base frame ``[num_envs, 12 + num_joints]`` (excludes hold time)."""
        return self.cmd_buf[:, 1, : 12 + self.num_joints]

    def _update_metrics(self):
        for group_idx, name in enumerate(self._error_group_names):
            self.metrics[name] = self._err[:, group_idx]
        self.metrics["instant_success"] = torch.all(self._err < self._reward_scales, dim=1).float()

        if self.success_rates is not None:
            offsets = self.table.offsets
            for cmd_id, name in enumerate(self._command_names):
                start = int(offsets[cmd_id].item())
                end = int(offsets[cmd_id + 1].item())
                if end > start:
                    self._success_per_cmd[cmd_id] = self.success_rates[start:end].mean()
                else:
                    self._success_per_cmd[cmd_id] = 0.0
                self._env.extras["log"]["Metrics/goal_point/success_rate_" + name] = self._success_per_cmd[
                    cmd_id
                ].item()

    def resample_indices(self, env_ids: torch.Tensor):
        indices = torch.randint(0, self.table.num_tasks, (env_ids.numel(),), device=self.device)
        self.cmd_indices[env_ids] = indices
        offsets = self.table.offsets
        for cmd_id in range(self.table.kind.shape[0]):
            in_range = (indices >= offsets[cmd_id]) & (indices < offsets[cmd_id + 1])
            self.cmd_ids[env_ids[in_range]] = cmd_id

    def _resample_command(self, env_ids: torch.Tensor):
        self.resample_indices(env_ids)
        task_idx = self.cmd_indices[env_ids]
        target_state_idx = self.table.target_index[task_idx]
        spawn_state_idx = self.table.spawn_index[task_idx]

        # Build the target row in a local tensor then write it back with a single
        # __setitem__; ``cmd_buf[env_ids, 0]`` is advanced indexing and would copy.
        target = torch.zeros(env_ids.numel(), self.state_dim, device=self.device)
        target[:, :3] = self.table.spawn_states[target_state_idx, :3] + self.table.params[task_idx, :3]
        target[:, 3:12] = self.table.params[task_idx, 3:12]
        target[:, 12 : 12 + self.num_joints] = self.table.spawn_states[target_state_idx, 7:]
        target[:, self.time_idx] = self.table.params[task_idx, 12]
        self.cmd_buf[env_ids, 0] = target

        self.cmd_buf[env_ids, 2, self.time_idx] = 0.0
        self.cmd_mask[env_ids] = self.table.task_mask[task_idx]

        # Teleport robot to the spawn pose associated with this task.
        spawn_state = self.table.spawn_states[spawn_state_idx]
        num_envs = env_ids.shape[0]
        zero_root_vel = torch.zeros(num_envs, 6, device=self.device)
        zero_joint_vel = torch.zeros(num_envs, self.num_joints, device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=spawn_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=zero_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=spawn_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=zero_joint_vel, env_ids=env_ids)

    def _update_command(self):
        """Recompute delta state from current robot state. Minimizes temporaries."""
        root_state_w = wp.to_torch(self.robot.data.root_state_w)
        root_quat = wp.to_torch(self.robot.data.root_quat_w)
        joint_pos = wp.to_torch(self.robot.data.joint_pos)

        current = self.cmd_buf[:, 2]
        target = self.cmd_buf[:, 0]
        delta = self.cmd_buf[:, 1]

        ti = self.time_idx
        nj = self.num_joints

        current[:, :3] = root_state_w[:, :3]
        current[:, 3], current[:, 4], current[:, 5] = euler_xyz_from_quat(root_quat)
        current[:, 6:12] = root_state_w[:, 7:13]
        current[:, 12 : 12 + nj] = joint_pos

        # Delta position (world -> body frame, masked)
        torch.sub(target[:, :3], current[:, :3], out=delta[:, :3])
        delta[:, :3] *= self.cmd_mask[:, :3]
        delta[:, :3] = quat_apply_inverse(root_quat, delta[:, :3])

        # Delta orientation (axis-angle in body frame, masked)
        quat_des = quat_from_euler_xyz(target[:, 3], target[:, 4], target[:, 5])
        delta[:, 3:6] = axis_angle_from_quat(quat_mul(quat_inv(root_quat), quat_des))
        delta[:, 3:6] *= self.cmd_mask[:, 3:6]

        # Delta velocity (world -> body frame, masked)
        torch.sub(target[:, 6:9], current[:, 6:9], out=delta[:, 6:9])
        delta[:, 6:9] = quat_apply_inverse(root_quat, delta[:, 6:9])
        torch.sub(target[:, 9:12], current[:, 9:12], out=delta[:, 9:12])
        delta[:, 9:12] = quat_apply_inverse(root_quat, delta[:, 9:12])
        delta[:, 6:12] *= self.cmd_mask[:, 6:12]

        # Delta joints (masked, in-place)
        torch.sub(target[:, 12 : 12 + nj], joint_pos, out=delta[:, 12 : 12 + nj])
        delta[:, 12 : 12 + nj] *= self.cmd_mask[:, 12:]

        # Success tracking: hold time is trailing scalar at time_idx
        self.compute_state_error()
        current[:, ti] += self._env.step_dt * torch.all(self._err < self._reward_scales, dim=1)
        torch.sub(target[:, ti], current[:, ti], out=delta[:, ti])

    def compute_state_error(self):
        """Compute per-group error norms.

        Groups: pos(3), rot(3), lin_vel(3), ang_vel(3), joints(num_joints).
        """
        delta = self.cmd_buf[:, 1]
        self._err[:, 0] = delta[:, 0:3].norm(dim=-1)
        self._err[:, 1] = delta[:, 3:6].norm(dim=-1)
        self._err[:, 2] = delta[:, 6:9].norm(dim=-1)
        self._err[:, 3] = delta[:, 9:12].norm(dim=-1)
        self._err[:, 4] = delta[:, 12 : 12 + self.num_joints].norm(dim=-1)

    def get_state_error(self):
        return self._err

    def get_task_done(self) -> torch.Tensor:
        return self.cmd_buf[:, 1, self.time_idx] <= 0.0

    def get_task_reward(self) -> torch.Tensor:
        return (self.cmd_buf[:, 1, self.time_idx] <= 0.0).float()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            from isaaclab.markers import VisualizationMarkers

            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        kinds = self.table.kind[self.cmd_ids.long()]
        pos_task_ids = self._env.scene._ALL_INDICES[kinds == 0]
        pose_task_ids = self._env.scene._ALL_INDICES[kinds == 1]
        vel_task_ids = self._env.scene._ALL_INDICES[kinds == 2]

        goal_translations = []
        goal_orientations = []
        goal_scales = []
        goal_marker_indices = []

        if len(pos_task_ids) > 0:
            goal_pos = self.cmd_buf[pos_task_ids, 0, :3].clone()
            goal_pos[:, 2] += 0.5
            goal_translations.append(goal_pos)
            goal_orientations.append(self._identity_quat[: len(pos_task_ids)])
            goal_scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(len(pos_task_ids), 1))
            goal_marker_indices.append(torch.full((len(pos_task_ids),), 1, device=self.device, dtype=torch.long))

        if len(pose_task_ids) > 0:
            goal_pos = self.cmd_buf[pose_task_ids, 0, :3].clone()
            goal_pos[:, 2] += 0.5
            goal_translations.append(goal_pos)
            euler = self.cmd_buf[pose_task_ids, 0, 3:6]
            quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            goal_orientations.append(quat)
            goal_scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(len(pose_task_ids), 1))
            goal_marker_indices.append(torch.full((len(pose_task_ids),), 2, device=self.device, dtype=torch.long))

        if len(vel_task_ids) > 0:
            base_pos_w = wp.to_torch(self.robot.data.root_pos_w)[vel_task_ids].clone()
            base_pos_w[:, 2] += 0.5
            xy_cmd = self.cmd_buf[vel_task_ids, 0, 6:8]
            scale, quat = self._resolve_xy_velocity_to_arrow(
                xy_cmd,
                self._identity_quat[: len(vel_task_ids)],
                self.cfg.goal_visualizer_cfg.markers["vel_arrow"].scale,
            )
            goal_translations.append(base_pos_w)
            goal_orientations.append(quat)
            goal_scales.append(scale)
            goal_marker_indices.append(torch.zeros((len(vel_task_ids),), device=self.device, dtype=torch.long))

        self.goal_visualizer.visualize(
            translations=torch.cat(goal_translations, dim=0),
            orientations=torch.cat(goal_orientations, dim=0),
            scales=torch.cat(goal_scales, dim=0),
            marker_indices=torch.cat(goal_marker_indices, dim=0),
        )

        base_pos_w = wp.to_torch(self.robot.data.root_pos_w).clone()
        base_pos_w[:, 2] += 0.5
        scale, quat = self._resolve_xy_velocity_to_arrow(
            wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2],
            wp.to_torch(self.robot.data.root_quat_w),
            self.cfg.current_vel_visualizer_cfg.markers["arrow"].scale,
        )
        self.current_vel_visualizer.visualize(base_pos_w, quat, scale)

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity: torch.Tensor,
        base_quat_w: torch.Tensor,
        default_scale: Sequence[float] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = default_scale if default_scale is not None else self.cfg.current_vel_visualizer_cfg
        arrow_scale = torch.tensor(scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
