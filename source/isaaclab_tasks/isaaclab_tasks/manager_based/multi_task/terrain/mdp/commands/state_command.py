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

The public ``command`` observation is a separate policy-facing tensor:
root delta ``[0:12]`` followed by target foot positions in the current
base frame ``[12:12+3*num_feet]``. Joint targets stay in ``cmd_buf`` for
reset and debug state bookkeeping, but are not exposed as command deltas.

Error groups: ``pos, rot, lin_vel, ang_vel, foot_pos`` -- each reduced
to a scalar norm. Success = all active group errors below threshold.
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

from isaaclab_tasks.manager_based.multi_task.mdp.util import (
    ArticulationResetStateAdapter,
    set_reset_state,
)

from .task_table_builder import _joint_order_from_names, build_task_table, synthesize_terrain_origins

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RelativeStateCommandCfg


class RelativeStateCommand(CommandTerm):
    """Track the delta between a target and current robot state.

    Computes per-group error norms for 5 groups:
    ``pos(3), rot(3), lin_vel(3), ang_vel(3), foot_pos(3 * num_feet)``.
    Inactive groups (per ``cmd_mask`` and terrain-foot mask) contribute
    zero error.
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
        """Active command mask ``[num_tasks, 12 + num_joints]``."""
        task_is_terrain: torch.Tensor = MISSING
        """Whether each task uses terrain-conforming target state data ``[num_tasks]``."""
        task_uses_feet: torch.Tensor = MISSING
        """Whether each task activates target-foot-position success ``[num_tasks]``."""
        offsets: torch.Tensor = MISSING
        """CSR offsets ``[num_cmd_types + 1]`` into the task table."""
        kind: torch.Tensor = MISSING
        """Command type tag ``[num_cmd_types]``: 0=pos, 1=pose, 2=vel."""
        spawn_states: torch.Tensor = MISSING
        """Zero-copy reference to the reset event's state buffer
        ``[num_states, 13 + 2 * num_joints]``."""

    def __init__(self, cfg: RelativeStateCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.num_joints = self.robot.num_joints
        self._joint_pos_slice = slice(13, 13 + self.num_joints)
        self._reset_state_adapters = [ArticulationResetStateAdapter(cfg.asset_name)]
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
        # purpose except for validation placeholders; overriding here would
        # desync the pipeline's USD /
        # default stance from the one the physics sim actually loads.
        from isaaclab.utils.assets import check_file_path, retrieve_file_path

        robot_articulation_cfg = self.robot.cfg
        usd_path = robot_articulation_cfg.spawn.usd_path
        # Newton's USD loader is local-filesystem only; retrieve remote
        # nucleus/S3 URLs so newton.ModelBuilder().add_usd() can open them.
        if check_file_path(usd_path) == 2:
            usd_path = retrieve_file_path(usd_path, force_download=False)

        pipeline_cfg = cfg.pipeline_cfg.replace(kin=cfg.pipeline_cfg.kin.copy())
        kin_cfg = pipeline_cfg.kin
        kin_cfg.usd_path = usd_path
        kin_cfg.default_pos = (0.0, 0.0, robot_articulation_cfg.init_state.pos[2])
        kin_cfg.default_joint_pos = robot_articulation_cfg.init_state.joint_pos
        kin_cfg.device = self.device

        # ``terrain_origins`` is ``None`` when the importer is configured
        # with ``use_terrain_origins=False`` (env_origins fall back to grid
        # spacing). Reproduce IsaacLab's tile-centre layout from the
        # generator cfg so the task table can still bound the sampler.
        terrain_origins = terrain.terrain_origins
        if terrain_origins is None:
            terrain_origins = synthesize_terrain_origins(
                num_rows=int(terrain_gen.num_rows),
                num_cols=int(terrain_gen.num_cols),
                size=terrain_gen.size,
                device=self.device,
            )

        table_data = build_task_table(
            terrain_mesh=terrain.terrain_mesh,
            terrain_origins=terrain_origins,
            cell_size=terrain_gen.size,
            pipeline_cfg=pipeline_cfg,
            commands=cfg.commands,
            num_joints=self.num_joints,
            device=self.device,
            pool_spacing=cfg.pool_spacing,
            pool_spacing_area_divisor=cfg.pool_spacing_area_divisor,
            pool_sampling_size=cfg.pool_sampling_size,
            robot_joint_names=self.robot.joint_names,
            exclude_self_pairs=cfg.exclude_self_pairs,
        )
        self._target_fk_kin = table_data.pop("kin")
        self._newton_joint_names = table_data.pop("newton_joint_names")
        self._foot_body_names = table_data.pop("foot_body_names")
        self._newton_foot_body_ids = table_data.pop("foot_body_ids")
        self._isaac_to_newton_joint_order = torch.tensor(
            _joint_order_from_names(self.robot.joint_names, self._newton_joint_names),
            device=self.device,
            dtype=torch.long,
        )
        self._foot_body_ids, foot_body_names = self.robot.find_bodies(self._foot_body_names, preserve_order=True)
        if foot_body_names != self._foot_body_names:
            raise RuntimeError(
                "PhysX foot body order does not match Newton foot body order: "
                f"physx={foot_body_names}, newton={self._foot_body_names}."
            )
        self.num_feet = len(self._foot_body_ids)
        self.command_dim = 12 + 3 * self.num_feet
        self.table = self.TaskTable(**table_data)

        self._command_names = list(cfg.commands.keys())
        self.success_rates: torch.Tensor | None = None
        self._success_per_cmd = torch.zeros(self.table.kind.shape[0], device=self.device)

        # [num_envs, {target=0, delta=1, current=2}, state_dim]
        self.cmd_buf = torch.zeros(self.num_envs, 3, self.state_dim, device=self.device).contiguous()
        self.cmd_buf[:, 1] = 1  # init delta to nonzero so nothing triggers success at t=0
        # which state columns are active: 12 root DOFs (pos/rot/lin_vel/ang_vel) + num_joints
        self.cmd_mask = torch.zeros(self.num_envs, 12 + self.num_joints, device=self.device, dtype=torch.bool)
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # task table row per env
        self._command_indices_bound = False
        reset_state_dim = self.table.spawn_states.shape[1]
        self._task_idx_scratch = torch.empty(self.num_envs, device=self.device, dtype=torch.long)
        self._target_state_idx_scratch = torch.empty_like(self._task_idx_scratch)
        self._spawn_state_idx_scratch = torch.empty_like(self._task_idx_scratch)
        self._target_reset_state_scratch = torch.empty(self.num_envs, reset_state_dim, device=self.device)
        self._spawn_reset_state_scratch = torch.empty_like(self._target_reset_state_scratch)
        self._task_params_scratch = torch.empty(self.num_envs, 13, device=self.device)
        self._task_mask_scratch = torch.empty_like(self.cmd_mask)
        self._terrain_task_scratch = torch.empty(self.num_envs, device=self.device, dtype=torch.bool)
        self._foot_task_scratch = torch.empty_like(self._terrain_task_scratch)
        self._target_row_scratch = torch.empty(self.num_envs, self.state_dim, device=self.device)
        self._target_foot_pos_w = torch.zeros(self.num_envs, self.num_feet, 3, device=self.device)
        self._target_foot_pos_resample_scratch = torch.empty_like(self._target_foot_pos_w)
        self._target_foot_offset_w = torch.empty_like(self._target_foot_pos_w)
        self._target_foot_pos_b = torch.zeros_like(self._target_foot_pos_w)
        self._current_foot_pos_w = torch.zeros_like(self._target_foot_pos_w)
        self._foot_delta_w = torch.empty_like(self._target_foot_pos_w)
        self._foot_delta_b = torch.zeros_like(self._target_foot_pos_w)
        self._foot_cross = torch.empty_like(self._target_foot_pos_w)
        self._foot_cross2 = torch.empty_like(self._target_foot_pos_w)
        self._foot_success_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._command_obs = torch.zeros(self.num_envs, self.command_dim, device=self.device)
        self._target_fk_capacity = 0
        self._target_fk_joint_q = torch.empty(0, device=self.device)
        self._target_fk_body_q_t = torch.empty(0, device=self.device)
        self._target_fk_joint_qd: wp.array | None = None
        self._target_fk_body_q: wp.array | None = None
        self._target_fk_body_qd: wp.array | None = None

        # Per-command-type success thresholds: pos, rot, lin_vel, ang_vel, foot_pos.
        # Each command-type cfg may override any field via its own ``pos_std`` / ``rot_std`` /
        # ``lin_vel_std`` / ``ang_vel_std`` / ``foot_pos_std`` (``None`` falls back to global).
        self.num_error_groups = 5
        std_attrs = ("pos_std", "rot_std", "lin_vel_std", "ang_vel_std", "foot_pos_std")
        global_stds = (cfg.pos_std, cfg.rot_std, cfg.lin_vel_std, cfg.ang_vel_std, cfg.foot_pos_std)
        reward_scales = torch.empty(len(cfg.commands), self.num_error_groups, device=self.device)
        for cmd_idx, cmd_cfg in enumerate(cfg.commands.values()):
            for grp_idx, attr in enumerate(std_attrs):
                override = getattr(cmd_cfg, attr, None)
                reward_scales[cmd_idx, grp_idx] = global_stds[grp_idx] if override is None else override
        self._reward_scales = reward_scales

        # Per-cmd-type inverse scale broadcast across the command-obs channels:
        # 12 root channels grouped as [pos x3, rot x3, lin_vel x3, ang_vel x3]
        # followed by 3 * num_feet foot-position channels.
        obs_group_widths = (3, 3, 3, 3, 3 * self.num_feet)
        obs_inv_scales = torch.empty(len(cfg.commands), self.command_dim, device=self.device)
        col = 0
        for grp_idx, width in enumerate(obs_group_widths):
            obs_inv_scales[:, col : col + width] = reward_scales[:, grp_idx : grp_idx + 1].reciprocal()
            col += width
        self._obs_inv_unit_scales = obs_inv_scales
        self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # per-group error norms: [num_envs, 5] for pos/rot/lin_vel/ang_vel/foot_pos
        self._err = torch.empty(self.num_envs, self.num_error_groups, device=self.device)

        self._error_group_names = ["error_pos", "error_rot", "error_linvel", "error_angvel", "error_foot_pos"]
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
        """Policy command: root delta + target foot positions in base frame."""
        return self._command_obs

    @property
    def cmd_ids(self) -> torch.Tensor:
        """Command-type ids derived from the currently selected task rows."""
        return torch.bucketize(self.cmd_indices, self.table.offsets[1:-1], right=True)

    def _update_metrics(self):
        for group_idx, name in enumerate(self._error_group_names):
            self.metrics[name] = self._err[:, group_idx]
        self.metrics["instant_success"] = torch.all(
            self._err < self._reward_scales[self.cmd_ids.long()], dim=1
        ).float()

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

    def bind_command_indices(self, command_indices: torch.Tensor) -> None:
        """Bind selected task-table rows to an externally-owned tensor.

        Args:
            command_indices: Per-env task-table row indices.
        """
        expected_shape = (self.num_envs,)
        if tuple(command_indices.shape) != expected_shape:
            raise ValueError(f"command_indices must have shape {expected_shape}, got {tuple(command_indices.shape)}.")
        if command_indices.device != self.cmd_indices.device:
            raise ValueError(
                f"command_indices must be on device {self.cmd_indices.device}, got {command_indices.device}."
            )
        if command_indices.dtype != self.cmd_indices.dtype:
            raise ValueError(f"command_indices must have dtype {self.cmd_indices.dtype}, got {command_indices.dtype}.")
        self.cmd_indices = command_indices
        self._command_indices_bound = True

    def resample_indices(self, env_ids: torch.Tensor):
        if self._command_indices_bound:
            return
        indices = self._task_idx_scratch[: env_ids.numel()]
        torch.randint(0, self.table.num_tasks, (env_ids.numel(),), device=self.device, out=indices)
        self.cmd_indices.index_copy_(0, env_ids, indices)

    def _ensure_target_fk_scratch(self, num_states: int) -> None:
        """Ensure Newton FK scratch buffers can evaluate ``num_states`` targets."""
        if num_states <= self._target_fk_capacity:
            return

        coord_count = int(self._target_fk_kin.model.joint_coord_count)
        dof_count = int(self._target_fk_kin.model.joint_dof_count)
        body_count = int(self._target_fk_kin.model.body_count)
        self._target_fk_capacity = num_states
        self._target_fk_joint_q = torch.empty(num_states, coord_count, device=self.device)
        self._target_fk_body_q_t = torch.empty(num_states, body_count, 7, device=self.device)
        self._target_fk_joint_qd = wp.zeros((num_states, dof_count), dtype=wp.float32, device=self.device)
        self._target_fk_body_q = wp.from_torch(self._target_fk_body_q_t, dtype=wp.transformf)
        self._target_fk_body_qd = wp.zeros((num_states, body_count), dtype=wp.spatial_vectorf, device=self.device)

    def _compute_target_foot_pos_w(self, target_state: torch.Tensor, out: torch.Tensor) -> None:
        """Write target foot origins [m] from packed reset states via Newton FK."""
        num_states = target_state.shape[0]
        self._ensure_target_fk_scratch(num_states)

        joint_q = self._target_fk_joint_q[:num_states]
        joint_q[:, :7] = target_state[:, :7]
        torch.index_select(
            target_state[:, self._joint_pos_slice],
            1,
            self._isaac_to_newton_joint_order,
            out=joint_q[:, 7:],
        )

        self._target_fk_kin.eval_fk_batched(
            wp.from_torch(joint_q),
            self._target_fk_joint_qd[:num_states],
            self._target_fk_body_q[:num_states],
            self._target_fk_body_qd[:num_states],
        )
        body_q_t = self._target_fk_body_q_t[:num_states]
        for foot_id, body_id in enumerate(self._newton_foot_body_ids):
            out[:num_states, foot_id].copy_(body_q_t[:, body_id, :3])

    def _resample_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        num_resets = env_ids.numel()
        self.resample_indices(env_ids)

        task_idx = self._task_idx_scratch[:num_resets]
        target_state_idx = self._target_state_idx_scratch[:num_resets]
        spawn_state_idx = self._spawn_state_idx_scratch[:num_resets]
        target_state = self._target_reset_state_scratch[:num_resets]
        spawn_state = self._spawn_reset_state_scratch[:num_resets]
        task_params = self._task_params_scratch[:num_resets]
        task_mask = self._task_mask_scratch[:num_resets]
        terrain_task = self._terrain_task_scratch[:num_resets]
        foot_task = self._foot_task_scratch[:num_resets]

        torch.index_select(self.cmd_indices, 0, env_ids, out=task_idx)
        torch.index_select(self.table.target_index, 0, task_idx, out=target_state_idx)
        torch.index_select(self.table.spawn_index, 0, task_idx, out=spawn_state_idx)
        torch.index_select(self.table.spawn_states, 0, target_state_idx, out=target_state)
        torch.index_select(self.table.spawn_states, 0, spawn_state_idx, out=spawn_state)
        torch.index_select(self.table.params, 0, task_idx, out=task_params)
        torch.index_select(self.table.task_mask, 0, task_idx, out=task_mask)
        torch.index_select(self.table.task_is_terrain, 0, task_idx, out=terrain_task)
        torch.index_select(self.table.task_uses_feet, 0, task_idx, out=foot_task)

        # Buffer states are sampled in tile-frame (single-patch sampler with
        # tile centered at world origin). Lift them into per-env world frame
        # by adding ``env_origins`` so each env operates in its own copy of
        # the terrain. Required when ``scene.env_spacing != 0``.
        env_origins = self._env.scene.env_origins[env_ids]
        target_state[:, :3].add_(env_origins)
        spawn_state[:, :3].add_(env_origins)

        # Build the target row in a local tensor then write it back with a single
        # index_copy_; ``cmd_buf[env_ids, 0]`` is advanced indexing and would copy.
        target = self._target_row_scratch[:num_resets]
        target.zero_()
        target[:, :3].copy_(target_state[:, :3])
        target[:, :3].add_(task_params[:, :3])
        target[:, 3:12].copy_(task_params[:, 3:12])
        target[:, 12 : 12 + self.num_joints].copy_(target_state[:, self._joint_pos_slice])
        if bool(terrain_task.any()):
            if bool(terrain_task.all()):
                target[:, :3].copy_(target_state[:, :3])
                target[:, 3], target[:, 4], target[:, 5] = euler_xyz_from_quat(target_state[:, 3:7])
            else:
                terrain_local_ids = terrain_task.nonzero(as_tuple=False).squeeze(-1)
                terrain_target_state = target_state[terrain_local_ids]
                target[terrain_local_ids, :3] = terrain_target_state[:, :3]
                target[terrain_local_ids, 3], target[terrain_local_ids, 4], target[terrain_local_ids, 5] = (
                    euler_xyz_from_quat(terrain_target_state[:, 3:7])
                )
        target[:, self.time_idx].copy_(task_params[:, 12])
        self.cmd_buf[:, 0].index_copy_(0, env_ids, target)

        self.cmd_buf[:, 2, self.time_idx].index_fill_(0, env_ids, 0.0)
        self.cmd_mask.index_copy_(0, env_ids, task_mask)
        target_foot_pos_w = self._target_foot_pos_resample_scratch[:num_resets]
        self._compute_target_foot_pos_w(target_state, target_foot_pos_w)
        self._target_foot_pos_w.index_copy_(0, env_ids, target_foot_pos_w)
        self._foot_success_mask.index_copy_(0, env_ids, foot_task)

        # Teleport robot to the spawn reset state associated with this task.
        set_reset_state(self._env, spawn_state, env_ids, self._reset_state_adapters)

    def _update_command(self):
        """Recompute delta state from current robot state. Minimizes temporaries."""
        root_state_w = wp.to_torch(self.robot.data.root_state_w)
        root_quat = wp.to_torch(self.robot.data.root_quat_w)
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        body_link_pos_w = wp.to_torch(self.robot.data.body_link_pos_w)

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

        # Joint targets stay in cmd_buf for reset/debug bookkeeping, but the
        # policy command and success criteria are foot-position based.
        delta[:, 12 : 12 + nj].zero_()

        for foot_id, body_id in enumerate(self._foot_body_ids):
            self._current_foot_pos_w[:, foot_id].copy_(body_link_pos_w[:, body_id])
        torch.sub(self._target_foot_pos_w, self._current_foot_pos_w, out=self._foot_delta_w)
        torch.sub(self._target_foot_pos_w, root_state_w[:, None, :3], out=self._target_foot_offset_w)
        quat_xyz = root_quat[:, :3]
        quat_w = root_quat[:, 3:4]
        for foot_id in range(self.num_feet):
            torch.cross(quat_xyz, self._foot_delta_w[:, foot_id], dim=-1, out=self._foot_cross[:, foot_id])
            self._foot_cross[:, foot_id].mul_(2.0)
            torch.mul(quat_w, self._foot_cross[:, foot_id], out=self._foot_delta_b[:, foot_id])
            self._foot_delta_b[:, foot_id].neg_()
            self._foot_delta_b[:, foot_id].add_(self._foot_delta_w[:, foot_id])
            torch.cross(quat_xyz, self._foot_cross[:, foot_id], dim=-1, out=self._foot_cross2[:, foot_id])
            self._foot_delta_b[:, foot_id].add_(self._foot_cross2[:, foot_id])

            target_foot_pos_b = self._target_foot_pos_b[:, foot_id]
            torch.cross(quat_xyz, self._target_foot_offset_w[:, foot_id], dim=-1, out=self._foot_cross[:, foot_id])
            self._foot_cross[:, foot_id].mul_(2.0)
            torch.mul(quat_w, self._foot_cross[:, foot_id], out=target_foot_pos_b)
            target_foot_pos_b.neg_()
            target_foot_pos_b.add_(self._target_foot_offset_w[:, foot_id])
            torch.cross(quat_xyz, self._foot_cross[:, foot_id], dim=-1, out=self._foot_cross2[:, foot_id])
            target_foot_pos_b.add_(self._foot_cross2[:, foot_id])

        self._target_foot_pos_b *= self._foot_success_mask.view(-1, 1, 1)
        self._command_obs[:, :12].copy_(delta[:, :12])
        self._command_obs[:, 12:].copy_(self._target_foot_pos_b.flatten(1))
        if self.cfg.normalize_command_obs:
            self._command_obs.mul_(self._obs_inv_unit_scales[self.cmd_ids.long()])

        # Success tracking: hold time is trailing scalar at time_idx
        self.compute_state_error()
        current[:, ti] += self._env.step_dt * torch.all(
            self._err < self._reward_scales[self.cmd_ids.long()], dim=1
        )
        torch.sub(target[:, ti], current[:, ti], out=delta[:, ti])

    def compute_state_error(self):
        """Compute per-group error norms.

        Groups: pos(3), rot(3), lin_vel(3), ang_vel(3), foot positions(3 * num_feet).
        """
        delta = self.cmd_buf[:, 1]
        self._err[:, 0] = delta[:, 0:3].norm(dim=-1)
        self._err[:, 1] = delta[:, 3:6].norm(dim=-1)
        self._err[:, 2] = delta[:, 6:9].norm(dim=-1)
        self._err[:, 3] = delta[:, 9:12].norm(dim=-1)
        foot_err = self._foot_delta_b.norm(dim=-1).amax(dim=1)
        self._err[:, 4] = torch.where(self._foot_success_mask, foot_err, torch.zeros_like(foot_err))
        # self.print_state_error()

    def print_state_error(self) -> None:
        """Print per-group state-error stats to stdout for terminal debug.

        One line per error group with mean / max across envs and the
        success-threshold the metric is graded against. ``ok%`` is the
        fraction of envs whose group error sits below threshold.

        Cheap but not free: one CPU sync per scalar (~15 syncs total),
        so call sparingly — e.g. every N policy steps, not every physics
        step.
        """
        err = self._err  # [num_envs, num_error_groups]
        # Per-env thresholds vary with cmd_ids; print the strictest per group as a summary.
        per_env_thresh = self._reward_scales[self.cmd_ids.long()]  # [num_envs, num_error_groups]
        thresh = per_env_thresh.amin(dim=0)  # [num_error_groups]
        mean = err.mean(dim=0)
        mx = err.amax(dim=0)
        ok_pct = 100.0 * (err < per_env_thresh).float().mean(dim=0)
        print(f"state error (n_envs={self.num_envs})")
        print(f"success_rate: {self._success_per_cmd.mean().item()}")
        for i, name in enumerate(self._error_group_names):
            short = name.removeprefix("error_")
            print(
                f"  {short:<9s} μ={mean[i].item():7.3f}  max={mx[i].item():7.3f}  "
                f"thr={thresh[i].item():.3f}  ok={ok_pct[i].item():5.1f}%"
            )

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
        if not self.robot.is_initialized or not hasattr(self, "table"):
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

        # One sphere per active foot target. Tasks that don't activate the
        # foot-pos error group (``_foot_success_mask = False``) are skipped
        # so the visualizer only shows markers the policy is actually graded
        # against.
        foot_active_ids = self._env.scene._ALL_INDICES[self._foot_success_mask]
        if len(foot_active_ids) > 0:
            foot_pos_w = self._target_foot_pos_w[foot_active_ids].reshape(-1, 3)
            n_markers = foot_pos_w.shape[0]
            goal_translations.append(foot_pos_w)
            goal_orientations.append(self._identity_quat[:1].expand(n_markers, -1))
            goal_scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(n_markers, 1))
            goal_marker_indices.append(torch.full((n_markers,), 3, device=self.device, dtype=torch.long))

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
