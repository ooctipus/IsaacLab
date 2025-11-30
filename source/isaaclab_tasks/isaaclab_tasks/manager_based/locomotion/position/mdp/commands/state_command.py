# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command term that tracks a full "relative state" target (pos, rot, lin vel, ang vel).

Interface:

- cmd_buf[:, 0, :]  → desired world state   (x,y,z, r,p,y, vx,vy,vz, wx,wy,wz)
- cmd_buf[:, 1, :]  → desired state in base frame (relative state)
- cmd_buf[:, 2, :]  → current root state in world frame

This term supports different "command kinds" via cfg:

- PositionCommands:  only x,y,z ranges
- PoseCommands:      x,y,z + r,p,y ranges
- VelocityCommands:  lin/ang velocity ranges
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_apply_inverse,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    quat_inv,
    axis_angle_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.terrains import TerrainImporter
    from .commands_cfg import RelativeStateCommandCfg


class RelativeStateCommand(CommandTerm):
    """Command term that generates and tracks a full relative state target.

    The command has 12 DOFs grouped into 4x3:

    - group 0: position      (x, y, z)
    - group 1: orientation   (roll, pitch, yaw) or axis-angle error in base frame
    - group 2: linear vel    (vx, vy, vz)
    - group 3: angular vel   (wx, wy, wz)

    The term samples a desired world state based on ranges in cfg.commands,
    then converts it into a relative state in the robot base frame for use
    in observations/rewards.
    """

    cfg: RelativeStateCommandCfg

    @configclass
    class CommandSpec:
        """Compiled specification for command sampling.

        Attributes:
            cardinal: Number of distinct command entries in cfg.commands.
            mask:     [cardinal, 12] bool mask indicating which DOFs are active per command.
            min:      [cardinal, 12] lower bounds for each DOF (inactive entries ignored).
            span:     [cardinal, 12] (max - min) per DOF (inactive entries ignored).
            kind:     [cardinal] int tag per command:
                      0 = PositionCommands
                      1 = PoseCommands
                      2 = VelocityCommands
        """

        cardinal: int = 1
        mask: torch.Tensor = MISSING
        min: torch.Tensor = MISSING
        span: torch.Tensor = MISSING
        kind: torch.Tensor = MISSING
        num_descretized_cmd: int = MISSING
        descretized_cmd: torch.Tensor = MISSING
        descretized_mask: torch.Tensor = MISSING

    def __init__(self, cfg: RelativeStateCommandCfg, env: ManagerBasedEnv):
        """Initialize the relative state command generator.

        Args:
            cfg: Configuration for ranges and command types.
            env: Manager-based environment providing robot and terrain.
        """
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.spec = self._build_spec(self.cfg.commands)

        # desired, error, current
        # cmd_buf[:, 0, :] → desired world state
        # cmd_buf[:, 1, :] → desired state in base frame (relative state)
        # cmd_buf[:, 2, :] → current state in world frame
        self.cmd_buf = torch.zeros(self.num_envs, 3, 13, device=self.device).contiguous()
        self.cmd_buf[:, 1] = 1  # important: initialize relative error and time to 1, nothing will trigger success.
        self.cmd_ids = torch.randint(0, self.spec.cardinal, size=(self.num_envs,), device=self.device, dtype=torch.int32)
        self.cmd_mask = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.bool)
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # reward scales used by get_task_reward() (group-wise scaling)
        reward_scale = [self.cfg.pos_std, self.cfg.rot_std, self.cfg.lin_vel_std, self.cfg.ang_vel_std]
        self._reward_scales = torch.tensor(reward_scale, device=self.device).view(1, 4)
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # scratch buffers
        self._rel = torch.empty(self.num_envs, 12, device=self.device)  # rel pos, rot, lin vel, ang vel
        self._err = torch.empty(self.num_envs, 4, device=self.device)  # error norm: pos, rot, lin_vel, ang_vel

        # metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_linvel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_angvel"] = torch.zeros(self.num_envs, device=self.device)

        self._warp_seed = 1

    def _build_spec(self, commands: dict[str, RelativeStateCommandCfg.Commands]) -> CommandSpec:
        """Compile cfg.commands into a CommandSpec used for fast sampling."""
        from .commands_cfg import RelativeStateCommandCfg

        self.terrains: TerrainImporter = self._env.scene["terrain"]
        has_spawn = "spawn" in self.terrains.flat_patches
        spawn_src = self.terrains.flat_patches["spawn"] if has_spawn else self._env.scene.terrain.terrain_origins
        if spawn_src.dim() == 3:
            spawn_src = spawn_src.unsqueeze(2)  # [row, col, 1, 3]

        num_row, num_col, num_spawn_per_terrain, _ = spawn_src.shape
        n_subterrains = num_row * num_col
        spawn_flat = spawn_src.clone().reshape(n_subterrains, num_spawn_per_terrain, 3)
        ranges = torch.zeros((len(commands), 13, 2), device=self.device)  # 0-12 pos,rot,lin_vel,ang_vel. 12 hold time
        mask = torch.zeros((len(commands), 12), device=self.device, dtype=torch.bool)
        kind = torch.zeros(len(commands), dtype=torch.int32, device=self.device)

        blocks = []
        mask_blocks = []
        n_samples = 20  # bins per spawn

        for cmd_id, val in enumerate(commands.values()):
            # --- collect tuple ranges for this cfg FIRST ---
            for data_id, data in enumerate(val.__dict__.values()):
                if data is not None and isinstance(data, tuple):
                    if data_id < 12:
                        mask[cmd_id, data_id] = True
                    ranges[cmd_id, data_id, 0] = data[0]
                    ranges[cmd_id, data_id, 1] = data[1]

            # --- TerrainCommands: per-tile Cartesian(spawn, target) ---
            if isinstance(val, RelativeStateCommandCfg.TerrainCommands):
                if "target" not in self.terrains.flat_patches:
                    raise RuntimeError(
                        "The terrain-based command generator requires a valid flat patch under 'target'"
                        f"in the terrain. Found: {list(self.terrains.flat_patches.keys())}"
                    )

                targets = self.terrains.flat_patches[val.target_key]  # [R,C,Pt,3] or compatible
                _, _, num_targets_per_terrain, _ = targets.shape
                targets_flat = targets.reshape(n_subterrains, num_targets_per_terrain, 3)
                val.pos_x = val.pos_y = val.pos_z = None  # TerrainCommands do not use pos_* ranges
                kind[cmd_id] = 1 if (val.roll or val.pitch or val.yaw) else 0

                spawn_exp = spawn_flat[:, :, None, :]
                target_exp = targets_flat[:, None, :, :]

                spawn_all = spawn_exp.expand(-1, num_spawn_per_terrain, num_targets_per_terrain, -1).reshape(-1, 3)
                target_all = target_exp.expand(-1, num_spawn_per_terrain, num_targets_per_terrain, -1).reshape(-1, 3)

                block = torch.zeros(spawn_all.shape[0], 16, device=self.device)
                # 0:3 spawn, 3:6 target, 6:15 unused, 15 hold time
                block[:, 0:3] = spawn_all
                block[:, 3:6] = target_all
                mi = ranges[cmd_id, 3:13, 0]
                rand = torch.rand(spawn_all.shape[0], 10, device=self.device)
                block[:, 6:16] = rand * (ranges[cmd_id, 3:13, 1] - mi).view(1, 10) + mi.view(1, 10)
                blocks.append(block)

                # mask for this block: spawn position is always "active"; plus any DOFs that had ranges
                block_mask = torch.zeros(block.shape[0], 12, device=self.device, dtype=torch.bool)
                block_mask[:, 0:3] = True  # we always care about pos error for terrain command
                block_mask |= mask[cmd_id].view(1, 12)  # keep any non-pos DOFs that have ranges in the original cfg
                mask_blocks.append(block_mask)

            # --- Non-terrain commands: num bins * num_spawn per tile ---
            else:
                if isinstance(val, RelativeStateCommandCfg.PositionCommands):
                    kind[cmd_id] = 0
                elif isinstance(val, RelativeStateCommandCfg.PoseCommands):
                    kind[cmd_id] = 1
                elif isinstance(val, RelativeStateCommandCfg.VelocityCommands):
                    kind[cmd_id] = 2

                _min = ranges[cmd_id, :, 0]
                span = ranges[cmd_id, :, 1] - _min

                count = n_subterrains * num_spawn_per_terrain * n_samples
                block = torch.zeros(count, 16, device=self.device)
                # 3-6: target position, 6-9 target rotation, 9-12 target lin_vel, 12-15 target ang_vel, 15 hold time
                block[:, 3:16] = torch.rand(count, 13, device=self.device) * span[:,].view(1, 13) + _min.view(1, 13)
                spawn_exp = spawn_flat[:, :, None, :].expand(n_subterrains, num_spawn_per_terrain, n_samples, 3)
                block[:, 0:3] = spawn_exp.reshape(-1, 3)  # add spawn
                block[:, 3:6] += spawn_exp.reshape(-1, 3)  # add target relative to spawn
                blocks.append(block)

                block_mask = mask[cmd_id].view(1, 12).expand(count, 12)
                mask_blocks.append(block_mask)
        # stack all discrete commands
        descretized_cmd = torch.cat(blocks, dim=0)
        descretized_mask = torch.cat(mask_blocks, dim=0)

        spec = self.CommandSpec(
            cardinal=len(commands),
            mask=mask,
            min=ranges[..., 0],
            span=ranges[..., 1] - ranges[..., 0],
            kind=kind,
            num_descretized_cmd=descretized_cmd.shape[0],
            descretized_cmd=descretized_cmd,
            descretized_mask=descretized_mask
        )
        return spec

    def __str__(self) -> str:
        msg = "RelativeStateCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Return the current relative state target in base frame.

        Returns:
            Tensor of shape [num_envs, 13] corresponding to cmd_buf[:, 1, :].
        """
        return self.cmd_buf[:, 1, :12]

    def _update_metrics(self):
        """Update error metrics based on the last computed _err buffer."""
        self.metrics["error_pos"] = self._err[:, 0]
        self.metrics["error_rot"] = self._err[:, 1]
        self.metrics["error_linvel"] = self._err[:, 2]
        self.metrics["error_angvel"] = self._err[:, 3]

    def resample_indices(self, env_ids: torch.Tensor):
        indices = torch.randint(0, self.spec.num_descretized_cmd, (env_ids.numel(),), device=self.device)
        self.cmd_indices[env_ids] = indices

    def _resample_command(self, env_ids: torch.Tensor):
        self.resample_indices(env_ids)
        idx = self.cmd_indices[env_ids]
        self.cmd_buf[env_ids, 0, :] = self.spec.descretized_cmd[idx , 3:]
        self.cmd_buf[env_ids, 0, 5] = self.cmd_buf[env_ids, 0, 5].uniform_(-3.14, 3.14)
        self.cmd_buf[env_ids, 2, 12] = 0.0
        self.cmd_mask[env_ids] = self.spec.descretized_mask[idx]

        spawns_locations = self.spec.descretized_cmd[idx, :3]
        self._env.scene.terrain.env_origins.index_copy_(0, env_ids, spawns_locations)

    def _update_command(self):
        """Update world-state row and recompute relative state for all envs.

        - Row 2 of cmd_buf is updated from the robot root state in world frame.
        - Row 1 of cmd_buf is recomputed as the target expressed in base frame:
          position, axis-angle rotation, linear velocity, angular velocity.
        """
        root_state_w = self.robot.data.root_state_w
        root_quat = self.robot.data.root_quat_w  # (N, 4) wxyz

        # world state row
        self.cmd_buf[:, 2, :3] = root_state_w[:, :3]
        torch.stack(euler_xyz_from_quat(root_quat), dim=-1, out=self.cmd_buf[:, 2, 3:6])
        self.cmd_buf[:, 2, 6:12] = root_state_w[:, 7:13]

        # relative position (world → body)
        torch.sub(self.cmd_buf[:, 0, :3], self.cmd_buf[:, 2, :3], out=self._rel[:, 0:3])
        pos_w = self._rel[:, 0:3] * self.cmd_mask[:, :3]
        self.cmd_buf[:, 1, :3] = quat_apply_inverse(root_quat, pos_w)

        # relative orientation (axis-angle in body)
        quat_des = quat_from_euler_xyz(
            self.cmd_buf[:, 0, 3],
            self.cmd_buf[:, 0, 4],
            self.cmd_buf[:, 0, 5],
        )
        quat_err = quat_mul(quat_inv(root_quat), quat_des)
        rot_vec = axis_angle_from_quat(quat_err) * self.cmd_mask[:, 3:6]
        self._rel[:, 3:6] = rot_vec
        self.cmd_buf[:, 1, 3:6] = rot_vec

        # relative velocities (world → body)
        torch.sub(self.cmd_buf[:, 0, 6:12], self.cmd_buf[:, 2, 6:12], out=self._rel[:, 6:12])
        lin_w = self._rel[:, 6:9]
        ang_w = self._rel[:, 9:12]
        lin_b = quat_apply_inverse(root_quat, lin_w)
        ang_b = quat_apply_inverse(root_quat, ang_w)
        vel_rel = torch.cat([lin_b, ang_b], dim=-1) * self.cmd_mask[:, 6:12]
        self._rel[:, 6:12] = vel_rel
        self.cmd_buf[:, 1, 6:12] = vel_rel

        self.compute_state_error()
        success = torch.all(self._err < self._reward_scales, dim=1)
        success_time = self.cmd_buf[:, 2, 12]
        success_time[success] += self._env.step_dt
        success_time[~success] = 0.0
        self.cmd_buf[:, 2, 12] = success_time
        # remaining time until success: target_hold - success_time
        torch.sub(self.cmd_buf[:, 0, 12], self.cmd_buf[:, 2, 12], out=self.cmd_buf[:, 1, 12])

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create / toggle visualization markers for the command targets."""
        if debug_vis:
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
        """Callback for Kit debug visualization of position/pose/velocity goals."""
        if not self.robot.is_initialized:
            return

        kinds = self.spec.kind[self.cmd_ids.long()]
        pos_task_ids = self._env.scene._ALL_INDICES[kinds == 0]
        pose_task_ids = self._env.scene._ALL_INDICES[kinds == 1]
        vel_task_ids = self._env.scene._ALL_INDICES[kinds == 2]

        goal_translations = []
        goal_orientations = []
        goal_scales = []
        goal_marker_indices = []

        # Position goals: visualize as cuboids
        if len(pos_task_ids) > 0:
            goal_pos = self.cmd_buf[pos_task_ids, 0, :3].clone()
            goal_pos[:, 2] += 0.5
            goal_translations.append(goal_pos)
            goal_orientations.append(self._identity_quat[:len(pos_task_ids)])
            goal_scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(len(pos_task_ids), 1))
            goal_marker_indices.append(torch.full((len(pos_task_ids),), 1, device=self.device, dtype=torch.long))

        # Pose goals: visualize as oriented arrows
        if len(pose_task_ids) > 0:
            goal_pos = self.cmd_buf[pose_task_ids, 0, :3].clone()
            goal_pos[:, 2] += 0.5
            goal_translations.append(goal_pos)
            euler = self.cmd_buf[pose_task_ids, 0, 3:6]
            quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            goal_orientations.append(quat)
            goal_scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(len(pose_task_ids), 1))
            goal_marker_indices.append(torch.full((len(pose_task_ids),), 2, device=self.device, dtype=torch.long))

        # Velocity goals: visualize as arrows from base
        if len(vel_task_ids) > 0:
            base_pos_w = self.robot.data.root_pos_w[vel_task_ids].clone()
            base_pos_w[:, 2] += 0.5
            xy_cmd = self.cmd_buf[vel_task_ids, 0, 6:8]
            scale, quat = self._resolve_xy_velocity_to_arrow(
                xy_cmd,
                self._identity_quat[:len(vel_task_ids)],
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

        # current velocity arrows
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        s, q = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2],
            self.robot.data.root_quat_w,
            self.cfg.current_vel_visualizer_cfg.markers["arrow"].scale,
        )
        self.current_vel_visualizer.visualize(base_pos_w, q, s)

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity: torch.Tensor,
        base_quat_w: torch.Tensor,
        default_scale: Sequence[float] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert an XY velocity command into an oriented arrow in world frame."""
        scale = default_scale if default_scale is not None else self.cfg.current_vel_visualizer_cfg
        arrow_scale = torch.tensor(scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    def compute_state_error(self):
        """Compute grouped norms of the relative state error.

        Groups:
            0: position      (x,y,z)
            1: orientation   (axis-angle vector)
            2: linear vel    (vx,vy,vz)
            3: angular vel   (wx,wy,wz)
        """
        rel = self.cmd_buf[:, 1, :12]
        rel_grouped = rel.view(rel.shape[0], 4, 3)
        torch.sum(rel_grouped * rel_grouped, dim=2, out=self._err)
        self._err.sqrt_()

    def get_state_error(self):
        return self._err

    def get_task_success(self):
        return self.cmd_buf[:, 1, 12] <= 0.0

    def get_task_reward(self):
        """Compute a multiplicative reward from grouped errors.

        Reward per env:
            r = prod_{groups} (1 - tanh(err_group / std_group))

        where std_group comes from cfg.{pos_std, rot_std, lin_vel_std, ang_vel_std}.
        """
        group_r = 1.0 - torch.tanh(self._err / self._reward_scales)
        return group_r.prod(dim=1)
