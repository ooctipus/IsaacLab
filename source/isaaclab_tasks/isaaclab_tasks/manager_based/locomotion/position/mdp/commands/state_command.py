# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_inv, quat_mul, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.terrains import TerrainImporter
    from .commands_cfg import RelativeStateCommandCfg


class RelativeStateCommand(CommandTerm):

    cfg: RelativeStateCommandCfg
    """Configuration for the command generator."""

    @configclass
    class CommandSpec:

        cardinal: int = 1

        mask: torch.Tensor = MISSING

        min: torch.Tensor = MISSING

        span: torch.Tensor = MISSING

    def __init__(self, cfg: RelativeStateCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.spec = self._build_spec(self.cfg.commands)

        # desired, error, current
        self.cmd_buf = torch.zeros(self.num_envs, 3, 12, device=self.device)
        self.cmd_ids = torch.randint(0, self.spec.cardinal, size=(self.num_envs,), device=self.device)
        self.cmd_mask = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.bool)
        self.rand = torch.empty(self.num_envs, 12, device=self.device)
        self.reward = torch.zeros(self.num_envs, device=self.device)
        # --- pre-allocated / constant tensors to avoid per-step allocations ---
        reward_scale = [self.cfg.pos_std, self.cfg.rot_std, self.cfg.lin_vel_std, self.cfg.ang_vel_std]
        self._reward_scales = torch.tensor(reward_scale, device=self.device).view(1, 4)
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # --- scratch buffers (avoid per-step allocations) ---
        self._rel = torch.empty(self.num_envs, 12, device=self.device)  # rel pos, euler (rot), lin vel, ang vel
        self._err = torch.empty(self.num_envs, 4, device=self.device)  # error norm: pos, rot, lin_vel, ang_vel
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_linvel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_angvel"] = torch.zeros(self.num_envs, device=self.device)

    def _build_spec(self, commands: dict[str, RelativeStateCommandCfg.Commands]) -> CommandSpec:
        from .commands_cfg import RelativeStateCommandCfg
        num_cmd = len(commands)
        ranges = torch.zeros((len(commands), 12, 2), device=self.device)
        mask = torch.zeros((num_cmd, 12,), device=self.device, dtype=torch.bool)

        for cmd_id, val in enumerate(commands.values()):
            for data_id, data in enumerate(val.__dict__.values()):
                if data is not None and isinstance(data, tuple):
                    mask[cmd_id, data_id] = True
                    ranges[cmd_id, data_id, 0] = data[0]
                    ranges[cmd_id, data_id, 1] = data[1]

            if isinstance(val, RelativeStateCommandCfg.TerrainCommands):
                self.terrains: TerrainImporter = self._env.scene["terrain"]
                if "target" not in self.terrains.flat_patches or "spawn" not in self.terrains.flat_patches:
                    raise RuntimeError(
                        "The terrain-based command generator requires a valid flat patch under 'target' and 'spawn'"
                        f"in the terrain. Found: {list(self.terrains.flat_patches.keys())}"
                    )
                self.valid_targets: torch.Tensor = self.terrains.flat_patches[val.target_key]
                self.valid_spawn: torch.Tensor = self.terrains.flat_patches[val.spawn_key]

        spec = self.CommandSpec(
            cardinal=len(commands),
            mask=mask,
            min=ranges[..., 0],
            span=ranges[..., 1] - ranges[..., 0],
        )

        return spec

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The relative state in base frame"""
        return self.cmd_buf[:, 1]

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # assumes _compute_group_errors() was called this step
        self.metrics["error_pos"] = self._err[:, 0]
        self.metrics["error_rot"] = self._err[:, 1]
        self.metrics["error_linvel"] = self._err[:, 2]
        self.metrics["error_angvel"] = self._err[:, 3]

    def _resample_command(self, env_ids: torch.Tensor):
        # pick a command type for each env
        cmd_ids = torch.randint(0, self.spec.cardinal, size=(len(env_ids),), device=self.device)
        cmd_mask = self.spec.mask[cmd_ids]
        self.cmd_ids[env_ids] = cmd_ids
        self.cmd_mask[env_ids] = cmd_mask

        # random in [0,1)
        r = (self.rand[env_ids].uniform_() * self.spec.span[cmd_ids] + self.spec.min[cmd_ids])

        # desired position (world)
        default_position = self._env.scene.env_origins[env_ids].clone()
        default_position[:, 2] += self.robot.data.default_root_state[env_ids, 2]
        default_position *= cmd_mask[:, :3].to(default_position.dtype)

        self.cmd_buf[env_ids, 0, :3] = default_position + r[:, :3]
        self.cmd_buf[env_ids, 0, 3:12] = r[:, 3:12]

    def _update_command(self):
        """Re-target the position command to the current root state."""
        root_state_w = self.robot.data.root_state_w
        root_quat = self.robot.data.root_quat_w  # (N, 4)

        # update correct state
        self.cmd_buf[:, 2, :3] = root_state_w[:, :3]
        torch.stack(euler_xyz_from_quat(root_quat), dim=-1, out=self.cmd_buf[:, 2, 3:6])
        self.cmd_buf[:, 2, 6:12] = root_state_w[:, 7:13]

        # relative position (world → body)
        torch.sub(self.cmd_buf[:, 0, :3], self.cmd_buf[:, 2, :3], out=self._rel[:, 0:3])
        self.cmd_buf[:, 1, :3] = quat_apply_inverse(root_quat, self._rel[:, 0:3] * self.cmd_mask[:, :3])

        # relative orientation (euler diff in body)
        quat_des = quat_from_euler_xyz(self.cmd_buf[:, 0, 3], self.cmd_buf[:, 0, 4], self.cmd_buf[:, 0, 5],)
        quat_err = quat_mul(quat_inv(root_quat), quat_des)
        torch.stack(euler_xyz_from_quat(quat_err, wrap_to_2pi=True), dim=-1, out=self._rel[:, 3:6])
        self.cmd_buf[:, 1, 3:6] = self._rel[:, 3:6] * self.cmd_mask[:, 3:6]

        # relative velocities (world → body)
        torch.sub(self.cmd_buf[:, 0, 6:12], self.cmd_buf[:, 2, 6:12], out=self._rel[:, 6:12])
        self._rel[:, 6:9] = quat_apply_inverse(root_quat, self._rel[:, 6:9])
        self._rel[:, 9:12] = quat_apply_inverse(root_quat, self._rel[:, 9:12])
        self.cmd_buf[:, 1, 6:12] = self._rel[:, 6:12] * self.cmd_mask[:, 6:12]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)

            # set their visibility to true
            self.goal_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return

        # current per-env command mask (already set in _resample_command)
        cmd_mask = self.cmd_mask  # (num_envs, 12)

        has_velocity = torch.any(cmd_mask[:, 6:], dim=1)   # any lin/ang vel dim
        has_orientation = torch.any(cmd_mask[:, 3:6], dim=1)
        has_position = torch.any(cmd_mask[:, :3], dim=1)

        vel_task_ids = self._env.scene._ALL_INDICES[has_velocity]
        pose_task_ids = self._env.scene._ALL_INDICES[~has_velocity & has_orientation]
        pos_task_ids = self._env.scene._ALL_INDICES[~has_velocity & ~has_orientation & has_position]

        goal_translations = []
        goal_orientations = []
        goal_scales = []
        goal_marker_indices = []

        # --- position-only goals (cuboids) ---
        if len(pos_task_ids) > 0:
            goal_pos = self.cmd_buf[pos_task_ids, 0, :3].clone()
            goal_pos[:, 2] += 0.5
            goal_translations.append(goal_pos)
            goal_orientations.append(self._identity_quat[:len(pos_task_ids)])
            goal_scales.append(torch.tensor((1., 1., 1.), device=self.device).repeat(len(pos_task_ids), 1))
            goal_marker_indices.append(torch.full((len(pos_task_ids),), 1, device=self.device, dtype=torch.long))

        # --- pose goals (arrow orientation) ---
        if len(pose_task_ids) > 0:
            goal_pos = self.cmd_buf[pose_task_ids, 0, :3].clone()
            goal_pos[:, 2] += 0.5
            goal_translations.append(goal_pos)
            # cmd_buf[..., 3:6] is euler (roll, pitch, yaw) → convert to quat
            euler = self.cmd_buf[pose_task_ids, 0, 3:6]
            quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            goal_orientations.append(quat)
            goal_scales.append(torch.tensor((1., 1., 1.), device=self.device).repeat(len(pose_task_ids), 1))
            # assume marker index 0 = arrow
            goal_marker_indices.append(torch.full((len(pose_task_ids),), 2, device=self.device, dtype=torch.long))

        # --- velocity goals (arrows from base) ---
        if len(vel_task_ids) > 0:
            base_pos_w = self.robot.data.root_pos_w[vel_task_ids].clone()
            base_pos_w[:, 2] += 0.5

            # xy commanded velocity: vx, vy (indices 6,7)
            xy_cmd = self.cmd_buf[vel_task_ids, 0, 6:8]
            # identity quats → no extra rotation applied
            scale, quat = self._resolve_xy_velocity_to_arrow(
                xy_cmd,
                self._identity_quat[:len(vel_task_ids)],
                self.cfg.goal_visualizer_cfg.markers["vel_arrow"].scale
            )

            goal_translations.append(base_pos_w)
            goal_orientations.append(quat)
            goal_scales.append(scale)
            goal_marker_indices.append(torch.zeros((len(vel_task_ids),), device=self.device, dtype=torch.long))

        # visualize goals
        self.goal_visualizer.visualize(
            translations=torch.cat(goal_translations, dim=0),
            orientations=torch.cat(goal_orientations, dim=0),
            scales=torch.cat(goal_scales, dim=0),
            marker_indices=torch.cat(goal_marker_indices, dim=0),
        )

        # visualize current velocity arrows (same as before, but a bit tidied)
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        s, q = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2],
            self.robot.data.root_quat_w,
            self.cfg.current_vel_visualizer_cfg.markers["arrow"].scale
        )
        self.current_vel_visualizer.visualize(base_pos_w, q, s)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, base_quat_w: torch.Tensor, default_scale: Sequence[float] | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        scale = default_scale if default_scale is not None else self.cfg.current_vel_visualizer_cfg
        arrow_scale = torch.tensor(scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        arrow_quat = quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def get_state_error(self):
        """Reward that minimizes position, rotation, lin/ang velocities with group scales."""
        # squared norms (no in-place on rel!)
        rel = self.cmd_buf[:, 1]  # (N, 12), maybe non-contiguous
        rel_grouped = rel.contiguous().view(rel.shape[0], 4, 3)
        torch.sum(rel_grouped * rel_grouped, dim=2, out=self._err)
        self._err.sqrt_()

        return self._err

    def get_task_reward(self):
        self.get_state_error()
        # vectorized scaling + nonlinearity
        group_r = 1.0 - torch.tanh(self._err / self._reward_scales)
        self.reward = group_r.prod(dim=1)
        return self.reward


class TerrainBasedRelativeStateCommand(RelativeStateCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TerrainBasedRelativeStateCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedRelativeStateCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]
        self.valid_spawn: torch.Tensor = self.terrain.flat_patches["spawn"]
        self.marker_indices = torch.cat((
            torch.zeros(self.valid_targets.view(-1, 3).shape[0], dtype=torch.long, device=self.device),
            torch.ones(self.valid_spawn.view(-1, 3).shape[0], dtype=torch.long, device=self.device)
        ), dim=0)

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.cmd_buf[env_ids] = 0.0
        cmd_mode = torch.randint(0, 3, size=(len(env_ids), ), device=self.device)
        self.cmd_mode[env_ids] = cmd_mode
        r = self.rand[env_ids].uniform_().mul_(self.span).add_(self.min)

        # position_cmd
        cmd_ids = env_ids[(cmd_mode == 0) | (cmd_mode == 1)]
        # self.cmd_buf[cmd_ids, 0, :3] = self._env.scene.env_origins[cmd_ids] + r[cmd_ids, 0, :3]
        # self.cmd_buf[cmd_ids, 0, 2] += self.robot.data.default_root_state[cmd_ids, 2]
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.cmd_buf[cmd_ids, 0, :3] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]

        # pose_cmd
        cmd_ids = env_ids[cmd_mode == 1]
        self.cmd_buf[cmd_ids, 0, 3:6] = r[cmd_ids, 3:6]

        # vel_cmd
        cmd_ids = env_ids[cmd_mode == 2]
        self.cmd_buf[cmd_ids, 1, 6 : 13] = r[:, 6 : 12]
