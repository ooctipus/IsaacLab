# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
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
        kind: torch.Tensor = MISSING
        num_descretized_cmd: int = MISSING
        descretized_cmd: torch.Tensor = MISSING
        """Per-row layout: ``[0:3]`` spawn pos, ``[3:6]`` target pos,
        ``[6:16]`` ranges (rot, vel, hold)."""
        descretized_mask: torch.Tensor = MISSING
        # row ranges for each command: rows for cmd i are [offsets[i] : offsets[i+1]]
        descretized_cmd_offsets: torch.Tensor = MISSING  # [cardinal + 1], long

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

        # command names in the same order as commands.values() used in _build_spec
        # (Python dict preserves insertion order)
        self._command_names = list(self.cfg.commands.keys())

        self.success_rates: torch.Tensor | None = None
        self._success_per_cmd = torch.zeros(self.spec.cardinal, device=self.device)
        offsets = self.spec.descretized_cmd_offsets
        self._disc_count_per_cmd = (offsets[1:] - offsets[:-1]).to(torch.float32)

        # desired, error, current
        # cmd_buf[:, 0, :] → desired world state
        # cmd_buf[:, 1, :] → desired state in base frame (relative state)
        # cmd_buf[:, 2, :] → current state in world frame
        self.cmd_buf = torch.zeros(self.num_envs, 3, 13, device=self.device).contiguous()
        self.cmd_buf[:, 1] = 1  # important: initialize relative error and time to 1, nothing will trigger success.
        self.cmd_ids = torch.randint(
            0, self.spec.cardinal, size=(self.num_envs,), device=self.device, dtype=torch.int32
        )
        self.cmd_mask = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.bool)
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # reward scales used by get_task_reward() (group-wise scaling)
        reward_scale = [self.cfg.pos_std, self.cfg.rot_std, self.cfg.lin_vel_std, self.cfg.ang_vel_std]
        self._reward_scales = torch.tensor(reward_scale, device=self.device).view(1, 4)
        self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # scratch buffers
        self._rel = torch.empty(self.num_envs, 12, device=self.device)  # rel pos, rot, lin vel, ang vel
        self._err = torch.empty(self.num_envs, 4, device=self.device)  # error norm: pos, rot, lin_vel, ang_vel

        # metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_linvel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_angvel"] = torch.zeros(self.num_envs, device=self.device)

        # Success-criterion bookkeeping: resolve support feet and prepare lazy
        # caches for body weight and characteristic limb length ``L_ref``. The
        # ``L_ref`` capture is deferred to the first ``get_task_done`` call so
        # the asset is fully spawned and articulated when we read body Z.
        foot_ids, _ = self.robot.find_bodies(self.cfg.foot_body_names)
        if len(foot_ids) == 0:
            raise ValueError(
                "RelativeStateCommandCfg.foot_body_names matched no bodies on asset"
                f" {self.cfg.asset_name!r}; cannot derive N_support_feet or L_ref."
            )
        self._foot_ids: list[int] = list(foot_ids)
        self._n_support_feet: int = len(self._foot_ids)
        self._weight: torch.Tensor | None = None  # [num_envs], m·g per env
        self._L_ref: float | None = None  # scalar, ⟨z_base − mean_f z_foot⟩_envs at first call

        # Resolve the joint-wrench sensor used by the success gate. We read the
        # joint reaction torque rather than ``applied_torque`` so that hard
        # joint-stop reactions (which are absorbed by constraints, not the
        # actuator) are counted toward the per-joint mechanical load.
        self._wrench_sensor = env.scene[self.cfg.joint_wrench_sensor_name]

        # Resolve the contact sensor used by the feet-bear-weight gate. We map
        # each foot body (by articulation index) to its channel index on the
        # contact sensor by body-name match, because the contact sensor's body
        # ordering need not coincide with the articulation's.
        self._contact_sensor = env.scene[self.cfg.contact_sensor_name]
        robot_body_names = list(self.robot.data.body_names)
        sensor_body_names = list(self._contact_sensor.body_names or [])
        contact_foot_channels: list[int] = []
        for fid in self._foot_ids:
            name = robot_body_names[fid]
            try:
                contact_foot_channels.append(sensor_body_names.index(name))
            except ValueError as err:
                raise RuntimeError(
                    f"ContactSensor {self.cfg.contact_sensor_name!r} does not cover foot body"
                    f" {name!r}; expand the sensor's prim_path regex to include all feet."
                ) from err
        self._contact_foot_channels: torch.Tensor = torch.tensor(
            contact_foot_channels, device=self.device, dtype=torch.long
        )

        # Bind a live per-env gravity view so the success effort gate remains
        # a true geometric ratio under per-env gravity randomization. Newton
        # exposes ``model.gravity`` as a per-world Warp array — a zero-copy
        # torch view here reflects subsequent gravity-event writes
        # automatically. PhysX only has a scene-wide gravity, so we cache the
        # PhysX simulation view and read it lazily after startup events fire
        # (mirrors how :class:`mdp.randomize_physics_scene_gravity` writes it).
        self._sim_gravity: torch.Tensor | None = None
        self._physics_sim_view = None
        try:
            import isaaclab_newton.physics.newton_manager as nm  # noqa: PLC0415

            model = nm.NewtonManager.get_model()
            if model is not None and model.gravity is not None:
                self._sim_gravity = wp.to_torch(model.gravity)
        except ImportError:
            pass
        if self._sim_gravity is None:
            self._physics_sim_view = env.sim.physics_sim_view

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
        spawn_flat = spawn_src[..., :3].clone().reshape(n_subterrains, num_spawn_per_terrain, 3)
        # Terrain patches are surface points; commands track the robot root pose above that surface.
        root_pos_offset = wp.to_torch(self.robot.data.default_root_pose)[0, :3]
        ranges = torch.zeros((len(commands), 13, 2), device=self.device)  # 0-12 pos,rot,lin_vel,ang_vel. 12 hold time
        mask = torch.zeros((len(commands), 12), device=self.device, dtype=torch.bool)
        kind = torch.zeros(len(commands), dtype=torch.int32, device=self.device)

        blocks = []
        mask_blocks = []
        row_counts = []  # number of rows per command
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

                targets_full = self.terrains.flat_patches[val.target_key]  # [R,C,Pt,3]
                targets = targets_full[..., :3]
                _, _, num_targets_per_terrain, _ = targets.shape
                targets_flat = targets.reshape(n_subterrains, num_targets_per_terrain, 3)
                val.pos_x = val.pos_y = val.pos_z = None  # TerrainCommands do not use pos_* ranges
                kind[cmd_id] = 1 if (val.roll or val.pitch or val.yaw) else 0

                spawn_pos_expanded = spawn_flat[:, :, None, :]
                target_pos_expanded = targets_flat[:, None, :, :]

                spawn_all = spawn_pos_expanded.expand(-1, num_spawn_per_terrain, num_targets_per_terrain, -1).reshape(
                    -1, 3
                )
                target_all = target_pos_expanded.expand(-1, num_spawn_per_terrain, num_targets_per_terrain, -1).reshape(
                    -1, 3
                )
                mi = ranges[cmd_id, :, 0].view(1, 13)
                rand_range = torch.rand(spawn_all.shape[0], 13, device=self.device) * (ranges[cmd_id, :, 1] - mi) + mi

                # block layout (16 cols):
                #   0:3   spawn position
                #   3:6   target position
                #   6:9   target rotation (roll, pitch, yaw)
                #   9:12  target linear velocity
                #   12:15 target angular velocity
                #   15    hold time
                block = torch.zeros(spawn_all.shape[0], 16, device=self.device)
                block[:, 0:3] = spawn_all
                block[:, 3:6] = target_all + root_pos_offset + rand_range[:, :3]
                block[:, 6:16] = rand_range[:, 3:]
                blocks.append(block)
                mask_blocks.append(mask[cmd_id].view(1, 12).expand(block.shape[0], 12))
                row_counts.append(block.shape[0])

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

                # same 16-col layout as TerrainCommands above
                count = n_subterrains * num_spawn_per_terrain * n_samples
                block = torch.zeros(count, 16, device=self.device)
                # 3:6 target pos, 6:9 target rot, 9:12 target lin_vel, 12:15 target ang_vel, 15 hold time
                block[:, 3:16] = torch.rand(count, 13, device=self.device) * span[:,].view(1, 13) + _min.view(1, 13)
                spawn_pos_expanded = spawn_flat[:, :, None, :].expand(
                    n_subterrains, num_spawn_per_terrain, n_samples, 3
                )
                block[:, 0:3] = spawn_pos_expanded.reshape(-1, 3)  # spawn position
                block[:, 3:6] += spawn_pos_expanded.reshape(-1, 3)  # target relative to spawn
                block[:, 3:6] += root_pos_offset  # target is the robot root pose, not the terrain contact point
                blocks.append(block)

                block_mask = mask[cmd_id].view(1, 12).expand(count, 12)
                mask_blocks.append(block_mask)
                row_counts.append(block.shape[0])

        # stack all discrete commands
        descretized_cmd = torch.cat(blocks, dim=0)
        descretized_mask = torch.cat(mask_blocks, dim=0)

        # build offsets so rows for cmd i are [offsets[i] : offsets[i+1]]
        counts = torch.tensor(row_counts, device=self.device, dtype=torch.long)
        descretized_cmd_offsets = torch.zeros(len(commands) + 1, device=self.device, dtype=torch.long)
        descretized_cmd_offsets[1:] = torch.cumsum(counts, dim=0)

        spec = self.CommandSpec(
            cardinal=len(commands),
            kind=kind,
            num_descretized_cmd=descretized_cmd.shape[0],
            descretized_cmd=descretized_cmd,
            descretized_mask=descretized_mask,
            descretized_cmd_offsets=descretized_cmd_offsets,
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

        if self.success_rates is not None:
            offsets = self.spec.descretized_cmd_offsets
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
        indices = torch.randint(0, self.spec.num_descretized_cmd, (env_ids.numel(),), device=self.device)
        self.cmd_indices[env_ids] = indices

    def _resample_command(self, env_ids: torch.Tensor):
        self.resample_indices(env_ids)
        idx = self.cmd_indices[env_ids]
        self.cmd_buf[env_ids, 0, :] = self.spec.descretized_cmd[idx, 3:16]
        self.cmd_mask[env_ids] = self.spec.descretized_mask[idx]

        rows = self.spec.descretized_cmd[idx]
        self._env.scene.terrain.env_origins.index_copy_(0, env_ids.long(), rows[:, 0:3])
        root_pose, root_velocity = self._reset_robot_to_spawn(env_ids, rows[:, 0:3])
        self._update_reset_command_state(env_ids, root_pose, root_velocity)

    def _reset_robot_to_spawn(
        self, env_ids: torch.Tensor, spawn_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place the robot root at the sampled terrain spawn patch."""
        root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids].clone()
        root_pose[:, :3].add_(spawn_pos_w)
        root_velocity = wp.to_torch(self.robot.data.default_root_vel)[env_ids].clone()
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
        return root_pose, root_velocity

    def _update_reset_command_state(
        self, env_ids: torch.Tensor, root_pose: torch.Tensor, root_velocity: torch.Tensor
    ) -> None:
        """Initialize command error at reset without advancing success hold time."""
        root_quat = root_pose[:, 3:7]
        self.cmd_buf[env_ids, 2, :3] = root_pose[:, :3]
        self.cmd_buf[env_ids, 2, 3:6] = torch.stack(euler_xyz_from_quat(root_quat), dim=-1)
        self.cmd_buf[env_ids, 2, 6:12] = root_velocity
        self.cmd_buf[env_ids, 2, 12] = 0.0

        pos_w = (self.cmd_buf[env_ids, 0, :3] - root_pose[:, :3]) * self.cmd_mask[env_ids, :3]
        self._rel[env_ids, 0:3] = quat_apply_inverse(root_quat, pos_w)
        self.cmd_buf[env_ids, 1, :3] = self._rel[env_ids, 0:3]

        quat_des = quat_from_euler_xyz(
            self.cmd_buf[env_ids, 0, 3],
            self.cmd_buf[env_ids, 0, 4],
            self.cmd_buf[env_ids, 0, 5],
        )
        quat_err = quat_mul(quat_inv(root_quat), quat_des)
        self._rel[env_ids, 3:6] = axis_angle_from_quat(quat_err) * self.cmd_mask[env_ids, 3:6]
        self.cmd_buf[env_ids, 1, 3:6] = self._rel[env_ids, 3:6]

        vel_rel_w = (self.cmd_buf[env_ids, 0, 6:12] - root_velocity) * self.cmd_mask[env_ids, 6:12]
        self._rel[env_ids, 6:9] = quat_apply_inverse(root_quat, vel_rel_w[:, :3])
        self._rel[env_ids, 9:12] = quat_apply_inverse(root_quat, vel_rel_w[:, 3:6])
        self.cmd_buf[env_ids, 1, 6:12] = self._rel[env_ids, 6:12]

        rel_grouped = self.cmd_buf[env_ids, 1, :12].view(env_ids.shape[0], 4, 3)
        self._err[env_ids] = torch.linalg.vector_norm(rel_grouped, dim=2)
        self.cmd_buf[env_ids, 1, 12] = self.cmd_buf[env_ids, 0, 12]

    def _update_command(self):
        """Update world-state row and recompute relative state for all envs.

        - Row 2 of cmd_buf is updated from the robot root state in world frame.
        - Row 1 of cmd_buf is recomputed as the target expressed in base frame:
          position, axis-angle rotation, linear velocity, angular velocity.
        """
        root_state_w = wp.to_torch(self.robot.data.root_state_w)
        root_quat = wp.to_torch(self.robot.data.root_quat_w)  # (N, 4) xyzw

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
        self.cmd_buf[:, 2, 12] = success_time
        # remaining time until success: target_hold - success_time
        torch.sub(self.cmd_buf[:, 0, 12], self.cmd_buf[:, 2, 12], out=self.cmd_buf[:, 1, 12])

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create / toggle visualization markers for the command targets."""
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
            goal_orientations.append(self._identity_quat[: len(pos_task_ids)])
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

        # current velocity arrows
        base_pos_w = wp.to_torch(self.robot.data.root_pos_w).clone()
        base_pos_w[:, 2] += 0.5
        s, q = self._resolve_xy_velocity_to_arrow(
            wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2],
            wp.to_torch(self.robot.data.root_quat_w),
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

    def get_task_done(self) -> torch.Tensor:
        """Return ``True`` per env where the task is considered successfully held.

        Terrain-invariant gates (all must hold):

        1. ``timer_done``: goal held for the required duration.
        2. ``settled``: every body's linear/angular speed below the configured
           thresholds.
        3. ``natural``: worst-joint specific effort
           ``max_j |τ_react,axis_j| / (m·|g|·L_ref) < multiplier / N_support_feet``.
        4. ``feet_bear_weight``: ``sum_f max(0, F_z[f]) / (m·|g|) >= min_foot_weight_fraction``.

        See :class:`RelativeStateCommandCfg` for the knobs.
        """
        if self._weight is None:
            body_mass = wp.to_torch(self.robot.data.body_mass)
            if self._sim_gravity is not None:
                g_mag = self._sim_gravity.norm(dim=-1)
            else:
                g = self._physics_sim_view.get_gravity()  # physx case
                g_vec = torch.tensor((g[0], g[1], g[2]), device=self.device, dtype=torch.float32)
                g_mag = g_vec.norm().expand(self.num_envs)
            self._weight = body_mass.sum(dim=-1) * g_mag
        if self._L_ref is None:
            body_pos_w = wp.to_torch(self.robot.data.body_pos_w)
            z_base = body_pos_w[:, 0, 2]
            z_feet = body_pos_w[:, self._foot_ids, 2].mean(dim=-1)
            self._L_ref = float((z_base - z_feet).mean().item())

        timer_done = self.cmd_buf[:, 1, 12] <= 0.0

        lin_speed_max = wp.to_torch(self.robot.data.body_lin_vel_w).norm(dim=-1).amax(dim=-1)
        ang_speed_max = wp.to_torch(self.robot.data.body_ang_vel_w).norm(dim=-1).amax(dim=-1)
        settled = (lin_speed_max < self.cfg.success_body_lin_speed_thresh) & (
            ang_speed_max < self.cfg.success_body_ang_speed_thresh
        )

        # Joint-axis (incoming-joint-frame x) component of the reaction wrench;
        # includes hard-stop constraint reactions that ``applied_torque`` misses.
        wrench_torque = wp.to_torch(self._wrench_sensor.data.torque)
        joint_axis_torque_max = wrench_torque[..., 0].abs().amax(dim=-1)
        specific_effort_max = joint_axis_torque_max / (self._weight * self._L_ref)
        threshold = self.cfg.success_effort_multiplier / float(self._n_support_feet)
        natural = specific_effort_max < threshold

        net_forces = wp.to_torch(self._contact_sensor.data.net_forces_w)
        foot_fz = net_forces[:, self._contact_foot_channels, 2].sum(dim=-1)
        weight_supported = foot_fz / self._weight
        feet_bear_weight = weight_supported >= self.cfg.success_min_foot_weight_fraction

        return timer_done & settled & natural & feet_bear_weight

    def get_task_reward(self) -> torch.Tensor:
        return self.get_task_done().float()
