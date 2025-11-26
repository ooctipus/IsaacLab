# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz, quat_mul, quat_inv, axis_angle_from_quat

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

        kind: torch.Tensor = MISSING

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
        self.is_torch_backend = True
        # desired, error, current
        self.cmd_buf = torch.zeros(self.num_envs, 3, 12, device=self.device)
        self.cmd_ids = torch.randint(0, self.spec.cardinal, size=(self.num_envs,), device=self.device, dtype=torch.int32)
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
        self._init_warp()
        self._warp_seed = 1

    def _init_warp(self):
        self._num_cmd = self.spec.cardinal
        self._wp_device = wp.device_from_torch(self.device)

        # no flattening needed anymore
        self._wp_spec_min = wp.from_torch(self.spec.min.contiguous(), requires_grad=False)
        self._wp_spec_span = wp.from_torch(self.spec.span.contiguous(), requires_grad=False)
        self._wp_spec_mask = wp.from_torch(self.spec.mask.contiguous(), requires_grad=False)

        self._wp_env_origins = wp.from_torch(self._env.scene.env_origins.contiguous(), requires_grad=False)
        self._default_root_z = self.robot.data.default_root_state[:, 2].contiguous()
        self._wp_default_root_z = wp.from_torch(self._default_root_z, requires_grad=False)

        self._wp_cmd_buf = wp.from_torch(self.cmd_buf.contiguous(), requires_grad=False)
        self._wp_cmd_ids = wp.from_torch(self.cmd_ids, requires_grad=False)
        self._wp_cmd_mask_2d = wp.from_torch(self.cmd_mask, requires_grad=False)
        self._wp_rel = wp.from_torch(self._rel, requires_grad=False)

    def _build_spec(self, commands: dict[str, RelativeStateCommandCfg.Commands]) -> CommandSpec:
        from .commands_cfg import RelativeStateCommandCfg
        num_cmd = len(commands)
        ranges = torch.zeros((len(commands), 12, 2), device=self.device)
        mask = torch.zeros((num_cmd, 12,), device=self.device, dtype=torch.bool)
        kind = torch.zeros(num_cmd, dtype=torch.int32, device=self.device)

        for cmd_id, val in enumerate(commands.values()):
            # tag kind
            if isinstance(val, RelativeStateCommandCfg.PositionCommands):
                kind[cmd_id] = 0
            elif isinstance(val, RelativeStateCommandCfg.PoseCommands):
                kind[cmd_id] = 1
            elif isinstance(val, RelativeStateCommandCfg.VelocityCommands):
                kind[cmd_id] = 2

            if isinstance(val, RelativeStateCommandCfg.TerrainCommands):
                self.terrains: TerrainImporter = self._env.scene["terrain"]
                if "target" not in self.terrains.flat_patches or "spawn" not in self.terrains.flat_patches:
                    raise RuntimeError(
                        "The terrain-based command generator requires a valid flat patch under 'target' and 'spawn'"
                        f"in the terrain. Found: {list(self.terrains.flat_patches.keys())}"
                    )
                self.valid_targets: torch.Tensor = self.terrains.flat_patches[val.target_key]
                self.valid_spawn: torch.Tensor = self.terrains.flat_patches[val.spawn_key]
                val.pos_x = None
                val.pos_y = None
                val.pos_z = None

            for data_id, data in enumerate(val.__dict__.values()):
                if data is not None and isinstance(data, tuple):
                    mask[cmd_id, data_id] = True
                    ranges[cmd_id, data_id, 0] = data[0]
                    ranges[cmd_id, data_id, 1] = data[1]

        spec = self.CommandSpec(
            cardinal=len(commands),
            mask=mask,
            min=ranges[..., 0],
            span=ranges[..., 1] - ranges[..., 0],
            kind=kind,
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
        if env_ids.numel() == 0:
            return

        if self.is_torch_backend:
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

        else:
            wp_env_ids = wp.from_torch(env_ids, requires_grad=False)
            self._warp_seed += 1

            wp.launch(
                resample_commands_kernel,
                dim=env_ids.shape[0],
                inputs=[
                    wp_env_ids,
                    self._wp_cmd_ids,
                    self._wp_cmd_mask_2d,
                    self._wp_spec_mask,
                    self._wp_spec_min,
                    self._wp_spec_span,
                    self._wp_env_origins,
                    self._wp_default_root_z,
                    self._wp_cmd_buf,
                    self._num_cmd,
                    self.num_envs,
                    self._warp_seed,
                ],
                device=self._wp_device,
            )

    def _update_command(self):
        root_state_w = self.robot.data.root_state_w
        root_quat = self.robot.data.root_quat_w  # still use in Torch

        # world state row
        self.cmd_buf[:, 2, :3] = root_state_w[:, :3]
        torch.stack(euler_xyz_from_quat(root_quat), dim=-1, out=self.cmd_buf[:, 2, 3:6])
        self.cmd_buf[:, 2, 6:12] = root_state_w[:, 7:13]
        wp_root_quat = wp.from_torch(root_quat, requires_grad=False)

        if self.is_torch_backend:
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

            # axis-angle rotation vector in body frame: (N, 3)
            torch.stack(euler_xyz_from_quat(quat_err, wrap_to_2pi=True), dim=-1, out=self._rel[:, 3:6])
            self.cmd_buf[:, 1, 3:6] = axis_angle_from_quat(quat_err) * self.cmd_mask[:, 3:6]

            # relative velocities (world → body)
            torch.sub(self.cmd_buf[:, 0, 6:12], self.cmd_buf[:, 2, 6:12], out=self._rel[:, 6:12])
            self._rel[:, 6:9] = quat_apply_inverse(root_quat, self._rel[:, 6:9])
            self._rel[:, 9:12] = quat_apply_inverse(root_quat, self._rel[:, 9:12])
            self.cmd_buf[:, 1, 6:12] = self._rel[:, 6:12] * self.cmd_mask[:, 6:12]
        else:
            # Warp uses the pre-wrapped view
            wp.launch(
                update_command_rel_kernel,
                dim=self.num_envs,
                inputs=[
                    wp_root_quat,
                    self._wp_cmd_buf,
                    self._wp_cmd_mask_2d,
                    self._wp_rel,
                    self.num_envs,
                ],
                device=self._wp_device,
            )

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
        kinds = self.spec.kind[self.cmd_ids.long()]
        pos_task_ids = self._env.scene._ALL_INDICES[kinds == 0]
        pose_task_ids = self._env.scene._ALL_INDICES[kinds == 1]
        vel_task_ids = self._env.scene._ALL_INDICES[kinds == 2]

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


@wp.kernel
def resample_commands_kernel(
    env_ids: wp.array(dtype=wp.int64),
    cmd_ids_out: wp.array(dtype=wp.int32),
    cmd_mask_out: wp.array(dtype=wp.bool, ndim=2),
    spec_mask: wp.array(dtype=wp.bool, ndim=2),
    spec_min: wp.array(dtype=wp.float32, ndim=2),
    spec_span: wp.array(dtype=wp.float32, ndim=2),
    env_origins: wp.array(dtype=wp.float32, ndim=2),
    default_root_z: wp.array(dtype=wp.float32),
    cmd_buf: wp.array(dtype=wp.float32, ndim=3),
    num_cmd: int,
    num_envs: int,
    seed: int,
):
    tid = wp.tid()
    if tid >= env_ids.shape[0]:
        return

    env64 = env_ids[tid]
    if env64 < 0 or env64 >= num_envs:
        return

    # cast to int32 for rand_init
    env = wp.int32(env64)

    state = wp.rand_init(seed, env)
    cmd_type = wp.randi(state, 0, num_cmd)

    cmd_ids_out[env] = cmd_type

    # base position = env origin + default root z
    base_x = env_origins[env, 0]
    base_y = env_origins[env, 1]
    base_z = env_origins[env, 2] + default_root_z[env]

    for d in range(12):
        m = spec_mask[cmd_type, d]
        cmd_mask_out[env, d] = m

        val = 0.0
        if m:
            lo = spec_min[cmd_type, d]
            span = spec_span[cmd_type, d]
            u = wp.randf(state)
            val = lo + span * u

        if d == 0:
            cmd_buf[env, 0, 0] = base_x + val if m else 0.0
        elif d == 1:
            cmd_buf[env, 0, 1] = base_y + val if m else 0.0
        elif d == 2:
            cmd_buf[env, 0, 2] = base_z + val if m else 0.0
        else:
            cmd_buf[env, 0, d] = val


@wp.kernel
def update_command_rel_kernel(
    root_quat_w: wp.array(dtype=wp.float32, ndim=2),  # [N, 4] in wxyz
    cmd_buf: wp.array(dtype=wp.float32, ndim=3),      # [N, 3, 12]
    cmd_mask: wp.array(dtype=wp.bool, ndim=2),        # [N, 12]
    rel: wp.array(dtype=wp.float32, ndim=2),          # [N, 12]
    num_envs: int,
):
    tid = wp.tid()
    if tid >= num_envs:
        return

    i = tid

    # ------------------------------------------------------------------
    # 0. load root quaternion (Isaac wxyz -> Warp xyzw)
    # ------------------------------------------------------------------
    qw = root_quat_w[i, 0]
    qx = root_quat_w[i, 1]
    qy = root_quat_w[i, 2]
    qz = root_quat_w[i, 3]

    q_root = wp.quat(qx, qy, qz, qw)
    q_root_inv = wp.quat_inverse(q_root)

    # ------------------------------------------------------------------
    # 1. orientation error (axis-angle in body frame)
    # ------------------------------------------------------------------
    # desired euler (roll, pitch, yaw) from command row
    roll = cmd_buf[i, 0, 3]
    pitch = cmd_buf[i, 0, 4]
    yaw = cmd_buf[i, 0, 5]

    # Warp's RPY matches IsaacLab convention as you stated
    q_des = wp.quat_rpy(roll, pitch, yaw)  # xyzw

    # q_err = q_root_inv * q_des (body frame error)
    q_err = quat_mul_wp(q_root_inv, q_des)

    axis, angle = wp.quat_to_axis_angle(q_err)
    rot_err = axis * angle  # vec3

    ex = rot_err[0]
    ey = rot_err[1]
    ez = rot_err[2]

    if not cmd_mask[i, 3]:
        ex = 0.0
    if not cmd_mask[i, 4]:
        ey = 0.0
    if not cmd_mask[i, 5]:
        ez = 0.0

    rel[i, 3] = ex
    rel[i, 4] = ey
    rel[i, 5] = ez

    cmd_buf[i, 1, 3] = ex
    cmd_buf[i, 1, 4] = ey
    cmd_buf[i, 1, 5] = ez

    # ------------------------------------------------------------------
    # 2. relative position (world -> body)
    # ------------------------------------------------------------------
    # world pose row (cmd_buf row 2) and desired pose row (row 0)
    px = cmd_buf[i, 2, 0]
    py = cmd_buf[i, 2, 1]
    pz = cmd_buf[i, 2, 2]

    dx = cmd_buf[i, 0, 0]
    dy = cmd_buf[i, 0, 1]
    dz = cmd_buf[i, 0, 2]

    rx = dx - px
    ry = dy - py
    rz = dz - pz

    v_w = wp.vec3(rx, ry, rz)

    if not cmd_mask[i, 0]:
        v_w[0] = 0.0
    if not cmd_mask[i, 1]:
        v_w[1] = 0.0
    if not cmd_mask[i, 2]:
        v_w[2] = 0.0

    v_b = wp.quat_rotate(q_root_inv, v_w)

    rel[i, 0] = v_b[0]
    rel[i, 1] = v_b[1]
    rel[i, 2] = v_b[2]

    cmd_buf[i, 1, 0] = v_b[0]
    cmd_buf[i, 1, 1] = v_b[1]
    cmd_buf[i, 1, 2] = v_b[2]

    # ------------------------------------------------------------------
    # 3. relative velocities (world -> body)
    # ------------------------------------------------------------------
    # desired/world vel slices are already written by Torch:
    #   cmd_buf[i, 0, 6:9] (lin), 0,9:12 (ang)
    #   cmd_buf[i, 2, 6:9],      2,9:12

    # linear vel diff
    lvx = cmd_buf[i, 0, 6] - cmd_buf[i, 2, 6]
    lvy = cmd_buf[i, 0, 7] - cmd_buf[i, 2, 7]
    lvz = cmd_buf[i, 0, 8] - cmd_buf[i, 2, 8]
    lv_w = wp.vec3(lvx, lvy, lvz)

    # angular vel diff
    avx = cmd_buf[i, 0, 9] - cmd_buf[i, 2, 9]
    avy = cmd_buf[i, 0, 10] - cmd_buf[i, 2, 10]
    avz = cmd_buf[i, 0, 11] - cmd_buf[i, 2, 11]
    av_w = wp.vec3(avx, avy, avz)

    # rotate into body
    lv_b = wp.quat_rotate(q_root_inv, lv_w)
    av_b = wp.quat_rotate(q_root_inv, av_w)

    # apply masks
    if not cmd_mask[i, 6]:
        lv_b[0] = 0.0
    if not cmd_mask[i, 7]:
        lv_b[1] = 0.0
    if not cmd_mask[i, 8]:
        lv_b[2] = 0.0

    if not cmd_mask[i, 9]:
        av_b[0] = 0.0
    if not cmd_mask[i, 10]:
        av_b[1] = 0.0
    if not cmd_mask[i, 11]:
        av_b[2] = 0.0

    rel[i, 6] = lv_b[0]
    rel[i, 7] = lv_b[1]
    rel[i, 8] = lv_b[2]
    rel[i, 9] = av_b[0]
    rel[i, 10] = av_b[1]
    rel[i, 11] = av_b[2]

    cmd_buf[i, 1, 6] = lv_b[0]
    cmd_buf[i, 1, 7] = lv_b[1]
    cmd_buf[i, 1, 8] = lv_b[2]
    cmd_buf[i, 1, 9] = av_b[0]
    cmd_buf[i, 1, 10] = av_b[1]
    cmd_buf[i, 1, 11] = av_b[2]


@wp.func
def quat_mul_wp(a: wp.quat, b: wp.quat) -> wp.quat:
    # a, b in xyzw
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]

    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]

    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw

    # Return in xyzw
    return wp.quat(x, y, z, w)
