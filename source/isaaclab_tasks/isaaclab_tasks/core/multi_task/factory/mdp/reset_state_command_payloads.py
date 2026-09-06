# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.warp.proxy_array import ProxyArray

from ...mdp.commands.state_command.reset_state_writer import ResetStateWriter
from ...utils.symmetry import Symmetry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from .reset_state_task_table import FactoryResetStateTaskTable


@wp.kernel
def _set_target_kernel(
    env_ids: wp.array(dtype=wp.int32),
    pos_w: wp.array(dtype=wp.vec3),
    quat_w: wp.array(dtype=wp.quatf),
    target_pos: wp.array(dtype=wp.vec3),
    target_quat: wp.array(dtype=wp.quatf),
):
    """Scatter sampled target poses into the per-env goal buffers."""
    i = wp.tid()
    e = env_ids[i]
    target_pos[e] = pos_w[i]
    target_quat[e] = quat_w[i]


@wp.kernel
def _command_update_kernel(
    held_pos: wp.array(dtype=wp.vec3),
    held_quat: wp.array(dtype=wp.quatf),
    robot_quat: wp.array(dtype=wp.quatf),
    target_pos: wp.array(dtype=wp.vec3),
    nearest_quat: wp.array(dtype=wp.quatf),
    orientation_error: wp.array(dtype=wp.float32),
    command: wp.array2d(dtype=wp.float32),
    error: wp.array2d(dtype=wp.float32),
):
    """Frame the symmetry-reduced alignment error as the robot-base-frame command.

    The :class:`~...utils.symmetry.Symmetry` has already written, per env,
    the NEAREST symmetry-equivalent target orientation (``nearest_quat``) and the
    geodesic ``orientation_error``. This kernel only does the observation framing:
    the position delta and the held->nearest rotation, both expressed in the robot
    base frame (one consistent frame), the rotation canonicalized to ``w >= 0``
    (no double-cover sign flip). At the goal it is ``(0, 0, 0, identity)``.
    """
    i = wp.tid()
    inv_robot = wp.quat_inverse(robot_quat[i])
    delta = target_pos[i] - held_pos[i]
    p_err = wp.quat_rotate(inv_robot, delta)
    q_world = wp.mul(nearest_quat[i], wp.quat_inverse(held_quat[i]))
    q_err = wp.mul(wp.mul(inv_robot, q_world), robot_quat[i])
    if q_err[3] < 0.0:
        q_err = wp.quatf(-q_err[0], -q_err[1], -q_err[2], -q_err[3])
    command[i, 0] = p_err[0]
    command[i, 1] = p_err[1]
    command[i, 2] = p_err[2]
    command[i, 3] = q_err[0]
    command[i, 4] = q_err[1]
    command[i, 5] = q_err[2]
    command[i, 6] = q_err[3]
    dist = wp.length(delta)
    error[i, 0] = orientation_error[i]
    error[i, 1] = dist


class FactoryAssemblyPayload:
    """Assembly progress and success semantics for the factory reset-state command.

    Owns the per-env goal pose, the symmetry-reduced alignment error, the
    threshold/hold-timer success state, and the held-asset debug marker. It
    interprets selected table rows, resolves their coordinate frame, and writes
    both spawn and target state; the command shell sees only opaque row ids.
    """

    error_names = ("orientation", "position")
    error_dim = 2
    command_dim = 7

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedRLEnv, table: FactoryResetStateTaskTable):
        payload_cfg = cfg.payload
        self._env = env
        self._device = env.device
        self._states_relative = cfg.states_relative
        self.table = table
        self.reset_assets = tuple(cfg.reset_assets)
        if table.states.layout.names != self.reset_assets:
            raise ValueError(
                "Factory table layout must exactly match StateCommandCfg.reset_assets: "
                f"{table.states.layout.names} != {self.reset_assets}."
            )
        self._reset_state_writer = ResetStateWriter(env, table.states, self.reset_assets, cfg.states_relative)
        self.held_asset: Articulation | RigidObject = env.scene[payload_cfg.held_asset_cfg.name]
        self.fixed_asset: Articulation | RigidObject = env.scene[payload_cfg.fixed_asset_cfg.name]
        self.robot: Articulation = env.scene[payload_cfg.robot_cfg.name]
        self._held_asset_index = table.states.layout.entity_index(payload_cfg.held_asset_cfg.name)

        # symmetry reducer: one asset type for the single-held-asset factory.
        # The single-cyclic fast path ignores type_id; the zero buffer keeps the
        # generic reducer call shape for non-cyclic/custom symmetry.
        self._symmetry = Symmetry([payload_cfg.symmetry], str(env.device))
        self._type_id = wp.zeros(env.num_envs, dtype=wp.int32, device=str(env.device))

        # success / hold-timer state (owned here -- the command reads it back via
        # get_task_done / get_task_reward / command_std)
        self.is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.duration_required = torch.zeros(env.num_envs, device=env.device)
        self.duration_held = torch.zeros(env.num_envs, device=env.device)

        # per-env active command variant: thresholds + active-channel mask + hold range
        self._command_names = list(cfg.commands.keys())
        self.randomize_command_indices = bool(cfg.randomize_command_indices)
        self.command_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._command_masks = torch.ones(len(self._command_names), self.error_dim, device=env.device, dtype=torch.bool)
        self._command_thresholds = torch.empty(len(self._command_names), self.error_dim, device=env.device)
        self._duration_ranges = torch.empty(len(self._command_names), 2, device=env.device)
        for command_idx, command in enumerate(cfg.commands.values()):
            self._command_thresholds[command_idx] = torch.tensor(
                (command.orientation_threshold, command.position_threshold), device=env.device
            )
            self._duration_ranges[command_idx] = torch.tensor(command.duration, device=env.device)
        self.cmd_mask = torch.zeros(env.num_envs, self.error_dim, device=env.device, dtype=torch.bool)
        self.command_thresholds = torch.empty(env.num_envs, self.error_dim, device=env.device)

        # warp-native numeric buffers wrapped as ProxyArray: ``.warp`` for the
        # kernels, cached zero-copy ``.torch`` view for the torch-side readers
        dev = str(env.device)
        target_quat = wp.zeros(env.num_envs, dtype=wp.quatf, device=dev)
        target_quat.fill_(wp.quatf(0.0, 0.0, 0.0, 1.0))
        self.target_pos = ProxyArray(wp.zeros(env.num_envs, dtype=wp.vec3, device=dev))
        self.target_quat = ProxyArray(target_quat)
        self.orientation_error = ProxyArray(wp.zeros(env.num_envs, dtype=wp.float32, device=dev))
        self._nearest_quat = ProxyArray(wp.zeros(env.num_envs, dtype=wp.quatf, device=dev))

        self._viz_cfg = payload_cfg.held_asset_visualizer_cfg

    def command_std(self) -> torch.Tensor:
        """Per-env success thresholds ``[N, 2]``: orientation [rad], position [m]."""
        return self.command_thresholds

    def get_task_done(self) -> torch.Tensor:
        """Per-env success: pose within threshold and held past the required duration."""
        return self.is_success

    def get_task_reward(self) -> torch.Tensor:
        """Sparse success reward: 1 when :meth:`get_task_done`, else 0."""
        return self.is_success.float()

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample task rows through the factory table's policy."""
        return self.table.sample_rows(count)

    def bind(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Bind selected assembly rows and write their simulator reset state."""
        spawn_rows, target_rows = self.table.gather(task_rows)
        target_origin = self._env.scene.env_origins[env_ids] if self._states_relative else None
        self._bind_target(env_ids, target_rows, target_origin)
        self._reset_state_writer.write(env_ids, spawn_rows)

    def bind_target(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Bind selected assembly targets and write their target simulator state."""
        _, target_rows = self.table.gather(task_rows)
        target_origin = self._env.scene.env_origins[env_ids] if self._states_relative else None
        self._bind_target(env_ids, target_rows, target_origin)
        self._reset_state_writer.write(env_ids, target_rows)

    def _bind_target(
        self,
        env_ids: torch.Tensor,
        target_rows: torch.Tensor,
        target_origin: torch.Tensor | None,
    ) -> None:
        """Sample the command variant + hold time and set the goal held-asset pose.

        The held-asset pose is read by entity index and lifted to world by
        target_origin when the table stores environment-local states.
        """
        if self.randomize_command_indices:
            self.command_indices[env_ids] = torch.randint(
                0, len(self._command_names), (env_ids.numel(),), device=self._device
            )
        self.cmd_mask[env_ids] = self._command_masks[self.command_indices[env_ids]]
        self.command_thresholds[env_ids] = self._command_thresholds[self.command_indices[env_ids]]
        ranges = self._duration_ranges[self.command_indices[env_ids]]
        self.duration_required[env_ids] = torch.empty(env_ids.numel(), device=env_ids.device).uniform_(0.0, 1.0)
        self.duration_required[env_ids] *= ranges[:, 1] - ranges[:, 0]
        self.duration_required[env_ids] += ranges[:, 0]
        self.duration_held[env_ids] = 0.0

        target_pose = self.table.states.root_pose[target_rows, self._held_asset_index]
        target_pos_w = target_pose[:, :3]
        if target_origin is not None:
            target_pos_w = target_pos_w + target_origin
        self.set_target(env_ids, target_pos_w, target_pose[:, 3:7])

    def set_target(self, env_ids: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Scatter the goal held-asset world pose into ``env_ids`` (the sampled target slot)."""
        wp.launch(
            _set_target_kernel,
            dim=env_ids.shape[0],
            inputs=[
                wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32),
                wp.from_torch(pos_w.contiguous().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(quat_w.contiguous().view(-1, 4), dtype=wp.quatf),
                self.target_pos.warp,
                self.target_quat.warp,
            ],
            device=str(self._device),
        )

    def update(self, step_dt: float, command_out: torch.Tensor, error_out: torch.Tensor) -> None:
        """Refresh the command observation, error, and threshold/hold success.

        Command = where the TARGET held pose is relative to the CURRENT held pose:
        the position delta in the robot base frame [m] plus the remaining rotation
        as a quaternion (xyzw). Success error is the symmetry-reduced orientation
        angle [rad] and the position distance [m]. The hold timer advances by
        ``step_dt`` while all active error groups are within threshold;
        :attr:`is_success` becomes true once it passes the per-env required duration
        and clears whenever the pose leaves the active thresholds.

        ``wp.from_torch`` is a zero-copy reinterpret, so no host work happens here.
        """
        held_quat = self.held_asset.data.root_quat_w.warp
        # symmetry-reduced orientation error + nearest-equivalent target
        self._symmetry.reduce_orientation(
            held_quat,
            self.target_quat.warp,
            self._type_id,
            self.orientation_error.warp,
            self._nearest_quat.warp,
        )
        # frame the result as the robot-base-frame command observation
        wp.launch(
            _command_update_kernel,
            dim=command_out.shape[0],
            inputs=[
                self.held_asset.data.root_pos_w.warp,
                held_quat,
                self.robot.data.root_quat_w.warp,
                self.target_pos.warp,
                self._nearest_quat.warp,
                self.orientation_error.warp,
                wp.from_torch(command_out),
                wp.from_torch(error_out),
            ],
            device=str(self._device),
        )
        # threshold + hold-timer success (masked-off error groups always pass)
        active_success = (error_out < self.command_thresholds) | ~self.cmd_mask
        instant_success = torch.all(active_success, dim=1)
        self.duration_held[:] = torch.where(instant_success, self.duration_held + step_dt, 0.0)
        self.is_success[:] = instant_success & (self.duration_held >= self.duration_required)

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Create (lazily) and toggle the held-asset target-frame marker."""
        if debug_vis:
            from isaaclab.markers import VisualizationMarkers

            if not hasattr(self, "held_asset_visualizer"):
                self.held_asset_visualizer = VisualizationMarkers(self._viz_cfg)
            self.held_asset_visualizer.set_visibility(True)
        elif hasattr(self, "held_asset_visualizer"):
            self.held_asset_visualizer.set_visibility(False)

    def debug_visualize(self, env: ManagerBasedRLEnv) -> None:
        """Draw the sampled target root-frame pose for the held asset."""
        if not self.fixed_asset.is_initialized:
            return
        self.held_asset_visualizer.visualize(
            translations=self.target_pos.torch,
            orientations=self.target_quat.torch,
        )
