# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .success_monitor_cfg import SuccessMonitorCfg

from ..assembly_data import Offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.datasets import EpisodeData

    from .commands import TaskCommand

# # viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


def get_nested_value(data: dict, keys: list):
    for key in keys:
        data = data[key]
    return data


def ee_held_asset_distance_tanh(
    env: ManagerBasedRLEnv,
    root_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    root_asset_offset: Offset,
    target_asset_offset: Offset,
    std: float = 0.1,
    use_rotation: bool = False,
    failure_rate_weight: Optional[str] = None,
) -> torch.Tensor:
    if failure_rate_weight:
        weight = eval(failure_rate_weight)
    else:
        weight = 1.0

    # log success rates and weights
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Metrics/ee_held_asset_distance_tanh_weight"] = weight

    root_asset: Articulation = env.scene[root_asset_cfg.name]
    root_asset_alignment_pos_w, root_asset_alignment_quat_w = root_asset_offset.combine(
        root_asset.data.body_link_pos_w[:, root_asset_cfg.body_ids].view(-1, 3),
        root_asset.data.body_link_quat_w[:, root_asset_cfg.body_ids].view(-1, 4),
    )
    target_asset_alignment_pos_w, target_asset_alignment_quat_w = target_asset_offset.apply(
        env.scene[target_asset_cfg.name]
    )
    target_asset_in_root_asset_frame_pos, target_asset_in_root_asset_frame_angle_axis = math_utils.compute_pose_error(
        root_asset_alignment_pos_w,
        root_asset_alignment_quat_w,
        target_asset_alignment_pos_w,
        target_asset_alignment_quat_w,
    )

    pos_distance = torch.norm(target_asset_in_root_asset_frame_pos, dim=1)
    if use_rotation:
        target_asset_in_root_asset_frame_angle_axis[:, 2] = 0.0
        rot_distance = torch.norm(target_asset_in_root_asset_frame_angle_axis, dim=1)
        return weight * (1 - torch.tanh((pos_distance + rot_distance) / std))
    else:
        return weight * (1 - torch.tanh(pos_distance / std))


class ProgressContext(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.held_asset: Articulation | RigidObject = env.scene[cfg.params.get("held_asset_cfg").name]  # type: ignore
        self.fixed_asset: Articulation | RigidObject = env.scene[cfg.params.get("fixed_asset_cfg").name]  # type: ignore
        self.held_asset_offset: Offset = cfg.params.get("held_asset_offset")  # type: ignore
        self.fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_centered = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.z_distance_reached = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_diff = torch.zeros((env.num_envs), device=env.device)
        self.xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.last_z_distance = torch.zeros((env.num_envs), device=env.device)
        self.z_distance = torch.zeros((env.num_envs), device=env.device)
        self.z_progress = torch.zeros((env.num_envs), device=env.device)
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100_000,
            num_monitored_data=1,
            device=env.device
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self.failure_rate = 1.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        held_asset_cfg: SceneEntityCfg,
        fixed_asset_cfg: SceneEntityCfg,
        held_asset_offset: Offset,
        fixed_asset_offset: Offset,
        command_context: str = "task_command",
    ) -> torch.Tensor:
        task_command: TaskCommand = env.command_manager.get_term(command_context)
        success_threshold = task_command.success_threshold
        held_asset_alignment_pos_w, held_asset_alignment_quat_w = self.held_asset_offset.apply(self.held_asset)
        fixed_asset_alignment_pos_w, fixed_asset_alignment_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)
        held_asset_in_fixed_asset_frame_pos, held_asset_in_fixed_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                fixed_asset_alignment_pos_w,
                fixed_asset_alignment_quat_w,
                held_asset_alignment_pos_w,
                held_asset_alignment_quat_w,
            )
        )
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(held_asset_in_fixed_asset_frame_quat)
        self.euler_xy_diff[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xy_distance[:] = torch.norm(held_asset_in_fixed_asset_frame_pos[:, 0:2], dim=1)
        self.z_distance[:] = held_asset_in_fixed_asset_frame_pos[:, 2]
        self.z_progress[:] = self.z_distance - self.last_z_distance
        self.last_z_distance[:] = self.z_distance
        self.orientation_aligned[:] = self.euler_xy_diff < 0.025
        self.position_centered[:] = self.xy_distance < 0.0025
        self.z_distance_reached[:] = self.z_distance < success_threshold
        self.success[:] = self.orientation_aligned & self.position_centered & self.z_distance_reached

        # Update continuous success counter
        self.continuous_success_counter[:] = torch.where(
            self.success,
            self.continuous_success_counter + 1,
            torch.zeros_like(self.continuous_success_counter)
        )

        # Update success monitor
        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device),
            self.success
        )
        success_rate = self.success_monitor.get_success_rate()
        self.failure_rate = (1 - success_rate).clamp(min=1e-6).item()

        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"][f"Metrics/failure_rate"] = self.failure_rate

        return torch.zeros(env.num_envs, device=env.device)


def progress_dense(
    env: ManagerBasedRLEnv,
    std: float,
    context: str = "progress_context",
    failure_rate_weight: Optional[str] = None,
) -> torch.Tensor:
    if failure_rate_weight:
        weight = eval(failure_rate_weight)
    else:
        weight = 1.0

    # log success rates and weights
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Metrics/progress_dense_weight"] = weight

    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    angle_diff: torch.Tensor = getattr(context_term, "euler_xy_diff")
    xy_distance: torch.Tensor = getattr(context_term, "xy_distance")
    z_distance: torch.Tensor = getattr(context_term, "z_distance")
    z_distance = torch.abs(z_distance)

    # Normalize the distances by std
    angle_diff = torch.exp(-angle_diff / std)
    xy_distance = torch.exp(-xy_distance / std)
    z_distance = torch.exp(-z_distance / std)
    # Stack the tensors and compute the mean along the new dimension
    stacked = torch.stack([angle_diff, xy_distance, z_distance], dim=0)
    return weight * torch.mean(stacked, dim=0)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")
    return torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)


def joint_torques_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1), 0, 1e4)


def joint_acc_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1), 0, 1e4)


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1), 0, 1e4)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.clamp(
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1), 0, 1e4
    )


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.clamp(torch.sum(torch.square(env.action_manager.action), dim=1), 0, 1e4)


def processed_action_thresholded_l2_clamped(env: ManagerBasedRLEnv, name: str = "jointpos", threshold: float = 0.1) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    processed_actions = env.action_manager._terms[name].processed_actions
    thresholded_processed_actions = torch.clamp(torch.abs(processed_actions) - threshold, 0, 1e4)
    return torch.clamp(torch.sum(torch.square(thresholded_processed_actions), dim=1), 0, 1e4)

def joint_force_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Compute L2 norm of joint forces with clamping for reward penalization."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.joint_ids]
    force_magnitudes = torch.norm(joint_forces, dim=2)
    total_force = torch.sum(force_magnitudes, dim=1)
    return torch.clamp(total_force, 0, 1e4)
