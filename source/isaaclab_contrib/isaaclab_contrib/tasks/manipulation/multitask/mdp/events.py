# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedEnv


def reset_multitask_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """

    def _asset_env_mapping(asset: AssetBase, requested_envs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map global environment indices to local indices for the asset.
        Args:
            asset: The asset to map the environment indices for.
            requested_envs: The global environment indices to filter.
        Returns:
            A tuple of the local environment indices and the global environment indices.
        """
        assigned_envs = getattr(asset, "assigned_envs", ())
        if len(assigned_envs) > 0:
            local_indices = asset._filter_env_ids(requested_envs)
            requested_list = requested_envs.cpu().tolist()
            global_indices = torch.tensor(
                [e for e in requested_list if e in assigned_envs],
                dtype=torch.long,
                device=requested_envs.device,
            )
            return local_indices, global_indices
        return requested_envs, requested_envs

    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        local_ids, global_ids = _asset_env_mapping(rigid_object, env_ids)
        if local_ids.numel() == 0:
            continue
        default_root_state = rigid_object.data.default_root_state[local_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[global_ids]
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=global_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=global_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        local_ids, global_ids = _asset_env_mapping(articulation_asset, env_ids)
        if local_ids.numel() == 0:
            continue
        default_root_state = articulation_asset.data.default_root_state[local_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[global_ids]
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=global_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=global_ids)
        default_joint_pos = articulation_asset.data.default_joint_pos[local_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[local_ids].clone()
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=global_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=global_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=global_ids)


def reset_multitask_robot_init_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
):
    """Reset robot root state from per-env init_state buffer."""
    init_state_tensor = getattr(env.cfg, "_robot_init_state_tensor", None)
    if init_state_tensor is None:
        return
    articulation = env.scene.articulations.get("robot")
    if articulation is None:
        return

    init_state_tensor = init_state_tensor.to(device=articulation.device)
    root_state = init_state_tensor[env_ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids].to(device=articulation.device)
    articulation.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    articulation.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    # Update default_root_state with env-relative values for future resets.
    articulation.data.default_root_state[env_ids] = init_state_tensor[env_ids]
