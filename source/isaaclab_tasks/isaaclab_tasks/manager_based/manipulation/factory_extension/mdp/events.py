# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import inspect
from typing import TYPE_CHECKING

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from ..state_machine import utils as pose_math_utils
from . import key_point_maths

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from isaaclab.envs import ManagerBasedRLEnv

    from .data import KeyPointsTracker
    from .data_cfg import KeyPointDataCfg, DataCfg
    from ..tasks import Offset

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


def reset_collection_asset(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    offset: Offset,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    asset: RigidObjectCollection = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_object_state[env_ids, asset_cfg.object_collection_ids].clone()
    num_objs = len(asset_cfg.object_collection_ids)

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    root_pos = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    root_quat = root_states[:, 3:7]

    offset_pose = torch.tensor(offset.pose, device=asset.device).repeat(len(env_ids) * num_objs, 1)
    target_pos, target_quat = math_utils.combine_frame_transforms(
        root_pos.view(-1, 3), root_quat.view(-1, 4), offset_pose[:, :3], offset_pose[:, 3:],
    )

    positions = target_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(target_quat, orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples
    root_pose = pose_math_utils.offset_subtract(
        torch.cat([positions.view(-1, 3), orientations.view(-1, 4)], dim=1), offset_pose
    )

    # set into the physics simulation
    asset.write_object_pose_to_sim(
        object_pose=root_pose.view(len(env_ids), num_objs, 7),
        env_ids=env_ids,
        object_ids=asset_cfg.object_collection_ids
    )
    asset.write_object_velocity_to_sim(
        velocities.view(len(env_ids), num_objs, 6),
        env_ids=env_ids,
        object_ids=asset_cfg.object_collection_ids
    )


def reset_assets(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    asset_reset_key_points_cfg: KeyPointDataCfg,
) -> None:
    data_term: KeyPointsTracker = env.data_manager.get_term(asset_reset_key_points_cfg.term)
    data_term._update_data(env_ids, reset=True)
    # (all_envs, n_key_points, n_assets, n_offsets(1), 7)
    reset_kp, reset_kp_mask = asset_reset_key_points_cfg.get(env.data_manager)
    # (n_envs, n_key_points, n_assets, 7), (n_envs, n_key_points, n_assets)
    reset_kp, reset_kp_mask = reset_kp[env_ids, :, :, 0], reset_kp_mask[env_ids, :, :, 0]
    chosen_reset_kp, _ = key_point_maths.select_valid(reset_kp, reset_kp_mask, dim=2, strategy="random")
    assets: RigidObjectCollection = env.scene[asset_cfg.name]
    collection_vel = assets.data.object_vel_w[env_ids[:, None], asset_cfg.object_collection_ids, :6].clone()
    collection_vel[:] = 0.0
    assets.write_object_pose_to_sim(chosen_reset_kp, env_ids, object_ids=asset_cfg.object_collection_ids)
    assets.write_object_velocity_to_sim(collection_vel, env_ids, object_ids=asset_cfg.object_collection_ids)


def reset_task_assets(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    asset_reset_spec_cfg: KeyPointDataCfg,
    asset_reset_kp_offset_cfg: KeyPointDataCfg,
    align_asset_id_cfg: KeyPointDataCfg,
):
    assets: RigidObjectCollection = env.scene[asset_cfg.name]
    align_asset_id, align_asset_mask = align_asset_id_cfg.get(env.data_manager)  # (num_envs, 1, max_num_assets)
    offsets, offset_masks = asset_reset_kp_offset_cfg.get(env.data_manager)  # (n_contexts, n_key_points, n_assets, n_offsets, 7)
    asset_reset_spec = asset_reset_spec_cfg.get(env.data_manager)

    align_asset_id, align_asset_mask = align_asset_id.squeeze(1)[env_ids], align_asset_mask.squeeze(1)[env_ids]  # (num_env_id, max_num_assets)
    chosen_assets_id, _ = key_point_maths.select_valid(align_asset_id, align_asset_mask, dim=1, strategy="first")  # (num_env_id,)
    collection_id_to_reset_id_list = [asset_reset_spec.key_points_name.index(n) if n in asset_reset_spec.key_points_name else -1 for n in assets.data.object_names]
    collection_id_to_reset_id = torch.tensor(collection_id_to_reset_id_list, device=env.device)
    reset_kp_ids = collection_id_to_reset_id[chosen_assets_id]   # (num_env_id,)

    expanded_offsets = offsets.unsqueeze(0).repeat(len(env_ids), 1, 1, 1, 1, 1)  # (num_env_ids, num_contexts, n_kp, num_assets, n_offsets, 7)
    expanded_offset_masks = offset_masks.unsqueeze(0).repeat(len(env_ids), 1, 1, 1, 1)  # (num_env_ids, num_contexts, n_kp, num_assets, n_offsets)
    # (num_contexts, num_key_points, max_num_assets) -> (num_env_ids, num_contexts, num_key_points, max_num_assets)
    expanded_reset_kp_asset_id = asset_reset_spec.key_points_asset_id.unsqueeze(0).repeat(len(reset_kp_ids), 1, 1, 1)
    expanded_reset_kp_asset_id_mask = asset_reset_spec.key_points_asset_id_mask.unsqueeze(0).repeat(len(reset_kp_ids), 1, 1, 1)

    num_env_ids, num_contexts, num_key_points, num_assets, num_offsets = expanded_offset_masks.size()
    gather_reset_kp_ids = reset_kp_ids[:, None, None, None].expand(num_env_ids, num_contexts, 1, num_assets)
    # (num_env_ids, num_contexts, num_assets)
    asset_id = expanded_reset_kp_asset_id.gather(dim=2, index=gather_reset_kp_ids).squeeze(2)
    asset_id_mask = expanded_reset_kp_asset_id_mask.gather(dim=2, index=gather_reset_kp_ids).squeeze(2)
    filtered_asset_id_mask = (asset_id != assets.data.object_names.index('nist_board')) & asset_id_mask
    # (num_env_ids, num_contexts, num_assets, n_offsets, 7)
    gather_reset_kp_ids = reset_kp_ids[:, None, None, None, None].expand(num_env_ids, num_contexts, 1, num_assets, num_offsets)
    kp_sel_offsets = expanded_offsets.gather(dim=2, index=gather_reset_kp_ids[..., None].expand(*gather_reset_kp_ids.shape, 7)).squeeze(2)
    kp_sel_offset_masks = expanded_offset_masks.gather(dim=2, index=gather_reset_kp_ids).squeeze(2)
    # (num_env_ids, num_contexts, num_assets, 7)
    off_sel_kp_sel_offsets, _ = key_point_maths.select_valid(kp_sel_offsets, kp_sel_offset_masks, dim=3, strategy="random")

    pick_idx = key_point_maths.select_random_valid_idx(filtered_asset_id_mask.view(len(env_ids), -1), dim=1)
    selected_offset = off_sel_kp_sel_offsets.view(len(env_ids), -1, 7)[torch.arange(len(env_ids)), pick_idx]
    final_selected_asset_id = asset_id.view(len(env_ids), -1)[torch.arange(len(env_ids)), pick_idx]

    key_points_src_reset = assets.data.object_state_w[env_ids, :, :7].clone()
    task_key_points_target_pos, task_key_points_target_quat = math_utils.combine_frame_transforms(
        key_points_src_reset[torch.arange(len(env_ids)), final_selected_asset_id, :3].view(-1, 3),
        key_points_src_reset[torch.arange(len(env_ids)), final_selected_asset_id, 3:].view(-1, 4),
        selected_offset[:, :3],
        selected_offset[:, 3:],
    )
    key_points_src_reset[torch.arange(len(env_ids)), chosen_assets_id] = torch.cat([task_key_points_target_pos, task_key_points_target_quat], dim=1)
    assets.write_object_pose_to_sim(key_points_src_reset, env_ids)

class ChainedResetTerms(ManagerTermBase):

    def __init__(self, cfg: EventTermCfg, env: DataManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terms: dict[str, callable] = cfg.params["terms"]  # type: ignore
        self.params: dict[str, dict[str, any]] = cfg.params["params"]  # type: ignore
        self.class_terms = {}

        for term_name, term_func in self.terms.items():
            if inspect.isclass(term_func):
                self.class_terms[term_name] = term_func

        for term_name, term_cfg in self.params.items():
            for val in term_cfg.values():
                if isinstance(val, SceneEntityCfg):
                    val.resolve(env.scene)

        class ParamsAttrMock:
            def __init__(self, params):
                self.params = params

        for term_name, term_cls in self.class_terms.items():
            params_attr_mock = ParamsAttrMock(self.params[term_name])
            self.terms[term_name] = term_cls(params_attr_mock, env)

    def __call__(
        self,
        env: DataManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, callable],
        params: dict[str, dict[str, any]],
        probability: float = 1.0,
    ) -> None:
        keep = torch.rand(env_ids.size(0), device=env_ids.device) < probability
        if not keep.any():
            return
        env_ids_to_reset = env_ids[keep]
        for func_name, func in terms.items():
            func(env, env_ids_to_reset, **params[func_name])  # type: ignore
            for name, term in self._env.data_manager._terms.items():
                term._update_data(env_ids)


def reset_held_asset_against(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    aligning_point_offset_cfg: KeyPointDataCfg,
    aligning_point_asset_id_cfg: KeyPointDataCfg,
    point_align_against_cfg: KeyPointDataCfg,
    held_asset_range: dict[str, tuple[float, float]] = {},
):
    env_id_arange = torch.arange(env_ids.shape[0], device=env_ids.device)
    # (n_envs, n_key_points(1), n_assets, n_offsets, 7), (n_envs, n_key_points(1), n_assets, n_offsets)
    aligning_offset, aligning_offset_mask = aligning_point_offset_cfg.get(env.data_manager)
    point_align_against_pose_w, point_align_against_mask = point_align_against_cfg.get(env.data_manager)
    # (n_envs, 1, n_assets), (n_envs, 1, n_assets)
    aligning_point_asset_id, aligning_point_asset_id_mask = aligning_point_asset_id_cfg.get(env.data_manager)

    env_len = len(env_ids)
    aligning_offset = aligning_offset.squeeze(1)[env_ids].view(env_len, -1, 7)  # (n_envs, n_assets * n_offsets, 7)
    aligning_offset_mask = aligning_offset_mask.squeeze(1)[env_ids].view(env_len, -1)  # (n_envs, n_assets * n_offsets)
    point_align_against_pose_w = point_align_against_pose_w.squeeze(1)[env_ids].view(env_len, -1, 7)  # (n_envs, n_assets * n_offsets, 7)
    point_align_against_mask = point_align_against_mask.squeeze(1)[env_ids].view(env_len, -1)  # (n_envs, n_assets * n_offsets)
    aligning_point_asset_id = aligning_point_asset_id.squeeze(1)[env_ids]  # (n_envs, n_assets)
    aligning_point_asset_id_mask = aligning_point_asset_id_mask.squeeze(1)[env_ids]  # (n_envs, n_assets)

    point_align_against_pose_w, _ = key_point_maths.select_valid(point_align_against_pose_w, point_align_against_mask, dim=1, strategy="first")
    aligning_offset, _ = key_point_maths.select_valid(aligning_offset, aligning_offset_mask, dim=1)
    chosen_assets, _ = key_point_maths.select_valid(aligning_point_asset_id, aligning_point_asset_id_mask, dim=1)

    translated_held_asset_pose = pose_math_utils.offset_subtract(point_align_against_pose_w, aligning_offset)

    # Add randomization
    range_list = [held_asset_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    new_pos_w = translated_held_asset_pose[:, :3] + samples[:, 0:3]
    quat_b = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    new_quat_w = math_utils.quat_mul(translated_held_asset_pose[:, 3:], quat_b)

    # reset the held asset pose
    assets: RigidObjectCollection = env.scene["assets"]
    assets_pose = assets.data.object_state_w[env_ids, :, :7].clone()
    assets_pose[env_id_arange, chosen_assets] = torch.cat([new_pos_w, new_quat_w], dim=1)
    assets.write_object_pose_to_sim(assets_pose, env_ids=env_ids)  # type: ignore


def grasp_held_asset(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_diameter_cfg: DataCfg,
) -> None:
    held_asset_diameter: torch.Tensor = held_asset_diameter_cfg.get(env.data_manager)
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][env_ids].clone().detach()
    joint_pos[:, :] = held_asset_diameter[env_ids].unsqueeze(1) / 2 * 1.0
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), robot_cfg.joint_ids, env_ids)  # type: ignore


def reset_attachments(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    attach_to_asset_cfg: SceneEntityCfg,
):
    asset: RigidObjectCollection = env.scene[asset_cfg.name]
    attach_to_asset: Articulation = env.scene[attach_to_asset_cfg.name]
    target_pose = attach_to_asset.data.body_link_state_w[env_ids[:, None], attach_to_asset_cfg.body_ids, :7]
    asset.write_object_pose_to_sim(target_pose, env_ids=env_ids, object_ids=asset_cfg.object_collection_ids)


class reset_end_effector_round_fixed_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: DataManagerBasedRLEnv):
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore

    def __call__(
        self,
        env: DataManagerBasedRLEnv,
        env_ids: torch.Tensor,
        ee_reset_point_cfg: KeyPointDataCfg,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
    ) -> None:

        reset_point_pose_w, reset_point_mask = ee_reset_point_cfg.get(env.data_manager)
        reset_point_pose_w = reset_point_pose_w.squeeze(1)[env_ids].view(len(env_ids), -1, 7)  # (n_envs, n_assets, n_offsets, 7)
        reset_point_mask = reset_point_mask.squeeze(1)[env_ids].view(len(env_ids), -1)  # (n_envs, n_assets, n_offsets)

        reset_point_pose_w, _ = key_point_maths.select_valid(reset_point_pose_w, reset_point_mask, dim=1, strategy="first")

        # add randomization around the key point
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        pos_w = reset_point_pose_w[:, :3] + samples[:, 0:3]

        # for those non_reset_id, we will let ik solve for its current position
        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids], self.robot.data.root_link_quat_w[env_ids], pos_w, quat_w
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Error Rate 75% ^ 10 = 0.05 (final error)
        for _ in range(10):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )
