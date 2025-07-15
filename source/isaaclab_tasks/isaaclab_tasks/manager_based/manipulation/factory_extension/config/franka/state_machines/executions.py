from typing import TYPE_CHECKING, Optional
import torch
import isaaclab.utils.math as math_utils

from ....state_machine import utils as pose_utils
from ....state_machine import skills
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from ....mdp.data_cfg import AlignmentDataCfg as Align
from ....mdp.data_cfg import KeyPointDataCfg as Kp
from ....mdp.data_cfg import AlignmentMetric
from ....mdp import key_point_maths as key_point_utils
from ....tasks import Offset
if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection


# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform_execution"))


def gripper_action(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    gripper_command: float
):
    return gripper_command


def gripper_stay_still(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    action_idx: int
):
    last_act = env.action_manager.prev_action[env_ids, action_idx]
    return last_act


def ee_stay_still(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ee_src_cfg: Kp,
):
    ee_src, _ = ee_src_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    ee_src = ee_src[env_ids, 0, 0, 0, :]
    return ee_src


def lift_up_execution(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    supporting_asset_cfg: SceneEntityCfg,
    object_holding_kp_cfg: Kp,
    object_holding_offset_cfg: Kp,
):
    object_holding_kp, _ = object_holding_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_kp = object_holding_kp[env_ids, 0, 0, 0, :]
    object_holding_offset, _ = object_holding_offset_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_offset = object_holding_offset[env_ids, 0, 0, 0, :]
    support_asset: RigidObjectCollection = env.scene[supporting_asset_cfg.name]
    support_pose = support_asset.data.object_state_w[env_ids, supporting_asset_cfg.object_collection_ids, :7].view(-1, 7)
    up_pose = torch.tensor([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env_ids.shape[0], 1)

    lifted_up_ee_pose = pose_utils.poses_combine(support_pose, up_pose)

    object_holding_des = pose_utils.correct_pose(
        cur_pose=object_holding_kp,
        des_pose=lifted_up_ee_pose,
        pos_mask=torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(len(env_ids), 1),
        rot_mask=torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(len(env_ids), 1)
    )

    ee_src_destination = pose_utils.offset_subtract(
        align_against_pose_w=object_holding_des,
        aligning_offset_pose=object_holding_offset,
    )

    return ee_src_destination


def insertion_execution(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    entry_alignment_cfg: Align,
    grasp_alignment_cfg: Align,
    entry_kp_cfg: Kp,
    align_kp_cfg: Kp,
    align_src_cfg: Kp,
    align_offset_cfg: Kp,
    asset_align_kp_cfg: Kp,
    object_holding_kp_cfg: Kp,
    object_holding_offset_cfg: Kp,
    object_holding_src_cfg: Kp,
):
    des_ee_pose = align_holding_asset(
        env,
        env_ids,
        entry_alignment_cfg,
        grasp_alignment_cfg,
        entry_kp_cfg,
        align_src_cfg,
        align_offset_cfg,
        asset_align_kp_cfg,
        object_holding_kp_cfg,
        object_holding_offset_cfg
    )

    align_kp, _ = align_kp_cfg.get(env.data_manager)
    cur_ee_kp, _ = object_holding_src_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    grasp_met: AlignmentMetric.AlignmentData = grasp_alignment_cfg.get(env.data_manager)  # type: ignore
    cur_ee_kp = cur_ee_kp[env_ids, 0, grasp_met.align_asset_index[env_ids], grasp_met.align_offset_index[env_ids]]

    entry_met: AlignmentMetric.AlignmentData = entry_alignment_cfg.get(env.data_manager)
    align_asset_id, align_offset_id = entry_met.align_asset_index[env_ids], entry_met.align_offset_index[env_ids]
    insertion_kp = align_kp[env_ids, 0, align_asset_id, align_offset_id]

    insertion_vector = torch.zeros_like(insertion_kp[:, :3])
    delta_magnitude = torch.norm(entry_met.pos_delta[env_ids], dim=1).clamp(max=0.001)
    insertion_vector[:, 2] = -delta_magnitude
    inserted_pos, _ = math_utils.combine_frame_transforms(
        insertion_kp[:, :3],
        insertion_kp[:, 3:],
        insertion_vector,)

    corrected_position = torch.where(  # correction
        (entry_met.pos_delta[env_ids, 2] > -0.00015).unsqueeze(-1),
        cur_ee_kp[:, :3] - entry_met.pos_delta[env_ids, :3],
        cur_ee_kp[:, :3]
    )
    corrected_position += (inserted_pos - insertion_kp[:, :3])  # insertion

    return torch.cat((corrected_position, des_ee_pose[:, 3:]), dim=1)


def align_holding_asset(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_alignment_cfg: Align,
    grasp_alignment_cfg: Align,
    asset_align_against_kp_cfg: Kp,
    asset_align_src_cfg: Kp,
    asset_align_offset_cfg: Kp,
    asset_align_kp_cfg: Kp,
    object_holding_kp_cfg: Kp,
    object_holding_offset_cfg: Kp,
    offset: Optional[Offset] = None,
):
    asset_align_against_kp, asset_align_against_mask = asset_align_against_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    align_src, _ = asset_align_src_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    align_offset, _ = asset_align_offset_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    asset_align_kp, asset_align_mask = asset_align_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_kp, object_holding_mask = object_holding_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_offset, _ = object_holding_offset_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)

    asset_align_against_kp, asset_align_against_mask = asset_align_against_kp[env_ids, 0], asset_align_against_mask[env_ids, 0]
    align_src = align_src[env_ids, 0]
    align_offset = align_offset[env_ids, 0]
    asset_align_kp, asset_align_mask = asset_align_kp[env_ids, 0], asset_align_mask[env_ids, 0]
    object_holding_kp, object_holding_mask = object_holding_kp[env_ids, 0], object_holding_mask[env_ids, 0]
    object_holding_offset = object_holding_offset[env_ids, 0]

    asset_align_against_met: AlignmentMetric.AlignmentData = asset_alignment_cfg.get(env.data_manager)  # type: ignore
    chosen_object_holding_kp, _ = key_point_utils.select_valid(  # (n_envs, 7)
        object_holding_kp.view(len(env_ids), -1, 7), object_holding_mask.view(len(env_ids), -1), 1
    )
    chosen_object_holding_z_axis = math_utils.matrix_from_quat(chosen_object_holding_kp[..., 3:7])[..., 2]
    filtered_asset_align_mask = key_point_utils.z_axis_alignment_filter(
        asset_align_kp, asset_align_mask, chosen_object_holding_z_axis, -0.05
    )
    # safety guard for the case that all the key points are filtered out
    valid = filtered_asset_align_mask.any(dim=1, keepdim=True)  # → (E,1)
    filtered_asset_align_mask = torch.where(
        valid,
        filtered_asset_align_mask,
        asset_align_mask 
    )
    _, align_idx, asset_align_against_pose_w, _ = key_point_utils.minimum_error_pair_selection(
        asset_align_kp.view(len(env_ids), -1, 7),
        filtered_asset_align_mask.view(len(env_ids), -1).bool(),
        asset_align_against_kp.view(len(env_ids), -1, 7),
        asset_align_against_mask.view(len(env_ids), -1).bool(),
        torch.cat((asset_align_against_met.pos_mask[env_ids], asset_align_against_met.rot_mask[env_ids]), dim=1)
    )

    env_id_arange = torch.arange(len(env_ids), device=env.device)
    align_src = align_src.view(len(env_ids), -1, 7)[env_id_arange, align_idx].view(-1, 7)
    insertion_offset = align_offset.view(len(env_ids), -1, 7)[env_id_arange, align_idx].view(-1, 7)

    grasp_met: AlignmentMetric.AlignmentData = grasp_alignment_cfg.get(env.data_manager)  # type: ignore
    grasp_asset_id, grasp_offset_id = grasp_met.align_asset_index[env_ids], grasp_met.align_offset_index[env_ids]
    obj_held_pose_w = object_holding_kp[env_id_arange, grasp_asset_id, grasp_offset_id].view(-1, 7)
    obj_held_offset = object_holding_offset[env_id_arange, grasp_asset_id, grasp_offset_id].view(-1, 7)

    des_ee_pose_w = skills.align_held_asset_key_point(
        aligning_pose_w=asset_align_against_pose_w,
        asset_pose_w=align_src,
        asset_key_point_offset=insertion_offset,
        object_held_pose_w=obj_held_pose_w,
        object_held_offset=obj_held_offset,
    )

    if offset is not None:
        offset_pose = torch.tensor(offset.pose, device=env.device).repeat(env_ids.shape[0], 1)
        des_ee_pose_w[:, :3] += offset_pose[:, :3]

    return des_ee_pose_w


def screw_execution(
    env: DataManagerBasedRLEnv,
    mask: torch.Tensor,
    nist_board_cfg: SceneEntityCfg,
    task_alignment_cfg: Align,
    grasp_alignment_cfg: Align,
    align_src_cfg: Kp,
    align_kp_cfg: Kp,
    align_offset_cfg: Kp,
    align_against_kp_cfg: Kp,
    object_holding_src_cfg: Kp,
    object_holding_kp_cfg: Kp,
    object_holding_offset_cfg: Kp,
):
    assets: RigidObjectCollection = env.scene["assets"]
    supporting_asset_pose_w = assets.data.object_state_w[mask, nist_board_cfg.object_collection_ids, :7]
    support_asset_z_axis = math_utils.matrix_from_quat(supporting_asset_pose_w[:, 3:])[..., 2]

    insertion_vector = support_asset_z_axis * 0.005
    turning_vector = support_asset_z_axis * 3.1415 / 8
    turning_quat = math_utils.quat_from_euler_xyz(turning_vector[:, 0], turning_vector[:, 1], turning_vector[:, 2])

    align_src_pose_w, _ = align_src_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    align_insertion_kp, _ = align_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    align_offset, _ = align_offset_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_src, _ = object_holding_src_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_kp, _ = object_holding_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_offset, _ = object_holding_offset_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    align_against_kp, _ = align_against_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)

    assembly_metric: AlignmentMetric.AlignmentData = task_alignment_cfg.get(env.data_manager)  # type: ignore
    align_i, align_offset_i = assembly_metric.align_asset_index[mask], assembly_metric.align_offset_index[mask]
    against_i, against_offset_i = assembly_metric.align_against_asset_index[mask], assembly_metric.align_against_offset_index[mask]
    align_src_pose_w = align_src_pose_w[mask, 0, align_i, align_offset_i]
    align_kp_pose_w = align_insertion_kp[mask, 0, align_i, align_offset_i]
    insertion_offset = align_offset[mask, 0, align_i, align_offset_i]
    align_against_kp_pose_w = align_against_kp[mask, 0, against_i, against_offset_i]

    grasp_met: AlignmentMetric.AlignmentData = grasp_alignment_cfg.get(env.data_manager)  # type: ignore
    align_i, align_offset_i = grasp_met.align_asset_index[mask], grasp_met.align_offset_index[mask]
    obj_held_src_w = object_holding_src[mask, 0, align_i, align_offset_i]
    obj_held_pose_w = object_holding_kp[mask, 0, align_i, align_offset_i]
    obj_held_offset = object_holding_offset[mask, 0, align_i, align_offset_i]

    ee_pose_w_screwing_held_asset = skills.turning_held_asset(
        des_fixed_asset_held_assembled_pose_w=align_against_kp_pose_w,
        held_asset_pose_w=align_src_pose_w,
        held_asset_insertion_offset_w=insertion_offset,
        held_asset_insertion_pose_w=align_kp_pose_w,
        ee_pose_w=obj_held_src_w,
        ee_object_held_offset_w=obj_held_offset,
        ee_object_held_pose_w=obj_held_pose_w,
        insertion_vector=insertion_vector,
        turning_quat=turning_quat
    )
    return ee_pose_w_screwing_held_asset


def unwind_execution(
    env: DataManagerBasedRLEnv,
    mask: torch.Tensor,
    nist_board_cfg: SceneEntityCfg,
    grasp_alignment_cfg: Align,
    grasp_kp_cfg: Kp,
    object_holding_src_cfg: Kp,
    object_holding_kp_cfg: Kp,
):
    assets: RigidObjectCollection = env.scene["assets"]
    supporting_asset_pose_w = assets.data.object_state_w[mask, nist_board_cfg.object_collection_ids, :7]
    support_asset_z_axis = math_utils.matrix_from_quat(supporting_asset_pose_w[:, 3:])[..., 2]

    unwinding_vector = support_asset_z_axis * -3.1415 / 4
    unwinding_quat = math_utils.quat_from_euler_xyz(unwinding_vector[:, 0], unwinding_vector[:, 1], unwinding_vector[:, 2])

    object_holding_src, _ = object_holding_src_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_kp, _ = object_holding_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    grasp_kp, _ = grasp_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)

    grasp_met: AlignmentMetric.AlignmentData = grasp_alignment_cfg.get(env.data_manager) # type: ignore
    align_i, align_offset_i = grasp_met.align_asset_index[mask], grasp_met.align_offset_index[mask]
    against_i, against_offset_i = grasp_met.align_against_asset_index[mask], grasp_met.align_against_offset_index[mask]
    obj_held_src_w = object_holding_src[mask, 0, align_i, align_offset_i]
    obj_held_pose_w = object_holding_kp[mask, 0, align_i, align_offset_i]
    grasp_pose_w = grasp_kp[mask, 0, against_i, against_offset_i]

    ee_pose = skills.turning_with_alignment(
        ee_pose_w=obj_held_src_w,
        ee_object_held_pose_w=obj_held_pose_w,
        held_asset_grasp_pose_w=grasp_pose_w,
        turning_quat=unwinding_quat
    )
    return ee_pose


def align_gripper_to_held_asset_grasp_point(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor,
    supporting_asset_cfg: SceneEntityCfg,
    grasp_alignment_cfg: Align,
    grasp_kp_cfg: Kp,
    object_holding_offset_cfg: Kp,
    object_holding_kp_cfg: Kp,
    offset: Optional[Offset] = None,
):
    grasp_kp, _ = grasp_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_held_offset, _ = object_holding_offset_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    object_holding_kp, _ = object_holding_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    grasp_met: AlignmentMetric.AlignmentData = grasp_alignment_cfg.get(env.data_manager)  # type: ignore
    align_i, align_offset_i = grasp_met.align_asset_index[env_ids], grasp_met.align_offset_index[env_ids]
    against_i, against_offset_i = grasp_met.align_against_asset_index[env_ids], grasp_met.align_against_offset_index[env_ids]
    object_holding_kp = object_holding_kp[env_ids, 0, align_i, align_offset_i]
    object_held_offset = object_held_offset[env_ids, 0, align_i, align_offset_i]
    grasp_kp = grasp_kp[env_ids, 0, against_i, against_offset_i]

    support_asset: RigidObjectCollection = env.scene[supporting_asset_cfg.name]
    support_quat = support_asset.data.object_quat_w[env_ids, supporting_asset_cfg.object_collection_ids].view(-1, 4)
    support_height = support_asset.data.object_pos_w[env_ids, supporting_asset_cfg.object_collection_ids, 2].view(-1)
    z_axis = math_utils.matrix_from_quat(support_quat)[..., 2]

    ENV_IDS = torch.arange(env_ids.shape[0], device=env.device)
    interpolate_env_mask = torch.any(~grasp_met.rot_mask[env_ids], dim=1)
    interpolate_id, non_interpolate_id = ENV_IDS[interpolate_env_mask].view(-1), ENV_IDS[~interpolate_env_mask].view(-1)
    object_holding_kp[non_interpolate_id, 3:] = grasp_kp[non_interpolate_id, 3:]
    if torch.any(interpolate_env_mask):
        interpolated_grasp_quat = pose_utils.align_quat_axis(
            object_holding_kp[interpolate_id, 3:], grasp_kp[interpolate_id, 3:], axis=(~grasp_met.rot_mask[interpolate_id]).float()
        )
        object_holding_kp[interpolate_id, 3:] = interpolated_grasp_quat

    # step 2:
    # special treatment for falling objects because gripper now, gripper can not touch the ground,
    # compare the grasp kp z diff with supporting asset z diff
    # if greater than 60 degrees, then interpolate, otherwise, skip
    grasp_z = math_utils.matrix_from_quat(object_holding_kp[:, 3:])[..., 2]
    dot = (grasp_z * z_axis).sum(dim=-1)
    dot_limit = torch.cos(torch.tensor(1.57 * 2 / 3, device=env.device))
    object_holding_kp[:, 3:] = torch.where(
        (dot < dot_limit).unsqueeze(-1),
        pose_utils.interpolate_grasp_quat(object_holding_kp[:, 3:], z_axis, interpolation_factor=0.5),
        object_holding_kp[:, 3:]
    )

    grasp_kp_too_low = (grasp_kp[:, 2] - support_height) < 0.01
    grasp_kp[:, 2] = torch.where(grasp_kp_too_low, support_height + 0.01, grasp_kp[:, 2])
    ee_grasp_held_asset_pose_w = pose_utils.offset_subtract(
        align_against_pose_w=torch.cat((grasp_kp[:, :3], object_holding_kp[:, 3:]), dim=1),
        aligning_offset_pose=object_held_offset,
    )

    if offset is not None:
        offset_pose = torch.tensor(offset.pose, device=env.device).repeat(env_ids.shape[0], 1)
        ee_grasp_held_asset_pose_w[:, :3] += offset_pose[:, :3]

    return ee_grasp_held_asset_pose_w
