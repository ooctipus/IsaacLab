import torch
from torch import Tensor

import isaaclab.utils.math as math_utils
from . import utils as poses_utils


@torch.jit.script
def align_held_asset(
    desired_asset_pose_w: Tensor, asset_pose_w: Tensor, object_held_pose_w: Tensor, object_held_offset: Tensor,
) -> Tensor:
    curr_offset = poses_utils.poses_subtract(asset_pose_w, object_held_pose_w)
    grasp_target = poses_utils.poses_combine(desired_asset_pose_w, curr_offset)
    return poses_utils.offset_subtract(grasp_target, object_held_offset)


@torch.jit.script
def align_held_asset_key_point(
    aligning_pose_w: Tensor, asset_pose_w: Tensor, asset_key_point_offset: Tensor, object_held_pose_w: Tensor, object_held_offset: Tensor,
) -> Tensor:
    desired_asset_pose_w = poses_utils.offset_subtract(aligning_pose_w, asset_key_point_offset)
    curr_offset = poses_utils.poses_subtract(asset_pose_w, object_held_pose_w)
    grasp_target = poses_utils.poses_combine(desired_asset_pose_w, curr_offset)
    return poses_utils.offset_subtract(grasp_target, object_held_offset)


@torch.jit.script
def turning_held_asset(
    des_fixed_asset_held_assembled_pose_w: torch.Tensor,
    held_asset_pose_w: torch.Tensor,
    held_asset_insertion_offset_w: torch.Tensor,
    held_asset_insertion_pose_w: torch.Tensor,
    ee_pose_w: torch.Tensor,
    ee_object_held_offset_w: torch.Tensor,
    ee_object_held_pose_w: torch.Tensor,
    insertion_vector: torch.Tensor,
    turning_quat: torch.Tensor,
) -> torch.Tensor:

    pos_mask = torch.tensor([1.0, 1.0, 0.0], device=ee_pose_w.device).repeat(ee_pose_w.shape[0], 1)
    rot_mask = torch.tensor([0.0, 0.0, 1.0], device=ee_pose_w.device).repeat(ee_pose_w.shape[0], 1)
    p = poses_utils.correct_pose(held_asset_insertion_pose_w, des_fixed_asset_held_assembled_pose_w, pos_mask, rot_mask)

    # screwing - find the screwed leg pose
    p[:, :3] = p[:, :3] + insertion_vector
    
    # turning_quat needs broadcasting
    turning_quat = turning_quat.expand(p.shape[0], 4)
    p[:, 3:] = math_utils.quat_mul(p[:, 3:], turning_quat)

    return align_held_asset_key_point(p, held_asset_pose_w, held_asset_insertion_offset_w, ee_object_held_pose_w, ee_object_held_offset_w)


@torch.jit.script
def turning_with_alignment(
    ee_pose_w: torch.Tensor,
    ee_object_held_pose_w: torch.Tensor,
    held_asset_grasp_pose_w: torch.Tensor,
    turning_quat: torch.Tensor,
):
    pos_mask = torch.tensor([1.0, 1.0, 0.0], device=ee_pose_w.device).repeat(ee_pose_w.shape[0], 1)
    rot_mask = torch.tensor([0.0, 0.0, 1.0], device=ee_pose_w.device).repeat(ee_pose_w.shape[0], 1)
    corrected_pose = poses_utils.correct_pose(ee_object_held_pose_w, held_asset_grasp_pose_w, pos_mask, rot_mask)
    corrected_pose[:, 3:] = math_utils.quat_mul(turning_quat, corrected_pose[:, 3:])
    ee_object_held_offset_pose = poses_utils.poses_subtract(ee_pose_w, ee_object_held_pose_w)
    ee_pose_w_when_turning = poses_utils.offset_subtract(corrected_pose, ee_object_held_offset_pose)
    return ee_pose_w_when_turning
