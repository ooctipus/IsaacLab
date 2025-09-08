import torch
from torch import Tensor
from typing import Optional

import isaaclab.utils.math as math_utils
from ....state_machine import utils as poses_utils

# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


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


def interpolate_grasp_quat(
    held_asset_grasp_point_quat_w: torch.Tensor,
    grasped_object_quat_in_ee_frame: torch.Tensor,
    secondary_z_axis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if secondary_z_axis is not None:
        table_z_axis = secondary_z_axis
        leg_grasp_z_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 2]
        leg_grasp_point_z_axis = math_utils.normalize(0.7 * leg_grasp_z_axis + 0.3 * table_z_axis)

        # determine the closest y axis
        leg_grasp_point_y_axis_case1 = leg_grasp_point_z_axis.cross(table_z_axis)
        leg_grasp_point_y_axis_case2 = -leg_grasp_point_z_axis.cross(table_z_axis)
        robot_grasp_y = math_utils.matrix_from_quat(grasped_object_quat_in_ee_frame)[..., 1]
        dot_dist_1 = (robot_grasp_y * leg_grasp_point_y_axis_case1).sum(dim=1)
        dot_dist_2 = (robot_grasp_y * leg_grasp_point_y_axis_case2).sum(dim=1)
        leg_grasp_point_y_axis = torch.where(
            (dot_dist_1 > dot_dist_2).view(-1, 1), leg_grasp_point_y_axis_case1, leg_grasp_point_y_axis_case2
        )
        leg_grasp_point_x_axis = leg_grasp_point_y_axis.cross(leg_grasp_point_z_axis)
    else:
        leg_grasp_z_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 2]
        leg_grasp_point_z_axis = math_utils.normalize(leg_grasp_z_axis)
        robot_grasp_y = math_utils.matrix_from_quat(grasped_object_quat_in_ee_frame)[..., 1]

        leg_grasp_x_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 0]
        leg_grasp_y_axis = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)[..., 1]
        leg_grasp_neg_x_axis = -leg_grasp_x_axis.clone()
        leg_grasp_neg_y_axis = -leg_grasp_y_axis.clone()

        # Stack all candidate axes into a tensor of shape (num_envs, 4, 3)
        candidate_axes = torch.stack(
            [leg_grasp_x_axis, leg_grasp_neg_x_axis, leg_grasp_y_axis, leg_grasp_neg_y_axis], dim=1
        )  # shape: (N, 4, 3)

        # Compute dot products between each candidate axis and robot_grasp_y.
        # robot_grasp_y is (N, 3) and unsqueezed to (N, 1, 3) so that broadcasting gives (N, 4, 3).
        dot_products = (candidate_axes * robot_grasp_y.unsqueeze(1)).sum(dim=2)  # shape: (N, 4)
        # Get the index of the candidate with the maximum dot product for each environment.
        max_indices = dot_products.argmax(dim=1)  # shape: (N,)

        # Index the best candidate out.
        leg_grasp_point_y_axis = candidate_axes[torch.arange(candidate_axes.shape[0]), max_indices]  # shape: (N, 3)
        leg_grasp_point_x_axis = leg_grasp_point_y_axis.cross(leg_grasp_z_axis)

    # compose to grasp quat
    des_leg_grasp_quat_w = math_utils.quat_from_matrix(
        torch.stack((leg_grasp_point_x_axis, leg_grasp_point_y_axis, leg_grasp_point_z_axis), dim=1).transpose(1, 2)
    )
    return des_leg_grasp_quat_w


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
