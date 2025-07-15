import torch
from torch import Tensor
from typing import Tuple
import isaaclab.utils.math as math_utils


@torch.jit.script
def subtract_frame_transforms(t01: Tensor, q01: Tensor, t02: Tensor, q02: Tensor) -> Tuple[Tensor, Tensor]:
    # compute orientation
    q10 = math_utils.quat_inv(q01)
    q12 = math_utils.quat_mul(q10, q02)
    # compute translation
    t12 = math_utils.quat_apply(q10, t02 - t01)
    return t12, q12


@torch.jit.script
def combine_frame_transforms(t01: Tensor, q01: Tensor, t12: Tensor, q12: Tensor) -> tuple[Tensor, Tensor]:
    # compute orientation
    q02 = math_utils.quat_mul(q01, q12)
    # compute translation
    t02 = t01 + math_utils.quat_apply(q01, t12)
    return t02, q02


@torch.jit.script
def delta_align_quat_axis(q1: Tensor, q2: Tensor, axis: Tensor) -> Tuple[Tensor, Tensor]:
    # world‐space axes
    v1 = math_utils.quat_apply(q1, axis)
    v2 = math_utils.quat_apply(q2, axis)

    # axis–angle correction
    cross = torch.cross(v1, v2, dim=-1)
    axis_norm = torch.sqrt((cross * cross).sum(-1, True)).clamp(min=1e-8)
    axis = cross / axis_norm
    dot = (v1 * v2).sum(dim=-1).clamp(-1, 1)
    angle = torch.acos(dot)
    return angle, axis


@torch.jit.script
def align_quat_axis(q1: Tensor, q2: Tensor, axis: Tensor) -> Tensor:
    angle, axis = delta_align_quat_axis(q1, q2, axis)
    q_corr = math_utils.quat_from_angle_axis(angle, axis)
    return math_utils.quat_mul(q_corr, q1)


@torch.jit.script
def compute_pose_error(t01: Tensor, q01: Tensor, t02: Tensor, q02: Tensor) -> Tuple[Tensor, Tensor]:
    # Compute quaternion error (i.e., difference quaternion)
    # Reference: https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
    # q_current_norm = q_current * q_current_conj
    source_quat_norm = math_utils.quat_mul(q01, math_utils.quat_conjugate(q01))[:, 0]
    # q_current_inv = q_current_conj / q_current_norm
    source_quat_inv = math_utils.quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
    # q_error = q_target * q_current_inv
    quat_error = math_utils.quat_mul(q02, source_quat_inv)
    # Compute position error
    pos_error = t02 - t01
    axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
    return pos_error, axis_angle_error


@torch.jit.script
def compute_poses_error(p1: Tensor, p2: Tensor) -> Tensor:
    pos_error, rot_error = compute_pose_error(p1[:, :3], p1[:, 3:], p2[:, :3], p2[:, 3:])
    return torch.cat((pos_error, rot_error), dim=1)


@torch.jit.script
def root_pose_given_target_n_offset(
    target_pos_w: Tensor, target_quat_w: Tensor, offset_pos: Tensor, offset_quat: Tensor
) -> Tuple[Tensor, Tensor]:
    # TA←W ​= {TB←A}-1 ​∘ TC←W​   where  ​combine_transform(a,b): b∘a
    inv_pos_ba = -math_utils.quat_apply(math_utils.quat_inv(offset_quat), offset_pos)
    inv_quat_ba = math_utils.quat_inv(offset_quat)
    return combine_frame_transforms(target_pos_w, target_quat_w, inv_pos_ba, inv_quat_ba)


@torch.jit.script
def poses_combine(root_pose: Tensor, offset_pose: Tensor) -> Tensor:
    pos_w, quat_w = combine_frame_transforms(
        root_pose[:, :3], root_pose[:, 3:], offset_pose[:, :3], offset_pose[:, 3:]
    )
    return torch.cat((pos_w, quat_w), dim=1)


@torch.jit.script
def poses_subtract(root_pose: Tensor, target_pose: Tensor) -> Tensor:
    pos_w, quat_w = subtract_frame_transforms(
        root_pose[:, :3], root_pose[:, 3:], target_pose[:, :3], target_pose[:, 3:]
    )
    return torch.cat((pos_w, quat_w), dim=1)


@torch.jit.script
def offset_subtract(align_against_pose_w: Tensor, aligning_offset_pose: Tensor) -> Tensor:
    aligning_asset_root_pos, aligning_asset_root_quat = root_pose_given_target_n_offset(
        align_against_pose_w[:, :3], align_against_pose_w[:, 3:], aligning_offset_pose[:, :3], aligning_offset_pose[:, 3:]
    )
    return torch.cat((aligning_asset_root_pos, aligning_asset_root_quat), dim=1)


@torch.jit.script
def correct_pose(cur_pose: Tensor, des_pose: Tensor, pos_mask: Tensor, rot_mask: Tensor) -> Tensor:
    delta_pos = des_pose[:, :3] - cur_pose[:, :3]
    p_new = cur_pose[:, :3] + pos_mask * delta_pos
    q_new = align_quat_axis(cur_pose[:, 3:], des_pose[:, 3:], rot_mask)
    return torch.cat((p_new, q_new), dim=1)


@torch.jit.script
def se3_step(ee_pose: Tensor, des_ee_pose: Tensor, max_pos: float, max_rot: float) -> Tensor:
    # Calculate the vector from current to desired position
    pos_error, rot_error = compute_pose_error(
        ee_pose[:, :3], ee_pose[:, 3:], des_ee_pose[:, :3], des_ee_pose[:, 3:]
    )
    # interpolate the position
    distance = torch.norm(pos_error, dim=1).clamp(max=max_pos)
    new_pos_error = math_utils.normalize(pos_error) * distance.view(-1, 1)
    new_pos = ee_pose[:, :3] + new_pos_error

    # interpolate the orientation
    angle = torch.norm(rot_error, dim=1).clamp(max=max_rot)
    axis = math_utils.normalize(rot_error) * angle.view(-1, 1)
    new_quat_error = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
        math_utils.quat_from_angle_axis(angle, axis),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=ee_pose.device),
    )
    new_quat = math_utils.quat_mul(new_quat_error, ee_pose[:, 3:])
    new_pose = torch.cat((new_pos, new_quat), dim=1)

    return new_pose


@torch.jit.script
def interpolate_grasp_quat(
    held_asset_grasp_point_quat_w: torch.Tensor,
    secondary_z_axis: torch.Tensor,
    interpolation_factor: float,
) -> torch.Tensor:
    orig_mat: torch.Tensor = math_utils.matrix_from_quat(held_asset_grasp_point_quat_w)
    orig_y_axis = orig_mat[..., 1]
    # interpolate z
    leg_grasp_z_axis = orig_mat[..., 2]
    leg_grasp_point_z_axis = math_utils.normalize(interpolation_factor * leg_grasp_z_axis + (1 - interpolation_factor) * secondary_z_axis)
    # determine the closest y axis
    y1: torch.Tensor = math_utils.normalize(torch.linalg.cross(leg_grasp_point_z_axis, secondary_z_axis))
    y2: torch.Tensor = math_utils.normalize(torch.linalg.cross(secondary_z_axis, leg_grasp_point_z_axis))

    dot1 = (y1 * orig_y_axis).sum(dim=-1)   # (…)
    dot2 = (y2 * orig_y_axis).sum(dim=-1)
    leg_grasp_point_y_axis = torch.where((dot1 >= dot2).unsqueeze(-1), y1, y2)

    leg_grasp_point_x_axis = math_utils.normalize(torch.linalg.cross(leg_grasp_point_y_axis, leg_grasp_point_z_axis))
    # compose to grasp quat
    des_leg_grasp_quat_w = math_utils.quat_from_matrix(
        torch.stack((leg_grasp_point_x_axis, leg_grasp_point_y_axis, leg_grasp_point_z_axis), dim=1).transpose(1, 2)
    )
    return des_leg_grasp_quat_w
