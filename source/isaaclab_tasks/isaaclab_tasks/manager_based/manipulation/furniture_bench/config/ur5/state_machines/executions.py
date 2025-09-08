import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

from ....state_machine import utils as pose_utils
from ....state_machine import StateMachine as FSM
from . import skills


def gripper_action(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor, gripper_command: float):
    return gripper_command


def lift_up_execution(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    des_ee_pose = align_gripper_to_held_asset_grasp_point(env, sm, mask, interpolate=True)
    des_ee_pose[:, 2] += 0.1
    return des_ee_pose


def insertion_execution(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    des_ee_pose = aligning_held_asset_insertion_to_fixed_asset_entry(env, sm, mask)
    hole0_midpoint_distance = 0.8 * (sm.fixed_asset_entry_pose_w[mask, :3] - sm.fixed_asset_held_assembled_pose_w[mask, :3])
    return torch.cat((des_ee_pose[:, :3] - hole0_midpoint_distance, des_ee_pose[:, 3:]), dim=1)


def aligning_held_asset_insertion_way_above_fixed_asset_entry(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    hole0_tip_pos_w_to_align, _ = math_utils.combine_frame_transforms(
        sm.fixed_asset_entry_pose_w[mask, :3], sm.fixed_asset_entry_pose_w[mask, 3:], sm.way_above_hole_offset_xyz[mask]
    )
    aligning_pose_w = torch.cat((hole0_tip_pos_w_to_align, sm.fixed_asset_entry_pose_w[mask, 3:]), dim=1)
    des_ee_pose_w = skills.align_held_asset_key_point(
        aligning_pose_w=aligning_pose_w,
        asset_pose_w=sm.held_asset_pose_w[mask],
        asset_key_point_offset=sm.held_asset_insertion_offset_pose[mask],
        object_held_pose_w=sm.ee_object_held_pose_w[mask],
        object_held_offset=sm.ee_object_held_offset_pose[mask],
    )
    return des_ee_pose_w


def aligning_held_asset_insertion_to_fixed_asset_entry(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    hole0_tip_pos_w_to_align, _ = math_utils.combine_frame_transforms(
        sm.fixed_asset_entry_pose_w[mask, :3], sm.fixed_asset_entry_pose_w[mask, 3:], sm.approach_above_hole_offset_xyz[mask]
    )
    aligning_pose_w = torch.cat((hole0_tip_pos_w_to_align, sm.fixed_asset_entry_pose_w[mask, 3:]), dim=1)
    des_ee_pose_w = skills.align_held_asset_key_point(
        aligning_pose_w=aligning_pose_w,
        asset_pose_w=sm.held_asset_pose_w[mask],
        asset_key_point_offset=sm.held_asset_insertion_offset_pose[mask],
        object_held_pose_w=sm.ee_object_held_pose_w[mask],
        object_held_offset=sm.ee_object_held_offset_pose[mask],
    )
    return des_ee_pose_w


def screw_execution(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    insertion_vector = sm.support_asset_z_axis[mask] * sm.screw_pos_delta
    turning_vector = sm.support_asset_z_axis[mask] * sm.screw_turn_delta
    turning_quat = math_utils.quat_from_euler_xyz(turning_vector[:, 0], turning_vector[:, 1], turning_vector[:, 2])
    ee_pose_w_screwing_held_asset = skills.turning_held_asset(
        des_fixed_asset_held_assembled_pose_w=sm.init_hole_pose[mask],
        held_asset_pose_w=sm.held_asset_pose_w[mask],
        held_asset_insertion_offset_w=sm.held_asset_insertion_offset_pose[mask],
        held_asset_insertion_pose_w=sm.held_asset_insertion_pose_w[mask],
        ee_pose_w=sm.ee_pose_w[mask],
        ee_object_held_offset_w=sm.ee_object_held_offset_pose[mask],
        ee_object_held_pose_w=sm.ee_object_held_pose_w[mask],
        insertion_vector=insertion_vector,
        turning_quat=turning_quat
    )
    return ee_pose_w_screwing_held_asset


def unwind_execution(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    unwinding_vector = sm.support_asset_z_axis[mask] * sm.unwind_delta
    unwinding_quat = math_utils.quat_from_euler_xyz(unwinding_vector[:, 0], unwinding_vector[:, 1], unwinding_vector[:, 2])
    ee_pose = skills.turning_with_alignment(
        ee_pose_w=sm.ee_pose_w[mask], ee_object_held_pose_w=sm.ee_object_held_pose_w[mask],
        held_asset_grasp_pose_w=sm.held_asset_grasp_pose_w[mask], turning_quat=unwinding_quat
    )
    return ee_pose


def align_gripper_to_held_asset_grasp_point(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor, interpolate=False):
    if interpolate:
        desired_held_asset_grasp_pose_w = torch.cat((sm.held_asset_grasp_pose_w[mask, :3], sm.desired_held_asset_grasp_quat_w[mask]), dim=1)
    else:
        desired_held_asset_grasp_pose_w = sm.held_asset_grasp_pose_w[mask]
    ee_grasp_held_asset_pose_w = pose_utils.offset_subtract(
        align_against_pose_w=desired_held_asset_grasp_pose_w,
        aligning_offset_pose=sm.ee_object_held_offset_pose[mask],
    )
    return ee_grasp_held_asset_pose_w


def keep_ee_pose(env: ManagerBasedRLEnv, sm: FSM, mask: torch.Tensor):
    """Keep the current end-effector pose unchanged."""
    return sm.ee_pose_w[mask]
