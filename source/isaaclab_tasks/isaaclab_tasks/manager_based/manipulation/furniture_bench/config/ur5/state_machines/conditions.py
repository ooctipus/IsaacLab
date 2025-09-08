import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from ....state_machine import StateMachine as FSM

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def always(env: ManagerBasedRLEnv, sm: FSM):
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def gripper_open(env: ManagerBasedRLEnv, sm: FSM, robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].view(-1)
    return less(gripper_joint_pos, 0.05)


def gripper_grasp_object(env: ManagerBasedRLEnv, sm: FSM, robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].view(-1)
    return greater(gripper_joint_pos, 0.49)


def wrist_counter_clockwise_limit_reached(env: ManagerBasedRLEnv, sm: FSM, robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    wrist_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].view(-1)
    wrist_joint_limits = robot.data.joint_pos_limits[:, robot_cfg.joint_ids].view(env.num_envs, -1)
    upper_soft_limit = wrist_joint_limits[:, 0] + (wrist_joint_limits[:, 1] - wrist_joint_limits[:, 0]) * 0.9
    return wrist_joint_pos > upper_soft_limit


def wrist_clockwise_limit_reached(env: ManagerBasedRLEnv, sm: FSM, robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    wrist_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].view(-1)
    wrist_joint_limits = robot.data.joint_pos_limits[:, robot_cfg.joint_ids].view(env.num_envs, -1)
    lower_soft_limit = wrist_joint_limits[:, 0] + (wrist_joint_limits[:, 1] - wrist_joint_limits[:, 0]) * 0.1
    return wrist_joint_pos < lower_soft_limit


def held_asset_lifted(env, sm: FSM):
    return greater(sm.held_asset_insertion_pose_w[:, 2], 0.05)


def gripper_aligned_with_held_asset(env, sm: FSM, interpolate: bool = False, only_pos: bool = False):
    pos_criterion = pos_error(sm.ee_object_held_pose_w[:, :3], sm.held_asset_grasp_pose_w[:, :3], threshold=0.05)
    if only_pos:
        return pos_criterion
    if interpolate:
        rot_criterion = quat_error(sm.ee_object_held_pose_w[:, 3:], sm.desired_held_asset_grasp_quat_w, threshold=0.05)
    else:
        rot_criterion = quat_error(sm.ee_object_held_pose_w[:, 3:], sm.held_asset_grasp_pose_w[:, 3:], threshold=0.05)
    return pos_criterion & rot_criterion


def held_asset_insertion_aligned_with_fixed_asset_entry(env, sm: FSM, check_fully_inserted=False, pos_threshold=0.01, rot_threshold=0.1):
    if check_fully_inserted:
        entry_pos = sm.held_asset_inserted_in_fixed_asset_pos[:, :3]
    else:
        entry_pos = sm.fixed_asset_entry_pose_w[:, :3]
    pos_criterion = pos_error(entry_pos, sm.held_asset_insertion_pose_w[:, :3], threshold=pos_threshold)
    rot_criterion = quat_error(sm.fixed_asset_entry_pose_w[:, 3:], sm.held_asset_insertion_pose_w[:, 3:], threshold=rot_threshold, components="xy")
    return pos_criterion & rot_criterion


def held_asset_insertion_aligned_with_way_above_fixed_asset_entry(env, sm: FSM, check_fully_inserted=False):
    if check_fully_inserted:
        entry_pos = sm.held_asset_inserted_in_fixed_asset_pos[:, :3]
    else:
        entry_pos, _ = math_utils.combine_frame_transforms(
            sm.fixed_asset_entry_pose_w[:, :3], sm.fixed_asset_entry_pose_w[:, 3:], sm.way_above_hole_offset_xyz
        )
    pos_criterion = pos_error(entry_pos, sm.held_asset_insertion_pose_w[:, :3], threshold=0.01)
    rot_criterion = quat_error(sm.fixed_asset_entry_pose_w[:, 3:], sm.held_asset_insertion_pose_w[:, 3:], threshold=0.1, components="xy")
    return pos_criterion & rot_criterion


def held_asset_fully_assembled_on_fixed_asset(env, sm: FSM):
    pos_criterion = pos_error(sm.held_asset_insertion_pose_w[:, :3], sm.fixed_asset_held_assembled_pose_w[:, :3], threshold=0.0015)
    rot_criterion = quat_error(sm.fixed_asset_entry_pose_w[:, 3:], sm.held_asset_insertion_pose_w[:, 3:], threshold=0.1, components="xy")
    return pos_criterion & rot_criterion


def pos_error(pos1: torch.Tensor, pos2: torch.Tensor, threshold: float, p=2) -> torch.Tensor:
    return torch.norm(pos2 - pos1, p=p, dim=1) < threshold


def quat_error(quat1: torch.Tensor, quat2: torch.Tensor, threshold: float, components: str = "xyz", p=2) -> torch.Tensor:
    source_quat_norm = math_utils.quat_mul(quat1, math_utils.quat_conjugate(quat1))[:, 0]
    source_quat_inv = math_utils.quat_conjugate(quat1) / source_quat_norm.unsqueeze(-1)
    quat_error = math_utils.quat_mul(quat2, source_quat_inv)
    axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
    if "x" not in components:
        axis_angle_error[:, 0] = 0.0
    if "y" not in components:
        axis_angle_error[:, 1] = 0.0
    if "z" not in components:
        axis_angle_error[:, 2] = 0.0

    return torch.norm(axis_angle_error, p=p, dim=1) < threshold


def quat_axis_angle_error(quat1: torch.Tensor, quat2: torch.Tensor, component: str = "z") -> torch.Tensor:
    source_quat_norm = math_utils.quat_mul(quat1, math_utils.quat_conjugate(quat1))[:, 0]
    source_quat_inv = math_utils.quat_conjugate(quat1) / source_quat_norm.unsqueeze(-1)
    quat_error = math_utils.quat_mul(quat2, source_quat_inv)
    axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
    if component == "x":
        return axis_angle_error[:, 0]
    elif component == "y":
        return axis_angle_error[:, 1]
    elif component == "z":
        return axis_angle_error[:, 2]
    else:
        raise ValueError(f"Invalid component: {component}")


def greater(val1: torch.Tensor, val2: float):
    return val1 > val2


def less(val1: torch.Tensor, val2: float):
    return val1 < val2
