import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from ....state_machine import ConditionCfg
from isaaclab.envs import ManagerBasedRLEnv
from ....mdp.data_cfg import AlignmentDataCfg as Align
from ....mdp.data_cfg import KeyPointDataCfg as Kp
from ....mdp.data_cfg import DataCfg as Data
from ....mdp.data_cfg import AlignmentMetric
from ....mdp.command import AssemblyTaskCommand

# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform2"))


if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection


def always(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    return torch.ones(len(env_ids), device=env.device, dtype=torch.bool)


def gripper_open(env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, threshold: float):
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_pos = robot.data.joint_pos[env_ids, robot_cfg.joint_ids].view(-1)
    return greater(gripper_joint_pos, threshold)


def gripper_grasp_object(env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, diameter_data: Data):
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_pos = robot.data.joint_pos[env_ids, robot_cfg.joint_ids].view(-1)
    diameter = diameter_data.get(env.data_manager)
    return (gripper_joint_pos < (diameter[env_ids] * 1.1 / 2))


class Grasped:
    def __init__(self, condition_cfg: ConditionCfg, env: ManagerBasedRLEnv):
        self.robot_cfg: SceneEntityCfg = condition_cfg.args["robot_cfg"]
        self.history_length: int = condition_cfg.args["history_length"]
        self.robot: Articulation = env.scene[self.robot_cfg.name]
        self.gripper_pos_history = torch.zeros((env.num_envs, self.history_length), device=env.device)
        self.history_pointer = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.history_size = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.std_threshold = 1e-4

    def reset(self, env_ids):
        self.gripper_pos_history[env_ids] = 0.0
        self.history_pointer[env_ids] = 0
        self.history_size[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, history_length=5):
        gripper_joint_pos = self.robot.data.joint_pos[env_ids, self.robot_cfg.joint_ids].view(-1)
        self.gripper_pos_history[env_ids, self.history_pointer[env_ids]] = gripper_joint_pos

        self.history_pointer[env_ids] = (self.history_pointer[env_ids] + 1) % self.history_length
        self.history_size[env_ids] = torch.clamp(self.history_size[env_ids] + 1, max=self.history_length)

        sum_hist = self.gripper_pos_history[env_ids].sum(dim=1)
        mean_joint_pos = sum_hist / self.history_size[env_ids]

        var = ((self.gripper_pos_history[env_ids] - mean_joint_pos.unsqueeze(1))**2).sum(dim=1) / self.history_size[env_ids].clamp(min=1)
        std = torch.sqrt(var)

        has_enough_data = self.history_size[env_ids] > (self.history_length // 2)
        is_stable = std < self.std_threshold

        return is_stable & has_enough_data


def task_success(env: ManagerBasedRLEnv, env_ids: torch.Tensor, task_alignment_cfg: Align):
    task_alignment: AlignmentMetric.AlignmentData = task_alignment_cfg.get(env.data_manager)  # type: ignore
    return task_alignment.pos_aligned[env_ids] & task_alignment.rot_aligned[env_ids]


def task_not_success(env: ManagerBasedRLEnv, env_ids: torch.Tensor, task_alignment_cfg: Align):
    task_alignment: AlignmentMetric.AlignmentData = task_alignment_cfg.get(env.data_manager)  # type: ignore
    return ~task_alignment.pos_aligned[env_ids] | ~task_alignment.rot_aligned[env_ids]


def task_type(env: ManagerBasedRLEnv, env_ids: torch.Tensor, task_type: list[int]):
    command: AssemblyTaskCommand = env.command_manager.get_term("task_command")
    categories = command.cur_task_categories[env_ids]
    types = torch.tensor(task_type, device=env.device)
    return torch.isin(categories, types)


def not_during_screw_task(env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, task_alignment_cfg: Align):
    not_skip_screw = ~(task_type(env, env_ids, [2]) 
             & ~wrist_counter_clockwise_limit_reached(env, env_ids, robot_cfg) 
             & task_not_success(env, env_ids, task_alignment_cfg))
    return torch.where(
        task_type(env, env_ids, [1, 0]), task_success(env, env_ids, task_alignment_cfg), not_skip_screw
    )
        
class SpeedLow:
    def __init__(self, condition_cfg: ConditionCfg, env: ManagerBasedRLEnv):
        self.robot_cfg: SceneEntityCfg = condition_cfg.args["robot_cfg"]
        self.history_length: int = condition_cfg.args["history_length"]
        self.robot: Articulation = env.scene[self.robot_cfg.name]
        self.speed_history = torch.zeros((env.num_envs, self.history_length), device=env.device)
        self.history_pointer = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.history_size = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.speed_limit = condition_cfg.args["speed_limit"]

    def reset(self, env_ids):
        self.speed_history[env_ids] = 0.0
        self.history_pointer[env_ids] = 0
        self.history_size[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg, speed_limit, history_length=5):
        ee_vel = self.robot.data.body_link_lin_vel_w[env_ids, self.robot_cfg.body_ids].view(len(env_ids), -1)
        speeds = torch.norm(ee_vel, p=2, dim=1)
        self.speed_history[env_ids, self.history_pointer[env_ids]] = speeds

        self.history_pointer[env_ids] = (self.history_pointer[env_ids] + 1) % self.history_length
        self.history_size[env_ids] = torch.clamp(self.history_size[env_ids] + 1, max=self.history_length)

        sum_hist = self.speed_history[env_ids].sum(dim=1)
        mean_speed = sum_hist / self.history_size[env_ids]
        return (mean_speed < self.speed_limit)


class GraspAssetSpeedLow:
    def __init__(self, condition_cfg: ConditionCfg, env: ManagerBasedRLEnv):
        self.grasp_kp_asset_id_cfg: Kp = condition_cfg.args["grasp_kp_asset_id_cfg"]
        self.history_length: int = condition_cfg.args["history_length"]
        self.asset: RigidObjectCollection = env.scene[condition_cfg.args["asset_cfg"].name]
        self.speed_history = torch.zeros((env.num_envs, self.history_length), device=env.device)
        self.history_pointer = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.history_size = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.speed_limit = condition_cfg.args["speed_limit"]

    def reset(self, env_ids):
        self.speed_history[env_ids] = 0.0
        self.history_pointer[env_ids] = 0
        self.history_size[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        grasp_kp_asset_id_cfg: Kp,
        speed_limit,
        entry_alignment_cfg: Align,
        task_alignment_cfg: Align,
        history_length=5
    ):
        entry_met: AlignmentMetric.AlignmentData = entry_alignment_cfg.get(env.data_manager)
        inserted = entry_met.pos_delta[env_ids, 2] < -0.0005
        inserted_env_ids = env_ids[inserted]
        if len(inserted_env_ids) > 0:
            asset_id, _ = self.grasp_kp_asset_id_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
            asset_id = asset_id[inserted_env_ids].view(len(inserted_env_ids), -1)
            asset_vel = self.asset.data.object_lin_vel_w[inserted_env_ids[:, None], asset_id].view(len(inserted_env_ids), 3)
            speeds = torch.norm(asset_vel, p=2, dim=1)
            self.speed_history[inserted_env_ids, self.history_pointer[inserted_env_ids]] = speeds
            self.history_pointer[inserted_env_ids] = (self.history_pointer[inserted_env_ids] + 1) % self.history_length
            self.history_size[inserted_env_ids] = torch.clamp(self.history_size[inserted_env_ids] + 1, max=self.history_length)

        sum_hist = self.speed_history[env_ids].sum(dim=1)
        mean_speed = sum_hist / (self.history_size[env_ids].clamp(min=1))
        # print(f"mean_speed: {mean_speed}, history_size: {self.history_size[env_ids]}, history: {self.speed_history[env_ids]}")
        return ((mean_speed < self.speed_limit) & (self.history_size[env_ids] > (self.history_length // 2))) | task_success(env, env_ids, task_alignment_cfg)


def entry_met(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    entry_alignment_cfg: Align
):
    entry_met: AlignmentMetric.AlignmentData = entry_alignment_cfg.get(env.data_manager)
    inserted = entry_met.pos_delta[env_ids, 2] < -0.005
    return inserted


def wrist_counter_clockwise_limit_reached(env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    wrist_joint_pos = robot.data.joint_pos[env_ids, robot_cfg.joint_ids].view(-1)
    wrist_joint_limits = robot.data.joint_pos_limits[env_ids, robot_cfg.joint_ids].view(len(env_ids), -1)
    upper_soft_limit = wrist_joint_limits[:, 0] + (wrist_joint_limits[:, 1] - wrist_joint_limits[:, 0]) * 0.9
    return wrist_joint_pos > upper_soft_limit


def wrist_clockwise_limit_reached(env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    wrist_joint_pos = robot.data.joint_pos[env_ids, robot_cfg.joint_ids].view(-1)
    wrist_joint_limits = robot.data.joint_pos_limits[env_ids, robot_cfg.joint_ids].view(len(env_ids), -1)
    lower_soft_limit = wrist_joint_limits[:, 0] + (wrist_joint_limits[:, 1] - wrist_joint_limits[:, 0]) * 0.1
    return wrist_joint_pos < lower_soft_limit


def held_asset_lifted(env, env_ids: torch.Tensor, aligning_key_point_cfg: Kp, threshold: float = 0.05):
    align_kp, align_kp_mask = aligning_key_point_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    align_kp = align_kp[env_ids].view(len(env_ids), -1, 7)  # (n_envs, n_assets * n_offsets, 7)
    align_kp_mask = align_kp_mask[env_ids].view(len(env_ids), -1)  # (n_envs, n_assets * n_offsets, 7)

    align_kp[~align_kp_mask][:, 2] = threshold
    align_kp_all_above = torch.all(align_kp[..., 2] >= threshold, dim=1)  # (n_envs, )

    return align_kp_all_above


def gripper_aligned_with_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    manipulation_alignment_cfg: Align,
    only_pos: bool = False,
    pos_threshold: tuple[float, float, float] | None = None,
    rot_threshold: tuple[float, float, float] | None = None,
):
    manipulation_alignment: AlignmentMetric.AlignmentData = manipulation_alignment_cfg.get(env.data_manager)  # type: ignore
    if pos_threshold is not None:
        pos_threshold = torch.tensor(pos_threshold, device=env.device)
        pos_aligned = torch.all(manipulation_alignment.pos_error[env_ids] < pos_threshold, dim=1)
    else:
        pos_aligned = manipulation_alignment.pos_aligned[env_ids]
    if only_pos:
        return pos_aligned

    return pos_aligned & manipulation_alignment.rot_aligned[env_ids]


def held_asset_insertion_aligned_with_fixed_asset_entry(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    auxiliary_alignment_cfg: Align,
    pos_threshold: tuple[float, float, float] | None = None,
):
    auxiliary_alignment: AlignmentMetric.AlignmentData = auxiliary_alignment_cfg.get(env.data_manager)  # type: ignore
    if pos_threshold is not None:
        pos_threshold = torch.tensor(pos_threshold, device=env.device)
        pos_aligned = torch.all(auxiliary_alignment.pos_error[env_ids] < pos_threshold, dim=1)
    else:
        pos_aligned = auxiliary_alignment.pos_aligned[env_ids]

    return pos_aligned & auxiliary_alignment.rot_aligned[env_ids]


def held_asset_insertion_position_aligned_with_fixed_asset_entry(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    auxiliary_alignment_cfg: Align,
    pos_threshold: tuple[float, float, float] | None = None,
):
    auxiliary_alignment: AlignmentMetric.AlignmentData = auxiliary_alignment_cfg.get(env.data_manager)  # type: ignore
    if pos_threshold is not None:
        pos_threshold = torch.tensor(pos_threshold, device=env.device)
        pos_aligned = torch.all(auxiliary_alignment.pos_error[env_ids] < pos_threshold, dim=1)
    else:
        pos_aligned = auxiliary_alignment.pos_aligned[env_ids]

    return pos_aligned


def held_asset_insertion_not_position_aligned_with_fixed_asset_entry(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    auxiliary_alignment_cfg: Align,
    pos_threshold: tuple[float, float, float] | None = None,
):
    return ~held_asset_insertion_position_aligned_with_fixed_asset_entry(env, env_ids, auxiliary_alignment_cfg, pos_threshold)


def held_asset_fully_assembled_on_fixed_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    task_alignment_cfg: Align,
):
    task_alignment: AlignmentMetric.AlignmentData = task_alignment_cfg.get(env.data_manager)  # type: ignore
    return task_alignment.pos_aligned[env_ids] & task_alignment.rot_aligned[env_ids]


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
