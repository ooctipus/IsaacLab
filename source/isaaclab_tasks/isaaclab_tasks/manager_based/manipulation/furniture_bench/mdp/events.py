# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import threading 
import os
import random
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, Sdf

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Sequence, Any, Literal

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.controllers import MultiConstraintDifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    MultiConstraintsDifferentialInverseKinematicsActionCfg,
)
from ..state_machine import utils as sm_math_utils

from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ...ur5.assembly_data import Offset

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))
def reset_held_asset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    holding_body_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    robot_object_held_offset: Offset,
    held_asset_graspable_offset: Offset,
    held_asset_inhand_range: dict[str, tuple[float, float]],
):
    robot: Articulation = env.scene[holding_body_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    end_effector_quat_w = robot.data.body_link_quat_w[env_ids, holding_body_cfg.body_ids].view(-1, 4)
    end_effector_pos_w = robot.data.body_link_pos_w[env_ids, holding_body_cfg.body_ids].view(-1, 3)
    obj_target_pos_w, obj_target_quat_w = robot_object_held_offset.combine(end_effector_pos_w, end_effector_quat_w)
    held_graspable_pos_b = torch.tensor(held_asset_graspable_offset.pos, device=env.device).repeat(len(env_ids), 1)
    held_graspable_quat_b = torch.tensor(held_asset_graspable_offset.quat, device=env.device).repeat(len(env_ids), 1)

    translated_held_asset_pos, translated_held_asset_quat = _pose_a_when_frame_ba_aligns_pose_c(
        pos_c=obj_target_pos_w, quat_c=obj_target_quat_w, pos_ba=held_graspable_pos_b, quat_ba=held_graspable_quat_b
    )

    # Add randomization
    range_list = [held_asset_inhand_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    new_pos_w = translated_held_asset_pos + samples[:, 0:3]
    quat_b = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    new_quat_w = math_utils.quat_mul(translated_held_asset_quat, quat_b)

    held_asset.write_root_link_pose_to_sim(torch.cat([new_pos_w, new_quat_w], dim=1), env_ids=env_ids)  # type: ignore
    held_asset.write_root_com_velocity_to_sim(held_asset.data.default_root_state[env_ids, 7:], env_ids=env_ids)  # type: ignore


class robotiq_gripper_grasp_held_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        robot_ik_solver_cfg = MultiConstraintsDifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=MultiConstraintDifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore

        left_open_offset = torch.tensor([0.005, -0.007, 0.0], device=env.device).repeat(env.num_envs, 1)
        right_open_offset = torch.tensor([0.005, 0.007, 0.0], device=env.device).repeat(env.num_envs, 1)
        left_finger_close_offset = torch.tensor([0.02, -0.04, 0.0], device=env.device).repeat(env.num_envs, 1)
        right_finger_close_offset = torch.tensor([0.02, 0.04, 0.0], device=env.device).repeat(env.num_envs, 1)
        self.fully_open_length = 0.08
        self.left_finger_closeness_per_meter = (left_finger_close_offset - left_open_offset) / self.fully_open_length
        self.right_finger_closeness_per_meter = (right_finger_close_offset - right_open_offset) / self.fully_open_length

    def __call__(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor, robot_ik_cfg: SceneEntityCfg, held_asset_diameter: float
    ) -> None:
        # gripper_pos_b, gripper_quat_b = self.solver._compute_frame_pose()

        gripper_base_pos_w = self.robot.data.body_link_pos_w[:, 7]
        gripper_base_quat_w = self.robot.data.body_link_quat_w[:, 7]

        desired_left_width_offset = self.left_finger_closeness_per_meter * (
            self.fully_open_length - held_asset_diameter / 2
        )
        desired_right_width_offset = self.right_finger_closeness_per_meter * (
            self.fully_open_length - held_asset_diameter / 2
        )

        closed_left_inner_finger_pos_w, closed_left_inner_finger_quat_w = math_utils.combine_frame_transforms(
            gripper_base_pos_w, gripper_base_quat_w, desired_left_width_offset
        )
        closed_right_inner_finger_pos_w, closed_right_inner_finger_quat_w = math_utils.combine_frame_transforms(
            gripper_base_pos_w, gripper_base_quat_w, desired_right_width_offset
        )

        closed_inner_finger_pos_b, closed_inner_finger_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w.repeat_interleave(2, dim=0),
            self.robot.data.root_link_quat_w.repeat_interleave(2, dim=0),
            torch.stack((closed_left_inner_finger_pos_w, closed_right_inner_finger_pos_w), dim=1).view(-1, 3),
            torch.stack((closed_left_inner_finger_quat_w, closed_right_inner_finger_quat_w), dim=1).view(-1, 4),
        )
        self.solver.process_actions(
            torch.cat([closed_inner_finger_pos_b, closed_inner_finger_quat_b], dim=1).view(-1, 14)
        )

        # Error Rate 75% ^ 10 = 0.05 (final error)
        for _ in range(50):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )


class reset_end_effector_round_fixed_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        robot_object_held_offset: Offset = cfg.params.get("robot_object_held_offset")  # type: ignore

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.robot_object_held_offset: Offset = robot_object_held_offset
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
        self.reset_velocity = torch.zeros((env.num_envs, self.robot.data.joint_vel.shape[1]), device=env.device)
        self.reset_position = torch.zeros((env.num_envs, self.robot.data.joint_pos.shape[1]), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        robot_object_held_offset: Offset = None,
    ) -> None:
        if fixed_asset_offset is None:
            fixed_tip_pos_w, fixed_tip_quat_w = (
                env.scene[fixed_asset_cfg.name].data.root_pos_w,
                env.scene[fixed_asset_cfg.name].data.root_quat_w,
            )
        else:
            fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)

        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (env.num_envs, 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w, pos_w, quat_w
        )

        if self.robot_object_held_offset is not None:
            pos_b, quat_b = self.robot_object_held_offset.subtract(pos_b, quat_b)

        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Error Rate 75% ^ 10 = 0.05 (final error)
        for i in range(10):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )


class reset_end_effector_grasp_fixed_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        robot_object_held_offset: Offset = cfg.params.get("robot_object_held_offset")  # type: ignore
        support_asset_cfg: SceneEntityCfg = cfg.params.get("support_asset_cfg", SceneEntityCfg("table"))  # type: ignore
        grasp_angle_range: float = cfg.params.get("grasp_angle_range", 0.0)  # type: ignore
        yaw_choices: list[float] = cfg.params.get("yaw_choices", [0.0])

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.robot_object_held_offset: Offset = robot_object_held_offset
        self.support_asset: RigidObject = env.scene[support_asset_cfg.name]
        self.grasp_angle_range: float = grasp_angle_range
        self.yaw_choices: list[float] = torch.tensor(yaw_choices, device=env.device)
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
        self.reset_velocity = torch.zeros((env.num_envs, self.robot.data.joint_vel.shape[1]), device=env.device)
        self.reset_position = torch.zeros((env.num_envs, self.robot.data.joint_pos.shape[1]), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        robot_object_held_offset: Offset = None,
        support_asset_cfg: SceneEntityCfg = None,
        grasp_angle_range: float = 0.0,
        yaw_choices: list[float] = [0.0],
    ) -> None:
        if fixed_asset_offset is None:
            fixed_tip_pos_w, fixed_tip_quat_w = (
                env.scene[fixed_asset_cfg.name].data.root_pos_w,
                env.scene[fixed_asset_cfg.name].data.root_quat_w,
            )
        else:
            fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)

        # Sample random position offsets
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (env.num_envs, 3), device=env.device)
        pos_w = fixed_tip_pos_w + samples[:, 0:3]

        # Get the current end effector pose
        pos_b, quat_b = self.solver._compute_frame_pose()

        # Get the support asset's z-axis for interpolation
        support_asset_z_axis = math_utils.matrix_from_quat(self.support_asset.data.root_quat_w)[..., 2]

        grasp_angle = math_utils.sample_uniform(self.grasp_angle_range[0], self.grasp_angle_range[1], (env.num_envs, 1), device=env.device)
        # Interpolate the grasp quaternion
        desired_grasp_quat_w = sm_math_utils.interpolate_grasp_quat(
            held_asset_grasp_point_quat_w=fixed_tip_quat_w,
            grasped_object_quat_in_ee_frame=quat_b,
            secondary_z_axis=support_asset_z_axis,
            secondary_z_axis_weight=grasp_angle
        )

        # Randomize yaw in robot frame (0, pi, or 2*pi)
        yaw_indices = torch.randint(0, len(self.yaw_choices), (env.num_envs,), device=env.device)
        yaw_angles = self.yaw_choices[yaw_indices]  # Shape: [num_envs]
        yaw_quat = math_utils.quat_from_angle_axis(
            yaw_angles, 
            torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)  # Shape: [num_envs, 3]
        )  # Shape: [num_envs, 4]
        desired_grasp_quat_w = math_utils.quat_mul(desired_grasp_quat_w, yaw_quat)

        # Convert to body frame
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w, 
            self.robot.data.root_link_quat_w, 
            pos_w, 
            desired_grasp_quat_w
        )

        if self.robot_object_held_offset is not None:
            pos_b, quat_b = self.robot_object_held_offset.subtract(pos_b, quat_b)

        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Error Rate 75% ^ 10 = 0.05 (final error)
        for i in range(10):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )


def _pose_a_when_frame_ba_aligns_pose_c(
    pos_c: torch.Tensor, quat_c: torch.Tensor, pos_ba: torch.Tensor, quat_ba: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TA←W ​= {TB←A}-1 ​∘ TC←W​   where  ​combine_transform(a,b): b∘a
    inv_pos_ba = -math_utils.quat_apply(math_utils.quat_inv(quat_ba), pos_ba)
    inv_quat_ba = math_utils.quat_inv(quat_ba)
    return math_utils.combine_frame_transforms(pos_c, quat_c, inv_pos_ba, inv_quat_ba)


def sample_from_nested_dict(nested_dict: dict, idx) -> dict:
    """
    Extract a specified number of elements from a nested dictionary starting at a given index.

    Args:
        nested_dict (dict): The nested dictionary to sample from.
        start_idx (int): The starting index for sampling.
        num_elements (int): The number of elements to sample.

    Returns:
        dict: A new nested dictionary containing the sampled elements.
    """
    sampled_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # Recurse into nested dictionaries
            sampled_dict[key] = sample_from_nested_dict(value, idx)
        elif isinstance(value, torch.Tensor):
            # Slice the tensor to get the desired elements
            sampled_dict[key] = value[idx].clone()
        else:
            raise TypeError(f"Unsupported type in nested dictionary: {type(value)}")
    return sampled_dict


def sample_state_data_set(sample_state, episode_data, idx):
    """
    Sample state from episode data using given indices in a nested dictionary.

    Args:
        sample_state (dict): The dictionary to store the sampled state.
        episode_data (dict): The nested dictionary containing episode data.
        idx (torch.Tensor): The indices to sample.

    Returns:
        None
    """
    for key, value in episode_data.items():
        if isinstance(value, dict):
            # Recurse into nested dictionaries
            if key not in sample_state:
                sample_state[key] = {}
            sample_state_data_set(sample_state[key], value, idx)
        elif isinstance(value, list):
            # Handle lists of tensors
            if key not in sample_state:
                sample_state[key] = []
            sampled_tensors = [value[i] for i in idx.tolist()]
            sample_state[key] = torch.stack(sampled_tensors, dim=0)  # Stack into a tensor
        else:
            raise TypeError(f"Unsupported type in episode data: {type(value)}")


def move_dict_to_device(data: dict | torch.Tensor, device: torch.device) -> dict | torch.Tensor:
    """
    Recursively create a new dictionary with all tensors moved to the specified device.
    """
    if isinstance(data, dict):
        # Create a new dictionary and recurse on each key-value pair
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = move_dict_to_device(value, device)
        return new_dict
    elif isinstance(data, torch.Tensor):
        # Clone the tensor and move it to the specified device
        return data.clone().to(device)
    else:
        # If it's not a tensor, return it as is
        return data


class MultiResetManager(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Get configuration parameters as lists
        dataset_files: list[str] = cfg.params.get("datasets", [])
        probabilities: list[float] = cfg.params.get("probs", [])
        failure_rate_sampling: bool = cfg.params.get("failure_rate_sampling", False)

        if len(dataset_files) == 0:
            raise ValueError("No datasets provided")
        if len(dataset_files) != len(probabilities):
            raise ValueError("Number of datasets must match number of probabilities")

        self.failure_rate_sampling = failure_rate_sampling

        self.datasets = []
        self.num_states = []
        # Load all datasets
        for i, dataset_file in enumerate(dataset_files):
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Dataset file {dataset_file} does not exist.")

            dataset = torch.load(dataset_file)
            self.num_states.append(len(dataset["initial_state"]["articulation"]["robot"]["joint_position"]))
            init_indices = torch.arange(self.num_states[i], device=env.device)
            processed_dataset = {}
            sample_state_data_set(processed_dataset, dataset, init_indices)
            processed_dataset = move_dict_to_device(processed_dataset, env.device)
            self.datasets.append(processed_dataset)
        # Convert and normalize probabilities
        self.probs = torch.tensor(probabilities, device=env.device)
        self.probs = self.probs / self.probs.sum()

        # Store dataset lengths
        self.num_states = torch.tensor(self.num_states, device=env.device)

        # Initialize success monitor
        self.num_tasks = len(self.datasets)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100_000,
            num_monitored_data=self.num_tasks,
            device=env.device
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        datasets: list[str],
        probs: list[float],
        success: str,
        failure_rate_sampling: bool = False
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)

        # logs current data
        success_mask = torch.where(eval(success)[env_ids], 1.0, 0.0)
        reset_task_ids = self.task_id[env_ids]
        self.success_monitor.success_update(reset_task_ids, success_mask)

        # Calculate success rate
        num_envs = len(env_ids)

        success_rates = self.success_monitor.get_success_rate()
        if self.failure_rate_sampling:
            failure_rate = (1 - success_rates).clamp(min=1e-6)
            sample_probs = failure_rate.view(-1) * self.probs
        else:
            sample_probs = self.probs
        normalized_probs = sample_probs / sample_probs.sum()

        # Log success rates and probs for each task
        if "log" not in self._env.extras:
            self._env.extras["log"] = {}
        for task_idx in range(self.num_tasks):
            self._env.extras["log"][f"Metrics/task_{task_idx}_success_rate"] = success_rates[task_idx].item()
            self._env.extras["log"][f"Metrics/task_{task_idx}_prob"] = sample_probs[task_idx].item()
            self._env.extras["log"][f"Metrics/task_{task_idx}_normalized_prob"] = normalized_probs[task_idx].item()

        # Sample which dataset to use for each environment
        dataset_indices = torch.multinomial(normalized_probs, num_envs, replacement=True)
        self.task_id[env_ids] = dataset_indices

        # Process each dataset's environments in parallel
        for dataset_idx in range(len(self.datasets)):
            # Get environments that use this dataset
            mask = dataset_indices == dataset_idx
            if not mask.any():
                continue

            current_env_ids = env_ids[mask]

            # Sample states for these environments
            state_indices = torch.randint(0, self.num_states[dataset_idx],
                                        (len(current_env_ids),),
                                        device=self._env.device)

            # Reset these environments using sampled states
            states_to_reset_from = sample_from_nested_dict(self.datasets[dataset_idx], state_indices)
            self._env.scene.reset_to(states_to_reset_from['initial_state'],
                                   env_ids=current_env_ids,
                                   is_relative=True)

        # Reset velocities for all environments at once
        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(
            torch.zeros_like(robot.data.joint_vel[env_ids]),
            env_ids=env_ids
        )


def reset_root_states_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfgs: dict[str, SceneEntityCfg] = dict(),
):
    """Reset multiple assets' root states to random positions and velocities uniformly within given ranges.

    This function randomizes the root position and velocity of multiple assets using the same random offsets.
    This keeps the relative positioning between assets intact while randomizing their global position.

    * It samples the root position from the given ranges and adds them to each asset's default root position
    * It samples the root orientation from the given ranges and sets them into the physics simulation
    * It samples the root velocity from the given ranges and sets them into the physics simulation

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Args:
        env: The environment instance
        env_ids: The environment IDs to reset
        pose_range: Dictionary of position and orientation ranges
        velocity_range: Dictionary of linear and angular velocity ranges
        asset_cfgs: List of asset configurations to reset (all receive same random offset)
    """
    if not asset_cfgs:
        return

    # Generate a single set of random offsets that will be applied to all assets
    # Sample the device from the first asset
    asset_cfgs = list(asset_cfgs.values())
    device = env.scene[asset_cfgs[0].name].device

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    rand_pose_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=device)

    # Create orientation delta quaternion from the random Euler angles
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
    )

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    rand_vel_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=device)

    # Apply the same random offsets to each asset
    for asset_cfg in asset_cfgs:
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # Get default root state for this asset
        root_states = asset.data.default_root_state[env_ids].clone()

        # Apply position offset
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]

        # Apply orientation offset
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        # Apply velocity offset
        velocities = root_states[:, 7:13] + rand_vel_samples

        # Set the new pose and velocity into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)



# Add this function to your task_mdp.py module
def randomize_hdri(
    env,
    env_ids: torch.Tensor,
    light_path: str = "/World/skyLight",
    hdri_paths: list[str] = [f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"],
    intensity_range: tuple = (500.0, 1000.0),
) -> None:
    """Randomizes the HDRI texture and intensity.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        light_path: Path to the dome light prim.
        intensity_range: Range for intensity randomization (min, max).
    """
    # Get stage
    stage = omni.usd.get_context().get_stage()

    # Only change the global dome light once, regardless of how many environments we have
    # Get the light prim
    light_prim = stage.GetPrimAtPath(light_path)

    if not light_prim.IsValid():
        print(f"Light at {light_path} not found!")
        return

    # Get the dome light
    dome_light = UsdLux.DomeLight(light_prim)
    if not dome_light:
        print(f"Prim at {light_path} is not a dome light!")
        return

    # Choose a random HDRI
    random_hdri = random.choice(hdri_paths)

    # Set the texture file path
    try:
        texture_attr = dome_light.GetTextureFileAttr()
        texture_attr.Set(random_hdri)

        # Randomize intensity
        intensity_attr = dome_light.GetIntensityAttr()
        intensity = random.randint(intensity_range[0], intensity_range[1])
        intensity_attr.Set(intensity)

        # print(f"Sky HDRI set to: {random_hdri}, intensity: {intensity}")
    except Exception as e:
        print(f"Error setting sky HDRI: {e}")


def randomize_tiled_cameras(
    env,
    env_ids: torch.Tensor,
    camera_path_template: str,
    base_position: tuple,
    base_rotation: tuple,
    position_deltas: dict,
    euler_deltas: dict
) -> None:
    """Randomizes tiled cameras with XYZ and Euler angle deltas from base values.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        camera_path_template: Template string for camera path with {} for env index.
        base_position: Base position (x,y,z) from the camera config.
        base_rotation: Base rotation quaternion (w,x,y,z) from the camera config.
        position_deltas: Dictionary with x,y,z delta ranges to apply to base position.
        euler_deltas: Dictionary with pitch,yaw,roll delta ranges in degrees.
    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Process each environment separately
    for env_idx in env_ids:
        env_idx_value = env_idx.item() if hasattr(env_idx, 'item') else env_idx

        # Get the camera path for this environment using the template
        camera_path = camera_path_template.format(env_idx_value)

        # Get the stage
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)

        if not camera_prim.IsValid():
            print(f"Camera at {camera_path} not found!")
            continue

        # === Randomize Position ===
        pos_delta_x = random.uniform(*position_deltas["x"])
        pos_delta_y = random.uniform(*position_deltas["y"])
        pos_delta_z = random.uniform(*position_deltas["z"])

        new_pos = (
            base_position[0] + pos_delta_x,
            base_position[1] + pos_delta_y,
            base_position[2] + pos_delta_z
        )

        # === Randomize Rotation (Euler deltas in degrees, convert to radians) ===
        # Convert base quaternion (w, x, y, z) to GfQuatf
        base_quat = Gf.Quatf(base_rotation[0], Gf.Vec3f(base_rotation[1], base_rotation[2], base_rotation[3]))
        base_rot = Gf.Rotation(base_quat)

        # Create delta rotation from Euler angles (ZYX order: yaw, pitch, roll)
        delta_pitch = random.uniform(*euler_deltas["pitch"])
        delta_yaw = random.uniform(*euler_deltas["yaw"])
        delta_roll = random.uniform(*euler_deltas["roll"])

        delta_rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), delta_yaw) * \
            Gf.Rotation(Gf.Vec3d(0, 1, 0), delta_pitch) * \
            Gf.Rotation(Gf.Vec3d(1, 0, 0), delta_roll)

        # Apply delta rotation to base rotation
        new_rot = delta_rot * base_rot
        new_quat = new_rot.GetQuat()

        # === Apply pose to the USD prim ===
        xform = UsdGeom.Xformable(camera_prim)
        xform_ops = xform.GetOrderedXformOps()

        if not xform_ops:
            xform.AddTransformOp()

        # Set translation and orientation
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*new_pos))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(new_quat)


def randomize_camera_focal_length(
    env,
    env_ids: torch.Tensor,
    camera_path_template: str,
    focal_length_range: tuple = (0.8, 1.8)
) -> None:
    """Randomizes the focal length of cameras.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        camera_path_template: Template for camera path with {} for env index.
        focal_length_range: Range for focal length randomization (min, max) in mm.
    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Get the USD stage
    stage = omni.usd.get_context().get_stage()

    # Process each environment
    for env_idx in env_ids:
        # Get the camera path for this environment
        camera_path = camera_path_template.format(env_idx)

        # Get the camera prim
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            print(f"Camera at {camera_path} not found!")
            continue

        # Generate random values within specified ranges
        focal_length = random.uniform(focal_length_range[0], focal_length_range[1])

        # Set the focal length
        focal_attr = camera_prim.GetAttribute("focalLength")
        if focal_attr.IsValid():
            focal_attr.Set(focal_length)
            # print(f"Set focal length to {focal_length} for camera {camera_path}")
        else:
            print(f"Focal length attribute not found for camera {camera_path}")


class randomize_visual_color_multiple_meshes(ManagerTermBase):
    """Randomize the visual color of multiple mesh bodies on an asset using Replicator API.

    This function randomizes the visual color of multiple mesh bodies of the asset using the Replicator API.
    The function samples a single random color and applies it to all specified mesh bodies of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and mesh names ["body_0/mesh", "body_1/mesh"], the prim paths for the meshes would be
    "/World/asset/body_0/mesh" and "/World/asset/body_1/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term."""
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        colors = cfg.params.get("colors")
        event_name = cfg.params.get("event_name")
        mesh_names: list[str] = cfg.params.get("mesh_names", [])  # type: ignore

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual color with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # create the affected prim paths for all mesh names
        mesh_prim_paths = []
        for mesh_name in mesh_names:
            if not mesh_name.startswith("/"):
                mesh_name = "/" + mesh_name
            mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
            mesh_prim_paths.append(mesh_prim_path)

        # parse the colors into replicator format
        if isinstance(colors, dict):
            # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
            color_low = [colors[key][0] for key in ["r", "g", "b"]]
            color_high = [colors[key][1] for key in ["r", "g", "b"]]
            colors = rep.distribution.uniform(color_low, color_high)
        else:
            colors = list(colors)

        # Create the omni-graph node for the randomization term
        def rep_texture_randomization():
            # Apply the same color to all mesh prims
            for mesh_prim_path in mesh_prim_paths:
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)

                with prims_group:
                    rep.randomizer.color(colors=colors)

            return

        # Register the event to the replicator
        with rep.trigger.on_custom_event(event_name=event_name):
            rep_texture_randomization()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_names: list[str] = [],
    ):
        # import replicator
        import omni.replicator.core as rep

        # only send the event to the replicator
        rep.utils.send_og_event(event_name)
