# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import inspect
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from ..assembly_keypoints import KEYPOINTS_NISTBOARD
from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..assembly_keypoints import Offset

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


def reset_fixed_assets(env: ManagerBasedRLEnv, env_ids: torch.tensor, asset_list: list[str]):
    nistboard: RigidObject = env.scene["nistboard"]
    for asset_str in asset_list:
        asset: Articulation | RigidObject = env.scene[asset_str]
        asset_offset_on_nist_board: Offset = getattr(KEYPOINTS_NISTBOARD, asset_str)
        asset_on_board_pos, asset_on_board_quat = asset_offset_on_nist_board.apply(nistboard)
        root_pose = torch.cat((asset_on_board_pos, asset_on_board_quat), dim=1)[env_ids]
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(asset.data.root_vel_w[env_ids]), env_ids=env_ids)


def reset_held_asset_on_fixed_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    assembled_offset: Offset,
    entry_offset: Offset,
    assembly_fraction_range: tuple[float, float],
    assembly_ratio: tuple[float, float, float], # m / radian
    fixed_asset_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
):
    fixed_asset: RigidObject = env.scene[fixed_asset_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]
    
    assembly_fraction = math_utils.sample_uniform(
        assembly_fraction_range[0], assembly_fraction_range[1], (len(env_ids), 1), device=env.device
    )
    pos_delta = torch.tensor(entry_offset.pos, device=env.device) - torch.tensor(assembled_offset.pos, device=env.device)
    pos_delta = pos_delta.repeat(len(env_ids), 1) * assembly_fraction
    ratio = torch.tensor(assembly_ratio, device=env.device)
    rot_delta = math_utils.wrap_to_pi(torch.where(ratio != 0, 1 / ratio * pos_delta, 0.0))
    quat_delta = math_utils.quat_from_euler_xyz(rot_delta[:, 0], rot_delta[:, 1], rot_delta[:, 2])
    held_asset_on_fixed_asset_pose = torch.cat(math_utils.combine_frame_transforms(
        fixed_asset.data.root_pos_w[env_ids], fixed_asset.data.root_quat_w[env_ids],
        pos_delta, quat_delta
    ), dim=1)
    held_asset.write_root_pose_to_sim(held_asset_on_fixed_asset_pose, env_ids=env_ids)


def reset_held_asset_in_gripper(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    holding_body_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    held_asset_graspable_offset: Offset,
    held_asset_inhand_range: dict[str, tuple[float, float]],
):
    robot: Articulation = env.scene[holding_body_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    end_effector_quat_w = robot.data.body_link_quat_w[env_ids, holding_body_cfg.body_ids].view(-1, 4)
    end_effector_pos_w = robot.data.body_link_pos_w[env_ids, holding_body_cfg.body_ids].view(-1, 3)
    held_graspable_pos_b = torch.tensor(held_asset_graspable_offset.pos, device=env.device).repeat(len(env_ids), 1)
    held_graspable_quat_b = torch.tensor(held_asset_graspable_offset.quat, device=env.device).repeat(len(env_ids), 1)

    flip_z_quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=env.device).repeat(len(env_ids), 1)
    translated_held_asset_pos, translated_held_asset_quat = _pose_a_when_frame_ba_aligns_pose_c(
        pos_c=end_effector_pos_w,
        quat_c=math_utils.quat_mul(end_effector_quat_w, flip_z_quat),
        pos_ba=held_graspable_pos_b,
        quat_ba=held_graspable_quat_b,
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


def grasp_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_diameter: float,
) -> None:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][env_ids].clone()
    joint_pos[:, :] = held_asset_diameter / 2 * 1.05
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), robot_cfg.joint_ids, env_ids)  # type: ignore


class reset_end_effector_around_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = None  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        ik_iterations: int = 10,
    ) -> None:
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w[env_ids] + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids], self.robot.data.root_link_quat_w[env_ids], pos_w, quat_w
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
        n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        
        # Error Rate 75% ^ 10 = 0.05 (final error)
        for i in range(ik_iterations):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )
        self.robot.root_physx_view.get_jacobians()


def reset_root_state_uniform_on_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    offset: Offset,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples
    offset_pose = torch.tensor(offset.pose, device=env.device).repeat(len(env_ids), 1)
    positions, orientations = _pose_a_when_frame_ba_aligns_pose_c(
        positions.view(-1, 3), orientations.view(-1, 4), offset_pose[:, :3], offset_pose[:,3:]
    )

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


class TermChoice(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_partitions: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        self.num_partitions = len(self.term_partitions)
        sampling_strategy = cfg.params.get("sampling_strategy", "uniform")  # type: ignore
        for term_name, term_cfg in self.term_partitions.items():
            for key, val in term_cfg.params.items():
                if isinstance(val, SceneEntityCfg):
                    val.resolve(env.scene)

        for term_name, term_cfg in self.term_partitions.items():
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(term_cfg, env)  # type: ignore
        
        self.term_samples = torch.zeros((env.num_envs,), dtype=torch.int, device=env.device)
        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self.num_partitions,
            device=env.device,
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, ManagerTermBase],
        sampling_strategy: Literal["uniform", "failure_rate"] = "uniform",
    ) -> None:
        success_rate = self.success_monitor.get_success_rate()
        log = {f"Metrics/{name}": success_rate[i].item() for i, name in enumerate(self.term_partitions.keys())}

        context_term: ManagerTermBase = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
        orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")[env_ids]
        position_centered: torch.Tensor = getattr(context_term, "position_centered")[env_ids]
        z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")[env_ids]
        term_successes = torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)
        self.success_monitor.success_update(self.term_samples[env_ids], term_successes)
        
        if sampling_strategy == "uniform":
            self.term_samples[env_ids] = torch.randint(0, self.num_partitions, (env_ids.size(0),), device=env_ids.device, dtype=self.term_samples.dtype)
        else:
            self.term_samples[env_ids] = self.success_monitor.failure_rate_sampling(env_ids)

        i = 0
        for term_name, term_cfg in self.term_partitions.items():
            # get the env_ids that belong to the current term
            term_ids = env_ids[self.term_samples[env_ids] == i]
            if term_ids.numel() > 0:
                term_cfg.func(env, term_ids, **term_cfg.params)
            i += 1
        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"].update(log)  # type: ignore
        


class ChainedResetTerms(ManagerTermBase):

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
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
        env: ManagerBasedRLEnv,
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

def _pose_a_when_frame_ba_aligns_pose_c(
    pos_c: torch.Tensor, quat_c: torch.Tensor, pos_ba: torch.Tensor, quat_ba: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TA←W ​= {TB←A}-1 ​∘ TC←W​   where  ​combine_transform(a,b): b∘a
    inv_pos_ba = -math_utils.quat_apply(math_utils.quat_inv(quat_ba), pos_ba)
    inv_quat_ba = math_utils.quat_inv(quat_ba)
    return math_utils.combine_frame_transforms(pos_c, quat_c, inv_pos_ba, inv_quat_ba)
