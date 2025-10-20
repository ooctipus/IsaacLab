# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from . import utils as dexsuite_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloudDebug")
ray_cfg.markers["hit"].radius = 0.005
visualizer = VisualizationMarkers(ray_cfg)


class reset_accumulator(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.acceptance_conditions = cfg.params.get("acceptance_conditions")
        for key, val in self.acceptance_conditions.items():
            if hasattr(val, "class_type"):
                self.acceptance_conditions[key] = val.class_type(val, env)

        asset_keys = cfg.params.get("reset_assets")
        total_state_dim = dexsuite_utils.get_reset_state(
            self._env, torch.tensor([0], device=env.device), asset_keys
        ).shape[-1]
        self.max_size = 32
        self.valid_tensor = torch.zeros((env.num_envs, self.max_size, total_state_dim), device=env.device)
        self.valid_state_tensor_size = torch.zeros((env.num_envs,), device=env.device, dtype=torch.int)
        self.valid_ptr = torch.zeros((env.num_envs,), device=env.device, dtype=torch.int)
        self.precollecting_phase = True

        reset_term: EventTermCfg = cfg.params.get("reset_term")
        if inspect.isclass(reset_term.func):
            reset_term.func = reset_term.func(reset_term, env)
        while (self.valid_state_tensor_size < self.max_size).any():
            env_ids = torch.arange(env.num_envs, device=env.device)[self.valid_state_tensor_size < self.max_size]
            self.__call__(env, env_ids, **self.cfg.params)

        self.precollecting_phase = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reset_term: EventTermCfg,
        reset_assets: list[str],
        acceptance_conditions: dict,
    ):
        reset_term.func(env, env_ids, **reset_term.params)
        valid_mask = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
        for key, val in self.acceptance_conditions.items():
            valid_mask &= val(env, env_ids)

        valid_env_ids = env_ids[valid_mask]
        invalid_env_ids = env_ids[~valid_mask]

        states = dexsuite_utils.get_reset_state(self._env, valid_env_ids, reset_assets)
        idx = self.valid_ptr[valid_env_ids]
        self.valid_tensor[valid_env_ids, idx] = states
        self.valid_ptr[valid_env_ids] = (idx + 1) % self.max_size
        self.valid_state_tensor_size[valid_env_ids] = torch.clamp(
            self.valid_state_tensor_size[valid_env_ids] + 1, max=self.max_size
        )
        if not self.precollecting_phase:
            rand_idx = torch.randint(0, self.max_size, (len(invalid_env_ids),), device=env.device)
            sampled_states = self.valid_tensor[invalid_env_ids, rand_idx]
            dexsuite_utils.set_reset_state(self._env, sampled_states, invalid_env_ids, reset_assets)


class reset_end_effector_around_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        reach_asset_cfg: SceneEntityCfg = cfg.params.get("reach_asset_cfg")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.reach_asset: Articulation | RigidObject = env.scene[reach_asset_cfg.name]
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
        reach_asset_cfg: SceneEntityCfg,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
        ik_iterations: int = 10,
    ) -> None:
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_tip_pos_w, fixed_tip_quat_w = self.reach_asset.data.root_pos_w, self.reach_asset.data.root_quat_w
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
            env.sim.render()
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )
        self.robot.root_physx_view.get_jacobians()


class chained_reset_terms(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terms: dict[str, EventTermCfg] = cfg.params["terms"]  # type: ignore
        for term_name, term_cfg in self.terms.items():
            for key, val in term_cfg.params.items():
                if isinstance(val, SceneEntityCfg):
                    val.resolve(env.scene)

        for term_name, term_cfg in self.terms.items():
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(term_cfg, env)  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        terms: dict[str, callable],
    ) -> None:
        env_ids_to_reset = env_ids
        for func_name, term in terms.items():
            term.func(env, env_ids_to_reset, **term.params)  # type: ignore
