# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import inspect
import warp as wp
from pxr import UsdPhysics
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.utils import get_first_matching_child_prim
import isaaclab.sim as sim_utils
from . import utils as dexsuite_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from pxr import Usd
    
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloudDebug")
ray_cfg.markers["hit"].radius = 0.005
visualizer = VisualizationMarkers(ray_cfg)


class reset_asset_collision_free(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.max_dist = 0.5 
        self.num_points = 32
        asset_cfg = cfg.params.get("collision_check_asset_cfg")
        asset: RigidObject = env.scene[asset_cfg.name]
        body_names = asset.body_names if asset_cfg.body_names is None else asset_cfg.body_names
        if isinstance(body_names, str):
            body_names = [body_names]

        self.body_ids = []
        self.local_pts = []
        for body_name in body_names:
            prim = get_first_matching_child_prim(
                asset.cfg.prim_path.replace(".*", "0", 1),  # we use the 0th env prim as template
                predicate=lambda p: p.GetName() == body_name and p.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            local_pts = dexsuite_utils.sample_object_point_cloud(
                num_envs=env.num_envs,
                num_points=self.num_points,
                prim_path_pattern=str(prim.GetPath()).replace("env_0", "env_.*", 1),
                device=env.device
            )
            if local_pts is not None:
                self.local_pts.append(local_pts.view(env.num_envs, 1, self.num_points, 3))
                self.body_ids.append(asset.body_names.index(body_name))
        self.local_pts = torch.cat(self.local_pts, dim=1)
        self.body_ids = torch.tensor(self.body_ids, dtype=torch.int, device=env.device)
        obstacle_cfgs: list[SceneEntityCfg] = cfg.params.get("collision_check_against_asset_cfg")
        self.obstacle_meshes: list[tuple[wp.Mesh, Usd.Prim]] = []
        all_handles = []
        prim_counts = []
        for i in range(env.num_envs):
            ids = []
            for obstacle_cfg in obstacle_cfgs:
                obj_path = env.scene[obstacle_cfg.name].cfg.prim_path.replace(".*", str(i))
                prims = sim_utils.get_all_matching_child_prims(
                    obj_path,
                    predicate=lambda p: p.GetTypeName() in ("Cube","Sphere","Cylinder","Capsule","Cone","Mesh") and p.HasAPI(UsdPhysics.CollisionAPI)
                )
                for p in prims:
                    # convert each USD prim → Warp mesh...
                    wp_mesh = dexsuite_utils.prim_to_warp_mesh(p, device=env.device, relative_to_world=True)
                    self.obstacle_meshes.append((wp_mesh, p))
                    ids.append(int(wp_mesh.id))
            all_handles.append(ids)
            prim_counts.append(len(ids))

        self.max_prims = max(prim_counts)
        assert self.max_prims > 0, f"No collision primitives found under {obstacle_cfgs}"
        padded = [ids + [0]*(self.max_prims - len(ids)) for ids in all_handles]
        self.handles_tensor = torch.tensor(padded, dtype=torch.int64, device=env.device)
        self.prim_counts = torch.tensor(prim_counts, dtype=torch.int32, device=env.device)

        self.asset_keys = [asset_cfg.name, *[cfg.name for cfg in obstacle_cfgs]]
        total_state_dim = dexsuite_utils.get_reset_state(self._env, torch.tensor([0], device=env.device), self.asset_keys).shape[-1]
        self.max_size = 32
        self.collision_free_tensor = torch.zeros((env.num_envs, self.max_size, total_state_dim), device=env.device)
        self.collision_free_state_tensor_size = torch.zeros((env.num_envs,), device=env.device, dtype=torch.int)
        self.collision_free_ptr = torch.zeros((env.num_envs,), device=env.device, dtype=torch.int)
        self.precollecting_phase = True
        
        reset_term: EventTermCfg = cfg.params.get("reset_term")
        if inspect.isclass(reset_term.func):
            reset_term.func = reset_term.func(reset_term, env)
        while (self.collision_free_state_tensor_size < self.max_size).any():
            env_ids = torch.arange(env.num_envs, device=env.device)[self.collision_free_state_tensor_size < self.max_size]
            self.__call__(env, env_ids, **self.cfg.params)
            
        self.precollecting_phase = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        reset_term: EventTermCfg,
        collision_check_asset_cfg: SceneEntityCfg,
        collision_check_against_asset_cfg: SceneEntityCfg,
    ):
        reset_term.func(env, env_ids, **reset_term.params)
        asset: RigidObject = env.scene[collision_check_asset_cfg.name]
        pos_w = asset.data.body_link_pos_w[env_ids][:, self.body_ids].unsqueeze(2).expand(-1, -1, self.num_points, 3)
        quat_w = asset.data.body_link_quat_w[env_ids][:, self.body_ids].unsqueeze(2).expand(-1, -1, self.num_points, 4)
        cloud = math_utils.quat_apply(quat_w, self.local_pts[env_ids]) + pos_w

        total_points = len(self.body_ids) * self.num_points
        handles_sub = self.handles_tensor[env_ids]
        counts_sub = self.prim_counts[env_ids]
        queries_w = wp.from_torch(cloud.reshape(-1,3), dtype=wp.vec3)
        handles_w = wp.from_torch(handles_sub.reshape(-1), dtype=wp.uint64)
        counts_w = wp.from_torch(counts_sub, dtype=wp.int32)
        sign_w  = wp.zeros((len(env_ids) * self.num_points * len(self.body_ids),), dtype=float, device=env.device)
        wp.launch(
            dexsuite_utils.get_sign_distance,
            dim=len(env_ids) * total_points,
            inputs=[queries_w, handles_w, counts_w, float(self.max_dist), total_points, self.max_prims],
            outputs=[sign_w],
            device=env.device,
        )
        signs = wp.to_torch(sign_w).view(len(env_ids), len(self.body_ids), self.num_points)

        # collision_points = cloud[(signs < 0.0)]
        # collision_against_points = torch.cat([wp.to_torch(mesh[0].points) for mesh in self.obstacle_meshes], dim=0)
        # while True:
        #     env.sim.render()
        #     visualizer.visualize(torch.cat((collision_points.view(-1, 3), collision_against_points.view(-1, 3)), dim=0))

        cooll_free_mask = (signs >= 0.0).view(len(env_ids), -1).all(dim=1).bool()
        coll_free_id = env_ids[cooll_free_mask]
        coll_id = env_ids[~cooll_free_mask]

        states = dexsuite_utils.get_reset_state(self._env, coll_free_id, self.asset_keys)
        idx = self.collision_free_ptr[coll_free_id]
        self.collision_free_tensor[coll_free_id, idx] = states
        self.collision_free_ptr[coll_free_id] = (idx + 1) % self.max_size
        self.collision_free_state_tensor_size[coll_free_id] = torch.clamp(self.collision_free_state_tensor_size[coll_free_id] + 1, max=self.max_size)
        if not self.precollecting_phase:
            rand_idx = torch.randint(0, self.max_size, (len(coll_id),), device=env.device)
            sampled_states = self.collision_free_tensor[coll_id, rand_idx]
            dexsuite_utils.set_reset_state(self._env, sampled_states, coll_id, self.asset_keys)


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