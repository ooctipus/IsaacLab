# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.events import reset_root_state_uniform
import isaaclab.sim as sim_utils
from .utils import sample_object_point_cloud, prim_to_warp_mesh, get_pen_multi_agg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from pxr import Usd
    
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloudDebug")
ray_cfg.markers["hit"].radius = 0.01
visualizer = VisualizationMarkers(ray_cfg)


class reset_asset_collision_free(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.max_dist = 0.5
        self.num_points = 128
        asset: RigidObject = env.scene[cfg.params.get("collision_check_asset_cfg").name]
        obstacle: RigidObject = env.scene[cfg.params.get("collision_check_against_asset_cfg").name]
        self.local_pts = sample_object_point_cloud(
            num_envs=env.num_envs, num_points=self.num_points, prim_path=asset.cfg.prim_path, device=env.device
        )

        self.obstacle_meshes: list[tuple[wp.Mesh, Usd.Prim]] = []
        all_handles = []
        prim_counts = []
        for i in range(env.num_envs):
            obj_path = obstacle.cfg.prim_path.replace(".*", str(i))
            prims = sim_utils.get_all_matching_child_prims(
                obj_path,
                predicate=lambda p: p.GetTypeName() in ("Cube","Sphere","Cylinder","Capsule","Cone","Mesh")
            )
            ids = []
            for p in prims:
                # convert each USD prim → Warp mesh...
                wp_mesh = prim_to_warp_mesh(p, device=env.device)
                self.obstacle_meshes.append((wp_mesh, p))
                ids.append(int(wp_mesh.id))
            all_handles.append(ids)
            prim_counts.append(len(ids))

        self.max_prims = max(prim_counts)
        assert self.max_prims > 0, f"No collision primitives found under {obstacle.cfg.prim_path}"
        padded = [ids + [0]*(self.max_prims - len(ids)) for ids in all_handles]
        self.handles_tensor = torch.tensor(padded, dtype=torch.int64, device=env.device)
        self.prim_counts = torch.tensor(prim_counts, dtype=torch.int32, device=env.device)
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,  # [E] initial env indices
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        collision_check_asset_cfg: SceneEntityCfg,
        collision_check_against_asset_cfg: SceneEntityCfg,
    ):
        asset = env.scene[collision_check_asset_cfg.name]
        collision_ids = env_ids.clone()
        while True:
            while collision_ids.numel() > 0:
                reset_root_state_uniform(env, collision_ids, pose_range, velocity_range, collision_check_asset_cfg)

                pos_w = asset.data.root_pos_w[collision_ids].unsqueeze(1).expand(-1, self.num_points, -1)
                quat_w = asset.data.root_quat_w[collision_ids].unsqueeze(1).expand(-1, self.num_points, -1)
                cloud = math_utils.quat_apply(quat_w, self.local_pts[collision_ids]) + pos_w
                env_id_left = cloud.shape[0]

                handles_sub = self.handles_tensor[collision_ids]
                counts_sub = self.prim_counts[collision_ids]
                queries_w = wp.from_torch(cloud.reshape(-1,3), dtype=wp.vec3)
                handles_w = wp.from_torch(handles_sub.reshape(-1), dtype=wp.uint64)
                counts_w = wp.from_torch(counts_sub, dtype=wp.int32)
                is_pen_w = wp.zeros(env_id_left * self.num_points, dtype=wp.bool)

                M = env_id_left * self.num_points
                res_w = wp.zeros(M, dtype=wp.bool)
                sign_w  = wp.zeros(M, dtype=float)
                # wp.launch(debug_sign_multi, dim=M, inputs=[queries_w, handles_w, res_w, sign_w, float(self.max_dist), self.num_points])
                
                wp.launch(
                    get_pen_multi_agg,
                    dim=M,
                    inputs=[queries_w, handles_w, counts_w, float(self.max_dist), self.num_points, self.max_prims],
                    outputs=[sign_w]
                )
                signs = wp.to_torch(sign_w)
                collision_points = []
                for mesh, prim in self.obstacle_meshes:
                    collision_points.append(wp.to_torch(mesh.points))
                
                # collision_p = torch.cat(collision_points, dim=0)
                # pen_points = cloud.view(-1, 3)[signs < 0]

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