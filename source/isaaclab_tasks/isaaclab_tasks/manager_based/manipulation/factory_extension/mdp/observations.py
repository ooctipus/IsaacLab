# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from . import key_point_maths as advanced_math
# from . import maths as advanced_math
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.assets import RigidObject, RigidObjectCollection, Articulation
import trimesh
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    # from isaaclab.assets import RigidObject, Articulation
    # from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

    from .data_cfg import KeyPointDataCfg
    from .data_cfg import AlignmentDataCfg as Align
    from .data_cfg import AlignmentMetric


def task_encoding(env: DataManagerBasedRLEnv, command_term="task_command") -> torch.Tensor:
    """Returns the task encoding for the given task ID."""
    command_term = env.command_manager.get_term(command_term)
    return command_term.one_hot_command


def target_asset_pose_in_root_asset_frame_min_error(
    env: DataManagerBasedRLEnv,
    alignment_cfg: Align,
):
    alignment: AlignmentMetric.AlignmentData = alignment_cfg.get(env.data_manager)
    return torch.cat([alignment.pos_delta, alignment.rot_delta], dim=1)  # [envs, 6]


def target_asset_pose_in_root_asset_frame(
    env: DataManagerBasedRLEnv,
    target_kp_cfg: KeyPointDataCfg,
    root_kp_cfg: KeyPointDataCfg,
):
    target_pose_w, target_pose_mask = target_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)
    root_pose_w, root_pose_mask = root_kp_cfg.get(env.data_manager)  # (n_envs, 1, n_assets, n_offsets, 7)

    set2 = target_pose_w.view(env.num_envs, -1, 7)
    mask2 = target_pose_mask.view(env.num_envs, -1).bool()
    set1 = root_pose_w.view(env.num_envs, -1, 7)
    mask1 = root_pose_mask.view(env.num_envs, -1).bool()

    # s: [B, N, 7], m: [B, N]
    s1_pairs, s2_pairs, pair_mask = advanced_math.cartesian_pairwise(set1, set2, mask1, mask2)
    N = s1_pairs.size(1)
    out_flat = torch.zeros(env.num_envs * N, 7, device=env.device)
    if torch.any(pair_mask):
        root_flat = s1_pairs.reshape(env.num_envs * N, 7)
        target_flat = s2_pairs.reshape(env.num_envs * N, 7)
        mask_flat = pair_mask.reshape(env.num_envs * N)

        r_sel = root_flat[mask_flat]  # [N_valid, 7]
        t_sel = target_flat[mask_flat]  # [N_valid, 7]

        pos_delta, quat_delta = math_utils.subtract_frame_transforms(  # [N_valid, 3], [N_valid, 4]
            r_sel[:, :3], r_sel[:, 3:],
            t_sel[:, :3], t_sel[:, 3:],
        )
        delta = torch.cat([pos_delta, quat_delta], dim=1)  # [N_valid, 7]
        out_flat[mask_flat] = delta

    out = out_flat.view(env.num_envs, -1)

    return out


def fps(xyz: torch.Tensor, npoint: int):
    """
    Input:
        xyz: (N, 3) point‐cloud
        npoint: number of samples
    Return:
        centroids: (npoint,) indices of sampled points
    """
    device = xyz.device
    N, _ = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    # keep track of min distance to any chosen centroid
    distances = torch.full((N,), float('inf'), device=device)
    # start from a random point
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()

    for i in range(npoint):
        centroids[i] = farthest
        # compute squared Euclidean dist from the newest centroid
        diff = xyz - xyz[farthest].unsqueeze(0)      # (N,3)
        dist2 = torch.sum(diff*diff, dim=1)          # (N,)
        # update the running minimum distances
        distances = torch.minimum(distances, dist2)
        # next farthest is the point with the largest min‐distance
        farthest = torch.argmax(distances).item()

    return centroids


class PointCloud(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: DataManagerBasedRLEnv):
        from pxr import UsdGeom
        from isaaclab.sim.utils import get_all_matching_child_prims
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        self.robot: RigidObject | Articulation = env.scene[self.robot_cfg.name]

        self.object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self.object: RigidObjectCollection = env.scene[self.object_cfg.name]

        self.num_points: int = cfg.params["num_points"]

        if not isinstance(self.robot, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.robot_cfg.name}'"
                f" with type: '{type(self.robot)}'."
            )

        if not isinstance(self.object, (RigidObjectCollection)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.object_cfg.name}'"
                f" with type: '{type(self.object)}'."
            )

        # uncomment to visualize
        if cfg.params["visualize"]:
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            from isaaclab.markers import VisualizationMarkers
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
            ray_cfg.markers["hit"].radius = 0.001
            self.visualizer = VisualizationMarkers(ray_cfg)

        self.points = torch.zeros((env.num_envs, len(self.object.cfg.rigid_objects), self.num_points, 3), device=self.device)
        self.object = env.scene[cfg.params["object_cfg"].name]
        for i in range(env.num_envs):
            
            # self.object is actually a RigidObjectCollection
            # object_names_list = list(self.object.cfg.rigid_objects.keys())
            object_cfgs_list = list(self.object.cfg.rigid_objects.values())

            for id in cfg.params["object_cfg"].object_collection_ids:
                # this gets points at (0, 0, 0) pos and rot so we don't need to transform points
                object_cfg = object_cfgs_list[id]
                prim_path = object_cfg.prim_path
                prims = get_all_matching_child_prims(prim_path.replace(".*", str(i)), predicate=lambda prim: prim.GetTypeName() == "Mesh")
                vertex_points_w = torch.tensor([], device=self.device)

                for prim in prims:
                    mesh = UsdGeom.Mesh(prim)
                    vertices = np.array(mesh.GetPointsAttr().Get())

                    # load face‐counts and face‐indices
                    counts = mesh.GetFaceVertexCountsAttr().Get()
                    indices = mesh.GetFaceVertexIndicesAttr().Get()

                    # triangulate “poly” faces into a (F,3) array
                    faces = []
                    it = iter(indices)
                    for cnt in counts:
                        poly = [next(it) for _ in range(cnt)]
                        # fan‐triangulate
                        for k in range(1, cnt-1):
                            faces.append([poly[0], poly[k], poly[k+1]])
                    
                    faces = np.array(faces, dtype=np.int64)

                    # build trimesh and sample
                    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    samples, __ = tm.sample(self.num_points, return_index=True)
                    self.points[i, id] = torch.from_numpy(samples).to(self.device)
                    
                    # mesh_points = torch.tensor(UsdGeom.Mesh(prim).GetPointsAttr().Get(), device=self.device)
                    # vertex_points_w = torch.cat((vertex_points_w, mesh_points), dim=0)

                # rand_indices = fps(vertex_points_w, self.num_points)
                # self.points[i, id] = vertex_points_w[rand_indices]
                # self.points[i, id] = vertex_points_w[:self.num_points]


    def __call__(
        self,
        env: DataManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        num_points: int = 1000,
        visualize: bool = True
    ):
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObjectCollection = env.scene[object_cfg.name]

        num_objects = object.data.object_pos_w.shape[1]

        object_pos_w = object.data.object_pos_w.unsqueeze(2).repeat(1, 1, num_points, 1)
        object_rot_w = object.data.object_quat_w.unsqueeze(2).repeat(1, 1, num_points, 1)

        # apply rotation + translation
        object_point_cloud_w = math_utils.quat_apply(object_rot_w.reshape(-1, num_points, 4), self.points.reshape(-1, num_points, 3))
        object_point_cloud_w = object_point_cloud_w + object_pos_w.reshape(-1, num_points, 3)

        if visualize:
            self.visualizer.visualize(translations=object_point_cloud_w.reshape(-1, 3))

        # transform to robot frame
        object_point_cloud_b, _ = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_points, 1).reshape(-1, num_points, 3), 
            robot.data.root_quat_w.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_points, 1).reshape(-1, num_points, 4),
            object_point_cloud_w
        )

        return object_point_cloud_b.reshape(env.num_envs, num_objects * num_points * 3)
