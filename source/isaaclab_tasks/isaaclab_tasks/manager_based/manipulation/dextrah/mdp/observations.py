# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse, quat_apply, quat_inv, quat_mul
from .utils import sample_object_point_cloud
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCamera, Camera, RayCasterCamera


def object_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position and quaternion of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_quat_w
    )
    return torch.cat((object_pos_b, object_quat_b), dim=1)

def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply(quat_inv(robot.data.root_quat_w), object.data.root_pos_w - robot.data.root_pos_w)
    return object_pos_b

def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The quaternion of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)

def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].clone().view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1) 
    return out.view(env.num_envs, -1)



def body_pos_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
    flatten: bool = False,
) -> torch.Tensor:
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    body_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w)
    # concate and return
    return body_pos_b.view(env.num_envs, -1) if flatten else body_pos_b.view(env.num_envs, -1, 3)

class objects_point_cloud_b(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        object_cfgs: list[SceneEntityCfg] = cfg.params.get("object_cfgs", [SceneEntityCfg('object')])
        self.num_points: list[int] = cfg.params.get("num_points", [10])
        self.visualize = cfg.params.get("visualize", True)
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.statics: list[bool] = cfg.params.get("statics", [False])
        self.objects: list[RigidObject] = [env.scene[object_cfg.name] for object_cfg in object_cfgs]
        self.body_ids = []
        self.body_names: list = [object.data.body_names if isinstance(cfg.body_ids, slice) else cfg.body_names for object, cfg in zip(self.objects, object_cfgs)]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]

        if self.visualize:
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            from isaaclab.markers import VisualizationMarkers
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0055
            self.visualizer = VisualizationMarkers(ray_cfg)

        points = []
        for object, n_point, body_name in zip(self.objects, self.num_points, self.body_names):
            if len(body_name) == 1:
                local_pts = sample_object_point_cloud(env.num_envs, n_point, object.cfg.prim_path, device=env.device)
                if local_pts is not None:
                    points.append(local_pts)
                    self.body_ids.append([0])
                else:
                    raise(f"Found no Collider to sample point cloud in {object.cfg.prim_path}")
            else:
                body_ids = []
                for bn in body_name:
                    local_pts = sample_object_point_cloud(env.num_envs, n_point, f"{object.cfg.prim_path}/{bn}", device=env.device)
                    if local_pts is not None:
                        points.append(local_pts)
                        body_ids.append(object.body_names.index(bn))
                self.body_ids.append(body_ids)
        self.points_b = torch.cat(points, dim=1)
        self.points_w = torch.zeros_like(self.points_b)
    
    def reset(self, env_ids = slice(None)):
        idx = 0
        for i, object in enumerate(self.objects):
            if self.statics[i]:
                object_pos_w = object.data.root_pos_w[env_ids].unsqueeze(1).repeat(1, self.num_points[i], 1)
                object_quat_w = object.data.root_quat_w[env_ids].unsqueeze(1).repeat(1, self.num_points[i], 1)
                self.points_w[env_ids, idx : idx + self.num_points[i]] = (quat_apply(object_quat_w, self.points_b[env_ids, idx : idx + self.num_points[i]]) + object_pos_w)
            idx += self.num_points[i]
            
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfgs: list[SceneEntityCfg] = [SceneEntityCfg("object")],
        num_points: list[int] = [10],
        statics: list[bool] = [False],
        visualize: bool = True,
        normalize: bool = False,
        flatten: bool = False,
    ):
        idx = 0
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, self.points_w.shape[1], 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, self.points_w.shape[1], 1)
        for object, body_id, static, n_pts in zip(self.objects, self.body_ids, statics, num_points):
            if not static:
                pos_w = object.data.body_link_pos_w[:, body_id].unsqueeze(2).expand(-1, -1, n_pts, 3).reshape(env.num_envs, -1, 3)
                quat_w = object.data.body_link_quat_w[:, body_id].unsqueeze(2).expand(-1, -1, n_pts, 4).reshape(env.num_envs, -1, 4)
                self.points_w[:, idx : idx + n_pts * len(body_id)] = quat_apply(quat_w, self.points_b[:, idx : idx + n_pts * len(body_id)]) + pos_w
            idx += n_pts

        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)
        if normalize:
            object_point_cloud_pos_b = object_point_cloud_pos_b - object_point_cloud_pos_b.mean(dim=1, keepdim=True)
            d = torch.norm(object_point_cloud_pos_b, dim=-1)
            m = d.max(dim=1, keepdim=True)[0]
            object_point_cloud_pos_b = object_point_cloud_pos_b / (m.unsqueeze(-1) + 1e-6)
        return object_point_cloud_pos_b.view(self.num_envs, -1) if flatten else object_point_cloud_pos_b


class object_point_cloud_b(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg('object'))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 10)
        self.visualize = cfg.params.get("visualize", True)
        self.static: bool = cfg.params.get("static", False)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # uncomment to visualize
        if self.visualize:
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            from isaaclab.markers import VisualizationMarkers
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0015
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_b = sample_object_point_cloud(env.num_envs, self.num_points, self.object.cfg.prim_path, device=env.device)
        self.points_w = torch.zeros_like(self.points_b)

    def reset(self, env_ids: slice | torch.Tensor = slice(None)):
        if self.static:
            object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
            object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)
            # apply rotation + translation
            self.points_w = quat_apply(object_quat_w, self.points_b) + object_pos_w
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        static: bool = False,
        flatten: bool = False,
        visualize: bool = True
    ):
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)
        if not static:
            object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, self.num_points, 1)
            object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, self.num_points, 1)
            # apply rotation + translation
            self.points_w = quat_apply(object_quat_w, self.points_b) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    return torch.cat((thumb_contact, index_contact, middle_contact, ring_contact), dim=1)


def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    normalize: bool = True,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    # obtain the input image
    images = sensor.data.output["depth"]
    # depth image normalization
    if normalize:
        images = torch.tanh(images / 2) * 2
        images -= torch.mean(images, dim=(1, 2), keepdim=True)

    return images