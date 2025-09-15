# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sim import get_first_matching_child_prim
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms, combine_frame_transforms, compute_pose_error

from .utils import sample_object_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera, MultiMeshRayCasterCamera, MultiMeshRayCaster


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
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

        object_cfgs: list[SceneEntityCfg] = cfg.params.get("object_cfgs", [SceneEntityCfg("object")])
        self.num_points: list[int] = cfg.params.get("num_points", [10])
        self.visualize = cfg.params.get("visualize", True)
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.statics: list[bool] = cfg.params.get("statics", [False])
        self.objects: list[RigidObject] = [env.scene[object_cfg.name] for object_cfg in object_cfgs]
        self.body_ids = []
        self.body_names: list = [
            object.data.body_names if isinstance(cfg.body_ids, slice) else cfg.body_names
            for object, cfg in zip(self.objects, object_cfgs)
        ]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]

        if self.visualize:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

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
                    raise (f"Found no Collider to sample point cloud in {object.cfg.prim_path}")
            else:
                body_ids = []
                for bn in body_name:
                    prim = get_first_matching_child_prim(
                        object.cfg.prim_path.replace(".*", "0", 1),
                        predicate=lambda p: p.GetName() == bn,
                        traverse_instance_prims=True,
                    )
                    expression = str(prim.GetPrimPath()).replace("/env_0/", "/env_.*/", 1)
                    local_pts = sample_object_point_cloud(env.num_envs, n_point, expression, device=env.device)
                    if local_pts is not None:
                        points.append(local_pts)
                        body_ids.append(object.body_names.index(bn))
                self.body_ids.append(body_ids)
        self.points_b = torch.cat(points, dim=1)
        self.points_w = torch.zeros_like(self.points_b)

    def reset(self, env_ids=slice(None)):
        idx = 0
        for i, object in enumerate(self.objects):
            if self.statics[i]:
                object_pos_w = object.data.root_pos_w[env_ids].unsqueeze(1).repeat(1, self.num_points[i], 1)
                object_quat_w = object.data.root_quat_w[env_ids].unsqueeze(1).repeat(1, self.num_points[i], 1)
                self.points_w[env_ids, idx : idx + self.num_points[i]] = (
                    quat_apply(object_quat_w, self.points_b[env_ids, idx : idx + self.num_points[i]]) + object_pos_w
                )
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
        object_point_cloud_pos_b.clamp_(-5.0, 5.0)
        if normalize:
            object_point_cloud_pos_b = object_point_cloud_pos_b - object_point_cloud_pos_b.mean(dim=1, keepdim=True)
            d = torch.norm(object_point_cloud_pos_b, dim=-1)
            m = d.max(dim=1, keepdim=True)[0]
            object_point_cloud_pos_b = object_point_cloud_pos_b / (m.unsqueeze(-1) + 1e-6)
        return object_point_cloud_pos_b.view(self.num_envs, -1) if flatten else object_point_cloud_pos_b


def task_pose_error(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The pose error between the object and the target object in the world frame."""
    asset: RigidObject = env.scene[object_cfg.name]
    command_pose_b = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command_pose_b[:, :3], command_pose_b[:, 3:7]
    )
    pos_error, quat_error = compute_pose_error(des_pos_w, des_quat_w, asset.data.root_pos_w, asset.data.root_quat_w)
    return torch.cat((pos_error, quat_error), dim=1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b.view(env.num_envs, -1)


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


def depth_image_ray_caster_camera(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    normalize: bool = True,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera | MultiMeshRayCasterCamera = env.scene.sensors[sensor_cfg.name]
    # obtain the input image
    images = sensor.data.output["distance_to_image_plane"]
    images = torch.nan_to_num(images, nan=10.0)
    # depth image normalization
    if normalize:
        images = torch.tanh(images / 2) * 2
        images -= torch.mean(images, dim=(1, 2), keepdim=True)

    return images


def depth_image_ray_caster(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    normalize: bool = True,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]
    # obtain the input image
    point_clouds_w = sensor.data.ray_hits_w
    width, height = sensor.cfg.pattern_cfg.size
    resolution = sensor.cfg.pattern_cfg.resolution
    # torch.norm(point_clouds_w - sensor.data.pos_w, dim=1)
    return point_clouds_w.view(env.num_envs, int(width / resolution) + 1, int(height / resolution) + 1, 3)[..., 0].unsqueeze(-1)
