# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from dataclasses import MISSING
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns, MultiMeshRayCasterCameraCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import dexsuite_env_cfg as dexsuite_state_impl
from ... import mdp
from . import dexsuite_kuka_allegro_env_cfg as kuka_allegro_dexsuite


@configclass
class KukaAllegroSingleRayCasterCameraSceneCfg(kuka_allegro_dexsuite.KukaAllegroSceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""
    width: int = 64
    height: int = 64

    base_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint",
        mesh_prim_paths=[
            "/World/GroundPlane",
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=False, target_prim_expr="{ENV_REGEX_NS}/Object"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/Robot/ee_link/.*_link.*"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/Robot/.*_link_.*"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/table", track_mesh_transforms=False),
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.57, -0.8, 0.5), rot=(0.6124, 0.6124, 0.3536, 0.3536), convention="opengl",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(height=64, width=64),
        max_distance=10,
        depth_clipping_behavior="max"
    )

    def __post_init__(self):
        super().__post_init__()
        self.base_camera.pattern_cfg.width = self.width
        self.base_camera.pattern_cfg.height = self.height
        del self.width
        del self.height


@configclass
class KukaAllegroDuoRayCasterCameraSceneCfg(KukaAllegroSingleRayCasterCameraSceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    wrist_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/palm_link",
        mesh_prim_paths=[
            "/World/GroundPlane",
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=False, target_prim_expr="{ENV_REGEX_NS}/Object"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/Robot/ee_link/.*_link.*"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/table"),
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.038, -0.38, -0.18), rot=(0.299, 0.641, 0.641, -0.299), convention="opengl",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(height=64, width=64),
        max_distance=10,
        depth_clipping_behavior="max"
    )

    def __post_init__(self):
        super().__post_init__()
        self.wrist_camera.data_types = self.base_camera.data_types
        self.wrist_camera.pattern_cfg.width = self.base_camera.pattern_cfg.width
        self.wrist_camera.pattern_cfg.height = self.base_camera.pattern_cfg.height


@configclass
class KukaAllegroSingleTiledCameraSceneCfg(kuka_allegro_dexsuite.KukaAllegroSceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    camera_type: str = "distance_to_image_plane"
    width: int = 64
    height: int = 64

    base_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.57, -0.8, 0.5), rot=(0.6124, 0.6124, 0.3536, 0.3536), convention="opengl",
        ),
        data_types=MISSING,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
        width=MISSING,
        height=MISSING,
    )

    def __post_init__(self):
        super().__post_init__()
        self.base_camera.data_types = [self.camera_type]
        self.base_camera.width = self.width
        self.base_camera.height = self.height
        del self.camera_type
        del self.width
        del self.height


@configclass
class KukaAllegroDuoTiledCameraSceneCfg(KukaAllegroSingleTiledCameraSceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/ee_link/palm_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.038, -0.38, -0.18), rot=(0.299, 0.641, 0.641, -0.299), convention="opengl",
        ),
        data_types=MISSING,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
        width=MISSING,
        height=MISSING,
    )

    def __post_init__(self):
        super().__post_init__()
        self.wrist_camera.data_types = self.base_camera.data_types
        self.wrist_camera.width = self.base_camera.width
        self.wrist_camera.height = self.base_camera.height


@configclass
class KukaAllegroSingleCameraObservationsCfg(kuka_allegro_dexsuite.KukaAllegroObservationCfg):
    """Observation specifications for the MDP."""

    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for policy group."""

        object_observation_b = ObsTerm(
            func=mdp.vision_camera,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )

    base_image: BaseImageObsCfg = BaseImageObsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in self.__dataclass_fields__.values():
            obs_group = getattr(self, group.name)
            obs_group.history_length = None



@configclass
class KukaAllegroDuoCameraObservationsCfg(KukaAllegroSingleCameraObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class WristImageObsCfg(ObsGroup):
        wrist_observation = ObsTerm(
            func=mdp.vision_camera,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

    wrist_image: WristImageObsCfg = WristImageObsCfg()


@configclass
class KukaAllegroSingleCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroSingleTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroSingleCameraObservationsCfg = KukaAllegroSingleCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        sa = {"num_envs": 4096, "env_spacing": 3, "replicate_physics": False}
        self.variants.setdefault("scene", {}).update({
            "64x64tiled_depth": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "distance_to_image_plane", "width": 64, "height": 64}),
            "64x64tiled_rgb": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "rgb", "width": 64, "height": 64}),
            "64x64tiled_albedo": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "diffuse_albedo", "width": 64, "height": 64}),
            "64x64raycaster": KukaAllegroSingleRayCasterCameraSceneCfg(**{**sa, "width": 64, "height": 64}),
            "128x128tiled_depth": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "distance_to_image_plane", "width": 128, "height": 128}),
            "128x128tiled_rgb": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "rgb", "width": 128, "height": 128}),
            "128x128tiled_albedo": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "diffuse_albedo", "width": 128, "height": 128}),
            "128x128raycaster": KukaAllegroSingleRayCasterCameraSceneCfg(**{**sa, "width": 128, "height": 128}),
            "256x256tiled_depth": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "distance_to_image_plane", "width": 256, "height": 256}),
            "256x256tiled_rgb": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "rgb", "width": 256, "height": 256}),
            "256x256tiled_albedo": KukaAllegroSingleTiledCameraSceneCfg(**{**sa, "camera_type": "diffuse_albedo", "width": 256, "height": 256}),
            "256x256raycaster": KukaAllegroSingleRayCasterCameraSceneCfg(**{**sa, "width": 256, "height": 256}),
        })

@configclass
class KukaAllegroDuoCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroDuoTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroDuoCameraObservationsCfg = KukaAllegroDuoCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        sa = {"num_envs": 4096, "env_spacing": 3, "replicate_physics": False}
        self.variants.setdefault("scene", {}).update({
            "64x64tiled_depth": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "distance_to_image_plane", "width": 64, "height": 64}),
            "64x64tiled_rgb": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "rgb", "width": 64, "height": 64}),
            "64x64tiled_albedo": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "diffuse_albedo", "width": 64, "height": 64}),
            "64x64raycaster": KukaAllegroDuoRayCasterCameraSceneCfg(**{**sa, "width": 64, "height": 64}),
            "128x128tiled_depth": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "distance_to_image_plane", "width": 128, "height": 128}),
            "128x128tiled_rgb": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "rgb", "width": 128, "height": 128}),
            "128x128tiled_albedo": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "diffuse_albedo", "width": 128, "height": 128}),
            "128x128raycaster": KukaAllegroDuoRayCasterCameraSceneCfg(**{**sa, "width": 128, "height": 128}),
            "256x256tiled_depth": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "distance_to_image_plane", "width": 256, "height": 256}),
            "256x256tiled_rgb": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "rgb", "width": 256, "height": 256}),
            "256x256tiled_albedo": KukaAllegroDuoTiledCameraSceneCfg(**{**sa, "camera_type": "diffuse_albedo", "width": 256, "height": 256}),
            "256x256raycaster": KukaAllegroDuoRayCasterCameraSceneCfg(**{**sa, "width": 256, "height": 256}),
        })

# SingleCamera
@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg(
    KukaAllegroSingleCameraMixinCfg,
    dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    pass


@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg_PLAY(
    KukaAllegroSingleCameraMixinCfg,
    dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY
):
    pass


# DuoCamera
@configclass
class DexsuiteKukaAllegroLiftDuoCameraEnvCfg(
    KukaAllegroDuoCameraMixinCfg,
    dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    pass


@configclass
class DexsuiteKukaAllegroLiftDuoCameraEnvCfg_PLAY(
    KukaAllegroDuoCameraMixinCfg,
    dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY
):
    pass
