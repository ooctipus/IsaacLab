# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import dextrah_env_cfg as dextrah_state_impl
from ... import mdp
from .dextrah_kuka_allegro_env_cfg import KukaAllegroMixinCfg


@configclass
class KukaAllegroDepthCameraSceneCfg(dextrah_state_impl.SceneCfg):
    """Dextrah scene for multi-objects Lifting/Reorientation"""

    base_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.57, -0.8, 0.5),
            rot=(0.61237, 0.61237, 0.35355, 0.35355),  # (x: 90 degree, y: 60 degree, z: 0 degree)
            convention="opengl",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=64,
        height=64,
    )

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/palm_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.038, -0.38, -0.18),
            rot=(0.29884, 0.64086, 0.64086, -0.29884),  # (x: 130 degree, y: 0 degree, z: -90 degree)
            convention="opengl",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=64,
        height=64,
    )

    wall = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 3.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.2, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

@configclass
class KukaAllegroRGBCameraSceneCfg(KukaAllegroDepthCameraSceneCfg):
    """Dextrah scene for multi-objects Lifting/Reorientation"""

    def __post_init__(self):
        self.base_camera.data_types=["rgb"]
        self.wrist_camera.data_types=["rgb"]


@configclass
class KukaAllegroDepthCameraObservationsCfg(dextrah_state_impl.ObservationsCfg):
    """Observation specifications for the MDP."""
    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for policy group."""
        object_observation_b = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0., n_max=0.),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")})
    
    @configclass
    class WristImageObsCfg(ObsGroup):
        wrist_observation = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0., n_max=0.),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")})


    base_image: BaseImageObsCfg = BaseImageObsCfg()
    wrist_image: WristImageObsCfg = WristImageObsCfg()

    def __post_init__(self):
        # Hack: my encoder doesn't support multiple different observation source
        self.privileged.perception.params["flatten"] = True


@configclass
class KukaAllegroRGBPreTrainedResNet18ObservationsCfg(dextrah_state_impl.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class RGBImageObsCfg(dextrah_state_impl.ObservationsCfg.PolicyCfg):
        """Camera observations for policy group."""

        base_observation_b = ObsTerm(
            func=mdp.image_features,
            params={"model_device": "cuda", "sensor_cfg": SceneEntityCfg("base_camera")},
        )
        wrist_observation_b = ObsTerm(
            func=mdp.image_features,
            clip=(-1.0, 1.0),
            params={"model_device": "cuda", "sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = None

    policy: RGBImageObsCfg = RGBImageObsCfg()


@configclass
class KukaAllegroDepthCameraMixinCfg(KukaAllegroMixinCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.75, -1.75, 0.75), lookat=(-0.5, 0.0, 0.50), origin_type="env")
    scene: KukaAllegroDepthCameraSceneCfg = KukaAllegroDepthCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroDepthCameraObservationsCfg = KukaAllegroDepthCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class KukaAllegroRGBCameraMixinCfg(KukaAllegroMixinCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.75, -1.75, 0.75), lookat=(-0.5, 0.0, 0.50), origin_type="env")
    scene: KukaAllegroRGBCameraSceneCfg = KukaAllegroRGBCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroRGBPreTrainedResNet18ObservationsCfg = KukaAllegroRGBPreTrainedResNet18ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class DextrahKukaAllegroReorientDepthCameraEnvCfg(KukaAllegroDepthCameraMixinCfg, dextrah_state_impl.DextrahReorientEnvCfg):
    pass


@configclass
class DextrahKukaAllegroReorientDepthCameraEnvCfg_PLAY(
    KukaAllegroDepthCameraMixinCfg, dextrah_state_impl.DextrahReorientEnvCfg_PLAY
):
    pass


@configclass
class DextrahKukaAllegroLiftDepthCameraEnvCfg(KukaAllegroDepthCameraMixinCfg, dextrah_state_impl.DextrahLiftEnvCfg):
    pass


@configclass
class DextrahKukaAllegroLiftDepthCameraEnvCfg_PLAY(KukaAllegroDepthCameraMixinCfg, dextrah_state_impl.DextrahLiftEnvCfg_PLAY):
    pass

@configclass
class DextrahKukaAllegroReorientRGBCameraEnvCfg(KukaAllegroRGBCameraMixinCfg, dextrah_state_impl.DextrahReorientEnvCfg):
    pass


@configclass
class DextrahKukaAllegroReorientRGBCameraEnvCfg_PLAY(
    KukaAllegroRGBCameraMixinCfg, dextrah_state_impl.DextrahReorientEnvCfg_PLAY
):
    pass


@configclass
class DextrahKukaAllegroLiftRGBCameraEnvCfg(KukaAllegroRGBCameraMixinCfg, dextrah_state_impl.DextrahLiftEnvCfg):
    pass


@configclass
class DextrahKukaAllegroLiftRGBCameraEnvCfg_PLAY(KukaAllegroRGBCameraMixinCfg, dextrah_state_impl.DextrahLiftEnvCfg_PLAY):
    pass