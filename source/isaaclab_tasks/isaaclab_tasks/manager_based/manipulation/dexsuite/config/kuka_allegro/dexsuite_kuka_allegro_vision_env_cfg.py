# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import dexsuite_env_cfg as dexsuite_state_impl
from ... import mdp
from .dexsuite_kuka_allegro_env_cfg import KukaAllegroMixinCfg


@configclass
class KukaAllegroCameraSceneCfg(dexsuite_state_impl.SceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

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

    # wrist_camera = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/palm_link/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.038, -0.38, -0.18),
    #         rot=(0.29884, 0.64086, 0.64086, -0.29884),  # (x: 130 degree, y: 0 degree, z: -90 degree)
    #         convention="opengl",
    #     ),
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=64,
    #     height=64,
    # )

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
class KukaAllegroCameraObservationsCfg(dexsuite_state_impl.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for policy group."""

        object_observation_b = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )

    # @configclass
    # class WristImageObsCfg(ObsGroup):
    #     wrist_observation = ObsTerm(
    #         func=mdp.depth_image,
    #         noise=Unoise(n_min=-0.0, n_max=0.0),
    #         clip=(-1.0, 1.0),
    #         params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
    #     )

    base_image: BaseImageObsCfg = BaseImageObsCfg()
    # wrist_image: WristImageObsCfg = WristImageObsCfg()


@configclass
class KukaAllegroDepthCameraMixinCfg(KukaAllegroMixinCfg):
    scene: KukaAllegroCameraSceneCfg = KukaAllegroCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroCameraObservationsCfg = KukaAllegroCameraObservationsCfg()


@configclass
class DexsuiteKukaAllegroReorientDepthCameraEnvCfg(KukaAllegroDepthCameraMixinCfg, dexsuite_state_impl.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientDepthCameraEnvCfg_PLAY(
    KukaAllegroDepthCameraMixinCfg, dexsuite_state_impl.DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteKukaAllegroLiftDepthCameraEnvCfg(KukaAllegroDepthCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftDepthCameraEnvCfg_PLAY(KukaAllegroDepthCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY):
    pass
