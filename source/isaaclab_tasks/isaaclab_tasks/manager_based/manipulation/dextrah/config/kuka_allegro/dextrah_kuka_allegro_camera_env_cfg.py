# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import dextrah_env_cfg as dextrah_state_impl
from ... import mdp
from .dextrah_kuka_allegro_env_cfg import KukaAllegroMixinCfg


@configclass
class KukaAllegroCameraSceneCfg(dextrah_state_impl.SceneCfg):
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
class KukaAllegroCameraObservationsCfg(dextrah_state_impl.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class DepthImageObsCfg(dextrah_state_impl.ObservationsCfg.PolicyCfg):
        """Camera observations for policy group."""

        object_observation_b = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )
        wrist_observation_b = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = None

    policy: DepthImageObsCfg = DepthImageObsCfg()


@configclass
class KukaAllegroCameraMixinCfg(KukaAllegroMixinCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.75, -1.75, 0.75), lookat=(-0.5, 0.0, 0.50), origin_type="env")
    scene: KukaAllegroCameraSceneCfg = KukaAllegroCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroCameraObservationsCfg = KukaAllegroCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class DextrahKukaAllegroReorientCameraEnvCfg(KukaAllegroCameraMixinCfg, dextrah_state_impl.DexSuiteReorientEnvCfg):
    pass


@configclass
class DextrahKukaAllegroReorientCameraEnvCfg_PLAY(
    KukaAllegroCameraMixinCfg, dextrah_state_impl.DexSuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DextrahKukaAllegroLiftCameraEnvCfg(KukaAllegroCameraMixinCfg, dextrah_state_impl.DexSuiteLiftEnvCfg):
    pass


@configclass
class DextrahKukaAllegroLiftCameraEnvCfg_PLAY(KukaAllegroCameraMixinCfg, dextrah_state_impl.DexSuiteLiftEnvCfg_PLAY):
    pass
