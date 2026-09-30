# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera (vision) variants of the KukaAllegro Lift and Reorient tasks.

Each task is exposed as a :class:`~isaaclab_tasks.utils.PresetCfg` whose ``single_camera`` /
``duo_camera`` variants add base (and wrist) cameras plus the matching image observations on top of
the state env config in :mod:`.kuka_allegro_env_cfg`. The camera data type / resolution and
renderer backend remain ``presets=`` selectable through the camera configs.
"""

from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    WristTiledCameraCfg,
)
from .kuka_allegro_env_cfg import (
    KukaAllegroLiftEnvCfg,
    KukaAllegroReorientEnvCfg,
    KukaAllegroSceneCfg,
)


@configclass
class SingleCameraSceneCfg(KukaAllegroSceneCfg):
    """KukaAllegro scene with a single base-mounted camera."""

    camera: CameraCfg = BaseTiledCameraCfg()


@configclass
class DuoCameraSceneCfg(SingleCameraSceneCfg):
    """KukaAllegro scene with base-mounted and wrist-mounted cameras."""

    wrist_camera: CameraCfg = WristTiledCameraCfg()


@configclass
class KukaAllegroReorientCameraEnvCfg(PresetCfg):
    single_camera = KukaAllegroReorientEnvCfg(scene=SingleCameraSceneCfg(), observations=SingleCameraObservationsCfg())
    duo_camera = KukaAllegroReorientEnvCfg(scene=DuoCameraSceneCfg(), observations=DuoCameraObservationsCfg())
    default = single_camera


@configclass
class KukaAllegroLiftCameraEnvCfg(PresetCfg):
    single_camera = KukaAllegroLiftEnvCfg(scene=SingleCameraSceneCfg(), observations=SingleCameraObservationsCfg())
    duo_camera = KukaAllegroLiftEnvCfg(scene=DuoCameraSceneCfg(), observations=DuoCameraObservationsCfg())
    default = single_camera
