# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video playback configurations for Factory environments."""

import math

from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import VideoRecorderCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg, WindowCfg

from .factory_env_cfg import FactoryBaseEnvCfg, FactoryEnvCfg
from .factory_variant_env_cfg import FactoryVariantEnvCfg


def set_factory_video_view(env, env_ids, asset_cfg: SceneEntityCfg) -> None:
    """Resolve the robot once; camera offsets are owned by the visualizer declaration."""
    origin = env.scene[asset_cfg.name].data.root_pos_w.torch[0]
    visualizer = env.sim.visualizers[0]
    camera = visualizer.cfg.cameras[0]
    eye = tuple((origin + origin.new_tensor(camera.eye)).tolist())
    target = tuple((origin + origin.new_tensor(camera.lookat)).tolist())
    visualizer.set_camera_view(eye, target)


def configure_factory_video(
    cfg: FactoryEnvCfg, fps: int, output_prefix: str, backend: NewtonGLBackendCfg | NewtonRTXBackendCfg
) -> None:
    """Configure the shared wide studio recording setup."""
    cfg.scene.studio_backdrop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StudioBackdrop",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 6.0, 5.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=1.0, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 1.5)),
    )
    if isinstance(backend, NewtonRTXBackendCfg):
        half_pitch = 0.5 * math.atan2(0.1, 3.0)
        cfg.scene.video_camera = CameraCfg(
            prim_path=cfg.scene.robot.prim_path + "/FactoryVideoCamera",
            width=1280,
            height=720,
            data_types=["rgb"],
            renderer_cfg=backend.renderer_cfg,
            spawn=sim_utils.PinholeCameraCfg(focal_length=42.0),
            offset=CameraCfg.OffsetCfg(
                pos=(3.0, 0.0, 0.4), rot=(-math.sin(half_pitch), 0.0, math.cos(half_pitch), 0.0), convention="world"
            ),
        )
        cameras = [SceneCameraCfg(prim_path=cfg.scene.video_camera.prim_path)]
    else:
        cameras = [PerspectiveCameraCfg(eye=(3.0, 0.0, 0.4), lookat=(0.0, 0.0, 0.3), focal_length=42.0)]
        cfg.events.video_camera = EventTermCfg(
            func=set_factory_video_view, mode="startup", params={"asset_cfg": SceneEntityCfg("robot")}
        )
    cfg.sim.visualizer_cfgs = [
        VisualizerCfg(
            backend=backend,
            headless=True,
            cameras=cameras,
            window=WindowCfg(size=(1280, 720)),
            visible_env_indices=[0],
            enable_markers=False,
            enable_live_plots=False,
        )
    ]
    cfg.video_recorders = [
        VideoRecorderCfg(
            source=f"visualizer:{backend.visualizer_type}",
            capture_on_render=True,
            fps=fps,
            output_filename_prefix=output_prefix,
        )
    ]


@configclass
class FactoryVideoEnvCfg(FactoryBaseEnvCfg):
    """Record a static Factory assembly with the established wide studio camera."""

    def play_mode(self) -> None:
        """Configure evaluation recording."""
        super().play_mode()
        self.events.reset_strategies.params["state_table_size"] = 64
        configure_factory_video(self, fps=30, output_prefix="assembly_40s", backend=NewtonRTXBackendCfg())


@configclass
class FactoryVariantVideoEnvCfg(FactoryVariantEnvCfg):
    """Record Factory variants with the established wide studio camera."""

    def play_mode(self) -> None:
        """Configure evaluation recording."""
        super().play_mode()
        self.sim.render_interval = 1
        configure_factory_video(self, fps=100, output_prefix="factory_variant", backend=NewtonGLBackendCfg())
