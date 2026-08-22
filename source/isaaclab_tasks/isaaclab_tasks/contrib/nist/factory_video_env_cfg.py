# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video playback configuration for a single Factory V1 assembly."""

from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import VideoRecorderCfg
from isaaclab.utils.configclass import configclass

from .factory_env_cfg import FactoryBaseEnvCfg, FactoryEnvCfg


def configure_factory_video(cfg: FactoryEnvCfg, fps: int, output_prefix: str) -> None:
    """Configure the shared wide studio recording setup."""
    cfg.viewer.eye = (7.5, 7.5, 7.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.scene.studio_backdrop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StudioBackdrop",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 6.0, 5.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=1.0, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 1.5)),
    )
    cfg.sim.visualizer_cfgs = [
        NewtonRTXVisualizerCfg(
            headless=True,
            eye=(3.0, 0.0, 0.4),
            lookat=(0.0, 0.0, 0.3),
            focal_length=42.0,
            camera_target_prim_path="/World/envs/env_.*/Robot",
            camera_target_env_index=0,
            visible_env_indices=[0],
            rtx_environment="default",
            rtx_default_light_rotation=(0.0, 65.0, 0.0),
            ground_color=(0.18, 0.18, 0.18),
            ground_roughness=1.0,
            enable_markers=False,
            enable_live_plots=False,
        )
    ]
    cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:newton_rtx", capture_on_render=True, fps=fps, output_filename_prefix=output_prefix
        )
    ]


@configclass
class FactoryVideoEnvCfg(FactoryBaseEnvCfg):
    """Record Factory V1 with the established wide studio camera."""

    def play_mode(self) -> None:
        """Configure evaluation recording."""
        super().play_mode()
        self.events.reset_strategies.params["state_table_size"] = 64
        configure_factory_video(self, fps=30, output_prefix="assembly_40s")
