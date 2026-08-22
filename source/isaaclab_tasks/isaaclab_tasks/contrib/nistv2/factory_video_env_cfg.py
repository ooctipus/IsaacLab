# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video playback configuration for Factory V2."""

from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import VideoRecorderCfg
from isaaclab.utils.configclass import configclass

from .factory_env_cfg import FactoryBaseEnvCfg


@configclass
class FactoryVideoEnvCfg(FactoryBaseEnvCfg):
    """Record Factory V2 with the established wide studio camera."""

    def play_mode(self) -> None:
        """Configure evaluation recording."""
        super().play_mode()
        self.sim.render_interval = 1

        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.scene.studio_backdrop = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/StudioBackdrop",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 6.0, 5.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=1.0, metallic=0.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 1.5)),
        )

        self.sim.visualizer_cfgs = [
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
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_rtx",
                capture_on_render=True,
                fps=100,
                output_filename_prefix="factory_v2",
            ),
        ]
