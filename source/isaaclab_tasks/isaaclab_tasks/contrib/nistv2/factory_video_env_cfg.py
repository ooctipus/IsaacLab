# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video playback composition for homogeneous Factory."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.factory_video_env_cfg import configure_factory_video

from .factory_env_cfg import FactoryBaseEnvCfg


@configclass
class FactoryVideoEnvCfg(FactoryBaseEnvCfg):
    """Record homogeneous Factory with the shared studio camera."""

    def play_mode(self) -> None:
        super().play_mode()
        self.sim.render_interval = 1
        configure_factory_video(self, fps=100, output_prefix="factory_v2")
