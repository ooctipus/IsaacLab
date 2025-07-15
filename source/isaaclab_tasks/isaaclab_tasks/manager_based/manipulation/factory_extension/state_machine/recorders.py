# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets import Articulation
from isaaclab.managers.recorder_manager import RecorderTerm


class EndEffectorStateRecorder(RecorderTerm):
    def record_post_step(self):
        if not hasattr(self, "end_effector_idx"):
            setattr(self, "end_effector_idx", self._env.scene["robot"].data.body_names.index("robotiq_base_link"))
        robot: Articulation = self._env.scene["robot"]
        state = robot.data.body_link_state_w[:, self.end_effector_idx]
        state[:, :3] -= self._env.scene.env_origins
        return "end_effector", state
