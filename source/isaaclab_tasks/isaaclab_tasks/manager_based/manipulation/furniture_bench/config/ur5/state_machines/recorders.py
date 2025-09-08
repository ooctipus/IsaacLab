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
        state = robot.data.body_link_state_w[:, self.end_effector_idx].clone()
        state[:, :3] -= self._env.scene.env_origins
        return "end_effector", state


class StableStateRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))
