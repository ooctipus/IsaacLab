# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for motion-imitation environments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionPenaltyScaleCurriculum(ManagerTermBase):
    """Adapt the motion penalty scale from the running average episode length.

    The update reproduces the released G1 motion curriculum. IsaacLab invokes
    curriculum terms before clearing :attr:`episode_length_buf`, so each call
    consumes the completed lengths of exactly the environments being reset.
    Calling :meth:`reset` does not clear the running curriculum state.
    """

    _SAMPLE_BUDGET = 10_000
    _SCALE_DEGREE = 1.0e-5
    _LOWER_THRESHOLD = 40.0
    _UPPER_THRESHOLD = 42.0

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the released penalty curriculum state.

        Args:
            cfg: Curriculum term configuration.
            env: Manager-based motion environment.
        """
        super().__init__(cfg, env)
        self._scale = 0.1
        self._average_episode_length = 0.0

    @property
    def scale(self) -> float:
        """Current multiplier for curriculum-scaled penalty channels."""
        return self._scale

    @property
    def average_episode_length(self) -> float:
        """Exponentially weighted average completed episode length [steps]."""
        return self._average_episode_length

    def evaluation_state_dict(self) -> dict[str, float]:
        """Return the persistent released-curriculum state."""
        return {
            "scale": self._scale,
            "average_episode_length": self._average_episode_length,
        }

    def load_evaluation_state_dict(self, state: Mapping[str, torch.Tensor | float]) -> None:
        """Restore state captured by :meth:`evaluation_state_dict`."""
        if set(state) != {"scale", "average_episode_length"}:
            raise ValueError("Motion penalty evaluation state has unexpected fields.")
        scale = state["scale"]
        average = state["average_episode_length"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("Motion penalty scale evaluation state must be a scalar.")
        if isinstance(average, bool) or not isinstance(average, (int, float)):
            raise TypeError("Motion penalty average evaluation state must be a scalar.")
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("Motion penalty scale evaluation state must lie in [0, 1].")
        if not math.isfinite(average) or average < 0.0:
            raise ValueError("Motion penalty average evaluation state must be finite and nonnegative.")
        self._scale = float(scale)
        self._average_episode_length = float(average)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice | torch.Tensor,
    ) -> dict[str, float]:
        """Update the curriculum from completed episode lengths.

        Args:
            env: Manager-based motion environment.
            env_ids: Environments whose completed episodes are being reset.

        Returns:
            Current scale and running average for logging.
        """
        episode_lengths = env.episode_length_buf[env_ids]
        sample_count = episode_lengths.numel()
        if sample_count > 0:
            batch_average = torch.mean(episode_lengths, dtype=torch.float32).item()
            batch_weight = sample_count / self._SAMPLE_BUDGET
            self._average_episode_length = (
                self._average_episode_length * (1.0 - batch_weight) + batch_average * batch_weight
            )

            if self._average_episode_length < self._LOWER_THRESHOLD:
                self._scale *= 1.0 - self._SCALE_DEGREE
            elif self._average_episode_length > self._UPPER_THRESHOLD:
                self._scale *= 1.0 + self._SCALE_DEGREE
            self._scale = min(max(self._scale, 0.0), 1.0)

        return {
            "penalty_scale": self._scale,
            "average_episode_length": self._average_episode_length,
        }
