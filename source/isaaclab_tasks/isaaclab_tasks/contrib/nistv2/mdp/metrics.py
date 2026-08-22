# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode metrics for full-board assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from ..board_layout import NUM_ASSEMBLIES
from .assembly_state import _assembly_state
from .reset import board_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_TILE_SHAPE = (4, 5)
_UNFINISHED_LABELS = tuple(tuple(str(5 * row + column + 1) for column in range(5)) for row in range(4))
_ASSET_LABELS = (
    ("N4", "N8", "N12", "N16", "Gs"),
    ("Gm", "Gl", "R4", "R8", "R12"),
    ("R16", "P4", "P8", "P12", "P16"),
    ("USB", "WP", "BNC", "DS", "RJ"),
)


class BoardMetrics(ManagerTermBase):
    """Accumulate full-board reset and terminal outcome metrics."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset = env.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("BoardMetrics requires the resolved board reset term.")
        self._reset = reset
        self._state = _assembly_state(env)
        self._episode_counts = torch.zeros(NUM_ASSEMBLIES, dtype=torch.long, device=env.device)
        self._success_counts = torch.zeros_like(self._episode_counts)
        self._asset_start_counts = torch.zeros_like(self._episode_counts)
        self._asset_unassembled_counts = torch.zeros_like(self._episode_counts)
        self._publish(env)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        ended = env_ids[(env.episode_length_buf[env_ids] > 0) & env.reset_buf[env_ids]]
        if ended.numel() == 0:
            return

        unfinished = self._reset.unfinished_count[ended].long() - 1
        success = self._state.all_success[ended]
        self._episode_counts.add_(torch.bincount(unfinished, minlength=NUM_ASSEMBLIES))
        self._success_counts.add_(torch.bincount(unfinished[success], minlength=NUM_ASSEMBLIES))

        started_unfinished = self._reset.initial_unfinished[ended]
        self._asset_start_counts.add_(started_unfinished.sum(dim=0))
        self._asset_unassembled_counts.add_((started_unfinished & ~self._state.asset_assembled[ended]).sum(dim=0))

    def _publish(self, env: ManagerBasedRLEnv) -> None:
        env.extras["heatmap"] = {
            "Metrics/ResetProbs": {
                "numerator": self._reset.sample_counts.view(_TILE_SHAPE),
                "denominator": self._reset.sample_total,
                "cell_labels": _UNFINISHED_LABELS,
                "color_label": "Share of reset samples",
                "cmap": "magma",
                "colorbar_percent": True,
            },
            "Metrics/ResetSuccessRate": {
                "numerator": self._success_counts.view(_TILE_SHAPE),
                "denominator": self._episode_counts.view(_TILE_SHAPE),
                "cell_labels": _UNFINISHED_LABELS,
                "color_label": "Whole-board success rate",
                "cmap": "RdYlGn",
                "vmax": 1.0,
            },
            "Metrics/AssetUnassembledRate": {
                "numerator": self._asset_unassembled_counts.view(_TILE_SHAPE),
                "denominator": self._asset_start_counts.view(_TILE_SHAPE),
                "cell_labels": _ASSET_LABELS,
                "color_label": "Still unassembled at episode end",
                "cmap": "RdYlGn_r",
                "vmax": 1.0,
            },
        }
