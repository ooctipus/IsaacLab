# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-bank metrics for the configured board assemblies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .reset import board_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tile_labels(labels: tuple[str, ...]) -> tuple[tuple[int, int], tuple[tuple[str, ...], ...]]:
    columns = min(5, len(labels))
    rows = (len(labels) + columns - 1) // columns
    padded = labels + ("",) * (rows * columns - len(labels))
    return (rows, columns), tuple(padded[start : start + columns] for start in range(0, len(padded), columns))


class BoardMetrics(ManagerTermBase):
    """Publish reset-bank metrics for the selected assemblies."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset = env.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("BoardMetrics requires the resolved board reset term.")
        self._reset = reset
        self._num_slots = reset.layout.num_slots
        self._num_variants = reset.layout.num_variants
        self._unfinished_tile_shape, self._unfinished_labels = _tile_labels(
            tuple(str(count) for count in range(1, self._num_slots + 1))
        )
        self._asset_tile_shape, self._asset_labels = _tile_labels(reset.layout.asset_labels)
        self._unfinished_facet_labels = tuple(f"U={count}" for count in range(1, self._num_slots + 1))
        self._facet_columns = min(5, self._num_slots)
        facet_rows = (self._num_slots + self._facet_columns - 1) // self._facet_columns
        self._figure_size = (3.2 * self._facet_columns, 2.5 * facet_rows)
        self._publish(env)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        del env, env_ids

    def _publish(self, env: ManagerBasedRLEnv) -> None:
        env.extras["heatmap"] = {
            "Metrics/ResetProbs": {
                "numerator": self._reset.reset_probability_mass,
                "denominator": self._reset.reset_probability_total,
                "tile_shape": self._unfinished_tile_shape,
                "cell_labels": self._unfinished_labels,
                "color_label": "Reset probability",
                "cmap": "magma",
                "colorbar_percent": True,
            },
            "Metrics/ResetSuccessRate": {
                "numerator": self._reset.reset_success_sum,
                "denominator": self._reset.reset_state_count,
                "tile_shape": self._unfinished_tile_shape,
                "cell_labels": self._unfinished_labels,
                "color_label": "Estimated task success rate",
                "cmap": "RdYlGn",
                "vmax": 1.0,
            },
            "Metrics/AssetUnassembledRate": {
                "numerator": self._reset.asset_unassembled_sum,
                "denominator": self._reset.asset_unfinished_count,
                "tile_shape": self._asset_tile_shape,
                "cell_labels": self._asset_labels,
                "facet_labels": self._unfinished_facet_labels,
                "facet_columns": self._facet_columns,
                "figure_size": self._figure_size,
                "color_label": "Estimated failure when asset starts unfinished",
                "cmap": "RdYlGn_r",
                "vmax": 1.0,
            },
        }
