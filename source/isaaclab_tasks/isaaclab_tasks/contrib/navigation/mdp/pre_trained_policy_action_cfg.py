# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg, ObservationGroupCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.configclass import configclass

_GOAL_MARKER_CFG = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Actions/velocity_goal")
_GOAL_MARKER_CFG.markers["arrow"].scale = (0.5, 0.5, 0.5)
_CURRENT_MARKER_CFG = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Actions/velocity_current")
_CURRENT_MARKER_CFG.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type | str = "{DIR}.pre_trained_policy_action:PreTrainedPolicyAction"
    """Class of the action term."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""

    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""

    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""

    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""

    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""

    goal_visualizer_cfg: VisualizationMarkersCfg = _GOAL_MARKER_CFG
    """Marker configuration for the commanded velocity."""

    current_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_MARKER_CFG
    """Marker configuration for the measured velocity."""
