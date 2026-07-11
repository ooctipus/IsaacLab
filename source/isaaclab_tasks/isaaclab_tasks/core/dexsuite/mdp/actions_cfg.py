# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for temporal-response randomized action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import TemporalDRRelativeJointPositionAction


@configclass
class TemporalDRRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`TemporalDRRelativeJointPositionAction`."""

    class_type: type[TemporalDRRelativeJointPositionAction] | str = (
        "{DIR}.actions:TemporalDRRelativeJointPositionAction"
    )

    delay_steps: tuple[int, int] = (0, 2)
    """Per-episode action latency range [control steps], inclusive."""

    ema_beta_range: tuple[float, float] = (0.5, 1.0)
    """Per-episode first-order low-pass coefficient range (1.0 = no filtering)."""
