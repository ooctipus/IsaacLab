# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the G1 joint-position action term."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import G1JointPositionAction


@configclass
class G1JointPositionActionCfg(JointActionCfg):
    """Configuration for the articulation-derived G1 position-control action."""

    class_type: type[G1JointPositionAction] | str = "{DIR}.actions:G1JointPositionAction"

    scale: float | dict[str, float] = 5.0
    """Scale from behavior action to controller-normalized action."""

    offset: float | dict[str, float] = 0.0
    """Offset added to the controller-normalized action."""

    clip: dict[str, tuple[float, float]] | None = {".*": (-5.0, 5.0)}
    """Controller-normalized action limits."""

    action_scale: float = 0.25
    """Scale applied to the effort-over-stiffness target displacement."""

    default_joint_offset_range: tuple[float, float] = (0.0, 0.0)
    """Uniform episodic default-pose offset range [rad]."""
