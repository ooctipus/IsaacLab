# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-launch-safe configuration for motion action terms."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import MotionJointPositionAction, MotionMujocoControlAction


@configclass
class MotionJointPositionActionCfg(ActionTermCfg):
    """Configuration for the articulation-derived G1 position-control action."""

    class_type: type[MotionJointPositionAction] | str = "{DIR}.actions:MotionJointPositionAction"

    joint_names: list[str] = MISSING
    """Joint names or regular expressions controlled by this action."""

    preserve_order: bool = False
    """Whether the resolved joints retain the order of :attr:`joint_names`."""

    normalize_to: float = 5.0
    """Multiplier from behavior action to controller-normalized action."""

    action_clip: float = 5.0
    """Symmetric bound on the controller-normalized action."""

    action_scale: float = 0.25
    """Scale applied to the effort-over-stiffness target displacement."""

    default_joint_offset_range: tuple[float, float] = (0.0, 0.0)
    """Uniform episodic default-pose offset range [rad]."""

    def __post_init__(self) -> None:
        """Reject the inherited action clip because this term owns its real clamp."""
        if self.clip is not None:
            raise ValueError("MotionJointPositionActionCfg does not use ActionTermCfg.clip; use action_clip.")


@configclass
class MotionMujocoControlActionCfg(ActionTermCfg):
    """Configuration for the native fixed-width MuJoCo control action."""

    class_type: type[MotionMujocoControlAction] | str = "{DIR}.actions:MotionMujocoControlAction"

    action_width: int = MISSING
    """Number of dimensionless native control inputs."""

    def __post_init__(self) -> None:
        """Reject clipping that the identity native-control path does not implement."""
        if self.clip is not None:
            raise ValueError("MotionMujocoControlActionCfg does not use ActionTermCfg.clip.")


__all__ = ["MotionJointPositionActionCfg", "MotionMujocoControlActionCfg"]
