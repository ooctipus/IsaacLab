# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for native MuJoCo control."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .native_mujoco_action import NativeMujocoControlAction


@configclass
class NativeMujocoControlActionCfg(ActionTermCfg):
    """Configuration for the native fixed-width MuJoCo control action."""

    class_type: type[NativeMujocoControlAction] | str = "{DIR}.native_mujoco_action:NativeMujocoControlAction"

    action_width: int = MISSING
    """Number of dimensionless native control inputs."""

    def __post_init__(self) -> None:
        """Reject clipping that the identity native-control path does not implement."""
        if self.clip is not None:
            raise ValueError("NativeMujocoControlActionCfg does not use ActionTermCfg.clip.")
