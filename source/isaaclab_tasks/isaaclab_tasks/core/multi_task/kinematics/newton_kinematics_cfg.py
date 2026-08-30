# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-dataclass configuration for :class:`NewtonKinematics`.

Separated from :mod:`newton_kinematics` so cfg files can be constructed
without triggering ``import newton`` (which transitively pulls ``pxr``).
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class NewtonKinematicsCfg:
    """Configuration for building a :class:`NewtonKinematics` model.

    Mirrors the constructor arguments so the model can be instantiated
    with ``NewtonKinematics(cfg)``.
    """

    usd_path: str = MISSING  # type: ignore[assignment]
    """Local path or remote URL to the robot USD file."""

    device: str = "cuda:0"
    """Warp device string."""

    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.6)
    """Default root position ``(x, y, z)`` [m]."""

    default_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Default root orientation ``(x, y, z, w)`` quaternion."""

    default_joint_pos: dict[str, float] | None = None
    """Default revolute joint positions as ``{regex: value}`` dict, or ``None``."""

    collapse_fixed_joints: bool = False
    """Merge fixed joints for a simpler kinematic tree."""
