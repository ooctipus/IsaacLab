# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for curriculum sampling strategies."""

from __future__ import annotations

from isaaclab.utils import configclass

from .sampling_strategies import BetaSignal, FrontierSignal, UniformSignal


@configclass
class BetaSignalCfg:
    """Blueprint for a :class:`BetaSignal`."""

    class_type: type[BetaSignal] | str = "{DIR}.sampling_strategies:BetaSignal"
    target: float = 0.66
    kappa: float = 1.0


@configclass
class FrontierSignalCfg:
    """Blueprint for a :class:`FrontierSignal`."""

    class_type: type[FrontierSignal] | str = "{DIR}.sampling_strategies:FrontierSignal"
    k: int = 8
    dilation_steps: int = 1


@configclass
class UniformSignalCfg:
    """Blueprint for a :class:`UniformSignal`."""

    class_type: type[UniformSignal] | str = "{DIR}.sampling_strategies:UniformSignal"


SignalCfg = BetaSignalCfg | FrontierSignalCfg | UniformSignalCfg
"""Discriminated union of signal cfg types."""
