# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the retargeting pipeline."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class SamplerBaseCfg:
    """Base configuration for a pipeline sampler.

    Subclass this to define a concrete sampling strategy and set
    :attr:`class_type` to the corresponding :class:`SamplerBase` subclass.
    """

    class_type: type = None  # type: ignore[assignment]
    """Implementation class.  Must be a :class:`SamplerBase` subclass."""


@configclass
class RetargetPipelineCfg:
    """General retarget pipeline configuration."""

    max_candidates: int = 2000
    """Maximum number of candidates in the buffer."""

    ik_iterations: int = 200
    """Maximum number of IK solver iterations."""

    ik_convergence_threshold: float = 0.01
    """Stop IK early when mean cost change falls below this threshold."""

    device: str = "cuda:0"
    """Warp device."""
