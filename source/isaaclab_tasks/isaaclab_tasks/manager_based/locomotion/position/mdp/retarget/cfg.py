# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the retargeting pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

from ...utils.kinematic.newton_kinematics import NewtonKinematicsCfg


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
    """Full retarget pipeline configuration.

    Nests the kinematics, sampler, and foot specification so the
    pipeline can be constructed with ``RetargetPipeline(cfg)``.
    """

    kin: NewtonKinematicsCfg = MISSING  # type: ignore[assignment]
    """Kinematics model configuration."""

    sampler: SamplerBaseCfg = MISSING  # type: ignore[assignment]
    """Sampler configuration (with ``class_type`` set)."""

    foot_body_names: list[str] = MISSING  # type: ignore[assignment]
    """Body names of the feet (exact match against Newton body names)."""

    max_candidates: int = 2000
    """Maximum number of candidates in the buffer."""

    ik_iterations: int = 200
    """Maximum number of IK solver iterations."""

    ik_convergence_threshold: float = 0.01
    """Stop IK early when mean cost change falls below this threshold."""

    extra_objectives_factory: Callable | None = None
    """Optional callable ``(kin, foot_ids, n_problems, sampler, wp_mesh) -> list[IKObjective]``.

    Returns additional IK objectives beyond the standard set (foot position,
    base pose, joint limits).  The extra objectives are appended to the
    solver's objective list.
    """
