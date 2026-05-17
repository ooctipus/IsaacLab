# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for retarget acceptance criteria.

Each :class:`CriterionBaseCfg` subclass declares the static parameters of
a criterion and sets :attr:`class_type` to the criterion implementation.
The pipeline instantiates criteria via ``cfg.class_type(cfg, pipeline,
wp_mesh)``; the criterion's ``__init__`` pulls any runtime state it
needs (kinematics, foot indices, solver costs) from ``pipeline``.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class CriterionBaseCfg:
    """Base configuration for a retarget acceptance criterion.

    Subclasses set :attr:`class_type` to the criterion implementation
    (resolvable ``"{DIR}.module:ClassName"`` string). :attr:`name`
    becomes the key in the pipeline's rejection summary.
    """

    name: str = MISSING  # type: ignore[assignment]
    """Key under which this criterion's rejections are reported."""

    class_type: type | str = MISSING  # type: ignore[assignment]
    """Criterion implementation class. Called as ``class_type(cfg, pipeline, wp_mesh)``."""


@configclass
class CollisionCheckCfg(CriterionBaseCfg):
    """Config for :class:`CollisionCheck`.

    Probes every non-foot body against the terrain mesh and rejects
    candidates whose max probe penetration exceeds :attr:`max_pen`.
    """

    name: str = "collision"
    class_type: type | str = "{DIR}.criteria:CollisionCheck"

    n_samples: int = 16
    """Surface probe points per body."""

    max_pen: float = 0.02
    """Maximum allowed probe penetration depth [m]."""


@configclass
class LateralHipLimitCfg(CriterionBaseCfg):
    """Config for :class:`LateralHipLimit`.

    :attr:`joint_pattern` is resolved on the criterion side: when ``None``,
    falls back to :attr:`RetargetPipelineCfg.lateral_hip_joint_pattern`
    (the robot-preset regex). Skip the criterion entirely by omitting it
    from the pipeline config's ``criteria`` list.
    """

    name: str = "lateral_hip_limit"
    class_type: type | str = "{DIR}.criteria:LateralHipLimit"

    joint_pattern: str | None = None
    """Override regex for lateral hip joint names. ``None`` uses the pipeline cfg's pattern."""

    max_angle: float = 1.05
    """Maximum absolute lateral hip angle [rad]."""


@configclass
class JointWithinLimitCfg(CriterionBaseCfg):
    """Config for :class:`JointWithinLimit`.

    Shrinks every non-root joint's effective retarget interval around
    its center and rejects candidates outside the reduced interval. The
    effective interval is the Newton joint limit intersected with the
    sampler's FK joint range around the default pose. With the default
    ``limit_ratio = 0.9``, a symmetric interval ``[-L, L]`` becomes
    ``[-0.9 L, 0.9 L]``.
    """

    name: str = "joint_limit"
    class_type: type | str = "{DIR}.criteria:JointWithinLimit"

    limit_ratio: float = 0.9
    """Allowed fraction of the effective retarget joint interval."""


@configclass
class SupportPolygonStabilityCfg(CriterionBaseCfg):
    """Config for :class:`SupportPolygonStability`.

    The lateral tolerance only applies when the number of contacts is
    exactly two (support collapses to a segment). For ``nc >= 3`` the
    criterion is parameter-free (strict hull inclusion).
    """

    name: str = "stability"
    class_type: type | str = "{DIR}.criteria:SupportPolygonStability"

    segment_tol_frac: float = 0.05
    """``nc == 2`` lateral tolerance fraction [unitless].

    Effective tolerance is ``segment_tol_frac × segment_length``;
    segment-length scaling is a finite-foot-footprint regularization of
    the otherwise measure-zero segment-support balance condition. Unused
    when ``nc != 2``.
    """


@configclass
class FootPositionErrorCfg(CriterionBaseCfg):
    """Config for :class:`FootPositionError`.

    ``num_bodies`` and ``foot_ids`` are pulled from the pipeline's
    :class:`NewtonKinematics` at construction time by the criterion.
    """

    name: str = "foot_err"
    class_type: type | str = "{DIR}.criteria:FootPositionError"

    max_err: float = 0.1
    """Error threshold [m] applied to the aggregated per-foot error."""

    aggregate: str = "sum"
    """``"max"`` to bound the worst foot, ``"sum"`` to bound total drift."""


@configclass
class SolverCostOutlierCfg(CriterionBaseCfg):
    """Config for :class:`SolverCostOutlier` (residual IK-quality filter)."""

    name: str = "cost"
    class_type: type | str = "{DIR}.criteria:SolverCostOutlier"

    threshold_multiplier: float = 3.0
    """Multiplier on batch-median cost above which a candidate is rejected."""
