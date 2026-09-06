# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for retarget acceptance criteria.

Each :class:`CriterionBaseCfg` subclass declares one post-solve predicate.
Its :attr:`class_type` is called by the shared family executor with the
configuration and the explicit solved candidate data.
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
    class_type: type | str = "{DIR}.criteria:evaluate_collision_check"

    n_samples: int = 16
    """Surface probe points per body."""

    max_pen: float = 0.02
    """Maximum allowed probe penetration depth [m]."""


@configclass
class LateralHipLimitCfg(CriterionBaseCfg):
    """Config for :class:`LateralHipLimit`.

    Robot presets resolve :attr:`joint_pattern` explicitly at the composition
    root. Skip the criterion by omitting it from the family.
    """

    name: str = "lateral_hip_limit"
    class_type: type | str = "{DIR}.criteria:evaluate_lateral_hip_limit"

    joint_pattern: str | None = None
    """Regex for lateral hip joint names."""

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
    class_type: type | str = "{DIR}.criteria:evaluate_joint_within_limit"

    limit_ratio: float = 0.9
    """Allowed fraction of the effective retarget joint interval."""


@configclass
class SupportPolygonStabilityCfg(CriterionBaseCfg):
    """Acceptance bounds for the objective's cached signed stability margin."""

    name: str = "stability"
    class_type: type | str = "{DIR}.criteria:evaluate_support_polygon_stability"

    minimum_contacts: int = 3
    """Minimum active contacts required to define a support polygon."""

    minimum_margin: float = 0.0
    """Minimum accepted signed CoM-to-support-edge margin [m]."""


@configclass
class FootPositionErrorCfg(CriterionBaseCfg):
    """Acceptance bound for the cached final-FK foot position error."""

    name: str = "foot_err"
    class_type: type | str = "{DIR}.criteria:evaluate_foot_position_error"

    max_err: float = 0.1
    """Error threshold [m] applied to the aggregated per-foot error."""

    aggregate: str = "sum"
    """``"max"`` to bound the worst foot, ``"sum"`` to bound total drift."""


@configclass
class SolverCostOutlierCfg(CriterionBaseCfg):
    """Config for :class:`SolverCostOutlier` (residual IK-quality filter)."""

    name: str = "cost"
    class_type: type | str = "{DIR}.criteria:evaluate_solver_cost_outlier"

    threshold_multiplier: float = 3.0
    """Multiplier on batch-median cost above which a candidate is rejected."""
