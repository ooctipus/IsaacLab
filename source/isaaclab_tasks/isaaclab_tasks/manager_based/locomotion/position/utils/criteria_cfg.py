# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for retarget acceptance criteria.

Mirrors :mod:`.criteria`: each :class:`CriterionBaseCfg` subclass
declares the static parameters of a criterion and knows how to
instantiate it against the live :class:`RetargetPipeline` (which
provides the kinematics model, foot indices, and per-run solver
costs the criterion needs at build time).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    import warp as wp

    from ..mdp.retarget.buffer import RetargetBuffer
    from ..mdp.retarget.pipeline import RetargetPipeline

    CriterionFn = Callable[[RetargetBuffer, int], torch.Tensor]


@configclass
class CriterionBaseCfg:
    """Base configuration for a retarget acceptance criterion.

    Subclasses set :attr:`class_type` to the criterion implementation
    (resolvable ``"{DIR}.module:ClassName"`` string) and implement
    :meth:`build` to plumb runtime state from the pipeline into the
    criterion's constructor. :attr:`name` becomes the key in the
    pipeline's rejection summary.
    """

    name: str = MISSING  # type: ignore[assignment]
    """Key under which this criterion's rejections are reported."""

    class_type: type | str = MISSING  # type: ignore[assignment]
    """Criterion implementation class (resolvable string or direct type)."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> CriterionFn:
        """Instantiate the criterion against a live pipeline.

        Args:
            pipeline: The initialized :class:`RetargetPipeline`
                (provides ``kin``, ``foot_body_ids``, ``_solver_costs``).
            wp_mesh: Terrain warp mesh for the current ``run`` call.

        Returns:
            A callable ``(buffer, N) -> bool[N]`` ready to append to
            :meth:`RetargetPipeline.run`'s criteria dict.
        """
        raise NotImplementedError


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

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> CriterionFn:
        from .criteria import CollisionCheck

        return CollisionCheck(
            kin=pipeline.kin,
            wp_mesh=wp_mesh,
            exclude_bodies=pipeline.foot_body_ids,
            n_samples=self.n_samples,
            max_pen=self.max_pen,
        )


@configclass
class HaaLimitCfg(CriterionBaseCfg):
    """Config for :class:`HaaLimit`.

    :attr:`joint_pattern` is resolved on the pipeline side: when ``None``,
    falls back to :attr:`RetargetPipelineCfg.haa_joint_pattern` (the
    robot-preset regex). Skip the criterion entirely by omitting it
    from the pipeline config's ``criteria`` list.
    """

    name: str = "haa_limit"
    class_type: type | str = "{DIR}.criteria:HaaLimit"

    joint_pattern: str | None = None
    """Override regex for HAA joint names. ``None`` uses the pipeline cfg's pattern."""

    max_angle: float = 1.05
    """Maximum absolute HAA angle [rad]."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> CriterionFn:
        from .criteria import HaaLimit

        pattern = self.joint_pattern if self.joint_pattern is not None else pipeline.cfg.haa_joint_pattern
        if pattern is None:
            raise ValueError(
                "HaaLimitCfg requires a joint_pattern. Either set HaaLimitCfg.joint_pattern or "
                "RetargetPipelineCfg.haa_joint_pattern (typically resolved per robot preset)."
            )
        return HaaLimit(kin=pipeline.kin, joint_pattern=pattern, max_angle=self.max_angle)


@configclass
class SupportPolygonStabilityCfg(CriterionBaseCfg):
    """Config for :class:`SupportPolygonStability` (no parameters)."""

    name: str = "stability"
    class_type: type | str = "{DIR}.criteria:SupportPolygonStability"

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> CriterionFn:
        from .criteria import SupportPolygonStability

        return SupportPolygonStability()


@configclass
class FootPositionErrorCfg(CriterionBaseCfg):
    """Config for :class:`FootPositionError`.

    Number-of-bodies and foot-indices are pulled from the pipeline's
    :class:`NewtonKinematics` at :meth:`build` time.
    """

    name: str = "foot_err"
    class_type: type | str = "{DIR}.criteria:FootPositionError"

    max_err: float = 0.1
    """Error threshold [m] applied to the aggregated per-foot error."""

    aggregate: str = "sum"
    """``"max"`` to bound the worst foot, ``"sum"`` to bound total drift."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> CriterionFn:
        from .criteria import FootPositionError

        return FootPositionError(
            num_bodies=pipeline.kin.model.body_count,
            foot_ids=pipeline.foot_body_ids,
            max_err=self.max_err,
            aggregate=self.aggregate,
        )


@configclass
class SolverCostOutlierCfg(CriterionBaseCfg):
    """Config for :class:`SolverCostOutlier` (residual IK-quality filter)."""

    name: str = "cost"
    class_type: type | str = "{DIR}.criteria:SolverCostOutlier"

    threshold_multiplier: float = 3.0
    """Multiplier on batch-median cost above which a candidate is rejected."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> CriterionFn:
        from .criteria import SolverCostOutlier

        return SolverCostOutlier(pipeline=pipeline, threshold_multiplier=self.threshold_multiplier)
