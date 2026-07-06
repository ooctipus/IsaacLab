# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builders for standard Newton position, rotation, and joint-limit objectives."""

from __future__ import annotations

import re

import newton.ik as ik
import warp as wp

from .cfg import BodyPointsCfg, EntityPositionCfg, EntityRotationCfg
from .context import IKObjectiveBuild, IKObjectiveBuildContext, IKPositionObjectiveBuildContext


def _validate_asset(asset: str, context: IKObjectiveBuildContext) -> None:
    if asset != context.asset_name:
        raise ValueError(f"Objective asset {asset!r} does not match solved asset {context.asset_name!r}.")


def _resolve_body_ids(patterns: list[str] | tuple[str, ...] | str, context: IKObjectiveBuildContext) -> tuple[int, ...]:
    patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    names: list[str] = []
    for pattern in patterns:
        matched = [name for name in context.kinematics.body_names if re.fullmatch(pattern, name)]
        if not matched:
            raise ValueError(f"Body pattern {pattern!r} matched no bodies on {context.asset_name!r}.")
        names.extend(matched)
    if len(names) != len(set(names)):
        raise ValueError(f"Body patterns resolve duplicate bodies: {patterns!r}.")
    return tuple(context.kinematics.find_body_indices(names))


def build_position_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build body-position objectives against a named generated target field."""
    device = context.kinematics.device
    if isinstance(cfg.current, BodyPointsCfg):
        _validate_asset(cfg.current.asset, context)
        body_ids = _resolve_body_ids(cfg.current.bodies, context)
        if isinstance(context, IKPositionObjectiveBuildContext):
            point_count = len(body_ids)
            if context.body_offsets.shape != (point_count, 3):
                raise ValueError("Position-objective body offsets must have shape [point_count, 3].")
            offsets = context.body_offsets
        else:
            offsets = ((0.0, 0.0, 0.0),) * len(body_ids)
        targets = tuple(wp.zeros(context.batch_size, dtype=wp.vec3, device=device) for _ in body_ids)
        objectives = tuple(
            ik.IKObjectivePosition(
                link_index=body_id,
                link_offset=wp.vec3(*offsets[index]),
                target_positions=targets[index],
                weight=cfg.weight,
            )
            for index, body_id in enumerate(body_ids)
        )
    elif isinstance(cfg.current, EntityPositionCfg):
        _validate_asset(cfg.current.asset, context)
        objectives = (
            ik.IKObjectivePosition(
                link_index=0,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.zeros(context.batch_size, dtype=wp.vec3, device=device),
                weight=cfg.weight,
            ),
        )
    else:
        raise TypeError(f"Unsupported position current descriptor: {type(cfg.current).__name__}.")
    return IKObjectiveBuild(objectives=objectives, target_bind=cfg.target_bind)


def build_rotation_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build the root-rotation objective against generated base rotations."""
    if not isinstance(cfg.current, EntityRotationCfg):
        raise TypeError(f"Unsupported rotation current descriptor: {type(cfg.current).__name__}.")
    _validate_asset(cfg.current.asset, context)
    objective = ik.IKObjectiveRotation(
        link_index=0,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.zeros(context.batch_size, dtype=wp.vec4, device=context.kinematics.device),
        weight=cfg.weight,
    )
    return IKObjectiveBuild(objectives=(objective,), target_bind=cfg.target_bind)


def build_joint_limit_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build a Newton joint-limit objective from the declared mechanics."""
    model = context.kinematics.model
    objective = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=cfg.weight,
    )
    return IKObjectiveBuild(objectives=(objective,))


def build_gravity_torque_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build static gravity-torque regularization."""
    from .gravity_torque import IKObjectiveGravityTorque

    return IKObjectiveBuild(objectives=(IKObjectiveGravityTorque(cfg, context),))


def build_joint_default_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build default-joint regularization."""
    from .joint_default import IKObjectiveJointDefault

    return IKObjectiveBuild(objectives=(IKObjectiveJointDefault(cfg, context),))


def build_joint_regularize_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build selected-joint regularization."""
    from .joint_regularize import IKObjectiveJointRegularize

    return IKObjectiveBuild(objectives=(IKObjectiveJointRegularize(cfg, context),))


def build_stability_margin_objective(cfg, context: IKObjectiveBuildContext) -> IKObjectiveBuild:
    """Build support-polygon stability regularization."""
    from .stability_margin import IKObjectiveStabilityMargin

    return IKObjectiveBuild(objectives=(IKObjectiveStabilityMargin(cfg, context),))
