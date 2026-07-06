# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for custom Newton IK objectives.

Each :class:`IKObjectiveBaseCfg` subclass declares one numerical term and
sets :attr:`class_type` to a builder. Builders receive only the objective
configuration and an explicit :class:`IKObjectiveBuildContext`; objectives
never capture a domain pipeline.
"""

from __future__ import annotations

from dataclasses import MISSING, field

from isaaclab.utils.configclass import configclass


@configclass
class IKObjectiveBaseCfg:
    """Base configuration for a retarget IK objective.

    Subclasses set :attr:`class_type` to an objective builder resolvable from
    a ``"{DIR}.module:symbol"`` string. Builders are called as
    ``class_type(cfg, context)``.
    """

    class_type: type | str = MISSING  # type: ignore[assignment]
    """Objective implementation class."""


@configclass
class BodyPointsCfg:
    """Body origins on the articulation represented by one kinematic model."""

    asset: str = MISSING  # type: ignore[assignment]
    bodies: list[str] | tuple[str, ...] | str = MISSING  # type: ignore[assignment]


@configclass
class EntityPositionCfg:
    """Root position of the articulation represented by one kinematic model."""

    asset: str = MISSING  # type: ignore[assignment]


@configclass
class EntityRotationCfg:
    """Root rotation of the articulation represented by one kinematic model."""

    asset: str = MISSING  # type: ignore[assignment]


@configclass
class IKObjectivePositionCfg(IKObjectiveBaseCfg):
    """Body-position objective over a generated target field."""

    class_type: type | str = "{DIR}.standard:build_position_objective"
    name: str = MISSING  # type: ignore[assignment]
    current: BodyPointsCfg | EntityPositionCfg = MISSING  # type: ignore[assignment]
    target_bind: str = MISSING  # type: ignore[assignment]
    weight: float = 1.0


@configclass
class IKObjectiveRotationCfg(IKObjectiveBaseCfg):
    """Root-rotation objective over generated base rotations."""

    class_type: type | str = "{DIR}.standard:build_rotation_objective"
    name: str = "base_rotation"
    current: EntityRotationCfg = MISSING  # type: ignore[assignment]
    target_bind: str = MISSING  # type: ignore[assignment]
    weight: float = 1.0


@configclass
class IKObjectiveJointLimitCfg(IKObjectiveBaseCfg):
    """Newton joint-limit objective."""

    class_type: type | str = "{DIR}.standard:build_joint_limit_objective"
    name: str = "joint_limit"
    weight: float = 10.0


@configclass
class IKObjectiveJointPinCfg(IKObjectiveBaseCfg):
    """Config for per-problem joint-coordinate targets."""

    class_type: type | str = "{DIR}.joint_pin:build_joint_pin_objective"

    weight: float = 10.0
    """Residual weight [unitless]."""


@configclass
class IKObjectiveMeshCollisionCfg(IKObjectiveBaseCfg):
    """Config for collision probes against one generated obstacle mesh.

    Probes attached to active contact bodies are gated by the solve context;
    all other probes remain active.
    """

    class_type: type | str = "{DIR}.mesh_collision:build_mesh_collision_objective"

    weight: float = 3.0
    """Residual weight [unitless]."""

    margin: float = 0.05
    """Softplus temperature [m]. Larger values soften the penalty's knee."""

    n_samples: int = 4
    """Surface probe points per body."""

    max_distance: float = 2.0
    """Mesh query radius [m]."""

    one_sided_up_axis: tuple[float, float, float] | None = (0.0, 0.0, 1.0)
    """Optional world up axis for one-sided surface penetration."""


@configclass
class IKObjectiveStabilityMarginCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveStabilityMargin`.

    Foot identities and per-candidate contact masks come from the explicit
    objective-build context.
    """

    class_type: type | str = "{DIR}.standard:build_stability_margin_objective"

    weight: float = 1.0
    """Residual weight [unitless]."""


@configclass
class IKObjectiveGravityTorqueCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveGravityTorque`.

    Penalizes the per-revolute static gravity-compensation torque
    :math:`\\tau_j = \\hat{a}_j \\cdot (r_j \\times m_j\\, g)` where
    :math:`\\hat{a}_j` is the joint axis, :math:`r_j` the vector from
    the joint to the subtree COM, and :math:`m_j` the subtree mass.
    The zero-torque configuration is the subtree COM directly below the
    joint axis (a "hanging" pose), so this term pulls free DOFs --
    e.g. a raised foot's leg in an nc<4 stance -- toward a
    gravitationally natural posture without fighting the constrained
    DOFs required to hit the contact targets.

    Weight should be small enough that the foot-contact, base-pose, and
    joint-regularize objectives dominate: empirically ``0.01``--``0.05``
    gives natural hanging/folded postures for unconstrained limbs
    without biasing the stance legs away from their contact solutions.
    """

    class_type: type | str = "{DIR}.standard:build_gravity_torque_objective"

    weight: float = 0.02
    """Residual weight [unitless] applied uniformly across revolute joints."""


@configclass
class IKObjectiveJointDefaultCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveJointDefault`.

    Penalizes deviation of every (non-root) joint DOF from its angle in
    :attr:`~isaaclab_tasks.core.multi_task.kinematics.NewtonKinematics.default_joint_q`,
    i.e. the asset's default joint configuration captured at kinematics
    init. Differs from :class:`IKObjectiveJointRegularizeCfg` in two ways:

    * **Coverage:** every (non-root) DOF is regularized; there is no
      regex selection.
    * **Targets:** targets are fixed to the per-DOF entries of
      ``default_joint_q``, so each joint is pulled toward the robot's
      nominal pose rather than a user-supplied angle.

    Use a small weight (typical ``0.02``--``0.1``) so the foot-contact,
    base-pose, and stability objectives still dominate the solve and the
    objective only nudges the IK toward the robot's nominal pose where
    the contact constraints leave slack.
    """

    class_type: type | str = "{DIR}.standard:build_joint_default_objective"

    weight: float = 0.05
    """Uniform residual weight [unitless] applied to every (non-root) DOF."""

    skip_root: bool = True
    """Exclude the free-root joint's 6 DOFs from regularization."""


@configclass
class IKObjectiveJointRegularizeCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveJointRegularize`.

    Each entry in :attr:`joint_targets` maps a joint-name regex to the
    target angle [rad] its matched DOFs are pulled toward. DOFs not
    matched by any pattern are left free. If multiple patterns match the
    same DOF, the **last** matching entry's target wins (Python dict
    insertion order). Patterns that match zero joints on the current
    robot are silently skipped -- useful for multi-robot presets where
    each robot uses a different joint-naming convention.

    An empty :attr:`joint_targets` is invalid. Robot presets must resolve the
    desired mapping at the composition root; there is no pipeline fallback.

    Example::

        IKObjectiveJointRegularizeCfg(
            joint_targets={
                ".*HAA": 0.0,  # ANYmal-C HAA -> 0 rad
                ".*hip_joint": 0.0,  # go2/b2 HAA   -> 0 rad
                ".*hip_x": 0.0,  # spot HAA     -> 0 rad
            },
            weight=3.0,
        )
    """

    class_type: type | str = "{DIR}.standard:build_joint_regularize_objective"

    joint_targets: dict[str, float] = field(default_factory=dict)
    """Mapping of joint-name regex to target angle [rad]."""

    weight: float = 1.0
    """Uniform residual weight [unitless] applied to every matched DOF."""
