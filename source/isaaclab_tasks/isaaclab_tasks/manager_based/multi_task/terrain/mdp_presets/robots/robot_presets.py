# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-axis preset *classes* for the position locomotion task.

Every :class:`PresetCfg` declared here represents one robot-specific scalar
that the base environment (and rsl_rl agent) references inline instead of
hard-coding.  Per-robot modules in this package (e.g. :mod:`anymal_c`,
:mod:`go2`) populate their own field on these classes at import time, and the
preset resolver then substitutes the picked field at every consumption site.

To register a new robot, drop a new module alongside this one and set class
attributes such as::

    from .robot_presets import BaseBodyNameCfg, RobotArticulationCfg, ...

    RobotArticulationCfg.<robot> = <ROBOT>_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    BaseBodyNameCfg.<robot> = "..."
    ...

Only add fields; **never reassign** ``default`` -- it is the fallback used
when no robot preset is selected.  Package-level defaults preserve the
strings previously hard-coded in the base env / reward / termination configs
so the task still loads without a robot picked (except for
:class:`RobotArticulationCfg.default`, which is :data:`MISSING` and will fail
loudly if resolution happens with no robot selected).
"""

from dataclasses import MISSING, field

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class RobotArticulationCfg(PresetCfg):
    """Full :class:`ArticulationCfg` for the robot asset at ``scene.robot``.

    Required -- no sensible default exists, so resolving without picking a
    robot preset leaves ``scene.robot`` as :data:`MISSING`.
    """

    default: ArticulationCfg = MISSING  # type: ignore[assignment]


@configclass
class HeightScannerPrimPathCfg(PresetCfg):
    """USD prim path anchoring the height-scanner ray caster."""

    default: str = "{ENV_REGEX_NS}/Robot/base"


@configclass
class BaseBodyNameCfg(PresetCfg):
    """Robot base / torso body name.

    Used by events that target the base mass (``add_base_mass``) and by the
    viewer (``viewer.body_name``).
    """

    default: str = "base"


@configclass
class BaseContactBodyNamesCfg(PresetCfg):
    """Body-name regex for bodies that must not contact the terrain.

    Drives the ``base_contact`` termination's ``sensor_cfg.body_names``.
    """

    default: str = "^(?!.*foot).*$"


@configclass
class FootBodyNamesCfg(PresetCfg):
    """Body-name regex matching the robot's feet.

    Drives reward terms that index foot bodies (e.g. ``foot_touchdown``).
    """

    default: str = ".*FOOT.*"


@configclass
class NonFootBodyNamesCfg(PresetCfg):
    """Body-name regex matching every body **except** the feet.

    Drives the ``undesired_contact`` reward's ``sensor_cfg.body_names``.
    """

    default: str = "^(?!.*(?:(FOOT))).*$"


@configclass
class AsyncFootPairsCfg(PresetCfg):
    """Foot pairs whose air/contact times should differ (gait diagnostic)."""

    default: tuple[tuple[str, str], ...] = ()


@configclass
class SyncFootPairsCfg(PresetCfg):
    """Foot pairs whose air/contact times should match (gait diagnostic)."""

    default: tuple[tuple[str, str], ...] = ()


@configclass
class ExperimentNameCfg(PresetCfg):
    """rsl_rl experiment name, picked up by ``PositionLocomotionPPORunnerCfg``."""

    default: str = "position_command"


@configclass
class RetargetFootBodyNamesCfg(PresetCfg):
    """Exact foot body names used by the retarget pipeline.

    Consumed by :attr:`RetargetPipelineCfg.foot_body_names` to resolve
    Newton body indices for the feet. Must match the body names produced
    by :class:`NewtonKinematics` when loading the robot's USD (case-
    sensitive, no regex).

    Required -- no sensible default exists, so resolving without picking a
    robot preset (or for a robot without a retarget preset) leaves the
    field as :data:`MISSING` and fails loudly at pipeline construction.
    """

    default: list[str] = MISSING  # type: ignore[assignment]


@configclass
class RetargetHaaJointPatternCfg(PresetCfg):
    """Regex matching hip-abduction/adduction joint names for retarget.

    Consumed by :attr:`RetargetPipelineCfg.haa_joint_pattern`. ``None``
    disables the :class:`HaaLimit` criterion -- appropriate for robots
    with no abduction joints or where over-splay is not a concern.
    """

    default: str | None = None


@configclass
class RetargetJointRegularizeTargetsCfg(PresetCfg):
    """Per-robot joint-name regex -> target-angle dict for retarget IK.

    Consumed by :attr:`RetargetPipelineCfg.joint_regularize_targets`.
    Each entry pulls its matched DOFs toward the listed angle during
    IK; unmatched DOFs are left free. Empty dict disables the
    regularizer entirely -- appropriate for robots where no joint-
    space prior is needed.
    """

    default: dict[str, float] = field(default_factory=dict)


@configclass
class RetargetBasePosWeightCfg(PresetCfg):
    """Per-robot weight for the base-position IK objective [unitless].

    Consumed by :attr:`RetargetPipelineCfg.base_pos_weight`. Keeps the
    IK near the sampler's plane-fit base position. Small by default so
    the foot-contact targets (weight ``1.0``) dominate -- the base is a
    soft anchor, not a hard target. Raise for robots where base-xy
    placement is critical (e.g. narrow support polygon bipeds).
    """

    default: float = 0.05


@configclass
class RetargetBaseRotWeightCfg(PresetCfg):
    """Per-robot weight for the base-orientation IK objective [unitless].

    Consumed by :attr:`RetargetPipelineCfg.base_rot_weight`. Pulls the
    base quaternion toward the sampler's plane-fit target. The correct
    value is **body-plan-specific**:

    * **Quadruped, full 4-contact stance**: ``0.5`` (default). The
      sampler's plane fit is informative; this weight agrees with the
      stability-margin objective and keeps the base aligned with the
      terrain-fit plane.
    * **Quadruped, nc < 4 stance** (rearing, handstand): prefer
      ``0.0``. The plane fit degenerates to identity (rank-deficient
      cross-covariance at 2-3 contacts), which actively fights a
      natural forward/backward tilt. Set per-call via CLI override,
      not per-robot.
    * **Biped**: ``2.0``. The identity base target *is* upright
      (correct posture), but the stability-margin objective (weight
      ``1.0``) is strong enough at the default ``0.5`` to tilt the
      torso to project the COM onto the narrow 2-foot segment. A
      stronger weight holds the torso upright; if the base were
      allowed to fall, downstream policy training would inherit
      non-standing seed poses.
    """

    default: float = 0.5


@configclass
class RetargetGravityWeightCfg(PresetCfg):
    """Per-robot weight for the gravity-torque IK objective [unitless].

    Consumed by the :class:`IKObjectiveGravityTorque` built for the
    retarget pipeline's extra-objective list. Pulls unconstrained
    revolute DOFs toward their gravity-stable hang-down pose (e.g.
    raised legs in nc<4 stances, arms on bipeds). ``0.0`` disables the
    objective entirely.
    """

    default: float = 0.02
