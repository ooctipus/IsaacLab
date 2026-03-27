# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for batched group-aware action terms.

``BatchedDiffIKActionCfg`` and ``BatchedBinaryGripperActionCfg``
take a ``groups`` dict mapping group keys to per-group parameters
(multi-group, shared columns — safe because column semantics are
uniform across robots).

``BatchedRelJointPosActionCfg`` handles **one group per term** so
that each robot gets independent action columns — avoiding semantic
ambiguity when joint counts or meanings differ across robots.
"""

from __future__ import annotations

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

OffsetCfg = DifferentialInverseKinematicsActionCfg.OffsetCfg
"""Re-exported from core for convenience."""


# ============================================================
# Per-group parameter configs
# ============================================================


@configclass
class DiffIKGroupCfg:
    """Per-group parameters for :class:`BatchedDiffIKActionCfg`.

    Specifies arm joint patterns and optional body offset for one
    robot group.  The robot asset and IK body are inferred from
    ``robot_meta[group_key].asset_cfg``.
    """

    joint_names: list[str] = MISSING
    """Arm joint name patterns for inverse kinematics."""

    body_offset: DifferentialInverseKinematicsActionCfg.OffsetCfg | None = None
    """Optional IK target frame offset (quaternion in ``(x, y, z, w)`` order)."""

    scale: float | None = None
    """Per-group action scale override. If ``None``, uses the shared default."""


@configclass
class GripperGroupCfg:
    """Per-group parameters for :class:`BatchedBinaryGripperActionCfg`."""

    joint_names: list[str] = MISSING
    """Gripper joint name patterns."""

    open_command_expr: dict[str, float] = MISSING
    """Joint name pattern to value mapping for the open position."""

    close_command_expr: dict[str, float] = MISSING
    """Joint name pattern to value mapping for the close position."""


# ============================================================
# Top-level action term configs
# ============================================================


@configclass
class BatchedDiffIKActionCfg(ActionTermCfg):
    """Batched differential IK action for multi-robot environments.

    Iterates ``robot_meta`` to discover robot groups and maintains
    per-group :class:`DifferentialIKController` instances.  All groups
    share the same action columns since their env rows are disjoint.

    Example::

        @configclass
        class ActionsCfg:
            arm_action = BatchedDiffIKActionCfg(
                robot_meta=ROBOT_META,
                controller=DifferentialIKControllerCfg(...),
                scale=0.5,
                groups={
                    "openarm_reach": DiffIKGroupCfg(
                        joint_names=["openarm_joint.*"],
                    ),
                    "franka_reach": DiffIKGroupCfg(
                        joint_names=["panda_joint.*"],
                        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                            pos=(0.0, 0.0, 0.107),
                        ),
                    ),
                },
            )
    """

    class_type: type | str = "{DIR}.batched_actions:BatchedDiffIKAction"

    asset_name: str = "__batched__"
    """Unused — asset names come from ``robot_meta``."""

    robot_meta: dict = {}
    """Mapping from group key to group config (e.g., ReachGroupCfg, LiftGroupCfg)."""

    controller: DifferentialIKControllerCfg = MISSING
    """Shared IK controller configuration."""

    scale: float | tuple = 0.5
    """Default action scale for all groups."""

    groups: dict = MISSING
    """Per-group DiffIK parameters (group_key -> :class:`DiffIKGroupCfg`)."""


@configclass
class BatchedBinaryGripperActionCfg(ActionTermCfg):
    """Batched binary gripper action for multi-robot environments.

    Example::

        @configclass
        class ActionsCfg:
            gripper_action = BatchedBinaryGripperActionCfg(
                robot_meta=ROBOT_META,
                groups={
                    "openarm_lift": GripperGroupCfg(
                        joint_names=["openarm_finger_joint.*"],
                        open_command_expr={"openarm_finger_joint.*": 0.044},
                        close_command_expr={"openarm_finger_joint.*": 0.0},
                    ),
                },
            )
    """

    class_type: type | str = "{DIR}.batched_actions:BatchedBinaryGripperAction"

    asset_name: str = "__batched__"
    """Unused — asset names come from ``robot_meta``."""

    robot_meta: dict = {}
    """Mapping from group key to group config."""

    groups: dict = MISSING
    """Per-group gripper parameters (group_key -> :class:`GripperGroupCfg`)."""


@configclass
class BatchedRelJointPosActionCfg(ActionTermCfg):
    """Relative joint position action for one robot group.

    Each group registers its own term so that different robots get
    independent action columns — avoiding semantic ambiguity in the
    policy output layer.

    Example::

        @configclass
        class ActionsCfg:
            franka_joints = BatchedRelJointPosActionCfg(
                robot_meta=ROBOT_META,
                group_name="franka_cabinet",
                joint_names=["panda_joint.*"],
                scale=0.1,
            )
            ur10_joints = BatchedRelJointPosActionCfg(
                robot_meta=ROBOT_META,
                group_name="ur10_reach",
                joint_names=[".*"],
                scale=0.1,
            )
    """

    class_type: type | str = "{DIR}.batched_actions:BatchedRelJointPosAction"

    asset_name: str = "__batched__"
    """Unused — asset name comes from ``robot_meta[group_name]``."""

    robot_meta: dict = {}
    """Mapping from group key to group config."""

    group_name: str = MISSING
    """Clone-group name (must match a key in ``robot_meta``)."""

    joint_names: list[str] = MISSING
    """Joint name patterns for relative position control."""

    scale: float = 0.1
    """Action scale."""
