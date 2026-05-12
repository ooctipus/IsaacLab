# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sawyer robot module for the Meta-World multi-task port.

Wraps :data:`~isaaclab_contrib.tasks.manipulation.metaworld.metaworld_assets_cfg.SAWYER_METAWORLD_CFG`
for use in :class:`~...registry.MultiTaskRegistry`. The arm action is the
3-d xyz delta + workspace-clamped DiffIK from the Meta-World env (column key
``"metaworld_arm"``); the gripper action is the 1-d Meta-World scalar
gripper (column key ``"gripper"``).
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.tasks.manipulation.metaworld.mdp import (
    MetaworldArmActionCfg,
    MetaworldGripperActionCfg,
)
from isaaclab_contrib.tasks.manipulation.metaworld.metaworld_assets_cfg import SAWYER_METAWORLD_CFG
from isaaclab_contrib.tasks.manipulation.multitask import mdp

from ._base import RobotModuleCfg


@dataclass
class SawyerMetaworldRobotCfg(RobotModuleCfg):
    """Meta-World Sawyer arm + 5-body box gripper for the multi-task registry.

    Action specs match Meta-World's standalone env exactly:

    * **arm column**: 3-d xyz delta via DiffIK pinv (k_val=25), with
      workspace clamping (``hand_low`` / ``hand_high``). Column key
      ``"metaworld_arm"`` so it does not collide with Franka's 6-d ``"arm"``.
    * **gripper column**: 1-d Meta-World gripper scalar (open=-1, close=+1).
      Column key ``"gripper"``.
    """

    workspace_low: tuple[float, float, float] = (-0.5, 0.40, 0.05)
    """Lower bound of the EE workspace clamp [m]. Matches MW ``hand_low``."""

    workspace_high: tuple[float, float, float] = (0.5, 1.0, 0.5)
    """Upper bound of the EE workspace clamp [m]. Matches MW ``hand_high``."""

    arm_scale: float = 0.01
    """Per-step xyz delta scale [m]. Matches MW ``action_scale``."""

    @property
    def name(self) -> str:
        return "sawyer"

    @property
    def all_joint_names(self) -> list[str]:
        return ["right_j[0-6]", "[rl]_close"]

    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------

    def scene_assets(self, group: str) -> dict[str, object]:
        return {
            "sawyer_robot": SAWYER_METAWORLD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
            ),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_specs(self, group: str | None = None) -> dict[str, tuple[int, object]]:
        arm = MetaworldArmActionCfg(
            asset_name="sawyer_robot",
            joint_names=["right_j[0-6]"],
            body_name="hand",
            scale=self.arm_scale,
            controller=DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="pinv",
                ik_params={"k_val": 25.0},
            ),
            workspace_low=self.workspace_low,
            workspace_high=self.workspace_high,
        )
        gripper = MetaworldGripperActionCfg(asset_name="sawyer_robot")
        return {
            "metaworld_arm": (3, arm),
            "gripper": (1, gripper),
        }

    # ------------------------------------------------------------------
    # Robot-side scatter obs (none — Meta-World's full 39-d state is
    # contributed by the *task* module so it can reference per-group
    # cube / command).
    # ------------------------------------------------------------------

    def scatter_obs_terms(self, group: str) -> dict[str, tuple[int | None, TermCfg]]:
        return {}

    # ------------------------------------------------------------------
    # Reset events
    # ------------------------------------------------------------------

    def reset_events(self, group: str) -> dict[str, EventTerm]:
        # The Sawyer is a *global* asset (shared across all groups), so we
        # only register the reset event once: from the first group that asks.
        # The registry de-duplicates by event-name key, so we use a fixed key.
        return {
            "sawyer_reset_to_default": EventTerm(
                func=mdp.reset_to_default,
                mode="reset",
                params={
                    "reset_joint_targets": True,
                    "asset_cfgs": [SceneEntityCfg("sawyer_robot")],
                },
            ),
            "sawyer_reset_joints": EventTerm(
                func="isaaclab.envs.mdp:reset_joints_by_offset",
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("sawyer_robot"),
                    "position_range": (-0.05, 0.05),
                    "velocity_range": (0.0, 0.0),
                },
            ),
        }


SAWYER_METAWORLD = SawyerMetaworldRobotCfg()
"""Default Sawyer module instance for Meta-World multi-task registry use."""
