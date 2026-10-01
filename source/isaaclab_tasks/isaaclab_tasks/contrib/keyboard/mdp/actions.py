# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint controls written directly into Newton's flat control arrays."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import torch
import warp as wp

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..newton_selection import (
    JOINT_COORD,
    JOINT_DOF,
    scalar_field_active,
    scalar_field_read,
    scalar_field_write,
)
from ..selection_paths import NewtonSelectorCfg


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _relative_joint_targets(
    q: Any,
    qd: Any,
    ke: Any,
    kd: Any,
    limit: Any,
    target_q: Any,
    target_qd: Any,
    force: Any,
    action: wp.array2d[float],
    applied_effort: wp.array2d[float],
):
    world, slot = wp.tid()
    position = scalar_field_read(q, world, slot)
    target = position
    effort = float(0.0)
    if scalar_field_active(qd, world, slot):
        target = position + action[world, slot]
        # Keep target rounding and separate products identical to the tensor term.
        effort = scalar_field_read(ke, world, slot) * (target - position)
        effort -= scalar_field_read(kd, world, slot) * scalar_field_read(qd, world, slot)
        maximum = scalar_field_read(limit, world, slot)
        if effort < -maximum:
            effort = -maximum
        if effort > maximum:
            effort = maximum
        if wp.isnan(maximum):
            effort = maximum
    scalar_field_write(target_q, world, slot, target)
    scalar_field_write(target_qd, world, slot, 0.0)
    scalar_field_write(force, world, slot, 0.0)
    applied_effort[world, slot] = effort


@configclass
class NewtonRelativeJointPositionActionCfg(ActionTermCfg):
    """Relative scalar-joint targets [rad], using the USD-authored implicit drives."""

    class_type: str = "{DIR}.actions:NewtonRelativeJointPositionAction"
    joints: NewtonSelectorCfg = MISSING
    dofs: NewtonSelectorCfg = MISSING
    scale: float = 0.02


class NewtonRelativeJointPositionAction(ActionTerm):
    """Apply q + scaled action every physics step and retain implicit-PD telemetry."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.joints.frequency != JOINT_COORD or cfg.dofs.frequency != JOINT_DOF:
            raise ValueError("Relative joint targets require coordinate and DOF selectors.")
        self.joints, self.dofs = cfg.joints, cfg.dofs
        if self.joints.static_counts != self.dofs.static_counts:
            raise ValueError("Relative joint targets require scalar joints (one coordinate per DOF).")
        if not torch.equal(wp.to_torch(self.joints.joint_ids), wp.to_torch(self.dofs.joint_ids)):
            raise ValueError("Coordinate and DOF selectors must refer to the same scalar joints in the same order.")
        self._targets = self.joints if self.joints.use_coord_layout_targets else self.dofs
        self._raw = torch.zeros((env.num_envs, self.joints.width), device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self.applied_effort = torch.zeros_like(self._raw)
        self._processed_wp = wp.from_torch(self._processed)
        self._effort_wp = wp.from_torch(self.applied_effort)

    @property
    def action_dim(self):
        return self._raw.shape[1]

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._processed.copy_(actions * self.cfg.scale)

    def apply_actions(self):
        wp.launch(
            _relative_joint_targets,
            dim=self._processed.shape,
            inputs=[
                self.joints.scalar_field("state", "joint_q"),
                self.dofs.scalar_field("state", "joint_qd"),
                self.dofs.scalar_field("model", "joint_target_ke"),
                self.dofs.scalar_field("model", "joint_target_kd"),
                self.dofs.scalar_field("model", "joint_effort_limit"),
                self._targets.scalar_field("control", "joint_target_q"),
                self.dofs.scalar_field("control", "joint_target_qd"),
                self.dofs.scalar_field("control", "joint_f"),
                self._processed_wp,
            ],
            outputs=[self._effort_wp],
            device=self._processed_wp.device,
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._processed[ids] = 0.0
        self.applied_effort[ids] = 0.0
