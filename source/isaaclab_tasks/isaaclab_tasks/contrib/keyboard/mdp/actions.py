# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint controls written directly into Newton's flat control arrays."""

from __future__ import annotations

from dataclasses import MISSING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..newton_selection import JOINT_COORD, JOINT_DOF, NewtonSelectorCfg


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
        self._q_ids, self._qd_ids = self.joints.dense_ids(), self.dofs.dense_ids()
        if self._q_ids.shape != self._qd_ids.shape:
            raise ValueError("Relative joint targets require scalar joints (one coordinate per DOF).")
        if not torch.equal(wp.to_torch(self.joints.joint_ids), wp.to_torch(self.dofs.joint_ids)):
            raise ValueError("Coordinate and DOF selectors must refer to the same scalar joints in the same order.")
        model = NewtonManager.get_model()
        self._target_ids = self._q_ids if model.use_coord_layout_targets else self._qd_ids
        self._raw = torch.zeros(self._q_ids.shape, device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self.applied_effort = torch.zeros_like(self._raw)

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
        model, state, control = NewtonManager.get_model(), NewtonManager.get_state(), NewtonManager.get_control()
        q = wp.to_torch(state.joint_q)[self._q_ids]
        qd = wp.to_torch(state.joint_qd)[self._qd_ids]
        active = self.dofs.dense_active()
        target = torch.where(active, q + self._processed, q)
        wp.to_torch(control.joint_target_q)[self._target_ids] = target
        wp.to_torch(control.joint_target_qd)[self._qd_ids] = 0.0
        wp.to_torch(control.joint_f)[self._qd_ids] = 0.0
        ke = wp.to_torch(model.joint_target_ke)[self._qd_ids]
        kd = wp.to_torch(model.joint_target_kd)[self._qd_ids]
        limit = wp.to_torch(model.joint_effort_limit)[self._qd_ids]
        effort = (ke * (target - q) - kd * qd).clamp(-limit, limit)
        self.applied_effort.copy_(torch.where(active, effort, 0.0))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._processed[ids] = 0.0
        self.applied_effort[ids] = 0.0
