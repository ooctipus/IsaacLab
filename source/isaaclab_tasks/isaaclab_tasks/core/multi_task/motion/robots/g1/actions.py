# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 joint-position action term."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .actions_cfg import G1JointPositionActionCfg


class G1JointPositionAction(JointAction):
    """Apply normalized behavior actions through the native position-control law.

    The processed action retains the controller-normalized value used by
    history and action-rate evidence. The articulation target is held in the
    separate joint-position-target tensor.
    """

    cfg: G1JointPositionActionCfg

    def __init__(self, cfg: G1JointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        if self._num_joints == 0:
            raise ValueError("G1JointPositionAction must resolve at least one articulation joint.")
        joint_ids = range(self._asset.num_joints) if isinstance(self._joint_ids, slice) else self._joint_ids
        self._joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.int64, device=self.device)

        self._joint_position = torch.empty_like(self._raw_actions)
        self._joint_velocity = torch.empty_like(self._raw_actions)
        data = self._asset.data
        self.joint_default_position = data.default_joint_pos.torch[0, self._joint_ids].clone()
        self.joint_stiffness = data.joint_stiffness.torch[0, self._joint_ids].clone()
        self.joint_damping = data.joint_damping.torch[0, self._joint_ids].clone()
        self.joint_effort_limit = data.joint_effort_limits.torch[0, self._joint_ids].clone()
        self.default_joint_offset = torch.zeros_like(self._processed_actions)
        self._previous_processed_actions = torch.zeros_like(self._processed_actions)
        self.joint_position_target = torch.empty_like(self._processed_actions)
        self._reset_default_joint_offset = torch.empty_like(self._processed_actions)
        self._reset_joint_position_target = torch.empty_like(self._processed_actions)
        self._applied_torque = torch.zeros_like(self._processed_actions)
        self.joint_target_gain = cfg.action_scale * self.joint_effort_limit / self.joint_stiffness

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names in the declared behavior-action order."""
        return tuple(self._joint_names)

    @property
    def joint_ids(self) -> torch.Tensor:
        """Live-articulation joint indices in behavior-action order."""
        return self._joint_ids_tensor

    def process_actions(self, actions: torch.Tensor) -> None:
        """Normalize behavior actions and materialize joint targets [rad]."""
        self._previous_processed_actions.copy_(self._processed_actions)
        self._raw_actions.copy_(actions)
        torch.mul(self._raw_actions, self._scale, out=self._processed_actions)
        self._processed_actions.add_(self._offset)
        if self.cfg.clip is not None:
            torch.maximum(self._processed_actions, self._clip[:, :, 0], out=self._processed_actions)
            torch.minimum(self._processed_actions, self._clip[:, :, 1], out=self._processed_actions)
        torch.mul(self._processed_actions, self.joint_target_gain, out=self.joint_position_target)
        self.joint_position_target.add_(self.joint_default_position)
        self.joint_position_target.add_(self.default_joint_offset)

    def apply_actions(self) -> None:
        """Write the persistent joint-position target [rad] to the articulation."""
        torch.index_select(self._asset.data.joint_pos.torch, 1, self._joint_ids_tensor, out=self._joint_position)
        torch.index_select(self._asset.data.joint_vel.torch, 1, self._joint_ids_tensor, out=self._joint_velocity)
        torch.sub(self.joint_position_target, self._joint_position, out=self._applied_torque)
        self._applied_torque.mul_(self.joint_stiffness)
        self._applied_torque.addcmul_(self.joint_damping, self._joint_velocity, value=-1.0)
        torch.minimum(self._applied_torque, self.joint_effort_limit, out=self._applied_torque)
        self._applied_torque.neg_()
        torch.minimum(self._applied_torque, self.joint_effort_limit, out=self._applied_torque)
        self._applied_torque.neg_()
        self._asset.set_joint_position_target_index(target=self.joint_position_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear action state and sample the episodic default-pose offset [rad]."""
        full_reset = env_ids is None or (
            isinstance(env_ids, slice) and env_ids.start is None and env_ids.stop is None and env_ids.step is None
        )
        if full_reset:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._previous_processed_actions.zero_()
            self._applied_torque.zero_()
            offset = self.default_joint_offset
            offset.uniform_(*self.cfg.default_joint_offset_range)
            torch.add(self.joint_default_position, offset, out=self.joint_position_target)
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._previous_processed_actions[env_ids] = 0.0
            self._applied_torque[env_ids] = 0.0
            count = len(env_ids)
            offset = self._reset_default_joint_offset[:count]
            target = self._reset_joint_position_target[:count]
            offset.uniform_(*self.cfg.default_joint_offset_range)
            torch.add(self.joint_default_position, offset, out=target)
            self.default_joint_offset[env_ids] = offset
            self.joint_position_target[env_ids] = target

    @property
    def previous_processed_actions(self) -> torch.Tensor:
        """Controller-normalized actions from the preceding physical edge."""
        return self._previous_processed_actions

    @property
    def applied_torque(self) -> torch.Tensor:
        """Clipped PD torque retained immediately before the last physics substep [N m]."""
        return self._applied_torque


def _joint_position_action(env: ManagerBasedRLEnv, action_name: str) -> G1JointPositionAction:
    """Return the G1 controller that owns normalized actions and PD torques."""
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, G1JointPositionAction):
        raise TypeError("G1 controller evidence requires G1JointPositionAction.")
    return action


def controller_torques_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return the squared norm of the last clipped PD torques [N² m²]."""
    torque = _joint_position_action(env, action_name).applied_torque
    return torque.square().sum(dim=-1)


def controller_action_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return the squared change in controller-normalized actions."""
    action = _joint_position_action(env, action_name)
    return (action.processed_actions - action.previous_processed_actions).square().sum(dim=-1)


def controller_torque_limits(env: ManagerBasedRLEnv, action_name: str, soft_ratio: float) -> torch.Tensor:
    """Return clipped PD torque above a fraction of the effort limits [N m]."""
    action = _joint_position_action(env, action_name)
    excess = action.applied_torque.abs() - soft_ratio * action.joint_effort_limit
    return excess.clamp_min_(0.0).sum(dim=-1)
