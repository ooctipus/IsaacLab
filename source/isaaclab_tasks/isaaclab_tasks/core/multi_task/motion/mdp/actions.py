# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-control action terms for native and position-target backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import MotionJointPositionActionCfg, MotionMujocoControlActionCfg


class MotionJointPositionAction(ActionTerm):
    """Apply normalized behavior actions through the native position-control law.

    The processed action retains the controller-normalized value used by
    history and action-rate evidence. The articulation target is held in the
    separate joint-position-target tensor.
    """

    cfg: MotionJointPositionActionCfg

    def __init__(self, cfg: MotionJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        if not joint_ids:
            raise ValueError("MotionJointPositionAction must resolve at least one articulation joint.")
        self._num_joints = len(joint_ids)
        self._joint_names = tuple(joint_names)
        self._joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.int64, device=self.device)
        if self._num_joints == self._asset.num_joints and joint_ids == list(range(self._asset.num_joints)):
            self._joint_ids: slice | list[int] = slice(None)
        else:
            self._joint_ids = joint_ids

        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_position = torch.empty_like(self._raw_actions)
        self._joint_velocity = torch.empty_like(self._raw_actions)
        data = self._asset.data
        joint_ids = self._joint_ids
        self.joint_default_position = data.default_joint_pos.torch[0, joint_ids].clone()
        self.joint_stiffness = data.joint_stiffness.torch[0, joint_ids].clone()
        self.joint_damping = data.joint_damping.torch[0, joint_ids].clone()
        self.joint_effort_limit = data.joint_effort_limits.torch[0, joint_ids].clone()
        self.default_joint_offset = torch.zeros_like(self._processed_actions)
        self.joint_position_target = torch.empty_like(self._processed_actions)
        self._reset_default_joint_offset = torch.empty_like(self._processed_actions)
        self._reset_joint_position_target = torch.empty_like(self._processed_actions)
        self._applied_torque = torch.zeros_like(self._processed_actions)
        self.joint_target_gain = cfg.action_scale * self.joint_effort_limit / self.joint_stiffness

        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self._raw_actions.dtype)
        self._IO_descriptor.action_type = "JointPosition"
        self._IO_descriptor.extras = {
            "joint_names": list(self._joint_names),
            "normalize_to": cfg.normalize_to,
            "processed_action_clip": (-cfg.action_clip, cfg.action_clip),
            "effort_over_stiffness_scale": cfg.action_scale,
            "default_joint_offset_range_rad": cfg.default_joint_offset_range,
        }

    @property
    def action_dim(self) -> int:
        """Number of controlled articulation joints."""
        return self._num_joints

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names in the declared behavior-action order."""
        return self._joint_names

    @property
    def joint_ids(self) -> torch.Tensor:
        """Live-articulation joint indices in behavior-action order."""
        return self._joint_ids_tensor

    @property
    def joint_position(self) -> torch.Tensor:
        """Joint positions in behavior-action order [rad]."""
        torch.index_select(self._asset.data.joint_pos.torch, 1, self._joint_ids_tensor, out=self._joint_position)
        return self._joint_position

    @property
    def joint_velocity(self) -> torch.Tensor:
        """Joint velocities in behavior-action order [rad/s]."""
        torch.index_select(self._asset.data.joint_vel.torch, 1, self._joint_ids_tensor, out=self._joint_velocity)
        return self._joint_velocity

    @property
    def raw_actions(self) -> torch.Tensor:
        """Dimensionless behavior actions before native normalization."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Controller-normalized actions after the native symmetric clamp."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Normalize behavior actions and materialize joint targets [rad]."""
        self._raw_actions.copy_(actions)
        torch.mul(self._raw_actions, self.cfg.normalize_to, out=self._processed_actions)
        self._processed_actions.clamp_(-self.cfg.action_clip, self.cfg.action_clip)
        torch.mul(self._processed_actions, self.joint_target_gain, out=self.joint_position_target)
        self.joint_position_target.add_(self.joint_default_position)
        self.joint_position_target.add_(self.default_joint_offset)

    def apply_actions(self) -> None:
        """Write the persistent joint-position target [rad] to the articulation."""
        joint_position = self.joint_position
        joint_velocity = self.joint_velocity
        torch.sub(
            self.joint_position_target,
            joint_position,
            out=self._applied_torque,
        )
        self._applied_torque.mul_(self.joint_stiffness)
        self._applied_torque.addcmul_(
            self.joint_damping,
            joint_velocity,
            value=-1.0,
        )
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
            self._applied_torque.zero_()
            offset = self.default_joint_offset
            offset.uniform_(*self.cfg.default_joint_offset_range)
            torch.add(self.joint_default_position, offset, out=self.joint_position_target)
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._applied_torque[env_ids] = 0.0
            count = len(env_ids)
            offset = self._reset_default_joint_offset[:count]
            target = self._reset_joint_position_target[:count]
            offset.uniform_(*self.cfg.default_joint_offset_range)
            torch.add(self.joint_default_position, offset, out=target)
            self.default_joint_offset[env_ids] = offset
            self.joint_position_target[env_ids] = target

    @property
    def applied_torque(self) -> torch.Tensor:
        """Clipped PD torque retained immediately before the last physics substep [N m]."""
        return self._applied_torque


class MotionMujocoControlAction(ActionTerm):
    """Write raw behavior actions to one native MuJoCo control vector."""

    cfg: MotionMujocoControlActionCfg

    def __init__(self, cfg: MotionMujocoControlActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._actions = torch.zeros(self.num_envs, cfg.action_width, dtype=torch.float32, device=self.device)
        self._control_source = wp.from_torch(self._actions.view(-1), dtype=wp.float32)

        control = env.sim.physics_manager.get_control()
        namespace = getattr(control, "mujoco", None)
        destination = getattr(namespace, "ctrl", None)
        if destination is None:
            raise RuntimeError("The selected simulator does not expose a native MuJoCo control input.")
        if destination.dtype != wp.float32:
            raise TypeError("Native MuJoCo control inputs must use float32 storage.")
        if destination.shape != self._control_source.shape:
            raise ValueError(
                f"Native MuJoCo control input shape {destination.shape} differs from {self._control_source.shape}."
            )
        if destination.device != self._control_source.device:
            raise ValueError(
                f"Native MuJoCo control input device {destination.device} differs from {self._control_source.device}."
            )
        self._control_destination = destination
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self._actions.dtype)
        self._IO_descriptor.action_type = "NativeMujocoControl"
        self._IO_descriptor.extras = {"control_width": self.action_dim}

    @property
    def action_dim(self) -> int:
        """Return the native control width."""
        return self.cfg.action_width

    @property
    def raw_actions(self) -> torch.Tensor:
        """Return raw behavior actions."""
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Return the identity-processed native control actions."""
        return self._actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Copy one environment-step behavior action into persistent storage."""
        self._actions.copy_(actions)

    def apply_actions(self) -> None:
        """Enqueue the persistent control vector copy without allocation or sync."""
        wp.copy(self._control_destination, self._control_source)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear native control rows for reset environments."""
        if env_ids is None:
            self._actions.zero_()
        else:
            self._actions[env_ids] = 0.0


__all__ = ["MotionJointPositionAction", "MotionMujocoControlAction"]
