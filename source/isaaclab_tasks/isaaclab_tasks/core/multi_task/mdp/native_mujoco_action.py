# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native MuJoCo control action shared by manager-based environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .native_mujoco_action_cfg import NativeMujocoControlActionCfg


class NativeMujocoControlAction(ActionTerm):
    """Write raw behavior actions to one native MuJoCo control vector."""

    cfg: NativeMujocoControlActionCfg

    def __init__(self, cfg: NativeMujocoControlActionCfg, env: ManagerBasedEnv) -> None:
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
