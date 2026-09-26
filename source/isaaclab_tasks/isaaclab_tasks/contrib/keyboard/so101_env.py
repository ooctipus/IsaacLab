# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composition root for the Newton SO101 keyboard task."""

from dataclasses import fields, is_dataclass

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import JointTargetMode, JointType, ModelFlags

from isaaclab.envs import ManagerBasedRLEnv

from .newton_selection import NewtonSelections, NewtonSelectorCfg


class SO101KeyboardEnv(ManagerBasedRLEnv):
    """Bind task selectors after model finalization and before MDP construction."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg.copy(), render_mode=render_mode, **kwargs)

    def load_managers(self):
        self.selections = NewtonSelections(NewtonManager.get_model())

        def bind(value):
            if isinstance(value, NewtonSelectorCfg):
                return self.selections.resolve(value)
            if isinstance(value, dict):
                for key, item in value.items():
                    value[key] = bind(item)
            elif isinstance(value, (tuple, list)):
                return type(value)(bind(item) for item in value)
            elif is_dataclass(value):
                for field in fields(value):
                    setattr(value, field.name, bind(getattr(value, field.name)))
            return value

        for cfg in (
            self.cfg.commands,
            self.cfg.actions,
            self.cfg.observations,
            self.cfg.rewards,
            self.cfg.terminations,
            self.event_manager.cfg,
        ):
            bind(cfg)
        # Authored robot USD can carry a calibrated pose. The task's reset seed is zero.
        state = NewtonManager.get_state()
        robot = self.cfg.actions.action.joints
        wp.to_torch(state.joint_q)[robot.dense_ids()] = 0.0
        wp.to_torch(state.joint_qd).zero_()
        model = NewtonManager.get_model()
        root_ids = self.selections.root_joint_ids[self.cfg.commands.typing.reset_roots.dense_ids()]
        if torch.any(root_ids < 0) or torch.any(wp.to_torch(model.joint_type)[root_ids] != int(JointType.FIXED)):
            raise ValueError("Keyboard reset snapshots require fixed articulation roots.")
        wp.to_torch(model.joint_target_mode)[self.cfg.actions.action.dofs.dense_ids()] = int(
            JointTargetMode.POSITION_VELOCITY
        )
        NewtonManager.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        NewtonManager.invalidate_fk()
        self.sim.forward()
        self._property_world_mask = wp.zeros(self.num_envs + 1, dtype=wp.bool, device=self.device)
        super().load_managers()
