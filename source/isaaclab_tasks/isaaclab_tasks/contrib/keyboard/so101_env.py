# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composition root for the Newton SO101 keyboard task."""

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import JointTargetMode, JointType, ModelFlags

from isaaclab.envs import ManagerBasedRLEnv

from .keyboard_variants import KeyboardVariants
from .newton_selection import NewtonSelections, bind_selectors


class SO101KeyboardEnv(ManagerBasedRLEnv):
    """Bind task selectors after model finalization and before MDP construction."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg.copy(), render_mode=render_mode, **kwargs)

    def load_managers(self):
        self.episode_interrupted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.selections = NewtonSelections(
            NewtonManager.get_model(), state=NewtonManager.get_state(), control=NewtonManager.get_control()
        )

        for cfg in (
            self.cfg.commands,
            self.cfg.actions,
            self.cfg.observations,
            self.cfg.rewards,
            self.cfg.terminations,
            self.event_manager.cfg,
        ):
            bind_selectors(cfg, self.selections.resolve)
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
        self.keyboard_variants = (
            KeyboardVariants(self, self.cfg.keyboard_variants) if self.cfg.keyboard_variants else None
        )
        super().load_managers()

    @property
    def all_env_ids(self):
        """Stable logical actor IDs used by the task's MDP."""
        return self.scene._ALL_INDICES

    @property
    def env_origins(self):
        """Physical world origins for portable reset snapshots."""
        return self.scene.env_origins

    def forward(self):
        """Reconcile task-authored state and update native forward kinematics."""
        self.sim.forward()

    def reset_variant_ids(self, env_ids):
        """Return the committed keyboard variants used by snapshot sampling."""
        return self.keyboard_variants.variant_ids[env_ids]

    def restore_reset_snapshot(self, env_ids, variant_ids, snapshot):
        """Restore physical snapshot data before the command publishes its typing state."""
        from .mdp.reset import restore_reset_state

        if self.keyboard_variants is not None:
            torch._assert_async(
                (variant_ids == self.keyboard_variants.variant_ids[env_ids]).all(),
                "Snapshot keyboard differs from the committed variant.",
            )
        command = self.cfg.commands.typing
        restore_reset_state(self, snapshot, env_ids, command.reset_roots, command.reset_coords, command.reset_dofs)

    def curriculum_worlds(self, variant):
        """Provide scratch worlds of one keyboard variant during initial curriculum construction."""
        if self.keyboard_variants is not None:
            self.keyboard_variants.apply(self.all_env_ids, torch.full_like(self.all_env_ids, variant))
        return self.all_env_ids

    def finish_curriculum(self, original_variants):
        """Restore keyboard assignments after constructing reset snapshots."""
        if self.keyboard_variants is not None:
            self.keyboard_variants.apply(self.all_env_ids, original_variants)

    def _reset_idx(self, env_ids, *, variant_ids=None):
        ids = self.scene._ALL_INDICES[env_ids]
        if self.keyboard_variants is not None:
            if variant_ids is None:
                count = len(self.keyboard_variants.layouts)
                # A nonzero cyclic offset excludes the previous variant; a one-entry bank stays at zero.
                offsets = torch.randint(1, max(count, 2), ids.shape, device=self.device)
                variants = (self.keyboard_variants.variant_ids[ids] + offsets) % count
            else:
                variants = torch.as_tensor(variant_ids, dtype=torch.long, device=self.device)
            self.keyboard_variants.apply(ids, variants)
        super()._reset_idx(ids)

    def invalidate_fk(self, env_ids) -> None:
        """Mark task-authored coordinates for reconciliation before native state reads."""
        NewtonManager.invalidate_fk(env_ids=wp.from_torch(env_ids.to(torch.int32)))

    def notify_model_changed(self, flags, env_ids, *, root_poses_only=False) -> None:
        """Synchronize task-authored model properties in the selected logical worlds."""
        self._property_world_mask.zero_()
        wp.to_torch(self._property_world_mask)[env_ids] = True
        NewtonManager.notify_model_changed(flags, world_mask=self._property_world_mask, root_poses_only=root_poses_only)

    def reset_keyboard(self, env_ids, variant_ids) -> None:
        """Reset selected episodes to registered keyboard variants, then reconcile forward kinematics."""
        if self.keyboard_variants is None:
            raise RuntimeError("This configuration has no registered keyboard variants.")
        self._reset_idx(env_ids, variant_ids=variant_ids)
        self.sim.forward()
