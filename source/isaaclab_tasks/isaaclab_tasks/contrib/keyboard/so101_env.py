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
from isaaclab.utils import clone
from isaaclab.utils.warp.utils import warp_on_torch_stream

from .keyboard_variants import KeyboardVariants
from .newton_selection import NewtonSelections
from .selection_paths import bind_selectors, resolve_selection


class SO101KeyboardEnv(ManagerBasedRLEnv):
    """Bind task selectors after model finalization and before MDP construction."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        with warp_on_torch_stream(cfg.sim.device):
            super().__init__(clone(cfg), render_mode=render_mode, **kwargs)

    def reset(self, env_ids=slice(None), *, seed=None, options=None):
        """Reset episodes with physics and task work ordered on the caller's stream."""
        with warp_on_torch_stream(self.device):
            return super().reset(env_ids, seed=seed, options=options)

    def step(self, action):
        """Advance actions, physics and the MDP in one ordered stream scope."""
        with warp_on_torch_stream(self.device):
            return super().step(action)

    def reset_to(self, state, env_ids=slice(None), seed=None, is_relative=False):
        """Restore an explicit scene state with the same ordering as ordinary resets."""
        with warp_on_torch_stream(self.device):
            return super().reset_to(state, env_ids, seed, is_relative)

    def load_managers(self):
        self.episode_interrupted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.selections = NewtonSelections(
            NewtonManager.get_model(), state=NewtonManager.get_state_0(), control=NewtonManager.get_control()
        )

        authored_cfg = self.cfg
        self.cfg = clone(authored_cfg)
        try:
            for cfg in (
                self.cfg.commands,
                self.cfg.actions,
                self.cfg.observations,
                self.cfg.rewards,
                self.cfg.terminations,
                self.event_manager.cfg,
            ):
                bind_selectors(cfg, lambda query: resolve_selection(self.selections, query))
            # Task defaults own the initial pose and the matching actuator targets.
            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            control = NewtonManager.get_control()
            robot_q, robot_qd = self.cfg.actions.action.joints, self.cfg.actions.action.dofs
            positions = wp.array(
                [
                    self.cfg.robot_joint_positions[model.joint_label[joint].rsplit("/", 1)[-1]]
                    for joint in robot_q.joint_ids.numpy()
                ],
                dtype=wp.float32,
                device=model.device,
            )
            wp.indexedarray(model.joint_q, robot_q.ids).assign(positions)
            wp.indexedarray(state.joint_q, robot_q.ids).assign(positions)
            model.joint_qd.zero_()
            state.joint_qd.zero_()
            targets = robot_q if model.use_coord_layout_targets else robot_qd
            wp.indexedarray(model.joint_target_q, targets.ids).assign(positions)
            wp.indexedarray(control.joint_target_q, targets.ids).assign(positions)
            model.joint_target_qd.zero_()
            control.joint_target_qd.zero_()
            root_ids = self.selections.root_joint_ids[self.cfg.commands.typing.reset_roots.dense_ids()]
            if torch.any(root_ids < 0) or torch.any(wp.to_torch(model.joint_type)[root_ids] != int(JointType.FIXED)):
                raise ValueError("Keyboard reset snapshots require fixed articulation roots.")
            wp.indexedarray(model.joint_target_mode, robot_qd.ids).fill_(int(JointTargetMode.POSITION_VELOCITY))
            NewtonManager.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            NewtonManager.invalidate_fk()
            self.sim.forward()
            self._property_world_mask = wp.zeros(self.num_envs + 1, dtype=wp.bool, device=self.device)
            self.keyboard_variants = (
                KeyboardVariants(self, self.cfg.keyboard_variants) if self.cfg.keyboard_variants else None
            )
            super().load_managers()
        finally:
            # Managers retain their bound copies; reproducibility uses authored selectors.
            self.cfg = authored_cfg

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
        with warp_on_torch_stream(self.device):
            self.sim.forward()

    def _compute_final_observations(self):
        """Preview terminal observation history from the completed typing transition."""
        with self.command_manager.get_term("typing").preview_housekeeping(self.step_dt):
            return self.observation_manager.preview()

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
        command = self.command_manager.get_term("typing").cfg
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
        with warp_on_torch_stream(self.device):
            self._reset_idx(env_ids, variant_ids=variant_ids)
            self.sim.forward()
