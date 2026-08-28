# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native assembly transforms, completion, and validity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.contrib.nist.assembly_profile import AssemblyProfile

from ..newton_selection import NewtonBodySelectorCfg
from .reset import board_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg


@wp.kernel(enable_backward=False)
def _update_assembly_state(
    body_q: wp.array(dtype=wp.transformf),
    held_body_ids: wp.array2d(dtype=wp.int32),
    board_body_ids: wp.array2d(dtype=wp.int32),
    robot_root_body_ids: wp.array2d(dtype=wp.int32),
    variant_ids: wp.array2d(dtype=wp.uint8),
    unfinished_count: wp.array(dtype=wp.uint8),
    required_assembly_gain: wp.array(dtype=wp.uint8),
    env_origins: wp.array(dtype=wp.vec3f),
    held_align: wp.array(dtype=wp.transformf),
    fixed_tip_in_board: wp.array(dtype=wp.transformf),
    goal_in_board: wp.array(dtype=wp.transformf),
    num_variants: int,
    num_slots: int,
    num_fixed: int,
    assembly_contact_forces: wp.array2d(dtype=wp.vec3f),
    contact_force_threshold_sq: float,
    success_threshold: float,
    lower: wp.vec3f,
    upper: wp.vec3f,
    assembly_frames: wp.array2d(dtype=wp.transformf),
    variant_active: wp.array2d(dtype=wp.float32),
    asset_assembled: wp.array2d(dtype=wp.bool),
    all_success: wp.array(dtype=wp.bool),
    task_success: wp.array(dtype=wp.bool),
    assembly_contact_force_exceeded: wp.array(dtype=wp.bool),
    any_held_asset_out_of_bound: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    count = int(0)
    excessive_contact = bool(False)
    outside = bool(False)
    root_inverse = wp.transform_inverse(body_q[robot_root_body_ids[world, 0]])
    board_root = body_q[board_body_ids[world, 0]]
    for variant in range(num_variants):
        tip_frame = wp.transform_multiply(board_root, fixed_tip_in_board[variant])
        assembly_frames[world, 2 * variant + 1] = wp.transform_multiply(root_inverse, tip_frame)
        assembly_frames[world, 2 * variant] = wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 0.0))
        variant_active[world, variant] = 0.0
        asset_assembled[world, variant] = False

    for slot in range(num_slots):
        variant = int(variant_ids[world, slot])
        variant_active[world, variant] = 1.0
        held_root = body_q[held_body_ids[world, slot]]
        held_frame = wp.transform_multiply(held_root, held_align[variant])
        goal_frame = wp.transform_multiply(board_root, goal_in_board[variant])
        assembly_frames[world, 2 * variant] = wp.transform_multiply(root_inverse, held_frame)

        error = wp.transform_multiply(wp.transform_inverse(goal_frame), held_frame)
        position = wp.transform_get_translation(error)
        rotation = wp.transform_get_rotation(error)
        roll = wp.atan2(
            2.0 * (rotation[3] * rotation[0] + rotation[1] * rotation[2]),
            1.0 - 2.0 * (rotation[0] * rotation[0] + rotation[1] * rotation[1]),
        )
        pitch = wp.asin(wp.clamp(2.0 * (rotation[3] * rotation[1] - rotation[2] * rotation[0]), -1.0, 1.0))
        centered = wp.sqrt(position[0] * position[0] + position[1] * position[1]) < 0.0025
        success = wp.abs(roll) + wp.abs(pitch) < 0.025 and centered and position[2] < success_threshold
        asset_assembled[world, variant] = success
        if success:
            count += 1

        contact_row = world * num_slots + slot
        for fixed in range(num_fixed):
            force = assembly_contact_forces[contact_row, fixed]
            excessive_contact = excessive_contact or wp.length_sq(force) > contact_force_threshold_sq

        root_position = wp.transform_get_translation(held_root) - env_origins[world]
        finite = wp.isfinite(root_position[0]) and wp.isfinite(root_position[1]) and wp.isfinite(root_position[2])
        outside = outside or not finite
        outside = outside or root_position[0] < lower[0] or root_position[0] > upper[0]
        outside = outside or root_position[1] < lower[1] or root_position[1] > upper[1]
        outside = outside or root_position[2] < lower[2] or root_position[2] > upper[2]

    target_count = num_slots - int(unfinished_count[world]) + int(required_assembly_gain[world])
    all_success[world] = count == num_slots
    task_success[world] = count >= target_count
    assembly_contact_force_exceeded[world] = excessive_contact
    any_held_asset_out_of_bound[world] = outside


@wp.kernel(enable_backward=False)
def _gather_held_asset_in_fixed_asset_frame(
    body_q: wp.array(dtype=wp.transformf),
    held_body_ids: wp.array2d(dtype=wp.int32),
    fixed_body_ids: wp.array2d(dtype=wp.int32),
    fixed_kind_by_slot: wp.array2d(dtype=wp.int32),
    fixture_index_by_variant: wp.array(dtype=wp.int32),
    variant_ids: wp.array2d(dtype=wp.uint8),
    held_align: wp.array(dtype=wp.transformf),
    fixed_tip: wp.array(dtype=wp.transformf),
    num_fixed: int,
    output: wp.array2d(dtype=wp.transformf),
):
    world, slot = wp.tid()
    variant = int(variant_ids[world, slot])
    fixture = fixture_index_by_variant[variant]
    fixed_slot = int(0)
    for candidate in range(num_fixed):
        if fixed_kind_by_slot[world, candidate] == fixture:
            fixed_slot = candidate
    held = wp.transform_multiply(body_q[held_body_ids[world, slot]], held_align[variant])
    fixed = wp.transform_multiply(body_q[fixed_body_ids[world, fixed_slot]], fixed_tip[variant])
    output[world, slot] = wp.transform_multiply(wp.transform_inverse(fixed), held)


class AssemblyState(ManagerTermBase):
    """Compute all assembly frames, completion, and validity in one refresh."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset = env.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("AssemblyState requires the resolved board reset term.")
        self._reset = reset
        layout = reset.layout
        self._num_variants = layout.num_variants
        self._num_slots = layout.num_slots

        model = NewtonManager.get_model()
        held = cfg.params["held_bodies"].resolve(model)
        fixed = cfg.params["fixed_bodies"].resolve(model)
        board = cfg.params["board_body"].resolve(model)
        robot_root = cfg.params["robot_root_body"].resolve(model)
        contact_sensor_cfg: SceneEntityCfg = cfg.params["contact_sensor_cfg"]
        contact_view = env.scene.sensors[contact_sensor_cfg.name].contact_view
        held_shape = (env.num_envs, self._num_slots)
        fixed_shape = (env.num_envs, layout.num_fixed_slots)
        if held.shape != held_shape or fixed.shape != fixed_shape:
            raise ValueError(
                f"Assembly body selectors must resolve to {held_shape} and {fixed_shape}, "
                f"got {held.shape} and {fixed.shape}."
            )
        root_shape = (env.num_envs, 1)
        if board.shape != root_shape or robot_root.shape != root_shape:
            raise ValueError(
                f"Board and robot root selectors must resolve to {root_shape}, got {board.shape} and "
                f"{robot_root.shape}."
            )

        held_ids = np.asarray(held.ids, dtype=np.int32)
        fixed_ids = np.asarray(fixed.ids, dtype=np.int32)
        sensing_ids = np.asarray(contact_view.sensing_indices, dtype=np.int32)
        self._assembly_contact_forces = contact_view.force_matrix
        if self._assembly_contact_forces is None:
            raise RuntimeError("Assembly contact sensor does not expose filtered forces.")

        sensing_shape = (env.num_envs * self._num_slots,)
        if sensing_ids.shape != sensing_shape or not np.array_equal(sensing_ids.reshape(held_shape), held_ids):
            raise ValueError("Assembly contact sensor rows must follow held-asset slot order.")
        num_fixed = layout.num_fixed_slots
        counterpart_shape = (env.num_envs, self._num_slots, num_fixed)
        try:
            flat_counterpart_ids = np.asarray(contact_view.counterpart_indices, dtype=np.int32)
        except ValueError as error:
            raise ValueError(
                "Assembly contact sensor must expose every selected fixed fixture for every held-asset row."
            ) from error
        if flat_counterpart_ids.shape != (sensing_shape[0], num_fixed):
            raise ValueError(
                "Assembly contact sensor must expose every selected fixed fixture for every held-asset row."
            )
        counterpart_ids = flat_counterpart_ids.reshape(counterpart_shape)
        if tuple(self._assembly_contact_forces.shape) != (sensing_shape[0], num_fixed):
            raise ValueError("Assembly contact sensor force matrix does not match the selected board layout.")

        expected_counterparts = np.broadcast_to(fixed_ids[:, None, :], counterpart_shape)
        if not np.array_equal(np.sort(counterpart_ids, axis=2), np.sort(expected_counterparts, axis=2)):
            raise ValueError("Assembly contact sensor counterparts do not match the selected fixed fixtures.")

        self._held_body_ids = wp.array(held.ids, dtype=wp.int32, device=env.device)
        self._fixed_body_ids = wp.array(fixed.ids, dtype=wp.int32, device=env.device)
        self._board_body_ids = wp.array(board.ids, dtype=wp.int32, device=env.device)
        self._robot_root_body_ids = wp.array(robot_root.ids, dtype=wp.int32, device=env.device)
        self._variant_ids = wp.from_torch(reset.variant_ids, dtype=wp.uint8)
        self._unfinished_count = wp.from_torch(reset.unfinished_count, dtype=wp.uint8)
        self._required_assembly_gain = wp.from_torch(reset.required_assembly_gain, dtype=wp.uint8)
        self._fixed_kind_by_slot = wp.from_torch(reset.fixed_kind_by_slot)
        self._fixture_index_by_variant = wp.array(layout.fixture_index_by_variant, dtype=wp.int32, device=env.device)
        self._env_origins = wp.from_torch(env.scene.env_origins.contiguous(), dtype=wp.vec3f)
        self._held_align = self._offsets([variant.held_align.pose for variant in layout.variants], env.device)
        self._fixed_tip = self._offsets([variant.fixed_tip.pose for variant in layout.variants], env.device)
        board_offsets = [wp.transformf(*variant.board_offset.pose) for variant in layout.variants]
        self._fixed_tip_in_board = wp.array(
            [
                wp.transform_multiply(board_offset, wp.transformf(*variant.fixed_tip.pose))
                for board_offset, variant in zip(board_offsets, layout.variants, strict=True)
            ],
            dtype=wp.transformf,
            device=env.device,
        )
        self._goal_in_board = wp.array(
            [
                wp.transform_multiply(
                    board_offset, wp.transformf(*AssemblyProfile(variant.profile).assembled_offset.pose)
                )
                for board_offset, variant in zip(board_offsets, layout.variants, strict=True)
            ],
            dtype=wp.transformf,
            device=env.device,
        )
        self._num_fixed = num_fixed
        self._contact_force_threshold_sq = float(cfg.params["contact_force_threshold"]) ** 2
        self._success_threshold = float(cfg.params["success_threshold"])
        bounds = cfg.params["workspace"]
        self._lower = wp.vec3f(*(bounds[axis][0] for axis in ("x", "y", "z")))
        self._upper = wp.vec3f(*(bounds[axis][1] for axis in ("x", "y", "z")))

        self._assembly_frames = wp.empty((env.num_envs, 2 * self._num_variants), dtype=wp.transformf, device=env.device)
        self._held_asset_in_fixed_asset_frame = wp.empty(
            (env.num_envs, self._num_slots), dtype=wp.transformf, device=env.device
        )
        self._variant_active = wp.empty((env.num_envs, self._num_variants), dtype=wp.float32, device=env.device)
        self._asset_assembled = wp.empty((env.num_envs, self._num_variants), dtype=wp.bool, device=env.device)
        self._all_success = wp.empty(env.num_envs, dtype=wp.bool, device=env.device)
        self._task_success = wp.empty(env.num_envs, dtype=wp.bool, device=env.device)
        self._assembly_contact_force_exceeded = wp.empty(env.num_envs, dtype=wp.bool, device=env.device)
        self._any_held_asset_out_of_bound = wp.empty(env.num_envs, dtype=wp.bool, device=env.device)
        self._assembly_frames_torch = wp.to_torch(self._assembly_frames).view(env.num_envs, -1)
        self._held_asset_in_fixed_asset_frame_torch = wp.to_torch(self._held_asset_in_fixed_asset_frame).view(
            env.num_envs, -1
        )
        self._variant_active_torch = wp.to_torch(self._variant_active)
        self._asset_assembled_torch = wp.to_torch(self._asset_assembled)
        self._all_success_torch = wp.to_torch(self._all_success)
        self._task_success_torch = wp.to_torch(self._task_success)
        self._assembly_contact_force_exceeded_torch = wp.to_torch(self._assembly_contact_force_exceeded)
        self._any_held_asset_out_of_bound_torch = wp.to_torch(self._any_held_asset_out_of_bound)
        self._stamp = (-1, -1)

    @staticmethod
    def _offsets(poses: list[tuple[float, ...]], device: str) -> wp.array:
        return wp.array([wp.transformf(*pose) for pose in poses], dtype=wp.transformf, device=device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        held_bodies: NewtonBodySelectorCfg,
        fixed_bodies: NewtonBodySelectorCfg,
        board_body: NewtonBodySelectorCfg,
        robot_root_body: NewtonBodySelectorCfg,
        contact_sensor_cfg: SceneEntityCfg,
        contact_force_threshold: float,
        success_threshold: float,
        workspace: dict[str, tuple[float, float]],
    ) -> None:
        self._stamp = (-1, -1)

    @property
    def assembly_frames(self) -> torch.Tensor:
        """Canonical held-align and fixed-tip poses in the robot root frame."""
        self._refresh()
        return self._assembly_frames_torch

    @property
    def held_asset_in_fixed_asset_frame(self) -> torch.Tensor:
        """Slot-ordered held-align poses in their matching fixed-tip frames."""
        self._refresh()
        return self._held_asset_in_fixed_asset_frame_torch

    @property
    def all_success(self) -> torch.Tensor:
        """Whether every active assembly is complete."""
        self._refresh()
        return self._all_success_torch

    @property
    def task_success(self) -> torch.Tensor:
        """Whether the episode's required net assembly gain is complete."""
        self._refresh()
        return self._task_success_torch

    @property
    def variant_active(self) -> torch.Tensor:
        """Active assembly variants as a zero-copy float mask."""
        self._refresh()
        return self._variant_active_torch

    @property
    def asset_assembled(self) -> torch.Tensor:
        """Whether each active canonical assembly is complete; inactive entries are false."""
        self._refresh()
        return self._asset_assembled_torch

    @property
    def assembly_contact_force_exceeded(self) -> torch.Tensor:
        """Whether a held asset exceeds its matching fixture's force limit."""
        self._refresh()
        return self._assembly_contact_force_exceeded_torch

    @property
    def any_held_asset_out_of_bound(self) -> torch.Tensor:
        """Whether any held asset is outside the workspace."""
        self._refresh()
        return self._any_held_asset_out_of_bound_torch

    def _refresh(self) -> None:
        stamp = (self._env.common_step_counter, self._reset.revision)
        if stamp == self._stamp:
            return
        wp.launch(
            _update_assembly_state,
            dim=self.num_envs,
            inputs=[
                NewtonManager.get_state_0().body_q,
                self._held_body_ids,
                self._board_body_ids,
                self._robot_root_body_ids,
                self._variant_ids,
                self._unfinished_count,
                self._required_assembly_gain,
                self._env_origins,
                self._held_align,
                self._fixed_tip_in_board,
                self._goal_in_board,
                self._num_variants,
                self._num_slots,
                self._num_fixed,
                self._assembly_contact_forces,
                self._contact_force_threshold_sq,
                self._success_threshold,
                self._lower,
                self._upper,
            ],
            outputs=[
                self._assembly_frames,
                self._variant_active,
                self._asset_assembled,
                self._all_success,
                self._task_success,
                self._assembly_contact_force_exceeded,
                self._any_held_asset_out_of_bound,
            ],
            device=self.device,
        )
        wp.launch(
            _gather_held_asset_in_fixed_asset_frame,
            dim=(self.num_envs, self._num_slots),
            inputs=[
                NewtonManager.get_state_0().body_q,
                self._held_body_ids,
                self._fixed_body_ids,
                self._fixed_kind_by_slot,
                self._fixture_index_by_variant,
                self._variant_ids,
                self._held_align,
                self._fixed_tip,
                self._num_fixed,
            ],
            outputs=[self._held_asset_in_fixed_asset_frame],
            device=self.device,
        )
        self._stamp = stamp


def _assembly_state(env: ManagerBasedRLEnv) -> AssemblyState:
    return env.event_manager.get_term_cfg("assembly_state").func


def assembly_frames_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return canonical held-align and fixed-tip poses in the robot root frame.

    Fixed-tip frames are always populated. Held-align frames are all-zero for inactive variants.
    """
    return _assembly_state(env).assembly_frames


def held_asset_in_fixed_asset_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return slot-ordered held-align poses in matching fixed-tip frames."""
    return _assembly_state(env).held_asset_in_fixed_asset_frame


def assembly_variant_active_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the active canonical assembly variants as a float mask."""
    return _assembly_state(env).variant_active


def any_held_asset_out_of_bound(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return environments where any held asset left the workspace."""
    return _assembly_state(env).any_held_asset_out_of_bound


def assembly_contact_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return environments where a held asset overloads its matching fixture."""
    return _assembly_state(env).assembly_contact_force_exceeded


class assembly_progress_context(ManagerTermBase):
    """Expose the active assembly goal through Factory's progress-context contract."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._state = _assembly_state(env)
        self._dummy = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._terminal_success = torch.zeros_like(self._dummy)

    @property
    def is_success(self) -> torch.Tensor:
        """Success state captured during the latest termination evaluation."""
        return self._terminal_success

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._terminal_success.copy_(self._state.task_success)
        env.extras["successes"] = self._terminal_success
        return self._dummy


def assembly_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward environments that complete their active assembly goal."""
    return _assembly_state(env).task_success.float()
