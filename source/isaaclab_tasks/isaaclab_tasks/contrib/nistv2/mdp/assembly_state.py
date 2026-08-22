# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native assembly transforms, completion, and validity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_tasks.contrib.nist.assembly_profile import AssemblyProfile
from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANTS

from ..board_layout import NUM_ASSEMBLIES
from ..newton_selection import NewtonBodySelectorCfg
from .reset import board_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg


@wp.kernel(enable_backward=False)
def _update_assembly_state(
    body_q: wp.array(dtype=wp.transformf),
    held_body_ids: wp.array2d(dtype=wp.int32),
    fixed_body_ids: wp.array2d(dtype=wp.int32),
    robot_root_body_ids: wp.array2d(dtype=wp.int32),
    variant_ids: wp.array2d(dtype=wp.uint8),
    env_origins: wp.array(dtype=wp.vec3f),
    held_align: wp.array(dtype=wp.transformf),
    fixed_tip: wp.array(dtype=wp.transformf),
    assembled_offsets: wp.array(dtype=wp.transformf),
    lower: wp.vec3f,
    upper: wp.vec3f,
    assembly_frames: wp.array2d(dtype=wp.transformf),
    asset_assembled: wp.array2d(dtype=wp.bool),
    all_success: wp.array(dtype=wp.bool),
    any_held_asset_out_of_bound: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    count = int(0)
    outside = bool(False)
    root_inverse = wp.transform_inverse(body_q[robot_root_body_ids[world, 0]])
    for slot in range(NUM_ASSEMBLIES):
        variant = int(variant_ids[world, slot])
        held_root = body_q[held_body_ids[world, slot]]
        fixed_root = body_q[fixed_body_ids[world, variant]]
        held_frame = wp.transform_multiply(held_root, held_align[variant])
        tip_frame = wp.transform_multiply(fixed_root, fixed_tip[variant])
        goal_frame = wp.transform_multiply(fixed_root, assembled_offsets[variant])
        assembly_frames[world, 2 * variant] = wp.transform_multiply(root_inverse, held_frame)
        assembly_frames[world, 2 * variant + 1] = wp.transform_multiply(root_inverse, tip_frame)

        error = wp.transform_multiply(wp.transform_inverse(goal_frame), held_frame)
        position = wp.transform_get_translation(error)
        rotation = wp.transform_get_rotation(error)
        roll = wp.atan2(
            2.0 * (rotation[3] * rotation[0] + rotation[1] * rotation[2]),
            1.0 - 2.0 * (rotation[0] * rotation[0] + rotation[1] * rotation[1]),
        )
        pitch = wp.asin(wp.clamp(2.0 * (rotation[3] * rotation[1] - rotation[2] * rotation[0]), -1.0, 1.0))
        centered = wp.sqrt(position[0] * position[0] + position[1] * position[1]) < 0.0025
        success = wp.abs(roll) + wp.abs(pitch) < 0.025 and centered and position[2] < 0.001
        asset_assembled[world, variant] = success
        if success:
            count += 1

        root_position = wp.transform_get_translation(held_root) - env_origins[world]
        finite = wp.isfinite(root_position[0]) and wp.isfinite(root_position[1]) and wp.isfinite(root_position[2])
        outside = outside or not finite
        outside = outside or root_position[0] < lower[0] or root_position[0] > upper[0]
        outside = outside or root_position[1] < lower[1] or root_position[1] > upper[1]
        outside = outside or root_position[2] < lower[2] or root_position[2] > upper[2]

    all_success[world] = count == NUM_ASSEMBLIES
    any_held_asset_out_of_bound[world] = outside


class AssemblyState(ManagerTermBase):
    """Compute all assembly frames, completion, and validity in one launch."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        model = NewtonManager.get_model()
        held = cfg.params["held_bodies"].resolve(model)
        fixed = cfg.params["fixed_bodies"].resolve(model)
        robot_root = cfg.params["robot_root_body"].resolve(model)
        expected = (env.num_envs, NUM_ASSEMBLIES)
        if held.shape != expected or fixed.shape != expected:
            raise ValueError(f"Assembly body selectors must resolve to {expected}, got {held.shape} and {fixed.shape}.")
        if robot_root.shape != (env.num_envs, 1):
            raise ValueError(f"Robot root selector must resolve to {(env.num_envs, 1)}, got {robot_root.shape}.")

        reset = env.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("AssemblyState requires the resolved board reset term.")
        self._reset = reset
        self._held_body_ids = wp.array(held.ids, dtype=wp.int32, device=env.device)
        self._fixed_body_ids = wp.array(fixed.ids, dtype=wp.int32, device=env.device)
        self._robot_root_body_ids = wp.array(robot_root.ids, dtype=wp.int32, device=env.device)
        self._variant_ids = wp.from_torch(reset.variant_ids, dtype=wp.uint8)
        self._env_origins = wp.from_torch(env.scene.env_origins.contiguous(), dtype=wp.vec3f)
        self._held_align = self._offsets([variant.held_align.pose for variant in ASSEMBLY_VARIANTS], env.device)
        self._fixed_tip = self._offsets([variant.fixed_tip.pose for variant in ASSEMBLY_VARIANTS], env.device)
        self._assembled = self._offsets(
            [AssemblyProfile(variant.profile).assembled_offset.pose for variant in ASSEMBLY_VARIANTS], env.device
        )
        bounds = cfg.params["workspace"]
        self._lower = wp.vec3f(*(bounds[axis][0] for axis in ("x", "y", "z")))
        self._upper = wp.vec3f(*(bounds[axis][1] for axis in ("x", "y", "z")))

        self._assembly_frames = wp.empty((env.num_envs, 2 * NUM_ASSEMBLIES), dtype=wp.transformf, device=env.device)
        self._asset_assembled = wp.empty((env.num_envs, NUM_ASSEMBLIES), dtype=wp.bool, device=env.device)
        self._all_success = wp.empty(env.num_envs, dtype=wp.bool, device=env.device)
        self._any_held_asset_out_of_bound = wp.empty(env.num_envs, dtype=wp.bool, device=env.device)
        self._assembly_frames_torch = wp.to_torch(self._assembly_frames).view(env.num_envs, -1)
        self._asset_assembled_torch = wp.to_torch(self._asset_assembled)
        self._all_success_torch = wp.to_torch(self._all_success)
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
        robot_root_body: NewtonBodySelectorCfg,
        workspace: dict[str, tuple[float, float]],
    ) -> None:
        self._stamp = (-1, -1)

    @property
    def assembly_frames(self) -> torch.Tensor:
        """Canonical held-align and fixed-tip poses in the robot root frame."""
        self._refresh()
        return self._assembly_frames_torch

    @property
    def all_success(self) -> torch.Tensor:
        """Whether every assembly is complete."""
        self._refresh()
        return self._all_success_torch

    @property
    def asset_assembled(self) -> torch.Tensor:
        """Whether each canonical assembly is complete."""
        self._refresh()
        return self._asset_assembled_torch

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
                self._fixed_body_ids,
                self._robot_root_body_ids,
                self._variant_ids,
                self._env_origins,
                self._held_align,
                self._fixed_tip,
                self._assembled,
                self._lower,
                self._upper,
            ],
            outputs=[
                self._assembly_frames,
                self._asset_assembled,
                self._all_success,
                self._any_held_asset_out_of_bound,
            ],
            device=self.device,
        )
        self._stamp = stamp


def _assembly_state(env: ManagerBasedRLEnv) -> AssemblyState:
    return env.event_manager.get_term_cfg("assembly_state").func


def assembly_frames_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return canonical held-align and fixed-tip poses in the robot root frame."""
    return _assembly_state(env).assembly_frames


def any_held_asset_out_of_bound(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return environments where any held asset left the workspace."""
    return _assembly_state(env).any_held_asset_out_of_bound


class assembly_progress_context(ManagerTermBase):
    """Expose all-pairs completion through Factory's progress-context contract."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._state = _assembly_state(env)
        self._dummy = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def is_success(self) -> torch.Tensor:
        return self._state.all_success

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        env.extras["successes"] = self.is_success
        return self._dummy


def assembly_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward environments only when every assembly pair is complete."""
    return _assembly_state(env).all_success.float()
