# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-ready reset transforms selected by motion manager presets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.math import convert_quat, quat_apply, quat_from_angle_axis, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH

from ..trajectory.smpl import smpl_live_joint_source_names
from .commands import MotionStatePayload


def _select_reset_state(
    reference: MotionStatePayload.ResetState,
    alternative: MotionStatePayload.ResetState,
    use_alternative: torch.Tensor,
) -> MotionStatePayload.ResetState:
    """Select one of two simulator-ready states for each reset row."""

    def select(reference_value: torch.Tensor, alternative_value: torch.Tensor) -> torch.Tensor:
        mask = use_alternative
        while mask.ndim < reference_value.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, alternative_value, reference_value)

    return MotionStatePayload.ResetState(
        root_position=select(reference.root_position, alternative.root_position),
        root_rotation_xyzw=select(reference.root_rotation_xyzw, alternative.root_rotation_xyzw),
        root_linear_velocity_world=select(
            reference.root_linear_velocity_world,
            alternative.root_linear_velocity_world,
        ),
        root_angular_velocity_world=select(
            reference.root_angular_velocity_world,
            alternative.root_angular_velocity_world,
        ),
        joint_position=select(reference.joint_position, alternative.joint_position),
        joint_velocity=select(reference.joint_velocity, alternative.joint_velocity),
    )


class SmplMocapAndFallReset:
    """Mix reference frames with an exact HumEnv fall-state reservoir."""

    reset_source_names = ("motion", "fall")
    _POOL_SIZE = 8192

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        random_actions_high_exclusive: int,
        physics_dt_seconds: float,
        physics_steps_per_action: int,
    ) -> None:
        if random_actions_high_exclusive < 1 or physics_steps_per_action < 1:
            raise ValueError("SMPL fall synthesis requires positive action and physics-step counts.")
        if not np.isfinite(physics_dt_seconds) or physics_dt_seconds <= 0.0:
            raise ValueError("SMPL fall synthesis requires a finite positive physics timestep [s].")
        self.seed = env.cfg.seed
        self._random_actions_high_exclusive = random_actions_high_exclusive
        self._physics_dt_seconds = physics_dt_seconds
        self._physics_steps_per_action = physics_steps_per_action
        source_skeleton = env.cfg.commands.motion.task_table.source.build_skeleton()
        live_source_names = smpl_live_joint_source_names(tuple(env.scene["robot"].joint_names))
        self._live_from_source = torch.tensor(
            [source_skeleton.joint_names.index(name) for name in live_source_names],
            dtype=torch.int64,
            device=env.device,
        )
        self._pool: MotionStatePayload.ResetState | None = None

    def _build_pool(self, device: torch.device) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("SMPL MoCapAndFall reset synthesis requires the existing MuJoCo package.") from exc

        model = mujoco.MjModel.from_xml_path(SMPL_HUMENV_MJCF_PATH)
        if (model.nq, model.nv, model.nu) != (76, 75, 69):
            raise ValueError("The SMPL fall MJCF must expose qpos[76], qvel[75], and action[69].")
        model.opt.timestep = self._physics_dt_seconds
        data = mujoco.MjData(model)
        random = np.random.default_rng(self.seed)
        qpos = np.empty((self._POOL_SIZE, model.nq), dtype=np.float32)
        qvel = np.empty((self._POOL_SIZE, model.nv), dtype=np.float32)
        for index in range(self._POOL_SIZE):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[2] = 1.0
            data.qpos[3:7] = random.random(4)
            mujoco.mj_forward(model, data)
            for _ in range(int(random.integers(0, self._random_actions_high_exclusive))):
                data.ctrl[:] = random.random(model.nu) - 0.5
                mujoco.mj_step(model, data, nstep=self._physics_steps_per_action)
            qpos[index] = data.qpos
            qvel[index] = data.qvel

        qpos_t = torch.as_tensor(qpos, device=device)
        qvel_t = torch.as_tensor(qvel, device=device)
        root_rotation = convert_quat(qpos_t[:, 3:7], to="xyzw")
        self._pool = MotionStatePayload.ResetState(
            root_position=qpos_t[:, :3],
            root_rotation_xyzw=root_rotation,
            root_linear_velocity_world=qvel_t[:, :3],
            root_angular_velocity_world=quat_apply(root_rotation, qvel_t[:, 3:6]),
            joint_position=qpos_t[:, 7:].index_select(1, self._live_from_source),
            joint_velocity=qvel_t[:, 6:].index_select(1, self._live_from_source),
        )

    def __call__(
        self,
        reference: MotionStatePayload.ResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionStatePayload.ResetState:
        """Select source zero as motion and source one as synthesized fall."""
        if self._pool is None:
            self._build_pool(reference.root_position.device)
        assert self._pool is not None
        pool_indices = torch.randint(
            0,
            self._POOL_SIZE,
            (reset_source_indices.shape[0],),
            device=reset_source_indices.device,
            generator=generator,
        )
        alternative = MotionStatePayload.ResetState(
            root_position=self._pool.root_position[pool_indices],
            root_rotation_xyzw=self._pool.root_rotation_xyzw[pool_indices],
            root_linear_velocity_world=self._pool.root_linear_velocity_world[pool_indices],
            root_angular_velocity_world=self._pool.root_angular_velocity_world[pool_indices],
            joint_position=self._pool.joint_position[pool_indices],
            joint_velocity=self._pool.joint_velocity[pool_indices],
        )
        return _select_reset_state(reference, alternative, reset_source_indices == 1)


class G1ReferenceAndLieDownReset:
    """Select reference and released G1 lie-down states."""

    reset_source_names = ("reference", "lie_down")

    def __init__(self, env: ManagerBasedRLEnv, *, lie_down_root_height_m: float) -> None:
        del env
        if not np.isfinite(lie_down_root_height_m) or lie_down_root_height_m <= 0.0:
            raise ValueError("G1 lie-down root height must be finite and positive [m].")
        self._lie_down_root_height_m = lie_down_root_height_m

    def __call__(
        self,
        reference: MotionStatePayload.ResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionStatePayload.ResetState:
        """Select source zero as reference and source one as native G1 lie-down."""
        root_position = reference.root_position.clone()
        root_rotation = reference.root_rotation_xyzw.clone()
        lie_down = reset_source_indices == 1
        root_position[lie_down, 2] = self._lie_down_root_height_m

        sign = torch.where(
            torch.rand((), device=root_rotation.device, generator=generator) < 0.5,
            root_rotation.new_tensor(1.0),
            root_rotation.new_tensor(-1.0),
        )
        angle = (sign * (-0.5 * torch.pi)).expand(root_rotation.shape[0])
        axis = root_rotation.new_tensor((1.0, 0.0, 0.0)).expand(root_rotation.shape[0], 3)
        rotated = quat_mul(quat_from_angle_axis(angle, axis), root_rotation)
        root_rotation[lie_down] = rotated[lie_down]
        return MotionStatePayload.ResetState(
            root_position=root_position,
            root_rotation_xyzw=root_rotation,
            root_linear_velocity_world=reference.root_linear_velocity_world,
            root_angular_velocity_world=reference.root_angular_velocity_world,
            joint_position=reference.joint_position,
            joint_velocity=reference.joint_velocity,
        )


__all__ = ["G1ReferenceAndLieDownReset", "SmplMocapAndFallReset"]
