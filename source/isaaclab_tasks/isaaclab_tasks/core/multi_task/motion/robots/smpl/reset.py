# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SMPL motion-frame and synthesized-fall reset transform."""

from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.utils.math import convert_quat, quat_apply

from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_JOINT_NAMES, SMPL_HUMENV_MJCF_PATH

from ...data import MotionResetState
from .articulation import smpl_live_joint_mujoco_names


class SmplHumEnvMocapAndFallReset:
    """Mix reference frames with an exact HumEnv fall-state reservoir."""

    reset_source_names = ("reference", "fall")

    def __init__(
        self,
        *,
        seed: int,
        device: str | torch.device,
        capacity: int,
        live_joint_names: tuple[str, ...],
        physics_dt_seconds: float,
        physics_steps_per_action: int,
        random_actions_high_exclusive: int,
        fall_pool_size: int,
        initial_root_height_m: float,
        initial_root_quaternion_component_range: tuple[float, float],
        control_range: tuple[float, float],
    ) -> None:
        if random_actions_high_exclusive < 1 or physics_steps_per_action < 1:
            raise ValueError("SMPL fall synthesis requires positive action and physics-step counts.")
        if fall_pool_size < 1:
            raise ValueError("SMPL fall synthesis requires a positive pool size.")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("SMPL reset capacity must be a positive integer.")
        if not math.isfinite(physics_dt_seconds) or physics_dt_seconds <= 0.0:
            raise ValueError("SMPL fall synthesis requires a finite positive physics timestep [s].")
        if not math.isfinite(initial_root_height_m) or initial_root_height_m <= 0.0:
            raise ValueError("SMPL fall initial root height must be finite and positive [m].")
        quaternion_low, quaternion_high = initial_root_quaternion_component_range
        control_low, control_high = control_range
        if not all(math.isfinite(value) for value in initial_root_quaternion_component_range + control_range):
            raise ValueError("SMPL fall randomization ranges must be finite.")
        if quaternion_low >= quaternion_high or control_low >= control_high:
            raise ValueError("SMPL fall randomization ranges must be increasing.")

        device = torch.device(device)
        self.seed = seed
        self._capacity = capacity
        self._random_actions_high_exclusive = random_actions_high_exclusive
        self._pool_size = fall_pool_size
        self._physics_dt_seconds = physics_dt_seconds
        self._physics_steps_per_action = physics_steps_per_action
        self._initial_root_height_m = initial_root_height_m
        self._initial_root_quaternion_component_low = quaternion_low
        self._initial_root_quaternion_component_width = quaternion_high - quaternion_low
        self._control_low = control_low
        self._control_width = control_high - control_low
        live_mujoco_names = smpl_live_joint_mujoco_names(live_joint_names)
        self._live_from_mujoco = torch.tensor(
            [MUJOCO_JOINT_NAMES.index(name) for name in live_mujoco_names],
            dtype=torch.int64,
            device=device,
        )
        self._pool = self._build_pool(device)
        joint_count = len(live_joint_names)
        self._pool_indices = torch.empty(capacity, dtype=torch.int64, device=device)
        self._use_fall = torch.empty(capacity, dtype=torch.bool, device=device)
        self._output = MotionResetState(
            root_position=torch.empty(capacity, 3, device=device),
            root_rotation_xyzw=torch.empty(capacity, 4, device=device),
            root_linear_velocity_world=torch.empty(capacity, 3, device=device),
            root_angular_velocity_world=torch.empty(capacity, 3, device=device),
            joint_position=torch.empty(capacity, joint_count, device=device),
            joint_velocity=torch.empty(capacity, joint_count, device=device),
        )

    def _build_pool(self, device: torch.device) -> MotionResetState:
        """Synthesize and freeze the CPU MuJoCo reservoir during construction."""
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("SMPL HumEnv mocap/fall reset synthesis requires the existing MuJoCo package.") from exc

        model = mujoco.MjModel.from_xml_path(SMPL_HUMENV_MJCF_PATH)
        if (model.nq, model.nv, model.nu) != (76, 75, 69):
            raise ValueError("The SMPL fall MJCF must expose qpos[76], qvel[75], and action[69].")
        model.opt.timestep = self._physics_dt_seconds
        data = mujoco.MjData(model)
        random = np.random.default_rng(self.seed)
        qpos = np.empty((self._pool_size, model.nq), dtype=np.float32)
        qvel = np.empty((self._pool_size, model.nv), dtype=np.float32)
        for index in range(self._pool_size):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[2] = self._initial_root_height_m
            data.qpos[3:7] = (
                random.random(4) * self._initial_root_quaternion_component_width
                + self._initial_root_quaternion_component_low
            )
            mujoco.mj_forward(model, data)
            for _ in range(int(random.integers(0, self._random_actions_high_exclusive))):
                data.ctrl[:] = random.random(model.nu) * self._control_width + self._control_low
                mujoco.mj_step(model, data, nstep=self._physics_steps_per_action)
            qpos[index] = data.qpos
            qvel[index] = data.qvel

        qpos_t = torch.as_tensor(qpos, device=device)
        qvel_t = torch.as_tensor(qvel, device=device)
        root_rotation = convert_quat(qpos_t[:, 3:7], to="xyzw")
        return MotionResetState(
            root_position=qpos_t[:, :3],
            root_rotation_xyzw=root_rotation,
            root_linear_velocity_world=qvel_t[:, :3],
            root_angular_velocity_world=quat_apply(root_rotation, qvel_t[:, 3:6]),
            joint_position=qpos_t[:, 7:].index_select(1, self._live_from_mujoco),
            joint_velocity=qvel_t[:, 6:].index_select(1, self._live_from_mujoco),
        )

    @staticmethod
    def _select(
        pool: torch.Tensor,
        reference: torch.Tensor,
        pool_indices: torch.Tensor,
        use_fall: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Gather pool rows and select fall rows into caller-stable storage."""
        torch.index_select(pool, 0, pool_indices, out=output)
        mask = use_fall
        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)
        torch.where(mask, output, reference, out=output)

    def __call__(
        self,
        reference: MotionResetState,
        reset_source_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> MotionResetState:
        """Select source zero as reference and source one as synthesized fall."""
        count = reset_source_indices.shape[0]
        if count > self._capacity:
            raise ValueError(f"SMPL reset batch {count} exceeds capacity {self._capacity}.")
        pool_indices = self._pool_indices[:count]
        use_fall = self._use_fall[:count]
        torch.randint(0, self._pool_size, (count,), generator=generator, out=pool_indices)
        torch.eq(reset_source_indices, 1, out=use_fall)

        output = self._output
        root_position = output.root_position[:count]
        root_rotation = output.root_rotation_xyzw[:count]
        root_linear_velocity = output.root_linear_velocity_world[:count]
        root_angular_velocity = output.root_angular_velocity_world[:count]
        joint_position = output.joint_position[:count]
        joint_velocity = output.joint_velocity[:count]
        self._select(self._pool.root_position, reference.root_position, pool_indices, use_fall, root_position)
        self._select(
            self._pool.root_rotation_xyzw,
            reference.root_rotation_xyzw,
            pool_indices,
            use_fall,
            root_rotation,
        )
        self._select(
            self._pool.root_linear_velocity_world,
            reference.root_linear_velocity_world,
            pool_indices,
            use_fall,
            root_linear_velocity,
        )
        self._select(
            self._pool.root_angular_velocity_world,
            reference.root_angular_velocity_world,
            pool_indices,
            use_fall,
            root_angular_velocity,
        )
        self._select(self._pool.joint_position, reference.joint_position, pool_indices, use_fall, joint_position)
        self._select(self._pool.joint_velocity, reference.joint_velocity, pool_indices, use_fall, joint_velocity)
        return MotionResetState(
            root_position=root_position,
            root_rotation_xyzw=root_rotation,
            root_linear_velocity_world=root_linear_velocity,
            root_angular_velocity_world=root_angular_velocity,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
        )
