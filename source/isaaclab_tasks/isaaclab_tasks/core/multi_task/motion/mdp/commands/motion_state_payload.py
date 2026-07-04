# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion descriptors and simulator reset rows for StateCommand."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from ...data.reset_state import MotionResetState
from .motion_sampler import MotionSampler
from .motion_task_table import MotionTaskTable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg


class MotionStatePayload:
    """Sample motion descriptors and materialize simulator reset rows."""

    class _ReferenceResolver:
        """Fixed-capacity in-place interpolation scratch for one reset batch."""

        class _Field:
            """One fixed output field plus interpolation scratch."""

            def __init__(self, table: MotionTaskTable, name: str, capacity: int) -> None:
                self.interpolation = table.interpolation(name)
                source = table.field(name)
                shape = source.shape[1:]
                width = math.prod(shape)
                self.source = source.view(source.shape[0], width)
                self.output = torch.empty((capacity, *shape), dtype=torch.float32, device=table.device)
                self.output_flat = self.output.view(capacity, width)
                self.value0 = torch.empty_like(self.output_flat)
                self.value1 = torch.empty_like(self.output_flat)
                if self.interpolation != "slerp":
                    return

                quaternion_count = width // 4
                scratch_shape = (capacity, quaternion_count)
                self.product = torch.empty((capacity, quaternion_count, 4), device=table.device)
                self.dot = torch.empty(scratch_shape, device=table.device)
                self.angle = torch.empty(scratch_shape, device=table.device)
                self.denominator = torch.empty(scratch_shape, device=table.device)
                self.weight0 = torch.empty(scratch_shape, device=table.device)
                self.weight1 = torch.empty(scratch_shape, device=table.device)
                self.sign = torch.empty(scratch_shape, device=table.device)
                self.norm = torch.empty(scratch_shape, device=table.device)
                self.negative = torch.empty(scratch_shape, dtype=torch.bool, device=table.device)
                self.near_linear = torch.empty(scratch_shape, dtype=torch.bool, device=table.device)
                self.temp = torch.empty((capacity, quaternion_count, 4), device=table.device)

            def resolve(self, frame0: torch.Tensor, frame1: torch.Tensor, alpha: torch.Tensor) -> None:
                """Gather and interpolate one reset-batch prefix without replacing storage."""
                count = frame0.shape[0]
                value0 = self.value0[:count]
                value1 = self.value1[:count]
                output_flat = self.output_flat[:count]
                torch.index_select(self.source, 0, frame0, out=value0)
                torch.index_select(self.source, 0, frame1, out=value1)
                if self.interpolation == "linear":
                    torch.lerp(value0, value1, alpha.view(-1, 1), out=output_flat)
                    return

                quaternion_count = output_flat.shape[1] // 4
                quaternion_shape = (count, quaternion_count, 4)
                q0 = value0.view(quaternion_shape)
                q1 = value1.view(quaternion_shape)
                output = output_flat.view(quaternion_shape)
                product = self.product[:count]
                dot = self.dot[:count]
                angle = self.angle[:count]
                denominator = self.denominator[:count]
                weight0 = self.weight0[:count]
                weight1 = self.weight1[:count]
                sign = self.sign[:count]
                norm = self.norm[:count]
                negative = self.negative[:count]
                near_linear = self.near_linear[:count]
                temp = self.temp[:count]
                quaternion_alpha = alpha.view(count, 1).expand(count, quaternion_count)

                torch.mul(q0, q1, out=product)
                torch.sum(product, dim=-1, out=dot)
                torch.lt(dot, 0.0, out=negative)
                sign.fill_(1.0)
                sign.masked_fill_(negative, -1.0)
                q1.mul_(sign.unsqueeze(-1))
                dot.abs_().clamp_(max=1.0)

                torch.acos(dot, out=angle)
                torch.sin(angle, out=denominator)
                denominator.clamp_min_(torch.finfo(torch.float32).eps)
                weight0.copy_(quaternion_alpha).neg_().add_(1.0).mul_(angle).sin_().div_(denominator)
                weight1.copy_(quaternion_alpha).mul_(angle).sin_().div_(denominator)
                torch.mul(q0, weight0.unsqueeze(-1), out=output)
                torch.mul(q1, weight1.unsqueeze(-1), out=temp)
                output.add_(temp)

                torch.sub(q1, q0, out=temp)
                temp.mul_(quaternion_alpha.unsqueeze(-1)).add_(q0)
                torch.linalg.vector_norm(temp, dim=-1, out=norm)
                norm.clamp_min_(1.0e-12)
                temp.div_(norm.unsqueeze(-1))
                torch.gt(dot, 0.9995, out=near_linear)
                torch.where(near_linear.unsqueeze(-1), temp, output, out=output)

        def __init__(self, table: MotionTaskTable, capacity: int, field_names: tuple[str, ...]) -> None:
            self.table = table
            device = table.device
            self.clip_indices = torch.empty(capacity, dtype=torch.int64, device=device)
            self.reset_time_ranges_seconds = torch.empty(capacity, 2, device=device)
            self.reset_source_indices = torch.empty(capacity, dtype=torch.int64, device=device)
            self.reference_time_seconds = torch.empty(capacity, device=device)
            self.source_fps = torch.empty(capacity, device=device)
            self.frame_counts = torch.empty(capacity, dtype=torch.int64, device=device)
            self.last_frame = torch.empty(capacity, dtype=torch.int64, device=device)
            self.last_frame_float = torch.empty(capacity, device=device)
            self.position = torch.empty(capacity, device=device)
            self.clamped_position = torch.empty(capacity, device=device)
            self.floor_position = torch.empty(capacity, device=device)
            self.local_frame0 = torch.empty(capacity, dtype=torch.int64, device=device)
            self.local_frame1 = torch.empty(capacity, dtype=torch.int64, device=device)
            self.global_frame0 = torch.empty(capacity, dtype=torch.int64, device=device)
            self.global_frame1 = torch.empty(capacity, dtype=torch.int64, device=device)
            self.clip_offset = torch.empty(capacity, dtype=torch.int64, device=device)
            self.alpha = torch.empty(capacity, device=device)
            self._fields = {name: self._Field(table, name, capacity) for name in field_names}
            self.reference = MappingProxyType({name: field.output for name, field in self._fields.items()})

        def resolve(self, count: int) -> None:
            """Resolve only the populated reset-batch prefix."""
            clip_indices = self.clip_indices[:count]
            source_fps = self.source_fps[:count]
            frame_counts = self.frame_counts[:count]
            last_frame = self.last_frame[:count]
            last_frame_float = self.last_frame_float[:count]
            position = self.position[:count]
            clamped_position = self.clamped_position[:count]
            floor_position = self.floor_position[:count]
            local_frame0 = self.local_frame0[:count]
            local_frame1 = self.local_frame1[:count]
            global_frame0 = self.global_frame0[:count]
            global_frame1 = self.global_frame1[:count]
            clip_offset = self.clip_offset[:count]
            alpha = self.alpha[:count]

            torch.index_select(self.table.source_fps, 0, clip_indices, out=source_fps)
            torch.index_select(self.table.frame_counts, 0, clip_indices, out=frame_counts)
            torch.sub(frame_counts, 1, out=last_frame)
            last_frame_float.copy_(last_frame)
            torch.index_select(self.table.clip_offsets, 0, clip_indices, out=clip_offset)
            torch.mul(self.reference_time_seconds[:count], source_fps, out=position)
            torch.clamp(position, min=0.0, out=clamped_position)
            torch.minimum(clamped_position, last_frame_float, out=clamped_position)
            torch.floor(clamped_position, out=floor_position)
            local_frame0.copy_(floor_position)
            torch.add(local_frame0, 1, out=local_frame1)
            torch.minimum(local_frame1, last_frame, out=local_frame1)
            torch.sub(clamped_position, floor_position, out=alpha)
            torch.add(clip_offset, local_frame0, out=global_frame0)
            torch.add(clip_offset, local_frame1, out=global_frame1)
            for field in self._fields.values():
                field.resolve(global_frame0, global_frame1, alpha)

    error_names: tuple[str, ...] = ()
    error_dim = 0
    command_dim = 0

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedRLEnv, table: MotionTaskTable):
        payload_cfg = cfg.payload
        self._env = env
        self._device = torch.device(env.device)
        self.table = table
        self.sampler = MotionSampler(table, payload_cfg.reset_sources, capacity=env.num_envs, seed=env.cfg.seed)
        self.num_envs = env.num_envs
        if payload_cfg.root_velocity_frame not in ("link", "center_of_mass"):
            raise ValueError("root_velocity_frame must be 'link' or 'center_of_mass'.")
        self._root_velocity_frame = payload_cfg.root_velocity_frame
        self._states_relative = cfg.states_relative
        if self.table.device != self._device:
            raise ValueError("Motion payload and MotionTaskTable must use the environment device.")
        reset_fields = (
            "joint_position",
            "joint_velocity",
            "root_position",
            "root_rotation",
            "root_linear_velocity",
            "root_angular_velocity",
        )
        if not set(reset_fields).issubset(self.table.frames.available_fields):
            raise ValueError("The table does not contain one complete simulator reset state.")

        self.robot = env.scene[payload_cfg.robot_asset_name]
        if self.table.joint_names != tuple(self.robot.joint_names):
            raise ValueError("Trajectory joint names differ from the live articulation order.")
        if self.table.reference_frame_names:
            live_body_names = tuple(self.robot.body_names)
            if self.table.reference_frame_names[: len(live_body_names)] != live_body_names:
                raise ValueError("Trajectory reference frames do not begin with the live articulation body order.")
        reset_transform_binds = {
            name: eval(expression, {}, {"env": env, "payload": self})  # noqa: S307
            for name, expression in payload_cfg.reset_transform_binds.items()
        }
        reset_transform = payload_cfg.reset_transform_factory(
            capacity=self.num_envs,
            device=self._device,
            **reset_transform_binds,
            **payload_cfg.reset_transform_params,
        )
        if getattr(reset_transform, "reset_source_names", None) != self.sampler.reset_source_names:
            raise ValueError(
                "Reset-transform source names differ from the motion sampler: "
                f"expected {self.sampler.reset_source_names}, "
                f"got {getattr(reset_transform, 'reset_source_names', None)}."
            )
        self._reset_transform = reset_transform

        self._resolver = self._ReferenceResolver(table, self.num_envs, reset_fields)
        self._root_state = torch.empty(self.num_envs, 13, device=self._device)
        self._env_origins = torch.empty(self.num_envs, 3, device=self._device)

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample motion descriptor rows through the payload-owned policy."""
        return self.sampler.sample_rows(count)

    def bind(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Resolve one task-row batch and write only its selected simulator rows."""
        count = env_ids.shape[0]
        clip_indices = self._resolver.clip_indices[:count]
        reset_ranges = self._resolver.reset_time_ranges_seconds[:count]
        reset_sources = self._resolver.reset_source_indices[:count]
        reset_times = self._resolver.reference_time_seconds[:count]
        torch.index_select(self.table.clip_indices, 0, task_rows, out=clip_indices)
        torch.index_select(self.table.reset_time_ranges_seconds, 0, task_rows, out=reset_ranges)
        self.sampler.sample_reset_sources(reset_sources)
        self.sampler.sample_reset_times(reset_ranges, reset_times)
        self._resolver.resolve(count)
        self._write_reset(env_ids, reset_sources, count)

    def _write_reset(self, env_ids: torch.Tensor, reset_sources: torch.Tensor, count: int) -> None:
        """Write one resolved reset-batch prefix to selected simulator rows."""
        reference = self._resolver.reference
        decoded = MotionResetState(
            root_position=reference["root_position"][:count],
            root_rotation_xyzw=reference["root_rotation"][:count],
            root_linear_velocity_world=reference["root_linear_velocity"][:count],
            root_angular_velocity_world=reference["root_angular_velocity"][:count],
            joint_position=reference["joint_position"][:count],
            joint_velocity=reference["joint_velocity"][:count],
        )
        decoded = self._reset_transform(decoded, reset_sources, self.sampler.generator)
        expected_shapes = (
            (decoded.root_position, (count, 3)),
            (decoded.root_rotation_xyzw, (count, 4)),
            (decoded.root_linear_velocity_world, (count, 3)),
            (decoded.root_angular_velocity_world, (count, 3)),
            (decoded.joint_position, (count, len(self.robot.joint_names))),
            (decoded.joint_velocity, (count, len(self.robot.joint_names))),
        )
        if any(
            value.shape != shape or value.dtype is not torch.float32 or value.device != self._device
            for value, shape in expected_shapes
        ):
            raise ValueError("Motion reset state has a wrong shape, dtype, or device.")

        root_state = self._root_state[:count]
        root_state[:, :3].copy_(decoded.root_position)
        if self._states_relative:
            env_origins = self._env_origins[:count]
            torch.index_select(self._env.scene.env_origins, 0, env_ids, out=env_origins)
            root_state[:, :3].add_(env_origins)
        root_state[:, 3:7].copy_(decoded.root_rotation_xyzw)
        root_state[:, 7:10].copy_(decoded.root_linear_velocity_world)
        root_state[:, 10:13].copy_(decoded.root_angular_velocity_world)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        if self._root_velocity_frame == "link":
            self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        else:
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=decoded.joint_position, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=decoded.joint_velocity, env_ids=env_ids)

    def update(self, step_dt: float, command_out: torch.Tensor, error_out: torch.Tensor) -> None:
        """Keep the empty policy-command output unchanged between resets."""
        del step_dt, command_out, error_out

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Accept the shared debug-visualization lifecycle without markers."""
        del debug_vis

    def debug_visualize(self, env: ManagerBasedRLEnv) -> None:
        """Leave motion visualization to explicit observation/reference tools."""
        del env
