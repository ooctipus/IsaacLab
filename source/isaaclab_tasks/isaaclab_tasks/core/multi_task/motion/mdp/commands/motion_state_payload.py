# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-specific payload for the shared state-command lifecycle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import torch

from ..history import AppliedTransitionHistory, AppliedTransitionHistoryLayout
from .motion_task_table import MotionTaskTable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg


class MotionTransitionState(Protocol):
    """Optional resettable transition state attached after manager construction."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear rows whose reset seed must not enter later transition history."""


class _MotionReferenceResolver:
    """Fixed-size in-place continuous-time lookup over a motion task table."""

    class _Field:
        """One fixed output field plus interpolation scratch."""

        def __init__(
            self,
            table: MotionTaskTable,
            name: str,
            num_envs: int,
            alpha: torch.Tensor,
        ) -> None:
            self.interpolation = table.interpolation(name)
            source = table.field(name)
            self.shape = source.shape[1:]
            self.width = math.prod(self.shape)
            self.source = source.view(source.shape[0], self.width)
            self.output = torch.empty((num_envs, *self.shape), dtype=torch.float32, device=table.device)
            self.output_flat = self.output.view(num_envs, self.width)
            self.value0 = torch.empty_like(self.output_flat)
            self.value1 = torch.empty_like(self.output_flat)
            if self.interpolation != "slerp":
                return

            quaternion_count = self.width // 4
            shape = (num_envs, quaternion_count)
            self.alpha = alpha.view(num_envs, 1).expand(shape)
            self.product = torch.empty((num_envs, quaternion_count, 4), device=table.device)
            self.dot = torch.empty(shape, device=table.device)
            self.angle = torch.empty(shape, device=table.device)
            self.denominator = torch.empty(shape, device=table.device)
            self.weight0 = torch.empty(shape, device=table.device)
            self.weight1 = torch.empty(shape, device=table.device)
            self.sign = torch.empty(shape, device=table.device)
            self.norm = torch.empty(shape, device=table.device)
            self.negative = torch.empty(shape, dtype=torch.bool, device=table.device)
            self.near_linear = torch.empty(shape, dtype=torch.bool, device=table.device)
            self.temp = torch.empty((num_envs, quaternion_count, 4), device=table.device)

        def resolve(self, frame0: torch.Tensor, frame1: torch.Tensor, alpha: torch.Tensor) -> None:
            """Gather and interpolate this field without replacing output storage."""
            torch.index_select(self.source, 0, frame0, out=self.value0)
            if self.interpolation == "left":
                self.output_flat.copy_(self.value0)
                return

            torch.index_select(self.source, 0, frame1, out=self.value1)
            if self.interpolation == "linear":
                torch.lerp(self.value0, self.value1, alpha.view(-1, 1), out=self.output_flat)
                return

            q0 = self.value0.view(*self.product.shape)
            q1 = self.value1.view(*self.product.shape)
            output = self.output_flat.view(*self.product.shape)
            torch.mul(q0, q1, out=self.product)
            torch.sum(self.product, dim=-1, out=self.dot)
            torch.lt(self.dot, 0.0, out=self.negative)
            self.sign.fill_(1.0)
            self.sign.masked_fill_(self.negative, -1.0)
            q1.mul_(self.sign.unsqueeze(-1))
            self.dot.abs_().clamp_(max=1.0)

            torch.acos(self.dot, out=self.angle)
            torch.sin(self.angle, out=self.denominator)
            self.denominator.clamp_min_(torch.finfo(torch.float32).eps)

            self.weight0.copy_(self.alpha).neg_().add_(1.0).mul_(self.angle).sin_().div_(self.denominator)
            self.weight1.copy_(self.alpha).mul_(self.angle).sin_().div_(self.denominator)
            torch.mul(q0, self.weight0.unsqueeze(-1), out=output)
            torch.mul(q1, self.weight1.unsqueeze(-1), out=self.temp)
            output.add_(self.temp)

            torch.sub(q1, q0, out=self.temp)
            self.temp.mul_(self.alpha.unsqueeze(-1)).add_(q0)
            torch.linalg.vector_norm(self.temp, dim=-1, out=self.norm)
            self.norm.clamp_min_(1.0e-12)
            self.temp.div_(self.norm.unsqueeze(-1))
            torch.gt(self.dot, 0.9995, out=self.near_linear)
            torch.where(self.near_linear.unsqueeze(-1), self.temp, output, out=output)

    def __init__(
        self,
        table: MotionTaskTable,
        clip_indices: torch.Tensor,
        reference_time_seconds: torch.Tensor,
        resolved_fields: tuple[str, ...],
    ) -> None:
        self.table = table
        self.clip_indices = clip_indices
        self.reference_time_seconds = reference_time_seconds
        self.num_envs = clip_indices.shape[0]
        device = table.device

        self.source_fps = torch.empty(self.num_envs, device=device)
        self.frame_counts = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.last_frame = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.last_frame_float = torch.empty(self.num_envs, device=device)
        self.position = torch.empty(self.num_envs, device=device)
        self.clamped_position = torch.empty(self.num_envs, device=device)
        self.floor_position = torch.empty(self.num_envs, device=device)
        self.local_frame0 = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.local_frame1 = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.global_frame0 = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.global_frame1 = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.clip_offset = torch.empty(self.num_envs, dtype=torch.int64, device=device)
        self.alpha = torch.empty(self.num_envs, device=device)
        self.tail_valid = torch.empty(self.num_envs, dtype=torch.bool, device=device)
        self._nonnegative = torch.empty(self.num_envs, dtype=torch.bool, device=device)
        self._before_end = torch.empty(self.num_envs, dtype=torch.bool, device=device)
        self.reference_end_time_seconds = torch.empty(self.num_envs, device=device)
        self.tail_elapsed_seconds = torch.empty(self.num_envs, device=device)
        self.reference_phase = torch.empty(self.num_envs, device=device)
        self._phase_denominator = torch.empty(self.num_envs, device=device)

        self._fields = {name: self._Field(table, name, self.num_envs, self.alpha) for name in resolved_fields}
        self.reference = MappingProxyType({name: field.output for name, field in self._fields.items()})

    def flat(self, name: str) -> torch.Tensor:
        """Return a persistent flattened view of one resolved field."""
        return self._fields[name].output_flat

    def bind(self, field_names: tuple[str, ...]) -> None:
        """Bind clip-constant metadata, then resolve reset and step fields."""
        torch.index_select(self.table.source_fps, 0, self.clip_indices, out=self.source_fps)
        torch.index_select(self.table.frame_counts, 0, self.clip_indices, out=self.frame_counts)
        torch.sub(self.frame_counts, 1, out=self.last_frame)
        self.last_frame_float.copy_(self.last_frame)
        torch.index_select(self.table.clip_offsets, 0, self.clip_indices, out=self.clip_offset)
        torch.div(self.last_frame_float, self.source_fps, out=self.reference_end_time_seconds)
        self._phase_denominator.copy_(self.last_frame_float).clamp_min_(1.0)
        self.resolve(field_names)

    def resolve(self, field_names: tuple[str, ...]) -> None:
        """Advance scalar timing and refresh only requested trajectory fields."""
        torch.mul(self.reference_time_seconds, self.source_fps, out=self.position)
        torch.ge(self.position, 0.0, out=self._nonnegative)
        torch.le(self.position, self.last_frame_float, out=self._before_end)
        torch.logical_and(self._nonnegative, self._before_end, out=self.tail_valid)

        torch.clamp(self.position, min=0.0, out=self.clamped_position)
        torch.minimum(self.clamped_position, self.last_frame_float, out=self.clamped_position)
        if field_names:
            torch.floor(self.clamped_position, out=self.floor_position)
            self.local_frame0.copy_(self.floor_position)
            torch.add(self.local_frame0, 1, out=self.local_frame1)
            torch.minimum(self.local_frame1, self.last_frame, out=self.local_frame1)
            torch.sub(self.clamped_position, self.floor_position, out=self.alpha)
            torch.add(self.clip_offset, self.local_frame0, out=self.global_frame0)
            torch.add(self.clip_offset, self.local_frame1, out=self.global_frame1)
        torch.sub(
            self.reference_time_seconds,
            self.reference_end_time_seconds,
            out=self.tail_elapsed_seconds,
        )
        self.tail_elapsed_seconds.clamp_min_(0.0)
        torch.div(self.clamped_position, self._phase_denominator, out=self.reference_phase)

        for name in field_names:
            self._fields[name].resolve(self.global_frame0, self.global_frame1, self.alpha)


class MotionStatePayload:
    """Bind motion descriptors and own per-environment runtime motion facts.

    The payload consumes one completed :class:`MotionTaskTable`; it never
    decodes source files or owns learned-provider state. A reward/evidence term
    calls :meth:`record_step` once after physics and before final-observation
    capture. Command-manager updates only expose the already-current reference,
    so terminal history/evidence is never advanced too late.
    """

    error_names: tuple[str, ...] = ()
    error_dim = 0

    @dataclass(frozen=True, slots=True)
    class ResetState:
        """Simulator-ready root and joint state in fixed world-frame and xyzw semantics."""

        root_position: torch.Tensor
        """Root-link position [m], shape [batch, 3], float."""

        root_rotation_xyzw: torch.Tensor
        """Root-link xyzw orientation, shape [batch, 4], float."""

        root_linear_velocity_world: torch.Tensor
        """Root-link linear velocity [m/s], shape [batch, 3], float."""

        root_angular_velocity_world: torch.Tensor
        """Root-link angular velocity [rad/s], shape [batch, 3], float."""

        joint_position: torch.Tensor
        """Simulator-ordered joint positions [rad], shape [batch, joint_count], float."""

        joint_velocity: torch.Tensor
        """Simulator-ordered joint velocities [rad/s], shape [batch, joint_count], float."""

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedRLEnv, table: MotionTaskTable):
        payload_cfg = cfg.payload
        self._env = env
        self._device = torch.device(env.device)
        self._step_dt = float(env.step_dt)
        self.table = table
        self.num_envs = env.num_envs
        if payload_cfg.root_velocity_frame not in ("link", "center_of_mass"):
            raise ValueError("root_velocity_frame must be 'link' or 'center_of_mass'.")
        self._root_velocity_frame = payload_cfg.root_velocity_frame
        self._states_relative = cfg.states_relative
        if not math.isfinite(self._step_dt) or self._step_dt <= 0.0:
            raise ValueError("Environment step_dt must be finite and positive.")
        if self.table.device != self._device:
            raise ValueError("Motion payload and MotionTaskTable must use the environment device.")
        if payload_cfg.episode_length_steps < 1:
            raise ValueError("episode_length_steps must be positive.")
        step_fields = tuple(payload_cfg.step_fields)
        if len(set(step_fields)) != len(step_fields):
            raise ValueError("step_fields must be unique.")
        declared_fields = set(self.table.frames.available_fields)
        unknown_step = set(step_fields) - declared_fields
        if unknown_step:
            raise ValueError(f"Unknown motion step fields: {sorted(unknown_step)}.")
        reset_fields = [
            "joint_position",
            "joint_velocity",
            "root_position",
            "root_rotation",
            "root_linear_velocity",
            "root_angular_velocity",
        ]
        self._reset_fields = tuple(reset_fields)
        if not set(self._reset_fields).issubset(declared_fields):
            raise ValueError("The table does not contain one complete simulator reset state.")
        self._step_fields = step_fields
        self._resolved_fields = tuple(dict.fromkeys((*self._reset_fields, *self._step_fields)))
        if len(set(payload_cfg.command_fields)) != len(payload_cfg.command_fields) or not set(
            payload_cfg.command_fields
        ).issubset(self._step_fields):
            raise ValueError("command_fields must be unique members of step_fields.")

        self.robot = env.scene[payload_cfg.robot_asset_name]
        if self.table.joint_names != tuple(self.robot.joint_names):
            raise ValueError("Trajectory joint names differ from the live articulation order.")
        if self.table.reference_frame_names:
            live_body_names = tuple(self.robot.body_names)
            if self.table.reference_frame_names[: len(live_body_names)] != live_body_names:
                raise ValueError("Trajectory reference frames do not begin with the live articulation body order.")
        if payload_cfg.reset_transform_factory is None:
            if len(self.table.reset_source_names) != 1:
                raise ValueError("Multiple reset sources require one reset transform.")
            self._reset_transform = None
        else:
            reset_transform = payload_cfg.reset_transform_factory(env)
            if getattr(reset_transform, "reset_source_names", None) != self.table.reset_source_names:
                raise ValueError(
                    "Reset-transform source names differ from the task table: "
                    f"expected {self.table.reset_source_names}, "
                    f"got {getattr(reset_transform, 'reset_source_names', None)}."
                )
            self._reset_transform = reset_transform

        first_clip = table.clip_indices[:1]
        first_range = table.reset_time_ranges_seconds[:1]
        self.clip_indices = first_clip.expand(self.num_envs).clone()
        self.reset_time_ranges_seconds = first_range.expand(self.num_envs, 2).clone()
        self.reset_source_indices = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.reference_time_seconds = self.reset_time_ranges_seconds[:, 0].clone()
        self.episode_relative_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.episode_time_seconds = torch.zeros(self.num_envs, device=self._device)
        self.episode_phase = torch.zeros(self.num_envs, device=self._device)
        self.prior_edge_applied = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        history_fields = tuple(payload_cfg.history_fields)
        history_length = payload_cfg.history_length
        if bool(history_fields) != (history_length > 0):
            raise ValueError("history_fields and a positive history_length must be configured together.")
        if history_fields:
            history_layout = AppliedTransitionHistoryLayout(history_fields, history_length)
            self.history_value = torch.zeros(self.num_envs, history_layout.width, device=self._device)
            history_sources = {
                name: torch.empty(self.num_envs, width, device=self._device) for name, width in history_fields
            }
            self.history_fields = MappingProxyType(history_sources)
            self.history = AppliedTransitionHistory(
                history_layout,
                self.history_value,
                fields=history_sources,
                applied=self.prior_edge_applied,
            )
        else:
            self.history_value = torch.empty(self.num_envs, 0, device=self._device)
            self.history_fields = MappingProxyType({})
            self.history = None
        self._episode_step_float = torch.empty(self.num_envs, device=self._device)
        self._episode_phase_scale = 1.0 / payload_cfg.episode_length_steps

        self._resolver = _MotionReferenceResolver(
            self.table,
            self.clip_indices,
            self.reference_time_seconds,
            self._resolved_fields,
        )
        self._resolved_reference = self._resolver.reference
        self.reference = MappingProxyType({name: self._resolved_reference[name] for name in self._step_fields})

        num_joints = len(self.robot.joint_names)
        self.reset_state = self.ResetState(
            root_position=torch.empty(self.num_envs, 3, device=self._device),
            root_rotation_xyzw=torch.empty(self.num_envs, 4, device=self._device),
            root_linear_velocity_world=torch.empty(self.num_envs, 3, device=self._device),
            root_angular_velocity_world=torch.empty(self.num_envs, 3, device=self._device),
            joint_position=torch.empty(self.num_envs, num_joints, device=self._device),
            joint_velocity=torch.empty(self.num_envs, num_joints, device=self._device),
        )

        evidence_specs = tuple(payload_cfg.raw_evidence)
        evidence_names = tuple(spec.name for spec in evidence_specs)
        if len(evidence_names) != len(set(evidence_names)):
            raise ValueError("Raw evidence names must be unique.")
        for spec in evidence_specs:
            if not spec.name or spec.width < 1 or not spec.anchor:
                raise ValueError("Raw evidence requires a name, positive width, and timing anchor.")
        evidence_by_name = {spec.name: spec for spec in evidence_specs}
        auxiliary_evidence_names = tuple(payload_cfg.auxiliary_evidence)
        if len(auxiliary_evidence_names) != len(set(auxiliary_evidence_names)) or any(
            not name for name in auxiliary_evidence_names
        ):
            raise ValueError("Auxiliary evidence names must be unique and nonempty.")
        unknown_auxiliary = set(auxiliary_evidence_names) - evidence_by_name.keys()
        if unknown_auxiliary:
            raise ValueError(f"Unknown auxiliary evidence channels: {sorted(unknown_auxiliary)}.")
        if any(evidence_by_name[name].width != 1 for name in auxiliary_evidence_names):
            raise ValueError("Auxiliary evidence channels must be scalar raw evidence.")
        self.raw_evidence_specs = evidence_specs
        self.raw_evidence_names = evidence_names
        self.auxiliary_evidence_names = auxiliary_evidence_names
        self.auxiliary_evidence_specs = tuple(evidence_by_name[name] for name in auxiliary_evidence_names)
        self.raw_evidence_value = torch.zeros(
            self.num_envs, sum(spec.width for spec in evidence_specs), device=self._device
        )
        raw_evidence: dict[str, torch.Tensor] = {}
        evidence_offset = 0
        for spec in evidence_specs:
            evidence_end = evidence_offset + spec.width
            raw_evidence[spec.name] = self.raw_evidence_value[:, evidence_offset:evidence_end]
            evidence_offset = evidence_end
        self.raw_evidence = MappingProxyType(raw_evidence)

        self.motion_facts = MappingProxyType(
            {
                "clip_index": self.clip_indices,
                "episode_relative_step": self.episode_relative_step,
                "episode_time_seconds": self.episode_time_seconds,
                "episode_phase": self.episode_phase,
                "reference_time_seconds": self.reference_time_seconds,
                "reference_phase": self._resolver.reference_phase,
                "reset_source": self.reset_source_indices,
                "reset_time_range_seconds": self.reset_time_ranges_seconds,
                "tail_valid": self._resolver.tail_valid,
                "tail_elapsed_seconds": self._resolver.tail_elapsed_seconds,
                "action_applied": self.prior_edge_applied,
            }
        )

        self._command_fields: list[tuple[slice, torch.Tensor]] = []
        offset = 0
        for name in payload_cfg.command_fields:
            field = self._resolver.flat(name)
            end = offset + field.shape[1]
            self._command_fields.append((slice(offset, end), field))
            offset = end
        self.command_dim = offset
        self._command_std = torch.empty(self.num_envs, 0, device=self._device)
        self._task_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._task_reward = torch.zeros(self.num_envs, device=self._device)
        self._transition_state: MotionTransitionState | None = None
        self._resolver.bind(self._resolved_fields)

    def attach_transition_state(self, state: MotionTransitionState) -> None:
        """Attach the one caller-owned history/transition-state coordinator.

        This late attachment supports manager construction order: a reward term
        may attach after the observation manager exists, then call
        :meth:`record_step` before final-observation capture.
        """
        if self._transition_state is not None:
            raise RuntimeError("Motion transition state is already attached.")
        self._transition_state = state

    def command_std(self) -> torch.Tensor:
        """Return the empty success-threshold layout for non-tracking actors."""
        return self._command_std

    def get_task_done(self) -> torch.Tensor:
        """Return false; reference tail does not implicitly terminate an episode."""
        return self._task_done

    def get_task_reward(self) -> torch.Tensor:
        """Return zero; scalar reward composition is not command state."""
        return self._task_reward

    def bind(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Bind motion descriptors, sample reset time, and clear reset rows."""
        clip_indices, reset_ranges = self.table.select(task_rows)
        reset_sources = self.table.sample_reset_sources(env_ids.shape[0])
        self.clip_indices.index_copy_(0, env_ids, clip_indices)
        self.reset_time_ranges_seconds.index_copy_(0, env_ids, reset_ranges)
        self.reset_source_indices.index_copy_(0, env_ids, reset_sources)
        reset_fraction = torch.rand(env_ids.shape[0], device=self._device, generator=self.table.generator)
        reset_time = torch.lerp(reset_ranges[:, 0], reset_ranges[:, 1], reset_fraction)
        self.reference_time_seconds.index_copy_(0, env_ids, reset_time)
        self._finish_bind(env_ids)

    def bind_clip_start(self, env_ids: torch.Tensor, clip_indices: torch.Tensor) -> None:
        """Bind exact clips at time zero through the normal reset semantics."""
        if (
            env_ids.ndim != 1
            or env_ids.dtype is not torch.int64
            or env_ids.device != self._device
            or clip_indices.shape != env_ids.shape
            or clip_indices.dtype is not torch.int64
            or clip_indices.device != self._device
        ):
            raise ValueError("Exact motion reset ids must be aligned int64 tensors on the environment device.")
        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(self.table.clip_index.clips))),
            "Exact motion reset clip indices are outside the motion table.",
        )
        torch._assert_async(
            torch.all(self.table.clip_valid[clip_indices]),
            "Exact motion reset includes an invalid clip.",
        )
        self.clip_indices.index_copy_(0, env_ids, clip_indices)
        self.reset_time_ranges_seconds.index_fill_(0, env_ids, 0.0)
        self.reset_source_indices.index_fill_(0, env_ids, 0)
        self.reference_time_seconds.index_fill_(0, env_ids, 0.0)
        self._finish_bind(env_ids)

    def _finish_bind(self, env_ids: torch.Tensor) -> None:
        """Clear episode state, resolve references, and write simulator reset rows."""
        self.episode_relative_step.index_fill_(0, env_ids, 0)
        self.episode_time_seconds.index_fill_(0, env_ids, 0.0)
        if self.history is not None:
            self.history.reset(env_ids)
        self.episode_phase.index_fill_(0, env_ids, 0.0)
        self.prior_edge_applied.index_fill_(0, env_ids, False)
        if self._transition_state is not None:
            self._transition_state.reset(env_ids)
        self._resolver.bind(self._resolved_fields)
        self._write_reset(env_ids)

    def _write_reset(self, env_ids: torch.Tensor) -> None:
        """Decode the already-interpolated reference into simulator reset state."""
        root_position = self._resolved_reference["root_position"][env_ids]
        root_rotation = self._resolved_reference["root_rotation"][env_ids]
        root_linear_velocity = self._resolved_reference["root_linear_velocity"][env_ids]
        root_angular_velocity = self._resolved_reference["root_angular_velocity"][env_ids]
        decoded = self.ResetState(
            root_position=root_position,
            root_rotation_xyzw=root_rotation,
            root_linear_velocity_world=root_linear_velocity,
            root_angular_velocity_world=root_angular_velocity,
            joint_position=self._resolved_reference["joint_position"][env_ids],
            joint_velocity=self._resolved_reference["joint_velocity"][env_ids],
        )
        if self._reset_transform is not None:
            decoded = self._reset_transform(
                decoded,
                self.reset_source_indices[env_ids],
                self.table.generator,
            )
        expected_shapes = (
            (decoded.root_position, (env_ids.shape[0], 3)),
            (decoded.root_rotation_xyzw, (env_ids.shape[0], 4)),
            (decoded.root_linear_velocity_world, (env_ids.shape[0], 3)),
            (decoded.root_angular_velocity_world, (env_ids.shape[0], 3)),
            (decoded.joint_position, (env_ids.shape[0], len(self.robot.joint_names))),
            (decoded.joint_velocity, (env_ids.shape[0], len(self.robot.joint_names))),
        )
        if any(
            value.shape != shape or value.dtype is not torch.float32 or value.device != self._device
            for value, shape in expected_shapes
        ):
            raise ValueError("Motion reset state has a wrong shape, dtype, or device.")

        root_position = decoded.root_position
        if self._states_relative:
            root_position = root_position + self._env.scene.env_origins[env_ids]
        self.reset_state.root_position.index_copy_(0, env_ids, root_position)
        self.reset_state.root_rotation_xyzw.index_copy_(0, env_ids, decoded.root_rotation_xyzw)
        self.reset_state.root_linear_velocity_world.index_copy_(0, env_ids, decoded.root_linear_velocity_world)
        self.reset_state.root_angular_velocity_world.index_copy_(0, env_ids, decoded.root_angular_velocity_world)
        self.reset_state.joint_position.index_copy_(0, env_ids, decoded.joint_position)
        self.reset_state.joint_velocity.index_copy_(0, env_ids, decoded.joint_velocity)

        root_state = torch.cat(
            (
                root_position,
                decoded.root_rotation_xyzw,
                decoded.root_linear_velocity_world,
                decoded.root_angular_velocity_world,
            ),
            dim=-1,
        )
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        if self._root_velocity_frame == "link":
            self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        else:
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(
            position=decoded.joint_position,
            env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=decoded.joint_velocity,
            env_ids=env_ids,
        )

    def record_step(self) -> None:
        """Record one reached edge before terminal observation capture.

        The selected runtime writes raw evidence directly into
        :attr:`raw_evidence`; this method advances only common temporal state.
        Every row corresponds to one applied vector-environment action.
        """
        # Append the captured current node under the previous edge's
        # eligibility. Immediately after reset that mask is false, so the reset
        # seed cannot enter history. The current edge becomes eligible only for
        # the next pre-step capture.
        if self.history is not None:
            self.history.append()
        self.prior_edge_applied.fill_(True)
        self.reference_time_seconds.add_(self._step_dt)
        self.episode_time_seconds.add_(self._step_dt)
        self.episode_relative_step.add_(1)
        self._episode_step_float.copy_(self.episode_relative_step).mul_(self._episode_phase_scale)
        torch.clamp(self._episode_step_float, max=1.0, out=self.episode_phase)
        self._resolver.resolve(self._step_fields)

    def update(self, step_dt: float, command_out: torch.Tensor, error_out: torch.Tensor) -> None:
        """Expose the reference already advanced by :meth:`record_step`.

        The manager calls this after reset, which is too late to mutate reached
        evidence or history. Consequently ``step_dt`` is intentionally not used
        here and the fixed zero-width error tensor remains untouched.
        """
        del step_dt, error_out
        for destination, source in self._command_fields:
            command_out[:, destination].copy_(source)

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Accept the shared debug-visualization lifecycle without markers."""
        del debug_vis

    def debug_visualize(self, env: ManagerBasedRLEnv) -> None:
        """Leave motion visualization to explicit observation/reference tools."""
        del env
