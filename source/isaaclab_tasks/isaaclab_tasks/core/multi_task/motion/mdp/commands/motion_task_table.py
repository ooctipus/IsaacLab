# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion trajectory tensors and task descriptors owned by one command table."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import torch

from ...data._identity import canonical_sha256, validate_nonempty, validate_sha256
from ...data.clip_index import MotionClipIndex
from ...data.sample_grid import MotionSampleGrid
from ...data.skeleton import MotionSkeleton

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg

Interpolation = Literal["linear", "slerp", "left"]


class MotionFrameSource(Protocol):
    """Decoded clips consumed once in their declared order."""

    def inspect(self) -> MotionClipIndex:
        """Return compact ordered clip metadata."""

    def clips(self) -> Iterator[tuple[str, Mapping[str, object]]]:
        """Yield decoded clips in the order returned by :meth:`inspect`."""

    def close(self) -> None:
        """Release source-format resources retained during table construction."""


class MotionFrameBuilder(Protocol):
    """Convert source clips into simulator-ordered trajectory tensors.

    The builder is created from the selected preset and live articulation. It
    resolves robot names and ordering once, before the table exists. The table
    therefore owns trajectory data, not a second robot model.
    """

    source_skeleton: MotionSkeleton
    joint_names: tuple[str, ...]
    reference_frame_names: tuple[str, ...]
    version: str
    construction_identity_sha256: str

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
        """Allocate exact-capacity output tensors for all source frames."""

    def build_frames(self, fields: Mapping[str, object], *, device: str | torch.device) -> MotionTaskTable.Frames:
        """Convert one decoded clip to simulator-ordered trajectory tensors."""


_FRAME_INTERPOLATION: dict[str, Interpolation] = {
    "root_position": "linear",
    "root_rotation": "slerp",
    "root_linear_velocity": "linear",
    "root_angular_velocity": "linear",
    "joint_position": "linear",
    "joint_velocity": "linear",
    "body_position": "linear",
    "body_rotation": "slerp",
    "body_linear_velocity": "linear",
    "body_angular_velocity": "linear",
    "observation": "left",
}


def _quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
    """Interpolate unit quaternions along the sign-invariant shortest arc."""
    dot = torch.sum(q0 * q1, dim=-1)
    q1 = torch.where((dot < 0.0).unsqueeze(-1), -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)
    while fraction.ndim < dot.ndim:
        fraction = fraction.unsqueeze(-1)

    angle = torch.acos(dot)
    sine = torch.sin(angle)
    safe_sine = sine.clamp_min(torch.finfo(q0.dtype).eps)
    first_weight = torch.sin((1.0 - fraction) * angle) / safe_sine
    second_weight = torch.sin(fraction * angle) / safe_sine
    spherical = first_weight.unsqueeze(-1) * q0 + second_weight.unsqueeze(-1) * q1
    linear = torch.nn.functional.normalize(q0 + fraction.unsqueeze(-1) * (q1 - q0), dim=-1)
    return torch.where((dot > 0.9995).unsqueeze(-1), linear, spherical)


def _table_identity(
    clip_index: MotionClipIndex,
    frames: MotionTaskTable.Frames,
    joint_names: tuple[str, ...],
    reference_frame_names: tuple[str, ...],
    frame_builder_version: str,
    frame_builder_identity_sha256: str,
    task_row_mode: Literal["source_frames", "clip_time_ranges"],
    reset_sources: tuple[tuple[str, float], ...],
    expert_sample_grid: MotionSampleGrid,
) -> str:
    """Return deterministic trajectory-data provenance without robot ownership."""
    validate_nonempty("frame_builder_version", frame_builder_version)
    validate_sha256("frame_builder_identity_sha256", frame_builder_identity_sha256)
    stored_column_shapes = {name: tuple(frames.field(name).shape[1:]) for name in frames.stored_fields}
    return canonical_sha256(
        {
            "source_content_hash": clip_index.content_identity_sha256,
            "source_skeleton_hash": clip_index.skeleton_sha256,
            "frame_builder_version": frame_builder_version,
            "joint_names": joint_names,
            "reference_frame_names": reference_frame_names,
            "frame_builder_identity_sha256": frame_builder_identity_sha256,
            "task_row_mode": task_row_mode,
            "reset_sources": reset_sources,
            "expert_sample_grid": {
                "mode": expert_sample_grid.mode.value,
                "step_seconds": expert_sample_grid.step_seconds,
            },
            "stored_columns": stored_column_shapes,
            "root_storage": frames.root_storage,
        }
    )


def _task_rows(
    clip_index: MotionClipIndex,
    device: torch.device,
    mode: Literal["source_frames", "clip_time_ranges"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic descriptor rows from one controlled mode."""
    if mode not in ("source_frames", "clip_time_ranges"):
        raise ValueError("task_row_mode must be 'source_frames' or 'clip_time_ranges'.")
    clip_valid = torch.tensor([clip.valid for clip in clip_index.clips], dtype=torch.bool, device=device)
    frame_counts = torch.tensor([clip.frame_count for clip in clip_index.clips], dtype=torch.int64, device=device)
    source_fps = torch.tensor([clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device)
    valid_clips = torch.arange(len(clip_index.clips), device=device)[clip_valid]
    if mode == "clip_time_ranges":
        end = (frame_counts[valid_clips] - 1) / source_fps[valid_clips]
        return valid_clips, torch.stack((torch.zeros_like(end), end), dim=-1)

    valid_frame_counts = frame_counts[valid_clips]
    clips = torch.repeat_interleave(valid_clips, valid_frame_counts)
    output_offsets = torch.cumsum(valid_frame_counts, dim=0) - valid_frame_counts
    repeated_output_offsets = torch.repeat_interleave(output_offsets, valid_frame_counts)
    local_frames = torch.arange(clips.shape[0], device=device) - repeated_output_offsets
    times = local_frames / source_fps[clips]
    return clips, torch.stack((times, times), dim=-1)


def _reset_source_data(
    reset_sources: tuple[tuple[str, float], ...],
    device: torch.device,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Validate ordered reset-source data and place probabilities on the table device."""
    if not isinstance(reset_sources, tuple) or not reset_sources:
        raise ValueError("reset_sources must be a nonempty tuple of name/probability pairs.")
    names: list[str] = []
    probabilities: list[float] = []
    for source in reset_sources:
        if not isinstance(source, tuple) or len(source) != 2:
            raise ValueError("Each reset source must be one (name, probability) tuple.")
        name, probability = source
        if not isinstance(name, str) or not name:
            raise ValueError("Reset-source names must be nonempty strings.")
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ValueError("Reset-source probabilities must be real scalars.")
        probability = float(probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("Reset-source probabilities must be finite and nonnegative.")
        names.append(name)
        probabilities.append(probability)
    if len(set(names)) != len(names):
        raise ValueError("Reset-source names must be unique.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("Reset-source probabilities must sum to one.")
    return tuple(names), torch.tensor(probabilities, dtype=torch.float32, device=device)


class MotionTaskTable:
    """Exact-capacity trajectory tensors plus selectable motion descriptors.

    Robot identity, kinematic structure, control, and defaults remain owned by
    the selected preset and scene articulation. Builders resolve source data to
    live simulator order once; this table then owns only concrete task tensors,
    clip boundaries, descriptor rows, priorities, and sampling state.
    """

    @dataclass(frozen=True, slots=True)
    class Frames:
        """Concrete trajectory columns sharing one frame axis and device.

        Every present tensor is contiguous, detached float32, and read-only
        after construction by contract. Joint columns are in live simulator
        order. Positions and velocities are world-frame SI values, and rotations
        are xyzw quaternions. Root columns are required simulator-reset facts;
        body columns are optional reference/evidence frames ordered by the
        table's ``reference_frame_names`` and may append non-rigid derived
        frames. ``observation`` is an optional already-derived source observation
        whose temporal rule is left-sample.
        """

        root_position: torch.Tensor | None = None
        """Root-link position [m], shape [frame_count, 3], float."""

        root_rotation: torch.Tensor | None = None
        """Root-link xyzw orientation, shape [frame_count, 4], float."""

        root_linear_velocity: torch.Tensor | None = None
        """Root-link linear velocity [m/s], shape [frame_count, 3], float."""

        root_angular_velocity: torch.Tensor | None = None
        """Root-link angular velocity [rad/s], shape [frame_count, 3], float."""

        joint_position: torch.Tensor | None = None
        """Simulator-ordered joint positions [rad], shape [frame_count, joint_count], float."""

        joint_velocity: torch.Tensor | None = None
        """Simulator-ordered joint velocities [rad/s], shape [frame_count, joint_count], float."""

        body_position: torch.Tensor | None = None
        """Reference-frame positions [m], shape [frame_count, reference_frame_count, 3], float."""

        body_rotation: torch.Tensor | None = None
        """Reference-frame xyzw orientations, shape [frame_count, reference_frame_count, 4], float."""

        body_linear_velocity: torch.Tensor | None = None
        """Reference-frame linear velocities [m/s], shape [frame_count, reference_frame_count, 3], float."""

        body_angular_velocity: torch.Tensor | None = None
        """Reference-frame angular velocities [rad/s], shape [frame_count, reference_frame_count, 3], float."""

        observation: torch.Tensor | None = None
        """Optional derived observation, shape [frame_count, observation_width], float."""

        _NAMES: ClassVar[tuple[str, ...]] = tuple(_FRAME_INTERPOLATION)
        _ROOT_FIELDS: ClassVar[tuple[str, ...]] = (
            "root_position",
            "root_rotation",
            "root_linear_velocity",
            "root_angular_velocity",
        )

        def __post_init__(self) -> None:
            """Validate fixed trajectory semantics and a shared frame axis."""
            values = {name: getattr(self, name) for name in self._NAMES}
            present = {name: value for name, value in values.items() if value is not None}
            if not present:
                raise ValueError("Motion trajectory frames must contain at least one column.")
            first = next(iter(present.values()))
            frame_count = first.shape[0] if first.ndim > 0 else -1
            for name, value in present.items():
                if (
                    value.ndim < 2
                    or value.shape[0] != frame_count
                    or value.dtype is not torch.float32
                    or value.device != first.device
                    or not value.is_contiguous()
                    or value.requires_grad
                ):
                    raise ValueError(
                        f"Trajectory column {name!r} must be contiguous detached float32 with "
                        f"frame axis {frame_count} on {first.device}."
                    )

            self._validate_group(values, ("joint_position", "joint_velocity"), required=True)
            self._validate_group(
                values,
                ("root_position", "root_rotation", "root_linear_velocity", "root_angular_velocity"),
            )
            self._validate_group(
                values,
                ("body_position", "body_rotation", "body_linear_velocity", "body_angular_velocity"),
            )
            root_stored = values["root_position"] is not None
            body_stored = values["body_position"] is not None
            if root_stored == body_stored:
                raise ValueError("Trajectory frames require exactly one root owner: explicit root or body row zero.")

            expected_tail = {
                "root_position": (3,),
                "root_rotation": (4,),
                "root_linear_velocity": (3,),
                "root_angular_velocity": (3,),
            }
            for name, shape in expected_tail.items():
                value = values[name]
                if value is not None and value.shape[1:] != shape:
                    raise ValueError(f"Trajectory column {name!r} must end in {shape}.")

            joint_position = values["joint_position"]
            joint_velocity = values["joint_velocity"]
            assert joint_position is not None and joint_velocity is not None
            if joint_position.ndim != 2 or joint_velocity.shape != joint_position.shape:
                raise ValueError("Joint position and velocity must share shape [frame_count, joint_count].")

            body_position = values["body_position"]
            if body_position is not None:
                body_rotation = values["body_rotation"]
                body_linear_velocity = values["body_linear_velocity"]
                body_angular_velocity = values["body_angular_velocity"]
                assert body_rotation is not None
                assert body_linear_velocity is not None
                assert body_angular_velocity is not None
                body_shape = body_position.shape[:2]
                if (
                    body_position.shape != (*body_shape, 3)
                    or body_rotation.shape != (*body_shape, 4)
                    or body_linear_velocity.shape != (*body_shape, 3)
                    or body_angular_velocity.shape != (*body_shape, 3)
                ):
                    raise ValueError("Reference-frame columns must share [frame_count, reference_frame_count, ...].")

            observation = values["observation"]
            if observation is not None and observation.ndim != 2:
                raise ValueError("Source observations must have shape [frame_count, observation_width].")

        def validate_values(self) -> None:
            """Reject nonfinite trajectory values and nonunit stored quaternions.

            This validation runs only while a table is constructed. Runtime
            reference lookup therefore retains its allocation-free hot path.
            """
            for name in self.stored_fields:
                value = self.field(name)
                if not bool(torch.all(torch.isfinite(value))):
                    raise ValueError(f"Trajectory column {name!r} must contain only finite values.")
            for name in ("root_rotation", "body_rotation"):
                value = getattr(self, name)
                if value is None:
                    continue
                norms = torch.linalg.vector_norm(value, dim=-1)
                if not torch.allclose(norms, torch.ones_like(norms), rtol=1.0e-5, atol=1.0e-5):
                    raise ValueError(f"Trajectory column {name!r} must contain unit quaternions.")

        @staticmethod
        def _validate_group(
            values: dict[str, torch.Tensor | None], names: tuple[str, ...], *, required: bool = False
        ) -> None:
            present = tuple(values[name] is not None for name in names)
            if required and not all(present):
                raise ValueError(f"Trajectory columns {names} are required together.")
            if any(present) and not all(present):
                raise ValueError(f"Trajectory columns {names} must be all present or all absent.")

        @property
        def stored_fields(self) -> tuple[str, ...]:
            """Names of concrete columns present in this trajectory."""
            return tuple(name for name in self._NAMES if getattr(self, name) is not None)

        @property
        def available_fields(self) -> tuple[str, ...]:
            """Names of logical columns available to runtime consumers."""
            return tuple(name for name in self._NAMES if getattr(self, name) is not None or name in self._ROOT_FIELDS)

        @property
        def root_storage(self) -> Literal["explicit", "body_row_zero"]:
            """Physical owner of the logical root columns."""
            return "explicit" if self.root_position is not None else "body_row_zero"

        @property
        def frame_count(self) -> int:
            """Number of stored trajectory frames."""
            return self.field(self.stored_fields[0]).shape[0]

        @property
        def device(self) -> torch.device:
            """Shared tensor device."""
            return self.field(self.stored_fields[0]).device

        @property
        def memory_bytes(self) -> int:
            """Resident bytes in all concrete trajectory columns."""
            return sum(self.field(name).numel() * self.field(name).element_size() for name in self.stored_fields)

        def field(self, name: str) -> torch.Tensor:
            """Return one stored column or a logical root view over body row zero."""
            if name not in self._NAMES:
                raise KeyError(f"Unknown motion trajectory field: {name!r}.")
            value = getattr(self, name)
            if value is None and name in self._ROOT_FIELDS:
                body_value = getattr(self, name.replace("root_", "body_", 1))
                if body_value is not None:
                    return body_value[:, 0]
            if value is None:
                raise KeyError(f"Motion trajectory field {name!r} is absent in this composition.")
            return value

        def copy_clip_(self, start: int, end: int, source: MotionTaskTable.Frames) -> None:
            """Copy one built clip into its exact destination range."""
            if source.stored_fields != self.stored_fields:
                raise ValueError(
                    "Built clip columns differ from the allocated table columns: "
                    f"expected {self.stored_fields}, got {source.stored_fields}."
                )
            if source.frame_count != end - start or source.device != self.device:
                raise ValueError("Built clip frame count or device differs from its destination range.")
            for name in self.stored_fields:
                destination = self.field(name)[start:end]
                value = source.field(name)
                if value.shape[1:] != destination.shape[1:]:
                    raise ValueError(
                        f"Built {name!r} shape {tuple(value.shape)} differs from "
                        f"destination {tuple(destination.shape)}."
                    )
                destination.copy_(value)

    class Writer:
        """Single-pass exact-capacity writer that seals into one task table."""

        __slots__ = (
            "_clip_index",
            "_finished",
            "_expert_sample_grid",
            "_frame_builder_identity_sha256",
            "_frame_builder_version",
            "_frames",
            "_joint_names",
            "_next_clip",
            "_reference_frame_names",
            "_reset_sources",
            "_seed",
            "_task_row_mode",
        )

        def __init__(
            self,
            clip_index: MotionClipIndex,
            frames: MotionTaskTable.Frames,
            joint_names: tuple[str, ...],
            reference_frame_names: tuple[str, ...],
            frame_builder_version: str,
            frame_builder_identity_sha256: str,
            task_row_mode: Literal["source_frames", "clip_time_ranges"],
            reset_sources: tuple[tuple[str, float], ...],
            expert_sample_grid: MotionSampleGrid,
            seed: int,
        ) -> None:
            if frames.frame_count != clip_index.total_frames:
                raise ValueError("Allocated trajectory capacity must equal the source frame count exactly.")
            validate_nonempty("frame_builder_version", frame_builder_version)
            validate_sha256("frame_builder_identity_sha256", frame_builder_identity_sha256)
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError("Motion task seed must be an integer.")
            self._clip_index = clip_index
            self._frames = frames
            self._frame_builder_version = frame_builder_version
            self._joint_names = joint_names
            self._reference_frame_names = reference_frame_names
            self._frame_builder_identity_sha256 = frame_builder_identity_sha256
            self._task_row_mode = task_row_mode
            self._reset_sources = reset_sources
            self._expert_sample_grid = expert_sample_grid
            self._seed = seed
            self._next_clip = 0
            self._finished = False

        @property
        def frames(self) -> MotionTaskTable.Frames:
            """Preallocated writable construction tensors."""
            return self._frames

        def write_clip(self, clip_id: str, frames: MotionTaskTable.Frames) -> None:
            """Copy the next built clip into its exact flat range."""
            if self._finished:
                raise RuntimeError("The writer is already finished.")
            if self._next_clip == len(self._clip_index.clips):
                raise RuntimeError("Every declared clip is already written.")
            clip = self._clip_index.clips[self._next_clip]
            if clip_id != clip.clip_id:
                raise ValueError(f"Writer expected clip {clip.clip_id!r}, got {clip_id!r}.")
            start = self._clip_index.offsets[self._next_clip]
            end = self._clip_index.offsets[self._next_clip + 1]
            self._frames.copy_clip_(start, end, frames)
            self._next_clip += 1

        def finish(self) -> MotionTaskTable:
            """Seal complete trajectory and descriptor tensors without copying."""
            if self._finished:
                raise RuntimeError("The writer is already finished.")
            if self._next_clip != len(self._clip_index.clips):
                raise ValueError(
                    f"Motion table is incomplete: wrote {self._next_clip} of {len(self._clip_index.clips)} clips."
                )
            table = MotionTaskTable.from_storage(
                self._clip_index,
                self._frames,
                self._joint_names,
                self._reference_frame_names,
                self._frame_builder_version,
                self._frame_builder_identity_sha256,
                self._task_row_mode,
                self._reset_sources,
                self._expert_sample_grid,
                seed=self._seed,
            )
            self._finished = True
            return table

    class ReferenceView:
        """Batched continuous-time reference lookup with explicit tail validity."""

        __slots__ = (
            "_alpha",
            "_global_frame0",
            "_global_frame1",
            "table",
            "clip_indices",
            "local_frame0",
            "local_frame1",
            "tail_valid",
            "time_seconds",
        )

        def __init__(
            self,
            table: MotionTaskTable,
            clip_indices: torch.Tensor,
            time_seconds: torch.Tensor,
            local_frame0: torch.Tensor,
            local_frame1: torch.Tensor,
            global_frame0: torch.Tensor,
            global_frame1: torch.Tensor,
            alpha: torch.Tensor,
            tail_valid: torch.Tensor,
        ) -> None:
            self.table = table
            self.clip_indices = clip_indices
            self.time_seconds = time_seconds
            self.local_frame0 = local_frame0
            self.local_frame1 = local_frame1
            self._global_frame0 = global_frame0
            self._global_frame1 = global_frame1
            self._alpha = alpha
            self.tail_valid = tail_valid

        def field(self, name: str) -> torch.Tensor:
            """Interpolate one named trajectory field at every requested time."""
            values = self.table.field(name)
            value0 = values[self._global_frame0]
            interpolation = self.table.interpolation(name)
            if interpolation == "left":
                return value0
            value1 = values[self._global_frame1]
            if interpolation == "slerp":
                return _quaternion_slerp(value0, value1, self._alpha)
            fraction = self._alpha
            while fraction.ndim < value0.ndim:
                fraction = fraction.unsqueeze(-1)
            return torch.lerp(value0, value1, fraction)

    __slots__ = (
        "_cache_identity",
        "_clip_ids",
        "_clip_index",
        "_clip_offsets",
        "_clip_valid",
        "_frame_builder_identity_sha256",
        "_frame_builder_version",
        "_frame_counts",
        "_frames",
        "_joint_names",
        "_reference_frame_names",
        "_row_priorities",
        "_sealed",
        "_source_fps",
        "_task_row_mode",
        "clip_indices",
        "clip_priorities",
        "generator",
        "expert_sample_grid",
        "reset_source_probabilities",
        "reset_source_names",
        "reset_time_ranges_seconds",
        "seed",
    )

    def __init__(
        self,
        clip_index: MotionClipIndex,
        frames: MotionTaskTable.Frames,
        joint_names: tuple[str, ...],
        reference_frame_names: tuple[str, ...],
        frame_builder_version: str,
        frame_builder_identity_sha256: str,
        task_row_mode: Literal["source_frames", "clip_time_ranges"],
        reset_sources: tuple[tuple[str, float], ...],
        expert_sample_grid: MotionSampleGrid,
        *,
        seed: int,
    ) -> None:
        if frames.frame_count != clip_index.total_frames:
            raise ValueError("Trajectory capacity must equal the declared source frame count exactly.")
        validate_nonempty("frame_builder_version", frame_builder_version)
        if (
            not isinstance(joint_names, tuple)
            or not joint_names
            or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("Trajectory joint_names must be a nonempty tuple of unique names.")
        if len(joint_names) != frames.field("joint_position").shape[1]:
            raise ValueError("Trajectory joint_names must match the joint-column axis exactly.")
        if (
            not isinstance(reference_frame_names, tuple)
            or any(not isinstance(name, str) or not name for name in reference_frame_names)
            or len(set(reference_frame_names)) != len(reference_frame_names)
        ):
            raise ValueError("Trajectory reference_frame_names must be a tuple of unique names.")
        body_position = frames.body_position
        if body_position is None:
            if reference_frame_names:
                raise ValueError("Trajectory reference-frame names require reference-frame columns.")
        elif len(reference_frame_names) != body_position.shape[1]:
            raise ValueError("Trajectory reference_frame_names must match the reference-frame column axis exactly.")

        validate_sha256("frame_builder_identity_sha256", frame_builder_identity_sha256)
        if not isinstance(expert_sample_grid, MotionSampleGrid):
            raise TypeError("expert_sample_grid must be MotionSampleGrid.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("Motion task seed must be an integer.")
        frames.validate_values()

        device = frames.device
        clip_offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=device)
        frame_counts = torch.tensor([clip.frame_count for clip in clip_index.clips], dtype=torch.int64, device=device)
        source_fps = torch.tensor([clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device)
        clip_valid = torch.tensor([clip.valid for clip in clip_index.clips], dtype=torch.bool, device=device)
        clip_indices, reset_time_ranges_seconds = _task_rows(clip_index, device, task_row_mode)
        reset_source_names, reset_source_probabilities = _reset_source_data(reset_sources, device)
        normalized_reset_sources = tuple((name, float(probability)) for name, probability in reset_sources)
        num_tasks = clip_indices.shape[0]
        if (
            clip_indices.ndim != 1
            or clip_indices.dtype is not torch.int64
            or reset_time_ranges_seconds.shape != (num_tasks, 2)
            or not reset_time_ranges_seconds.is_floating_point()
            or reset_source_probabilities.ndim != 1
            or reset_source_probabilities.shape[0] < 1
            or not reset_source_probabilities.is_floating_point()
            or num_tasks == 0
        ):
            raise ValueError(
                "Motion tasks require clip [N], reset-time-range [N, 2], and reset-source-probability [S] tensors."
            )
        if any(
            value.device != device for value in (clip_indices, reset_time_ranges_seconds, reset_source_probabilities)
        ):
            raise ValueError("Motion task and trajectory tensors must share one device.")

        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(clip_index.clips))),
            "Motion task clip indices are outside the stored clips.",
        )
        torch._assert_async(torch.all(clip_valid[clip_indices]), "Motion task rows include an invalid source clip.")
        low, high = reset_time_ranges_seconds.unbind(-1)
        clip_end = (frame_counts[clip_indices] - 1) / source_fps[clip_indices]
        torch._assert_async(
            torch.all(torch.isfinite(reset_time_ranges_seconds) & (reset_time_ranges_seconds >= 0.0)),
            "Motion reset-time ranges must be finite and nonnegative.",
        )
        torch._assert_async(
            torch.all((high >= low) & (high <= clip_end)),
            "A motion reset-time range crosses its clip boundary.",
        )
        torch._assert_async(
            torch.all(torch.isfinite(reset_source_probabilities) & (reset_source_probabilities >= 0.0)),
            "Motion reset-source probabilities must be finite and nonnegative.",
        )
        torch._assert_async(
            torch.isclose(reset_source_probabilities.sum(), reset_source_probabilities.new_tensor(1.0)),
            "Motion reset-source probabilities must sum to one.",
        )
        valid_indices = tuple(index for index, clip in enumerate(clip_index.clips) if clip.valid)
        expected = torch.tensor(valid_indices, dtype=torch.int64, device=device)
        if not torch.equal(torch.unique(clip_indices, sorted=True), expected):
            raise ValueError("Motion task rows must cover every valid clip in stable source order.")

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        object.__setattr__(self, "_clip_index", clip_index)
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_frame_builder_version", frame_builder_version)
        object.__setattr__(self, "_frame_builder_identity_sha256", frame_builder_identity_sha256)
        object.__setattr__(self, "_joint_names", joint_names)
        object.__setattr__(self, "_reference_frame_names", reference_frame_names)
        object.__setattr__(self, "_task_row_mode", task_row_mode)
        object.__setattr__(self, "_clip_offsets", clip_offsets)
        object.__setattr__(self, "_frame_counts", frame_counts)
        object.__setattr__(self, "_source_fps", source_fps)
        object.__setattr__(self, "_clip_valid", clip_valid)
        object.__setattr__(self, "clip_indices", clip_indices)
        object.__setattr__(self, "reset_time_ranges_seconds", reset_time_ranges_seconds)
        object.__setattr__(self, "reset_source_probabilities", reset_source_probabilities)
        object.__setattr__(self, "reset_source_names", reset_source_names)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "expert_sample_grid", expert_sample_grid)
        object.__setattr__(self, "generator", generator)
        object.__setattr__(self, "_clip_ids", tuple(clip_index.clip_ids[index] for index in valid_indices))
        object.__setattr__(self, "clip_priorities", torch.ones(len(valid_indices), dtype=torch.float32, device=device))
        object.__setattr__(self, "_row_priorities", torch.empty(0, dtype=torch.float32, device=device))
        object.__setattr__(
            self,
            "_cache_identity",
            _table_identity(
                clip_index,
                frames,
                joint_names,
                reference_frame_names,
                frame_builder_version,
                frame_builder_identity_sha256,
                task_row_mode,
                normalized_reset_sources,
                expert_sample_grid,
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MotionTaskTable metadata is immutable.")
        object.__setattr__(self, name, value)

    @classmethod
    def writer(
        cls,
        clip_index: MotionClipIndex,
        frames: MotionTaskTable.Frames,
        joint_names: tuple[str, ...],
        reference_frame_names: tuple[str, ...],
        frame_builder_version: str,
        frame_builder_identity_sha256: str,
        task_row_mode: Literal["source_frames", "clip_time_ranges"],
        reset_sources: tuple[tuple[str, float], ...],
        expert_sample_grid: MotionSampleGrid,
        *,
        seed: int,
    ) -> Writer:
        """Bind exact-capacity construction tensors to a single-pass writer."""
        return cls.Writer(
            clip_index,
            frames,
            joint_names,
            reference_frame_names,
            frame_builder_version,
            frame_builder_identity_sha256,
            task_row_mode,
            reset_sources,
            expert_sample_grid,
            seed,
        )

    @classmethod
    def build(
        cls,
        source: MotionFrameSource,
        clip_index: MotionClipIndex,
        frame_builder: MotionFrameBuilder,
        task_row_mode: Literal["source_frames", "clip_time_ranges"],
        reset_sources: tuple[tuple[str, float], ...],
        expert_sample_grid: MotionSampleGrid,
        *,
        seed: int,
        device: str | torch.device,
    ) -> MotionTaskTable:
        """Stream one decoded source directly into one exact-capacity task table."""
        source_clip_index = source.inspect()
        if source_clip_index.identity_sha256 != clip_index.identity_sha256:
            raise ValueError("Motion source clip index identity differs from the declared clip index identity.")
        if clip_index.skeleton_sha256 != frame_builder.source_skeleton.identity_sha256:
            raise ValueError("The source clip index and frame-builder skeleton identities differ.")
        frames = frame_builder.allocate(clip_index.total_frames, device=device)
        writer = cls.writer(
            clip_index,
            frames,
            frame_builder.joint_names,
            frame_builder.reference_frame_names,
            frame_builder.version,
            frame_builder.construction_identity_sha256,
            task_row_mode,
            reset_sources,
            expert_sample_grid,
            seed=seed,
        )
        for clip_id, fields in source.clips():
            writer.write_clip(clip_id, frame_builder.build_frames(fields, device=device))
        return writer.finish()

    @classmethod
    def from_storage(
        cls,
        clip_index: MotionClipIndex,
        frames: MotionTaskTable.Frames,
        joint_names: tuple[str, ...],
        reference_frame_names: tuple[str, ...],
        frame_builder_version: str,
        frame_builder_identity_sha256: str,
        task_row_mode: Literal["source_frames", "clip_time_ranges"],
        reset_sources: tuple[tuple[str, float], ...],
        expert_sample_grid: MotionSampleGrid,
        *,
        seed: int,
    ) -> MotionTaskTable:
        """Bind validated trajectory and descriptor tensors without copying."""
        return cls(
            clip_index,
            frames,
            joint_names,
            reference_frame_names,
            frame_builder_version,
            frame_builder_identity_sha256,
            task_row_mode,
            reset_sources,
            expert_sample_grid,
            seed=seed,
        )

    @property
    def frames(self) -> Frames:
        """Concrete trajectory tensor owner."""
        return self._frames

    @property
    def clip_index(self) -> MotionClipIndex:
        """Ordered source clip metadata."""
        return self._clip_index

    @property
    def frame_builder_version(self) -> str:
        """Readable version of source-to-trajectory conversion."""
        return self._frame_builder_version

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of every joint trajectory column."""
        return self._joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Ordered semantic labels of the optional reference-frame columns."""
        return self._reference_frame_names

    @property
    def frame_builder_identity_sha256(self) -> str:
        """Construction identity closing reference kinematics and ordered mappings."""
        return self._frame_builder_identity_sha256

    @property
    def task_row_mode(self) -> Literal["source_frames", "clip_time_ranges"]:
        """Controlled rule used to derive selectable descriptor rows."""
        return self._task_row_mode

    @property
    def cache_identity(self) -> str:
        """Deterministic source, builder, columns, and timing identity."""
        return self._cache_identity

    @property
    def clip_offsets(self) -> torch.Tensor:
        """Clip prefix offsets on the trajectory frame axis."""
        return self._clip_offsets

    @property
    def frame_counts(self) -> torch.Tensor:
        """Frames per clip."""
        return self._frame_counts

    @property
    def source_fps(self) -> torch.Tensor:
        """Source sample rate [Hz] per clip."""
        return self._source_fps

    @property
    def clip_valid(self) -> torch.Tensor:
        """Declared source validity per clip."""
        return self._clip_valid

    @property
    def clip_ids(self) -> tuple[str, ...]:
        """Valid clip identifiers covered by selectable descriptor rows."""
        return self._clip_ids

    @property
    def source_clip_ids(self) -> tuple[str, ...]:
        """All clip identifiers in trajectory storage order."""
        return self._clip_index.clip_ids

    @property
    def splits(self) -> tuple[str, ...]:
        """Dataset split per stored clip."""
        return tuple(clip.split for clip in self._clip_index.clips)

    @property
    def device(self) -> torch.device:
        """Shared trajectory and descriptor device."""
        return self._frames.device

    @property
    def memory_bytes(self) -> int:
        """Resident bytes owned by trajectory, clip, descriptor, and priority tensors."""
        tensors = (
            self._clip_offsets,
            self._frame_counts,
            self._source_fps,
            self._clip_valid,
            self.clip_indices,
            self.reset_time_ranges_seconds,
            self.reset_source_probabilities,
            self.clip_priorities,
            self._row_priorities,
        )
        return self._frames.memory_bytes + sum(value.numel() * value.element_size() for value in tensors)

    @property
    def num_tasks(self) -> int:
        """Number of selectable motion descriptors."""
        return self.clip_indices.shape[0]

    @property
    def clip_priorities_active(self) -> bool:
        """Whether row sampling currently uses external clip priorities."""
        return self._row_priorities.numel() > 0

    def field(self, name: str) -> torch.Tensor:
        """Return one concrete trajectory tensor without indirection or copying."""
        return self._frames.field(name)

    def interpolation(self, name: str) -> Interpolation:
        """Return the fixed temporal rule for one concrete trajectory field."""
        self._frames.field(name)
        return _FRAME_INTERPOLATION[name]

    def _validate_clip_indices(self, clip_indices: torch.Tensor) -> None:
        if clip_indices.ndim != 1 or clip_indices.dtype is not torch.int64 or clip_indices.device != self.device:
            raise ValueError("clip_indices must be a 1D int64 tensor on the table device.")
        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(self._clip_index.clips))),
            "clip_indices are outside the motion table.",
        )
        torch._assert_async(torch.all(self._clip_valid[clip_indices]), "clip_indices include an invalid source clip.")

    def reference_view(self, clip_indices: torch.Tensor, time_seconds: torch.Tensor) -> ReferenceView:
        """Resolve clamped continuous-time reference interpolation."""
        self._validate_clip_indices(clip_indices)
        if (
            time_seconds.ndim != 1
            or not time_seconds.is_floating_point()
            or time_seconds.device != self.device
            or time_seconds.shape != clip_indices.shape
        ):
            raise ValueError("time_seconds must match clip_indices as a floating tensor on the table device.")
        torch._assert_async(torch.all(torch.isfinite(time_seconds)), "time_seconds must be finite.")

        frame_counts = self._frame_counts[clip_indices]
        position = time_seconds * self._source_fps[clip_indices]
        last = frame_counts - 1
        tail_valid = (position >= 0.0) & (position <= last)
        clamped = torch.minimum(position.clamp_min(0.0), last.to(position.dtype))
        local_frame0 = torch.floor(clamped).to(torch.int64)
        local_frame1 = torch.minimum(local_frame0 + 1, last)
        alpha = clamped - local_frame0.to(clamped.dtype)
        offset = self._clip_offsets[clip_indices]
        return self.ReferenceView(
            self,
            clip_indices,
            time_seconds,
            local_frame0,
            local_frame1,
            offset + local_frame0,
            offset + local_frame1,
            alpha,
            tail_valid,
        )

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample table rows from mutable stable-clip priorities."""
        if not self.clip_priorities_active:
            return torch.randint(self.num_tasks, (count,), device=self.device, generator=self.generator)
        return torch.multinomial(self._row_priorities, count, replacement=True, generator=self.generator)

    def validate_clip_priorities(self, clip_ids: tuple[str, ...], priorities: torch.Tensor) -> None:
        """Validate a stable-ID priority update without mutating table state."""
        if clip_ids != self.clip_ids:
            raise ValueError("Motion priority clip ids do not match the task table.")
        if (
            priorities.shape != self.clip_priorities.shape
            or not priorities.is_floating_point()
            or priorities.device != self.device
            or priorities.requires_grad
        ):
            raise ValueError("Motion priorities must be detached floating point values on the table device.")
        if not torch.all(torch.isfinite(priorities)):
            raise ValueError("Motion priorities must be finite.")
        if torch.any(priorities < 0.0) or not torch.any(priorities > 0.0):
            raise ValueError("Motion priorities must be non-negative with positive total mass.")

    def set_clip_priorities(self, clip_ids: tuple[str, ...], priorities: torch.Tensor) -> None:
        """Commit one validated stable-ID priority update."""
        self.validate_clip_priorities(clip_ids, priorities)
        self.clip_priorities.copy_(priorities)
        valid_clip_indices = torch.arange(len(self._clip_index.clips), device=self.device)[self._clip_valid]
        priorities_by_clip = torch.zeros(len(self._clip_index.clips), device=self.device)
        priorities_by_clip[valid_clip_indices] = priorities
        row_priorities = priorities_by_clip[self.clip_indices]
        if self._row_priorities.shape != row_priorities.shape:
            object.__setattr__(self, "_row_priorities", row_priorities)
        else:
            self._row_priorities.copy_(row_priorities)

    def reset_clip_priorities(self) -> None:
        """Restore the allocation-free uniform row-sampling state."""
        self.clip_priorities.fill_(1.0)
        object.__setattr__(self, "_row_priorities", self.clip_priorities.new_empty(0))

    def sample_reset_sources(self, count: int) -> torch.Tensor:
        """Sample reset-source indices independently from motion descriptors."""
        return torch.multinomial(
            self.reset_source_probabilities,
            count,
            replacement=True,
            generator=self.generator,
        )

    def select(self, task_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source clip indices and continuous reset ranges [s]."""
        if task_rows.ndim != 1 or task_rows.dtype is not torch.int64 or task_rows.device != self.device:
            raise ValueError("task_rows must be a one-dimensional int64 tensor on the table device.")
        return self.clip_indices[task_rows], self.reset_time_ranges_seconds[task_rows]


def build_motion_task_table(cfg: StateCommandCfg, env: ManagerBasedRLEnv) -> MotionTaskTable:
    """Stream the selected source split directly into its runtime task table."""
    table_cfg = cfg.task_table
    source_cfg = table_cfg.source
    split = source_cfg.train if table_cfg.motion_split == "train" else source_cfg.evaluation
    source = source_cfg.open_split(table_cfg.source_artifact_root, split)
    try:
        clip_index = source.inspect()
        if (
            clip_index.source_content_sha256 != split.source_content_sha256
            or len(clip_index.clips) != split.clip_count
            or clip_index.total_frames != split.frame_count
        ):
            raise ValueError(
                "Motion source identity/counts differ from the selected split: "
                f"hash={clip_index.source_content_sha256}, clips={len(clip_index.clips)}, "
                f"frames={clip_index.total_frames}."
            )
        seed = env.cfg.seed
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("Motion environments require an integer seed.")
        return MotionTaskTable.build(
            source,
            clip_index,
            table_cfg.frame_builder_factory(env),
            table_cfg.task_row_mode,
            table_cfg.reset_sources,
            table_cfg.expert_sample_grid,
            seed=seed,
            device=env.device,
        )
    finally:
        source.close()
