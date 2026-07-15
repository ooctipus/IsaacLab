# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime motion table storage, interpolation, and expert sampling."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.utils.math import quat_slerp

from ...data.clip_index import MotionClipIndex
from ...data.frames import Interpolation, MotionFrames
from ...identity import canonical_sha256, validate_nonempty, validate_sha256

if TYPE_CHECKING:
    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from ....mdp.commands.state_command.task_table_view import TaskTableView


_TRAJECTORY_METRIC_NAMES = (
    "source_required_position_max_m",
    "source_required_distal_position_max_m",
    "source_required_distal_direction_max_rad",
    "source_root_rotation_max_rad",
    "source_all_position_max_m",
    "source_all_distal_position_max_m",
    "source_all_landmark_direction_max_rad",
    "source_all_distal_direction_max_rad",
    "source_nonroot_rotation_max_rad",
    "contact_gap_max_m",
    "contact_tilt_max_rad",
    "contact_slip_speed_max_mps",
    "contact_cumulative_drift_max_m",
    "contact_applicable",
    "contact_stable_frame_channel_count",
    "source_contact_confidence_mean",
)
_TRAJECTORY_INSPECTION_CAPTURE_STAGE_NAMES = (
    "frame_seed",
    "source_attempt",
    "physical_attempt",
    "post_physical",
    "contact_attempt",
)
_TRAJECTORY_INSPECTION_STAGE_NAMES = (*_TRAJECTORY_INSPECTION_CAPTURE_STAGE_NAMES, "final")
_TRAJECTORY_INSPECTION_QUALITY_PREFIX = "inspection_stage/"
_TARGET_COORDINATE_QUALITY_NAMES = (
    "coordinate_finite",
    "root_quaternion_norm_error",
    "joint_position_limit_violation_max_rad",
    "joint_position_limits_satisfied",
    "canonical_joint_velocity_limit_ratio",
    "canonical_joint_velocity_limits_satisfied",
    "fk_finite",
    "ground_penetration_max_m",
)
_DYNAMICS_QUALITY_NAMES = (
    "dynamics_balance_force_max_n",
    "dynamics_balance_torque_max_nm",
    "dynamics_effort_margin_ratio_min",
    "dynamics_normal_force_min_n",
    "dynamics_friction_margin_min_n",
    "dynamics_contact_transition_count",
)
_TRAJECTORY_QUALITY_NAMES = (
    "trajectory_route",
    "accepted",
    *_TRAJECTORY_METRIC_NAMES,
    "constraint_geometry_feasible",
    "inner_solve_converged",
    "nonlinear_refinement_required",
    "nonlinear_phases_converged",
)
_QUALITY_TRAJECTORY_ROUTE = _TRAJECTORY_QUALITY_NAMES.index("trajectory_route")
_QUALITY_ACCEPTED = _TRAJECTORY_QUALITY_NAMES.index("accepted")
_QUALITY_CONSTRAINT_GEOMETRY_FEASIBLE = _TRAJECTORY_QUALITY_NAMES.index("constraint_geometry_feasible")
_QUALITY_INNER_SOLVE_CONVERGED = _TRAJECTORY_QUALITY_NAMES.index("inner_solve_converged")
_QUALITY_NONLINEAR_REFINEMENT_REQUIRED = _TRAJECTORY_QUALITY_NAMES.index("nonlinear_refinement_required")
_QUALITY_NONLINEAR_PHASES_CONVERGED = _TRAJECTORY_QUALITY_NAMES.index("nonlinear_phases_converged")
_TRAJECTORY_METRIC_START = _QUALITY_ACCEPTED + 1
_TRAJECTORY_METRIC_STOP = _QUALITY_CONSTRAINT_GEOMETRY_FEASIBLE
_TARGET_COORDINATE_QUALITY_START = len(_TRAJECTORY_QUALITY_NAMES)
_TARGET_COORDINATE_QUALITY_STOP = _TARGET_COORDINATE_QUALITY_START + len(_TARGET_COORDINATE_QUALITY_NAMES)
_DYNAMICS_QUALITY_START = _TARGET_COORDINATE_QUALITY_STOP
_DYNAMICS_QUALITY_STOP = _DYNAMICS_QUALITY_START + len(_DYNAMICS_QUALITY_NAMES)
_QUALITY_NAMES = (
    *_TRAJECTORY_QUALITY_NAMES,
    *_TARGET_COORDINATE_QUALITY_NAMES,
    *_DYNAMICS_QUALITY_NAMES,
    "base_priority",
)


def _table_identity(
    clip_index: MotionClipIndex,
    decoder_version: str,
    frames: MotionFrames,
    joint_names: tuple[str, ...],
    reference_frame_names: tuple[str, ...],
    construction_version: str,
    construction_identity_sha256: str,
    task_row_mode: Literal["source_frames", "clip_time_ranges"],
    family_name: str,
    family_identity_sha256: str,
) -> str:
    """Return deterministic trajectory-data provenance without robot ownership."""
    validate_nonempty("decoder_version", decoder_version)
    validate_nonempty("construction_version", construction_version)
    validate_sha256("construction_identity_sha256", construction_identity_sha256)
    validate_nonempty("family_name", family_name)
    validate_sha256("family_identity_sha256", family_identity_sha256)
    stored_column_shapes = {name: tuple(frames.field(name).shape[1:]) for name in frames.stored_fields}
    return canonical_sha256(
        {
            "source_content_hash": clip_index.source_content_sha256,
            "decoder_version": decoder_version,
            "source_skeleton_hashes": clip_index.skeleton_identity_sha256s,
            "source_clips": [
                {
                    "clip_id": clip.clip_id,
                    "frame_count": clip.frame_count,
                    "source_fps": clip.source_fps,
                    "content_sha256": clip.content_sha256,
                    "skeleton_id": clip.skeleton_id,
                    "source_clip_id": clip.source_clip_id,
                    "source_frame_start": clip.source_frame_start,
                    "source_frame_stop": clip.source_frame_stop,
                }
                for clip in clip_index.clips
            ],
            "construction_version": construction_version,
            "joint_names": joint_names,
            "reference_frame_names": reference_frame_names,
            "construction_identity_sha256": construction_identity_sha256,
            "family_name": family_name,
            "family_identity_sha256": family_identity_sha256,
            "task_row_mode": task_row_mode,
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
    frame_counts = torch.tensor([clip.frame_count for clip in clip_index.clips], dtype=torch.int64, device=device)
    source_fps = torch.tensor([clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device)
    clip_indices = torch.arange(len(clip_index.clips), dtype=torch.int64, device=device)
    if mode == "clip_time_ranges":
        end = (frame_counts - 1) / source_fps
        return clip_indices, torch.stack((torch.zeros_like(end), end), dim=-1)

    clips = torch.repeat_interleave(clip_indices, frame_counts)
    output_offsets = torch.cumsum(frame_counts, dim=0) - frame_counts
    repeated_output_offsets = torch.repeat_interleave(output_offsets, frame_counts)
    local_frames = torch.arange(clips.shape[0], device=device) - repeated_output_offsets
    times = local_frames / source_fps[clips]
    return clips, torch.stack((times, times), dim=-1)


class MotionTaskTable:
    """Exact-capacity trajectory tensors plus selectable motion descriptors.

    Robot identity, kinematic structure, control, and defaults remain owned by
    the selected preset and scene articulation. Builders resolve source data to
    live simulator order once; this table then owns only concrete task tensors,
    clip boundaries, and descriptor rows.
    """

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
            value1 = values[self._global_frame1]
            if interpolation == "slerp":
                return quat_slerp(value0, value1, self._alpha)
            fraction = self._alpha
            while fraction.ndim < value0.ndim:
                fraction = fraction.unsqueeze(-1)
            return torch.lerp(value0, value1, fraction)

    class SampledSequence:
        """Clip-safe trajectory view on one declared sample clock."""

        __slots__ = ("_field", "clip_ids", "clip_offsets", "data_hash", "dataset_id", "device", "source")

        def __init__(
            self,
            source: MotionTaskTable,
            clip_offsets: tuple[int, ...],
            sampling_mode: Literal["source_rows", "uniform_before_source_end"],
            sampling_step_seconds: float | None,
            field: Callable[[str], torch.Tensor],
        ) -> None:
            self.source = source
            self.device = source.device
            self.clip_ids = source.clip_ids
            self.clip_offsets = clip_offsets
            self.dataset_id = f"{source.clip_index.source_content_sha256}:{source.construction_version}"
            self.data_hash = canonical_sha256(
                {
                    "source": source.cache_identity,
                    "sampling_mode": sampling_mode,
                    "sampling_step_seconds": sampling_step_seconds,
                }
            )
            self._field = field

        def field(self, name: str) -> torch.Tensor:
            """Return one sampled trajectory field."""
            return self._field(name)

    __slots__ = (
        "_cache_identity",
        "_clip_index",
        "_clip_offsets",
        "_clip_start_rows",
        "_construction_identity_sha256",
        "_construction_version",
        "_frame_counts",
        "_family_name",
        "_family_identity_sha256",
        "_frames",
        "_joint_names",
        "_reference_frame_names",
        "_sealed",
        "_task_row_mode",
        "_source_fps",
        "_view",
        "clip_indices",
        "reset_time_ranges_seconds",
    )

    def __init__(
        self,
        clip_index: MotionClipIndex,
        frames: MotionFrames,
        joint_names: tuple[str, ...],
        reference_frame_names: tuple[str, ...],
        construction_version: str,
        construction_identity_sha256: str,
        task_row_mode: Literal["source_frames", "clip_time_ranges"],
        decoder_version: str,
        family_name: str,
        family_identity_sha256: str,
        view: TaskTableView,
    ) -> None:
        validate_nonempty("decoder_version", decoder_version)
        if frames.frame_count != clip_index.total_frames:
            raise ValueError("Trajectory capacity must equal the declared source frame count exactly.")
        validate_nonempty("construction_version", construction_version)
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

        validate_sha256("construction_identity_sha256", construction_identity_sha256)
        validate_nonempty("family_name", family_name)
        validate_sha256("family_identity_sha256", family_identity_sha256)
        if (
            view.state_bank.row_count != frames.frame_count
            or view.state_bank.layout.names != ("robot",)
            or view.state_bank.layout.joint_names != (joint_names,)
            or view.sequences.sequence_count != len(clip_index.clips)
            or view.sequences.frame_count != clip_index.total_frames
        ):
            raise ValueError("Motion task view must address the exact stored robot frames and clip offsets.")

        device = frames.device
        clip_offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=device)
        frame_counts = torch.tensor([clip.frame_count for clip in clip_index.clips], dtype=torch.int64, device=device)
        source_fps = torch.tensor([clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device)
        clip_indices, reset_time_ranges_seconds = _task_rows(clip_index, device, task_row_mode)
        num_tasks = clip_indices.shape[0]
        if task_row_mode == "source_frames":
            clip_start_rows = torch.tensor(clip_index.offsets[:-1], dtype=torch.int64, device=device)
        else:
            clip_start_rows = torch.arange(len(clip_index.clips), dtype=torch.int64, device=device)
        if (
            clip_indices.ndim != 1
            or clip_indices.dtype is not torch.int64
            or reset_time_ranges_seconds.shape != (num_tasks, 2)
            or not reset_time_ranges_seconds.is_floating_point()
            or num_tasks == 0
        ):
            raise ValueError("Motion tasks require clip [N] and reset-time-range [N, 2] tensors.")
        if clip_indices.device != device or reset_time_ranges_seconds.device != device:
            raise ValueError("Motion task and trajectory tensors must share one device.")

        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(clip_index.clips))),
            "Motion task clip indices are outside the stored clips.",
        )
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
        expected = torch.arange(len(clip_index.clips), dtype=torch.int64, device=device)
        if not torch.equal(torch.unique(clip_indices, sorted=True), expected):
            raise ValueError("Motion task rows must cover every clip in stable source order.")

        object.__setattr__(self, "_clip_index", clip_index)
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_construction_version", construction_version)
        object.__setattr__(self, "_construction_identity_sha256", construction_identity_sha256)
        object.__setattr__(self, "_joint_names", joint_names)
        object.__setattr__(self, "_reference_frame_names", reference_frame_names)
        object.__setattr__(self, "_clip_offsets", clip_offsets)
        object.__setattr__(self, "_clip_start_rows", clip_start_rows)
        object.__setattr__(self, "_frame_counts", frame_counts)
        object.__setattr__(self, "_source_fps", source_fps)
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "clip_indices", clip_indices)
        object.__setattr__(self, "reset_time_ranges_seconds", reset_time_ranges_seconds)
        object.__setattr__(
            self,
            "_cache_identity",
            _table_identity(
                clip_index,
                decoder_version,
                frames,
                joint_names,
                reference_frame_names,
                construction_version,
                construction_identity_sha256,
                task_row_mode,
                family_name,
                family_identity_sha256,
            ),
        )
        object.__setattr__(self, "_task_row_mode", task_row_mode)
        object.__setattr__(self, "_family_name", family_name)
        object.__setattr__(self, "_family_identity_sha256", family_identity_sha256)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MotionTaskTable metadata is immutable.")
        object.__setattr__(self, name, value)

    @property
    def frames(self) -> MotionFrames:
        """Concrete trajectory tensor owner."""
        return self._frames

    @property
    def view(self) -> TaskTableView:
        """Canonical physical states, exact mechanics, and clip sequence view."""
        return self._view

    @property
    def base_priorities(self) -> torch.Tensor:
        """Immutable source-balanced sampling mass per retained sequence."""
        quality = self._view.quality
        if quality is None or quality.scope != "sequence" or "base_priority" not in quality.names:
            raise ValueError("Motion table quality must expose one sequence-scoped base_priority column.")
        return quality.values[:, quality.names.index("base_priority")]

    @property
    def family_name(self) -> str:
        """Exact or semantic coordinate family selected for this table."""
        return self._family_name

    @property
    def family_identity_sha256(self) -> str:
        """Complete selected generation, solve, acceptance, and selection policy identity."""
        return self._family_identity_sha256

    @property
    def clip_index(self) -> MotionClipIndex:
        """Ordered source clip metadata."""
        return self._clip_index

    @property
    def construction_version(self) -> str:
        """Readable version of source-to-trajectory conversion."""
        return self._construction_version

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of every joint trajectory column."""
        return self._joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Ordered semantic labels of the optional reference-frame columns."""
        return self._reference_frame_names

    @property
    def construction_identity_sha256(self) -> str:
        """Construction identity closing reference kinematics and ordered mappings."""
        return self._construction_identity_sha256

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
    def clip_start_rows(self) -> torch.Tensor:
        """First selectable task row per source clip."""
        return self._clip_start_rows

    @property
    def frame_counts(self) -> torch.Tensor:
        """Frames per clip."""
        return self._frame_counts

    @property
    def source_fps(self) -> torch.Tensor:
        """Source sample rate [Hz] per clip."""
        return self._source_fps

    @property
    def clip_ids(self) -> tuple[str, ...]:
        """Clip identifiers covered by selectable descriptor rows."""
        return self._clip_index.clip_ids

    @property
    def device(self) -> torch.device:
        """Shared trajectory and descriptor device."""
        return self._frames.device

    @property
    def num_tasks(self) -> int:
        """Number of selectable motion descriptors."""
        return self.clip_indices.shape[0]

    def field(self, name: str) -> torch.Tensor:
        """Return one concrete trajectory tensor without indirection or copying."""
        return self._frames.field(name)

    def interpolation(self, name: str) -> Interpolation:
        """Return the fixed temporal rule for one concrete trajectory field."""
        return self._frames.interpolation(name)

    def _validate_clip_indices(self, clip_indices: torch.Tensor) -> None:
        if clip_indices.ndim != 1 or clip_indices.dtype is not torch.int64 or clip_indices.device != self.device:
            raise ValueError("clip_indices must be a 1D int64 tensor on the table device.")
        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(self._clip_index.clips))),
            "clip_indices are outside the motion table.",
        )

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

    def sample(
        self,
        mode: Literal["source_rows", "uniform_before_source_end"],
        step_seconds: float | None,
    ) -> SampledSequence:
        """Return clips on one source-row or uniform sample clock."""
        if mode == "source_rows":
            if step_seconds is not None:
                raise ValueError("Source-row sampling does not declare step_seconds.")
        elif mode == "uniform_before_source_end":
            if step_seconds is None or not math.isfinite(step_seconds) or step_seconds <= 0.0:
                raise ValueError("Uniform sampling requires finite positive step_seconds.")
        else:
            raise ValueError(f"Unsupported sampling mode: {mode!r}.")

        counts = tuple(
            clip.frame_count
            if mode == "source_rows"
            else math.ceil((clip.frame_count - 1) / clip.source_fps / step_seconds)
            for clip in self.clip_index.clips
        )
        if any(count < 1 for count in counts):
            raise ValueError("Every sampled clip must contain at least one sample before its source endpoint.")
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        clip_offsets = tuple(offsets)

        if mode == "source_rows":
            return self.SampledSequence(self, clip_offsets, mode, step_seconds, self.field)

        counts_tensor = torch.tensor(counts, dtype=torch.int64, device=self.device)
        clip_positions = torch.repeat_interleave(
            torch.arange(len(self.clip_index.clips), dtype=torch.int64, device=self.device),
            counts_tensor,
        )
        flat_indices = torch.arange(clip_offsets[-1], dtype=torch.int64, device=self.device)
        starts = torch.tensor(clip_offsets[:-1], dtype=torch.int64, device=self.device)
        local_samples = flat_indices - starts[clip_positions]
        clip_indices = clip_positions
        sample_times = local_samples * step_seconds
        reference = self.reference_view(clip_indices, sample_times)
        return self.SampledSequence(self, clip_offsets, mode, step_seconds, reference.field)


def build_motion_task_table(command_cfg: StateCommandCfg, scene_cfg: object, device: str) -> MotionTaskTable:
    """Build the exact production motion task table."""
    from .motion_task_table_builder import build_motion_task_table as build

    return build(command_cfg, scene_cfg, device)
