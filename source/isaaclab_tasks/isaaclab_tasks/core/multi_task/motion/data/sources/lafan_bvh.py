# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode raw LAFAN1 BVH clips without a simulator or SciPy dependency."""

from __future__ import annotations

import csv
import hashlib
import math
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ....kinematics import time_unwrap_angles
from ...identity import canonical_sha256, validate_sha256
from ..clip_index import MotionClipIndex
from ..skeleton import MotionSkeleton
from ..source import MotionSourceCfg

_POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")
_ROTATION_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
_NATIVE_TO_WORLD_XYZW = np.array((2.0**-0.5, 0.0, 0.0, 2.0**-0.5), dtype=np.float64)
_NATIVE_TO_WORLD_WXYZ = (2.0**-0.5, 2.0**-0.5, 0.0, 0.0)
_LANDMARK_BODIES = (
    ("pelvis", "Hips"),
    ("left_hip", "LeftUpLeg"),
    ("left_knee", "LeftLeg"),
    ("left_ankle", "LeftFoot"),
    ("left_toe", "LeftToe"),
    ("right_hip", "RightUpLeg"),
    ("right_knee", "RightLeg"),
    ("right_ankle", "RightFoot"),
    ("right_toe", "RightToe"),
    ("torso", "Spine"),
    ("spine", "Spine1"),
    ("chest", "Spine2"),
    ("neck", "Neck"),
    ("left_thorax", "LeftShoulder"),
    ("right_thorax", "RightShoulder"),
    ("head", "Head"),
    ("left_shoulder", "LeftArm"),
    ("left_elbow", "LeftForeArm"),
    ("left_wrist", "LeftHand"),
    ("right_shoulder", "RightArm"),
    ("right_elbow", "RightForeArm"),
    ("right_wrist", "RightHand"),
)


@dataclass(frozen=True, slots=True)
class LafanBvhHierarchy:
    """One exact LAFAN BVH hierarchy with source-local offsets.

    The hierarchy retains every BVH offset in its declared source-local frame.
    :func:`lafan_bvh_skeleton` converts only the root zero-channel basis into
    IsaacLab's world frame; descendants remain parent-local source mechanics.

    Attributes:
        body_names: BVH joint names in topological order.
        parent_indices: Parent index for each body, with ``-1`` for the root.
        rest_translation_m: Parent-local rest translations [m]. The root row is
            zero because root position channels replace the BVH root offset.
        channels: Ordered native BVH channels for every body.
        end_sites: Declared end-site offsets [m] and their parent body indices.
        source_sha256: SHA-256 of the complete source BVH bytes.
    """

    @dataclass(frozen=True, slots=True)
    class EndSite:
        """One unnamed BVH end site relative to its parent body."""

        parent_index: int
        rest_translation_m: tuple[float, float, float]

    body_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    rest_translation_m: tuple[tuple[float, float, float], ...]
    channels: tuple[tuple[str, ...], ...]
    end_sites: tuple[EndSite, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete trees and unsupported LAFAN channel mechanics."""
        body_count = len(self.body_names)
        if body_count < 1 or len(set(self.body_names)) != body_count:
            raise ValueError("BVH body names must be nonempty and unique.")
        if len(self.parent_indices) != body_count or self.parent_indices[0] != -1:
            raise ValueError("BVH parent indices must contain one root.")
        if any(parent < 0 or parent >= body for body, parent in enumerate(self.parent_indices[1:], start=1)):
            raise ValueError("BVH bodies must be in topological order.")
        if len(self.rest_translation_m) != body_count or any(len(value) != 3 for value in self.rest_translation_m):
            raise ValueError("BVH rest translations must contain one xyz row per body.")
        if any(not math.isfinite(component) for value in self.rest_translation_m for component in value):
            raise ValueError("BVH rest translations must be finite.")
        if len(self.channels) != body_count:
            raise ValueError("BVH channels must contain one ordered row per body.")

        root_channels = self.channels[0]
        if (
            len(root_channels) != 6
            or {channel for channel in root_channels if channel.endswith("position")} != set(_POSITION_CHANNELS)
            or {channel for channel in root_channels if channel.endswith("rotation")} != set(_ROTATION_CHANNELS)
        ):
            raise ValueError("A LAFAN BVH root must declare xyz position and rotation channels exactly once.")
        for channels in self.channels[1:]:
            if len(channels) != 3 or set(channels) != set(_ROTATION_CHANNELS):
                raise ValueError("Every non-root LAFAN BVH body must declare exactly three rotation channels.")
        for end_site in self.end_sites:
            if end_site.parent_index < 0 or end_site.parent_index >= body_count:
                raise ValueError("BVH end sites must reference an existing parent body.")
            if len(end_site.rest_translation_m) != 3 or any(
                not math.isfinite(component) for component in end_site.rest_translation_m
            ):
                raise ValueError("BVH end-site offsets must be finite xyz rows.")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("BVH source_sha256 must be a lowercase SHA-256 digest.")

    @property
    def rotation_orders(self) -> tuple[str, ...]:
        """Per-body intrinsic rotation-channel order as lowercase axis names."""
        return tuple(
            "".join(channel[0].lower() for channel in channels if channel.endswith("rotation"))
            for channels in self.channels
        )


@dataclass(frozen=True, slots=True)
class LafanBvhClip:
    """One raw LAFAN clip decoded into robot-independent local kinematics.

    Attributes:
        hierarchy: Exact per-file BVH hierarchy and channel declaration.
        frame_time_seconds: Native time between samples [s].
        root_position_m: Root world positions [m], shape ``[frame_count, 3]``.
        local_rotation_xyzw: Unit xyzw rotations, shape
            ``[frame_count, body_count, 4]``. Row zero is world-root rotation;
            remaining rows are parent-local rotations.
        joint_coordinate_rad: Native non-root Euler channels [rad], shape
            ``[frame_count, 3 * (body_count - 1)]``.
    """

    hierarchy: LafanBvhHierarchy
    frame_time_seconds: float
    root_position_m: np.ndarray
    local_rotation_xyzw: np.ndarray
    joint_coordinate_rad: np.ndarray

    def __post_init__(self) -> None:
        """Validate decoded shapes and make the trajectory arrays immutable."""
        frame_count = self.root_position_m.shape[0] if self.root_position_m.ndim == 2 else 0
        if not math.isfinite(self.frame_time_seconds) or self.frame_time_seconds <= 0.0:
            raise ValueError("BVH frame time must be finite and positive [s].")
        if frame_count < 1 or self.root_position_m.shape != (frame_count, 3):
            raise ValueError("BVH root positions must have shape [frame_count, 3].")
        if self.local_rotation_xyzw.shape != (frame_count, len(self.hierarchy.body_names), 4):
            raise ValueError("BVH local rotations must have shape [frame_count, body_count, 4].")
        expected_coordinate_shape = (frame_count, 3 * (len(self.hierarchy.body_names) - 1))
        if self.joint_coordinate_rad.shape != expected_coordinate_shape:
            raise ValueError(f"BVH joint coordinates must have shape {expected_coordinate_shape}.")
        arrays = (self.root_position_m, self.local_rotation_xyzw, self.joint_coordinate_rad)
        if any(value.dtype != np.float64 for value in arrays):
            raise ValueError("Raw BVH decoding must preserve float64 source precision.")
        if not all(value.flags.c_contiguous for value in arrays):
            raise ValueError("Decoded BVH arrays must be C-contiguous.")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Decoded BVH arrays must contain only finite values.")
        norm = np.linalg.norm(self.local_rotation_xyzw, axis=-1)
        if not np.allclose(norm, 1.0, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("Decoded BVH rotations must be unit quaternions.")
        self.root_position_m.setflags(write=False)
        self.local_rotation_xyzw.setflags(write=False)
        self.joint_coordinate_rad.setflags(write=False)

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""
        return int(self.root_position_m.shape[0])

    @property
    def source_fps(self) -> float:
        """Native sample rate [Hz]."""
        return 1.0 / self.frame_time_seconds

    def slice(self, start: int, stop: int) -> LafanBvhClip:
        """Return one immutable frame view without copying decoded source arrays."""
        if type(start) is not int or type(stop) is not int or start < 0 or stop <= start or stop > self.frame_count:
            raise ValueError("LAFAN clip slices must be nonempty frame ranges inside the decoded source clip.")
        return LafanBvhClip(
            hierarchy=self.hierarchy,
            frame_time_seconds=self.frame_time_seconds,
            root_position_m=self.root_position_m[start:stop],
            local_rotation_xyzw=self.local_rotation_xyzw[start:stop],
            joint_coordinate_rad=self.joint_coordinate_rad[start:stop],
        )

    def free_root_coordinates(
        self, source_skeleton: MotionSkeleton, *, device: str | torch.device
    ) -> tuple[torch.Tensor, None]:
        """Return the declared BVH free-root and ordered Euler coordinates."""
        _validate_lafan_bvh_skeleton(self.hierarchy, source_skeleton)
        root_position = torch.tensor(self.root_position_m, dtype=torch.float32, device=device)
        local_rotation = torch.tensor(self.local_rotation_xyzw, dtype=torch.float32, device=device)
        coordinate = torch.tensor(self.joint_coordinate_rad, dtype=torch.float32, device=device)
        coordinate = time_unwrap_angles(coordinate)
        return torch.cat((root_position, local_rotation[:, 0], coordinate), dim=-1), None

    def local_pose(
        self, source_skeleton: MotionSkeleton, *, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world-root positions and root/local rotations."""
        _validate_lafan_bvh_skeleton(self.hierarchy, source_skeleton)
        return (
            torch.tensor(self.root_position_m, dtype=torch.float32, device=device),
            torch.tensor(self.local_rotation_xyzw, dtype=torch.float32, device=device),
        )


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply broadcast-compatible xyzw quaternions."""
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _euler_channels_to_quaternion_xyzw(angles_degrees: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    """Compose intrinsic BVH rotation channels in their declared order."""
    rotation_channels = tuple(channel for channel in channels if channel.endswith("rotation"))
    rotation = np.zeros((*angles_degrees.shape[:-1], 4), dtype=np.float64)
    rotation[..., 3] = 1.0
    for channel_index, channel in enumerate(rotation_channels):
        half_angle = np.deg2rad(angles_degrees[..., channel_index]) * 0.5
        axis_rotation = np.zeros_like(rotation)
        axis_rotation[..., _AXIS_INDEX[channel[0]]] = np.sin(half_angle)
        axis_rotation[..., 3] = np.cos(half_angle)
        rotation = _quaternion_multiply_xyzw(rotation, axis_rotation)
    return rotation


def _make_rotations_continuous(rotation: np.ndarray) -> None:
    """Choose quaternion signs continuously over time in-place."""
    if rotation.shape[0] < 2:
        return
    edge_sign = np.where(np.sum(rotation[:-1] * rotation[1:], axis=-1) < 0.0, -1.0, 1.0)
    sign = np.concatenate((np.ones((1, rotation.shape[1])), np.cumprod(edge_sign, axis=0)), axis=0)
    rotation *= sign[..., None]


def _canonical_world_vector(native: np.ndarray) -> np.ndarray:
    """Convert native LAFAN xyz centimeters to IsaacLab xyz meters."""
    return np.stack((native[..., 0], -native[..., 2], native[..., 1]), axis=-1) * 0.01


def load_lafan_bvh(path: str | Path) -> LafanBvhClip:
    """Load one raw LAFAN1 BVH file.

    Args:
        path: Raw LAFAN1 BVH file.

    Returns:
        The decoded robot-independent clip.
    """
    return decode_lafan_bvh(Path(path).read_bytes())


def _parse_lafan_bvh_hierarchy(lines: list[str], source_sha256: str) -> LafanBvhHierarchy:  # noqa: C901
    """Parse one complete hierarchy header without reading motion rows."""
    body_names: list[str] = []
    parent_indices: list[int] = []
    offsets_cm: list[tuple[float, float, float] | None] = []
    channels: list[tuple[str, ...] | None] = []
    end_sites_cm: list[tuple[int, tuple[float, float, float]]] = []
    stack: list[int | None] = []
    pending: int | None = None
    pending_end_site = False
    for raw_line in lines:
        fields = raw_line.strip().split()
        if not fields or fields[0] == "HIERARCHY":
            continue
        if fields[0] in ("ROOT", "JOINT"):
            if len(fields) != 2:
                raise ValueError("BVH ROOT and JOINT declarations require one name.")
            parent = next((body for body in reversed(stack) if body is not None), -1)
            if fields[0] == "ROOT" and body_names:
                raise ValueError("BVH must contain exactly one root body.")
            if fields[0] == "JOINT" and parent < 0:
                raise ValueError("BVH joints must be nested below the root.")
            pending = len(body_names)
            body_names.append(fields[1])
            parent_indices.append(parent)
            offsets_cm.append(None)
            channels.append(None)
            continue
        if fields[:2] == ["End", "Site"]:
            if not stack or stack[-1] is None:
                raise ValueError("BVH end sites must be nested below a body.")
            pending_end_site = True
            continue
        if fields[0] == "{":
            if pending is not None:
                stack.append(pending)
                pending = None
            elif pending_end_site:
                stack.append(None)
                pending_end_site = False
            else:
                raise ValueError("BVH contains an opening brace without a body declaration.")
            continue
        if fields[0] == "}":
            if not stack:
                raise ValueError("BVH contains an unmatched closing brace.")
            stack.pop()
            continue
        if fields[0] == "OFFSET":
            if len(fields) != 4 or not stack:
                raise ValueError("BVH OFFSET declarations require three values inside a body.")
            try:
                offset = tuple(float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError("BVH OFFSET values must be finite numbers.") from error
            if any(not math.isfinite(value) for value in offset):
                raise ValueError("BVH OFFSET values must be finite numbers.")
            if stack[-1] is None:
                parent = next((body for body in reversed(stack[:-1]) if body is not None), -1)
                end_sites_cm.append((parent, offset))
            else:
                offsets_cm[stack[-1]] = offset
            continue
        if fields[0] == "CHANNELS":
            if not stack or stack[-1] is None or len(fields) < 3:
                raise ValueError("BVH CHANNELS declarations must belong to a body.")
            try:
                channel_count = int(fields[1])
            except ValueError as error:
                raise ValueError("BVH CHANNELS count must be an integer.") from error
            declared = tuple(fields[2:])
            if channel_count != len(declared):
                raise ValueError("BVH CHANNELS count differs from its declared channel names.")
            if any(channel not in (*_POSITION_CHANNELS, *_ROTATION_CHANNELS) for channel in declared):
                raise ValueError("BVH declares an unsupported motion channel.")
            channels[stack[-1]] = declared
            continue
        raise ValueError(f"Unsupported BVH hierarchy declaration: {raw_line.strip()!r}.")
    if stack or pending is not None or pending_end_site:
        raise ValueError("BVH hierarchy braces are incomplete.")
    if not body_names or any(offset is None for offset in offsets_cm) or any(value is None for value in channels):
        raise ValueError("Every BVH body must declare OFFSET and CHANNELS rows.")
    rest_translation_m = []
    for body_index, offset in enumerate(offsets_cm):
        assert offset is not None
        converted = np.zeros(3, dtype=np.float64) if body_index == 0 else np.asarray(offset) * 0.01
        rest_translation_m.append(tuple(float(value) for value in converted))
    return LafanBvhHierarchy(
        body_names=tuple(body_names),
        parent_indices=tuple(parent_indices),
        rest_translation_m=tuple(rest_translation_m),
        channels=tuple(value for value in channels if value is not None),
        end_sites=tuple(
            LafanBvhHierarchy.EndSite(parent, tuple(float(value) * 0.01 for value in offset))
            for parent, offset in end_sites_cm
        ),
        source_sha256=source_sha256,
    )


def _read_lafan_bvh_hierarchy(stream, source_sha256: str) -> LafanBvhHierarchy:
    """Read only the bounded hierarchy prefix of one BVH member."""
    lines = []
    while raw_line := stream.readline():
        line = raw_line.decode("utf-8-sig")
        if line.strip() == "MOTION":
            return _parse_lafan_bvh_hierarchy(lines, source_sha256)
        lines.append(line)
    raise ValueError("BVH file does not contain a MOTION section.")


def decode_lafan_bvh(source_bytes: bytes) -> LafanBvhClip:
    """Decode one immutable BVH byte sequence without extracting it to disk.

    Args:
        source_bytes: Complete bytes of one LAFAN1 BVH member.

    Returns:
        Robot-independent local kinematics decoded at source precision.
    """
    lines = source_bytes.decode("utf-8-sig").splitlines()
    try:
        motion_line = next(index for index, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as error:
        raise ValueError("BVH file does not contain a MOTION section.") from error
    hierarchy = _parse_lafan_bvh_hierarchy(lines[:motion_line], hashlib.sha256(source_bytes).hexdigest())
    motion_lines = [line.strip() for line in lines[motion_line + 1 :] if line.strip()]
    metadata_valid = (
        len(motion_lines) >= 3 and motion_lines[0].startswith("Frames:") and motion_lines[1].startswith("Frame Time:")
    )
    if not metadata_valid:
        raise ValueError("BVH MOTION metadata must declare Frames then Frame Time.")
    try:
        frame_count = int(motion_lines[0].split(":", 1)[1])
        frame_time_seconds = float(motion_lines[1].split(":", 1)[1])
    except ValueError as error:
        raise ValueError("BVH frame count and frame time must be numeric.") from error
    if frame_count < 1 or len(motion_lines[2:]) != frame_count:
        raise ValueError("BVH frame count differs from the number of motion rows.")
    channel_count = sum(len(value) for value in hierarchy.channels)
    if any(len(row.split()) != channel_count for row in motion_lines[2:]):
        raise ValueError("BVH motion rows have the wrong width or contain nonnumeric values.")
    motion = np.fromstring("\n".join(motion_lines[2:]), dtype=np.float64, sep=" ")
    if motion.size != frame_count * channel_count:
        raise ValueError("BVH motion rows have the wrong width or contain nonnumeric values.")
    motion = motion.reshape(frame_count, channel_count)
    if not np.isfinite(motion).all():
        raise ValueError("BVH motion rows have the wrong width or contain nonfinite values.")

    root_position_native = np.empty((frame_count, 3), dtype=np.float64)
    local_rotation = np.empty((frame_count, len(hierarchy.body_names), 4), dtype=np.float64)
    joint_coordinates = []
    cursor = 0
    for body_index, body_channels in enumerate(hierarchy.channels):
        values = motion[:, cursor : cursor + len(body_channels)]
        cursor += len(body_channels)
        if body_index == 0:
            for channel_index, channel in enumerate(body_channels):
                if channel.endswith("position"):
                    root_position_native[:, _AXIS_INDEX[channel[0]]] = values[:, channel_index]
        rotation_indices = [index for index, channel in enumerate(body_channels) if channel.endswith("rotation")]
        rotation_degrees = values[:, rotation_indices]
        local_rotation[:, body_index] = _euler_channels_to_quaternion_xyzw(rotation_degrees, body_channels)
        if body_index > 0:
            joint_coordinates.append(np.deg2rad(rotation_degrees))
    local_rotation[:, 0] = _quaternion_multiply_xyzw(_NATIVE_TO_WORLD_XYZW, local_rotation[:, 0])
    _make_rotations_continuous(local_rotation)
    return LafanBvhClip(
        hierarchy=hierarchy,
        frame_time_seconds=frame_time_seconds,
        root_position_m=np.ascontiguousarray(_canonical_world_vector(root_position_native)),
        local_rotation_xyzw=np.ascontiguousarray(local_rotation),
        joint_coordinate_rad=np.ascontiguousarray(np.concatenate(joint_coordinates, axis=-1)),
    )


def lafan_bvh_skeleton(hierarchy: LafanBvhHierarchy) -> MotionSkeleton:
    """Construct source mechanics with a world-converted root and source-local offsets."""
    mechanics = {
        "body_names": hierarchy.body_names,
        "parent_indices": hierarchy.parent_indices,
        "rest_translation_m": hierarchy.rest_translation_m,
        "channels": hierarchy.channels,
        "end_sites": tuple((site.parent_index, site.rest_translation_m) for site in hierarchy.end_sites),
        "zero_channel_world_basis": "lafan_y_up_to_z_up_x_forward",
    }
    mechanics_sha256 = canonical_sha256(mechanics)
    joint_names = []
    joint_children = []
    joint_axes = []
    body_rows = zip(hierarchy.body_names[1:], hierarchy.channels[1:], strict=True)
    for body_index, (body_name, channels) in enumerate(body_rows, start=1):
        for channel in channels:
            joint_names.append(f"{body_name}_{channel[0].lower()}")
            joint_children.append(body_index)
            axis = [0.0, 0.0, 0.0]
            axis[_AXIS_INDEX[channel[0]]] = 1.0
            joint_axes.append(tuple(axis))
    body_names = set(hierarchy.body_names)
    landmarks = tuple(
        MotionSkeleton.Landmark(role, body_name, body_name)
        for role, body_name in _LANDMARK_BODIES
        if body_name in body_names
    )
    return MotionSkeleton(
        identifier=f"lafan_bvh_{mechanics_sha256[:16]}",
        content_sha256=mechanics_sha256,
        body_names=hierarchy.body_names,
        parent_indices=hierarchy.parent_indices,
        rest_translation_m=hierarchy.rest_translation_m,
        rest_rotation_wxyz=(_NATIVE_TO_WORLD_WXYZ, *((1.0, 0.0, 0.0, 0.0),) * (len(hierarchy.body_names) - 1)),
        joint_names=tuple(joint_names),
        joint_child_body_indices=tuple(joint_children),
        joint_axes=tuple(joint_axes),
        landmarks=landmarks,
        root_translation_frame="world",
        root_rotation_convention="xyzw",
        landmark_rotation_policy="anatomical_root",
    )


def _validate_lafan_bvh_skeleton(hierarchy: LafanBvhHierarchy, skeleton: MotionSkeleton) -> None:
    """Require the exact source mechanics derived from this clip's hierarchy."""
    expected = lafan_bvh_skeleton(hierarchy)
    if skeleton.identity_sha256 != expected.identity_sha256:
        raise ValueError("Raw LAFAN clip and declared source skeleton differ.")


@dataclass(frozen=True, slots=True)
class LafanClipRow:
    """One declared output clip backed by a frame range in an official BVH member."""

    clip_id: str
    member: str
    source_sha256: str
    source_frame_count: int
    source_fps: float
    frame_start: int
    frame_stop: int

    def __post_init__(self) -> None:
        """Require one concrete source-root member and nonempty frame range."""
        member = Path(self.member)
        if not self.clip_id or member.name != self.member or member.suffix.lower() != ".bvh":
            raise ValueError("LAFAN clip rows require a nonempty id and one root-level BVH member.")
        validate_sha256("LAFAN clip row source_sha256", self.source_sha256)
        if self.source_frame_count < 1 or not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError("LAFAN clip rows require positive source frame counts and sample rates [Hz].")
        if not 0 <= self.frame_start < self.frame_stop <= self.source_frame_count:
            raise ValueError("LAFAN clip row frame ranges must lie inside their declared source clip.")

    @property
    def frame_count(self) -> int:
        """Number of frames selected from the source clip."""
        return self.frame_stop - self.frame_start

    @property
    def content_sha256(self) -> str:
        """Identity of this exact source-byte frame range."""
        return canonical_sha256(
            {
                "source_sha256": self.source_sha256,
                "frame_start": self.frame_start,
                "frame_stop": self.frame_stop,
            }
        )


_LAFAN_CLIP_ROW_FIELDS = (
    "clip_id",
    "member",
    "source_sha256",
    "source_frame_count",
    "source_fps",
    "frame_start",
    "frame_stop",
)


def read_lafan_clip_rows(path: str | Path) -> tuple[LafanClipRow, ...]:
    """Read deterministic concrete clip rows from one UTF-8 CSV file."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _LAFAN_CLIP_ROW_FIELDS:
            raise ValueError(f"LAFAN clip-row columns must equal {_LAFAN_CLIP_ROW_FIELDS}.")
        try:
            rows = tuple(
                LafanClipRow(
                    clip_id=row["clip_id"],
                    member=row["member"],
                    source_sha256=row["source_sha256"],
                    source_frame_count=int(row["source_frame_count"]),
                    source_fps=float(row["source_fps"]),
                    frame_start=int(row["frame_start"]),
                    frame_stop=int(row["frame_stop"]),
                )
                for row in reader
            )
        except (TypeError, ValueError) as error:
            raise ValueError("LAFAN clip rows contain invalid numeric fields.") from error
    if not rows or len({row.clip_id for row in rows}) != len(rows):
        raise ValueError("LAFAN clip rows must be nonempty and have unique clip ids.")
    order = tuple((row.member, row.frame_start, row.frame_stop, row.clip_id) for row in rows)
    if order != tuple(sorted(order)):
        raise ValueError("LAFAN clip rows must be ordered by member and source frame range.")
    member_facts: dict[str, tuple[str, int, float]] = {}
    for row in rows:
        facts = (row.source_sha256, row.source_frame_count, row.source_fps)
        if row.member in member_facts and member_facts[row.member] != facts:
            raise ValueError("LAFAN rows for one BVH member must share source identity, count, and sample rate.")
        member_facts[row.member] = facts
    return rows


def write_lafan_clip_rows(path: str | Path, rows: tuple[LafanClipRow, ...]) -> None:
    """Write deterministic concrete clip rows with Unix line endings."""
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_LAFAN_CLIP_ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "member": row.member,
                    "source_sha256": row.source_sha256,
                    "source_frame_count": row.source_frame_count,
                    "source_fps": format(row.source_fps, ".17g"),
                    "frame_start": row.frame_start,
                    "frame_stop": row.frame_stop,
                }
            )


def lafan_clip_rows_content_sha256(zip_sha256: str, rows: tuple[LafanClipRow, ...]) -> str:
    """Return the ordered scientific identity of a zip dependency and concrete clip rows."""
    validate_sha256("LAFAN zip sha256", zip_sha256)
    return canonical_sha256(
        {
            "lafan_zip_sha256": zip_sha256,
            "clip_rows": [
                {
                    "clip_id": row.clip_id,
                    "member": row.member,
                    "source_sha256": row.source_sha256,
                    "source_frame_count": row.source_frame_count,
                    "source_fps": row.source_fps,
                    "frame_start": row.frame_start,
                    "frame_stop": row.frame_stop,
                }
                for row in rows
            ],
        }
    )


class LafanBvhZipClips:
    """Concrete clip rows over one verified official LAFAN zip dependency."""

    __slots__ = ("_decoded_members", "_index", "_path", "_rows", "_skeletons", "_source_sha256", "_used_rows")

    def __init__(
        self,
        path: str | Path,
        clip_rows_path: str | Path,
        *,
        verified_zip_sha256: str,
        verified_rows_sha256: str,
        expected_source_sha256: str,
    ) -> None:
        """Inspect member headers and deduplicate exact source mechanics."""
        self._path = Path(path)
        rows_path = Path(clip_rows_path)
        if not self._path.is_file() or not rows_path.is_file():
            raise FileNotFoundError(self._path if not self._path.is_file() else rows_path)
        for name, digest in (
            ("verified_zip_sha256", verified_zip_sha256),
            ("verified_rows_sha256", verified_rows_sha256),
            ("expected_source_sha256", expected_source_sha256),
        ):
            validate_sha256(name, digest)
        if hashlib.sha256(rows_path.read_bytes()).hexdigest() != verified_rows_sha256:
            raise ValueError("LAFAN clip-row bytes differ from their verified artifact digest.")
        self._rows = read_lafan_clip_rows(rows_path)
        self._source_sha256 = lafan_clip_rows_content_sha256(verified_zip_sha256, self._rows)
        if self._source_sha256 != expected_source_sha256:
            raise ValueError("LAFAN zip dependency and clip rows differ from the declared source identity.")

        member_rows = {}
        for row in self._rows:
            member_rows.setdefault(row.member, row)
        skeletons = []
        skeleton_ids = {}
        skeleton_id_by_identity = {}
        with zipfile.ZipFile(self._path) as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos if not info.is_dir())
            if not names or len(names) != len(infos) or len(set(names)) != len(names):
                raise ValueError("LAFAN zip must contain unique file members and no directories.")
            if any(Path(name).name != name or Path(name).suffix.lower() != ".bvh" for name in names):
                raise ValueError("LAFAN zip members must be root-level BVH files.")
            missing = sorted(set(member_rows).difference(names))
            if missing:
                raise ValueError(f"LAFAN clip rows reference missing zip members: {missing}.")
            for member, row in member_rows.items():
                with archive.open(member) as stream:
                    hierarchy = _read_lafan_bvh_hierarchy(stream, row.source_sha256)
                skeleton = lafan_bvh_skeleton(hierarchy)
                skeleton_id = skeleton_id_by_identity.get(skeleton.identity_sha256)
                if skeleton_id is None:
                    skeleton_id = len(skeletons)
                    skeleton_id_by_identity[skeleton.identity_sha256] = skeleton_id
                    skeletons.append(skeleton)
                skeleton_ids[member] = skeleton_id
        self._skeletons = tuple(skeletons)
        self._index = MotionClipIndex(
            source_content_sha256=self._source_sha256,
            skeleton_identity_sha256s=tuple(skeleton.identity_sha256 for skeleton in self._skeletons),
            clips=tuple(
                MotionClipIndex.Clip(
                    clip_id=row.clip_id,
                    frame_count=row.frame_count,
                    source_fps=row.source_fps,
                    content_sha256=row.content_sha256,
                    skeleton_id=skeleton_ids[row.member],
                    source_clip_id=Path(row.member).stem,
                    source_frame_start=row.frame_start,
                )
                for row in self._rows
            ),
        )
        self._used_rows: set[int] = set()
        self._decoded_members: set[str] = set()

    @staticmethod
    def _validate_member(row: LafanClipRow, source_bytes: bytes, clip: LafanBvhClip) -> None:
        """Verify decoded bytes against every immutable source fact in a clip row."""
        if hashlib.sha256(source_bytes).hexdigest() != row.source_sha256:
            raise ValueError(f"LAFAN member {row.member!r} differs from its clip-row source hash.")
        if clip.frame_count != row.source_frame_count or not math.isclose(
            clip.source_fps, row.source_fps, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"LAFAN member {row.member!r} count or sample rate differs from its clip rows.")

    def inspect(self) -> MotionClipIndex:
        """Return the concrete ordered row index after bounded header inspection."""
        return self._index

    def skeleton(self, skeleton_id: int) -> MotionSkeleton:
        """Return one deduplicated exact-mechanics source skeleton."""
        if type(skeleton_id) is not int or skeleton_id < 0 or skeleton_id >= len(self._skeletons):
            raise ValueError(f"Unknown skeleton id: {skeleton_id!r}.")
        return self._skeletons[skeleton_id]

    def clips(self, clip_indices: tuple[int, ...]) -> Iterator[tuple[int, LafanBvhClip]]:
        """Decode each selected member once in one sorted, single-mechanics group selection."""
        if any(type(index) is not int or index < 0 or index >= len(self._rows) for index in clip_indices):
            raise IndexError("LAFAN clip index is out of range.")
        if not clip_indices or clip_indices != tuple(sorted(set(clip_indices))):
            raise ValueError("Raw LAFAN group rows must be nonempty, sorted, and unique.")
        selected = set(clip_indices)
        if selected.intersection(self._used_rows):
            raise ValueError("Raw LAFAN clip rows may be consumed only once.")
        skeleton_ids = {self._index.clips[index].skeleton_id for index in clip_indices}
        if len(skeleton_ids) != 1:
            raise ValueError("One raw LAFAN stream call must select one exact-mechanics group.")
        selected_members = tuple(dict.fromkeys(self._rows[index].member for index in clip_indices))
        if set(selected_members).intersection(self._decoded_members):
            raise ValueError("Raw LAFAN BVH members may be decoded only once.")

        decoded_members: set[str] = set()
        current_member = None
        current_clip = None
        with zipfile.ZipFile(self._path) as archive:
            for clip_index in clip_indices:
                row = self._rows[clip_index]
                if row.member != current_member:
                    source_bytes = archive.read(row.member)
                    current_clip = decode_lafan_bvh(source_bytes)
                    self._validate_member(row, source_bytes, current_clip)
                    expected_skeleton = self._skeletons[self._index.clips[clip_index].skeleton_id]
                    if lafan_bvh_skeleton(current_clip.hierarchy).identity_sha256 != expected_skeleton.identity_sha256:
                        raise ValueError(f"LAFAN member {row.member!r} hierarchy changed after inspection.")
                    decoded_members.add(row.member)
                    current_member = row.member
                assert current_clip is not None
                yield clip_index, current_clip.slice(row.frame_start, row.frame_stop)
        self._used_rows.update(selected)
        self._decoded_members.update(decoded_members)

    def close(self) -> None:
        """End the source lifetime without retaining decoded motion arrays."""
        self._used_rows.update(range(len(self._rows)))
        self._decoded_members.update(row.member for row in self._rows)


def open_lafan_bvh_source(
    path: Path,
    source_root: Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
    verified_artifact_sha256: str,
) -> LafanBvhZipClips:
    """Open concrete clip rows over the official immutable LAFAN zip dependency."""
    if not path.is_relative_to(source_root):
        raise ValueError("LAFAN clip rows must reside below their explicit source root.")
    if verified_artifact_sha256 != split.artifact_sha256:
        raise ValueError("LAFAN clip-row digest was not verified by the source boundary.")
    dependencies = source.resolve_dependencies(source_root)
    if set(dependencies) != {"lafan_zip"}:
        raise ValueError("Raw LAFAN clip rows require exactly one dependency named 'lafan_zip'.")
    if source.source_fps is not None:
        raise ValueError("Raw LAFAN clips declare their frame time inside each BVH member.")
    if source.clip_directory is not None:
        raise ValueError("The official LAFAN source is a monolithic zip and must not declare clip_directory.")
    dependency = next(item for item in source.dependencies if item.name == "lafan_zip")
    return LafanBvhZipClips(
        dependencies["lafan_zip"],
        path,
        verified_zip_sha256=dependency.artifact_sha256,
        verified_rows_sha256=verified_artifact_sha256,
        expected_source_sha256=split.source_content_sha256,
    )
