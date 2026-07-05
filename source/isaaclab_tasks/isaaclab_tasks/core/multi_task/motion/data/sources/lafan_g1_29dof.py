# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode the released BFM G1-retargeted LAFAN 29-DoF monolithic joblib format."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import convert_quat, quat_from_rotation_vector

from ...identity import validate_sha256
from ..clip_index import MotionClipIndex
from ..skeleton import MotionSkeleton
from ..source import MotionSourceCfg

NativeField = np.ndarray | int | str


def _clip_sha256(fields: Mapping[str, NativeField], field_order: Sequence[str]) -> str:
    """Hash exact native arrays and honestly encoded Python scalars."""
    digest = hashlib.sha256()
    for name in field_order:
        value = fields[name]
        if isinstance(value, np.ndarray):
            if not value.flags.c_contiguous:
                raise ValueError(f"Native field {name!r} must be C-contiguous.")
            declaration = {"name": name, "dtype": value.dtype.str, "shape": value.shape}
            raw = memoryview(value).cast("B")
        elif type(value) is int:
            declaration = {"name": name, "scalar_type": "python:int", "scalar_encoding": "UTF-8 decimal"}
            raw = str(value).encode()
        elif type(value) is str:
            declaration = {"name": name, "scalar_type": "python:str", "scalar_encoding": "UTF-8"}
            raw = value.encode()
        else:
            raise TypeError(f"Native field {name!r} has unsupported type {type(value).__name__}.")
        metadata = json.dumps(declaration, separators=(",", ":"), sort_keys=True).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


_LAFAN_G1_EVALUATION_FIELDS = (
    "root_trans_offset",
    "pose_aa",
    "dof",
    "root_rot",
    "smpl_joints",
    "fps",
)
_LAFAN_G1_TRAINING_FIELDS = (*_LAFAN_G1_EVALUATION_FIELDS, "motion_name")
_FRAME_FIELD_SHAPES = {
    "root_trans_offset": (3,),
    "pose_aa": (30, 3),
    "dof": (29,),
    "root_rot": (4,),
    "smpl_joints": (24, 3),
}


def _joblib():
    try:
        import joblib
    except ImportError as error:
        raise ImportError("LAFAN G1 joblib import requires the optional joblib package.") from error
    return joblib


def _validate_clip(clip_id: str, fields: object) -> tuple[int, str]:
    if not isinstance(fields, dict):
        raise ValueError(f"LAFAN G1 clip {clip_id!r} must be a native field mapping.")
    field_order = tuple(fields)
    if field_order not in (_LAFAN_G1_EVALUATION_FIELDS, _LAFAN_G1_TRAINING_FIELDS):
        raise ValueError(f"LAFAN G1 clip {clip_id!r} fields differ from the native ordered contract.")
    if any(not isinstance(fields[name], np.ndarray) for name in _FRAME_FIELD_SHAPES):
        raise ValueError(f"LAFAN G1 clip {clip_id!r} frame fields must be NumPy arrays.")

    root = fields["root_trans_offset"]
    frame_count = int(root.shape[0]) if root.ndim == 2 else 0
    if frame_count < 1:
        raise ValueError(f"LAFAN G1 clip {clip_id!r} contains no frames.")
    for name, trailing_shape in _FRAME_FIELD_SHAPES.items():
        value = fields[name]
        if value.dtype != np.dtype(np.float32) or value.shape != (frame_count, *trailing_shape):
            raise ValueError(f"LAFAN G1 clip {clip_id!r} field {name!r} has the wrong dtype or shape.")
        if not value.flags.c_contiguous:
            raise ValueError(f"LAFAN G1 clip {clip_id!r} field {name!r} must be C-contiguous.")

    fps = fields["fps"]
    if type(fps) is not int or fps != 30:
        raise ValueError(f"LAFAN G1 clip {clip_id!r} must declare Python int fps=30.")
    motion_name = fields.get("motion_name")
    if motion_name is not None and (type(motion_name) is not str or not motion_name):
        raise ValueError(f"LAFAN G1 clip {clip_id!r} motion_name must be a nonempty Python string.")
    return frame_count, _clip_sha256(fields, field_order)


@dataclass(frozen=True, slots=True)
class LafanG1Clip:
    """One validated released BFM G1-retargeted LAFAN 29-DoF pose clip."""

    root_translation: np.ndarray
    """Root translation [m], shape ``[frame_count, 3]``."""

    pose_axis_angle: np.ndarray
    """Root rotation vector and hinge-axis vectors [rad], shape ``[frame_count, 30, 3]``."""

    source_fps: float
    """Native sample rate [Hz]."""

    def __post_init__(self) -> None:
        """Validate the typed LAFAN G1 source boundary once."""
        frame_count = int(self.root_translation.shape[0]) if self.root_translation.ndim == 2 else 0
        if frame_count < 3:
            raise ValueError("LAFAN G1 clips require at least three source frames.")
        if self.root_translation.shape != (frame_count, 3) or self.pose_axis_angle.shape != (frame_count, 30, 3):
            raise ValueError("LAFAN G1 clip tensors have invalid shapes.")
        if self.root_translation.dtype != np.float32 or self.pose_axis_angle.dtype != np.float32:
            raise ValueError("LAFAN G1 clip tensors must use float32.")
        if not self.root_translation.flags.c_contiguous or not self.pose_axis_angle.flags.c_contiguous:
            raise ValueError("LAFAN G1 clip tensors must be C-contiguous.")
        if self.source_fps != 30.0:
            raise ValueError("LAFAN G1 clips must use the native 30 Hz sample rate.")

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""
        return int(self.root_translation.shape[0])

    def local_body_rotation_wxyz(self, source_skeleton: MotionSkeleton, *, device: str | torch.device) -> torch.Tensor:
        """Decode the released G1 hinge rows as parent-local body rotations."""
        expected_children = tuple(range(1, source_skeleton.num_bodies))
        if (
            source_skeleton.num_bodies != self.pose_axis_angle.shape[1]
            or source_skeleton.joint_child_body_indices != expected_children
            or source_skeleton.root_rotation_convention != "axis_angle"
        ):
            raise ValueError("LAFAN G1 pose rows differ from the declared scalar-hinge source skeleton.")
        pose_axis_angle = torch.as_tensor(self.pose_axis_angle, device=device)
        if not torch.all(torch.isfinite(pose_axis_angle)):
            raise ValueError("LAFAN G1 pose rows must contain only finite values.")
        axes = pose_axis_angle.new_tensor(source_skeleton.joint_axes)
        joint_rotation = pose_axis_angle[:, 1:]
        coordinate = torch.sum(joint_rotation * axes, dim=-1, keepdim=True)
        if not torch.allclose(joint_rotation, coordinate * axes, atol=2.0e-6, rtol=2.0e-6):
            raise ValueError("LAFAN G1 non-root rotations must lie on their declared hinge axes.")
        return convert_quat(quat_from_rotation_vector(pose_axis_angle), to="wxyz")


class LafanG1JoblibClips:
    """One loaded LAFAN G1 mapping progressively released as clips are consumed."""

    __slots__ = (
        "_clips",
        "_index",
        "_source_sha256",
    )

    def __init__(self, clips: dict[str, dict[str, NativeField]], source_sha256: str) -> None:
        self._clips: dict[str, dict[str, NativeField]] | None = clips
        self._source_sha256 = source_sha256
        self._index: MotionClipIndex | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verified_artifact_sha256: str,
    ) -> LafanG1JoblibClips:
        """Load the unavoidable monolithic source mapping exactly once.

        Args:
            path: Native LAFAN G1 29-DoF joblib file.
            verified_artifact_sha256: File identity already verified by the source config.

        Returns:
            A source that releases its mapping progressively during iteration.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        validate_sha256("verified_artifact_sha256", verified_artifact_sha256)
        # A monolithic joblib mapping cannot be streamed from disk. Joblib
        # memmaps every stored ndarray independently, which exhausts ordinary
        # process file-descriptor limits on the native 862-clip corpus. Load
        # regular arrays once, then release each clip during trajectory construction.
        payload = _joblib().load(source_path)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("LAFAN G1 joblib must contain one nonempty ordered clip mapping.")
        if any(not isinstance(clip_id, str) or not isinstance(fields, dict) for clip_id, fields in payload.items()):
            raise ValueError("LAFAN G1 joblib clip identifiers and field mappings have invalid types.")
        return cls(payload, verified_artifact_sha256)

    def _require_open(self) -> dict[str, dict[str, NativeField]]:
        if self._clips is None:
            raise RuntimeError("LAFAN G1 joblib source is closed.")
        return self._clips

    def inspect(self) -> MotionClipIndex:
        """Validate all native clips while retaining only compact metadata."""
        if self._index is not None:
            return self._index
        clips = self._require_open()
        descriptors = []
        for clip_id, fields in clips.items():
            frame_count, digest = _validate_clip(clip_id, fields)
            descriptors.append(
                MotionClipIndex.Clip(
                    clip_id=clip_id,
                    frame_count=frame_count,
                    source_fps=30.0,
                    content_sha256=digest,
                )
            )
        self._index = MotionClipIndex(
            source_content_sha256=self._source_sha256,
            clips=tuple(descriptors),
        )
        return self._index

    def clips(self) -> Iterator[tuple[str, LafanG1Clip]]:
        """Pop native mappings and yield typed clips in declared order."""
        index = self.inspect()
        clips = self._require_open()
        for clip in index.clips:
            fields = clips.pop(clip.clip_id)
            yield (
                clip.clip_id,
                LafanG1Clip(
                    root_translation=fields["root_trans_offset"],
                    pose_axis_angle=fields["pose_aa"],
                    source_fps=float(fields["fps"]),
                ),
            )

    def close(self) -> None:
        """Release every native array not already yielded to the caller."""
        if self._clips is not None:
            self._clips.clear()
            self._clips = None


def open_lafan_g1_source(
    path: Path,
    source_root: Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
    verified_artifact_sha256: str,
) -> LafanG1JoblibClips:
    """Open one native G1 joblib artifact."""
    if not path.is_relative_to(source_root):
        raise ValueError("LAFAN artifact must reside below its explicit source root.")
    if verified_artifact_sha256 != split.artifact_sha256:
        raise ValueError("LAFAN artifact digest was not verified by the source boundary.")
    if verified_artifact_sha256 != split.source_content_sha256:
        raise ValueError("LAFAN monolithic artifact and declared source-content digests differ.")
    if source.clip_directory is not None:
        raise ValueError("LAFAN is monolithic and must not declare clip_directory.")
    return LafanG1JoblibClips.load(
        path,
        verified_artifact_sha256=verified_artifact_sha256,
    )
