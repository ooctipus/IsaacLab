# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode the native BFM-Zero monolithic G1 joblib format."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .._identity import validate_sha256
from ..clip_index import MotionClipIndex
from ._hashing import NativeField, clip_sha256, file_sha256

_BFM_G1_EVALUATION_FIELDS = (
    "root_trans_offset",
    "pose_aa",
    "dof",
    "root_rot",
    "smpl_joints",
    "fps",
)
_BFM_G1_TRAINING_FIELDS = (*_BFM_G1_EVALUATION_FIELDS, "motion_name")
_FRAME_BUILDER_FIELDS = ("root_trans_offset", "pose_aa", "fps")
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
        raise ImportError("BFM G1 joblib import requires the optional joblib package.") from error
    return joblib


def _validate_clip(clip_id: str, fields: object) -> tuple[int, str, str | None]:
    if not isinstance(fields, dict):
        raise ValueError(f"BFM clip {clip_id!r} must be a native field mapping.")
    field_order = tuple(fields)
    if field_order not in (_BFM_G1_EVALUATION_FIELDS, _BFM_G1_TRAINING_FIELDS):
        raise ValueError(f"BFM clip {clip_id!r} fields differ from the native ordered contract.")
    if any(not isinstance(fields[name], np.ndarray) for name in _FRAME_FIELD_SHAPES):
        raise ValueError(f"BFM clip {clip_id!r} frame fields must be NumPy arrays.")

    root = fields["root_trans_offset"]
    frame_count = int(root.shape[0]) if root.ndim == 2 else 0
    if frame_count < 1:
        raise ValueError(f"BFM clip {clip_id!r} contains no frames.")
    for name, trailing_shape in _FRAME_FIELD_SHAPES.items():
        value = fields[name]
        if value.dtype != np.dtype(np.float32) or value.shape != (frame_count, *trailing_shape):
            raise ValueError(f"BFM clip {clip_id!r} field {name!r} has the wrong dtype or shape.")
        if not value.flags.c_contiguous:
            raise ValueError(f"BFM clip {clip_id!r} field {name!r} must be C-contiguous.")

    fps = fields["fps"]
    if type(fps) is not int or fps != 30:
        raise ValueError(f"BFM clip {clip_id!r} must declare Python int fps=30.")
    motion_name = fields.get("motion_name")
    if motion_name is not None and (type(motion_name) is not str or not motion_name):
        raise ValueError(f"BFM clip {clip_id!r} motion_name must be a nonempty Python string.")
    return frame_count, clip_sha256(fields, field_order), motion_name


class BfmG1JoblibClips:
    """One loaded BFM mapping progressively released as clips are consumed."""

    __slots__ = (
        "_clips",
        "_index",
        "_license",
        "_path",
        "_remaining_frames",
        "_semantic_level",
        "_skeleton_sha256",
        "_source_sha256",
        "_split",
    )

    def __init__(
        self,
        path: Path,
        clips: dict[str, dict[str, NativeField]],
        source_sha256: str,
        *,
        skeleton_sha256: str,
        split: str,
        license: str,
        semantic_level: str,
    ) -> None:
        self._path = path
        self._clips: dict[str, dict[str, NativeField]] | None = clips
        self._source_sha256 = source_sha256
        self._skeleton_sha256 = skeleton_sha256
        self._split = split
        self._license = license
        self._semantic_level = semantic_level
        self._index: MotionClipIndex | None = None
        self._remaining_frames: int | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        artifact_sha256: str,
        skeleton_sha256: str,
        split: str,
        license: str,
        semantic_level: str = "robot_state",
    ) -> BfmG1JoblibClips:
        """Load the unavoidable monolithic source mapping exactly once.

        Args:
            path: Native BFM-Zero joblib file.
            artifact_sha256: File identity already verified by the source config.
            skeleton_sha256: Identity of the interpreting G1 source skeleton.
            split: Dataset split shared by the loaded clips.
            license: Source redistribution status or license identifier.
            semantic_level: Meaning of each native frame.

        Returns:
            A source that releases its mapping progressively during iteration.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        validate_sha256("artifact_sha256", artifact_sha256)
        # A monolithic joblib mapping cannot be streamed from disk. Joblib
        # memmaps every stored ndarray independently, which exhausts ordinary
        # process file-descriptor limits on the native 862-clip corpus. Load
        # regular arrays once, then release each clip during trajectory construction.
        payload = _joblib().load(source_path)
        after = file_sha256(source_path)
        if after != artifact_sha256:
            raise ValueError(f"BFM joblib changed during loading: {source_path}.")
        if not isinstance(payload, dict) or not payload:
            raise ValueError("BFM joblib must contain one nonempty ordered clip mapping.")
        if any(not isinstance(clip_id, str) or not isinstance(fields, dict) for clip_id, fields in payload.items()):
            raise ValueError("BFM joblib clip identifiers and field mappings have invalid types.")
        return cls(
            source_path,
            payload,
            artifact_sha256,
            skeleton_sha256=skeleton_sha256,
            split=split,
            license=license,
            semantic_level=semantic_level,
        )

    def _require_open(self) -> dict[str, dict[str, NativeField]]:
        if self._clips is None:
            raise RuntimeError("BFM joblib source is closed.")
        return self._clips

    def inspect(self) -> MotionClipIndex:
        """Validate all native clips while retaining only compact metadata."""
        if self._index is not None:
            return self._index
        clips = self._require_open()
        descriptors = []
        remaining_frames = 0
        for clip_id, fields in clips.items():
            frame_count, digest, motion_name = _validate_clip(clip_id, fields)
            remaining_frames += frame_count
            descriptors.append(
                MotionClipIndex.Clip(
                    clip_id=clip_id,
                    source_path=f"{self._path}#{clip_id}",
                    frame_count=frame_count,
                    source_fps=30.0,
                    split=self._split,
                    tags=() if motion_name is None else (motion_name,),
                    content_sha256=digest,
                )
            )
        self._index = MotionClipIndex(
            source_content_sha256=self._source_sha256,
            skeleton_sha256=self._skeleton_sha256,
            semantic_level=self._semantic_level,
            license=self._license,
            clips=tuple(descriptors),
        )
        self._remaining_frames = remaining_frames
        return self._index

    @property
    def remaining_clips(self) -> int:
        """Number of source mappings still retained by this importer."""
        return 0 if self._clips is None else len(self._clips)

    @property
    def remaining_frames(self) -> int:
        """Number of frames still retained by this importer."""
        if self._remaining_frames is None:
            self.inspect()
        assert self._remaining_frames is not None
        return self._remaining_frames

    def clips(self) -> Iterator[tuple[str, dict[str, NativeField]]]:
        """Pop native clips and yield one strict frame-builder field mapping."""
        index = self.inspect()
        clips = self._require_open()
        if file_sha256(self._path) != self._source_sha256:
            raise ValueError(f"BFM joblib changed before consumption: {self._path}.")
        for clip in index.clips:
            fields = clips.pop(clip.clip_id)
            assert self._remaining_frames is not None
            self._remaining_frames -= clip.frame_count
            yield clip.clip_id, {name: fields[name] for name in _FRAME_BUILDER_FIELDS}
        if file_sha256(self._path) != self._source_sha256:
            raise ValueError(f"BFM joblib changed during consumption: {self._path}.")

    def close(self) -> None:
        """Release every native array not already yielded to the caller."""
        if self._clips is not None:
            self._clips.clear()
            self._clips = None
        self._remaining_frames = 0

    def __enter__(self) -> BfmG1JoblibClips:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
