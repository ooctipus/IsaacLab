# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode the native HumEnv one-HDF5-per-clip format."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from ..clip_index import MotionClipIndex
from ._hashing import file_sha256, ordered_sources_sha256

HUMENV_HDF5_FIELDS = ("motion_id", "observation", "qpos", "qvel", "terminated", "truncated")
_FIELD_DTYPES = {
    "motion_id": np.dtype(np.int64),
    "observation": np.dtype(np.float64),
    "qpos": np.dtype(np.float32),
    "qvel": np.dtype(np.float32),
    "terminated": np.dtype(np.bool_),
    "truncated": np.dtype(np.bool_),
}
_FIELD_TRAILING_SHAPES = {
    "motion_id": (1,),
    "observation": (358,),
    "qpos": (76,),
    "qvel": (75,),
    "terminated": (1,),
    "truncated": (1,),
}


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError("HumEnv HDF5 import requires the optional h5py package.") from error
    return h5py


class HumEnvHdf5Clips:
    """Ordered native HumEnv files decoded one clip at a time."""

    __slots__ = (
        "_clip_ids",
        "_index",
        "_license",
        "_paths",
        "_semantic_level",
        "_skeleton_sha256",
        "_source_fps",
        "_split",
    )

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        clip_ids: Sequence[str] | None = None,
        source_fps: float,
        skeleton_sha256: str,
        split: str,
        license: str,
        semantic_level: str = "humenv_transition",
    ) -> None:
        """Declare files in caller-provided scientific order.

        Args:
            paths: One native HDF5 file per clip, in required clip order.
            clip_ids: Optional clip identifiers in the same order.
            source_fps: Source sample rate [Hz].
            skeleton_sha256: Identity of the interpreting source skeleton.
            split: Dataset split shared by these files.
            license: Source redistribution status or license identifier.
            semantic_level: Meaning of each native row.
        """
        self._paths = tuple(Path(path) for path in paths)
        if not self._paths:
            raise ValueError("HumEnv HDF5 paths must not be empty.")
        self._clip_ids = tuple(path.stem for path in self._paths) if clip_ids is None else tuple(clip_ids)
        if len(self._clip_ids) != len(self._paths):
            raise ValueError("HumEnv clip_ids must contain one identifier per path.")
        self._source_fps = source_fps
        self._skeleton_sha256 = skeleton_sha256
        self._split = split
        self._license = license
        self._semantic_level = semantic_level
        self._index: MotionClipIndex | None = None

    @staticmethod
    def _inspect_file(path: Path) -> tuple[int, str]:
        if not path.is_file():
            raise FileNotFoundError(path)
        before = file_sha256(path)
        h5py = _h5py()
        with h5py.File(path, "r") as stream:
            if tuple(stream.keys()) != ("ep_0",):
                raise ValueError(f"HumEnv file {path} must contain exactly the ep_0 group.")
            episode = stream["ep_0"]
            if set(episode.keys()) != set(HUMENV_HDF5_FIELDS):
                raise ValueError(f"HumEnv file {path} fields differ from the native contract.")
            frame_count: int | None = None
            for name in HUMENV_HDF5_FIELDS:
                dataset = episode[name]
                if dataset.dtype != _FIELD_DTYPES[name] or dataset.ndim < 1:
                    raise ValueError(f"HumEnv field {name!r} in {path} has the wrong dtype or rank.")
                if tuple(dataset.shape[1:]) != _FIELD_TRAILING_SHAPES[name]:
                    raise ValueError(f"HumEnv field {name!r} in {path} has the wrong trailing shape.")
                if frame_count is None:
                    frame_count = int(dataset.shape[0])
                elif dataset.shape[0] != frame_count:
                    raise ValueError(f"HumEnv fields in {path} do not share one frame count.")
            if frame_count is None or frame_count < 1:
                raise ValueError(f"HumEnv file {path} contains no frames.")
            terminated = np.asarray(episode["terminated"][:, 0], dtype=np.bool_)
            truncated = np.asarray(episode["truncated"][:, 0], dtype=np.bool_)
            done = terminated | truncated
            if np.any(terminated & truncated) or np.any(done[:-1]) or not bool(done[-1]):
                raise ValueError(f"HumEnv file {path} must contain one terminal final row.")
        after = file_sha256(path)
        if after != before:
            raise ValueError(f"HumEnv file changed during inspection: {path}.")
        return frame_count, before

    def inspect(self) -> MotionClipIndex:
        """Inspect exact native metadata without retaining decoded clip arrays."""
        if self._index is not None:
            return self._index
        clips = []
        sources = []
        for clip_id, path in zip(self._clip_ids, self._paths, strict=True):
            frame_count, digest = self._inspect_file(path)
            sources.append((clip_id, digest))
            clips.append(
                MotionClipIndex.Clip(
                    clip_id=clip_id,
                    source_path=str(path),
                    frame_count=frame_count,
                    source_fps=self._source_fps,
                    split=self._split,
                    tags=(),
                    content_sha256=digest,
                )
            )
        self._index = MotionClipIndex(
            source_content_sha256=ordered_sources_sha256(sources),
            skeleton_sha256=self._skeleton_sha256,
            semantic_level=self._semantic_level,
            license=self._license,
            clips=tuple(clips),
        )
        return self._index

    def clips(self) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        """Yield one fully decoded clip after closing its HDF5 file."""
        index = self.inspect()
        h5py = _h5py()
        for clip, path in zip(index.clips, self._paths, strict=True):
            if file_sha256(path) != clip.content_sha256:
                raise ValueError(f"HumEnv file changed after inspection: {path}.")
            with h5py.File(path, "r") as stream:
                episode = stream["ep_0"]
                fields = {name: np.asarray(episode[name][...]) for name in HUMENV_HDF5_FIELDS}
            if file_sha256(path) != clip.content_sha256:
                raise ValueError(f"HumEnv file changed during decoding: {path}.")
            yield clip.clip_id, fields

    def close(self) -> None:
        """Complete the controlled source lifetime; no file remains open between clips."""
