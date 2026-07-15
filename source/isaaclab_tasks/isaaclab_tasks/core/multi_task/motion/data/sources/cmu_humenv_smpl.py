# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode the native HumEnv one-HDF5-per-clip format."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import convert_quat

from ....kinematics import ordered_hinge_rotation
from ...identity import file_sha256, validate_sha256
from ..clip_index import MotionClipIndex
from ..skeleton import MotionSkeleton
from ..source import MotionSourceCfg
from .cmu_humenv_smpl_coordinates import cmu_humenv_smpl_skeleton

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


def _ordered_sources_sha256(sources: Sequence[tuple[str, str]]) -> str:
    """Hash ordered source identifiers and their file digests."""
    payload = json.dumps(sources, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CmuHumEnvSmplClip:
    """One validated CMU clip in native HumEnv SMPL coordinates."""

    generalized_position: np.ndarray
    """SMPL generalized positions, shape ``[frame_count, 76]``."""

    generalized_velocity: np.ndarray
    """SMPL generalized velocities [m/s, rad/s], shape ``[frame_count, 75]``."""

    source_fps: float
    """Native sample rate [Hz]."""

    def __post_init__(self) -> None:
        """Validate the typed CMU HumEnv SMPL source boundary once."""
        frame_count = int(self.generalized_position.shape[0]) if self.generalized_position.ndim == 2 else 0
        fields = (
            ("generalized_position", self.generalized_position, np.float32, (frame_count, 76)),
            ("generalized_velocity", self.generalized_velocity, np.float32, (frame_count, 75)),
        )
        if frame_count < 1:
            raise ValueError("CMU HumEnv SMPL clips must contain at least one frame.")
        for name, value, dtype, shape in fields:
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
                raise ValueError(f"CMU HumEnv SMPL field {name!r} has the wrong dtype or shape.")
            if not value.flags.c_contiguous:
                raise ValueError(f"CMU HumEnv SMPL field {name!r} must be C-contiguous.")
            if not np.isfinite(value).all():
                raise ValueError(f"CMU HumEnv SMPL field {name!r} must contain only finite values.")
        if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError("CMU HumEnv SMPL source_fps must be finite and positive [Hz].")

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""
        return int(self.generalized_position.shape[0])

    def free_root_coordinates(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode exact SMPL free-root coordinates in Newton conventions."""
        _validate_smpl_xyz_chains(source_skeleton)
        native_position = torch.as_tensor(self.generalized_position, device=device)
        native_velocity = torch.as_tensor(self.generalized_velocity, device=device)
        expected_position = (self.frame_count, 7 + source_skeleton.num_joints)
        expected_velocity = (self.frame_count, 6 + source_skeleton.num_joints)
        if (
            native_position.shape != expected_position
            or native_velocity.shape != expected_velocity
            or native_position.dtype is not torch.float32
            or native_velocity.dtype is not torch.float32
        ):
            raise ValueError("HumEnv generalized coordinates differ from the declared SMPL source skeleton.")
        root_rotation = convert_quat(native_position[:, 3:7], to="xyzw")
        position = torch.cat((native_position[:, :3], root_rotation, native_position[:, 7:]), dim=-1)
        velocity = native_velocity
        return position, velocity

    def local_pose(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the HumEnv world root rotation and non-root local pose deltas."""
        _validate_smpl_xyz_chains(source_skeleton)
        generalized_position = torch.as_tensor(self.generalized_position, device=device)
        expected_shape = (self.frame_count, 7 + source_skeleton.num_joints)
        if generalized_position.shape != expected_shape or generalized_position.dtype is not torch.float32:
            raise ValueError(f"HumEnv generalized positions must be float32 with shape {expected_shape}.")

        local_xyzw = torch.zeros(
            self.frame_count,
            source_skeleton.num_bodies,
            4,
            dtype=torch.float32,
            device=device,
        )
        local_xyzw[..., 3] = 1.0
        local_xyzw[:, 0].copy_(convert_quat(generalized_position[:, 3:7], to="xyzw"))
        coordinates = generalized_position[:, 7:].view(self.frame_count, source_skeleton.num_bodies - 1, 3)
        axes = generalized_position.new_tensor(source_skeleton.joint_axes).view(source_skeleton.num_bodies - 1, 3, 3)
        local_xyzw[:, 1:].copy_(ordered_hinge_rotation(coordinates, axes))
        return generalized_position[:, :3], local_xyzw


def _validate_smpl_xyz_chains(skeleton: MotionSkeleton) -> None:
    """Require one declared XYZ hinge chain for every non-root SMPL body."""
    expected_children = tuple(body for body in range(1, skeleton.num_bodies) for _ in range(3))
    xyz_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if skeleton.joint_child_body_indices != expected_children or skeleton.joint_axes != xyz_axes * (
        skeleton.num_bodies - 1
    ):
        raise ValueError("HumEnv SMPL coordinates must declare one ordered XYZ hinge chain per body.")
    if skeleton.root_rotation_convention != "wxyz":
        raise ValueError("HumEnv SMPL root rotations must use the declared wxyz convention.")


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError("HumEnv HDF5 import requires the optional h5py package.") from error
    return h5py


class CmuHumEnvSmplClips:
    """Ordered native HumEnv files decoded one clip at a time."""

    __slots__ = (
        "_clip_ids",
        "_index",
        "_file_sha256s",
        "_paths",
        "_source_fps",
        "_skeleton",
    )

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        file_sha256s: Sequence[str],
        clip_ids: Sequence[str] | None = None,
        source_fps: float,
    ) -> None:
        """Declare files in caller-provided scientific order.

        Args:
            paths: One native HDF5 file per clip, in required clip order.
            file_sha256s: Boundary-verified SHA-256 identity of every path.
            clip_ids: Optional clip identifiers in the same order.
            source_fps: Source sample rate [Hz].
        """
        self._paths = tuple(Path(path) for path in paths)
        if not self._paths:
            raise ValueError("HumEnv HDF5 paths must not be empty.")
        self._clip_ids = tuple(path.stem for path in self._paths) if clip_ids is None else tuple(clip_ids)
        self._file_sha256s = tuple(file_sha256s)
        if len(self._file_sha256s) != len(self._paths):
            raise ValueError("HumEnv file_sha256s must contain one digest per path.")
        for digest in self._file_sha256s:
            validate_sha256("file_sha256s entry", digest)
        if len(self._clip_ids) != len(self._paths):
            raise ValueError("HumEnv clip_ids must contain one identifier per path.")
        self._source_fps = source_fps
        self._skeleton = cmu_humenv_smpl_skeleton()
        self._index: MotionClipIndex | None = None

    @staticmethod
    def _inspect_file(path: Path) -> int:
        if not path.is_file():
            raise FileNotFoundError(path)
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
        return frame_count

    def inspect(self) -> MotionClipIndex:
        """Inspect exact native metadata without retaining decoded clip arrays."""
        if self._index is not None:
            return self._index
        clips = []
        sources = tuple(zip(self._clip_ids, self._file_sha256s, strict=True))
        for clip_id, path, digest in zip(self._clip_ids, self._paths, self._file_sha256s, strict=True):
            frame_count = self._inspect_file(path)
            clips.append(
                MotionClipIndex.Clip(
                    clip_id=clip_id,
                    frame_count=frame_count,
                    source_fps=self._source_fps,
                    content_sha256=digest,
                    skeleton_id=0,
                )
            )
        self._index = MotionClipIndex(
            source_content_sha256=_ordered_sources_sha256(sources),
            skeleton_identity_sha256s=(self._skeleton.identity_sha256,),
            clips=tuple(clips),
        )
        return self._index

    def skeleton(self, skeleton_id: int) -> MotionSkeleton:
        """Return the prepared SMPL coordinate system."""
        if type(skeleton_id) is not int or skeleton_id != 0:
            raise ValueError(f"Unknown skeleton id: {skeleton_id!r}.")
        return self._skeleton

    def clips(self, clip_indices: tuple[int, ...]) -> Iterator[tuple[int, CmuHumEnvSmplClip]]:
        """Yield selected typed clips after closing each HDF5 file."""
        index = self.inspect()
        h5py = _h5py()
        for clip_index in clip_indices:
            if type(clip_index) is not int or clip_index < 0 or clip_index >= len(index.clips):
                raise IndexError(f"Clip index is out of range: {clip_index!r}.")
            path = self._paths[clip_index]
            with h5py.File(path, "r") as stream:
                episode = stream["ep_0"]
                generalized_position = np.asarray(episode["qpos"][...])
                generalized_velocity = np.asarray(episode["qvel"][...])
            yield (
                clip_index,
                CmuHumEnvSmplClip(
                    generalized_position=generalized_position,
                    generalized_velocity=generalized_velocity,
                    source_fps=self._source_fps,
                ),
            )

    def close(self) -> None:
        """Complete the controlled source lifetime; no file remains open between clips."""


def open_cmu_humenv_smpl_source(
    path: Path,
    source_root: Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
    verified_artifact_sha256: str,
) -> CmuHumEnvSmplClips:
    """Open HDF5 clips in the declared split-list order."""
    if verified_artifact_sha256 != split.artifact_sha256:
        raise ValueError("SMPL-CMU split digest was not verified by the source boundary.")
    names = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if not names or any(not name for name in names):
        raise ValueError(f"SMPL-CMU split list must contain only nonempty clip names: {path}")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError(f"SMPL-CMU split entries must be data-root-relative names: {path}")
    if source.source_fps is None:
        raise ValueError("HumEnv HDF5 requires a declared uniform source_fps [Hz].")
    if source.clip_directory is None:
        raise ValueError("SMPL-CMU requires an explicit source-root-relative clip_directory.")
    data_root = source_root / source.clip_directory
    paths = tuple(data_root / name for name in names)
    missing = next((clip_path for clip_path in paths if not clip_path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"SMPL-CMU split clip does not exist: {missing}")
    file_sha256s = tuple(file_sha256(clip_path) for clip_path in paths)
    source_content_sha256 = _ordered_sources_sha256(tuple(zip(names, file_sha256s, strict=True)))
    if source_content_sha256 != split.source_content_sha256:
        raise ValueError(
            f"SMPL-CMU source content hash differs for {split.name}: "
            f"expected {split.source_content_sha256}, got {source_content_sha256}."
        )
    return CmuHumEnvSmplClips(
        paths,
        file_sha256s=file_sha256s,
        clip_ids=names,
        source_fps=source.source_fps,
    )
