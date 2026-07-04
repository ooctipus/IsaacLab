# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Narrow lifetime contract shared by decoded motion sources."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import MISSING
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import torch

from isaaclab.utils.configclass import configclass

from ..identity import file_sha256, validate_sha256
from .clip_index import MotionClipIndex
from .skeleton import MotionSkeleton

SourceClipT = TypeVar("SourceClipT", covariant=True)


@runtime_checkable
class MotionGeneralizedCoordinateClip(Protocol):
    """One free-root clip represented by generalized positions and velocities.

    Position rows contain world root translation, wxyz root rotation, then joint
    coordinates. Velocity rows contain world root-linear, root-local angular,
    then joint velocities.
    """

    @property
    def generalized_position(self) -> np.ndarray:
        """Free-root generalized positions [m, unit quaternion, rad], shape [frame_count, coordinate_count]."""

    @property
    def generalized_velocity(self) -> np.ndarray:
        """Free-root generalized velocities [m/s, rad/s], shape [frame_count, degree_of_freedom_count]."""

    @property
    def source_fps(self) -> float:
        """Native sample rate [Hz]."""

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""


@runtime_checkable
class MotionPoseAxisAngleClip(Protocol):
    """One target-body clip represented by world root translation and rotation vectors."""

    @property
    def root_translation(self) -> np.ndarray:
        """World-frame root translation [m], shape [frame_count, 3]."""

    @property
    def pose_axis_angle(self) -> np.ndarray:
        """Root and joint rotation vectors [rad], shape [frame_count, body_count, 3]."""

    @property
    def source_fps(self) -> float:
        """Native sample rate [Hz]."""

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""


@runtime_checkable
class MotionLocalBodyPoseClip(Protocol):
    """One source clip represented by world root translation and parent-local rotations."""

    @property
    def root_translation(self) -> np.ndarray:
        """World-frame root translation [m], shape [frame_count, 3]."""

    @property
    def source_fps(self) -> float:
        """Native sample rate [Hz]."""

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""

    def local_body_rotation_wxyz(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Decode parent-local body rotations on the requested device.

        Args:
            source_skeleton: Declared source-body topology and coordinate convention.
            device: Destination device for the decoded tensor.

        Returns:
            Parent-local unit wxyz quaternions, shape [frame_count, body_count, 4].
        """


class MotionClipSource(Protocol[SourceClipT]):
    """Decoded clips consumed once in their declared order."""

    def inspect(self) -> MotionClipIndex:
        """Return compact ordered clip metadata."""

    def clips(self) -> Iterator[tuple[str, SourceClipT]]:
        """Yield decoded clips in the order returned by :meth:`inspect`."""

    def close(self) -> None:
        """Release source-format resources retained during table construction."""


@configclass
class MotionSourceCfg:
    """One immutable native source and its construction-time opener."""

    @configclass
    class SplitCfg:
        """One selected artifact, ordered source identity, and exact size."""

        name: str = MISSING
        artifact: str = MISSING
        artifact_sha256: str = MISSING
        source_content_sha256: str = MISSING
        clip_count: int = MISSING
        frame_count: int = MISSING

        def __post_init__(self) -> None:
            """Reject incomplete or mutable split declarations."""
            if not self.name or not self.artifact:
                raise ValueError("Motion source split names and artifacts must be nonempty.")
            artifact = Path(self.artifact)
            if artifact.is_absolute() or ".." in artifact.parts:
                raise ValueError("Motion source split artifacts must be artifact-root-relative.")
            for name, digest in (
                ("artifact_sha256", self.artifact_sha256),
                ("source_content_sha256", self.source_content_sha256),
            ):
                validate_sha256(f"motion source split {name}", digest)
            if self.clip_count < 1 or self.frame_count < self.clip_count:
                raise ValueError("Motion source split counts are invalid.")

    identifier: str = MISSING
    open_source: Callable[[Path, Path, SplitCfg, MotionSourceCfg, str], MotionClipSource] = MISSING
    """Open a boundary-verified split artifact from its explicit deployment root."""

    format: str = MISSING
    semantic_level: str = MISSING
    skeleton_factory: Callable[[], MotionSkeleton] = MISSING
    """Construct declared source kinematics outside Hydra serialization."""

    source_fps: float | None = MISSING
    """Uniform source sample rate [Hz], or None when clips declare their own rates."""

    license: str = MISSING
    clip_directory: str | None = MISSING
    """Source-root-relative directory containing files named by a split artifact, if any."""

    train: SplitCfg = MISSING
    evaluation: SplitCfg = MISSING

    def __post_init__(self) -> None:
        """Validate the scientific identity and native source clock [Hz]."""
        if any(not value for value in (self.identifier, self.format, self.semantic_level, self.license)):
            raise ValueError("Motion source text fields must be nonempty.")
        if not callable(self.open_source):
            raise TypeError("Motion source open_source must be callable.")
        if self.source_fps is not None and (not math.isfinite(self.source_fps) or self.source_fps <= 0.0):
            raise ValueError("Motion source source_fps must be finite and positive [Hz].")
        if not callable(self.skeleton_factory):
            raise TypeError("Motion source skeleton_factory must be callable.")
        if self.clip_directory is not None:
            clip_directory = Path(self.clip_directory)
            if not self.clip_directory or clip_directory.is_absolute() or ".." in clip_directory.parts:
                raise ValueError("Motion source clip_directory must be source-root-relative when provided.")
        if self.train.name == self.evaluation.name:
            raise ValueError("Motion source train and evaluation split names must differ.")

    def build_skeleton(self) -> MotionSkeleton:
        """Construct and validate the immutable source coordinate system."""
        skeleton = self.skeleton_factory()
        if not isinstance(skeleton, MotionSkeleton):
            raise TypeError("Motion source skeleton_factory must return MotionSkeleton.")
        return skeleton

    def open_split(self, source_artifact_root: str | Path, split: SplitCfg) -> MotionClipSource:
        """Verify and open one source-root-relative split artifact."""
        if not source_artifact_root:
            raise ValueError("source_artifact_root must identify the selected motion deployment.")
        source_root = Path(source_artifact_root).expanduser()
        path = source_root / split.artifact
        if not path.is_file():
            raise FileNotFoundError(f"Motion source split artifact does not exist: {path}")
        actual = file_sha256(path)
        if actual != split.artifact_sha256:
            raise ValueError(
                f"Motion source split artifact hash differs for {split.artifact}: "
                f"expected {split.artifact_sha256}, got {actual}."
            )
        return self.open_source(path, source_root, split, self, actual)
