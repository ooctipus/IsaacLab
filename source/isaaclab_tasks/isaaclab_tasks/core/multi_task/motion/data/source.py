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
from typing import Literal, Protocol, TypeVar, runtime_checkable

import torch

from isaaclab.utils.configclass import configclass

from ..identity import file_sha256, validate_sha256
from .clip_index import MotionClipIndex
from .skeleton import MotionSkeleton

SourceClipT = TypeVar("SourceClipT", covariant=True)


@runtime_checkable
class MotionSourceClip(Protocol):
    """One decoded clip exposing exact coordinates and semantic pose lazily.

    Sources own representation decoding. A selected robot consumes exactly one
    of the available views after the table resolves its coordinate route.
    Generalized positions use world translation and an xyzw free-root
    quaternion. Generalized velocities, when native evidence exists, use world
    root linear velocity, root-local angular velocity, then joint rates.
    """

    @property
    def source_fps(self) -> float:
        """Native sample rate [Hz]."""

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""

    def free_root_coordinates(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode source free-root positions and optional velocities."""

    def local_pose(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode world root positions and xyzw root/local body rotations.

        Args:
            source_skeleton: Declared source-body topology and coordinate convention.
            device: Destination device for the decoded tensor.

        Returns:
            Root positions [m], shape [frame_count, 3], and unit xyzw
            rotations, shape [frame_count, body_count, 4]. Rotation row zero is
            world-root orientation; remaining rows are parent-local deltas.
        """


class MotionClipSource(Protocol[SourceClipT]):
    """Decoded clips consumed once in their declared order."""

    def inspect(self) -> MotionClipIndex:
        """Return compact ordered clip metadata."""

    def skeleton(self, skeleton_id: int) -> MotionSkeleton:
        """Return one immutable source coordinate system declared by :meth:`inspect`."""

    def clips(self, clip_indices: tuple[int, ...]) -> Iterator[tuple[int, SourceClipT]]:
        """Yield selected decoded clips as source-index and clip pairs."""

    def close(self) -> None:
        """Release source-format resources retained during table construction."""


@configclass
class MotionSourceCfg:
    """One immutable native source and its construction-time opener."""

    @configclass
    class DependencyCfg:
        """One named immutable artifact required by every split decoder."""

        name: str = MISSING
        artifact: str = MISSING
        artifact_sha256: str = MISSING

        def __post_init__(self) -> None:
            """Require a source-root-relative, hash-identified dependency."""
            if not self.name or not self.artifact:
                raise ValueError("Motion source dependency names and artifacts must be nonempty.")
            artifact = Path(self.artifact)
            if artifact.is_absolute() or ".." in artifact.parts:
                raise ValueError("Motion source dependencies must be source-root-relative.")
            validate_sha256("motion source dependency artifact_sha256", self.artifact_sha256)

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
    decoder_version: str = MISSING
    """Version of the native-format decoder and coordinate semantics."""

    source_fps: float | None = MISSING
    """Uniform source sample rate [Hz], or None when clips declare their own rates."""

    license: str = MISSING
    clip_directory: str | None = MISSING
    """Source-root-relative directory containing files named by a split artifact, if any."""
    dependencies: tuple[DependencyCfg, ...] = ()
    """Named immutable artifacts shared by both source splits."""

    train: SplitCfg = MISSING
    evaluation: SplitCfg = MISSING
    purpose: Literal["production", "oracle", "training-control"] = "production"
    """Whether this source may build runtime tables or is inspection evidence only.

    ``training-control`` is a campaign-scoped exception (bfm-env-20260805): it marks an
    explicitly registered clone of an oracle source as a control-arm training corpus for
    A/B data-source experiments. Oracle sources themselves remain runtime-refused.
    """

    def __post_init__(self) -> None:
        """Validate the scientific identity and native source clock [Hz]."""
        if any(
            not value
            for value in (self.identifier, self.format, self.semantic_level, self.decoder_version, self.license)
        ):
            raise ValueError("Motion source text fields must be nonempty.")
        if not callable(self.open_source):
            raise TypeError("Motion source open_source must be callable.")
        if self.source_fps is not None and (not math.isfinite(self.source_fps) or self.source_fps <= 0.0):
            raise ValueError("Motion source source_fps must be finite and positive [Hz].")
        if self.clip_directory is not None:
            clip_directory = Path(self.clip_directory)
            if not self.clip_directory or clip_directory.is_absolute() or ".." in clip_directory.parts:
                raise ValueError("Motion source clip_directory must be source-root-relative when provided.")
        dependency_names = tuple(dependency.name for dependency in self.dependencies)
        if len(set(dependency_names)) != len(dependency_names):
            raise ValueError("Motion source dependency names must be unique.")
        if self.train.name == self.evaluation.name:
            raise ValueError("Motion source train and evaluation split names must differ.")
        if self.purpose not in ("production", "oracle", "training-control"):
            raise ValueError("Motion source purpose must be 'production', 'oracle', or 'training-control'.")

    def resolve_dependencies(self, source_root: Path) -> dict[str, Path]:
        """Verify and return every named source dependency exactly once."""
        resolved: dict[str, Path] = {}
        for dependency in self.dependencies:
            path = source_root / dependency.artifact
            if not path.is_file():
                raise FileNotFoundError(f"Motion source dependency does not exist: {path}")
            actual = file_sha256(path)
            if actual != dependency.artifact_sha256:
                raise ValueError(
                    f"Motion source dependency hash differs for {dependency.artifact}: "
                    f"expected {dependency.artifact_sha256}, got {actual}."
                )
            resolved[dependency.name] = path
        return resolved

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
