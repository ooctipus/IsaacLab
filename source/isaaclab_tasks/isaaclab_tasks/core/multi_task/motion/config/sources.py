# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native motion-source configuration retained after optional decoding."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from ..data import MotionSkeleton
from ..data.importers import BfmG1JoblibClips, HumEnvHdf5Clips
from .source_skeletons import g1_lafan_source_skeleton, smpl_humenv_source_skeleton

if TYPE_CHECKING:
    from ..mdp.commands.motion_task_table import MotionFrameSource

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _open_humenv_source(
    path: str | Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
) -> MotionFrameSource:
    """Open HDF5 clips in the frozen HumEnv split-list order."""
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"SMPL-CMU split list does not exist: {split_path}")
    names = tuple(line.strip() for line in split_path.read_text(encoding="utf-8").splitlines())
    if not names or any(not name for name in names):
        raise ValueError(f"SMPL-CMU split list must contain only nonempty clip names: {split_path}")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError(f"SMPL-CMU split entries must be data-root-relative names: {split_path}")
    if source.source_fps is None:
        raise ValueError("HumEnv HDF5 requires a declared uniform source_fps [Hz].")
    data_root = split_path.parent.parent / "humenv_amass"
    paths = tuple(data_root / name for name in names)
    missing = next((clip_path for clip_path in paths if not clip_path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"SMPL-CMU split clip does not exist: {missing}")
    return HumEnvHdf5Clips(
        paths,
        clip_ids=names,
        source_fps=source.source_fps,
        skeleton_sha256=source.build_skeleton().identity_sha256,
        split=split.name,
        license=source.license,
        semantic_level=source.semantic_level,
    )


def _open_g1_source(
    path: str | Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
) -> MotionFrameSource:
    """Open one native G1 joblib artifact."""
    return BfmG1JoblibClips.load(
        path,
        artifact_sha256=split.artifact_sha256,
        skeleton_sha256=source.build_skeleton().identity_sha256,
        split=split.name,
        license=source.license,
        semantic_level=source.semantic_level,
    )


@configclass
class MotionSourceCfg:
    """One immutable native source and its construction-time opener.

    The opener normalizes file-format construction behind one callable. It is
    used only while constructing the command-owned trajectory table.
    """

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
                if _SHA256.fullmatch(digest) is None:
                    raise ValueError(f"Motion source split {name} must be a lowercase SHA-256 digest.")
            if self.clip_count < 1 or self.frame_count < self.clip_count:
                raise ValueError("Motion source split counts are invalid.")

    identifier: str = MISSING
    open_source: Callable[[str | Path, SplitCfg, MotionSourceCfg], MotionFrameSource] = MISSING
    """Open the selected split from one deployment path."""

    format: str = MISSING
    semantic_level: str = MISSING
    skeleton_factory: Callable[[], MotionSkeleton] = MISSING
    """Construct declared source kinematics outside Hydra serialization."""

    source_fps: float | None = MISSING
    """Uniform source sample rate [Hz], or ``None`` when clips declare their own rates."""

    license: str = MISSING
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
        if self.train.name == self.evaluation.name:
            raise ValueError("Motion source train and evaluation split names must differ.")

    def build_skeleton(self) -> MotionSkeleton:
        """Construct and validate the immutable source coordinate system."""
        skeleton = self.skeleton_factory()
        if not isinstance(skeleton, MotionSkeleton):
            raise TypeError("Motion source skeleton_factory must return MotionSkeleton.")
        return skeleton

    def open_split(self, source_artifact_root: str | Path, split: SplitCfg) -> MotionFrameSource:
        """Verify and open one source-root-relative split artifact."""
        if not source_artifact_root:
            raise ValueError("source_artifact_root must identify the selected motion deployment.")
        path = Path(source_artifact_root).expanduser() / split.artifact
        if not path.is_file():
            raise FileNotFoundError(f"Motion source split artifact does not exist: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != split.artifact_sha256:
            raise ValueError(
                f"Motion source split artifact hash differs for {split.artifact}: "
                f"expected {split.artifact_sha256}, got {actual}."
            )
        return self.open_source(path, split, self)


SMPL_CMU_SOURCE_CFG = MotionSourceCfg(
    identifier="smpl_cmu",
    open_source=_open_humenv_source,
    format="one_hdf5_file_per_clip_with_group_ep_0",
    semantic_level="smpl_robot_state_and_observation",
    skeleton_factory=smpl_humenv_source_skeleton,
    source_fps=30.0,
    license="amass_cmu_and_smpl_registered_source_required",
    train=MotionSourceCfg.SplitCfg(
        name="train",
        artifact="data_preparation/test_train_split/0-CMU_train_0.1.txt",
        artifact_sha256="99929805f4ab531a89bff89837d27c403625d4b4d89d1a4d381b88825548a996",
        source_content_sha256="fe17c0673e1f5d55d985ac135e36c895df81c2cac91417df0e966fd32eb3e6b6",
        clip_count=1_638,
        frame_count=730_307,
    ),
    evaluation=MotionSourceCfg.SplitCfg(
        name="test",
        artifact="data_preparation/test_train_split/0-CMU_test_0.1.txt",
        artifact_sha256="c9b77782f5c35e0a33b3daa18c110856554acbebed00c4b2877b836d53f9b1b7",
        source_content_sha256="2621cf6d60231a1a6c319d9ab1c44d66c13bb76de4c7bad7e7bdef2d57f0ed32",
        clip_count=182,
        frame_count=88_364,
    ),
)

G1_LAFAN_SOURCE_CFG = MotionSourceCfg(
    identifier="g1_lafan",
    open_source=_open_g1_source,
    format="joblib_pickle_mapping_clip_name_to_field_mapping",
    semantic_level="robot_pose_g1_not_canonical_lafan",
    skeleton_factory=g1_lafan_source_skeleton,
    source_fps=30.0,
    license="retargeted_lafan_redistribution_requires_provenance_review",
    train=MotionSourceCfg.SplitCfg(
        name="training",
        artifact="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
        artifact_sha256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
        source_content_sha256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
        clip_count=862,
        frame_count=258_600,
    ),
    evaluation=MotionSourceCfg.SplitCfg(
        name="evaluation",
        artifact="humanoidverse/data/lafan_29dof.pkl",
        artifact_sha256="f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
        source_content_sha256="f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
        clip_count=40,
        frame_count=264_705,
    ),
)


__all__ = [
    "G1_LAFAN_SOURCE_CFG",
    "SMPL_CMU_SOURCE_CFG",
    "MotionSourceCfg",
]
