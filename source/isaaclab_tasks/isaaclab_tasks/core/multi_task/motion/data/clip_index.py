# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact deterministic clip metadata retained after source decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ._identity import canonical_sha256, validate_nonempty, validate_sha256


@dataclass(frozen=True, slots=True)
class MotionClipIndex:
    """Immutable ordered index of motion source clips.

    Paths and licensing remain provenance. The content identity excludes
    location-only metadata so moving identical source bytes does not invalidate
    a robot-motion cache.
    """

    @dataclass(frozen=True, slots=True)
    class Clip:
        """One ordered source clip descriptor."""

        clip_id: str
        source_path: str
        frame_count: int
        source_fps: float
        split: str
        tags: tuple[str, ...]
        content_sha256: str
        valid: bool = True

        def __post_init__(self) -> None:
            """Validate one compact clip descriptor."""
            validate_nonempty("clip_id", self.clip_id)
            validate_nonempty("source_path", self.source_path)
            validate_nonempty("split", self.split)
            validate_sha256("content_sha256", self.content_sha256)
            if self.frame_count < 1:
                raise ValueError("frame_count must be positive.")
            if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
                raise ValueError("source_fps must be finite and positive.")
            if self.tags != tuple(sorted(set(self.tags))):
                raise ValueError("tags must be unique and sorted.")

    source_content_sha256: str
    skeleton_sha256: str
    semantic_level: str
    license: str
    clips: tuple[Clip, ...]
    identity_sha256: str = field(init=False)
    content_identity_sha256: str = field(init=False)
    clip_ids: tuple[str, ...] = field(init=False)
    offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Validate deterministic order and freeze metadata/content identities."""
        validate_sha256("source_content_sha256", self.source_content_sha256)
        validate_sha256("skeleton_sha256", self.skeleton_sha256)
        validate_nonempty("semantic_level", self.semantic_level)
        validate_nonempty("license", self.license)
        if not self.clips:
            raise ValueError("clips must not be empty.")
        clip_ids = tuple(clip.clip_id for clip in self.clips)
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("clip ids must be unique.")
        offsets = [0]
        for clip in self.clips:
            offsets.append(offsets[-1] + clip.frame_count)
        object.__setattr__(self, "clip_ids", clip_ids)
        object.__setattr__(self, "offsets", tuple(offsets))

        metadata = {
            "source_content_sha256": self.source_content_sha256,
            "skeleton_sha256": self.skeleton_sha256,
            "semantic_level": self.semantic_level,
            "license": self.license,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "source_path": clip.source_path,
                    "frame_count": clip.frame_count,
                    "source_fps": clip.source_fps,
                    "split": clip.split,
                    "tags": clip.tags,
                    "content_sha256": clip.content_sha256,
                    "valid": clip.valid,
                }
                for clip in self.clips
            ],
        }
        content = {
            "source_content_sha256": self.source_content_sha256,
            "skeleton_sha256": self.skeleton_sha256,
            "semantic_level": self.semantic_level,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "frame_count": clip.frame_count,
                    "source_fps": clip.source_fps,
                    "content_sha256": clip.content_sha256,
                    "valid": clip.valid,
                }
                for clip in self.clips
            ],
        }
        object.__setattr__(self, "identity_sha256", canonical_sha256(metadata))
        object.__setattr__(self, "content_identity_sha256", canonical_sha256(content))

    @property
    def total_frames(self) -> int:
        """Exact flat frame capacity required by this index."""
        return self.offsets[-1]
