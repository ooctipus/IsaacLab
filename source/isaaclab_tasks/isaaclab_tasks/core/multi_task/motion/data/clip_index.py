# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact deterministic clip metadata retained after source decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..identity import validate_nonempty, validate_sha256


@dataclass(frozen=True, slots=True)
class MotionClipIndex:
    """Immutable ordered runtime index of one motion source."""

    @dataclass(frozen=True, slots=True)
    class Clip:
        """One ordered source clip descriptor needed after inspection."""

        clip_id: str
        frame_count: int
        source_fps: float
        content_sha256: str
        skeleton_id: int
        source_clip_id: str | None = None
        source_frame_start: int = 0

        def __post_init__(self) -> None:
            """Validate one compact clip descriptor."""
            validate_nonempty("clip_id", self.clip_id)
            validate_sha256("content_sha256", self.content_sha256)
            if self.frame_count < 1:
                raise ValueError("frame_count must be positive.")
            if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
                raise ValueError("source_fps must be finite and positive.")
            if type(self.skeleton_id) is not int or self.skeleton_id < 0:
                raise ValueError("skeleton_id must be a non-negative integer.")
            if self.source_clip_id is not None:
                validate_nonempty("source_clip_id", self.source_clip_id)
            elif self.source_frame_start != 0:
                raise ValueError("Original clips must start at source frame zero.")
            if type(self.source_frame_start) is not int or self.source_frame_start < 0:
                raise ValueError("source_frame_start must be a non-negative integer.")

        @property
        def source_frame_stop(self) -> int:
            """Exclusive source-frame end retained by this clip."""
            return self.source_frame_start + self.frame_count

    source_content_sha256: str
    skeleton_identity_sha256s: tuple[str, ...]
    clips: tuple[Clip, ...]
    clip_ids: tuple[str, ...] = field(init=False)
    skeleton_ids: tuple[int, ...] = field(init=False)
    offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Validate deterministic order and derive immutable clip offsets."""
        validate_sha256("source_content_sha256", self.source_content_sha256)
        if not self.skeleton_identity_sha256s:
            raise ValueError("skeleton_identity_sha256s must not be empty.")
        for digest in self.skeleton_identity_sha256s:
            validate_sha256("skeleton_identity_sha256s entry", digest)
        if not self.clips:
            raise ValueError("clips must not be empty.")
        clip_ids = tuple(clip.clip_id for clip in self.clips)
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("clip ids must be unique.")
        skeleton_ids: list[int] = []
        seen_skeleton_ids: set[int] = set()
        offsets = [0]
        for clip in self.clips:
            if clip.skeleton_id >= len(self.skeleton_identity_sha256s):
                raise ValueError("clip skeleton_id is not declared by skeleton_identity_sha256s.")
            if clip.skeleton_id not in seen_skeleton_ids:
                if clip.skeleton_id != len(skeleton_ids):
                    raise ValueError("skeleton ids must be dense in first-occurrence order.")
                seen_skeleton_ids.add(clip.skeleton_id)
                skeleton_ids.append(clip.skeleton_id)
            offsets.append(offsets[-1] + clip.frame_count)
        if len(skeleton_ids) != len(self.skeleton_identity_sha256s):
            raise ValueError("Every declared skeleton identity must be used by at least one clip.")
        object.__setattr__(self, "clip_ids", clip_ids)
        object.__setattr__(self, "skeleton_ids", tuple(skeleton_ids))
        object.__setattr__(self, "offsets", tuple(offsets))

    @property
    def total_frames(self) -> int:
        """Exact flat frame capacity required by this index."""
        return self.offsets[-1]

    def for_skeleton(self, skeleton_id: int) -> tuple[int, ...]:
        """Return source-order clip indices using one declared skeleton."""
        if type(skeleton_id) is not int or skeleton_id not in self.skeleton_ids:
            raise ValueError(f"Unknown skeleton id: {skeleton_id!r}.")
        return tuple(index for index, clip in enumerate(self.clips) if clip.skeleton_id == skeleton_id)
