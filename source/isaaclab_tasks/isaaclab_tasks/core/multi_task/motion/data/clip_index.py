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
    clips: tuple[Clip, ...]
    clip_ids: tuple[str, ...] = field(init=False)
    offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Validate deterministic order and derive immutable clip offsets."""
        validate_sha256("source_content_sha256", self.source_content_sha256)
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

    @property
    def total_frames(self) -> int:
        """Exact flat frame capacity required by this index."""
        return self.offsets[-1]
