# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from isaaclab.utils.configclass import configclass


@configclass
class StateBufferCfg:
    """Configuration for :class:`StateBuffer`.

    Attributes:
        size: Target buffer size (number of states retained after any
            FPS thinning). Without oversample (``oversample_ratio=1.0``)
            this is also the buffer's hard capacity.
        oversample_ratio: When ``> 1.0``, the buffer accumulates up to
            ``size * oversample_ratio`` states then compacts back down
            to ``size`` via :func:`grid_bucket_downsample` over
            :paramref:`fps_features`. The 1.0 default reproduces the
            legacy ring-buffer-with-FIFO-wrap behavior.
        fps_features: Feature extractor used during compaction. Mirrors
            the locomotion pipeline's ``fps_features`` API: either
            a plain callable ``(states: Tensor[N, state_dim]) -> Tensor[N, F]``
            or an object exposing a ``.compute(states) -> Tensor[N, F]``
            method (the ``.compute`` form survives
            :class:`~isaaclab_tasks.utils.PresetCfg` field discovery,
            which filters bare callables out of class attributes).
            ``None`` defaults to ``states[:, :3]`` (xyz of the root pose).
        tag_names_bind: Eval expression to obtain tag name list at
            runtime.
        tag_indices_bind: Eval expression to obtain per-env tag index
            tensor at runtime.
    """

    size: int = 32768
    oversample_ratio: float = 1.0
    fps_features: Callable | None = None
    tag_names_bind: str | None = None
    tag_indices_bind: str | None = None
