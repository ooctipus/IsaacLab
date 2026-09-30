# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The streaming camera panel: one camera, a fixed list of env tiles, one composite image.

Every visualizer backend shows the same picture and differs only in where it puts it, so the
picture is built once here. :class:`StreamingView` resolves the camera and the env tiles when the
visualizer initializes and never resolves them again; a backend calls :meth:`StreamingView.composite`
and hands the result to its own sink.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.cloner.cloner_cfg import expand_env_regex_ns
from isaaclab.envs.utils.camera_colorizer import SUPPORTED_GT_TYPES, CameraFrameColorizer, sensor_key_for_gt_type

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera

    from .visualizer_cfg import VisualizerCfg

MAX_STREAMING_TILES = 100
"""Upper bound on the number of env tiles a streaming panel composites."""


def resolve_streaming_envs(num_envs: int, streaming_envs: int | list[int], sample_from: list[int] | None = None):
    """Resolve ``streaming_envs`` to a concrete list of env indices, capped at :data:`MAX_STREAMING_TILES`.

    Args:
        num_envs: Number of envs the camera covers.
        streaming_envs: ``int`` to sample that many envs, or the exact indices to show.
        sample_from: Envs to sample from when ``streaming_envs`` is an ``int``, such as the envs a
            visualizer keeps visible. ``None`` samples from every env.

    Returns:
        Sorted env indices, at most :data:`MAX_STREAMING_TILES` of them.
    """
    if isinstance(streaming_envs, list):
        if not streaming_envs or len(streaming_envs) > MAX_STREAMING_TILES:
            raise ValueError(f"streaming_envs must select between 1 and {MAX_STREAMING_TILES} environments.")
        duplicate = len(set(streaming_envs)) != len(streaming_envs)
        if duplicate or any(index not in range(num_envs) for index in streaming_envs):
            raise ValueError(f"streaming_envs must contain unique indices in [0, {num_envs}).")
        return sorted(streaming_envs)
    if streaming_envs <= 0 or streaming_envs > MAX_STREAMING_TILES:
        raise ValueError(f"streaming_envs must be between 1 and {MAX_STREAMING_TILES}.")
    pool = [index for index in sample_from if 0 <= index < num_envs] if sample_from is not None else range(num_envs)
    return sorted(random.sample(list(pool), min(streaming_envs, len(pool))))


def camera_gt_batch(camera: Camera, env_indices: list[int], sensor_key: str) -> torch.Tensor:
    """Return one camera output for selected env indices, as a torch tensor on the camera's device.

    Args:
        camera: Camera sensor to read.
        env_indices: Env indices to select, as indices into the camera's tiled output.
        sensor_key: Key in ``camera.data.output``, such as ``"rgb"`` or ``"distance_to_image_plane"``.

    Returns:
        Tensor of shape ``(len(env_indices), H, W, C)``.
    """
    raw = camera.data.output[sensor_key]
    if isinstance(raw, wp.array):
        raw = wp.to_torch(raw)
    elif hasattr(raw, "torch"):
        raw = raw.torch
    return raw.index_select(0, torch.tensor(env_indices, dtype=torch.long, device=raw.device))


def compose_streaming_grid(frames: list[np.ndarray], n_envs: int, n_gt: int, target_aspect: float = 1.0) -> np.ndarray:
    """Composite streaming frames into one tiled image.

    The layout minimises ``|log(composite_W/composite_H / target_aspect)|`` subject to keeping all
    GT columns of one env on the same row. Pass ``target_aspect=window_width/window_height`` to fill
    the panel.

    Args:
        frames: Flat ``uint8 (H, W, 3)`` frames ordered ``[env0_gt0, env0_gt1, ..., env1_gt0, ...]``.
        n_envs: Number of envs represented in ``frames``.
        n_gt: Number of GT types per env.
        target_aspect: Desired positive, finite width-to-height ratio.

    Returns:
        A ``uint8 (H, W, 3)`` composite.
    """
    if not frames:
        raise ValueError("A streaming composite requires at least one frame.")
    if not (math.isfinite(target_aspect) and target_aspect > 0):
        raise ValueError("target_aspect must be positive and finite.")
    h, w = frames[0].shape[:2]
    env_cols = _best_streaming_cols(n_envs, n_gt, h, w, target_aspect)
    env_rows = math.ceil(n_envs / env_cols)
    canvas = np.zeros((env_rows * h, env_cols * n_gt * w, 3), dtype=np.uint8)
    for env_idx in range(n_envs):
        row, col = divmod(env_idx, env_cols)
        for gt_idx in range(n_gt):
            y0, x0 = row * h, (col * n_gt + gt_idx) * w
            canvas[y0 : y0 + h, x0 : x0 + w] = frames[env_idx * n_gt + gt_idx][..., :3]
    return canvas


def _best_streaming_cols(n_envs: int, n_gt: int, frame_h: int, frame_w: int, target_aspect: float = 1.0) -> int:
    """Env-column count that best matches the target composite aspect ratio.

    Complete rows come first (fewest empty cells), then the closest aspect ratio, then more columns.
    """
    best_cols, best_score = 1, float("inf")
    for cols in range(1, n_envs + 1):
        rows = math.ceil(n_envs / cols)
        aspect_score = abs(math.log(cols * n_gt * frame_w / (rows * frame_h) / target_aspect))
        # Strong penalty for ragged rows; break ties by aspect then prefer more cols.
        score = (rows * cols - n_envs) * 10.0 + aspect_score - cols * 1e-6
        if score < best_score:
            best_score, best_cols = score, cols
    return best_cols


class StreamingView:
    """The camera a visualizer streams from and the composite it produces.

    :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_camera` says where the picture comes from,
    and it is read exactly once, here:

    * a ``str`` -- the planned prim-path expression of the scene camera to stream.

    The scene owns the camera's declaration, clone lifecycle, pose, and updates. This view only
    reads its outputs.
    """

    def __init__(
        self,
        cfg: VisualizerCfg,
        scene_cameras: dict[str, Camera],
        *,
        visible_env_ids: list[int] | None = None,
        target_aspect: float = 1.0,
    ):
        """Resolve the camera and the env tiles this panel shows.

        Args:
            cfg: Visualizer configuration carrying the ``streaming_*`` fields.
            scene_cameras: Plan-validated camera sensors keyed by prim-path expression.
            visible_env_ids: Envs the visualizer keeps visible, sampled from when
                :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_envs` is a count.
            target_aspect: Width-to-height ratio the composite should fill.

        Raises:
            ValueError: If a configured GT type is unsupported.
            RuntimeError: If the named camera is not in the scene.
        """
        self.cfg = cfg
        self._target_aspect = target_aspect
        self._composite: np.ndarray | None = None
        self._composite_token: object = object()

        if not cfg.streaming_gt_types or len(set(cfg.streaming_gt_types)) != len(cfg.streaming_gt_types):
            raise ValueError("streaming_gt_types must contain at least one unique output type.")
        unsupported = [gt for gt in cfg.streaming_gt_types if gt not in SUPPORTED_GT_TYPES]
        if unsupported:
            raise ValueError(
                f"streaming_gt_types contains unsupported type(s) {unsupported}. "
                f"Valid types: {sorted(SUPPORTED_GT_TYPES)}."
            )

        self._visible_env_ids = visible_env_ids
        self.scene_cameras = scene_cameras
        if not isinstance(cfg.streaming_camera, str):
            raise ValueError("streaming_view=True requires an explicit streaming_camera prim-path expression.")
        name = expand_env_regex_ns(cfg.streaming_camera)
        if name not in self.scene_cameras:
            raise RuntimeError(
                f"streaming_camera={name!r} is not a registered camera prim-path expression."
                f" Scene cameras: {sorted(self.scene_cameras)}."
            )
        self.camera: Camera = self.scene_cameras[name]
        self.env_ids = resolve_streaming_envs(self.camera.num_instances, cfg.streaming_envs, visible_env_ids)
        available = frozenset(self.camera.data.output)
        self._sensor_keys = tuple(sensor_key_for_gt_type(gt, available) for gt in cfg.streaming_gt_types)

    @property
    def last_composite(self) -> np.ndarray | None:
        """The composite built most recently, without building a new one.

        A backend reads this instead of :meth:`composite` when the simulation is paused: the picture
        cannot have changed, so re-rendering the camera would only burn GPU time and jitter the
        panel with floating-point differences.
        """
        return self._composite

    def select(self, camera: Camera) -> None:
        """Stream from another camera, re-resolving the env tiles for its instance count.

        Args:
            camera: Scene camera to stream from.
        """
        self.camera = camera
        self.env_ids = resolve_streaming_envs(self.camera.num_instances, self.cfg.streaming_envs, self._visible_env_ids)
        available = frozenset(self.camera.data.output)
        self._sensor_keys = tuple(sensor_key_for_gt_type(gt, available) for gt in self.cfg.streaming_gt_types)
        self.invalidate()

    def invalidate(self) -> None:
        """Drop the cached composite so the next :meth:`composite` rebuilds it."""
        self._composite = None
        self._composite_token = object()

    def composite(self, token: object | None = None) -> np.ndarray | None:
        """Return the tiled composite image, rebuilding it at most once per ``token``.

        Args:
            token: Value identifying the current step. Repeated calls with the same token reuse the
                cached composite; ``None`` always rebuilds.

        Returns:
            A ``uint8 (H, W, 3)`` composite, or ``None`` when the camera produces none of the
            configured GT types.
        """
        if token is not None and token == self._composite_token:
            return self._composite
        batches = {
            gt: camera_gt_batch(self.camera, self.env_ids, sensor_key)
            for gt, sensor_key in zip(self.cfg.streaming_gt_types, self._sensor_keys, strict=True)
        }
        frames = [
            CameraFrameColorizer.colorize(
                batches[gt][tile],
                gt,
                depth_min=self.cfg.streaming_depth_min,
                depth_max=self.cfg.streaming_depth_max,
            )
            for tile in range(len(self.env_ids))
            for gt in self.cfg.streaming_gt_types
        ]
        self._composite = compose_streaming_grid(
            frames, len(self.env_ids), len(self.cfg.streaming_gt_types), self._target_aspect
        )
        self._composite_token = token if token is not None else object()
        return self._composite
