# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampler contract for Position terrain-stance generation.

Hosts the :class:`SamplerBase` abstract interface, the
:class:`SamplerOutput` return dataclass, and the
:class:`SamplerSizing` / :func:`compute_sampler_sizing` helpers used by the
task-table builder to size :class:`RetargetBuffer` before each sampler call.
"""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

from ...kinematics import NewtonKinematics
from .buffer import RetargetBuffer
from .cfg import SamplerBaseCfg


def resolve_contact_body_names(contact_body_names: Sequence[str] | str, body_names: Sequence[str]) -> list[str]:
    """Resolve exact or regex contact-body names in model order."""
    if isinstance(contact_body_names, str):
        pattern = re.compile(contact_body_names)
        resolved = [name for name in body_names if pattern.fullmatch(name)]
        if not resolved:
            raise ValueError(
                f"Contact body regex {contact_body_names!r} matched no bodies; available={list(body_names)}."
            )
        return resolved
    missing = [name for name in contact_body_names if name not in body_names]
    if missing:
        raise ValueError(f"Contact bodies are missing from the model: missing={missing}, available={list(body_names)}.")
    return list(contact_body_names)


@dataclasses.dataclass(frozen=True)
class SamplerSizing:
    """Back-derived stage sizes for a :class:`SamplerBase` implementation.

    Returned by :meth:`SamplerBase.sizing` so the task-table builder can size the
    shared :class:`RetargetBuffer` before calling the sampler.
    """

    n_final: int
    """Final placements returned to the caller."""

    oversample_candidates: int
    """Polygons sent to IK per final placement (``n_polygons_to_ik / n_final``)."""

    max_neighborhoods: int
    """Neighborhoods (e.g. 4-foot polygon centers) the sampler will assemble.

    The sampler holds these K polygon candidates in *local* tensors during
    the polygon-FPS stage; they are independent of the
    :class:`RetargetBuffer`'s storage capacity and therefore unaffected by
    :attr:`ik_capacity`.
    """

    n_morph_patches: int
    """Morph-patch target for terrain contact sampling."""

    ik_capacity: int
    """Post-FPS workload size — sizes the :class:`RetargetBuffer`.

    The sampler runs polygon-FPS to thin its
    :attr:`max_neighborhoods` polygon pool down to this many rows before
    writing into the buffer; IK + FK + criteria + final-FPS all run at
    this rate. With the default cascade,
    ``ik_capacity = ceil(n_final * final_fps_oversample / criteria_yield)``
    -- i.e. polygon-FPS oversample (the part that would only flow through
    cheap per-polygon scratches) is *excluded* from the buffer's
    per-body slots.
    """


@dataclasses.dataclass
class SamplerOutput:
    """Typed return value from :meth:`SamplerBase.__call__`.

    A sampler writes candidate placements into the shared
    :class:`RetargetBuffer` in-place and returns this summary object.

    Fields:
        num_written: Number of rows written to the buffer.
        reject_stats: Ordered waterfall of rejection counts. Keys are
            sampler-specific (e.g. ``"out_of_reach"``,
            ``"non_convex_stance"``). Values are per-candidate counts
            that sum to ``(total_proposed - num_written)``.
        is_contact: Optional per-slot contact flag, shape
            ``[num_written, num_contacts]`` with bool dtype. ``None``
            means every slot is a hard contact. Stability / collision /
            foot-error criteria consume this mask when non-``None``
            to ignore air-targeted slots.
    """

    num_written: int
    reject_stats: dict[str, int]
    is_contact: torch.Tensor | None = None


def compute_sampler_sizing(
    n_final: int,
    *,
    final_fps_oversample: float = 1.5,
    criteria_yield: float = 0.5,
    polygon_fps_oversample: float = 2.0,
    polygon_assembly_yield: float = 0.8,
    patches_per_polygon: int = 4,
    morph_patch_oversample: float = 4.0,
) -> SamplerSizing:
    """Back-derive sampler stage sizes from the target final placement count.

    Walks the sampling stages backwards, applying an oversample multiplier at
    every downsampling stage and an expected yield rate at every filter
    stage. Produces a :class:`SamplerSizing` that scales with ``n_final``
    rather than hitting any fixed cap.

    The cascade (post → pre):

    1. ``n_final`` placements after final FPS.
    2. ``n_final × final_fps_oversample`` after criteria validation
       (FPS needs a bigger pool to achieve good spatial spread).
    3. ``(2) / criteria_yield`` polygons entering criteria
       (accounts for cost/base-z/collision/lateral-hip rejection).
    4. ``(3) × polygon_fps_oversample`` polygons entering polygon-FPS
       (grid-bucket FPS in the sampler needs a bigger pool too).
    5. ``(4) / polygon_assembly_yield`` neighborhoods -- the polygon-
       level yield captures the fraction passing base-height
       feasibility.
    6. ``n_final × patches_per_polygon × morph_patch_oversample`` morph-
       patches -- decoupled from K. The sampler reuses morph patches
       across centers (a single patch can serve many neighborhoods via
       different center/yaw combinations), so morph sizing only needs
       to cover the terrain densely enough for each foot-sector query
       to return at least one patch. Scaling with ``n_final`` (desired
       final count, a proxy for terrain area) is sufficient.

    Args:
        n_final: Target final placement count (e.g. ``area / spacing**2``).
        final_fps_oversample: Headroom multiplier into the final FPS.
        criteria_yield: Expected fraction of IK solves surviving criteria.
        polygon_fps_oversample: Headroom multiplier into the polygon FPS.
        polygon_assembly_yield: Expected fraction of neighborhoods whose
            assembled polygon passes base-height feasibility.
        patches_per_polygon: Foot contacts per polygon.
        morph_patch_oversample: Morph-patches per foot slot — ensures the
            neighborhood's in-range ball has several valid patches.

    Returns:
        :class:`SamplerSizing` with the stage sizes.
    """
    n_post_criteria = int(np.ceil(n_final * final_fps_oversample))
    n_polygons_pre_criteria = int(np.ceil(n_post_criteria / criteria_yield))
    n_polygons_pre_fps = int(np.ceil(n_polygons_pre_criteria * polygon_fps_oversample))
    K = int(np.ceil(n_polygons_pre_fps / polygon_assembly_yield))
    n_morph_patches = int(np.ceil(n_final * patches_per_polygon * morph_patch_oversample))
    oversample_candidates = max(1, int(np.ceil(n_polygons_pre_criteria / max(n_final, 1))))
    # Use the *integer-multiple* form so the sampler's
    # ``target_n = n_final * oversample_candidates`` never gets clamped by
    # ``buffer.max_candidates`` -- otherwise the post-FPS pool shrinks by
    # up to ``n_final - 1`` rows vs. the pre-fix behavior, costing a
    # small fraction of IK diversity. This is at most ``n_final - 1``
    # rows above ``n_polygons_pre_criteria``.
    ik_capacity = n_final * oversample_candidates
    return SamplerSizing(
        n_final=n_final,
        oversample_candidates=oversample_candidates,
        max_neighborhoods=K,
        n_morph_patches=n_morph_patches,
        ik_capacity=ik_capacity,
    )


class SamplerBase(ABC):
    """Abstract base for terrain-stance sampling strategies.

    Constructed from a cfg, a :class:`NewtonKinematics` instance, and
    the foot body indices.  Subclasses derive any robot geometry they
    need from ``kin`` and ``foot_body_ids`` instead of receiving it
    as explicit constructor arguments.
    """

    def __init__(
        self, cfg: SamplerBaseCfg, kin: NewtonKinematics, foot_body_ids: list[int], generator: torch.Generator
    ):
        self.cfg = cfg
        self.kin = kin
        self.foot_body_ids = foot_body_ids
        self.generator = generator

    @abstractmethod
    def sizing(self, n_desired: int) -> SamplerSizing:
        """Back-derive stage sizes from a target final-robot count.

        The task-table builder calls this before :meth:`__call__` to size the shared
        :class:`RetargetBuffer` — the sampler guarantees it will write at
        most :attr:`SamplerSizing.ik_capacity` rows into the buffer (the
        post-FPS workload).
        """
        ...

    @abstractmethod
    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
        *,
        seed: int,
    ) -> SamplerOutput:
        """Sample keypoints on geometry and write results to *buffer*.

        Args:
            wp_mesh: Terrain warp mesh.
            origin: Terrain origin offset ``[3]``.
            buffer: Pre-allocated retarget buffer (written in-place).
            n_desired: Number of valid candidates to aim for.

        Returns:
            A :class:`SamplerOutput` with ``num_written``, rejection
            waterfall, and optional per-placement metadata.
        """
        ...
