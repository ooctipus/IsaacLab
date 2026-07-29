# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feature extractors for the final FPS spatial-thinning step.

The retarget pipeline's final stage runs a grid-bucket downsampler over
*features* of the surviving IK-solved candidates to keep the chosen subset
spatially diverse. By default those features are just the root xyz, so the
sampler thins by Cartesian position only — but real "diversity" often means
more than position. Two states at the same xyz with different orientations,
or different joint configurations, are different *poses* the policy has to
learn.

This module exposes pluggable feature extractors that map a state slab
``[N_valid, joint_coord_count]`` to a feature tensor ``[N_valid, D]``.
Whatever ``D`` they return becomes the metric space the FPS thins in:
:func:`~isaaclab_tasks.core.multi_task.utils.grid_downsample.grid_bucket_downsample`
partitions that ``D``-dim bounding box into buckets sized
``(volume / k)^{1/D}`` and keeps one candidate per bucket.

**Mathematical note on mixing units.** Position is in meters, orientation
in radians, joint angles in radians. Euclidean distance only makes sense
once those quantities share a unit. Each non-position contribution gets a
"characteristic length" weight that expresses *how many meters of position
diversity 1 radian (or 1 rad of joint motion) is worth*. Pick the weights
to reflect what diversity matters most for your policy — typically
``rot_scale ∈ [0.2, 1.0]`` for full SO(3), and ``joint_scale ∈ [0.1, 0.5]``
per joint.

Adding orientation/joint dimensions automatically *refines* the sampling at
the same ``pool_spacing``: the bounding-box volume grows, more buckets
exist along the new axes, and the same target count covers more pose
diversity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import axis_angle_from_quat

from ...utils.grid_downsample import extract_features, grid_bucket_downsample

if TYPE_CHECKING:
    from .buffer import RetargetBuffer

FeatureExtractor = Callable[[torch.Tensor], torch.Tensor]
"""Either a raw callable ``(states: [N_valid, joint_coord_count]) -> [N_valid, D]``
*or* an object exposing a ``compute(states) -> [N_valid, D]`` method.
The caller (pipeline final-FPS step or
:class:`~isaaclab_tasks.core.multi_task.curriculum.StateBuffer`)
dispatches on which one is provided.

Implementations slice the state-row tensor (root layout:
``[xyz(3), quat_xyzw(4), joints]``) and return a tensor on whatever
device the input lives.

The cfg-class extractors below intentionally use ``compute`` instead of
``__call__`` so they survive walking through ``PresetCfg`` field discovery
(which filters callables out of class attributes).
"""


def xyz_features(states: torch.Tensor) -> torch.Tensor:
    """Default extractor — just the root xyz.

    Reproduces the original FPS behavior. Use when only spatial coverage
    matters and orientation/joints don't carry diversity (typical for
    flat terrain or short trajectories).

    Returns:
        ``[N_valid, 3]`` tensor of root translations [m].
    """
    return states[:, 0:3]


def bbox_target_count(features: torch.Tensor, spacing: float) -> int:
    """Derive a target sample count from the feature-space bounding box.

    Treats xy as a 2-D manifold (z is mostly noise from terrain height
    or vertical position) and counts any axes beyond xyz as real extra
    dimensions. The grid bucketer's per-cell side at the chosen
    ``spacing`` then partitions the bbox into ``count`` cells:

    .. code-block:: text

        bbox      = features.amax(0) - features.amin(0)
        xy_area   = bbox[0] * bbox[1]
        extra_vol = bbox[3:].prod() if D > 3 else 1
        D_eff     = 2 + max(0, D - 3)
        count     = max(1, int(xy_area * extra_vol / spacing**D_eff))

    Use when sampling diversity should track the *actual* feature-space
    extent of survivors rather than a fixed budget. Adding orientation
    or joint dimensions to the extractor naturally scales ``count`` up
    because the metric volume to fill is larger.

    Args:
        features: ``[N, D]`` feature tensor produced by an extractor.
        spacing: Desired per-cell side at the FPS metric.

    Returns:
        Target sample count: ``0`` when ``features`` is empty, otherwise
        ``>= 1``.
    """
    if features.shape[0] == 0:
        return 0
    bbox = features.amax(dim=0) - features.amin(dim=0)
    xy_area = float((bbox[0] * bbox[1]).clamp_min(1e-9).item())
    if features.shape[1] > 3:
        extra_vol = float(bbox[3:].clamp_min(1e-9).prod().item())
        d_eff = 2 + (features.shape[1] - 3)
    else:
        extra_vol = 1.0
        d_eff = 2
    return max(1, int(xy_area * extra_vol / float(spacing) ** d_eff))


def apply_final_fps(
    buffer: RetargetBuffer,
    n_desired: int,
    *,
    extractor: FeatureExtractor | None = None,
    spacing: float | None = None,
) -> None:
    """Run the terminal FPS spatial-thinning pass on a post-criteria buffer.

    Conceptually equivalent to:

    .. code-block:: python

        slab = buffer.joint_q_result_t[buffer._selected[:n_candidates]]
        sb = StateBuffer.from_states(slab, target_size=n_select, fps_features=extractor)
        keep = sb.compact()
        buffer.set_selected(buffer._selected[:n_candidates][keep])

    -- it's the in-place RetargetBuffer-flavoured entry point of the
    same operation. The xyz-default fast-path additionally skips the
    full-slab gather (``[N, state_dim]`` -> ``[N, 3]``) since the
    extractor only reads xyz; this saves ~88 MB of transient memory at
    ``N=1M``. For custom extractors the path delegates to
    :meth:`StateBuffer.from_states` so the dispatch lives in one place.

    Expects ``buffer._selected[:buffer.num_selected]`` to hold every
    post-criteria candidate (the shape :meth:`RetargetPipeline.run`
    returns, which always emits the un-thinned set). The function
    rewrites ``buffer._selected`` + ``buffer.num_selected`` in place
    to reflect the survivor subset.

    Args:
        buffer: Buffer with ``_selected[:num_selected]`` already
            populated with post-criteria candidates.
        n_desired: Fallback target count when :paramref:`spacing` is
            ``None``. Clamped to ``num_selected`` from above.
        extractor: Feature extractor; ``None`` -> xyz fast-path.
        spacing: When set, target count is derived from the feature-
            space bounding box via :func:`bbox_target_count`.
    """
    # Local import to avoid a curriculum -> terrain.retarget cycle at
    # module import time; the dependency is one-way at call time only.
    from ...curriculum import StateBuffer  # noqa: PLC0415

    n_candidates = int(buffer.num_selected)
    if n_candidates == 0:
        return

    candidates_idx = buffer._selected[:n_candidates].long()

    if extractor is None:
        # xyz fast-path: skip the full-slab gather entirely. Equivalent
        # to ``StateBuffer.from_states`` over the gathered xyz columns
        # only, but inlined so we don't materialise the unused state
        # columns.
        features = buffer.joint_q_result_t[candidates_idx, 0:3]
        if spacing is not None:
            n_select = min(bbox_target_count(features, spacing), n_candidates)
        else:
            n_select = min(int(n_desired), n_candidates)
        if n_select <= 0:
            buffer.num_selected = 0
            return
        local_idx = grid_bucket_downsample(features, n_select)
        buffer.set_selected(candidates_idx[local_idx])
        return

    # Custom-extractor path: delegate to StateBuffer.from_states so the
    # ``(states, extractor) -> features`` dispatch + thin lives in one
    # place. The slab gather is unavoidable here since custom
    # extractors may read quat / joint columns.
    slab = buffer.joint_q_result_t[candidates_idx]
    if spacing is not None:
        n_select = min(bbox_target_count(extract_features(slab, extractor), spacing), n_candidates)
    else:
        n_select = min(int(n_desired), n_candidates)
    if n_select <= 0:
        buffer.num_selected = 0
        return

    sb = StateBuffer.from_states(slab, target_size=n_select, fps_features=extractor)
    local_idx = sb.compact()
    buffer.set_selected(candidates_idx[local_idx])


@dataclass
class XYZYawFeatures:
    """Extractor returning ``[xyz, yaw_scale × yaw]``.

    Useful for ground locomotion where roll/pitch are usually tracked
    near zero but yaw is the direction the robot faces.

    Args:
        yaw_scale: How many meters of position-diversity one radian of
            yaw is "worth". ``1.0`` makes a full π rad turn equivalent
            to ~3.14 m of translation — usually too aggressive; try
            ``0.3 - 0.5`` for typical locomotion.
    """

    yaw_scale: float = 1.0

    def compute(self, states: torch.Tensor) -> torch.Tensor:
        xyz = states[:, 0:3]
        # Quaternion is (x, y, z, w); yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
        qx, qy, qz, qw = states[:, 3], states[:, 4], states[:, 5], states[:, 6]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return torch.cat([xyz, (self.yaw_scale * yaw).unsqueeze(-1)], dim=-1)

    def feature_volume_contribution(self) -> tuple[int, float]:
        """Extra dimensionality and a-priori volume factor beyond xy.

        Yaw spans ``[-π, π]`` rad; scaled by ``yaw_scale`` it contributes
        a 1-D axis of extent ``2π × yaw_scale``.
        """
        import math

        return 1, 2.0 * math.pi * self.yaw_scale


@dataclass
class XYZAxisAngleFeatures:
    """Extractor returning ``[xyz, rot_scale × axis_angle(quat)]``.

    Embeds full SO(3) orientation as a 3-vector via the quaternion log
    map (axis-angle), so the embedding has dimension 6 and Euclidean
    distance approximates SE(3) geodesic distance for small rotations.
    Has a continuity discontinuity at ``angle = π``, which only matters
    if your candidates routinely span half-rotations — usually not the
    case for robot reset poses.

    Args:
        rot_scale: How many meters of position-diversity one radian of
            rotation is "worth". ``0.5`` is a reasonable default for
            articulated robots; raise to emphasise orientation
            diversity, lower to emphasise position.
    """

    rot_scale: float = 0.5

    def compute(self, states: torch.Tensor) -> torch.Tensor:
        xyz = states[:, 0:3]
        rot_vec = axis_angle_from_quat(states[:, 3:7]) * self.rot_scale
        return torch.cat([xyz, rot_vec], dim=-1)

    def feature_volume_contribution(self) -> tuple[int, float]:
        """3 extra axes (axis-angle), each spanning ``[-π, π]`` rad scaled.

        Bounding-box view (matches what :func:`grid_bucket_downsample`
        uses); the actual SO(3) ball has ~½ this volume but the bucketer
        is axis-aligned regardless, so the bbox is the right estimate.
        """
        import math

        return 3, (2.0 * math.pi * self.rot_scale) ** 3


@dataclass
class XYZJointsFeatures:
    """Extractor returning ``[xyz, joint_scale × selected_joints]``.

    Useful when joint configuration carries diversity that translation
    alone misses — e.g. quadruped postures from "tucked" to "extended"
    that share xyz but should be sampled separately.

    Args:
        joint_scale: How many meters of position-diversity one radian
            of joint motion is "worth". For ``n`` joints contributing,
            joint-space contribution to the bounding box scales as
            ``joint_scale × √n × typical_joint_range``.
        joint_slice: Optional slice into the joint columns (after the
            7-element root pose) to restrict which joints contribute.
            ``None`` uses every joint.
    """

    joint_scale: float = 0.3
    joint_slice: slice | None = None

    def compute(self, states: torch.Tensor) -> torch.Tensor:
        xyz = states[:, 0:3]
        joints = states[:, 7:] if self.joint_slice is None else states[:, 7:][:, self.joint_slice]
        return torch.cat([xyz, self.joint_scale * joints], dim=-1)

    def feature_volume_contribution(self) -> tuple[int, float]:
        """Conservative no-op default — joint count isn't known a priori.

        The number of contributing joints depends on the kin model and
        :attr:`joint_slice`. Without it we'd have to guess wildly. Callers
        that need a feature-aware budget for joint-extended embeddings
        should pass an explicit ``--max_robots`` instead, or subclass and
        override this method with a known ``(n_joints, bbox_volume)``.
        """
        return 0, 1.0
