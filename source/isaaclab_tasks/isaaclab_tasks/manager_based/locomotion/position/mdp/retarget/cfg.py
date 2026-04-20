# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SupportSamplingCfg:
    """Configuration for 2D support polygon sampling."""

    num_candidates: int = 5000
    """Number of flat contact patches to sample on the terrain."""

    contact_radius: float = 0.04
    """Contact patch radius for morphological flatness check [m]."""

    max_height_diff: float = 0.03
    """Maximum height variation within a contact patch [m]."""

    horizontal_scale: float = 0.03
    """Heightmap rasterization grid spacing [m]."""

    oversample_ratio: float = 3.0
    """Oversample factor for farthest-point refinement of candidates."""

    search_radius: float = 0.5
    """Radius around each center point to search for support contacts [m]."""

    min_center_dist: float = 0.05
    """Minimum distance from center to candidate (avoid overlapping) [m]."""

    # -- Support polygon quality thresholds --
    min_diagonal_ratio: float = 0.3
    """Minimum ratio of shorter to longer diagonal (reject extreme skew)."""

    min_longitudinal_spread: float = 0.1
    """Minimum front-to-back spread [m]."""

    min_lateral_spread: float = 0.05
    """Minimum left-to-right spread [m]."""

    min_diagonal_length: float = 0.15
    """Minimum diagonal length [m]."""

    min_base_above_contacts: float = 0.2
    """Minimum standing height above contact centroid [m]."""

    max_base_above_contacts: float = 1.5
    """Maximum standing height above contact centroid [m]."""

    oversample_candidates: int = 3
    """Geometry candidates per desired output (for oversampling before IK)."""


@dataclass
class IKCfg:
    """Configuration for Stage 2+3: IK input computation and batched solve."""

    foot_weight: float = 1.0
    """Weight on foot position objectives."""

    base_pos_weight: float = 0.05
    """Weight on base position regularization (soft)."""

    base_rot_weight: float = 0.5
    """Weight on base orientation regularization."""

    joint_limit_weight: float = 10.0
    """Weight on joint limit penalty in IK."""

    iterations: int = 50
    """Number of IK solver iterations."""

    roll_damping: float = 0.3
    """Fraction of terrain roll to include in rotation target (0=level, 1=full terrain roll)."""

    # -- Joint limit tightening (applied to Newton model before IK) --
    haa_dof_indices: list[int] = field(default_factory=lambda: [6, 9, 12, 15])
    """DOF indices for hip abduction/adduction joints."""

    haa_max: float = 0.85
    """Maximum HAA angle for IK constraint [rad] (~49 deg)."""

    hfe_dof_indices: list[int] = field(default_factory=lambda: [7, 10, 13, 16])
    """DOF indices for hip flexion/extension joints."""

    hfe_max: float = 2.1
    """Maximum HFE angle for IK constraint [rad] (~120 deg)."""


@dataclass
class RetargetPipelineCfg:
    """Top-level configuration for the full retargeting pipeline."""

    sampling: SupportSamplingCfg = field(default_factory=SupportSamplingCfg)
    ik: IKCfg = field(default_factory=IKCfg)

    max_candidates: int = 2000
    """Maximum number of candidates in the buffer."""

    device: str = "cuda:0"
    """Warp device."""
