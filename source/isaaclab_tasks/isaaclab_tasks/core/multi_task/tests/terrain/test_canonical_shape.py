# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure shape-canonical helpers.

No Warp / kinematics dependency -- these exercise tensor ops only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.terrain.retarget.canonical_shape import (
    canonicalize_shape,
    yaw_from_xy_layout,
)


def _quadruped_nominals() -> torch.Tensor:
    """Nominal hip azimuths for a unit-square quadruped (FL/FR/HL/HR)."""
    offsets = torch.tensor(
        [
            [0.3, 0.2],
            [0.3, -0.2],
            [-0.3, 0.2],
            [-0.3, -0.2],
        ],
        dtype=torch.float32,
    )
    return torch.atan2(offsets[:, 1], offsets[:, 0])


def _default_stance(height: float = 0.5) -> torch.Tensor:
    """Default-stance foot world positions for a unit-square quadruped."""
    return torch.tensor(
        [
            [0.3, 0.2, -height],
            [0.3, -0.2, -height],
            [-0.3, 0.2, -height],
            [-0.3, -0.2, -height],
        ],
        dtype=torch.float32,
    )


def _rotz(yaw: float) -> torch.Tensor:
    c, s = np.cos(yaw), np.sin(yaw)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def test_yaw_zero_on_default_stance():
    nominals = _quadruped_nominals()
    feet = _default_stance()
    yaw = yaw_from_xy_layout(feet, nominals)
    assert torch.allclose(yaw, torch.zeros_like(yaw), atol=1e-6)


@pytest.mark.parametrize("true_yaw", [-1.2, -0.3, 0.0, 0.7, 1.5])
def test_yaw_recovers_applied_rotation(true_yaw):
    nominals = _quadruped_nominals()
    feet = _default_stance()
    R = _rotz(true_yaw)
    feet_rot = feet @ R.T + torch.tensor([2.5, -1.1, 0.4])
    yaw = yaw_from_xy_layout(feet_rot, nominals)
    assert torch.allclose(yaw, torch.tensor(true_yaw, dtype=torch.float32), atol=1e-5)


def test_canonical_invariant_under_translation_and_yaw():
    nominals = _quadruped_nominals()
    feet = _default_stance()
    base = canonicalize_shape(feet, nominals)
    R = _rotz(0.6)
    moved = feet @ R.T + torch.tensor([3.0, -2.0, 0.7])
    transformed = canonicalize_shape(moved, nominals)
    assert torch.allclose(base, transformed, atol=1e-5)


def test_canonical_batched_matches_unbatched():
    nominals = _quadruped_nominals()
    feet_a = _default_stance()
    feet_b = _default_stance() + torch.tensor([0.02, -0.01, 0.0])
    batch = torch.stack([feet_a, feet_b], dim=0)
    out = canonicalize_shape(batch, nominals)
    assert out.shape == (2, 4, 3)
    assert torch.allclose(out[0], canonicalize_shape(feet_a, nominals), atol=1e-6)
    assert torch.allclose(out[1], canonicalize_shape(feet_b, nominals), atol=1e-6)


def test_canonical_identity_only_flat_stance_z():
    """A flat stance has all feet at the centroid's z; canonical z is 0."""
    nominals = _quadruped_nominals()
    feet = _default_stance(height=0.5)
    canon = canonicalize_shape(feet, nominals)
    # rotating each foot by -nominal aligns it with +x axis -> canon_y ~ 0
    assert torch.allclose(canon[:, 1], torch.zeros(4), atol=1e-6)
    # all four feet share the centroid's z -> canon_z is identically zero.
    assert torch.allclose(canon[:, 2], torch.zeros(4), atol=1e-6)


@pytest.mark.parametrize("nc", [2, 3])
def test_canonical_rank_deficient_plane_does_not_nan(nc):
    """With nc < 3 (or collinear contacts) the plane fit is rank deficient;
    the ``flat`` branch should zero pitch/roll without producing NaN."""
    # place nc contacts along the +x axis at z=-0.4 (fully collinear).
    xs = torch.linspace(-0.2, 0.2, nc)
    feet = torch.stack([xs, torch.zeros(nc), torch.full((nc,), -0.4)], dim=-1)
    nominals = torch.linspace(-0.2, 0.2, nc)
    canon = canonicalize_shape(feet, nominals)
    assert torch.isfinite(canon).all()
    assert canon.shape == (nc, 3)


def test_canonical_flat_stance_is_height_invariant():
    """Since canonicalize centres on the polygon centroid, a flat-ground
    default stance produces identical canonical shapes at any uniform
    stance height."""
    nominals = _quadruped_nominals()
    s1 = _default_stance(height=0.5)
    s2 = _default_stance(height=0.6)
    c1 = canonicalize_shape(s1, nominals)
    c2 = canonicalize_shape(s2, nominals)
    assert torch.allclose(c1, c2, atol=1e-6)
