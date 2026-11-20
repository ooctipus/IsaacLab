# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the reorientation tasks' spawn clearance: the object's lift clears every link, and is the least such."""

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.reorient.clearance import ConvexClearance

pytestmark = pytest.mark.kitless

DRAWS = 4096
SIGNS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def random_quaternions(rng, count):
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)  # (x, y, z, w)


def matrices(q):
    x, y, z, w = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def boxes_overlap(ca, fa, ha, cb, fb, hb, tolerance):
    """Whether oriented boxes overlap deeper than ``tolerance``, by the fifteen separating axes, per draw."""
    axes = [fa[:, :, i] for i in range(3)] + [fb[:, :, j] for j in range(3)]
    axes += [np.cross(fa[:, :, i], fb[:, :, j]) for i in range(3) for j in range(3)]
    separated = np.zeros(len(ca), dtype=bool)
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        axis = axis / np.where(norm > 1e-9, norm, 1.0)[:, None]
        ra = sum(ha[i] * np.abs((fa[:, :, i] * axis).sum(1)) for i in range(3))
        rb = sum(hb[j] * np.abs((fb[:, :, j] * axis).sum(1)) for j in range(3))
        separated |= (norm > 1e-9) & (np.abs(((cb - ca) * axis).sum(1)) >= ra + rb - tolerance)
    return ~separated


def test_lift_is_the_least_that_clears_every_link():
    rng = np.random.default_rng(1701)
    object_half = np.array([0.036, 0.036, 0.036])
    link_halves = [np.array([0.03, 0.012, 0.014]), np.array([0.02, 0.015, 0.01]), np.array([0.05, 0.05, 0.005])]
    # links given as dense clouds (their boxes' corners plus inner points), as authored meshes are
    clouds = [np.concatenate((SIGNS * half, rng.uniform(-half, half, (50, 3)))) for half in link_halves]
    clearance = ConvexClearance(clouds, SIGNS * object_half, (0.0, 0.0, 1.0), "cpu")

    object_q = random_quaternions(rng, DRAWS)
    link_q = random_quaternions(rng, DRAWS * len(clouds)).reshape(DRAWS, len(clouds), 4)
    link_p = rng.uniform(-0.06, 0.06, (DRAWS, len(clouds), 3))
    object_pose = torch.tensor(np.concatenate((np.zeros((DRAWS, 3)), object_q), 1), dtype=torch.float32)
    link_pose = torch.tensor(np.concatenate((link_p, link_q), 2), dtype=torch.float32)
    lift = clearance.lift(object_pose, link_pose).numpy().astype(np.float64)

    def overlapping(t, tolerance):
        centre = np.zeros((DRAWS, 3))
        centre[:, 2] = t
        found = np.zeros(DRAWS, dtype=bool)
        for link, half in enumerate(link_halves):
            found |= boxes_overlap(
                centre, matrices(object_q), object_half, link_p[:, link], matrices(link_q[:, link]), half, tolerance
            )
        return found

    started_inside = overlapping(np.zeros(DRAWS), 1.0e-5)
    assert started_inside.mean() > 0.3
    assert (lift >= 0.0).all()
    assert (lift[~started_inside] < 1.0e-5).all()
    # clear after the lift, to float32 precision, and overlapping just below it: the least clearing lift
    assert not overlapping(lift, 1.0e-5).any()
    lifted = lift > 1.0e-4
    assert overlapping(lift - 1.0e-4, 0.0)[lifted].all()
