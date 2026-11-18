# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the NIST demo's loose-part spawn: no two parts' collision geometry overlaps after a reset, and a world
whose drawn parts did not overlap keeps its draws."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.kitless

DEMO = Path(__file__).resolve().parents[4] / "examples" / "demos" / "nist.py"
WORLDS = 8192


@pytest.fixture(scope="module")
def nist():
    spec = importlib.util.spec_from_file_location("nist_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def box_mesh(nist, centre, half):
    """A closed box mesh in a part's root frame, offset from the origin as the NIST parts' meshes are."""
    signs = torch.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    vertices = torch.tensor(centre) + signs * torch.tensor(half)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    triangles = torch.tensor([t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))])
    return nist.part_mesh(vertices, triangles)


def overlapping_worlds(positions, rotations, boxes):
    """Worlds with a pair of parts whose boxes overlap (separating axis test, exact for boxes)."""
    frames = np.stack([np.asarray(rotation_matrix(q)) for q in rotations.reshape(-1, 4)]).reshape(
        *rotations.shape[:2], 3, 3
    )
    positions = positions.numpy()
    found = np.zeros(positions.shape[0], dtype=bool)
    for b in range(10):
        for a in range(b):
            ca = positions[:, a] + frames[:, a] @ boxes[a][0]
            cb = positions[:, b] + frames[:, b] @ boxes[b][0]
            axes = [frames[:, a, :, i] for i in range(3)] + [frames[:, b, :, j] for j in range(3)]
            axes += [np.cross(frames[:, a, :, i], frames[:, b, :, j]) for i in range(3) for j in range(3)]
            separated = np.zeros_like(found)
            for axis in axes:
                norm = np.linalg.norm(axis, axis=1)
                axis = axis / np.where(norm > 1e-9, norm, 1.0)[:, None]
                ra = sum(boxes[a][1][i] * np.abs((frames[:, a, :, i] * axis).sum(1)) for i in range(3))
                rb = sum(boxes[b][1][j] * np.abs((frames[:, b, :, j] * axis).sum(1)) for j in range(3))
                separated |= (norm > 1e-9) & (np.abs(((cb - ca) * axis).sum(1)) > ra + rb)
            found |= ~separated
    return found


def rotation_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def test_seeded_reset_spawns_no_overlapping_parts_and_keeps_clean_worlds(nist):
    # Parts reaching about 9 cm from their origins, as the USB plug does, overlap in some worlds at a 10 cm spacing.
    boxes = [((0.0, 0.0, 0.06), (0.008, 0.012, 0.03 - 0.001 * slot)) for slot in range(10)]
    meshes = tuple(box_mesh(nist, centre, half) for centre, half in boxes)
    bounds = torch.tensor(nist.DROP_POSITION_RANGE)
    torch.manual_seed(1701)
    drawn = nist.draw_part_poses(WORLDS, bounds)
    stream = torch.get_rng_state()
    overlapped = overlapping_worlds(*drawn, boxes)
    assert overlapped.any()

    positions, rotations = drawn[0].clone(), drawn[1].clone()
    nist.separate_overlapping_parts(positions, rotations, bounds, meshes, torch.Generator().manual_seed(1701))

    assert not overlapping_worlds(positions, rotations, boxes).any()
    clean = torch.from_numpy(~overlapped)
    assert torch.equal(positions[clean], drawn[0][clean])
    assert torch.equal(rotations[clean], drawn[1][clean])
    # The redraws keep the drop region and the origins' spacing, and take nothing from the shared stream.
    assert ((positions >= bounds[:, 0]) & (positions <= bounds[:, 1])).all()
    gaps = (positions[:, :, None, :2] - positions[:, None, :, :2]).norm(dim=-1) + torch.eye(10)
    assert (gaps >= nist.PART_SPACING).all()
    assert torch.equal(torch.get_rng_state(), stream)
