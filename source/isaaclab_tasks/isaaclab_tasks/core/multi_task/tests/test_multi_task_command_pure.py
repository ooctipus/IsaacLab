# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for pure helpers exported by :mod:`multi_task_command`.

Under the ragged layout (Stage 2.0), the only remaining module-level pure helper is
:func:`pad_index_rows`. Per-env type masks that used to be materialized at resample
(``build_per_env_type_masks``) were removed: the command term now looks up
``spec.is_instant[env_subtask_ids]`` each step instead, which both avoids the mask
scatter and always reflects the current spec without a staleness window.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.core.multi_task.mdp.commands.spec import pad_index_rows


def test_pad_index_rows_uniform_rows():
    """Uniform-length rows produce a rectangular table with full-True valid mask."""
    rows = [[0, 1], [2, 3], [4, 5]]
    idx, valid = pad_index_rows(rows, device="cpu")
    assert idx.shape == (3, 2)
    assert valid.shape == (3, 2)
    assert torch.equal(idx, torch.tensor([[0, 1], [2, 3], [4, 5]]))
    assert torch.equal(valid, torch.ones(3, 2, dtype=torch.bool))


def test_pad_index_rows_ragged_rows():
    """Short rows pad with -1 and valid=False."""
    rows = [[0, 1, 2], [3], [4, 5]]
    idx, valid = pad_index_rows(rows, device="cpu")
    assert idx.shape == (3, 3)
    assert torch.equal(idx[0], torch.tensor([0, 1, 2]))
    assert torch.equal(valid[0], torch.tensor([True, True, True]))
    assert idx[1, 0].item() == 3
    assert idx[1, 1].item() == -1 and idx[1, 2].item() == -1
    assert torch.equal(valid[1], torch.tensor([True, False, False]))
    assert torch.equal(valid[2], torch.tensor([True, True, False]))


def test_pad_index_rows_empty_rows():
    """All-empty input returns a 1-column table with valid=False throughout."""
    rows: list[list[int]] = [[], [], []]
    idx, valid = pad_index_rows(rows, device="cpu")
    assert idx.shape == (3, 1)
    assert valid.shape == (3, 1)
    assert (idx == -1).all()
    assert not valid.any()


def test_pad_index_rows_mixed_with_empty():
    """An empty row padded alongside populated rows gets an all-False valid entry."""
    rows = [[1, 2], [], [3]]
    idx, valid = pad_index_rows(rows, device="cpu")
    assert idx.shape == (3, 2)
    assert idx[1, 0].item() == -1
    assert not valid[1].any()
