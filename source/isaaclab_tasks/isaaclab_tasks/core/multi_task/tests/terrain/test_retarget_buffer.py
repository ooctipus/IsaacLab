# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the meaningful RetargetBuffer storage contracts."""

import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cpu"


def test_float_views_share_one_backing_tensor_and_write_through():
    """All float views are slices of ``_data`` and writes hit the backing storage."""
    buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)

    assert buf._data.is_contiguous()
    assert buf._data.device == torch.device(DEVICE)
    assert buf.contact_targets_t.shape == (400, 3)
    assert buf.joint_q_init_t.shape == (100, 19)
    assert buf.joint_q_result_t.shape == (100, 19)
    assert buf.body_q_t.shape == (1700, 7)
    assert buf.base_target_pos_t.shape == (100, 3)
    assert buf.base_target_rot_t.shape == (100, 4)

    base = buf._data.data_ptr()
    assert buf.contact_targets_t.data_ptr() == base
    assert buf.joint_q_init_t.data_ptr() > base
    assert buf.joint_q_result_t.data_ptr() > buf.joint_q_init_t.data_ptr()

    buf.joint_q_result_t[3, 5] = 42.0
    backing_offset = buf._o_jr + 3 * 19 + 5
    assert buf._data[backing_offset].item() == 42.0


def test_reset_clears_runtime_state_without_reallocating():
    """Reuse must clear masks/counters while keeping the same allocation."""
    buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
    ptr_before = buf._data.data_ptr()
    buf._geom_valid[:3] = True
    buf._ik_valid[:2] = True
    buf._is_contact[:4] = False
    buf.num_written = 10
    buf.num_geometry_valid = 3
    buf.num_ik_valid = 2

    buf.reset()

    assert buf._data.data_ptr() == ptr_before
    assert not buf._geom_valid.any()
    assert not buf._ik_valid.any()
    assert bool(buf._is_contact.all())
    assert buf.num_written == 0
    assert buf.num_geometry_valid == 0
    assert buf.num_ik_valid == 0


def test_memory_bound_scales_with_capacity_not_written_count():
    small = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
    large = RetargetBuffer(1000, 19, 17, 4, device=DEVICE)
    ratio = large.memory_bytes / small.memory_bytes
    assert 9.0 < ratio < 11.0

    mem_empty = small.memory_bytes
    small.num_written = 50
    assert small.memory_bytes == mem_empty
