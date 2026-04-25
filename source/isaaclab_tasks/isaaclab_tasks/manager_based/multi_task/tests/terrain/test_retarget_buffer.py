# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RetargetBuffer: allocation, views, reset."""

import pytest
import torch
import warp as wp

from isaaclab_tasks.manager_based.multi_task.terrain.mdp.retarget.buffer import RetargetBuffer


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cpu"


class TestRetargetBufferAllocation:
    def test_views_shapes(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        assert buf.contact_targets_t.shape == (400, 3)
        assert buf.joint_q_init_t.shape == (100, 19)
        assert buf.joint_q_result_t.shape == (100, 19)
        assert buf.body_q_t.shape == (1700, 7)
        assert buf.base_target_pos_t.shape == (100, 3)
        assert buf.base_target_rot_t.shape == (100, 4)

    def test_single_allocation(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        assert buf._data.is_contiguous()
        # All float views should point into _data
        assert buf.contact_targets_t.data_ptr() == buf._data.data_ptr()

    def test_device(self):
        buf = RetargetBuffer(10, 19, 17, 4, device=DEVICE)
        assert buf._data.device == torch.device(DEVICE)

    def test_initial_counters(self):
        buf = RetargetBuffer(50, 19, 17, 4, device=DEVICE)
        assert buf.num_written == 0
        assert buf.num_geometry_valid == 0
        assert buf.num_ik_valid == 0
        assert buf.num_selected == 0


class TestRetargetBufferMasks:
    def test_set_geometry_valid(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        buf._geom_valid[0] = True
        buf._geom_valid[5] = True
        buf._geom_valid[10] = True
        assert buf._geom_valid.sum() == 3

    def test_reset_clears_masks(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        buf._geom_valid[0] = True
        buf.num_written = 10
        buf.num_geometry_valid = 1
        buf.reset()
        assert buf._geom_valid.sum() == 0
        assert buf.num_written == 0
        assert buf.num_geometry_valid == 0

    def test_cumulative_masks(self):
        buf = RetargetBuffer(10, 19, 17, 4, device=DEVICE)
        buf._geom_valid[:3] = True
        buf._ik_valid[0] = True
        buf._ik_valid[2] = True
        combined = buf._geom_valid & buf._ik_valid
        assert combined.sum() == 2
        assert combined[0] and combined[2] and not combined[1]


class TestRetargetBufferZeroCopy:
    def test_torch_view_writes_through(self):
        buf = RetargetBuffer(10, 19, 17, 4, device=DEVICE)
        buf.joint_q_result_t[3, 5] = 42.0
        assert buf.joint_q_result_t[3, 5] == pytest.approx(42.0)
        # Verify it's in the backing tensor
        s = buf._o_jr + 3 * 19 + 5
        assert buf._data[s] == pytest.approx(42.0)

    def test_memory_estimate(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        assert buf.memory_bytes > 0
        bigger = RetargetBuffer(200, 19, 17, 4, device=DEVICE)
        assert bigger.memory_bytes > buf.memory_bytes


class TestRetargetBufferReuse:
    def test_reset_preserves_capacity(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        buf.num_written = 50
        buf.num_geometry_valid = 30
        buf.reset()
        assert buf.max_candidates == 100
        assert buf.contact_targets_t.shape == (400, 3)
        assert buf.num_written == 0
