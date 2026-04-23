# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for zero-copy / single-allocation behaviour of the retarget buffer."""

import pytest
import warp as wp

from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cpu"


class TestSingleAllocation:
    def test_all_views_share_backing(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        base = buf._data.data_ptr()
        assert buf.contact_targets_t.data_ptr() == base
        assert buf.joint_q_init_t.data_ptr() > base
        assert buf.joint_q_result_t.data_ptr() > buf.joint_q_init_t.data_ptr()

    def test_pointer_stable_after_reset(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        ptr_before = buf._data.data_ptr()
        buf.reset()
        assert buf._data.data_ptr() == ptr_before

    def test_write_through(self):
        buf = RetargetBuffer(10, 19, 17, 4, device=DEVICE)
        buf.joint_q_result_t[3, 5] = 99.0
        s = buf._o_jr + 3 * 19 + 5
        assert buf._data[s] == pytest.approx(99.0)


class TestMemoryBound:
    def test_proportional_to_capacity(self):
        small = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        large = RetargetBuffer(1000, 19, 17, 4, device=DEVICE)
        ratio = large.memory_bytes / small.memory_bytes
        assert 9.0 < ratio < 11.0

    def test_independent_of_data_written(self):
        buf = RetargetBuffer(100, 19, 17, 4, device=DEVICE)
        mem_empty = buf.memory_bytes
        buf.num_written = 50
        assert buf.memory_bytes == mem_empty
