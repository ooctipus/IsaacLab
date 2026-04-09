# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab_tasks.manager_based.manipulation.factory_v1.mdp.util.state_buffer import StateBuffer


class TestStateBuffer:
    def test_empty_buffer(self):
        buf = StateBuffer(10, 3, torch.device("cpu"))
        assert len(buf) == 0
        assert not buf.is_full

    def test_add_and_len(self):
        buf = StateBuffer(10, 3, torch.device("cpu"))
        states = torch.randn(4, 3)
        start, n = buf.add(states)
        assert start == 0
        assert n == 4
        assert len(buf) == 4

    def test_is_full(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        buf.add(torch.randn(5, 2))
        assert buf.is_full
        assert len(buf) == 5

    def test_add_caps_at_boundary(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        buf.add(torch.randn(3, 2))
        start, n = buf.add(torch.randn(4, 2))
        assert start == 3
        assert n == 2, "Should cap at buffer boundary"
        assert len(buf) == 5

    def test_ring_wrap(self):
        buf = StateBuffer(4, 1, torch.device("cpu"))
        buf.add(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        assert buf.is_full
        start, n = buf.add(torch.tensor([[5.0], [6.0]]))
        assert start == 0
        assert n == 2
        assert buf.data[0].item() == pytest.approx(5.0)
        assert buf.data[1].item() == pytest.approx(6.0)

    def test_sample(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        data = torch.arange(10).reshape(5, 2).float()
        buf.add(data)
        indices = torch.tensor([0, 2, 4])
        sampled = buf.sample(indices)
        assert torch.equal(sampled, data[indices])

    def test_set_tag_names(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        buf.set_tag_names(["a", "b"])
        assert buf.tag_names == ["a", "b"]

    def test_set_tags(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        buf.set_tags(torch.tensor([0, 2, 4]), torch.tensor([1, 0, 1]))
        assert buf.tags[0].item() == 1
        assert buf.tags[1].item() == -1
        assert buf.tags[2].item() == 0
        assert buf.tags[4].item() == 1

    def test_success_rates_initially_none(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        assert buf.success_rates is None

    def test_max_size(self):
        buf = StateBuffer(42, 2, torch.device("cpu"))
        assert buf.max_size == 42
