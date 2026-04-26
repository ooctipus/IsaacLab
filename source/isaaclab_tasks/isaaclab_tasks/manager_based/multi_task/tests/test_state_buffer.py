# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab_tasks.manager_based.multi_task.mdp.util.state_buffer import StateBuffer


class TestStateBuffer:
    def test_add_caps_at_boundary_and_wraps_next_batch(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        first = torch.arange(6).reshape(3, 2).float()
        second = torch.arange(8).reshape(4, 2).float() + 10.0
        third = torch.tensor([[99.0, 100.0]])

        start, n = buf.add(first)
        assert (start, n) == (0, 3)
        assert len(buf) == 3

        start, n = buf.add(second)
        assert start == 3
        assert n == 2, "Should cap at buffer boundary"
        torch.testing.assert_close(buf.data[:3], first)
        torch.testing.assert_close(buf.data[3:5], second[:2])
        assert len(buf) == 5
        assert buf.is_full

        start, n = buf.add(third)
        assert start == 0
        assert n == 1
        torch.testing.assert_close(buf.data[0], third[0])

    def test_sample(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        data = torch.arange(10).reshape(5, 2).float()
        buf.add(data)
        indices = torch.tensor([0, 2, 4])
        sampled = buf.sample(indices)
        assert torch.equal(sampled, data[indices])

    def test_tags_follow_written_prefix_only(self):
        buf = StateBuffer(5, 2, torch.device("cpu"))
        buf.set_tag_names(["a", "b"])
        assert buf.tag_names == ["a", "b"]

        states = torch.arange(8).reshape(4, 2).float()
        start, n = buf.add_with_tags(states, torch.tensor([2, 1, 0, 1]))
        assert (start, n) == (0, 4)
        torch.testing.assert_close(buf.data[:4], states)
        torch.testing.assert_close(buf.tags[:4], torch.tensor([2, 1, 0, 1]))
        assert buf.tags[4].item() == -1
