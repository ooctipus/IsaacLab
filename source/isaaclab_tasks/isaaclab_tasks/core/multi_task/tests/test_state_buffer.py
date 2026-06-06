# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab_tasks.core.multi_task.curriculum.state_buffer import StateBuffer


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

    def test_oversample_compact_thins_to_target_and_zeros_tail(self):
        """When oversample fills, the buffer compacts down to target_size."""
        torch.manual_seed(0)
        target = 4
        capacity = 12
        buf = StateBuffer(capacity, 3, torch.device("cpu"), target_size=target)
        # Fill with linearly-spaced points along x so xyz is well-separated.
        xs = torch.linspace(0.0, 11.0, capacity).unsqueeze(-1)
        states = torch.cat([xs, torch.zeros(capacity, 2)], dim=-1)

        captured: list[torch.Tensor] = []
        buf.register_compact_callback(lambda idx: captured.append(idx.clone()))

        buf.add(states)
        # After fill+compact: only ``target`` slots are populated.
        assert len(buf) == target
        assert buf.is_full is False  # no longer at capacity post-compact
        # Tail beyond target must be zeroed so stale survivors don't leak.
        torch.testing.assert_close(buf.data[target:], torch.zeros(capacity - target, 3))
        # Surviving rows are a (sorted) subset of the input rows.
        survivors = buf.data[:target, 0]
        assert (survivors >= 0.0).all() and (survivors <= 11.0).all()
        assert torch.unique(survivors).numel() == target
        # Callback fired exactly once with target_size sorted indices.
        assert len(captured) == 1
        assert captured[0].shape == (target,)
        assert (captured[0][1:] >= captured[0][:-1]).all()

    def test_oversample_callback_indices_align_with_buffer_post_compact(self):
        """The callback's indices identify which pre-compact slots survive."""
        torch.manual_seed(1)
        target = 3
        capacity = 9
        buf = StateBuffer(capacity, 3, torch.device("cpu"), target_size=target)
        states = torch.cat([torch.arange(capacity).float().unsqueeze(-1), torch.zeros(capacity, 2)], dim=-1)

        captured: list[torch.Tensor] = []
        buf.register_compact_callback(lambda idx: captured.append(idx.clone()))
        buf.add(states)

        # Post-compact slot ``i`` holds the row that was at pre-compact index
        # ``captured[0][i]`` -- the buffer permutes by exactly that map.
        keep = captured[0]
        torch.testing.assert_close(buf.data[:target], states[keep])

    def test_default_features_use_first_three_columns(self):
        """``fps_features=None`` defaults to xyz (state[:, :3])."""
        torch.manual_seed(2)
        target = 2
        capacity = 6
        # state_dim=5 with first 3 columns being xyz, last 2 are noise.
        buf = StateBuffer(capacity, 5, torch.device("cpu"), target_size=target)
        xyz = torch.linspace(0.0, 5.0, capacity).unsqueeze(-1)
        states = torch.cat([xyz, torch.zeros(capacity, 2), torch.randn(capacity, 2) * 100.0], dim=-1)
        buf.add(states)
        # Buffer should still produce ``target`` survivors regardless of
        # the noisy non-xyz columns -- if those leaked into the FPS metric
        # the spatial coverage would be unstable.
        assert len(buf) == target

    def test_custom_callable_features(self):
        """``fps_features`` callable maps states → custom feature space."""
        torch.manual_seed(3)
        target = 2
        capacity = 6

        # Use the *last* column as the feature -- exercises the callable path.
        def last_col(states: torch.Tensor) -> torch.Tensor:
            return states[:, -1:].clone()

        buf = StateBuffer(capacity, 4, torch.device("cpu"), target_size=target, fps_features=last_col)
        states = torch.zeros(capacity, 4)
        states[:, -1] = torch.linspace(0.0, 5.0, capacity)
        buf.add(states)
        assert len(buf) == target

    def test_oversample_ratio_one_preserves_legacy_ring_behavior(self):
        """``target_size == max_size`` keeps the ring buffer's FIFO semantics."""
        buf = StateBuffer(4, 2, torch.device("cpu"))
        first = torch.arange(8).reshape(4, 2).float()
        # Fills exactly: no compaction trigger.
        start, n = buf.add(first)
        assert (start, n) == (0, 4)
        assert len(buf) == 4
        # Next batch wraps to slot 0 (the legacy ring path), not compaction.
        wrap = torch.tensor([[99.0, 100.0]])
        start, n = buf.add(wrap)
        assert start == 0 and n == 1
        torch.testing.assert_close(buf.data[0], wrap[0])

    def test_explicit_compact_is_noop_below_target(self):
        """compact() is idempotent when the buffer already fits target_size."""
        buf = StateBuffer(8, 3, torch.device("cpu"), target_size=4)
        states = torch.zeros(3, 3)
        states[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        buf.add(states)  # _size=3 <= target=4, no auto-trigger
        captured: list[torch.Tensor] = []
        buf.register_compact_callback(lambda idx: captured.append(idx.clone()))

        survivors = buf.compact()
        # No-op: survivors == arange(3), buffer unchanged, no callback fired.
        torch.testing.assert_close(survivors, torch.arange(3, dtype=torch.int64))
        assert len(buf) == 3
        torch.testing.assert_close(buf.data[:3], states)
        assert len(captured) == 0

    def test_from_states_view_wraps_without_copy(self):
        """``from_states`` aliases the input tensor; constructor allocates nothing."""
        torch.manual_seed(5)
        states = torch.zeros(8, 3)
        states[:, 0] = torch.linspace(0.0, 7.0, 8)
        sb = StateBuffer.from_states(states, target_size=4)
        # ``data`` is the same storage as ``states`` -- not a copy.
        assert sb.data.data_ptr() == states.data_ptr()
        assert sb.max_size == 8
        assert sb.target_size == 4
        assert len(sb) == 8

    def test_from_states_compact_does_not_mutate_caller_slab(self):
        """View-wrap compact allocates its own output; caller's slab is untouched."""
        torch.manual_seed(6)
        n, target = 8, 3
        states = torch.zeros(n, 3)
        states[:, 0] = torch.linspace(0.0, 7.0, n)
        original = states.clone()

        sb = StateBuffer.from_states(states, target_size=target)
        keep = sb.compact()

        # Caller's slab is unchanged.
        torch.testing.assert_close(states, original)
        # Buffer now owns a freshly-allocated [target, 3] tensor of survivors.
        assert sb.data.data_ptr() != states.data_ptr()
        assert sb.data.shape == (target, 3)
        assert len(sb) == target
        # Survivors are exactly ``states[keep]``.
        torch.testing.assert_close(sb.data, states[keep])

    def test_from_states_compact_idempotent_below_target(self):
        """compact() on a from_states with size <= target is a no-op view-wise."""
        states = torch.zeros(2, 3)
        sb = StateBuffer.from_states(states, target_size=4)
        keep = sb.compact()
        torch.testing.assert_close(keep, torch.arange(2, dtype=torch.int64))
        # No mutation of caller; data still aliases.
        assert sb.data.data_ptr() == states.data_ptr()

    def test_explicit_compact_thins_without_filling_capacity(self):
        """compact() can be triggered manually before the buffer overflows.

        This is the locomotion-pipeline shape: dump N candidates (where
        ``target_size <= N <= max_size``), compact() once, read survivors
        from buffer.data[:target_size]. Auto-trigger never fires because
        the buffer never hits max_size.
        """
        torch.manual_seed(4)
        target = 3
        capacity = 8
        buf = StateBuffer(capacity, 3, torch.device("cpu"), target_size=target)
        captured: list[torch.Tensor] = []
        buf.register_compact_callback(lambda idx: captured.append(idx.clone()))

        n_add = 6  # > target, but < capacity → no auto-compact
        states = torch.zeros(n_add, 3)
        states[:, 0] = torch.linspace(0.0, 5.0, n_add)
        buf.add(states)
        assert len(buf) == n_add  # auto-compact did not fire

        survivors = buf.compact()
        assert survivors.shape == (target,)
        assert (survivors[1:] >= survivors[:-1]).all()
        assert len(buf) == target
        torch.testing.assert_close(buf.data[:target], states[survivors])
        # Callback fires once with the same indices the caller received.
        assert len(captured) == 1
        torch.testing.assert_close(captured[0], survivors)
