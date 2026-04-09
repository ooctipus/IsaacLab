# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab_tasks.manager_based.manipulation.factory_v1.mdp.util.success_monitor import SuccessMonitor
from isaaclab_tasks.manager_based.manipulation.factory_v1.mdp.util.success_monitor_cfg import SuccessMonitorCfg


def _make_monitor(num_slots: int, history_len: int):
    """Helper to create a monitor with a fresh external rate tensor."""
    rate = torch.zeros(num_slots)
    cfg = SuccessMonitorCfg(num_monitored_data=num_slots, monitored_history_len=history_len, device="cpu")
    mon = cfg.class_type(cfg, rate)
    return mon, rate


class TestSuccessMonitor:

    def test_single_slot_single_update(self):
        """Slot 0 receives one success; others stay at 0."""
        mon, rate = _make_monitor(3, 5)
        mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
        assert rate[0].item() == pytest.approx(1.0)
        assert rate[1].item() == pytest.approx(0.0)
        assert rate[2].item() == pytest.approx(0.0)

    def test_single_slot_mixed_outcomes(self):
        """Three sequential updates (1, 0, 1) → rate = 2/3."""
        mon, rate = _make_monitor(3, 5)
        mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
        mon.success_update(torch.tensor([0]), torch.tensor([0.0]))
        mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
        assert rate[0].item() == pytest.approx(2.0 / 3.0)

    def test_circular_wrap_overwrites_oldest(self):
        """History=3, fill with 3 successes then 2 failures → rate = 1/3."""
        mon, rate = _make_monitor(1, 3)
        # Fill: [1, 1, 1]
        for _ in range(3):
            mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
        assert rate[0].item() == pytest.approx(1.0)
        # Overwrite 2 oldest: buffer becomes [0, 0, 1] in some order
        mon.success_update(torch.tensor([0]), torch.tensor([0.0]))
        mon.success_update(torch.tensor([0]), torch.tensor([0.0]))
        assert rate[0].item() == pytest.approx(1.0 / 3.0)

    def test_full_overwrite(self):
        """Fill with successes, then completely overwrite with failures."""
        mon, rate = _make_monitor(1, 3)
        for _ in range(3):
            mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
        assert rate[0].item() == pytest.approx(1.0)
        for _ in range(3):
            mon.success_update(torch.tensor([0]), torch.tensor([0.0]))
        assert rate[0].item() == pytest.approx(0.0)

    def test_independent_slots(self):
        """Slots updated independently produce independent rates."""
        mon, rate = _make_monitor(3, 4)
        for _ in range(4):
            mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
            mon.success_update(torch.tensor([1]), torch.tensor([0.0]))
        mon.success_update(torch.tensor([2]), torch.tensor([1.0]))
        mon.success_update(torch.tensor([2]), torch.tensor([0.0]))
        assert rate[0].item() == pytest.approx(1.0)
        assert rate[1].item() == pytest.approx(0.0)
        assert rate[2].item() == pytest.approx(0.5)

    def test_batch_update_same_slot(self):
        """Multiple envs map to the same slot in one call."""
        mon, rate = _make_monitor(3, 5)
        mon.success_update(torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 1.0]))
        assert rate[0].item() == pytest.approx(2.0 / 3.0)

    def test_batch_update_clamped_to_history_len(self):
        """When batch has more entries than history_len, only the last ones count."""
        mon, rate = _make_monitor(1, 2)
        mon.success_update(
            torch.tensor([0, 0, 0, 0, 0]),
            torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]),
        )
        # history_len=2, so only last 2 values (0, 0) are kept
        assert rate[0].item() == pytest.approx(0.0)

    def test_external_tensor_updated_in_place(self):
        """The monitor writes into the caller-provided tensor, not a copy."""
        rate = torch.zeros(2)
        cfg = SuccessMonitorCfg(num_monitored_data=2, monitored_history_len=5, device="cpu")
        mon = cfg.class_type(cfg, rate)
        mon.success_update(torch.tensor([0]), torch.tensor([1.0]))
        # rate should be the SAME object
        assert rate is mon.success_rate
        assert rate[0].item() == pytest.approx(1.0)

    def test_rate_zero_for_untouched_slots(self):
        """Slots that never receive updates remain at 0."""
        mon, rate = _make_monitor(5, 10)
        mon.success_update(torch.tensor([3]), torch.tensor([1.0]))
        assert rate[0].item() == pytest.approx(0.0)
        assert rate[1].item() == pytest.approx(0.0)
        assert rate[2].item() == pytest.approx(0.0)
        assert rate[3].item() == pytest.approx(1.0)
        assert rate[4].item() == pytest.approx(0.0)
