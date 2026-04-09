# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab_tasks.manager_based.manipulation.factory_v1.mdp.util.sampling import (
    beta_sampling_probs,
    tagged_report,
)


class TestBetaSamplingProbs:
    def test_output_sums_to_one(self):
        rates = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        probs = beta_sampling_probs(rates)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_peaks_at_target(self):
        rates = torch.linspace(0, 1, 101)
        probs = beta_sampling_probs(rates, target=0.5, kappa=4.0)
        peak_idx = probs.argmax().item()
        assert 45 <= peak_idx <= 55, f"Expected peak near index 50, got {peak_idx}"

    def test_peaks_shift_with_target(self):
        rates = torch.linspace(0, 1, 101)
        probs_low = beta_sampling_probs(rates, target=0.2, kappa=4.0)
        probs_high = beta_sampling_probs(rates, target=0.8, kappa=4.0)
        assert probs_low.argmax().item() < probs_high.argmax().item()

    def test_uniform_at_zero_kappa(self):
        rates = torch.tensor([0.1, 0.5, 0.9])
        probs = beta_sampling_probs(rates, kappa=0.0)
        assert torch.allclose(probs, torch.ones(3) / 3, atol=1e-4)

    def test_higher_kappa_concentrates(self):
        rates = torch.linspace(0, 1, 101)
        probs_low_k = beta_sampling_probs(rates, target=0.5, kappa=1.0)
        probs_high_k = beta_sampling_probs(rates, target=0.5, kappa=10.0)
        assert probs_high_k.max() > probs_low_k.max()

    def test_higher_temperature_flattens(self):
        rates = torch.linspace(0, 1, 101)
        probs_cold = beta_sampling_probs(rates, target=0.5, kappa=4.0, temperature=1.0)
        probs_hot = beta_sampling_probs(rates, target=0.5, kappa=4.0, temperature=10.0)
        assert probs_hot.max() < probs_cold.max()


class TestTaggedReport:
    def test_sum_reduction(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tags = torch.tensor([0, 1, 0, 1])
        result = tagged_report(values, tags, ["A", "B"], reduction="sum")
        assert result["A"] == pytest.approx(4.0)
        assert result["B"] == pytest.approx(6.0)

    def test_mean_reduction(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tags = torch.tensor([0, 1, 0, 1])
        result = tagged_report(values, tags, ["A", "B"], reduction="mean")
        assert result["A"] == pytest.approx(2.0)
        assert result["B"] == pytest.approx(3.0)

    def test_untagged_slots_excluded(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        tags = torch.tensor([0, -1, 0])
        result = tagged_report(values, tags, ["A"], reduction="sum")
        assert result["A"] == pytest.approx(4.0)

    def test_missing_tag_returns_zero(self):
        values = torch.tensor([1.0, 2.0])
        tags = torch.tensor([0, 0])
        result = tagged_report(values, tags, ["A", "B"], reduction="sum")
        assert result["B"] == pytest.approx(0.0)
