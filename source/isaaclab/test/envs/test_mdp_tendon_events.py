# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for fixed-tendon randomization events."""

from types import SimpleNamespace

from isaaclab.envs.mdp.events import randomize_fixed_tendon_parameters


def test_randomize_fixed_tendon_parameters_skips_assets_without_tendons():
    """An all-tendon event should be a no-op when an articulation has none."""
    event = randomize_fixed_tendon_parameters.__new__(randomize_fixed_tendon_parameters)
    event.asset = SimpleNamespace(num_fixed_tendons=0)

    event(
        env=None,
        env_ids=None,
        asset_cfg=None,
        stiffness_distribution_params=(0.75, 1.5),
        damping_distribution_params=(0.3, 3.0),
        operation="scale",
        distribution="log_uniform",
    )
