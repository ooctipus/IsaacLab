# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest


@pytest.fixture(autouse=True)
def isolate_simulation_context():
    """Give every simulation test exclusive ownership of its context."""
    from isaaclab.sim import SimulationContext

    SimulationContext.clear_instance()
    yield
    SimulationContext.clear_instance()
