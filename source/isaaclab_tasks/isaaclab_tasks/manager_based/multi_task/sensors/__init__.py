# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task-local sensor extensions.

This package contains sensor implementations specialized for the multi-task locomotion
setup. They subclass only :class:`isaaclab.sensors.SensorBase` (the env-side lifecycle
contract) and reuse no upstream sensor implementation, so a future rebase onto a new
IsaacLab version requires no merge work in shared sensor code — delete this package or
re-point the configs back to the upstream classes.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
