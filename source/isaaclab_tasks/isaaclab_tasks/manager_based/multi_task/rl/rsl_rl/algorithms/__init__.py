# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-level rsl_rl algorithm extensions.

This package keeps PPO subclasses that need task-specific bindings (e.g.
:class:`ValueShiftPPO`) at the task level. Names are re-exported here so
:func:`rsl_rl.utils.resolve_callable` can resolve the fully-qualified module
path declared on a runner cfg's ``algorithm.class_name``.
"""

from .value_shift_ppo import ValueShiftPPO

__all__ = ["ValueShiftPPO"]
