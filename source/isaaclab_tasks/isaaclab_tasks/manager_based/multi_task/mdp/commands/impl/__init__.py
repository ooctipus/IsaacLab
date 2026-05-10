# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private command backend implementations, grouped by dispatch strategy.

Public entry points stay in ``multi_task_command.py`` and
``multi_task_command_warp.py``. Strategy folders are the top-level boundary:

* ``mega_kernel``: current production backend. One branchy dispatch kernel.
* ``schedule_ordered_mega``: one branchy dispatch kernel over schedule-ordered slot ranks.
* ``packed_scatter``: fused-pipeline-sorted flat queue with legacy scatter.
* ``primitive_queue_local``: primitive queues plus local rows for reward composition.
* ``primitive_graph_local``: primitive graph queues with shared producer nodes.
"""

from .backend import CommandBackend, build_command_backend, build_command_output_store

__all__ = ["CommandBackend", "build_command_backend", "build_command_output_store"]
