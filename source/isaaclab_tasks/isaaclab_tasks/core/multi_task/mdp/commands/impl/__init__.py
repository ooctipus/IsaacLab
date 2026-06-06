# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private command backend implementations, grouped by dispatch strategy.

Public entry points stay in ``multi_task_command.py`` and
``multi_task_command_warp.py``. Each backend folder owns its full data-
management pipeline (bindings → read → execute → rotate → compose); the
only shared layer is the ``multi_task_command`` agreement.

* ``mega_kernel``: dense ``(env, slot)`` baseline. Supports
  ``slot_order="schedule"`` mode, publicly selected via
  ``dispatch_backend="schedule_ordered_mega"``.
* ``packed_scatter``: fused-pipeline-sorted flat queue with legacy scatter.
* ``primitive_queue_local``: primitive queues plus local rows for reward composition.
* ``primitive_graph_local``: primitive graph queues with shared producer nodes.
"""

from .backend import CommandBackend, build_command_backend

__all__ = ["CommandBackend", "build_command_backend"]
