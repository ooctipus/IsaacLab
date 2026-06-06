# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot preset registry for Factory tasks.

Importing this package eagerly imports every per-robot module (declared in
:file:`__init__.pyi`) so each robot's class-attribute assignments on
:class:`PresetCfg` subclasses execute before the preset resolver runs.

Add a new robot by dropping a ``<robot>.py`` module next to this file and
appending ``from .<robot> import *`` to the ``.pyi`` stub.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
