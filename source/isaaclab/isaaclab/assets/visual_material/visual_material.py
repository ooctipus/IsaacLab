# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_visual_material import BaseVisualMaterial

if TYPE_CHECKING:
    from isaaclab_newton.assets.visual_material import VisualMaterial as NewtonVisualMaterial


class VisualMaterial(FactoryBase, BaseVisualMaterial):
    """Factory for creating visual-material instances for the active physics backend.

    Kit/PhysX backends author shader inputs on the USD stage (RTX renders from the stage);
    the Newton backend writes the Newton model and renderer texture pools directly, since no
    Newton consumer reads the stage after import.
    """

    def __new__(cls, *args, **kwargs) -> BaseVisualMaterial | NewtonVisualMaterial:
        """Create a visual material of the backend-specific implementation class."""
        return super().__new__(cls, *args, **kwargs)


# The base implementation authors shader inputs on the USD stage, which is exactly what
# Kit-based RTX rendering consumes — the USD-stage backends ARE the base class. Registering
# them here means no backend package carries a passthrough module, and users only ever
# import from isaaclab.assets.
VisualMaterial.register("physx", BaseVisualMaterial)
VisualMaterial.register("ovphysx", BaseVisualMaterial)
