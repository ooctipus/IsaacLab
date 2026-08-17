# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from ..asset_base_cfg import AssetBaseCfg

if TYPE_CHECKING:
    from .rigid_object import RigidObject


@configclass
class RigidObjectCfg(AssetBaseCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    ##
    # Initialize configurations.
    ##

    class_type: type["RigidObject"] | str = "{DIR}.rigid_object:RigidObject"

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""

    mesh_variants_enabled: bool = False
    """Compile multi-asset collision meshes for reset-time selection.

    Newton MJWarp requires one rigid body and the same number of mesh colliders in every variant.
    Every variant must initially populate at least one environment so the model owns all mesh and
    SDF resources. Selection updates collision geometry, mass, and inertia. The Newton OpenGL
    visualizer follows the selected geometry; USD/Fabric visuals do not. The MuJoCo contact path
    rejects planar variants. PhysX ignores this option; Newton solvers other than MJWarp reject it.
    Defaults to ``False``.
    """

    mesh_variant_inertia_diagonal_offset: float = 0.0
    """Diagonal inertia added to every compiled mesh variant [kg·m²].

    The offset is packed into the variant bank, so selecting a mesh does not require another
    runtime update. Defaults to ``0.0``.
    """
