# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the ray-cast sensor."""

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from .multi_mesh_ray_caster import MultiMeshRayCaster


@configclass
class MultiMeshRayCasterCfg(RayCasterCfg):
    """Configuration for the multi-mesh ray-cast sensor."""

    @configclass
    class RaycastTargetCfg:
        """Configuration for different ray-cast targets."""

        prim_expr: str = MISSING
        """The regex to specify the target prim to ray cast against."""

        track_mesh_transforms: bool = True
        """Whether the mesh transformations should be tracked. Defaults to True.

        .. note::
            Not tracking the mesh transformations is recommended when the meshes are static to increase performance.
        """

    class_type: type["MultiMeshRayCaster"] | str = "{DIR}.multi_mesh_ray_caster:MultiMeshRayCaster"

    mesh_prim_paths: list[str | RaycastTargetCfg] = MISSING
    """The list of mesh primitive paths to ray cast against.

    If an entry is a string, it is internally converted to :class:`RaycastTargetCfg` with
    :attr:`~RaycastTargetCfg.track_mesh_transforms` disabled.
    """

    update_mesh_ids: bool = False
    """Whether to update the mesh ids of the ray hits in the :attr:`data` container."""
