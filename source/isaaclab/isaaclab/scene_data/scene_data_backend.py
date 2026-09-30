# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend interface and data formats for the scene data provider.

These types live in :mod:`isaaclab.scene_data` rather than
:mod:`isaaclab.scene` so that physics backends (``isaaclab_physx``,
``isaaclab_newton``) can subclass :class:`SceneDataBackend` without pulling
:mod:`isaaclab.scene` into the ``AppLauncher`` pre-launch import chain.
``AppLauncher._create_app`` pops ``*lab*`` modules from ``sys.modules``
during Kit init and any submodule imported during that window ends up
orphaned from its parent's ``__dict__`` after restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

# Under Sphinx ``autodoc_mock_imports``, ``wp.struct`` is a ``_MockObject``
# that replaces the decorated class with another mock, hiding its docstring
# and fields from autodoc. Fall back to an identity decorator when warp is
# mocked so the documentation builds from the source classes directly.
if getattr(wp, "__sphinx_mock__", False):

    def wp_struct(cls):
        return cls
else:
    wp_struct = wp.struct


class SceneDataFormat:
    """Warp struct variants describing the transform layouts that a
    :class:`SceneDataBackend` may publish to consumers.
    """

    @wp_struct
    class Vec3_Quat:
        """Separate position and quaternion arrays."""

        positions: wp.array(dtype=wp.vec3f) = None
        """Per-transform positions [m]."""

        orientations: wp.array(dtype=wp.quatf) = None
        """Per-transform orientations as quaternions."""

    @wp_struct
    class Vec3_Matrix33:
        """Separate position and rotation-matrix arrays."""

        positions: wp.array(dtype=wp.vec3f) = None
        """Per-transform positions [m]."""

        orientations: wp.array(dtype=wp.mat33f) = None
        """Per-transform orientations as 3x3 rotation matrices."""

    @wp_struct
    class Transform:
        """Packed warp transforms (position + quaternion)."""

        transforms: wp.array(dtype=wp.transformf) = None
        """Per-transform packed position + orientation transforms [m, -]."""

    @wp_struct
    class IndexedTransform:
        """Native packed transforms with canonical clone-plan gather indices."""

        transforms: wp.array(dtype=wp.transformf) = None
        """Native packed position + orientation transforms [m, -]."""

        source_indices: wp.array(dtype=wp.int32) = None
        """Canonical clone-plan body index to native transform index."""

    @wp_struct
    class TransposedMatrix44d:
        """Transposed double-precision 4x4 homogeneous transform matrices."""

        matrices: wp.array(dtype=wp.mat44d) = None
        """Per-transform transposed double-precision matrices [m]."""

    @dataclass(slots=True)
    class HostTransposedMatrix44d:
        """Host-resident transposed double-precision 4x4 transform matrices."""

        matrices: np.ndarray | None = None
        """Per-transform transposed double-precision matrices [m], shape ``[N, 4, 4]``."""

    @dataclass(slots=True)
    class FabricMatrix44:
        """Packed 4x4 transform matrices held by USD Fabric, indexed by prim."""

        matrices: Any = None
        """Per-prim ``omni:fabric:worldMatrix``, transposed and double precision [m]."""

        local_matrices: Any = None
        """Per-prim ``omni:fabric:localMatrix``, transposed and double precision [m]."""

        source_indices: Any = None
        """Per-prim index of the transform that drives it."""

    @dataclass(slots=True)
    class FabricMeshPoints:
        """Deformable visual meshes held by USD Fabric, one ragged point array per prim."""

        points: Any = None
        """Per-prim local-frame mesh points [m]."""

        world_matrices: Any = None
        """Per-prim ``omni:fabric:worldMatrix``, read to reach each prim's local frame [m]."""

        binding_slots: Any = None
        """Fabric selection slot for each canonical clone-plan point binding."""

    @wp_struct
    class Points:
        """Flat world-space nodal or particle positions."""

        points: wp.array(dtype=wp.vec3f) = None
        """World-space positions [m], shape [point_count]."""

    @dataclass(slots=True)
    class BodyPoints:
        """Native padded deformable-body point pointers and their plan bindings."""

        points: tuple[Any, ...] = ()
        """Padded world-space body positions [m], each shaped ``[body_count, max_point_count]``."""

        binding_ids: tuple[Any, ...] = ()
        """Clone-plan point-binding index for each body row."""

    @dataclass(slots=True)
    class CablePoints:
        """Native Newton state/model pointers and plan-owned cable topology."""

        body_q: Any = None
        """World-space body transforms [m, -]."""

        shape_body: Any = None
        """Body index for each Newton shape."""

        shape_transform: Any = None
        """Shape transforms relative to their bodies [m, -]."""

        shape_scale: Any = None
        """Shape half-extents [m]."""

        shape_ids: Any = None
        """Ordered Newton segment-shape indices."""

        shape_offsets: Any = None
        """First segment-shape index for each cable."""

        segment_counts: Any = None
        """Segment count for each cable."""

        binding_ids: Any = None
        """Clone-plan point-binding index for each cable."""

    @dataclass(slots=True)
    class HostPoints:
        """Host-resident flat world-space nodal or particle positions."""

        points: np.ndarray | None = None
        """World-space positions [m], shape ``[point_count, 3]``."""

    @dataclass(slots=True)
    class HostMeshPoints:
        """Host-resident plan-mapped drawable mesh points."""

        points: np.ndarray | None = None
        """World-space visual positions [m], shape ``[visual_point_count, 3]``."""


@dataclass(slots=True)
class SceneDataPublication:
    """One native-format pointer and its dirty latch."""

    data: Any
    dirty: bool


class SceneDataBackend:
    def _materialize(self, publication: SceneDataPublication) -> None:
        pass

    @property
    def transform_publication(self) -> SceneDataPublication:
        """Return the native transform pointer and dirty latch."""
        raise NotImplementedError

    @property
    def point_publications(self) -> dict[str, SceneDataPublication]:
        """Return named point pointers and dirty latches."""
        return {}
