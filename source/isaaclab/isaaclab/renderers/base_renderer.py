# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for renderer implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .camera_render_spec import CameraRenderSpec
from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray


@dataclass
class VisualMaterialWrite:
    """One batched visual-material attribute write, grouped by resolved shader attribute.

    Produced by :meth:`isaaclab.assets.VisualMaterial.write_channels`: all materials whose
    channel resolved to the same shader attribute name are batched into one entry, so detached
    renderers can mirror each entry with a single attribute write.
    """

    material_paths: list[str]
    """Absolute paths of the written ``UsdShade.Material`` prims."""
    shader_paths: list[str]
    """Absolute paths of the corresponding surface shader prims, aligned with :attr:`material_paths`."""
    attr_name: str
    """Namespaced shader attribute name (e.g. ``inputs:diffuse_tint``)."""
    semantic: str
    """How backends mirror the write: ``color``, ``float3``, ``scalar``, ``float2``, or ``texture``."""
    values: Any
    """CPU float tensor of shape (len(shader_paths), c), or ``list[str]`` for ``texture`` writes."""


class BaseRenderer(ABC):
    """Abstract base class for renderer implementations."""

    @classmethod
    def provides_temporal_camera_data(cls, data_type: str) -> bool:
        """Whether this renderer's ``data_type`` output carries temporal information.

        Under a physics backend without implicit damping (e.g. Newton), a camera policy
        needs a temporal cue to infer velocity. Renderers that accumulate frames over time
        (temporal AA / DLSS) supply it; pure rasterizers and non-beauty AOVs do not.

        The base default is ``False`` (assume no temporal information); renderer subclasses
        override per output type.

        Args:
            data_type: The camera output type, e.g. ``"rgb"`` or ``"depth"``.

        Returns:
            Whether the ``data_type`` output carries temporal information.
        """
        return False

    def initialize(self) -> None:
        """Post-physics one-time initialization hook. Called only once."""
        return

    def notify_visual_material_written(self, writes: Sequence[VisualMaterialWrite]) -> None:
        """Synchronize the backend after visual material channels were written to the USD stage.

        The stage is the authoritative representation: callers author the values on the material
        shaders before this hook fires. The default implementation is a no-op, which is correct for
        renderers that consume the live stage (e.g. Isaac RTX). Detached renderers override this to
        mirror the writes into their own representation, filtering by each write's semantic (e.g.
        Newton mirrors only ``color``-semantic writes into ``model.shape_color``).

        Args:
            writes: One entry per written shader attribute, each batching all materials that
                resolved a channel to the same attribute name.
        """
        return

    def register_visual_material_textures(self, texture_paths: Sequence[str]) -> None:
        """Declare candidate textures for the ``texture`` channel ahead of any swap.

        Backends that swap textures by index into a pre-built pool (e.g. Newton) load every
        declared texture once — at declaration time or at :meth:`initialize` — so later swaps
        are pure index writes with no I/O. The default implementation is a no-op, which is
        correct for renderers that resolve asset paths through the live stage (Isaac RTX).

        Args:
            texture_paths: Asset paths of the candidate textures.
        """
        return

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Pre-render per-camera setup the backend needs.

        The default implementation is a no-op. Renderer subclasses override
        to perform whatever per-camera initialization their backend requires
        — e.g. authoring stage attributes on the resolved camera prims,
        configuring per-tile GPU buffers, or any other state setup.

        Args:
            stage: Scene stage the camera prims live on, or ``None``
                when no stage context applies. Stage-less backends ignore it.
            spec: Immutable description of the tiled camera bundle.
        """
        return

    @abstractmethod
    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Per-output layout (channels + dtype) this renderer can produce.

        Outputs absent from the mapping are not produced by this backend.

        Returns:
            Mapping from supported :class:`RenderBufferKind` to its :class:`RenderBufferSpec`.
        """
        pass

    @abstractmethod
    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Prepare the stage for rendering before :meth:`create_render_data` is called.

        Some renderers need to export or preprocess the USD stage before
        creating render data. This method is called after the renderer is
        instantiated and before :meth:`create_render_data`.

        Args:
            stage: USD stage to prepare, or None if not applicable.
            num_envs: Number of environments.
        """
        pass

    @abstractmethod
    def create_render_data(self, spec: CameraRenderSpec) -> Any:
        """Create render data for the given camera :class:`CameraRenderSpec`.

        Args:
            spec: Immutable description of the tiled camera (paths, config, device).

        Returns:
            Renderer-specific data for subsequent :meth:`render` / :meth:`read_output` calls.
        """
        pass

    @abstractmethod
    def set_outputs(self, render_data: Any, output_data: dict[str, ProxyArray]) -> None:
        """Store reference to output buffers for writing during render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            output_data: Dictionary mapping output names (e.g. ``"rgb"``, ``"depth"``)
                to pre-allocated :class:`~isaaclab.utils.warp.ProxyArray` wrappers where
                rendered data will be written. Use ``.warp`` for the underlying warp array
                or ``.torch`` for a zero-copy tensor view.
        """
        pass

    @abstractmethod
    def update_transforms(self) -> None:
        """Update scene transforms before rendering.

        Called to sync physics/asset pose state into the renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_geometries(self) -> None:
        """Update mutable geometry attributes before rendering.

        Called to sync physics-driven geometry such as mesh points, extents, or other
        per-frame geometry buffers into the renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_camera(
        self,
        render_data: Any,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """Update camera poses and intrinsics for the next render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            positions: Camera positions in world frame. Shape ``(N,)``, dtype ``wp.vec3f``.
                Use ``.torch`` for a ``(N, 3)`` tensor view.
            orientations: Camera orientations as quaternions ``(x, y, z, w)``. Shape ``(N,)``,
                dtype ``wp.quatf``. Use ``.torch`` for a ``(N, 4)`` tensor view.
            intrinsics: Camera intrinsic matrices. Shape ``(N,)``, dtype ``wp.mat33f``.
                Use ``.torch`` for a ``(N, 3, 3)`` tensor view.
        """
        pass

    @abstractmethod
    def render(self, render_data: Any) -> None:
        """Perform rendering and write to output buffers.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
        """
        pass

    @abstractmethod
    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        """Read rendered outputs from the renderer into the camera data container.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            camera_data: The :class:`~isaaclab.sensors.camera.camera_data.CameraData`
                instance to populate.
        """
        pass

    @abstractmethod
    def cleanup(self, render_data: Any) -> None:
        """Release renderer resources associated with the given render data.

        Args:
            render_data: The render data object to clean up, or ``None``.
        """
        pass
