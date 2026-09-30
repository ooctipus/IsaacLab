# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for renderer implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .camera_render_spec import CameraRenderSpec
from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from isaaclab.cloner import ClonePlan
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray


@dataclass(frozen=True)
class VisualMaterialBatch:
    """One flat material-channel buffer and its aligned backend addresses."""

    channel: str
    material_paths: tuple[str, ...]
    shader_paths: tuple[str, ...]
    input_names: tuple[str, ...]
    values: torch.Tensor


class BaseRenderer(ABC):
    """Abstract base class for renderer implementations."""

    def initialize(self) -> None:
        """Post-physics one-time initialization hook. Called only once."""
        return

    @property
    def visual_material_writer(self) -> Callable[[tuple[VisualMaterialBatch, ...]], Any] | None:
        """Return the backend's shared material-writer factory, if supported.

        Its writer accepts ``None`` for a full sync or channel-to-material-offset device arrays plus
        one environment-id device array for partial writes, and provides an idempotent ``close()``.
        """
        return None

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

    def prepare_stage(self, stage: Any, plan: ClonePlan) -> None:
        """Prepare the stage for rendering before :meth:`create_render_data` is called.

        The default implementation is a no-op. A renderer that authors per-environment attributes
        reads where the environments are from ``plan`` rather than assuming a naming convention.

        Args:
            stage: USD stage to prepare, or None if not applicable.
            plan: Replication layout the stage was cloned from.

        Raises:
            ValueError: If stage preparation is attempted outside a clone-plan lifecycle.
        """
        if plan is None:
            raise ValueError("Renderer stage preparation requires an active clone plan.")
        return

    @abstractmethod
    def create_render_data(self, spec: CameraRenderSpec) -> Any:
        """Create render data for the given camera :class:`CameraRenderSpec`.

        Args:
            spec: Immutable description of the tiled camera (paths, config, device).

        Returns:
            Renderer-specific data for subsequent :meth:`update`, :meth:`render`, and
            :meth:`read_output` calls.
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
    def update(self, render_data: Any, intrinsics: ProxyArray) -> None:
        """Update scene geometry and camera state for the next render.

        The renderer requests every physics-derived field from the scene-data provider. Only
        camera-owned intrinsic metadata crosses this interface directly.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
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

    def close(self) -> None:
        """Release resources owned by the renderer itself rather than by a render data.

        Each camera owns its renderer instance, but renderer state outlives the camera's transient
        render data and cannot be released from :meth:`cleanup`. The simulation calls this once at
        teardown, while the stage and the underlying renderer backend are still alive.

        The default implementation is a no-op, for backends whose state lives entirely on the
        render data. Implementations must be idempotent.
        """
        return
