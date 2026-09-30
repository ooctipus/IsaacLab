# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The native OVRTX scene populated from one clone-plan snapshot."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from ovrtx import BindingFlag, DataAccess, PrimMode, Semantic

from isaaclab.scene_data import SceneDataFormat

if TYPE_CHECKING:
    from ovrtx import Renderer

logger = logging.getLogger(__name__)


class OvrtxScene:
    """One renderer-owned scene populated and bound from the clone plan."""

    transform_format = SceneDataFormat.TransposedMatrix44d
    point_format = SceneDataFormat.HostMeshPoints

    def __init__(self, renderer: Renderer):
        """Create the scene owned by the renderer.

        Args:
            renderer: The native renderer that draws this scene.
        """
        self._renderer: Renderer | None = renderer
        self._bindings: list[_SceneHandle] = []

    def open(self, usd_text: str) -> None:
        """Open the single clone-context USDA export."""
        logger.info("Loading USD into OVRTX...")
        self._renderer.open_usd_from_string(usd_text)

    def clone(self, source: str, targets: Sequence[str]) -> None:
        """Copy one planned source subtree onto its planned target paths."""
        self._renderer.clone_usd(source, list(targets))

    def write_tokens(self, paths: Sequence[str], attribute: str, tokens: Sequence[str]) -> None:
        """Write one token per planned prim."""
        self._renderer.write_attribute(
            list(paths), attribute, list(tokens), semantic=Semantic.TOKEN_STRING, prim_mode=PrimMode.CREATE_NEW
        )

    def write_reset_xform_stack(self, paths: Sequence[str]) -> None:
        """Pin world-space transforms on planned paths."""
        self._renderer.write_attribute(
            list(paths),
            "omni:resetXformStack",
            np.full(len(paths), True, dtype=np.bool_),
            prim_mode=PrimMode.MUST_EXIST,
        )

    def point_render_products_at(self, product_paths: Sequence[str], camera_paths: Sequence[str]) -> None:
        """Point tiled render products at every planned camera path."""
        self._renderer.write_array_attribute(
            list(product_paths),
            "camera",
            [list(camera_paths) for _ in product_paths],
            prim_mode=PrimMode.MUST_EXIST,
        )

    def bind(
        self,
        paths: Sequence[str],
        attribute: str = "omni:xform",
        *,
        dtype: Any = None,
        shape: tuple[int, ...] | None = None,
        is_array: bool = False,
    ) -> _SceneHandle:
        """Bind repeated writes to exact planned prim paths."""
        kwargs = {
            "prim_paths": list(paths),
            "attribute_name": attribute,
            "dtype": dtype,
            "shape": shape,
            "prim_mode": PrimMode.MUST_EXIST,
            "flags": BindingFlag.OPTIMIZE,
        }
        if is_array:
            binding = self._renderer.bind_array_attribute(**kwargs)
        else:
            binding = self._renderer.bind_attribute(
                **kwargs, semantic=Semantic.XFORM_MAT4x4 if attribute == "omni:xform" else Semantic.NONE
            )
        handle = _SceneHandle(binding, list(paths))
        self._bindings.append(handle)
        return handle

    def release(self, handle: _SceneHandle) -> None:
        """Release one persistent binding before scene teardown."""
        if handle not in self._bindings:
            return
        handle.binding.unbind()
        self._bindings.remove(handle)

    def pin_world_space(self, handle: _SceneHandle) -> None:
        """Prevent ancestor transforms from being applied to world-space point data."""
        self.write_reset_xform_stack(handle.paths)
        self._renderer.write_attribute(
            handle.paths,
            "omni:xform",
            np.tile(np.eye(4, dtype=np.float64), (len(handle.paths), 1, 1)),
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )

    def write_xforms(self, handle: _SceneHandle, transforms: wp.array) -> None:
        """Write the SDP matrix pointer directly through the persistent binding."""
        handle.binding.write(
            transforms,
            data_access=DataAccess.ASYNC,
            cuda_stream=wp.get_stream(str(transforms.device)).cuda_stream,
        )

    def write_points(self, binding: tuple[_SceneHandle, list[int], list[int]], points: np.ndarray) -> None:
        """Write SDP-formatted host point views through the persistent binding."""
        handle, offsets, counts = binding
        slices = [points[offset : offset + count] for offset, count in zip(offsets, counts, strict=True)]
        handle.binding.write(slices, data_access=DataAccess.ASYNC)

    def step(self, product_paths: Sequence[str]) -> dict:
        """Render the requested products."""
        return self._renderer.step(render_products=set(product_paths), delta_time=1.0 / 60.0)

    def close(self) -> None:
        """Release bindings and destroy the owned renderer."""
        for handle in tuple(self._bindings):
            try:
                self.release(handle)
            except Exception as exc:  # noqa: BLE001 - teardown must not mask the caller's error
                if "destroyed" not in str(exc).lower():
                    logger.warning("Error releasing a binding over %d prim(s): %s", len(handle.paths), exc)
        self._bindings.clear()
        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            renderer.destroy()


class _SceneHandle:
    """A persistent native binding and the exact paths it covers."""

    __slots__ = ("binding", "paths")

    def __init__(self, binding: Any, paths: list[str]):
        self.binding = binding
        self.paths = paths
