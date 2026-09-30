# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation.

Each instance is one camera client of the simulation's clone-context-owned native renderer and scene.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import torch
import warp as wp

import isaaclab.utils.warp  # noqa: F401  # initializes Warp runtime

# The ovrtx C library links to its own version of the USD libraries. Having
# the pxr Python package available can cause the C library to load an
# incompatible version of libusd, potentially leading to undefined behavior.
# By setting OVRTX_SKIP_USD_CHECK, we prevent the C library from loading the pxr Python package.
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"


def _preload_ovrtx_native_deps() -> None:
    """Pre-load ``libosdCPU.so`` from ``ovstage`` so ``ovrtx`` can resolve it.

    ``libovrtx.dylib.so`` depends on ``libosdCPU.so.3.6.0``, which ships inside the ``ovstage``
    wheel but is not on the system ``LD_LIBRARY_PATH``. Loading it explicitly places it in the
    process-wide ``dlopen`` cache, which is why this runs before ``ovrtx`` is imported below.
    """
    import ctypes  # noqa: PLC0415
    import importlib.util  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    spec = importlib.util.find_spec("ovstage")
    if spec is None or spec.origin is None:
        return
    lib = pathlib.Path(spec.origin).parent / "bin" / "plugins" / "libosdCPU.so.3.6.0"
    if lib.exists():
        with contextlib.suppress(OSError):
            ctypes.CDLL(str(lib))


_preload_ovrtx_native_deps()

try:
    from ovrtx import Device, Renderer, RendererConfig, TextureStreamingMode
except ModuleNotFoundError as exc:
    if exc.name != "ovrtx":
        raise
    raise ModuleNotFoundError(
        "The OVRTX renderer requires the optional 'ovrtx' runtime wheel, which is not installed. "
        "Run your command with: uv run --extra ovrtx <command> "
        "(or, manually: python -m pip install 'ovrtx==0.4.1.364340')."
    ) from exc

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import SimulationContext

from isaaclab_ov.cloner import OvReplicateContext

from .ovrtx_annotator_utils import (
    build_instance_id_to_labels_and_semantics,
    build_semantic_id_to_labels,
    decode_semantic_id_map,
    decode_stable_id_map,
    decode_stable_id_semantic_id_map,
)
from .ovrtx_compat import RENDER_VAR_FRAME_KEYS
from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
)
from .ovrtx_scene import OvrtxScene
from .ovrtx_usd import build_render_product_as_string

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

logger = logging.getLogger(__name__)

# The resolved integer value is assigned to the ``omni:rtx:minimal:mode`` attribute of the render product.
_RTX_MINIMAL_MODES = {
    RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE.value: 1,
    RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL.value: 2,
    RenderBufferKind.SIMPLE_SHADING_FULL_MDL.value: 3,
}

_DEPTH_RENDER_VAR_OUTPUTS = {
    "DistanceToImagePlaneSD": ("depth", "distance_to_image_plane"),
    "DistanceToCameraSD": ("distance_to_camera",),
}

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "It ships with the Isaac Lab wheel (`pip install isaaclab`); otherwise install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)

_DISABLE_LINUX_CUDA_CPU_SYNC_ENV = "ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC"


def _gpu_side_render_var_sync_enabled() -> bool:
    """Return whether render-var reads wait on the consuming CUDA stream."""
    if not sys.platform.startswith("linux"):
        return True
    value = os.environ.get(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid value for {_DISABLE_LINUX_CUDA_CPU_SYNC_ENV}: {value!r}. Expected '0' or '1'.")
    return value == "1"


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


def _resolve_rtx_minimal_mode(data_types: list[str]) -> int | None:
    """Resolve the RTX minimal mode from data types.

    RTX minimal mode is used to control the rendering quality. The higher the mode, the higher the quality.

    If no simple shading data types are requested, None is returned.

    Args:
        data_types: List of data types.

    Returns:
        The resolved RTX minimal mode if simple shading data types are requested, otherwise None.
    """
    filtered_data_types = [data_type for data_type in data_types if data_type in _RTX_MINIMAL_MODES]
    if not filtered_data_types:
        return None

    if len(filtered_data_types) > 1:
        raise ValueError(f"Multiple simple shading data types requested: {filtered_data_types}.")

    return _RTX_MINIMAL_MODES[filtered_data_types[0]]


class OVRTXRenderData:
    """OVRTX-specific RenderData. Holds warp output buffers sized from :class:`CameraRenderSpec`."""

    def __init__(self, spec: CameraRenderSpec, device):
        """Create render data from a camera render specification."""
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        self.num_envs = spec.num_instances
        self.data_types = spec.cfg.data_types
        self.transform_stream = spec.cfg.prim_path
        self.num_cols = math.ceil(math.sqrt(self.num_envs))
        self.num_rows = math.ceil(self.num_envs / self.num_cols)
        self.warp_buffers: dict[str, wp.array] = {}
        # Per-output metadata collected during render() and copied into CameraData.info by read_output().
        # Populated for "semantic_segmentation" (with an "idToLabels" mapping) and
        # "instance_segmentation" (with "idToLabels" and "idToSemantics" mappings).
        self.renderer_info: dict[str, Any] = {}
        # Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set.
        # ``isp_cfg`` is already fully normalized by ``prepare_cameras`` by the time it reaches here.
        self.ppisp_pipeline: PpispPipeline | None = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            self.ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)


class OVRTXRenderer(BaseRenderer):
    """OVRTX Renderer implementation using the ovrtx library.

    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    cfg: OVRTXRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this OVRTX backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        instance_seg_spec = (
            RenderBufferSpec(4, wp.uint8) if self.cfg.colorize_instance_segmentation else RenderBufferSpec(1, wp.int32)
        )
        # Semantic segmentation: colorized RGBA (uint8), else raw int32 IDs (matches Isaac RTX, whose
        # non-colorized per-pixel value is the semantic ID).
        semantic_seg_spec = (
            RenderBufferSpec(4, wp.uint8) if self.cfg.colorize_semantic_segmentation else RenderBufferSpec(1, wp.int32)
        )
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_FULL_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SEMANTIC_SEGMENTATION: semantic_seg_spec,
            RenderBufferKind.INSTANCE_SEGMENTATION: instance_seg_spec,
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._device = ""
        self._render_product_paths: list[str] = []
        self._render_product_usd = ""
        self._initialized_scene = False
        self._output_id_color_buffers: dict[str, wp.array] = {}
        # Handed over by the camera in :meth:`prepare_cameras`, before anything is cloned.
        self._spec: CameraRenderSpec | None = None
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("OVRTXRenderer requires an active SimulationContext.")
        self._clone_ctx = sim.get_or_create_backend(OvReplicateContext, sim, clone_role="scene")
        self._client_id = self._clone_ctx._add_renderer(self)
        scope_prefix = f"/Render_{self._client_id}/"
        self._render_var_keys = {
            source: scope_prefix + key.removeprefix("/Render/") if key.startswith("/Render/") else key
            for source, key in RENDER_VAR_FRAME_KEYS.items()
        }
        self._camera_xforms: Any = None

    @property
    def visual_material_writer(self):
        """Return the shared detached-scene material-writer factory."""
        return self._clone_ctx.create_visual_material_writer

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Take over the camera and apply OVRTX-specific USD overrides.

        The description is kept so the cloning context can build the scene while the cloner
        replicates, well before any camera initializes.

        When ``spec.cfg.isp_cfg`` is set, resolves it, pins ``exposure:*`` to neutral,
        and applies ``OmniRtxCameraExposureAPI_1`` so OVRTX's RTX exposure model does not
        compound on top of the ISP. Without an ISP, the authored exposure is left alone.

        Raises:
            RuntimeError: If this renderer was already assigned a camera specification.
        """
        if self._spec is not None:
            raise RuntimeError("An OVRTX renderer accepts exactly one CameraRenderSpec; construct one per camera.")
        device = wp.get_device(spec.device)
        if not device.is_cuda:
            raise ValueError(f"OVRTX requires a CUDA render device, got {spec.device!r}.")
        self._spec, self._device = spec, spec.device
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import apply_rtx_exposure_overrides, normalize_ppisp_cfg
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)
            spec.cfg.isp_cfg = normalize_ppisp_cfg(spec.cfg.isp_cfg)
            apply_rtx_exposure_overrides(stage, list(spec.camera_source_prim_paths))

        data_types = list(spec.cfg.data_types)
        if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
            data_types.append("rgb_hdr")
        self._render_product_usd, render_product_path = build_render_product_as_string(
            width=spec.cfg.width,
            height=spec.cfg.height,
            num_envs=spec.num_instances,
            data_types=data_types,
            camera_prim_path=spec.camera_source_prim_paths[0],
            minimal_mode=_resolve_rtx_minimal_mode(data_types),
            background_color=spec.cfg.background_color,
            device_id=device.ordinal,
            enable_shadows=self.cfg.enable_shadows,
            render_scope_name=f"Render_{self._client_id}",
        )
        self._render_product_paths = [render_product_path]
        config = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=True,
            keep_system_alive=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
            active_cuda_gpus=str(device.ordinal),
        )
        self._clone_ctx._configure_ovrtx(
            (device.ordinal, self.cfg.log_file_path, self.cfg.log_level),
            config,
            Renderer,
            OvrtxScene,
            self.cfg.temp_usd_dir,
        )
        logger.info("OVRTX camera %d uses the shared renderer on %s", self._client_id, device)

    def initialize(self) -> None:
        """Bind the scene built while cloning to the physics that now exists.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.initialize`. Rigid transforms
        and geometry layouts bind through the backend-neutral scene-data provider.

        Raises:
            RuntimeError: If the cloner never reached this renderer, which leaves it without the
                scene the bindings below need.
        """
        if self._spec is None:
            raise RuntimeError("OVRTX cannot initialize before a camera supplies its render specification.")
        env_prim_paths = self._clone_ctx.env_prim_paths
        if not env_prim_paths:
            raise RuntimeError(
                "OVRTX has no scene to bind: the cloner never reached this renderer. OVRTX draws its"
                " own copy of the scene, so its camera has to be declared with the scene it belongs"
                " to -- an InteractiveSceneCfg field, or a camera built inside an"
                " isaaclab.cloner.ReplicateSession -- rather than after the scene was replicated."
            )

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("OVRTX requires an active SimulationContext during initialization.")
        plan = sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("OVRTX requires a completed clone plan.")
        self._clone_ctx._initialize_ovrtx(sim.get_scene_data_provider(), plan)
        self._initialized_scene = True
        logger.info("OVRTX bound its scene over %d environment(s)", len(env_prim_paths))

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXRenderData:
        """Create OVRTX-specific RenderData with GPU buffers.

        Raises:
            RuntimeError: If this renderer's scene was never bound, which means the simulation was
                not played between the camera being cloned and initializing.
        """
        if not self._initialized_scene:
            raise RuntimeError(
                "OVRTX has no scene to render. Its scene is built while the cloner replicates and bound"
                " when the simulation is played, so the simulation has to be played before its camera"
                " initializes."
            )
        return OVRTXRenderData(spec, self._device)

    def set_outputs(self, render_data: OVRTXRenderData, output_data: dict[str, ProxyArray]) -> None:
        """Register pre-allocated warp output buffers for rendering.

        Each :class:`~isaaclab.utils.warp.ProxyArray` already carries the correct warp
        dtype from :meth:`~isaaclab.sensors.camera.CameraData.allocate`; store
        the underlying warp array directly. ``rgb`` is excluded because it is a
        non-contiguous strided view into ``rgba`` and is updated automatically.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`.
        """
        render_data.warp_buffers = {
            name: proxy.warp for name, proxy in output_data.items() if name != str(RenderBufferKind.RGB)
        }
        # When PPISP is composed but the user did not request the raw HDR AOV,
        # allocate an internal HDR scratch buffer under "rgb_hdr" so both the
        # HdrColor extractor and PPISP dispatch can use the same buffer map.
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGB_HDR) not in render_data.warp_buffers:
            ref_proxy = next(iter(output_data.values()))
            render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)] = wp.zeros(
                (render_data.num_envs, render_data.height, render_data.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
        if render_data.ppisp_pipeline is not None:
            if str(RenderBufferKind.RGBA) not in render_data.warp_buffers:
                raise ValueError(
                    "OVRTX renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                    " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                    " Camera.cfg.data_types when isp_cfg is set."
                )

    def update(self, render_data: OVRTXRenderData, intrinsics: ProxyArray) -> None:
        """Sync the scene and camera from SDP into OVRTX before rendering.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update`."""
        if self._camera_xforms is None:
            raise RuntimeError("OVRTX updates require clone-time camera bindings.")
        self._clone_ctx._update_ovrtx(self._camera_xforms, render_data.transform_stream)
        del intrinsics  # OVRTX takes the intrinsics from the render product.

    def read_output(
        self,
        render_data: OVRTXRenderData,
        camera_data: CameraData,
    ) -> None:
        """Forward per-output metadata collected during :meth:`render` into ``camera_data.info``.

        This is a *replace*, not a *merge*: every seeded output key is reset to this frame's metadata,
        which is ``None`` when its render var was absent. Because :meth:`render` rebuilds ``renderer_info``
        from scratch each frame (see the ``renderer_info.clear()`` in :meth:`_process_render_frame`), a render
        var that disappears on a later frame (e.g. a missing ``SemanticIdMap``) must clear the corresponding
        ``camera_data.info`` entry too, or downstream consumers would keep reading stale labels. ``renderer_info``
        only ever holds a subset of the outputs, so iterating ``camera_data.info`` both preserves its
        ``output``-mirroring key set and resets any dropped metadata to ``None``.

        Present entries are stored by reference (a shallow assignment, not a deep copy): ``camera_data.info``
        shares the same metadata dict objects as ``render_data.renderer_info`` (e.g. the semantic
        ``idToLabels`` mapping). Those references stay valid even after ``renderer_info`` is cleared or
        rebuilt, and each render builds a fresh metadata dict, so no aliased object is mutated in place.

        Pixel data needs no handling here: :meth:`set_outputs` wraps each ``camera_data.output`` tensor as a
        zero-copy warp array stored in ``render_data.warp_buffers``, and :meth:`render` writes the rendered
        tiles directly into those warp arrays.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`.
        """
        assert camera_data.info is not None, "CameraData.info should be created in CameraData.allocate"
        for output_name in camera_data.info:
            camera_data.info[output_name] = render_data.renderer_info.get(output_name)

    def _generate_random_colors_from_ids(self, input_ids: wp.array, output_colors: wp.array | None) -> wp.array:
        """Generate pseudo-random RGBA colors from uint32 IDs into a reusable output buffer.

        Args:
            input_ids: 3-D uint32 Warp array of shape (H, W, 1).
            output_colors: Existing color buffer to reuse, or None to allocate a new one.

        Returns:
            Color buffer containing the generated colors.
        """

        # Lazily allocate, and re-allocate if the shape changes.
        if output_colors is None or output_colors.shape != input_ids.shape:
            output_colors = wp.zeros(shape=input_ids.shape, dtype=wp.uint32, device=self._device)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=input_ids.shape,
            inputs=[input_ids, output_colors],
            device=self._device,
        )
        return output_colors

    @contextlib.contextmanager
    def _map_render_var_to_dlpack(self, render_var: Any) -> Iterator[wp.array]:
        """Map ``render_var`` for CUDA reads and yield it as a Warp array.

        The render is still in flight when the mapping returns, so reading it has to be ordered
        against render completion. Normally that is a ``cudaStreamWaitEvent`` on the Warp stream the
        consuming kernels run on, which is the ordering the OVRTX API is designed around.

        On Linux that GPU-side wait measures substantially slower end to end, so the mapping is
        instead requested with no GPU-side barrier and the calling thread blocks on the
        render-completion event. Setting :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` to ``1`` puts
        Linux back on the GPU-side wait; it is an escape hatch for platforms where that trade-off
        no longer holds, and is worth re-measuring before being relied on.

        Note that ``sync_stream=0`` is OVRTX's "no sync" sentinel, *not* the NULL CUDA stream: the
        field encodes ``0=no sync, 1=default stream, >1=specific stream``, so omitting the argument
        entirely means ``1``, not ``0``.

        The yielded array is a zero-copy view of the mapped memory and is only valid inside the
        ``with`` block -- the mapping is released on exit.

        Args:
            render_var: OVRTX ``RenderVarOutput`` to map (``frame.render_vars[name]``).

        Yields:
            The render var's contents as a Warp array, valid for the duration of the context.
        """
        gpu_side_sync = _gpu_side_render_var_sync_enabled()
        sync_stream = wp.get_stream(self._device).cuda_stream if gpu_side_sync else 0
        with render_var.map(device=Device.CUDA, sync_stream=sync_stream) as mapping:
            if not gpu_side_sync:
                mapping.wait()
            yield wp.from_dlpack(mapping)

    def _process_id_segmentation_render_var(
        self,
        render_data: OVRTXRenderData,
        frame,
        output_buffers: dict,
        render_var_name: str,
        buffer_key: str,
        colorize: bool,
    ) -> None:
        """Extract a uint32 ID-segmentation render var into ``output_buffers[buffer_key]``.

        Shared by ``semantic_segmentation`` (``SemanticSegmentation``) and ``instance_segmentation``
        (``NonStableInstanceSegmentation``), which only differ in the source render var, the destination buffer,
        and whether to colorize.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
            output_buffers: Destination warp buffers, keyed by data type.
            render_var_name: Name of the OVRTX render var to read.
            buffer_key: Data type key into ``output_buffers``.
            colorize: If True, IDs are mapped to RGBA colors; otherwise raw uint32 IDs are copied.
        """
        if render_var_name not in frame.render_vars or buffer_key not in output_buffers:
            return

        with self._map_render_var_to_dlpack(frame.render_vars[render_var_name]) as tiled_data:
            if tiled_data.dtype != wp.uint32:
                return

            if colorize:
                color_buffer = self._generate_random_colors_from_ids(
                    tiled_data, self._output_id_color_buffers.get(buffer_key)
                )
                self._output_id_color_buffers[buffer_key] = color_buffer

                colors_torch = wp.to_torch(color_buffer)
                colors_uint8 = colors_torch.view(torch.uint8)
                if colors_torch.dim() == 2:
                    h, w = colors_torch.shape
                    colors_uint8 = colors_uint8.reshape(h, w, 4)
                tiled_data = wp.from_torch(colors_uint8, dtype=wp.uint8)
                self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)
            else:
                # Non-colorized: ensure (TH, TW, 1) shape for the uint32 extraction kernel. Reshape the warp
                # array directly instead of round-tripping through torch, which raises on ``torch.uint32``
                # (newer torch exposes the dtype but ``wp.from_torch`` still rejects it).
                if tiled_data.ndim == 2:
                    tiled_data = tiled_data.reshape((*tiled_data.shape, 1))
                self._launch_extract_all_tiles(render_data, tiled_data, output_buffers[buffer_key])

    def _process_semantic_id_map(self, render_data: OVRTXRenderData, frame) -> None:
        """Decode the ``SemanticIdMap`` render var into ``render_data.renderer_info["semantic_segmentation"]``.

        Populates an ``"idToLabels"`` mapping compatible with Isaac RTX / Replicator: keys are the raw semantic
        IDs (``colorize_semantic_segmentation=False``) or the RGBA color tuples the segmentation buffer uses
        (``colorize_semantic_segmentation=True``); values are ``{semantic_type: label}`` dicts. The reserved
        BACKGROUND (ID 0) and UNLABELLED (ID 1) entries are always included.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
        """
        semantic_id_map = frame.render_vars.get(self._render_var_keys["SemanticIdMap"])
        if semantic_id_map is None:
            return

        with semantic_id_map.map(device=Device.CPU) as mapping:
            labels_by_id = decode_semantic_id_map(np.from_dlpack(mapping))

        render_data.renderer_info["semantic_segmentation"] = {
            "idToLabels": build_semantic_id_to_labels(
                labels_by_id, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
            )
        }

    def _process_instance_segmentation_maps(self, render_data: OVRTXRenderData, frame) -> None:
        """Decode the instance-segmentation map render vars into ``renderer_info["instance_segmentation"]``.

        An *instance pixel ID* is a compact integer that the renderer assigns to each visible object instance.
        Every pixel in the segmentation buffer holds the ID of the instance rendered at that location; the same
        ID maps to the same object across the entire frame.  ID 0 is reserved for BACKGROUND (no geometry), and
        ID 1 for UNLABELLED (geometry with no semantic annotation).  All other IDs are dynamically assigned per
        frame.

        Populates ``"idToLabels"`` (instance pixel ID -> USD prim path) and ``"idToSemantics"`` (instance pixel
        ID -> ``{semantic_type: label}``) compatible with Isaac RTX / Replicator. Resolving both requires all
        three map render vars — ``StableIdSemanticIdMap`` (pixel ID -> stable ID + semantic ID), ``StableIdMap``
        (stable ID -> prim path), and ``SemanticIdMap`` (semantic ID -> label). Keys are the raw pixel IDs
        (``colorize_instance_segmentation=False``) or the RGBA color tuples the segmentation buffer uses
        (``colorize_instance_segmentation=True``); the reserved BACKGROUND (ID 0) and UNLABELLED (ID 1) entries
        are always included.

        Raises:
            RuntimeError: If any of the three required render vars is absent from ``frame``.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
        """
        keys = tuple(
            self._render_var_keys[source] for source in ("StableIdSemanticIdMap", "StableIdMap", "SemanticIdMap")
        )
        resolved = {key: frame.render_vars.get(key) for key in keys}
        missing = [key for key, value in resolved.items() if value is None]
        if missing:
            raise RuntimeError(
                f"instance_segmentation was requested but the following render vars are missing from the "
                f"OVRTX frame: {missing}. Available vars: {list(frame.render_vars.keys())}"
            )

        with resolved[keys[0]].map(device=Device.CPU) as mapping:
            stable_id_semantic_id_map = decode_stable_id_semantic_id_map(np.from_dlpack(mapping))
        with resolved[keys[1]].map(device=Device.CPU) as mapping:
            stable_id_to_path = decode_stable_id_map(np.from_dlpack(mapping))
        with resolved[keys[2]].map(device=Device.CPU) as mapping:
            semantic_id_to_labels = decode_semantic_id_map(np.from_dlpack(mapping))

        id_to_labels, id_to_semantics = build_instance_id_to_labels_and_semantics(
            stable_id_semantic_id_map,
            stable_id_to_path,
            semantic_id_to_labels,
            colorize=self.cfg.colorize_instance_segmentation,
            device=self._device,
        )
        render_data.renderer_info["instance_segmentation"] = {
            "idToLabels": id_to_labels,
            "idToSemantics": id_to_semantics,
        }

    def _launch_extract_all_tiles(
        self, render_data: OVRTXRenderData, tiled_buffer: wp.array, output_buffer: wp.array
    ) -> None:
        """Launch ``extract_all_tiles_kernel`` for one tiled/output buffer pair.

        This is the only place that should launch ``extract_all_tiles_kernel``: it validates that
        ``output_buffer`` cannot read past the end of ``tiled_buffer`` (the kernel derives its per-thread
        channel loop bound from ``output_buffer``'s last dimension) before every launch, so callers cannot
        accidentally skip the check.

        Args:
            render_data: OVRTX render data for the current frame.
            tiled_buffer: 3D array of shape (H, W, C) holding all tiles packed into one buffer.
            output_buffer: 4D array of shape (num_envs, H, W, C) to receive the per-env tiles, with C no
                greater than ``tiled_buffer``'s channel count.

        Raises:
            ValueError: If ``output_buffer``'s channel count exceeds ``tiled_buffer``'s.
        """
        tiled_channels = tiled_buffer.shape[-1]
        output_channels = output_buffer.shape[-1]
        if output_channels > tiled_channels:
            raise ValueError(
                f"Output buffer has {output_channels} channels but the tiled buffer only has {tiled_channels};"
                " extract_all_tiles_kernel would read out of bounds."
            )

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(render_data.num_envs, render_data.height, render_data.width),
            inputs=[
                tiled_buffer,
                output_buffer,
                render_data.num_cols,
                render_data.width,
                render_data.height,
            ],
            device=self._device,
        )

    def _extract_rgba_tiles(
        self,
        render_data: OVRTXRenderData,
        tiled_data: wp.array,
        output_buffers: dict,
        buffer_key: str,
    ) -> None:
        """Extract per-env RGBA tiles from tiled buffer into output_buffers (single kernel launch)."""
        output_buffer = output_buffers[buffer_key]
        num_channels = output_buffer.shape[-1]
        if num_channels not in (3, 4):
            raise ValueError(f"Expected RGB (3 channels) or RGBA (4 channels), got {num_channels}")

        self._launch_extract_all_tiles(render_data, tiled_data, output_buffer)

    def _extract_hdr_color_tiles(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env HdrColor tiles into output_buffers."""
        if "rgb_hdr" not in output_buffers:
            return
        if tiled_data.dtype not in (wp.float16, wp.float32):
            raise TypeError(f"Unsupported OVRTX HdrColor dtype: {tiled_data.dtype}.")
        self._launch_extract_all_tiles(render_data, tiled_data, output_buffers["rgb_hdr"])

    def _prepare_ppisp_hdr_source(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> wp.array:
        """Return the PPISP HdrColor source on the output buffer device."""
        if render_data.ppisp_pipeline is None:
            return tiled_data

        output_device = str(output_buffers[str(RenderBufferKind.RGB_HDR)].device)
        if str(tiled_data.device) == output_device:
            return tiled_data

        # FIXME: OVRTX render var mapping can select a different CUDA device
        # than the camera/output buffers on MGPU systems. Keep this PPISP-only
        # bridge until render var mapping can be constrained like transform
        # bindings, whose maps pin ``device_id`` (see ``ovrtx_mapping``).
        return wp.clone(tiled_data, device=output_device)

    def _process_render_frame(self, render_data: OVRTXRenderData, frame, output_buffers: dict) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        # Reset per-output metadata so it is a snapshot of this frame only. Unlike pixel AOVs (always
        # present), metadata like the semantic ``idToLabels`` is only repopulated below when its render var
        # is available, so without this a missing SemanticIdMap on a later frame would leave a stale mapping.
        render_data.renderer_info.clear()

        ldr_color = frame.render_vars.get(self._render_var_keys["LdrColor"])
        if ldr_color is not None:
            ldr_outputs = (() if render_data.ppisp_pipeline is not None else ("rgba",)) + tuple(_RTX_MINIMAL_MODES)
            buffer_keys = [buffer_key for buffer_key in ldr_outputs if buffer_key in output_buffers]
            if buffer_keys:
                with self._map_render_var_to_dlpack(ldr_color) as tiled_data:
                    for buffer_key in buffer_keys:
                        self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)

        for depth_source, buffer_keys in _DEPTH_RENDER_VAR_OUTPUTS.items():
            depth_var = self._render_var_keys[depth_source]
            if depth_var not in frame.render_vars:
                continue
            if not any(buffer_key in output_buffers for buffer_key in buffer_keys):
                continue
            with self._map_render_var_to_dlpack(frame.render_vars[depth_var]) as tiled_depth_data:
                if tiled_depth_data.dtype == wp.uint32:
                    tiled_depth_data = wp.from_torch(
                        wp.to_torch(tiled_depth_data).view(torch.float32), dtype=wp.float32
                    )
                for buffer_key in buffer_keys:
                    if buffer_key in output_buffers:
                        self._launch_extract_all_tiles(render_data, tiled_depth_data, output_buffers[buffer_key])

        albedo = frame.render_vars.get(self._render_var_keys["DiffuseAlbedoSD"])
        if albedo is not None and "albedo" in output_buffers:
            with self._map_render_var_to_dlpack(albedo) as tiled_albedo_data:
                self._extract_rgba_tiles(render_data, tiled_albedo_data, output_buffers, "albedo")

        hdr_color = frame.render_vars.get(self._render_var_keys["HdrColor"])
        if hdr_color is not None and "rgb_hdr" in output_buffers:
            with self._map_render_var_to_dlpack(hdr_color) as tiled_hdr_data:
                tiled_hdr_data = self._prepare_ppisp_hdr_source(render_data, tiled_hdr_data, output_buffers)
                self._extract_hdr_color_tiles(render_data, tiled_hdr_data, output_buffers)

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            self._render_var_keys["SemanticSegmentation"],
            "semantic_segmentation",
            self.cfg.colorize_semantic_segmentation,
        )
        # Decode the SemanticIdMap into camera.data.info["semantic_segmentation"]["idToLabels"].
        if "semantic_segmentation" in output_buffers:
            self._process_semantic_id_map(render_data, frame)

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            self._render_var_keys["NonStableInstanceSegmentation"],
            "instance_segmentation",
            self.cfg.colorize_instance_segmentation,
        )
        # Decode the StableIdSemanticIdMap/StableIdMap/SemanticIdMap trio into
        # camera.data.info["instance_segmentation"]["idToLabels"] and ["idToSemantics"].
        if "instance_segmentation" in output_buffers:
            self._process_instance_segmentation_maps(render_data, frame)

        normals = frame.render_vars.get(self._render_var_keys["NormalSD"])
        if normals is not None and "normals" in output_buffers:
            with self._map_render_var_to_dlpack(normals) as tiled_normals_data:
                self._launch_extract_all_tiles(render_data, tiled_normals_data, output_buffers["normals"])

        # For motion vectors, extract only the first two (u, v) channels from the tiled buffer.
        # Note: mirrors the Isaac RTX renderer's handling of the "TargetMotionSD" AOV
        # (check: https://github.com/isaac-sim/IsaacLab/issues/2003).
        motion_vectors = frame.render_vars.get(self._render_var_keys["TargetMotionSD"])
        if motion_vectors is not None and "motion_vectors" in output_buffers:
            with self._map_render_var_to_dlpack(motion_vectors) as tiled_motion_vectors_data:
                self._launch_extract_all_tiles(render_data, tiled_motion_vectors_data, output_buffers["motion_vectors"])

    def render(self, render_data: OVRTXRenderData) -> None:
        """Render the scene into the provided RenderData.

        Raises:
            RuntimeError: If the scene was never bound; see :meth:`initialize`.
        """
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if not self._render_product_paths:
            raise RuntimeError("OVRTX initialized without its render product.")
        products = self._clone_ctx._render_ovrtx(self._render_product_paths)
        product_path = self._render_product_paths[0]
        if product_path not in products or not products[product_path].frames:
            raise RuntimeError(f"OVRTX produced no frame for render product {product_path!r}.")
        self._process_render_frame(render_data, products[product_path].frames[0], render_data.warp_buffers)

        # Post-render PPISP: HDR scene-linear -> LDR RGBA. Source and destination are the same warp
        # buffer map used by extraction.
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)],
                render_data.warp_buffers[str(RenderBufferKind.RGBA)],
            )

    def cleanup(self, render_data: OVRTXRenderData | None) -> None:
        """Release the render data's buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`.

        The scene, its bindings and render product outlive render data across stop/play cycles, so
        :meth:`close` releases them at simulation teardown.
        """
        if render_data is None:
            return
        render_data.warp_buffers.clear()
        render_data.renderer_info.clear()
        render_data.ppisp_pipeline = None

    def close(self) -> None:
        """Release this camera client and the shared scene after the final client."""
        self._clone_ctx._remove_renderer(self)
        self._camera_xforms = None
        self._render_product_paths.clear()
        self._render_product_usd = ""
        self._output_id_color_buffers.clear()
        self._initialized_scene = False
