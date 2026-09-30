# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp renderer for tiled camera rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import newton
import torch
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import replace_background_depth_wp

from isaaclab_newton.cloner.replicate import NewtonReplicateContext

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .segmentation import NewtonSegmentationMapper, NewtonSegmentationMapping

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.cloner import ClonePlan
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "It ships with the Isaac Lab wheel (`pip install isaaclab`); otherwise install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


class RenderData:
    # Maps each supported RenderBufferKind to (CameraOutputs field name, Newton warp dtype).
    # Newton reinterprets the allocated buffer memory: e.g. RGBA is allocated as (N,H,W,4) uint8
    # but the Newton sensor API consumes it as (world_count,1,H,W) uint32 (same bytes, packed view).
    #
    # The depth family (``distance_to_camera`` / ``distance_to_image_plane`` / ``depth``) is handled
    # separately in :meth:`set_outputs` rather than through this map, because Newton emits a single
    # ray-hit-distance buffer that must be reused as the source for the planar-depth conversion.
    #
    # The segmentation family (``semantic_segmentation`` / ``instance_segmentation``) is likewise
    # handled separately: Newton emits a single per-shape index buffer that is remapped into each
    # requested segmentation output by
    # :class:`~isaaclab_newton.renderers.segmentation.NewtonSegmentationMapper`.
    _OUTPUT_MAP: dict[str, tuple[str, type]] = {
        str(RenderBufferKind.RGBA): ("color_image", wp.uint32),
        str(RenderBufferKind.RGB_HDR): ("hdr_color_image", wp.vec3f),
        str(RenderBufferKind.ALBEDO): ("albedo_image", wp.uint32),
        str(RenderBufferKind.NORMALS): ("normals_image", wp.vec3f),
    }

    # Newton's native ``depth_image`` is the ray-hit (euclidean) distance from the camera optical
    # center, which is Isaac Lab's ``distance_to_camera``.
    _RAY_DEPTH_KIND: str = str(RenderBufferKind.DISTANCE_TO_CAMERA)
    # Planar-depth outputs (distance along the camera's forward axis). ``depth`` is Isaac Lab's alias
    # for ``distance_to_image_plane``. Both are derived from the ray depth via
    # ``convert_ray_depth_to_forward_depth``.
    _PLANE_DEPTH_KINDS: frozenset[str] = frozenset(
        {
            str(RenderBufferKind.DEPTH),
            str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE),
        }
    )

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=4) = None
        hdr_color_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) = None
        # Buffer Newton fills with ray-hit (euclidean) distance. Bound either to the caller's
        # ``distance_to_camera`` output or to an internal scratch buffer (see :meth:`set_outputs`).
        depth_image: wp.array(dtype=wp.float32, ndim=4) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        # Buffer Newton fills with the per-pixel shape index; the source for all segmentation outputs.
        shape_index_image: wp.array(dtype=wp.uint32, ndim=4) = None

    def __init__(
        self,
        newton_sensor: newton.sensors.SensorTiledCamera,
        spec: CameraRenderSpec,
        seg_mapper: NewtonSegmentationMapper | None = None,
        renderer_cfg: NewtonWarpRendererCfg | None = None,
    ):
        self.newton_sensor = newton_sensor
        # Shared, scene-static segmentation lookup builder (``None`` until segmentation is requested).
        self._seg_mapper = seg_mapper
        self._renderer_cfg = renderer_cfg

        self.num_cameras = 1

        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_transforms: wp.array(dtype=wp.transformf, ndim=2) = None
        self.transform_stream = spec.cfg.prim_path
        self.sensor_task_name: str | None = f"newton_warp_render:{id(self)}"
        self.outputs = RenderData.CameraOutputs()
        # Requested depth-family destination views keyed by data-type name. Each view aliases the
        # caller's output buffer as ``(world_count, 1, H, W)`` float32.
        self._depth_dests: dict[str, wp.array] = {}
        # Internal ray-depth buffer allocated only when a planar-depth output is requested without
        # ``distance_to_camera``; gives ``convert_ray_depth_to_forward_depth`` a source to read from.
        self._ray_depth_scratch: wp.array | None = None
        # Requested segmentation outputs keyed by data-type name -> (destination view, mapping). Each view
        # aliases the caller's output buffer as ``(world_count, 1, H, W)`` uint32.
        self._seg_dests: dict[str, tuple[wp.array, NewtonSegmentationMapping]] = {}
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        # Camera clipping planes [m] from ``spawn.clipping_range`` (``[0]`` near, ``[1]`` far).
        # Newton's ray tracer has no near-plane parameter, so only the far plane is enforced (through
        # the sensor's ``max_distance``); ``near_clip`` is captured for consumers but not applied.
        clipping_range = None if spec.cfg.spawn is None else spec.cfg.spawn.clipping_range
        self.near_clip: float | None = float(clipping_range[0]) if clipping_range is not None else None
        self.far_clip: float | None = float(clipping_range[1]) if clipping_range is not None else None

        # ABGR clear color packed as uint32 — Newton's SensorTiledCamera reads the low byte as R,
        # next as G, next as B, high byte as A (little-endian RGBA in memory). Default is 93% gray
        # (0xFFEEEEEE), matching the RTX renderer background and improving visibility of dark objects.
        background_color = spec.cfg.background_color
        if background_color is not None:
            r, g, b = (max(0, min(255, round(c * 255))) for c in background_color)
            self.clear_color: int = (0xFF << 24) | (b << 16) | (g << 8) | r
        else:
            self.clear_color = 0xFFEEEEEE

        # Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set.
        # ``isp_cfg`` is already fully normalized by ``prepare_cameras`` by the time it reaches here.
        self.ppisp_pipeline: PpispPipeline | None = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            self.ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)
        self._hdr_scratch_wp: wp.array | None = None
        """Internal HDR scratch buffer allocated when PPISP is composed but the
        user did not request ``"rgb_hdr"`` in ``data_types``. Also exposed to
        the Newton sensor through :attr:`CameraOutputs.hdr_color_image` as a
        vec3f reinterpretation of this same backing storage."""
        self._ppisp_hdr_source: wp.array | None = None
        """PPISP HDR source bound once in :meth:`set_outputs` from the caller's
        ``rgb_hdr`` output or :attr:`_hdr_scratch_wp`."""
        self._ppisp_rgba_dest: wp.array | None = None
        """PPISP LDR destination bound once in :meth:`set_outputs` from the
        caller's ``rgba`` output."""

    def _view(self, proxy: ProxyArray, dtype: type, shape: tuple[int, ...]) -> wp.array:
        """Alias the caller's output buffer as a ``(world_count, 1, H, W)`` warp array of ``dtype``.

        Newton reinterprets the backing memory in place (no copy), so the sensor writes directly
        into the camera's output buffer.
        """
        wp_arr = proxy.warp
        return wp.array(ptr=wp_arr.ptr, dtype=dtype, shape=shape, device=wp_arr.device, copy=False)

    def set_outputs(self, output_data: dict[str, ProxyArray]):
        shape = (self.newton_sensor.model.world_count, self.num_cameras, self.height, self.width)
        self._depth_dests = {}
        self._ray_depth_scratch = None
        self._seg_dests = {}
        self.outputs.shape_index_image = None
        ray_depth_dest: wp.array | None = None
        for output_name, proxy in output_data.items():
            # Depth family: bind each requested output to a float32 destination view. Newton fills
            # only the ray-hit distance; planar outputs are derived from it in :meth:`_convert_plane_depth`.
            if output_name == self._RAY_DEPTH_KIND or output_name in self._PLANE_DEPTH_KINDS:
                dest = self._view(proxy, wp.float32, shape)
                self._depth_dests[output_name] = dest
                if output_name == self._RAY_DEPTH_KIND:
                    ray_depth_dest = dest
                continue
            # Segmentation family: bind each requested output to a destination view — colorized RGBA
            # (uint32 packed) or raw int32 ids (matching the Isaac RTX / OVRTX contract).  Newton
            # fills only the shape-index scratch (uint32), which is remapped into each output in
            # :meth:`_convert_segmentation`.
            if output_name == RenderBufferKind.SEMANTIC_SEGMENTATION:
                colorize = bool(self._renderer_cfg.colorize_semantic_segmentation)
            elif output_name == RenderBufferKind.INSTANCE_SEGMENTATION:
                colorize = bool(self._renderer_cfg.colorize_instance_segmentation)
            else:
                colorize = None
            if colorize is not None:
                if self._seg_mapper is None:
                    raise RuntimeError(
                        f"Output '{output_name}' requires a segmentation mapper, but none was created. "
                        "Ensure the camera's data_types includes the segmentation output."
                    )
                seg_mapping = self._seg_mapper.get_mapping(output_name, colorize)
                dest = self._view(proxy, wp.uint32 if colorize else wp.int32, shape)
                self._seg_dests[output_name] = (dest, seg_mapping)
                continue
            if output_name == str(RenderBufferKind.RGB):
                continue
            try:
                field_name, dtype = self._OUTPUT_MAP[output_name]
            except KeyError as exc:
                raise ValueError(f"NewtonWarpRenderer does not support output {output_name!r}.") from exc
            setattr(self.outputs, field_name, self._view(proxy, dtype, shape))
        # Bind the buffer Newton fills with ray-hit distance. Write straight into the
        # ``distance_to_camera`` output when requested; otherwise allocate an internal scratch so the
        # planar-depth conversion has a source to read from.
        if ray_depth_dest is not None:
            self.outputs.depth_image = ray_depth_dest
        elif any(name in self._PLANE_DEPTH_KINDS for name in self._depth_dests):
            self._ray_depth_scratch = wp.zeros(shape, dtype=wp.float32, device=self.newton_sensor.model.device)
            self.outputs.depth_image = self._ray_depth_scratch
        else:
            self.outputs.depth_image = None
        # Allocate the shape-index buffer Newton fills when any segmentation output is requested; all
        # requested segmentation outputs are remapped from this single buffer in :meth:`_convert_segmentation`.
        if self._seg_dests:
            self.outputs.shape_index_image = wp.zeros(shape, dtype=wp.uint32, device=self.newton_sensor.model.device)
        # When PPISP is composed but the user did not request the raw HDR AOV,
        # allocate an internal HDR scratch buffer and route a vec3f-shaped view
        # of it as the Newton sensor's ``hdr_color_image`` so the renderer
        # fills it directly.
        if self.ppisp_pipeline is not None and self.outputs.hdr_color_image is None:
            ref_proxy = next(iter(output_data.values()))
            self._hdr_scratch_wp = wp.zeros(
                (self.newton_sensor.model.world_count, self.height, self.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
            self.outputs.hdr_color_image = wp.array(
                ptr=self._hdr_scratch_wp.ptr,
                dtype=wp.vec3f,
                shape=shape,
                device=self._hdr_scratch_wp.device,
                copy=False,
            )
        # Bind the two warp arrays the per-frame PPISP dispatch needs.
        if self.ppisp_pipeline is not None:
            if str(RenderBufferKind.RGBA) not in output_data:
                raise ValueError(
                    "Newton renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                    " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                    " Camera.cfg.data_types when isp_cfg is set."
                )
            hdr_proxy = output_data.get(str(RenderBufferKind.RGB_HDR))
            self._ppisp_hdr_source = hdr_proxy.warp if hdr_proxy is not None else self._hdr_scratch_wp
            self._ppisp_rgba_dest = output_data[str(RenderBufferKind.RGBA)].warp

    def get_output(self, output_name: str) -> wp.array:
        if output_name in self._depth_dests:
            return self._depth_dests[output_name]
        elif output_name in self._seg_dests:
            return self._seg_dests[output_name][0]
        elif output_name == RenderBufferKind.RGBA:
            return self.outputs.color_image
        elif output_name == RenderBufferKind.RGB_HDR:
            return self.outputs.hdr_color_image
        elif output_name == RenderBufferKind.ALBEDO:
            return self.outputs.albedo_image
        elif output_name == RenderBufferKind.NORMALS:
            return self.outputs.normals_image
        return None

    def _convert_segmentation(self):
        """Remap Newton's shape-index buffer into each requested segmentation output.

        Newton emits a single per-pixel shape index (:attr:`CameraOutputs.shape_index_image`);
        ``semantic_segmentation`` / ``instance_segmentation`` are each derived from it by a
        :class:`~isaaclab_newton.renderers.segmentation.NewtonSegmentationMapping`.
        No-op when no segmentation output was requested.
        """
        if self.outputs.shape_index_image is None:
            return
        for dest, seg_mapping in self._seg_dests.values():
            seg_mapping.convert_shape_index_to_output(self.outputs.shape_index_image, dest)

    def segmentation_info(self) -> dict[str, dict]:
        """Per-output ``idToLabels`` / ``idToSemantics`` info for the requested segmentation outputs."""
        return {name: seg_mapping.info for name, (_dest, seg_mapping) in self._seg_dests.items()}

    def _convert_plane_depth(self):
        """Fill any planar-depth outputs from the ray-hit distance Newton just rendered.

        Newton emits ``distance_to_camera`` (euclidean ray distance). ``depth`` and
        ``distance_to_image_plane`` are the projection of that distance onto the camera's forward
        axis, computed by :meth:`newton.sensors.SensorTiledCamera.Utils.convert_ray_depth_to_forward_depth`.
        No-op when only ``distance_to_camera`` (or no depth output) was requested.
        """
        assert self.outputs.depth_image is not None, "Expected a depth image to convert"
        for output_name, dest in self._depth_dests.items():
            if output_name in self._PLANE_DEPTH_KINDS:
                self.newton_sensor.utils.convert_ray_depth_to_forward_depth(
                    self.outputs.depth_image,
                    self.camera_transforms,
                    self.camera_rays,
                    out_depth=dest,
                )

    def _apply_depth_clipping(self, behavior: str):
        """Apply the renderer's depth-clipping behavior to the depth-family outputs.

        Newton writes ``0.0`` for rays that miss all geometry or fall beyond the far plane
        (``max_distance``), so ``"none"`` and ``"zero"`` both leave that ``0.0`` background. ``"max"``
        replaces the background with the far clip [m] to mirror the RTX renderer's
        :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.depth_clipping_behavior`. No-op when no
        depth output was requested or the camera did not provide a clipping range.

        """
        if behavior != "max" or self.far_clip is None:
            return
        for dest in self._depth_dests.values():
            replace_background_depth_wp(dest, self.far_clip, device=dest.device)

    def update(self, transforms: SceneDataFormat.Transform, intrinsics: ProxyArray):
        # SDP owns this persistent converted buffer. Alias it in Newton's tiled-camera shape so
        # sensor-graph capture sees the same pointer on every generation.
        self.camera_transforms = wp.array(
            ptr=transforms.transforms.ptr,
            dtype=wp.transformf,
            shape=(1, self.newton_sensor.model.world_count),
            device=transforms.transforms.device,
            copy=False,
        )

        if self.camera_rays is None:
            first_focal_length = intrinsics.torch[:, 1, 1][0:1]
            fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))

            fov_warp = wp.from_torch(fov_radians_all, dtype=wp.float32)
            self.camera_rays = self.newton_sensor.utils.compute_camera_rays_pinhole(
                self.width, self.height, camera_fovs=fov_warp
            )


class NewtonWarpRenderer(BaseRenderer):
    """Newton Warp backend for tiled camera rendering."""

    RenderData = RenderData

    def __init__(self, cfg: NewtonWarpRendererCfg):
        """Pre-physics initialization."""
        self.cfg = cfg
        self.newton_sensor: newton.sensors.SensorTiledCamera | None = None
        self._clone_plan: ClonePlan
        self._scene_data_provider = None
        # Shared, scene-static segmentation lookup builder, created lazily in ``create_render_data``.
        self._seg_mapper: NewtonSegmentationMapper | None = None
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("NewtonWarpRenderer requires an active SimulationContext.")
        self._newton_backend = sim.get_or_create_backend(NewtonReplicateContext, sim, clone_role="scene")
        self._newton_backend.load_visual_shapes = True

    def initialize(self) -> None:
        """Post-physics setup: read the built Newton model and construct the sensor."""
        self._scene_data_provider = SimulationContext.instance().get_scene_data_provider()
        self._newton_model = self._newton_backend.get_model()
        if self._newton_model is None:
            raise RuntimeError(
                "NewtonWarpRenderer requires a Newton model but its native resource has no model. "
                "This usually means the Newton model failed to build from the USD stage "
                "(e.g., unsupported PhysX schemas such as tendons). "
                "Check the log for earlier Newton model build errors."
            )

        self.newton_sensor = newton.sensors.SensorTiledCamera(
            self._newton_model,
            default_render_config=newton.sensors.SensorTiledCamera.RenderConfig(
                enable_textures=self.cfg.enable_textures,
                enable_shadows=self.cfg.enable_shadows,
                enable_ambient_lighting=self.cfg.enable_ambient_lighting,
                enable_backface_culling=self.cfg.enable_backface_culling,
                max_distance=self.cfg.max_distance,
                render_order=newton.sensors.SensorTiledCamera.RenderOrder.TILED,
                tile_width=self.cfg.tile_rendering_width,
                tile_height=self.cfg.tile_rendering_height,
            ),
        )

        if self.cfg.render_order == "pixel_priority":
            self.newton_sensor.default_render_config.render_order = (
                newton.sensors.SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
            )
        elif self.cfg.render_order == "view_priority":
            self.newton_sensor.default_render_config.render_order = (
                newton.sensors.SensorTiledCamera.RenderOrder.VIEW_PRIORITY
            )

    @property
    def visual_material_writer(self):
        """Return the shared Newton model color-writer factory."""
        return self._newton_backend.create_visual_material_writer

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this Newton Warp backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""

        def seg_spec(colorize: bool) -> RenderBufferSpec:
            # Colorized segmentation is RGBA uint8; raw segmentation is a single int32 id channel
            # (matching the Isaac RTX / OVRTX contract so backend-independent consumers see the same dtype).
            return RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.SEMANTIC_SEGMENTATION: seg_spec(self.cfg.colorize_semantic_segmentation),
            RenderBufferKind.INSTANCE_SEGMENTATION: seg_spec(self.cfg.colorize_instance_segmentation),
        }

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Normalize the camera's explicit PPISP cfg before rendering."""
        if spec.cfg.spawn is not None and spec.cfg.spawn.distortion is not None:
            raise NotImplementedError(
                "NewtonWarpRenderer does not implement the requested OpenCV lens-distortion model."
            )
        if spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import normalize_ppisp_cfg
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)

        spec.cfg.isp_cfg = normalize_ppisp_cfg(spec.cfg.isp_cfg)

    def prepare_stage(self, stage: Any, plan: ClonePlan) -> None:
        """Retain the clone plan that owns renderer-visible scene metadata."""
        if plan is None:
            raise ValueError("Newton Warp renderer stage preparation requires an active clone plan.")
        self._clone_plan = plan

    def create_render_data(self, spec: CameraRenderSpec) -> RenderData:
        """Create render data for the Newton tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""

        # Build the shared segmentation mapper and its per-kind lookup tables up-front for all
        # requested segmentation outputs.
        if (
            RenderBufferKind.SEMANTIC_SEGMENTATION in spec.cfg.data_types
            or RenderBufferKind.INSTANCE_SEGMENTATION in spec.cfg.data_types
        ):
            if self._seg_mapper is None:
                self._seg_mapper = NewtonSegmentationMapper(self._newton_model, self._clone_plan, self.cfg)
        if RenderBufferKind.SEMANTIC_SEGMENTATION in spec.cfg.data_types:
            self._seg_mapper.build_mapping(
                RenderBufferKind.SEMANTIC_SEGMENTATION, bool(self.cfg.colorize_semantic_segmentation)
            )
        if RenderBufferKind.INSTANCE_SEGMENTATION in spec.cfg.data_types:
            self._seg_mapper.build_mapping(
                RenderBufferKind.INSTANCE_SEGMENTATION, bool(self.cfg.colorize_instance_segmentation)
            )

        render_data = RenderData(self.newton_sensor, spec, seg_mapper=self._seg_mapper, renderer_cfg=self.cfg)
        self._newton_backend._register_sensor_task(
            render_data.sensor_task_name, lambda: self._launch_render(render_data)
        )
        return render_data

    def set_outputs(self, render_data: RenderData, output_data: dict[str, ProxyArray]):
        """Store output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        render_data.set_outputs(output_data)

    def update(self, render_data: RenderData, intrinsics: ProxyArray) -> None:
        """Request camera transforms through SDP before rendering.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update`."""
        transforms = self._scene_data_provider.request_transforms(
            SceneDataFormat.Transform, name=render_data.transform_stream
        )
        render_data.update(transforms, intrinsics)

    def render(self, render_data: RenderData):
        """Render and write to output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""

        self._newton_backend._update_sensor_tasks(render_data.sensor_task_name)

        # Post-render PPISP: HDR scene-linear → LDR RGBA. Source/destination
        # tensors were bound once in ``set_outputs``.
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data._ppisp_hdr_source,
                render_data._ppisp_rgba_dest,
            )

    def _launch_render(self, render_data: RenderData) -> None:
        """Launch the tiled-camera render kernels for sensor graph capture."""
        # default_render_config is shared state across all Newton sensors, so set max_distance
        # immediately before each render call rather than once in create_render_data.
        self.newton_sensor.default_render_config.max_distance = (
            render_data.far_clip if render_data.far_clip is not None else self.cfg.max_distance
        )

        # Use the renderer's clear value to fill distance_to_camera background when it is the only
        # depth output requested. This avoids a post-render kernel pass for that common case.
        # Planar-depth outputs (depth / distance_to_image_plane) are derived by
        # _convert_plane_depth(), which reads the same ray-depth buffer; pre-clearing to far_clip
        # would make it compute far_clip * cos(θ) per pixel instead of 0.0, so the <= 0.0
        # sentinel that _apply_depth_clipping relies on would no longer identify background pixels.
        _depth_kinds = set(render_data._depth_dests)
        _use_depth_clear = (
            self.cfg.depth_clipping_behavior == "max"
            and render_data.far_clip is not None
            and render_data._RAY_DEPTH_KIND in _depth_kinds
            and not (_depth_kinds & render_data._PLANE_DEPTH_KINDS)
        )

        self.newton_sensor.update(
            self._newton_backend._sensor_state,
            render_data.camera_transforms,
            render_data.camera_rays,
            color_image=render_data.outputs.color_image,
            hdr_color_image=render_data.outputs.hdr_color_image,
            albedo_image=render_data.outputs.albedo_image,
            depth_image=render_data.outputs.depth_image,
            normal_image=render_data.outputs.normals_image,
            shape_index_image=render_data.outputs.shape_index_image,
            # ARGB 93% gray to improve visibility of dark objects and align with RTX renderer background
            clear_data=newton.sensors.SensorTiledCamera.ClearData(
                clear_color=render_data.clear_color,
                **({"clear_depth": render_data.far_clip} if _use_depth_clear else {}),
            ),
            kernel_block_dim=self.cfg.kernel_block_dim,
        )

        if _depth_kinds & render_data._PLANE_DEPTH_KINDS:
            # Derive planar depth from the ray-hit distance, then clip. Deliberately no clear_depth
            # here: the ray-depth buffer feeds convert_plane_depth() (see the _use_depth_clear note).
            render_data._convert_plane_depth()
            render_data._apply_depth_clipping(self.cfg.depth_clipping_behavior)

        # Remap the shape-index buffer into the requested segmentation outputs.
        render_data._convert_segmentation()

    def read_output(self, render_data: RenderData, camera_data: CameraData) -> None:
        """Copy rendered outputs to the camera data buffers.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`."""
        for output_name in camera_data.output:
            if output_name == "rgb":
                continue
            image_data = render_data.get_output(output_name)
            if image_data is not None:
                output_wp = camera_data.output[output_name].warp
                if image_data.ptr != output_wp.ptr:
                    wp.copy(output_wp, image_data)

        # Publish the segmentation id-to-label metadata (idToLabels / idToSemantics) alongside the
        # pixel buffers.
        for output_name, info in render_data.segmentation_info().items():
            camera_data.info[output_name] = info

    def cleanup(self, render_data: RenderData | None):
        """Release resources and drop the camera's sensor task.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            if render_data.sensor_task_name is not None:
                self._newton_backend._unregister_sensor_task(render_data.sensor_task_name)
                render_data.sensor_task_name = None
