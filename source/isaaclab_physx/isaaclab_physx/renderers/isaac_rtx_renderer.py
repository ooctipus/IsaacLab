# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RTX renderer using Omniverse Replicator for tiled camera rendering."""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import astuple, dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import warp as wp

from pxr import Sdf

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.cloner import UsdReplicateContext
from isaaclab.cloner.query import env_root_paths
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import enable_extension
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab.utils.warp.kernels import reshape_tiled_image
from isaaclab.utils.warp.warp_math import clamp_depth_to_inf_wp, replace_inf_depth_wp

from .isaac_rtx_renderer_utils import (
    apply_isaac_rtx_determinism_settings,
    apply_isaac_rtx_global_settings,
    ensure_isaac_rtx_render_update,
    ensure_rtx_hydra_engine_attached,
)

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.cloner import ClonePlan

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from omni.replicator.core.scripts.utils.viewport_manager import HydraTexture

    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

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


# RTX simple-shading constants.
#
# Simple shading requires Kit's RTX "Minimal" render mode. Its shading level is
# selected by an integer:
#   0 = No Rendering (black output; only other AOVs are produced)
#   1 = Constant Diffuse (single constant color for all surfaces)
#   2 = Texture Diffuse  (diffuse shading using texture colors)
#   3 = Diffuse/Glossy/Emission (full material shading)
#
# The public data-type names we expose (``simple_shading_*``) are kept stable
# for backwards compatibility and map onto the Kit integer values below.
SIMPLE_SHADING_AOV = "SimpleShadingSD"
SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": 1,
    "simple_shading_diffuse_mdl": 2,
    "simple_shading_full_mdl": 3,
}

# Render-product attributes Kit maps the ``/rtx/rendermode`` and ``/rtx/minimal/mode`` carb
# settings onto (``OmniRtxSettingsCommonAPI_1`` and ``OmniRtxSettingsMinimalAPI_1``). Authoring
# them per render product keeps the process-wide settings — and therefore every other camera and
# the Kit viewport — on their configured render mode.
RTX_RENDER_MODE_ATTR = "omni:rtx:rendermode"
RTX_MINIMAL_MODE_ATTR = "omni:rtx:minimal:mode"
RTX_MINIMAL_RENDER_MODE = "Minimal"


def _camera_semantic_filter_predicate(semantic_filter: str | list[str]) -> str:
    """Build the instance-mapping predicate from the renderer's semantic filter.

    Replicator's semantic/instance segmentation annotators consume this via the synthetic-data pipeline.
    """
    if isinstance(semantic_filter, list):
        return ":*; ".join(semantic_filter) + ":*"
    return semantic_filter


@dataclass
class IsaacRtxRenderData:
    """Render data for Isaac RTX renderer."""

    annotators: dict[str, Any]
    render_product: HydraTexture
    spec: CameraRenderSpec
    output_data: dict[str, ProxyArray] | None = None
    renderer_info: dict[str, Any] = field(default_factory=dict)
    ppisp_pipeline: PpispPipeline | None = None
    """Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set."""
    _hdr_scratch_wp: wp.array | None = None
    """Internal HDR scratch buffer allocated when the user did not request
    ``"rgb_hdr"`` in ``data_types`` but the PPISP pipeline still needs
    somewhere to receive the HDR AOV before LDR conversion."""


class IsaacRtxRenderer(BaseRenderer):
    """Isaac RTX backend using Omniverse Replicator for tiled camera rendering.

    Requires Isaac Sim.
    """

    def __init__(self, cfg: IsaacRtxRendererCfg):
        self.cfg = cfg
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("IsaacRtxRenderer requires an active SimulationContext.")
        self._clone_ctx = sim.get_or_create_backend(UsdReplicateContext, sim.stage, clone_role="scene")
        runtime = sim.get_or_create_backend(_IsaacRtxRuntime, cfg.global_settings)
        if runtime._global_settings != astuple(cfg.global_settings):
            raise ValueError("Isaac RTX global settings must match across cameras because they are process-global.")
        self._camera_prim_paths: list[str] = []

    def initialize(self) -> None:
        """Bind the clone-owned Fabric destinations after physics initializes."""
        sim = SimulationContext.instance()
        plan = sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("IsaacRtxRenderer requires a completed clone plan.")
        self._scene_data_provider = sim.get_scene_data_provider()
        self._point_stream_names = plan.point_stream_names
        self._clone_ctx._prepare_fabric(self._scene_data_provider, sim.cfg.device, plan)

    @property
    def visual_material_writer(self):
        """Write material channels directly through Fabric."""
        return self._clone_ctx.create_fabric_visual_material_writer

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Normalize the explicit PPISP cfg and apply RTX-specific USD overrides.

        When ``spec.cfg.isp_cfg`` is set, pins ``exposure:*`` to neutral and applies
        ``OmniRtxCameraExposureAPI_1`` so RTX's physical-camera exposure model does not
        compound on top of the ISP. Without an ISP, the authored exposure is left alone.

        :attr:`~isaaclab.sensors.camera.CameraCfg.background_color` is applied
        per-render-product in :meth:`create_render_data` via USD attributes.
        """
        self._stage = stage
        self._camera_prim_paths = list(spec.camera_prim_paths)
        if "rgb_hdr" in spec.cfg.data_types or spec.cfg.isp_cfg is not None:
            get_settings_manager().set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
        if spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import apply_rtx_exposure_overrides, normalize_ppisp_cfg
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)

        prototypes = list(spec.camera_source_prim_paths)
        spec.cfg.isp_cfg = normalize_ppisp_cfg(spec.cfg.isp_cfg)
        apply_rtx_exposure_overrides(stage, prototypes)

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output Replicator layout this RTX backend writes.

        ``ALBEDO`` and the three ``SIMPLE_SHADING_*`` outputs require Isaac Sim 6.0+
        and are omitted on older versions. The three segmentation outputs report
        ``RenderBufferSpec(4, uint8)`` when the matching ``self.cfg.colorize_*`` flag is
        set, otherwise ``RenderBufferSpec(1, int32)``.
        """
        sim_major = get_isaac_sim_version().major

        specs: dict[RenderBufferKind, RenderBufferSpec] = {
            # Replicator's native layout for color output is rgba/uint8;
            # ``Camera`` aliases ``rgb`` as a view into ``rgba`` storage.
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }

        if sim_major >= 6:
            specs[RenderBufferKind.ALBEDO] = RenderBufferSpec(4, wp.uint8)
            for shading_type in SIMPLE_SHADING_MODES:
                specs[RenderBufferKind(shading_type)] = RenderBufferSpec(3, wp.uint8)

        seg_specs = (
            (RenderBufferKind.SEMANTIC_SEGMENTATION, self.cfg.colorize_semantic_segmentation),
            (RenderBufferKind.INSTANCE_SEGMENTATION, self.cfg.colorize_instance_segmentation),
            (RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST, self.cfg.colorize_instance_id_segmentation),
        )
        for name, colorize in seg_specs:
            specs[name] = RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return specs

    def prepare_stage(self, stage: Usd.Stage, plan: ClonePlan) -> None:
        """Author per-environment RTX scene partitions.

        Writes the inheriting primvar ``primvars:omni:scenePartition`` on each environment root
        the clone plan describes, and the matching non-primvar
        ``omni:scenePartition`` token on the plan-owned cameras handed over before cloning. RTX
        honors primvar inheritance, so the env-root primvar propagates to all descendant geometry
        and isolates each env's render tile.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`."""

        if plan is None:
            raise ValueError("Isaac RTX stage preparation requires an active clone plan.")

        root_layer = stage.GetRootLayer()
        env_paths = env_root_paths(plan)
        if not env_paths:
            return

        logger.debug("Authoring RTX scene partitions on %d plan-owned environment(s).", len(env_paths))
        partitions = {path: path.name for path in map(Sdf.Path, env_paths)}
        attributes = [
            (path.AppendProperty("primvars:omni:scenePartition"), token) for path, token in partitions.items()
        ]
        for camera_path in map(Sdf.Path, self._camera_prim_paths):
            token = next(
                (partitions[prefix] for prefix in reversed(camera_path.GetPrefixes()) if prefix in partitions), None
            )
            if token is not None:
                attributes.append((camera_path.AppendProperty("omni:scenePartition"), token))

        with Sdf.ChangeBlock():
            for attr_path, token in attributes:
                # Idempotent: a different renderer backend sharing this stage may have already
                # authored this attribute. Re-creating an existing spec raises, so only create
                # it when absent, then (re)assign the per-env token either way.
                attr_spec = root_layer.GetAttributeAtPath(attr_path)
                if attr_spec is None:
                    Sdf.JustCreatePrimAttributeInLayer(
                        root_layer, attr_path, Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform, True
                    )
                    attr_spec = root_layer.GetAttributeAtPath(attr_path)
                attr_spec.default = token

    def create_render_data(self, spec: CameraRenderSpec) -> IsaacRtxRenderData:
        """Create render product and annotators for the tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        import omni.replicator.core as rep

        isaac_sim_version = get_isaac_sim_version()

        simple_shading_mode = None
        needs_color_render = False
        if isaac_sim_version.major >= 6:
            simple_shading_mode = self._resolve_simple_shading_mode(spec)
            needs_color_render = any(
                data_type in spec.cfg.data_types for data_type in ("rgb", "rgba", str(RenderBufferKind.RGB_HDR))
            )
        else:
            unsupported = []
            if "albedo" in spec.cfg.data_types:
                unsupported.append("albedo")
            unsupported.extend(dt for dt in spec.cfg.data_types if dt in SIMPLE_SHADING_MODES)
            if unsupported:
                raise ValueError(
                    "Isaac RTX renderer does not support the following requested data types in"
                    " Isaac Sim versions before 6.0:"
                    f" {unsupported}."
                )

        stage = self._stage
        # The clone plan supplied these exact destinations before replication.
        cam_prim_paths = list(spec.camera_prim_paths)

        # Unique UUID name so concurrent tiled cameras and sequential env create/destroy
        # cycles in one Kit process do not reuse a stale Replicator / SyntheticData activation.
        # ``uuid4().hex`` (no hyphens) prefixed with ``rp_`` is a valid USD identifier.
        # Collision risk is negligible: uuid4 provides 122 random bits, so the birthday-paradox
        # chance among n names is ~n^2 / 2^123 (e.g. ~10^-25 for a million names).
        rp = rep.create.render_product_tiled(
            cameras=cam_prim_paths,
            tile_resolution=(spec.cfg.width, spec.cfg.height),
            name=f"rp_{uuid.uuid4().hex}",
        )

        # Apply background color as per-render-product USD attributes so each render product gets its own
        # background without touching the process-wide /rtx/background carb settings.
        background_color = spec.cfg.background_color
        if background_color is not None:
            edit_layer = stage.GetEditTarget().GetLayer()
            attributes = (
                ("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token, "color"),
                ("omni:rtx:background:source:color", Sdf.ValueTypeNames.Float3, tuple(background_color)),
            )
            with Sdf.ChangeBlock():
                for name, value_type, value in attributes:
                    attr_path = Sdf.Path(rp.path).AppendProperty(name)
                    if edit_layer.GetAttributeAtPath(attr_path) is None:
                        Sdf.JustCreatePrimAttributeInLayer(
                            edit_layer, attr_path, value_type, Sdf.VariabilityUniform, True
                        )
                    edit_layer.GetAttributeAtPath(attr_path).default = value

        # Register simple shading if needed
        if simple_shading_mode is not None:
            rep.AnnotatorRegistry.register_annotator_from_aov(
                aov=SIMPLE_SHADING_AOV, output_data_type=np.uint8, output_channels=4
            )

        needs_hdr_color = str(RenderBufferKind.RGB_HDR) in spec.cfg.data_types or (
            spec.cfg.isp_cfg is not None and any(data_type in ("rgb", "rgba") for data_type in spec.cfg.data_types)
        )
        if needs_hdr_color:
            rep.AnnotatorRegistry.register_annotator_from_aov(
                aov="HdrColor", output_data_type=np.float32, output_channels=4
            )

        # Define annotators based on requested data types
        annotators = {}
        for annotator_type in spec.cfg.data_types:
            if annotator_type == "rgba" or annotator_type == "rgb":
                if spec.cfg.isp_cfg is not None:
                    if str(RenderBufferKind.RGB_HDR) not in annotators:
                        annotator = rep.AnnotatorRegistry.get_annotator(
                            "HdrColor", device=spec.device, do_array_copy=False
                        )
                        annotators[str(RenderBufferKind.RGB_HDR)] = annotator
                else:
                    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=spec.device, do_array_copy=False)
                    annotators["rgba"] = annotator
            elif annotator_type == str(RenderBufferKind.RGB_HDR):
                if str(RenderBufferKind.RGB_HDR) not in annotators:
                    annotator = rep.AnnotatorRegistry.get_annotator("HdrColor", device=spec.device, do_array_copy=False)
                    annotators[str(RenderBufferKind.RGB_HDR)] = annotator
            elif annotator_type == "albedo":
                # TODO: this is a temporary solution because replicator has not exposed the annotator yet
                # once it's exposed, we can remove this
                rep.AnnotatorRegistry.register_annotator_from_aov(
                    aov="DiffuseAlbedoSD", output_data_type=np.uint8, output_channels=4
                )
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "DiffuseAlbedoSD", device=spec.device, do_array_copy=False
                )
                annotators["albedo"] = annotator
            elif annotator_type in SIMPLE_SHADING_MODES:
                annotator = rep.AnnotatorRegistry.get_annotator(
                    SIMPLE_SHADING_AOV, device=spec.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator
            elif annotator_type == "depth" or annotator_type == "distance_to_image_plane":
                # keep depth for backwards compatibility
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane", device=spec.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator
            # note: we are verbose here to make it easier to understand the code.
            #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
            #   if colorize is false, the data is returned as a uint32 image with ids as values.
            else:
                init_params = None
                if annotator_type == "semantic_segmentation":
                    init_params = {
                        "colorize": self.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                    }
                elif annotator_type == "instance_segmentation":
                    init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                elif annotator_type == "instance_id_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}

                # Map the user-facing key to the Replicator annotator name when they differ.
                _REP_ANNOTATOR_NAME = {
                    "instance_segmentation": "instance_segmentation_fast",
                }
                rep_annotator_name = _REP_ANNOTATOR_NAME.get(annotator_type, annotator_type)
                annotator = rep.AnnotatorRegistry.get_annotator(
                    rep_annotator_name, init_params, device=spec.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator

        # Attach annotators to render product
        for annotator in annotators.values():
            annotator.attach([rp.path])

        # Annotator attachment may resynchronize process-wide RTX settings onto the product.
        if simple_shading_mode is not None:
            self._apply_simple_shading_settings(
                stage,
                rp.path,
                simple_shading_mode,
                enable_minimal_render_mode=not needs_color_render,
            )

        ppisp_pipeline = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)

        return IsaacRtxRenderData(
            annotators=annotators,
            render_product=rp,
            spec=spec,
            ppisp_pipeline=ppisp_pipeline,
        )

    def _apply_simple_shading_settings(
        self,
        stage: Usd.Stage,
        render_product_path: str,
        shading_mode: int,
        *,
        enable_minimal_render_mode: bool,
    ) -> None:
        """Configure one render product for the requested simple-shading level.

        Simple shading only becomes cheaper than a full render when the render product's render
        mode is Minimal. Selecting a shading level while the product stays in
        ``RealTimePathTracing`` still pays for the path-tracing pipeline on every frame.

        The shading level is always authored per render product. Minimal render mode is enabled
        only when the product has no regular color output, preserving existing ``rgb``, ``rgba``,
        and ``rgb_hdr`` behavior for mixed requests. These values are not written through their
        process-wide carb settings, so color cameras, the Kit viewport, and
        :func:`~isaaclab_physx.renderers.isaac_rtx_renderer_utils.apply_isaac_rtx_determinism_settings`
        keep path tracing, and so cameras requesting different shading levels do not overwrite
        each other.

        Args:
            stage: Stage owning the render product.
            render_product_path: Prim path of the render product to configure.
            shading_mode: Minimal shading level, one of the values in :data:`SIMPLE_SHADING_MODES`.
            enable_minimal_render_mode: Whether to switch the render product to RTX Minimal mode.
        """
        layer = stage.GetSessionLayer()
        attributes = [(RTX_MINIMAL_MODE_ATTR, Sdf.ValueTypeNames.Int, shading_mode)]
        if enable_minimal_render_mode:
            attributes.append((RTX_RENDER_MODE_ATTR, Sdf.ValueTypeNames.Token, RTX_MINIMAL_RENDER_MODE))
        with Sdf.ChangeBlock():
            for name, value_type, value in attributes:
                path = Sdf.Path(render_product_path).AppendProperty(name)
                if layer.GetAttributeAtPath(path) is None:
                    Sdf.JustCreatePrimAttributeInLayer(layer, path, value_type, Sdf.VariabilityUniform, True)
                layer.GetAttributeAtPath(path).default = value

    def _resolve_simple_shading_mode(self, spec: CameraRenderSpec) -> int | None:
        """Resolve the requested simple shading mode from data types."""
        requested = [dt for dt in spec.cfg.data_types if dt in SIMPLE_SHADING_MODES]
        if not requested:
            return None
        if len(requested) > 1:
            raise ValueError(f"Multiple simple shading modes requested: {requested}.")
        return SIMPLE_SHADING_MODES[requested[0]]

    def set_outputs(self, render_data: IsaacRtxRenderData, output_data: dict[str, ProxyArray]):
        """Store reference to output buffers for writing during render.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGBA) not in output_data:
            raise ValueError(
                "Isaac RTX renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                " Camera.cfg.data_types when isp_cfg is set."
            )
        render_data.output_data = output_data
        # Allocate an internal HDR scratch buffer when PPISP is composed but
        # the user did not request the raw HDR AOV in ``data_types`` — the
        # PPISP kernel still needs somewhere to receive the HDR annotator
        # output before LDR conversion.
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGB_HDR) not in output_data:
            spec = render_data.spec
            hdr_spec = self.supported_output_types()[RenderBufferKind.RGB_HDR]
            assert hdr_spec.dtype is wp.float32
            render_data._hdr_scratch_wp = wp.zeros(
                (spec.num_instances, spec.cfg.height, spec.cfg.width, hdr_spec.channels),
                dtype=wp.float32,
                device=spec.device,
            )

    def update(self, render_data: IsaacRtxRenderData, intrinsics: ProxyArray) -> None:
        """Ask SDP to update the stage and camera before drawing it.

        Isaac RTX draws the USD stage rather than a scene of its own, so it does not copy state --
        it asks for the stage it is about to read in the format Fabric holds. The provider converts
        only when the backend's data has changed, so the Kit viewport asking for the same frame
        costs nothing.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update`."""
        self._scene_data_provider.request_transforms(SceneDataFormat.FabricMatrix44)
        self._clone_ctx._update_fabric_hierarchy()
        self._scene_data_provider.request_transforms(
            SceneDataFormat.FabricMatrix44, name=render_data.spec.cfg.prim_path
        )
        self._clone_ctx._update_fabric_hierarchy()
        for stream in self._point_stream_names:
            self._scene_data_provider.request_points(SceneDataFormat.FabricMeshPoints, stream)

    def render(self, render_data: IsaacRtxRenderData):
        """Extract data from annotators and write to output buffers.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""
        spec = render_data.spec
        output_data = render_data.output_data
        if output_data is None:
            raise RuntimeError("Isaac RTX outputs must be set before rendering.")

        if any("segmentation" in data_type for data_type in spec.cfg.data_types):
            from omni.syntheticdata import SyntheticData

            SyntheticData.Get().set_instance_mapping_semantic_filter(
                _camera_semantic_filter_predicate(self.cfg.semantic_filter)
            )

        # Pump RTX when this camera requests a fresh frame.
        ensure_isaac_rtx_render_update()

        view_count = spec.num_instances
        cfg = spec.cfg
        device = spec.device

        num_tiles_x = math.ceil(math.sqrt(view_count))

        # Extract the flattened image buffer
        for data_type, annotator in render_data.annotators.items():
            # check whether returned data is a dict (used for segmentation)
            output = annotator.get_data()
            if isinstance(output, dict):
                tiled_data_buffer = output["data"]
                render_data.renderer_info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            # convert data buffer to warp array
            if isinstance(tiled_data_buffer, np.ndarray):
                # Let warp infer the dtype from numpy array instead of hardcoding uint8
                # Different annotators return different dtypes: RGB(uint8), depth(float32), segmentation(uint32)
                tiled_data_buffer = wp.array(tiled_data_buffer, device=device)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=device)

            # process data for different segmentation types
            # Note: Replicator returns raw buffers of dtype uint32 for segmentation types
            #   so we need to convert them to uint8 4 channel images for colorized types
            if (
                (data_type == "semantic_segmentation" and self.cfg.colorize_semantic_segmentation)
                or (data_type == "instance_segmentation" and self.cfg.colorize_instance_segmentation)
                or (data_type == "instance_id_segmentation_fast" and self.cfg.colorize_instance_id_segmentation)
            ):
                tiled_data_buffer = wp.array(
                    ptr=tiled_data_buffer.ptr, shape=(*tiled_data_buffer.shape, 4), dtype=wp.uint8, device=device
                )

            # For motion vectors, use specialized kernel that reads 4 channels but only writes 2
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/2003)
            if data_type == "motion_vectors":
                tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()

            # For normals, we only require the first three channels of the tiled buffer
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/4239)
            if data_type == "normals":
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()
            if data_type in SIMPLE_SHADING_MODES:
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()
            if data_type == str(RenderBufferKind.RGB_HDR):
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()

            # The HDR annotator's destination is the user-visible ``output_data["rgb_hdr"]``
            # when they requested it explicitly; otherwise the renderer's internal
            # scratch buffer that the PPISP pipeline reads.
            if data_type == str(RenderBufferKind.RGB_HDR) and data_type not in output_data:
                assert render_data._hdr_scratch_wp is not None
                buf_wp = render_data._hdr_scratch_wp
            else:
                buf_wp = output_data[data_type].warp
            wp.launch(
                kernel=reshape_tiled_image,
                dim=(view_count, cfg.height, cfg.width),
                inputs=[
                    tiled_data_buffer.flatten(),
                    buf_wp,
                    *list(buf_wp.shape[1:]),
                    num_tiles_x,
                ],
                device=device,
            )

            # rgb is a strided warp view into rgba set up in CameraData.allocate();
            # no per-frame alias assignment needed.

            # NOTE: The `distance_to_camera` annotator returns the distance to the camera optical center.
            #       However, the replicator depth clipping is applied w.r.t. to the image plane which may result
            #       in values larger than the clipping range in the output. We apply an additional clipping to
            #       ensure values are within the clipping range for all the annotators.
            if data_type == "distance_to_camera":
                clamp_depth_to_inf_wp(buf_wp, cfg.spawn.clipping_range[1], device=device)

            # apply defined clipping behavior
            if (
                data_type in ("distance_to_camera", "distance_to_image_plane", "depth")
                and self.cfg.depth_clipping_behavior != "none"
            ):
                replacement = 0.0 if self.cfg.depth_clipping_behavior == "zero" else cfg.spawn.clipping_range[1]
                replace_inf_depth_wp(buf_wp, replacement, device=device)

        # Post-render PPISP: HDR scene-linear → LDR RGBA. The camera enforces
        # that ``rgba`` (or ``rgb`` aliasing into it) is present when an ISP is
        # configured, so writing to ``output_data["rgba"]`` is safe.
        if render_data.ppisp_pipeline is not None:
            hdr_proxy = output_data.get(str(RenderBufferKind.RGB_HDR))
            hdr_source = hdr_proxy.warp if hdr_proxy is not None else render_data._hdr_scratch_wp
            render_data.ppisp_pipeline.apply(hdr_source, output_data[str(RenderBufferKind.RGBA)].warp)

    def read_output(self, render_data: IsaacRtxRenderData, camera_data: CameraData) -> None:
        """Populate per-output metadata collected during render(). Pixel data already written in render().

        This is a *replace*, not a *merge*: every seeded output key is reset to this frame's metadata,
        which is ``None`` when its annotator produced none (``renderer_info`` only ever holds a subset of
        the outputs, so iterating ``camera_data.info`` both preserves its ``output``-mirroring key set and
        resets any metadata that went away to ``None``). Without this, a stale mapping from a previous frame
        would linger once an annotator stops emitting one.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`."""
        assert camera_data.info is not None, "CameraData.info should be created in CameraData.allocate"
        for output_name in camera_data.info:
            camera_data.info[output_name] = render_data.renderer_info.get(output_name)

    def cleanup(self, render_data: IsaacRtxRenderData | None):
        """Detach annotators, destroy the owned tiled render product, and drop held refs.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data is None:
            return

        for annotator in render_data.annotators.values():
            annotator.detach([render_data.render_product.path])

        render_data.render_product.destroy()

        render_data.annotators.clear()
        render_data.output_data = None
        render_data.renderer_info.clear()
        render_data.ppisp_pipeline = None
        render_data._hdr_scratch_wp = None


class _IsaacRtxRuntime:
    """Simulation-owned process-global RTX setup shared by every camera client."""

    def __init__(self, global_settings: IsaacRtxRendererGlobalSettingsCfg):
        self._global_settings = astuple(global_settings)
        # Load Replicator only when selected; Kit experiences otherwise resolve its bundled Warp at startup.
        enable_extension("omni.replicator.core")
        self._settings = get_settings_manager()
        self._settings.set_bool("/isaaclab/render/rtx_sensors", True)
        apply_isaac_rtx_global_settings(global_settings, self._settings)
        if self._settings.get("/isaaclab/render/deterministic", False):
            apply_isaac_rtx_determinism_settings(self._settings)
        ensure_rtx_hydra_engine_attached()
