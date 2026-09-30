# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD text construction for OVRTX render products."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

_OUTPUT_RENDER_VAR = {
    "rgb": "LdrColor",
    "rgba": "LdrColor",
    "rgb_hdr": "HdrColor",
    "albedo": "DiffuseAlbedoSD",
    "simple_shading_constant_diffuse": "LdrColor",
    "simple_shading_diffuse_mdl": "LdrColor",
    "simple_shading_full_mdl": "LdrColor",
    "semantic_segmentation": "SemanticSegmentation",
    "instance_segmentation": "NonStableInstanceSegmentation",
    "depth": "DistanceToImagePlaneSD",
    "distance_to_image_plane": "DistanceToImagePlaneSD",
    "distance_to_camera": "DistanceToCameraSD",
    "normals": "NormalSD",
    "motion_vectors": "TargetMotionSD",
}
_OUTPUT_RENDER_VAR_DEPENDENCIES = {
    "semantic_segmentation": ("SemanticIdMap",),
    "instance_segmentation": ("StableIdSemanticIdMap", "StableIdMap", "SemanticIdMap"),
}

_RENDER_VAR_PRIM_PATH_BY_SOURCE: Mapping[str, str] = MappingProxyType(
    {
        source: f"/Render/Vars/{source}"
        for source in dict.fromkeys(
            (
                *_OUTPUT_RENDER_VAR.values(),
                *(source for values in _OUTPUT_RENDER_VAR_DEPENDENCIES.values() for source in values),
            )
        )
    }
)


def render_var_prim_paths_by_source() -> Mapping[str, str]:
    """Return the canonical RenderVar prim path for every authored source."""
    return _RENDER_VAR_PRIM_PATH_BY_SOURCE


def build_render_scope_usd(
    camera_paths: list[str],
    render_product_name: str,
    render_vars: list[str],
    tiled_width: int,
    tiled_height: int,
    minimal_mode: int | None = None,
    background_color: tuple[float, float, float] | None = None,
    device_id: int | None = None,
    enable_shadows: bool = False,
    render_scope_name: str = "Render",
) -> str:
    """Build the Render scope USD string (def Scope Render, RenderProduct, Vars).

    Args:
        camera_paths: List of camera prim paths.
        render_product_name: Name of the render product.
        render_vars: Ordered OVRTX source names to author and attach.
        tiled_width: Width of the tiled image.
        tiled_height: Height of the tiled image.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        background_color: Solid background color as normalized RGB floats ``(r, g, b)`` in ``[0, 1]``.
            When set, the render product uses a solid color background instead of the dome light.
            When ``None``, the default dome-light background is used.
        device_id: CUDA device index assigned to the render product. ``None`` lets OVRTX choose.
        enable_shadows: Whether RTX Minimal mode casts shadows.
        render_scope_name: Name of the scope that owns this product and its render variables.

    Returns:
        The USD string for the render scope.
    """
    camera_rel_list = ", ".join([f"<{p}>" for p in camera_paths])
    device_ids_line = "" if device_id is None else f"\n        uint[] deviceIds = [{device_id}]"

    if background_color is None:
        bg_type_line = 'token omni:rtx:background:source:type = "domeLight"'
    else:
        r, g, b = background_color
        bg_type_line = (
            f'token omni:rtx:background:source:type = "color"\n'
            f"        color3f omni:rtx:background:source:color = ({r}, {g}, {b})"
        )

    if minimal_mode is None:
        render_mode_lines = ['token omni:rtx:rendermode = "RealTimePathTracing"']
    else:
        render_mode_lines = [
            'token omni:rtx:rendermode = "Minimal"',
            f"int omni:rtx:minimal:mode = {minimal_mode}",
            f"bool omni:rtx:minimal:castShadows = {'true' if enable_shadows else 'false'}",
        ]

    render_mode_block = "\n        ".join(render_mode_lines)
    ordered_vars = ", ".join(f"</{render_scope_name}/Vars/{source}>" for source in render_vars)
    render_var_defs = "\n".join(
        f'''        def RenderVar "{source}"
        {{
            uniform string sourceName = "{source}"
        }}'''
        for source in render_vars
    )

    return f'''
def Scope "{render_scope_name}"
{{
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{camera_rel_list}]{device_ids_line}
        {bg_type_line}
        float omni:rtx:rt:ambientLight:intensity = 1.0
        {render_mode_block}
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [{ordered_vars}]
        uniform int2 resolution = ({tiled_width}, {tiled_height})
    }}

    def "Vars"
    {{
{render_var_defs}
    }}
}}
'''


def _tiled_resolution(num_envs: int, width: int, height: int) -> tuple[int, int]:
    """Compute tiled width and height from env count and per-env resolution (same as Camera)."""
    num_cols = math.ceil(math.sqrt(num_envs))
    num_rows = math.ceil(num_envs / num_cols)
    return num_cols * width, num_rows * height


def build_render_product_as_string(
    width: int,
    height: int,
    num_envs: int,
    data_types: list[str],
    camera_prim_path: str,
    minimal_mode: int | None = None,
    background_color: tuple[float, float, float] | None = None,
    device_id: int | None = None,
    enable_shadows: bool = False,
    render_scope_name: str = "Render",
) -> tuple[str, str]:
    """Build the render product USD snippet as a string.

    This string is meant to be appended to an exported stage (ASCII) before loading into OVRTX.
    The initial camera relationship targets one prototype the clone plan names. Multi-environment
    rendering rewrites the relationship with the plan's exact destination paths after cloning.

    Args:
        width: Tile width from sensor config [px].
        height: Tile height from sensor config [px].
        num_envs: Number of environments from scene.
        data_types: Data types from sensor config.
        camera_prim_path: Absolute path of a camera prototype named by the clone plan.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        background_color: Solid background color as normalized RGB floats ``(r, g, b)`` in ``[0, 1]``.
            When set, the render product uses a solid color background instead of the dome light.
            When ``None``, the default dome-light background is used.
        device_id: CUDA device index assigned to the render product. ``None`` lets OVRTX choose.
        enable_shadows: Whether RTX Minimal mode casts shadows.
        render_scope_name: Name of the scope that owns this product and its render variables.

    Returns:
        Tuple of (render product USD snippet as a string, absolute render product prim path).
    """
    if not data_types:
        raise ValueError("OVRTX render products require at least one requested output.")
    unknown = [data_type for data_type in data_types if data_type not in _OUTPUT_RENDER_VAR]
    if unknown:
        raise ValueError(f"OVRTX does not define render variables for: {unknown}")
    primary = [_OUTPUT_RENDER_VAR[data_type] for data_type in data_types]
    dependencies = [source for data_type in data_types for source in _OUTPUT_RENDER_VAR_DEPENDENCIES.get(data_type, ())]
    render_vars = list(dict.fromkeys((*primary, *dependencies)))
    tiled_width, tiled_height = _tiled_resolution(num_envs, width, height)

    render_product_name = "RenderProduct"
    render_product_path = f"/{render_scope_name}/{render_product_name}"

    camera_content = build_render_scope_usd(
        [camera_prim_path],
        render_product_name,
        render_vars,
        tiled_width,
        tiled_height,
        minimal_mode=minimal_mode,
        background_color=background_color,
        device_id=device_id,
        enable_shadows=enable_shadows,
        render_scope_name=render_scope_name,
    )
    return camera_content, render_product_path
