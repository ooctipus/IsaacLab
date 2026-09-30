# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product and scene partition authoring."""

from __future__ import annotations

import importlib.util
import re

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_usd import (  # noqa: E402
        build_render_product_as_string,
        build_render_scope_usd,
    )
else:
    build_render_product_as_string = None
    build_render_scope_usd = None


def _render_vars(data_types: list[str]) -> list[str]:
    """Read the ordered render-variable relationship from a generated product."""
    product, _ = build_render_product_as_string(
        width=16,
        height=8,
        num_envs=1,
        data_types=data_types,
        camera_prim_path="/World/source/Camera",
    )
    relationship = re.search(r"rel orderedVars = \[(.*)\]", product)
    assert relationship is not None
    return re.findall(r"</Render/Vars/([^>]+)>", relationship.group(1))


def test_build_render_scope_usd_default_background_is_dome_light():
    """Default background (background_color=None) uses domeLight source type."""
    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_vars=["LdrColor"],
        tiled_width=16,
        tiled_height=8,
    )
    assert 'token omni:rtx:background:source:type = "domeLight"' in render_scope
    assert "omni:rtx:background:source:color" not in render_scope


def test_build_render_scope_usd_solid_background_color():
    """Providing background_color emits color source type and the color attribute."""
    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_vars=["LdrColor"],
        tiled_width=16,
        tiled_height=8,
        background_color=(1.0, 0.0, 0.5),
    )
    assert 'token omni:rtx:background:source:type = "color"' in render_scope
    assert "color3f omni:rtx:background:source:color = (1.0, 0.0, 0.5)" in render_scope
    assert 'token omni:rtx:background:source:type = "domeLight"' not in render_scope


@pytest.mark.parametrize(
    ("data_types", "expected"),
    [
        (["rgb_hdr"], ["HdrColor"]),
        (["motion_vectors"], ["TargetMotionSD"]),
        (["rgb", "motion_vectors"], ["LdrColor", "TargetMotionSD"]),
        (["depth", "distance_to_image_plane"], ["DistanceToImagePlaneSD"]),
    ],
)
def test_ovrtx_render_vars_follow_requested_outputs_once(data_types, expected):
    """Requested outputs compose stable render variables; aliases share one native AOV."""
    assert _render_vars(data_types) == expected


def test_ovrtx_render_product_rejects_empty_output_request():
    """The USD builder never invents an RGB request for an empty interface."""
    with pytest.raises(ValueError, match="at least one requested output"):
        _render_vars([])


def test_render_product_initially_targets_the_planned_source_camera():
    """The initial relationship uses the exact prototype path supplied by the clone plan."""
    render_product, render_product_path = build_render_product_as_string(
        width=16,
        height=8,
        num_envs=4,
        data_types=["rgb"],
        camera_prim_path="/World/prototypes/red/Robot/head_cam",
    )

    assert render_product_path == "/Render/RenderProduct"
    assert "rel camera = [</World/prototypes/red/Robot/head_cam>]" in render_product
    assert "/World/envs/env_0" not in render_product
    assert "uniform int2 resolution = (32, 16)" in render_product


def test_ovrtx_rgb_and_rgb_hdr_author_both_render_vars():
    """Requesting LDR RGB and RGB_HDR keeps both OVRTX render variables."""
    render_vars = _render_vars(["rgb", "rgb_hdr"])

    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_vars=render_vars,
        tiled_width=16,
        tiled_height=8,
    )

    assert "rel orderedVars = [</Render/Vars/LdrColor>, </Render/Vars/HdrColor>]" in render_scope
    assert 'def RenderVar "LdrColor"' in render_scope
    assert 'def RenderVar "HdrColor"' in render_scope


def test_ovrtx_semantic_segmentation_authors_semantic_and_id_map_render_vars():
    """Requesting semantic segmentation authors both SemanticSegmentation and SemanticIdMap render vars."""
    render_vars = _render_vars(["semantic_segmentation"])
    assert render_vars == ["SemanticSegmentation", "SemanticIdMap"]

    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_vars=render_vars,
        tiled_width=16,
        tiled_height=8,
    )

    assert "rel orderedVars = [</Render/Vars/SemanticSegmentation>, </Render/Vars/SemanticIdMap>]" in render_scope
    assert 'uniform string sourceName = "SemanticSegmentation"' in render_scope
    assert 'uniform string sourceName = "SemanticIdMap"' in render_scope


def test_ovrtx_instance_segmentation_authors_pixel_and_map_render_vars():
    """Requesting instance segmentation authors the pixel AOV plus the three ID/label map render vars."""
    render_vars = _render_vars(["instance_segmentation"])
    assert render_vars == [
        "NonStableInstanceSegmentation",
        "StableIdSemanticIdMap",
        "StableIdMap",
        "SemanticIdMap",
    ]

    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_vars=render_vars,
        tiled_width=16,
        tiled_height=8,
    )

    assert (
        "rel orderedVars = [</Render/Vars/NonStableInstanceSegmentation>, </Render/Vars/StableIdSemanticIdMap>,"
        " </Render/Vars/StableIdMap>, </Render/Vars/SemanticIdMap>]" in render_scope
    )
    assert 'uniform string sourceName = "StableIdSemanticIdMap"' in render_scope
    assert 'uniform string sourceName = "StableIdMap"' in render_scope


def test_ovrtx_semantic_and_instance_segmentation_share_a_single_semantic_id_map():
    """Requesting both segmentation outputs authors ``SemanticIdMap`` exactly once (it is shared)."""
    render_vars = _render_vars(["semantic_segmentation", "instance_segmentation"])
    assert render_vars.count("SemanticIdMap") == 1
    assert {"SemanticIdMap", "StableIdSemanticIdMap", "StableIdMap"} <= set(render_vars)


def test_ovrtx_rgb_composes_every_requested_pixel_aov():
    """RGB does not suppress independently requested depth, albedo, normals, or motion outputs."""
    assert _render_vars(["rgb", "depth", "albedo", "normals", "motion_vectors"]) == [
        "LdrColor",
        "DistanceToImagePlaneSD",
        "DiffuseAlbedoSD",
        "NormalSD",
        "TargetMotionSD",
    ]
