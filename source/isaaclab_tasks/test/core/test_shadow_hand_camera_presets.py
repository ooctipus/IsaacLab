# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Shadow Hand camera task semantics and preset resolution."""

import inspect
import types

import pytest
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers import RendererCfg
from isaaclab.sensors import CameraCfg

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import (
    ShadowHandCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(data_types: list[str], feature_extractor_enabled: bool = True):
    """Build a minimal mock cfg with a :meth:`validate_config` method.

    The mock reuses the real validation logic from :class:`ShadowHandCameraEnvCfg`.
    """
    cfg = types.SimpleNamespace()
    cfg.scene = types.SimpleNamespace(camera=CameraCfg(prim_path="/Camera", renderer_cfg=None, data_types=data_types))
    cfg.feature_extractor = types.SimpleNamespace(enabled=feature_extractor_enabled)
    cfg.validate_config = lambda: ShadowHandCameraEnvCfg.validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Task-specific validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_types", [["rgb"], ["albedo"], ["rgb", "depth"], ["depth"]])
def test_camera_data_must_match_feature_extractor_semantics(data_types):
    cfg = _make_cfg(data_types, feature_extractor_enabled=data_types != ["depth"])
    cfg.validate_config()


@pytest.mark.parametrize("data_types", [["depth"], ["distance_to_camera"], ["distance_to_image_plane"]])
def test_depth_only_camera_rejects_feature_extractor(data_types):
    cfg = _make_cfg(data_types)
    with pytest.raises(ValueError, match="Depth-only"):
        cfg.validate_config()


def test_manager_camera_features_declare_shape_without_probe_workaround():
    """Actor and critic camera features must not execute during manager construction."""
    assert mdp.ShadowHandCameraFeatures._output_shape == (27,)
    assert mdp.shadow_hand_camera_cached_features._output_shape == (27,)
    assert "_shape_probe_pending" not in inspect.getsource(mdp.ShadowHandCameraFeatures)


# ---------------------------------------------------------------------------
# Preset resolution — camera data types
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shadow_hand_camera_presets():
    """Collect all presets from ShadowHandCameraEnvCfg once for the module."""
    return collect_presets(ShadowHandCameraEnvCfg())


_CAMERA_DATA_TYPE_PRESETS = [
    # preset_name, expected_data_types
    ("default", ["rgb", "depth", "semantic_segmentation"]),
    ("full", ["rgb", "depth", "semantic_segmentation"]),
    ("rgb", ["rgb"]),
    ("albedo", ["albedo"]),
    ("simple_shading_constant_diffuse", ["simple_shading_constant_diffuse"]),
    ("simple_shading_diffuse_mdl", ["simple_shading_diffuse_mdl"]),
    ("simple_shading_full_mdl", ["simple_shading_full_mdl"]),
    ("depth", ["depth"]),
]


@pytest.mark.parametrize("preset_name,expected_data_types", _CAMERA_DATA_TYPE_PRESETS)
def test_camera_presets_resolve_to_valid_configs(shadow_hand_camera_presets, preset_name, expected_data_types):
    """Camera presets must be discoverable, request data, and have valid dimensions."""
    camera_presets = shadow_hand_camera_presets["scene.camera"]
    assert preset_name in camera_presets, f"Preset '{preset_name}' not found in camera presets"
    resolved = camera_presets[preset_name]
    assert resolved.data_types == expected_data_types, (
        f"Preset '{preset_name}': expected data_types={expected_data_types}, got {resolved.data_types}"
    )
    assert len(resolved.data_types) > 0, (
        f"Camera preset '{preset_name}' has an empty data_types list — nothing would be rendered."
    )
    assert resolved.width > 0, f"Camera preset '{preset_name}' has non-positive width: {resolved.width}"
    assert resolved.height > 0, f"Camera preset '{preset_name}' has non-positive height: {resolved.height}"


# ---------------------------------------------------------------------------
# Preset resolution — renderer
# ---------------------------------------------------------------------------

_RENDERER_PRESETS = [
    # preset_name, expected_class
    ("default", NewtonWarpRendererCfg),
    ("isaacsim_rtx", IsaacRtxRendererCfg),
    ("newton_renderer", NewtonWarpRendererCfg),
]


@pytest.mark.parametrize("preset_name,expected_class", _RENDERER_PRESETS)
def test_renderer_presets_resolve_to_expected_configs(shadow_hand_camera_presets, preset_name, expected_class):
    """Renderer presets must resolve to the expected configuration and renderer type."""
    renderer_presets = shadow_hand_camera_presets["scene.camera.renderer_cfg"]
    assert preset_name in renderer_presets, f"Preset '{preset_name}' not found in renderer presets"
    resolved = renderer_presets[preset_name]
    assert isinstance(resolved, expected_class), (
        f"Renderer preset '{preset_name}': expected {expected_class.__name__}, got {type(resolved).__name__}"
    )
    if preset_name == "newton_renderer":
        assert resolved.renderer_type == "newton_warp"

    rtx_cfg = renderer_presets["rtx"]
    assert isinstance(rtx_cfg, RendererCfg)
    assert rtx_cfg.renderer_type == "auto_rtx"
