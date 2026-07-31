# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer's shared-material color sync into the detached native stage."""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"
)

if not _MISSING_MODULES:
    import numpy as np
    import torch
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer


class _NativeRecorder:
    def __init__(self):
        self.writes = []
        self.resets = 0

    def write_attribute(self, prim_paths, attribute_name, tensor):
        self.writes.append((list(prim_paths), attribute_name, np.array(tensor)))

    def reset(self):
        self.resets += 1


def _renderer(initialized: bool = True):
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._initialized_scene = initialized
    renderer._renderer = _NativeRecorder()
    return renderer


def _write(paths, attr_name, values, semantic="color"):
    from isaaclab.renderers.base_renderer import VisualMaterialWrite

    return VisualMaterialWrite(
        material_paths=list(paths),
        shader_paths=[f"{path}/Shader" for path in paths],
        attr_name=attr_name,
        semantic=semantic,
        values=values,
    )


def test_notify_groups_writes_by_color_attribute():
    renderer = _renderer()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    renderer.notify_visual_material_written(
        [
            _write(["/World/Materials/a", "/World/Materials/c"], "inputs:diffuse_color_constant", colors[[0, 2]]),
            _write(["/World/Materials/b"], "inputs:diffuseColor", colors[[1]]),
        ]
    )

    native = renderer._renderer
    assert len(native.writes) == 2
    omnipbr_paths, omnipbr_attr, omnipbr_colors = native.writes[0]
    assert omnipbr_attr == "inputs:diffuse_color_constant"
    assert omnipbr_paths == ["/World/Materials/a/Shader", "/World/Materials/c/Shader"]
    assert np.allclose(omnipbr_colors, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    preview_paths, preview_attr, preview_colors = native.writes[1]
    assert preview_attr == "inputs:diffuseColor"
    assert preview_paths == ["/World/Materials/b/Shader"]
    assert np.allclose(preview_colors, [[0.0, 1.0, 0.0]])
    assert native.resets == 1


def test_notify_requires_initialized_scene():
    renderer = _renderer(initialized=False)

    with pytest.raises(RuntimeError, match="initialized"):
        renderer.notify_visual_material_written(
            [_write(["/World/Materials/a"], "inputs:diffuse_color_constant", torch.zeros(1, 3))]
        )


def test_notify_validates_color_shape():
    renderer = _renderer()

    with pytest.raises(ValueError, match="rows"):
        renderer.notify_visual_material_written(
            [_write(["/World/Materials/a"], "inputs:diffuse_color_constant", torch.zeros(2, 3))]
        )
    assert renderer._renderer.writes == []


def test_notify_with_no_materials_is_a_no_op():
    renderer = _renderer()

    renderer.notify_visual_material_written([])

    assert renderer._renderer.writes == [] and renderer._renderer.resets == 0


def test_notify_scalar_writes_flatten_to_one_dim():
    renderer = _renderer()

    renderer.notify_visual_material_written(
        [
            _write(
                ["/World/Materials/a", "/World/Materials/b"],
                "inputs:reflection_roughness_constant",
                torch.tensor([[0.1], [0.9]]),
                semantic="scalar",
            )
        ]
    )

    paths, attr, values = renderer._renderer.writes[0]
    assert attr == "inputs:reflection_roughness_constant"
    assert values.shape == (2,)


def test_notify_texture_semantic_is_rejected():
    renderer = _renderer()

    with pytest.raises(RuntimeError, match="texture"):
        renderer.notify_visual_material_written(
            [_write(["/World/Materials/a"], "inputs:diffuse_texture", ["/tmp/tex.png"], semantic="texture")]
        )
    assert renderer._renderer.writes == []
