# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPISP USD parsing helpers."""

from __future__ import annotations

import inspect

import isaaclab_ppisp
import pytest
from isaaclab_ppisp import PpispCfg, has_ppisp_camera_attrs, normalize_ppisp_cfg, ppisp_cfg_from_usd_camera
from isaaclab_ppisp.cfg import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN

from pxr import Gf, Sdf, Usd, Vt

_PPISP_FLOAT2_ATTRS = {
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
}


def _author_ppisp_attr(camera_prim: Usd.Prim, name: str, value):
    value_type = Sdf.ValueTypeNames.Float2 if name in _PPISP_FLOAT2_ATTRS else Sdf.ValueTypeNames.Float
    attr = camera_prim.CreateAttribute(f"ppisp:{name}", value_type)
    attr.Set(Gf.Vec2f(*value) if name in _PPISP_FLOAT2_ATTRS else value)
    return attr


def _author_camera(stage: Usd.Stage, camera_path: str = "/World/Camera") -> Usd.Prim:
    return stage.DefinePrim(camera_path, "Camera")


def _author_ppisp_camera(
    stage: Usd.Stage,
    camera_path: str = "/World/Camera_ppisp",
    *,
    inherits: str | None = "/World/Camera",
    attrs: dict | None = None,
    controller_weights: list[float] | None = None,
) -> Usd.Prim:
    camera_prim = _author_camera(stage, camera_path)
    if inherits is not None:
        _author_camera(stage, inherits)
        camera_prim.GetInherits().AddInherit(Sdf.Path(inherits))

    for name, value in (attrs or {}).items():
        _author_ppisp_attr(camera_prim, name, value)

    if controller_weights is not None:
        camera_prim.CreateAttribute("ppisp:controllerWeights", Sdf.ValueTypeNames.FloatArray).Set(
            Vt.FloatArray(controller_weights)
        )
    return camera_prim


def _controller_weights() -> list[float]:
    return [0.0] * PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN


def test_ppisp_camera_attr_import_uses_first_time_sample():
    stage = Usd.Stage.CreateInMemory()
    ppisp_camera = _author_ppisp_camera(stage, attrs={})

    exposure = ppisp_camera.CreateAttribute("ppisp:exposureOffset", Sdf.ValueTypeNames.Float)
    exposure.Set(1.0)
    exposure.Set(2.0, 10.0)
    exposure.Set(3.0, 20.0)

    color = ppisp_camera.CreateAttribute("ppisp:colorLatentBlue", Sdf.ValueTypeNames.Float2)
    color.Set(Gf.Vec2f(0.0, 0.0))
    color.Set(Gf.Vec2f(0.1, 0.2), 5.0)

    cfg = ppisp_cfg_from_usd_camera(ppisp_camera)

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentBlue"] == pytest.approx((0.1, 0.2))


def test_normalize_ppisp_cfg_is_explicit_and_fills_defaults():
    cfg = normalize_ppisp_cfg(PpispCfg(inputs={"responsivity": 3.0}))

    assert cfg.inputs["responsivity"] == pytest.approx(3.0)
    assert cfg.inputs["exposureOffset"] == pytest.approx(0.0)
    assert cfg.controller_responsivity == pytest.approx(3.0)


def test_ppisp_has_no_runtime_stage_discovery_surface():
    assert "camera_prim_path" not in PpispCfg.__dataclass_fields__
    assert tuple(inspect.signature(normalize_ppisp_cfg).parameters) == ("ppisp_cfg",)
    for name in ("auto_any_ppisp_cfg", "auto_camera_ppisp_cfg", "ppisp_cfg_from_usd_stage", "resolve_and_normalize"):
        assert not hasattr(isaaclab_ppisp, name)


def test_ppisp_cfg_from_usd_camera_requires_camera_attrs():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_camera(stage)

    with pytest.raises(ValueError, match="expected ppisp:\\* attributes"):
        ppisp_cfg_from_usd_camera(camera)


def test_has_ppisp_camera_attrs_ignores_unknown_ppisp_attrs():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_camera(stage)
    camera.CreateAttribute("ppisp:version", Sdf.ValueTypeNames.String).Set("1")

    assert not has_ppisp_camera_attrs(camera)


def test_ppisp_cfg_from_usd_camera_reads_controller_weights_from_camera_attrs():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_ppisp_camera(
        stage,
        attrs={
            "responsivity": 2.5,
            "vignettingAlpha1R": 0.25,
            "crfToeB": 0.125,
        },
        controller_weights=_controller_weights(),
    )

    cfg = ppisp_cfg_from_usd_camera(camera)

    assert cfg.inputs["responsivity"] == pytest.approx(2.5)
    assert cfg.inputs["vignettingAlpha1R"] == pytest.approx(0.25)
    assert cfg.inputs["crfToeB"] == pytest.approx(0.125)
    assert cfg.controller_prior_exposure == pytest.approx(0.0)
    assert cfg.controller_responsivity == pytest.approx(2.5)
    assert cfg.controller_weights is not None
    assert len(cfg.controller_weights) == PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN


def test_ppisp_cfg_from_usd_camera_validates_controller_weights_len():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_ppisp_camera(
        stage,
        attrs={"responsivity": 2.5},
        controller_weights=[0.0],
    )

    with pytest.raises(ValueError, match=f"Expected {PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}"):
        ppisp_cfg_from_usd_camera(camera)
