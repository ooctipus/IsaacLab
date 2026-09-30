# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP configuration and USD parsing helpers.

The implementation follows the physically plausible ISP model described in
https://arxiv.org/abs/2601.18336.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

from isaaclab.utils.configclass import configclass

PPISP_ATTR_NAMESPACE = "ppisp:"
"""Namespace prefix for authoritative PPISP attributes authored on a USD camera."""

PPISP_CONTROLLER_WEIGHTS_CAMERA_ATTR = "controllerWeights"
"""Camera ``ppisp:*`` attribute name containing flattened controller weights."""

PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN = 241_961
"""Flattened element count of the camera-authored controller weight array exported by NRE.

This is a frozen architectural constant tied to the exported controller network
shape (see :mod:`isaaclab_ppisp.kernels` for the offset layout). USD parsing
and Warp execution validate against it and fail loudly on a mismatch.
"""

PPISP_FLOAT2_INPUTS = {
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
}

PPISP_DEFAULT_INPUTS: dict[str, float | tuple[float, float]] = {
    "responsivity": 1.0,
    "exposureOffset": 0.0,
    "vignettingCenterR": (0.0, 0.0),
    "vignettingAlpha1R": 0.0,
    "vignettingAlpha2R": 0.0,
    "vignettingAlpha3R": 0.0,
    "vignettingCenterG": (0.0, 0.0),
    "vignettingAlpha1G": 0.0,
    "vignettingAlpha2G": 0.0,
    "vignettingAlpha3G": 0.0,
    "vignettingCenterB": (0.0, 0.0),
    "vignettingAlpha1B": 0.0,
    "vignettingAlpha2B": 0.0,
    "vignettingAlpha3B": 0.0,
    "colorLatentBlue": (0.0, 0.0),
    "colorLatentRed": (0.0, 0.0),
    "colorLatentGreen": (0.0, 0.0),
    "colorLatentNeutral": (0.0, 0.0),
    "crfToeR": 0.013659,
    "crfShoulderR": 0.013659,
    "crfGammaR": 0.378165,
    "crfCenterR": 0.0,
    "crfToeG": 0.013659,
    "crfShoulderG": 0.013659,
    "crfGammaG": 0.378165,
    "crfCenterG": 0.0,
    "crfToeB": 0.013659,
    "crfShoulderB": 0.013659,
    "crfGammaB": 0.378165,
    "crfCenterB": 0.0,
}


def default_ppisp_inputs() -> dict[str, float | tuple[float, float]]:
    """Return a copy of the PPISP identity/default input dictionary."""
    return dict(PPISP_DEFAULT_INPUTS)


@configclass
class PpispCfg:
    """Configuration for PPISP post-processing.

    PPISP inputs are static in IsaacLab. NRE exports store the authoritative
    values on a USD camera as ``ppisp:*`` attributes. If animated USD
    attributes are imported, the first authored time sample is used and later
    samples are ignored.
    """

    inputs: dict[str, float | tuple[float, float]] = field(default_factory=default_ppisp_inputs)
    """Flat PPISP values keyed by PPISP parameter name.

    Coordinate conventions for spatial inputs:

    * ``vignettingCenter{R,G,B}`` is a 2D offset in UV space normalised by
      ``max(width, height)`` with the image center at ``(0.0, 0.0)``. The
      :data:`PPISP_DEFAULT_INPUTS` defaults place every channel's
      optical center at the image center.
    * Radial vignetting coefficients ``vignettingAlpha{1,2,3}{R,G,B}`` are
      polynomial coefficients in the same normalised radius, applied as
      ``factor = clamp(1 + a1*r^2 + a2*r^4 + a3*r^6, 0, 1)``; with a square
      frame the image corners sit at ``r^2 = 0.5``.
    """

    controller_prior_exposure: float = 0.0
    """Controller prior exposure [EV] used by the native controller path."""

    controller_responsivity: float | None = None
    """Controller feature-extraction responsivity [dimensionless].

    When ``None``, the controller uses the static PPISP ``responsivity`` camera
    attribute so feature extraction sees the same responsivity-scaled HDR
    radiance as the image PPISP transform.
    """

    controller_weights: tuple[float, ...] | None = None
    """Flattened controller weights.

    USD imports read these from the camera's ``ppisp:controllerWeights``
    attribute. When present, the native controller predicts ``exposureOffset``
    and the four color latents from the HDR image each frame. Static PPISP
    inputs still provide responsivity, vignetting, and CRF.
    """


def normalize_ppisp_cfg(ppisp_cfg: PpispCfg | None) -> PpispCfg | None:
    """Normalise a :class:`PpispCfg` for downstream consumption.

    * If ``ppisp_cfg`` is ``None``, returns ``None``.
    * Otherwise validates ``ppisp_cfg.inputs`` and fills in defaults.
    """
    if ppisp_cfg is None:
        return None
    if not isinstance(ppisp_cfg, PpispCfg):
        raise TypeError(f"Unsupported PPISP configuration type: {type(ppisp_cfg)!r}")
    ppisp_cfg.inputs = _normalized_inputs(ppisp_cfg.inputs)
    _finalize_ppisp_cfg(ppisp_cfg)
    return ppisp_cfg


def ppisp_cfg_from_usd_camera(camera_prim: Any) -> PpispCfg:
    """Create :class:`PpispCfg` from a USD camera prim.

    PPISP values are read from camera ``ppisp:*`` attributes. Animated
    attributes are collapsed to their first authored time sample.
    """
    values = _read_ppisp_inputs_from_camera(camera_prim)
    controller_weights = _read_controller_weights_from_camera(camera_prim)
    if values is None and controller_weights is None:
        camera_path = str(camera_prim.GetPath()) if camera_prim and camera_prim.IsValid() else "<none>"
        raise ValueError(
            f"PPISP camera attributes were not found on camera {camera_path}; expected ppisp:* attributes."
        )

    cfg = PpispCfg(inputs=values or default_ppisp_inputs(), controller_weights=controller_weights)
    _finalize_ppisp_cfg(cfg)
    return cfg


def _normalized_inputs(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = default_ppisp_inputs()
    for input_name, value in inputs.items():
        if input_name not in values:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _normalize_input_value(input_name: str, value: Any) -> float | tuple[float, float]:
    if input_name in PPISP_FLOAT2_INPUTS:
        if len(value) != 2:
            raise ValueError(f"PPISP input '{input_name}' expects two values.")
        return (float(value[0]), float(value[1]))
    return float(value)


def _finalize_ppisp_cfg(ppisp_cfg: PpispCfg) -> None:
    if ppisp_cfg.controller_responsivity is None:
        ppisp_cfg.controller_responsivity = float(ppisp_cfg.inputs["responsivity"])
    else:
        ppisp_cfg.controller_responsivity = float(ppisp_cfg.controller_responsivity)


def _read_first_authored_value(attr: Any) -> Any:
    time_samples = attr.GetTimeSamples()
    if time_samples:
        return attr.Get(time_samples[0])
    return attr.Get()


def _read_ppisp_inputs_from_camera(camera_prim: Any | None) -> dict[str, float | tuple[float, float]] | None:
    if camera_prim is None or not camera_prim.IsValid():
        return None

    values = default_ppisp_inputs()
    found = False
    for input_name in values:
        attr = camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{input_name}")
        if not attr or not attr.IsValid():
            continue
        value = _read_first_authored_value(attr)
        if value is not None:
            values[input_name] = _normalize_input_value(input_name, value)
            found = True
    return values if found else None


def _read_controller_weights_from_camera(camera_prim: Any | None) -> tuple[float, ...] | None:
    if camera_prim is None or not camera_prim.IsValid():
        return None
    attr = camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{PPISP_CONTROLLER_WEIGHTS_CAMERA_ATTR}")
    if not attr or not attr.IsValid():
        return None
    value = _read_first_authored_value(attr)
    if value is None:
        return None
    weights = tuple(float(v) for v in value)
    if len(weights) != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
        raise ValueError(
            "Expected "
            f"{PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN} PPISP controller weights on camera "
            f"{camera_prim.GetPath()}, got {len(weights)}."
        )
    return weights


def has_ppisp_camera_attrs(camera_prim: Any | None) -> bool:
    """Return whether a USD camera prim contains recognized PPISP camera attributes.

    Args:
        camera_prim: USD prim to inspect.

    Returns:
        True when ``camera_prim`` is a camera with at least one recognized
        ``ppisp:*`` attribute, otherwise false.
    """
    if camera_prim is None or not camera_prim.IsValid() or camera_prim.GetTypeName() != "Camera":
        return False
    names = (*PPISP_DEFAULT_INPUTS, PPISP_CONTROLLER_WEIGHTS_CAMERA_ATTR)
    return any(
        (attr := camera_prim.GetAttribute(f"{PPISP_ATTR_NAMESPACE}{name}"))
        and attr.IsValid()
        and _read_first_authored_value(attr) is not None
        for name in names
    )
