# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Dexsuite KukaAllegro visual domain randomization.

Same golden-image contract as the Kit file, on the kit-less renderers. Per-environment colours reach
Newton-Warp through the Newton model's per-shape material map, which only exists under Newton
physics, and OVRTX through its material-write notification; both lanes therefore run Newton physics.
"""

from pathlib import Path

import pytest
from rendering_dexsuite_visual_randomization import (
    DEXSUITE_VISUAL_RANDOMIZATION_KITLESS_COMBINATIONS,
    rendering_test_dexsuite_visual_randomization,
)
from rendering_test_utils import (
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_require_ovlibs_install_fixture,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_COMPARISON_SCORES: list[dict] = []

# The determinism fixture is what makes this golden possible: it seeds the RNG the colour term
# samples from, and the reset pose the robot is rendered in.
_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize("physics_backend,renderer", DEXSUITE_VISUAL_RANDOMIZATION_KITLESS_COMBINATIONS)
def test_rendering_dexsuite_visual_randomization_kitless(physics_backend, renderer):
    """Camera output must match golden images (Dexsuite KukaAllegro, per-environment colour randomization)."""
    rendering_test_dexsuite_visual_randomization(physics_backend, renderer, _COMPARISON_SCORES)
