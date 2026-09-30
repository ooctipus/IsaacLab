# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visualizer material-writer contract."""

from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
from isaaclab_visualizers.newton.newton_visualizer import NewtonVisualizer


def test_newton_visualizer_exposes_shared_writer_factory() -> None:
    class Resource:
        def create_visual_material_writer(self, batches):
            return batches

    resource = Resource()
    visualizer = object.__new__(NewtonVisualizer)
    visualizer._newton_backend = resource
    renderer = object.__new__(NewtonWarpRenderer)
    renderer._newton_backend = resource

    assert visualizer.visual_material_writer == renderer.visual_material_writer
    assert visualizer.visual_material_writer == resource.create_visual_material_writer
