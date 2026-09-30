# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch real Scene UI extensions without starting an XR runtime."""

from isaaclab.app import AppLauncher

_SCENE_UI_KIT_ARGS = " ".join(
    (
        "--enable omni.kit.xr.core",
        "--enable omni.kit.scene_view.xr",
        "--enable omni.kit.scene_view.xr_utils",
    )
)
simulation_app = AppLauncher(
    headless=True,
    enable_cameras=True,
    device="cpu",
    kit_args=_SCENE_UI_KIT_ARGS,
).app

import pytest
import torch
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_teleop.camera_feed import _PanelDescriptor
from isaaclab_teleop.camera_feed_kit_scene_ui import _KitSceneUiCameraFeedPresenter

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.sensors.camera import Camera, CameraCfg

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


def test_real_scene_ui_imports_and_constructs_world_panel():
    """The real Scene UI extensions provide the signatures used by PiP."""
    sim_utils.create_new_stage()
    presenter = _KitSceneUiCameraFeedPresenter()
    descriptor = _PanelDescriptor(
        label="Camera",
        width_m=0.48,
        offset_m=(0.0, 0.0),
        distance_m=0.8,
        placement="world",
        world_position_m=(0.0, 0.8, 1.6),
        world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    panel = presenter.create_panel(descriptor, width=720, height=450)
    try:
        assert panel._container is not None
        assert panel._component is not None
    finally:
        panel.close()

    assert panel._closed


def test_real_panel_uploads_camera_public_rgba_buffer():
    """The real panel provider accepts the selected camera's public RGBA tensor."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(physics=PhysxCfg(), device="cpu", dt=1.0 / 60.0))
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        height=64,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    with cloner.ReplicateSession([camera_cfg], num_clones=1, env_spacing=0.0):
        camera = Camera(camera_cfg)
    panel = None

    try:
        sim.reset()
        sim.step()
        image = camera.data.output["rgba"].torch[0]
        presenter = _KitSceneUiCameraFeedPresenter()
        descriptor = _PanelDescriptor(
            label="Camera",
            width_m=0.48,
            offset_m=(0.0, 0.0),
            distance_m=0.8,
            placement="world",
            world_position_m=(0.0, 0.8, 1.6),
            world_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        panel = presenter.create_panel(descriptor, width=64, height=64)

        panel.upload(image)

        assert image.device.type == "cpu"
        assert image.dtype == torch.uint8
        assert tuple(image.shape) == (64, 64, 4)
    finally:
        if panel is not None:
            panel.close()
        camera._invalidate_initialize_callback(None)
        sim.stop()
        sim.clear_instance()
