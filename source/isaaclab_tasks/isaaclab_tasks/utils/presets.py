# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXStageVisualizerCfg, NewtonRTXVisualizerCfg
from isaaclab_visualizers.rerun import RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

_STREAMING_CAMERA_PATH = "{ENV_REGEX_NS}/Camera"


@configclass
class _AutoRtxRendererCfg(RendererCfg):
    renderer_type: str = "auto_rtx"


@configclass
class MultiBackendRendererCfg(PresetCfg):
    rtx: _AutoRtxRendererCfg = _AutoRtxRendererCfg()
    ovrtx: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    default: NewtonWarpRendererCfg = NewtonWarpRendererCfg()


@configclass
class MultiBackendVisualizerCfg(PresetCfg):
    """Interactive visualizers selectable through the typed ``visualizer`` preset."""

    default: list[VisualizerCfg] = []
    kit: KitVisualizerCfg = KitVisualizerCfg()
    newton_gl: NewtonGLVisualizerCfg = NewtonGLVisualizerCfg()
    newton_rtx: NewtonRTXVisualizerCfg = NewtonRTXVisualizerCfg(streaming_camera=_STREAMING_CAMERA_PATH)
    newton_rtx_stage: NewtonRTXStageVisualizerCfg = NewtonRTXStageVisualizerCfg()
    rerun: RerunVisualizerCfg = RerunVisualizerCfg()
    viser: ViserVisualizerCfg = ViserVisualizerCfg()


@configclass
class MultiBackendCameraCfg(PresetCfg):
    """Optional plan-owned camera selected with the Newton RTX visualizer."""

    default: CameraCfg | None = None
    newton_rtx: CameraCfg = CameraCfg(
        prim_path=_STREAMING_CAMERA_PATH,
        offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), convention="world"),
        spawn=PinholeCameraCfg(clipping_range=(0.1, 20.0)),
        width=64,
        height=64,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class MultiBackendSceneCfg(InteractiveSceneCfg):
    """Interactive scene with the optional plan-owned streaming camera."""

    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()


@configclass
class MultiBackendSimulationCfg(SimulationCfg):
    """Task simulation defaults with declarative visualizer selection."""

    physics: PhysxCfg = PhysxCfg()
    visualizer_cfgs: MultiBackendVisualizerCfg = MultiBackendVisualizerCfg()
