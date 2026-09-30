# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab port of Newton's granular implicit-MPM example.

A block of granular material is dropped onto static box colliders. The demo
shows the intended Isaac Lab MPM path: configure
:class:`~isaaclab_newton.physics.MPMSolverCfg`, add an
:class:`~isaaclab_newton.assets.MPMObject` to an
:class:`~isaaclab.scene.InteractiveScene`, and let
:class:`~isaaclab_newton.physics.NewtonMPMManager` own solver creation and
stepping.

.. code-block:: bash

    uv run python scripts/demos/mpm/newton_mpm_granular.py
    uv run python scripts/demos/mpm/newton_mpm_granular.py visualizer=kit
"""

from __future__ import annotations

import argparse
import math

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from isaaclab_visualizers.rerun import RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg, MultiBackendSimulationCfg, MultiBackendVisualizerCfg

parser = argparse.ArgumentParser(description="Newton implicit MPM granular demo.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Stop after this many frames; negative runs forever.",
)
parser.add_argument("--collider", default="cube", choices=["cube", "wedge", "concave", "none"], help="Collider set.")
parser.add_argument(
    "--voxel_size",
    type=float,
    default=0.1,
    help="MPM grid voxel size [m].",
)
parser.add_argument("--substeps", type=int, default=1, help="Solver substeps per frame.")
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)


FPS = 100.0
GRAVITY = (0.0, 0.0, -10.0)
VOXEL_SIZE = float(args_cli.voxel_size)

# Solver settings that differ from the MPMSolverCfg defaults. A fixed grid has a
# static topology, so the MPM step can be captured as a CUDA graph.
# ``grid_padding`` sizes the frozen grid to contain the block as it falls and
# spreads over the colliders, and ``max_active_cell_count`` bounds the per-step
# active set so the graph-captured allocations stay static.
GRID_TYPE = "fixed"
GRID_PADDING = 48
MAX_ACTIVE_CELL_COUNT = 1 << 18

# Granular block, emitted as a jittered particle grid.
EMIT_LO = (-1.0, -1.0, 2.0)
EMIT_HI = (1.0, 1.0, 3.5)
PARTICLES_PER_CELL = 3.0
PARTICLE_JITTER = VOXEL_SIZE / PARTICLES_PER_CELL

PARTICLE_COLOR = (0.7, 0.6, 0.4)

IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
Y_ROT_45_DEG = (0.0, math.sin(math.pi / 8.0), 0.0, math.cos(math.pi / 8.0))
Y_ROT_NEG_45_DEG = (0.0, -math.sin(math.pi / 8.0), 0.0, math.cos(math.pi / 8.0))


def create_sim_cfg():
    """Create the Isaac Lab simulation config that the MPM manager drives."""
    from isaaclab_newton.physics import MPMSolverCfg

    return MultiBackendSimulationCfg(
        dt=1.0 / FPS,
        device=args_cli.device,
        gravity=GRAVITY,
        visualizer_cfgs=MultiBackendVisualizerCfg(
            newton_gl=NewtonGLVisualizerCfg(show_particles=True, particle_color=PARTICLE_COLOR),
            rerun=RerunVisualizerCfg(show_particles=True),
            viser=ViserVisualizerCfg(show_particles=True),
        ),
        physics=MPMSolverCfg(
            voxel_size=VOXEL_SIZE,
            grid_type=GRID_TYPE,
            grid_padding=GRID_PADDING,
            max_active_cell_count=MAX_ACTIVE_CELL_COUNT,
            project_outside_colliders=True,
            num_substeps=args_cli.substeps,
        ),
    )


def create_scene_cfg():
    """Create an Isaac Lab scene config using declarative assets."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg

    def collider_cfg(prim_path: str, center, half_extents, orientation, friction: float = 0.1) -> AssetBaseCfg:
        return AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                size=(2.0 * half_extents[0], 2.0 * half_extents[1], 2.0 * half_extents[2]),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=friction,
                    dynamic_friction=friction,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=center, rot=orientation),
        )

    @configclass
    class GranularSceneCfg(InteractiveSceneCfg):
        """Scene containing static colliders and one Newton MPM object."""

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(6.0, 6.0), color=(0.30, 0.30, 0.30)),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/GranularMedia",
            spawn=MPMGridCfg(
                lower=EMIT_LO,
                upper=EMIT_HI,
                voxel_size=VOXEL_SIZE,
                particles_per_cell=PARTICLES_PER_CELL,
                particle_placement="cell_center",
                jitter=PARTICLE_JITTER,
                visual_color=PARTICLE_COLOR,
            ),
        )

        camera = preset(
            default=None,
            newton_rtx=CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera",
                offset=CameraCfg.OffsetCfg(
                    pos=(4.0, -6.0, 4.0),
                    rot=(0.5311168, 0.1608092, 0.2410710, 0.7962036),
                    convention="opengl",
                ),
                spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 20.0)),
                width=640,
                height=480,
                renderer_cfg=MultiBackendRendererCfg(),
            ),
        )

        if args_cli.collider == "cube":
            cube = collider_cfg("/World/Colliders/Cube", (0.75, 0.0, 0.8), (0.5, 2.0, 0.8), IDENTITY_ROT)
        elif args_cli.collider == "wedge":
            wedge = collider_cfg("/World/Colliders/Wedge", (0.1, 0.0, 0.5), (0.5, 2.0, 0.5), Y_ROT_45_DEG)
        elif args_cli.collider == "concave":
            left_ramp = collider_cfg("/World/Colliders/LeftRamp", (-0.7, 0.0, 0.8), (1.0, 2.0, 0.25), Y_ROT_45_DEG)
            right_ramp = collider_cfg("/World/Colliders/RightRamp", (0.7, 0.0, 0.8), (1.0, 2.0, 0.25), Y_ROT_NEG_45_DEG)

    return GranularSceneCfg(num_envs=1, env_spacing=0.0)


def particle_count(scene) -> int:
    """Return the number of MPM particles in the scene."""
    media = scene["media"]
    return media.num_instances * media.particles_per_object


def keep_running(sim, count: int) -> bool:
    """Return whether the demo loop should continue this frame."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    return sim.is_headless_or_exist_active_visualizer()


def run_simulator(sim, scene) -> None:
    """Run the MPM simulation using Isaac Lab's scene and physics-manager loop."""
    sim_dt = sim.get_physics_dt()
    count = 0
    while keep_running(sim, count):
        sim.step(render=False)
        scene.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Launch and run the Isaac Lab MPM granular demo."""
    cfg = resolve_config({"sim": create_sim_cfg(), "scene": create_scene_cfg()}, config_overrides)
    sim_cfg, scene_cfg = cfg["sim"], cfg["scene"]
    with launch_simulation(cfg, args_cli):
        import isaaclab.sim as sim_utils

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(4.0, -6.0, 4.0), target=(0.0, 0.0, 1.0))
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        print(
            f"[INFO]: Isaac Lab Newton granular MPM demo ready. Spawned {particle_count(scene)} particles.",
            flush=True,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
