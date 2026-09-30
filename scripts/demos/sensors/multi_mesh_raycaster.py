# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Example on using the Multi-Mesh Raycaster sensor.

.. code-block:: bash

    # with allegro hand
    python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type allegro_hand

    # with anymal-D bodies
    python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type anymal_d

    # with random multiple objects
    python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type objects

    # with Newton (MJWarp) physics
    python scripts/demos/sensors/multi_mesh_raycaster.py physics=newton_mjwarp

"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSceneCfg, MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Example on using the multi-mesh raycaster sensor.",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument(
    "--flat_ground",
    action="store_true",
    help="Use a lightweight flat mesh instead of the high-resolution rough terrain.",
)
parser.add_argument(
    "--asset_type",
    type=str,
    default="allegro_hand",
    help="Asset type to use.",
    choices=["allegro_hand", "anymal_d", "objects"],
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import torch
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

if args_cli.flat_ground:
    ground_spawn_cfg = sim_utils.MeshCuboidCfg(
        size=(20.0, 20.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    ground_init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05))
else:
    ground_spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        scale=(1, 1, 1),
    )
    ground_init_state = AssetBaseCfg.InitialStateCfg()

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

if args_cli.asset_type == "allegro_hand":
    asset_cfg = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/palm_link",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0, -0.1, 0.3)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/thumb_link_.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/index_link.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/middle_link_.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/ring_link_.*/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/palm_link/visuals_xform"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/allegro_mount/visuals_xform"),
        ],
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.005, size=(0.4, 0.4), direction=(0, 0, -1)),
        debug_vis=True,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

elif args_cli.asset_type == "anymal_d":
    asset_cfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0, -0.1, 0.3)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LF_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RF_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LH_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RH_.*/visuals"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/base/visuals"),
        ],
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(2.5, 2.5), direction=(0, 0, -1)),
        debug_vis=True,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )

elif args_cli.asset_type == "objects":
    object_assets_cfg = [
        sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        sim_utils.SphereCfg(
            radius=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        sim_utils.CylinderCfg(
            radius=0.2,
            height=0.5,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        sim_utils.CapsuleCfg(
            radius=0.15,
            height=0.5,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
        ),
        sim_utils.ConeCfg(
            radius=0.2,
            height=0.5,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
        ),
    ]
    asset_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=preset(
                default=object_assets_cfg,
                isaacsim_physx=object_assets_cfg,
                newton_mjwarp=object_assets_cfg[:1],
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0, 0.0, 0.6)),
        mesh_prim_paths=[
            "/World/Ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Object"),
        ],
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.6, 0.6), direction=(0, 0, -1)),
        debug_vis=True,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )
else:
    raise ValueError(f"Unknown asset type: {args_cli.asset_type}")


@configclass
class RaycasterSensorSceneCfg(MultiBackendSceneCfg):
    """Design the scene with sensors on the asset."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=ground_spawn_cfg,
        init_state=ground_init_state,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # asset
    asset = asset_cfg
    # ray caster
    ray_caster = ray_caster_cfg


@configclass
class DemoCfg:
    """Multi-mesh ray-caster demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physics=preset(
            default=PhysxCfg(),
            isaacsim_physx=PhysxCfg(),
            newton_mjwarp=MJWarpSolverCfg(),
        ),
    )
    scene: RaycasterSensorSceneCfg = RaycasterSensorSceneCfg(
        num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=True
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    from isaaclab.assets import Articulation

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while sim.is_headless_or_exist_active_visualizer():
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_pose = scene["asset"].data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            scene["asset"].write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = scene["asset"].data.default_root_vel.torch.clone()
            scene["asset"].write_root_velocity_to_sim_index(root_velocity=root_vel)

            if isinstance(scene["asset"], Articulation):
                # set joint positions with some noise
                joint_pos, joint_vel = (
                    scene["asset"].data.default_joint_pos.torch.clone(),
                    scene["asset"].data.default_joint_vel.torch.clone(),
                )
                joint_pos += torch.rand_like(joint_pos) * 0.1
                scene["asset"].write_joint_position_to_sim_index(position=joint_pos)
                scene["asset"].write_joint_velocity_to_sim_index(velocity=joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Asset state...")

        if isinstance(scene["asset"], Articulation):
            # -- generate actions/commands
            default_joint_pos = scene["asset"].data.default_joint_pos.torch
            targets = default_joint_pos + 5 * (torch.rand_like(default_joint_pos) - 0.5)
            # -- apply action to the asset
            scene["asset"].set_joint_position_target_index(target=targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg.sim, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
        scene = cfg.scene.class_type(cfg.scene)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
