# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass


@configclass
class TutorialCfg:
    """Complete declarative input for the direct clone lifecycle."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(physics=PhysxCfg(), dt=0.01)
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/lightDistant",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 10.0)),
    )
    cone_1 = AssetBaseCfg(
        prim_path="/World/Objects/Cone1",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 1.0, 1.0)),
    )
    cone_2 = cone_1.replace(
        prim_path="/World/Objects/Cone2", init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 1.0))
    )
    rigid_cone = AssetBaseCfg(
        prim_path="/World/Objects/ConeRigid",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0.0, 2.0), rot=(0.5, 0.0, 0.5, 0.0)),
    )
    deformable_cuboid = AssetBaseCfg(
        prim_path="/World/Objects/CuboidDeformable",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.5, 0.2),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            physics_material=PhysxDeformableBodyMaterialCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.0, 2.0)),
    )
    table = AssetBaseCfg(
        prim_path="/World/Objects/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05)),
    )


def main():
    """Main function."""
    cfg = TutorialCfg()
    cfg.sim.device = args_cli.device
    sim = sim_utils.SimulationContext(cfg.sim)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    with cloner.ReplicateSession(
        (cfg.ground, cfg.light, cfg.cone_1, cfg.cone_2, cfg.rigid_cone, cfg.deformable_cuboid, cfg.table),
        num_clones=1,
        env_spacing=0.0,
    ):
        for asset_cfg in (
            cfg.ground,
            cfg.light,
            cfg.cone_1,
            cfg.cone_2,
            cfg.rigid_cone,
            cfg.deformable_cuboid,
            cfg.table,
        ):
            asset_cfg.class_type(asset_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
