# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn deformable prims into the scene.

.. code-block:: bash

    # Usage with default PhysX physics and no visualizer.
    uv run --extra isaacsim --extra tetrahedralization python scripts/demos/deformables.py

    # Usage with Newton VBD backend and no visualizer.
    uv run --extra isaacsim --extra tetrahedralization python scripts/demos/deformables.py physics=newton_vbd

    # Install the optional dependencies for the repository launcher.
    ./isaaclab.sh -i tetrahedralization

    # Usage with OvPhysX backend without a visualizer.
    ./isaaclab.sh -p scripts/demos/deformables.py physics=ovphysx sim.visualizer_cfgs=[]

"""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

# create argparser
parser = argparse.ArgumentParser(
    description="This script demonstrates how to spawn deformable prims into the scene.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import (
    CloneCfg,
    InclusionSet,
    ReplicateSession,
    make_valid_clone_combinations,
    num_spawn_variants,
)

##
# Pre-defined configs
##
from isaaclab.assets import DeformableObjectCfg  # isort:skip
from isaaclab.utils.configclass import configclass  # isort:skip

from isaaclab_newton.physics import NewtonSoftContactCfg, VBDSolverCfg  # isort:skip
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg  # isort:skip
from isaaclab_newton.sim.spawners.materials import (  # isort:skip
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.physics import OvPhysxCfg  # isort:skip
from isaaclab_physx.physics import PhysxCfg  # isort:skip
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg  # isort:skip
from isaaclab_physx.sim.spawners.materials import (  # isort:skip
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject

_PHYSX_ASSET_CFGS = (
    PhysxDeformableBodyPropertiesCfg(),
    PhysxDeformableBodyMaterialCfg(),
    PhysxSurfaceDeformableBodyMaterialCfg(),
)


@configclass
class DemoCfg:
    """Deformable demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=preset(
            default=PhysxCfg(),
            isaacsim_physx=PhysxCfg(),
            newton_vbd=VBDSolverCfg(
                iterations=5,
                particle_enable_self_contact=True,
                particle_self_contact_radius=0.0001,
                particle_self_contact_margin=0.1,
                num_substeps=4,
                soft_contact_cfg=NewtonSoftContactCfg(
                    soft_contact_ke=1.0e5,
                    soft_contact_kd=1.0e0,
                    soft_contact_mu=0.01,
                ),
            ),
            ovphysx=OvPhysxCfg(),
        ),
    )
    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()
    asset_backend: tuple[object, object, object] = preset(
        default=_PHYSX_ASSET_CFGS,
        isaacsim_physx=_PHYSX_ASSET_CFGS,
        newton_vbd=(
            NewtonDeformableBodyPropertiesCfg(),
            NewtonDeformableBodyMaterialCfg(),
            NewtonSurfaceDeformableBodyMaterialCfg(),
        ),
        ovphysx=_PHYSX_ASSET_CFGS,
    )
    num_envs: int = 12
    env_spacing: float = 1.0
    clone_cfg: CloneCfg = CloneCfg(
        clone_combinations=[
            InclusionSet(assets=[name]) for name in ("sphere", "cuboid", "cylinder", "capsule", "cone", "surface")
        ],
    )
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    sphere: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.MeshSphereCfg(radius=0.4),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    sphere.visualizer_cfg.prim_path = "/Visuals/SphereDeformableTarget"
    cuboid: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cuboid",
        spawn=sim_utils.MeshCuboidCfg(size=(0.6, 0.6, 0.6)),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    cuboid.visualizer_cfg.prim_path = "/Visuals/CuboidDeformableTarget"
    cylinder: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.MeshCylinderCfg(radius=0.25, height=0.5),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    cylinder.visualizer_cfg.prim_path = "/Visuals/CylinderDeformableTarget"
    capsule: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Capsule",
        spawn=sim_utils.MeshCapsuleCfg(radius=0.35, height=0.5),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    capsule.visualizer_cfg.prim_path = "/Visuals/CapsuleDeformableTarget"
    cone: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cone",
        spawn=sim_utils.MeshConeCfg(radius=0.35, height=0.75),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    cone.visualizer_cfg.prim_path = "/Visuals/ConeDeformableTarget"
    surface: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Surface",
        spawn=sim_utils.MeshRectangleCfg(size=(1.5, 1.0), resolution=(21, 21)),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    surface.visualizer_cfg.prim_path = "/Visuals/SurfaceDeformableTarget"


def run_simulator(sim: "sim_utils.SimulationContext", objects: tuple["DeformableObject", ...]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        # reset
        if count % int(3.0 / sim_dt) == 0:
            # reset counters
            count = 0
            # reset deformable object state
            for deformable in objects:
                nodal_state = deformable.data.default_nodal_state_w.torch.clone()
                deformable.write_nodal_state_to_sim_index(nodal_state)
                deformable.reset()
            print("[INFO]: Resetting deformable object state...")
        # perform step
        sim.step()
        count += 1
        # update buffers
        for deformable in objects:
            deformable.update(sim_dt)


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    deformable_props, volume_material, surface_material = cfg.asset_backend
    volume_cfgs = (cfg.sphere, cfg.cuboid, cfg.cylinder, cfg.capsule, cfg.cone)
    deformable_cfgs = (*volume_cfgs, cfg.surface)
    for object_cfg in volume_cfgs:
        object_cfg.spawn.deformable_props = deformable_props.copy()
        object_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg()
        object_cfg.spawn.physics_material = volume_material.copy()
    cfg.surface.spawn.deformable_props = deformable_props.copy()
    cfg.surface.spawn.visual_material = sim_utils.PreviewSurfaceCfg()
    cfg.surface.spawn.physics_material = surface_material.copy()
    with launch_simulation(cfg, args_cli):
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(cfg.sim)
        # Set main camera
        sim.set_camera_view([4.0, 4.0, 3.0], [0.5, 0.5, 0.0])

        asset_names = ("sphere", "cuboid", "cylinder", "capsule", "cone", "surface")
        valid_set = make_valid_clone_combinations(
            asset_names,
            tuple(num_spawn_variants(object_cfg.spawn) for object_cfg in deformable_cfgs),
            cfg.clone_cfg.clone_combinations,
        )
        asset_cfgs = (
            cfg.ground,
            cfg.light,
            *deformable_cfgs,
            *(object_cfg.visualizer_cfg for object_cfg in deformable_cfgs),
            *((cfg.camera,) if cfg.camera is not None else ()),
        )
        with ReplicateSession(
            asset_cfgs,
            cfg.num_envs,
            cfg.env_spacing,
            clone_strategy=cfg.clone_cfg.clone_strategy,
            valid_set=valid_set,
            env_template=cfg.clone_cfg.clone_template,
            replicate_physics=cfg.clone_cfg.replicate_physics,
        ):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            objects = tuple(object_cfg.class_type(object_cfg) for object_cfg in deformable_cfgs)
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        run_simulator(sim, objects)
        print("[INFO]: Simulation complete...")


if __name__ == "__main__":
    # run the main function
    main()
