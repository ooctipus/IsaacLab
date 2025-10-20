import os
import random
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils.assets import LOCAL_ASSET_PATH_DIR, retrieve_file_path

UNIDEX_DIR = f"{LOCAL_ASSET_PATH_DIR}/Props/Unidex"


def _unidex_object_cfg(num_objects: int) -> RigidObjectCfg:
    # pick `num_objects` *random* entries from your full list
    with open(retrieve_file_path(f"{UNIDEX_DIR}/unidex.txt")) as f:
        objs = [line.rstrip("\n") for line in f]
    chosen = random.sample(objs, num_objects)
    usd_cfgs = [sim_utils.UsdFileCfg(usd_path=os.path.join(UNIDEX_DIR, "RawUSD", fname)) for fname in chosen]

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=usd_cfgs,
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                kinematic_enabled=False,
                disable_gravity=False,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )


PRIMITIVE_SHAPE = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
)


CUBE_SHAPE = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            CuboidCfg(size=(0.1, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
        ],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
)


UNIDEX_OBJECT_100 = _unidex_object_cfg(100)
UNIDEX_OBJECT_500 = _unidex_object_cfg(500)
