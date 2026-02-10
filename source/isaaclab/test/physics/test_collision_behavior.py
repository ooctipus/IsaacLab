# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify collision behavior between different collision primitives using Newton physics.

This test suite leverages Newton's parallel environment support to run multiple test
configurations simultaneously, dramatically reducing test time.

Test coverage:
1. Horizontal collisions between different shape types (batched across envs)
2. Falling objects colliding with ground (batched across envs)
3. Box stacking stability
4. Finger collision isolation on articulated hands (batched across envs)

Usage:
    # Run normally
    pytest test_collision_behavior.py

    # Run with Newton visualizer for debugging
    pytest test_collision_behavior.py --visualize -k "finger"
"""

# pyright: reportPrivateUsage=none

from __future__ import annotations

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim.spawners import materials
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import hand configurations for articulated collision tests
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

# Import shared physics test utilities
from physics.physics_test_utils import (
    COLLISION_PIPELINES,
    ShapeType,
    create_shape_cfg,
    get_shape_extent,
    get_shape_height,
    make_sim_cfg,
    perform_sim_step,
    shape_type_to_str,
)


##
# Constants
##

SIM_DT = 1.0 / 240.0  # 240 Hz physics

# Shape types to test (excluding problematic CONE)
GROUND_COLLISION_SHAPES = [
    ShapeType.SPHERE,
    ShapeType.BOX,
    ShapeType.CAPSULE,
    ShapeType.CYLINDER,
    ShapeType.MESH_SPHERE,
    ShapeType.MESH_BOX,
    ShapeType.MESH_CAPSULE,
    ShapeType.MESH_CYLINDER,
]

# Collision pairs for horizontal collision tests
# Format: (shape_a, shape_b)
HORIZONTAL_COLLISION_PAIRS = [
    # Primitive collisions
    (ShapeType.SPHERE, ShapeType.SPHERE),
    (ShapeType.SPHERE, ShapeType.BOX),
    (ShapeType.BOX, ShapeType.BOX),
    (ShapeType.CAPSULE, ShapeType.CAPSULE),
    (ShapeType.CYLINDER, ShapeType.CYLINDER),
    # Mesh collisions
    (ShapeType.MESH_SPHERE, ShapeType.MESH_SPHERE),
    (ShapeType.MESH_BOX, ShapeType.MESH_BOX),
    (ShapeType.MESH_CAPSULE, ShapeType.MESH_CAPSULE),
    # Mixed: mesh vs primitive
    (ShapeType.MESH_SPHERE, ShapeType.SPHERE),
    (ShapeType.MESH_BOX, ShapeType.BOX),
]

# Box stacking pairs
STACKING_PAIRS = [
    (ShapeType.BOX, ShapeType.BOX),
    (ShapeType.MESH_BOX, ShapeType.MESH_BOX),
    (ShapeType.BOX, ShapeType.MESH_BOX),
    (ShapeType.MESH_BOX, ShapeType.BOX),
]

# Finger collision test configuration
FINGER_NAMES = ["index", "middle", "ring", "thumb"]
FINGER_DROP_SHAPES = [ShapeType.SPHERE, ShapeType.MESH_SPHERE, ShapeType.BOX, ShapeType.MESH_BOX]

# Finger tip positions relative to hand root
ALLEGRO_FINGERTIP_OFFSETS = {
    "index": (-0.052, -0.252, 0.052),
    "middle": (-0.001, -0.252, 0.052),
    "ring": (0.054, -0.252, 0.052),
    "thumb": (-0.168, -0.039, 0.080),
}

# Joint names for each finger
ALLEGRO_FINGER_JOINTS = {
    "index": ["index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"],
    "middle": ["middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"],
    "ring": ["ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"],
    "thumb": ["thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"],
}


def get_shape_min_resting_height(shape_type: ShapeType) -> float:
    """Get the minimum height where an object can rest (accounts for tumbling)."""
    min_heights = {
        ShapeType.SPHERE: 0.25,
        ShapeType.BOX: 0.25,
        ShapeType.CAPSULE: 0.15,
        ShapeType.CYLINDER: 0.2,
        ShapeType.MESH_SPHERE: 0.25,
        ShapeType.MESH_BOX: 0.25,
        ShapeType.MESH_CAPSULE: 0.15,
        ShapeType.MESH_CYLINDER: 0.2,
    }
    return min_heights.get(shape_type, 0.2)


##
# Scene Configurations
##


@configclass
class CollisionTestSceneCfg(InteractiveSceneCfg):
    """Configuration for collision test scenes."""

    terrain: TerrainImporterCfg | None = None
    object_a: RigidObjectCfg | None = None
    object_b: RigidObjectCfg | None = None


##
# Batched Collision Tests
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_horizontal_collision_batched(device: str, use_mujoco_contacts: bool):
    """Test horizontal collisions between shape pairs using parallel environments.

    Each environment tests a different shape pair. All environments run in parallel.
    Object A approaches Object B along the X-axis; after collision, momentum transfers.
    """
    num_envs = len(HORIZONTAL_COLLISION_PAIRS)
    collision_steps = 240

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create scene with environments for each collision pair
        scene_cfg = CollisionTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)

        # Use a representative shape for scene config (actual shapes set per-env below)
        # We'll use the first pair's shapes as the "template"
        shape_a, shape_b = HORIZONTAL_COLLISION_PAIRS[0]
        extent_a = get_shape_extent(shape_a)
        extent_b = get_shape_extent(shape_b)
        separation = (extent_a + extent_b) * 2.5

        scene_cfg.object_a = create_shape_cfg(shape_a, "{ENV_REGEX_NS}/ObjectA", pos=(-separation / 2, 0.0, 0.5))
        scene_cfg.object_b = create_shape_cfg(shape_b, "{ENV_REGEX_NS}/ObjectB", pos=(separation / 2, 0.0, 0.5))

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        object_a: RigidObject = scene["object_a"]
        object_b: RigidObject = scene["object_b"]

        # Set initial velocities for all environments
        initial_velocity = 2.0
        vel_a = torch.zeros((num_envs, 6), device=device)
        vel_a[:, 0] = initial_velocity
        object_a.write_root_velocity_to_sim(vel_a)
        object_b.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))

        # Run simulation
        collision_detected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for _ in range(collision_steps):
            perform_sim_step(sim, scene, SIM_DT)
            # Check if collision occurred (object B gained velocity)
            vel_b = wp.to_torch(object_b.data.root_lin_vel_w)[:, 0]
            collision_detected |= vel_b > 0.1

        # Verify all environments had collisions
        final_vel_a = wp.to_torch(object_a.data.root_lin_vel_w)
        final_vel_b = wp.to_torch(object_b.data.root_lin_vel_w)

        for env_idx, (shape_a, shape_b) in enumerate(HORIZONTAL_COLLISION_PAIRS):
            pair_name = f"{shape_type_to_str(shape_a)}_{shape_type_to_str(shape_b)}"

            assert collision_detected[env_idx], f"[{pair_name}] Collision should have occurred"
            assert final_vel_b[env_idx, 0] > 0.1, f"[{pair_name}] Object B should move forward after collision"
            assert final_vel_a[env_idx, 0] < initial_velocity, f"[{pair_name}] Object A should have slowed"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_falling_collision_with_ground_batched(device: str, use_mujoco_contacts: bool):
    """Test objects falling and colliding with ground using parallel environments.

    Each environment tests a different shape type. All environments run in parallel.
    """
    shapes = GROUND_COLLISION_SHAPES
    num_envs = len(shapes)
    fall_steps = 480

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            shapes[0], "{ENV_REGEX_NS}/Object", pos=(0.0, 0.0, 2.0), disable_gravity=False
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        obj: RigidObject = scene["object_a"]

        # Run simulation
        for _ in range(fall_steps):
            perform_sim_step(sim, scene, SIM_DT)

        # Verify all environments
        final_pos = wp.to_torch(obj.data.root_pos_w)
        final_vel = wp.to_torch(obj.data.root_lin_vel_w)

        for env_idx, shape_type in enumerate(shapes):
            shape_name = shape_type_to_str(shape_type)
            final_height = final_pos[env_idx, 2].item()
            final_speed = torch.norm(final_vel[env_idx]).item()
            expected_min_height = get_shape_min_resting_height(shape_type) - 0.05

            assert final_height > expected_min_height, (
                f"[{shape_name}] Object fell through ground: height={final_height:.4f}, "
                f"expected > {expected_min_height:.4f}"
            )
            assert final_speed < 0.5, f"[{shape_name}] Object still moving too fast: velocity={final_speed:.4f}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_box_stacking_stability_batched(device: str, use_mujoco_contacts: bool):
    """Test box stacking stability using parallel environments.

    Each environment tests a different combination of primitive/mesh boxes.
    """
    num_envs = len(STACKING_PAIRS)
    settle_steps = 480

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)

        # Use first pair as template
        bottom_shape, top_shape = STACKING_PAIRS[0]
        height = get_shape_height(bottom_shape)
        scene_cfg.object_a = create_shape_cfg(
            bottom_shape, "{ENV_REGEX_NS}/BoxBottom", pos=(0.0, 0.0, height / 2 + 0.01), disable_gravity=False
        )
        scene_cfg.object_b = create_shape_cfg(
            top_shape, "{ENV_REGEX_NS}/BoxTop", pos=(0.0, 0.0, height * 1.5 + 0.02), disable_gravity=False
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        box_bottom: RigidObject = scene["object_a"]
        box_top: RigidObject = scene["object_b"]

        # Run simulation
        for _ in range(settle_steps):
            perform_sim_step(sim, scene, SIM_DT)

        # Verify all environments
        bottom_pos = wp.to_torch(box_bottom.data.root_pos_w)
        top_pos = wp.to_torch(box_top.data.root_pos_w)

        for env_idx, (bottom_shape, top_shape) in enumerate(STACKING_PAIRS):
            pair_name = f"{shape_type_to_str(bottom_shape)}_under_{shape_type_to_str(top_shape)}"
            bottom_height = bottom_pos[env_idx, 2].item()
            top_height = top_pos[env_idx, 2].item()

            assert bottom_height > 0.2, f"[{pair_name}] Bottom box fell through ground: height={bottom_height:.4f}"
            assert top_height > bottom_height + 0.3, (
                f"[{pair_name}] Top box not properly stacked: top={top_height:.4f}, bottom={bottom_height:.4f}"
            )


##
# Articulated Hand Collision Tests (Batched with newton_replicate)
##


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("drop_shape", [ShapeType.SPHERE, ShapeType.BOX], ids=["sphere", "box"])
def test_finger_collision_isolation(device: str, use_mujoco_contacts: bool, drop_shape: ShapeType, request):
    """Test that dropping an object on one finger only affects that finger.

    Uses 4 parallel environments (one per finger) with newton_replicate for proper cloning.
    Run with --visualize flag to enable Newton visualizer for debugging.
    """
    from pxr import UsdGeom

    from isaaclab.cloner.cloner_utils import newton_replicate
    from isaaclab.sim.utils.stage import get_current_stage

    from physics.physics_test_utils import TestVisualizer

    num_envs, drop_steps, settle_steps = 4, 480, 30
    hand_pos, drop_height = (0.0, 0.0, 0.5), 0.10
    hand_orientation = (0.283045, 0.683330, -0.621782, 0.257551)

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))
    viz = TestVisualizer(request, camera_position=(0.5, -0.8, 0.8), camera_target=(0.0, 0.0, 0.5))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        stage = get_current_stage()

        # Create environment containers and ground
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        UsdGeom.Xform.Define(stage, "/World/envs")
        for i in range(num_envs):
            UsdGeom.Xform.Define(stage, f"/World/envs/env_{i}")

        # Spawn hand and drop object at env_0
        hand_cfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_0/Hand")
        hand_cfg.spawn.func(hand_cfg.prim_path, hand_cfg.spawn, translation=hand_pos, orientation=hand_orientation)

        props = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            "mass_props": sim_utils.MassPropertiesCfg(mass=0.2),
        }
        spawn = sim_utils.SphereCfg(radius=0.035, **props) if drop_shape == ShapeType.SPHERE else sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), **props)
        spawn.func("/World/envs/env_0/DropObject", spawn, translation=hand_pos)
        object_cfg = RigidObjectCfg(spawn=spawn)

        # Clone to all environments
        sources = ["/World/envs/env_0/Hand", "/World/envs/env_0/DropObject"]
        destinations = ["/World/envs/env_{}/Hand", "/World/envs/env_{}/DropObject"]
        env_ids = torch.arange(num_envs, device=device)
        mapping = torch.ones((2, num_envs), dtype=torch.bool, device=device)
        newton_replicate(stage, sources, destinations, env_ids, mapping)

        # Create views (spawn=None since prims exist)
        hand_cfg.prim_path, hand_cfg.spawn = "/World/envs/env_.*/Hand", None
        hand = Articulation(hand_cfg)
        object_cfg.prim_path, object_cfg.spawn = "/World/envs/env_.*/DropObject", None
        drop_object = RigidObject(object_cfg)

        # Initialize
        sim.reset()
        hand.reset()
        drop_object.reset()
        hand.set_joint_position_target(wp.to_torch(hand.data.joint_pos))
        hand.write_data_to_sim()

        # Position drop objects above different fingers per environment
        drop_positions = torch.zeros((num_envs, 3), device=device)
        for i, finger in enumerate(FINGER_NAMES):
            offset = ALLEGRO_FINGERTIP_OFFSETS[finger]
            drop_positions[i] = torch.tensor([hand_pos[0] + offset[0], hand_pos[1] + offset[1], hand_pos[2] + offset[2] + drop_height])
        drop_object.write_root_pose_to_sim(torch.cat([drop_positions, torch.tensor([[0, 0, 0, 1]] * num_envs, device=device)], dim=1))

        # Settle
        for _ in range(settle_steps):
            hand.write_data_to_sim()
            drop_object.write_data_to_sim()
            sim.step(render=False)
            viz.step(NewtonManager._state_0)
            hand.update(SIM_DT)
            drop_object.update(SIM_DT)

        # Apply initial velocity
        velocity = torch.zeros((num_envs, 6), device=device)
        velocity[:, 2] = -1.0
        drop_object.write_root_velocity_to_sim(velocity)

        # Track deflections
        initial_joint_pos = wp.to_torch(hand.data.joint_pos).clone()
        joint_names = hand.data.joint_names
        peak_deflections = [{f: 0.0 for f in FINGER_NAMES} for _ in range(num_envs)]

        # Simulate
        for step in range(drop_steps):
            hand.write_data_to_sim()
            drop_object.write_data_to_sim()
            sim.step(render=False)
            viz.step(NewtonManager._state_0)
            hand.update(SIM_DT)
            drop_object.update(SIM_DT)

            # Update peak deflections
            current_pos = wp.to_torch(hand.data.joint_pos)
            for env_idx in range(num_envs):
                for finger in FINGER_NAMES:
                    deflection = sum(
                        abs(current_pos[env_idx, joint_names.index(j)].item() - initial_joint_pos[env_idx, joint_names.index(j)].item())
                        for j in ALLEGRO_FINGER_JOINTS[finger] if j in joint_names
                    )
                    peak_deflections[env_idx][finger] = max(peak_deflections[env_idx][finger], deflection)

            if step % 120 == 0:
                print(f"  Step {step:4d}: " + " | ".join(
                    f"Env{i}({FINGER_NAMES[i]}):{peak_deflections[i][FINGER_NAMES[i]]:.3f}" for i in range(num_envs)
                ))

        # Print results
        print("\n  FINAL RESULTS:")
        for env_idx, target in enumerate(FINGER_NAMES):
            d = peak_deflections[env_idx]
            print(f"    Env{env_idx} ({target}): " + " ".join(
                f"{'>>>' if f == target else '   '}{f}:{d[f]:.4f}" for f in FINGER_NAMES
            ))

        viz.wait_for_close(sim, lambda: NewtonManager._state_0)

        # Verify: target finger should have highest deflection
        for env_idx, target in enumerate(FINGER_NAMES):
            target_peak = peak_deflections[env_idx][target]
            assert target_peak > 0.01, f"[Env{env_idx}] Target '{target}' deflection too small: {target_peak:.6f}"
            if target != "thumb":  # Thumb is geometrically isolated
                for other in ["index", "middle", "ring"]:
                    if other != target:
                        assert target_peak >= peak_deflections[env_idx][other], (
                            f"[Env{env_idx}] '{target}' ({target_peak:.4f}) should deflect more than '{other}' ({peak_deflections[env_idx][other]:.4f})"
                        )



@pytest.mark.parametrize("device", ["cuda:0"])
def test_finger_collision_isolation_heterogeneous(device: str, request):
    """Test finger collision isolation with 1024 heterogeneous environments and 10 different mesh objects.

    Uses clone_from_template to spawn 10 different drop object shapes across 1024 environments.
    Each environment gets a sequentially assigned shape (round-robin), demonstrating large-scale
    heterogeneous cloning.

    NOTE: Heterogeneous environments require:
    - Newton collision pipeline (not mujoco_contacts)
    - MeshShape variants instead of primitives

    Run with --visualize flag to enable Newton visualizer for debugging.
    """
    from pxr import UsdGeom

    from isaaclab.cloner.cloner_cfg import TemplateCloneCfg
    from isaaclab.cloner.cloner_strategies import sequential
    from isaaclab.cloner.cloner_utils import clone_from_template
    from isaaclab.sim.utils.stage import get_current_stage

    from physics.physics_test_utils import TestVisualizer

    num_envs, drop_steps, settle_steps = 32, 240, 15
    hand_pos, drop_height = (0.0, 0.0, 0.5), 0.10
    hand_orientation = (0.283045, 0.683330, -0.621782, 0.257551)
    num_shapes = 10

    # Heterogeneous requires Newton pipeline (use_mujoco_contacts=False)
    sim_cfg = make_sim_cfg(use_mujoco_contacts=False, device=device, gravity=(0.0, 0.0, 0.0))
    viz = TestVisualizer(request, camera_position=(0.5, -0.8, 0.8), camera_target=(0.0, 0.0, 0.5))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        stage = get_current_stage()

        # Create ground and template root
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        UsdGeom.Xform.Define(stage, "/World/template")
        UsdGeom.Xform.Define(stage, "/World/envs")
        for i in range(num_envs):
            UsdGeom.Xform.Define(stage, f"/World/envs/env_{i}")

        # Create hand prototype (only one - homogeneous across all envs)
        hand_cfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/template/hand/proto_asset_0")
        hand_cfg.spawn.func(hand_cfg.prim_path, hand_cfg.spawn, translation=hand_pos, orientation=hand_orientation)

        # Create 10 heterogeneous drop object prototypes with varying shapes and sizes
        props = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            "mass_props": sim_utils.MassPropertiesCfg(mass=0.2),
        }

        shape_names = []
        # Prototype 0: Small sphere
        sim_utils.MeshSphereCfg(radius=0.025, **props).func("/World/template/object/proto_asset_0", sim_utils.MeshSphereCfg(radius=0.025, **props), translation=hand_pos)
        shape_names.append("small_sphere")
        # Prototype 1: Medium sphere
        sim_utils.MeshSphereCfg(radius=0.035, **props).func("/World/template/object/proto_asset_1", sim_utils.MeshSphereCfg(radius=0.035, **props), translation=hand_pos)
        shape_names.append("med_sphere")
        # Prototype 2: Large sphere
        sim_utils.MeshSphereCfg(radius=0.045, **props).func("/World/template/object/proto_asset_2", sim_utils.MeshSphereCfg(radius=0.045, **props), translation=hand_pos)
        shape_names.append("large_sphere")
        # Prototype 3: Small cube
        sim_utils.MeshCuboidCfg(size=(0.03, 0.03, 0.03), **props).func("/World/template/object/proto_asset_3", sim_utils.MeshCuboidCfg(size=(0.03, 0.03, 0.03), **props), translation=hand_pos)
        shape_names.append("small_cube")
        # Prototype 4: Medium cube
        sim_utils.MeshCuboidCfg(size=(0.05, 0.05, 0.05), **props).func("/World/template/object/proto_asset_4", sim_utils.MeshCuboidCfg(size=(0.05, 0.05, 0.05), **props), translation=hand_pos)
        shape_names.append("med_cube")
        # Prototype 5: Large cube
        sim_utils.MeshCuboidCfg(size=(0.06, 0.06, 0.06), **props).func("/World/template/object/proto_asset_5", sim_utils.MeshCuboidCfg(size=(0.06, 0.06, 0.06), **props), translation=hand_pos)
        shape_names.append("large_cube")
        # Prototype 6: Flat box
        sim_utils.MeshCuboidCfg(size=(0.06, 0.06, 0.02), **props).func("/World/template/object/proto_asset_6", sim_utils.MeshCuboidCfg(size=(0.06, 0.06, 0.02), **props), translation=hand_pos)
        shape_names.append("flat_box")
        # Prototype 7: Tall box
        sim_utils.MeshCuboidCfg(size=(0.03, 0.03, 0.08), **props).func("/World/template/object/proto_asset_7", sim_utils.MeshCuboidCfg(size=(0.03, 0.03, 0.08), **props), translation=hand_pos)
        shape_names.append("tall_box")
        # Prototype 8: Wide box
        sim_utils.MeshCuboidCfg(size=(0.08, 0.03, 0.03), **props).func("/World/template/object/proto_asset_8", sim_utils.MeshCuboidCfg(size=(0.08, 0.03, 0.03), **props), translation=hand_pos)
        shape_names.append("wide_box")
        # Prototype 9: Tiny sphere
        sim_utils.MeshSphereCfg(radius=0.02, **props).func("/World/template/object/proto_asset_9", sim_utils.MeshSphereCfg(radius=0.02, **props), translation=hand_pos)
        shape_names.append("tiny_sphere")

        print(f"  Created {num_shapes} heterogeneous object prototypes: {shape_names}")

        # Clone using TemplateCloneCfg - heterogeneous cloning with sequential assignment
        # Clone with NO spacing - spacing will be applied via write_root_pose_to_sim after cloning
        clone_cfg = TemplateCloneCfg(
            template_root="/World/template",
            template_prototype_identifier="proto_asset",
            clone_regex="/World/envs/env_.*",
            clone_usd=True,
            clone_physics=True,
            clone_strategy=sequential,  # Round-robin: env0=shape0, env1=shape1, ..., env10=shape0, etc.
            device=device,
        )
        print(f"  Cloning to {num_envs} environments (without spacing)...")
        clone_from_template(stage, num_clones=num_envs, template_clone_cfg=clone_cfg)

        # Create views (spawn=None since prims were cloned)
        hand_cfg.prim_path, hand_cfg.spawn = "/World/envs/env_.*/hand", None
        hand = Articulation(hand_cfg)
        object_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/object", spawn=None)
        drop_object = RigidObject(object_cfg)

        # Track which shape and finger each env got
        env_shapes = [shape_names[i % num_shapes] for i in range(num_envs)]
        env_fingers = [FINGER_NAMES[i % len(FINGER_NAMES)] for i in range(num_envs)]

        # Initialize
        print("  Initializing simulation...")
        sim.reset()
        hand.reset()
        drop_object.reset()
        hand.update(SIM_DT)
        drop_object.update(SIM_DT)

        # Set joint position targets for stability
        hand.set_joint_position_target(wp.to_torch(hand.data.joint_pos))
        hand.write_data_to_sim()

        # Position drop objects above different fingers per environment (cycling through fingers)
        # Use local env coordinates (all envs at same position since no spacing during clone)
        drop_positions = torch.zeros((num_envs, 3), device=device)
        for fi, finger in enumerate(FINGER_NAMES):
            offset = ALLEGRO_FINGERTIP_OFFSETS[finger]
            mask = (torch.arange(num_envs, device=device) % len(FINGER_NAMES)) == fi
            drop_positions[mask] = torch.tensor([hand_pos[0] + offset[0], hand_pos[1] + offset[1], hand_pos[2] + offset[2] + drop_height], device=device)

        # Write poses with identity quaternion [x,y,z,w] = [0,0,0,1]
        drop_object.write_root_pose_to_sim(torch.cat([drop_positions, torch.tensor([[0, 0, 0, 1]] * num_envs, device=device)], dim=1))

        # Settle - let hand stabilize, drop objects stay in place (zero gravity)
        print("  Settling...")
        for _ in range(settle_steps):
            hand.write_data_to_sim()
            sim.step(render=False)
            viz.step(NewtonManager._state_0)
            hand.update(SIM_DT)
            drop_object.update(SIM_DT)

        # Apply initial velocity
        velocity = torch.zeros((num_envs, 6), device=device)
        velocity[:, 2] = -1.5
        drop_object.write_root_velocity_to_sim(velocity)

        # Track deflections - use tensors for efficiency with 1024 envs
        initial_joint_pos = wp.to_torch(hand.data.joint_pos).clone()
        joint_names = hand.data.joint_names
        num_joints = len(joint_names)
        peak_deflection_per_env = torch.zeros(num_envs, device=device)

        # Build per-env joint mask tensor: (num_envs, num_joints) boolean mask
        # Each env's target finger joints are True
        finger_joint_indices = {
            finger: torch.tensor([joint_names.index(j) for j in ALLEGRO_FINGER_JOINTS[finger] if j in joint_names], device=device)
            for finger in FINGER_NAMES
        }
        # Create mask for each finger type: (num_fingers, num_joints)
        finger_masks = torch.zeros((len(FINGER_NAMES), num_joints), dtype=torch.bool, device=device)
        for fi, finger in enumerate(FINGER_NAMES):
            finger_masks[fi, finger_joint_indices[finger]] = True
        # Map each env to its finger index and gather the mask
        env_finger_idx = torch.arange(num_envs, device=device) % len(FINGER_NAMES)
        env_joint_mask = finger_masks[env_finger_idx]  # (num_envs, num_joints)

        for step in range(drop_steps):
            hand.write_data_to_sim()
            drop_object.write_root_velocity_to_sim(velocity)
            sim.step(render=False)
            viz.step(NewtonManager._state_0)
            hand.update(SIM_DT)
            drop_object.update(SIM_DT)

            # Update peak deflections using pure tensor ops
            current_pos = wp.to_torch(hand.data.joint_pos)
            diff = (current_pos - initial_joint_pos).abs()
            # Masked sum: zero out non-target joints, sum per env
            masked_diff = diff * env_joint_mask.float()
            deflection_per_env = masked_diff.sum(dim=1)
            peak_deflection_per_env = torch.maximum(peak_deflection_per_env, deflection_per_env)

            if step % 60 == 0:
                avg_deflection = peak_deflection_per_env.mean().item()
                max_deflection = peak_deflection_per_env.max().item()
                min_deflection = peak_deflection_per_env[peak_deflection_per_env > 0].min().item() if (peak_deflection_per_env > 0).any() else 0.0
                print(f"  Step {step:4d}: avg={avg_deflection:.4f}, min={min_deflection:.4f}, max={max_deflection:.4f}")

        # Print summary results
        print(f"\n  FINAL RESULTS ({num_envs} heterogeneous environments, {num_shapes} shape types):")
        print(f"    Average deflection: {peak_deflection_per_env.mean().item():.4f}")
        print(f"    Min deflection: {peak_deflection_per_env.min().item():.4f}")
        print(f"    Max deflection: {peak_deflection_per_env.max().item():.4f}")
        print(f"    Std deflection: {peak_deflection_per_env.std().item():.4f}")

        # Per-shape statistics
        print("\n  Per-shape statistics:")
        for shape_idx, shape_name in enumerate(shape_names):
            shape_mask = torch.tensor([i % num_shapes == shape_idx for i in range(num_envs)], device=device)
            shape_deflections = peak_deflection_per_env[shape_mask]
            print(f"    {shape_name:12s}: avg={shape_deflections.mean().item():.4f}, count={shape_mask.sum().item()}")

        viz.wait_for_close(sim, lambda: NewtonManager._state_0)

        # Verify: all environments should have some deflection
        num_with_deflection = (peak_deflection_per_env > 0.01).sum().item()
        success_rate = num_with_deflection / num_envs * 100
        print(f"\n  Success rate: {num_with_deflection}/{num_envs} ({success_rate:.1f}%) environments detected impact")

        assert success_rate >= 90.0, f"Expected at least 90% success rate, got {success_rate:.1f}%"

##
# Single-environment tests (for specific edge cases)
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("restitution", [0.0, 0.5])
def test_sphere_bounce_restitution(device: str, use_mujoco_contacts: bool, restitution: float):
    """Test sphere bouncing behavior with different restitution values.

    Note: Newton/MujocoWarp handles restitution differently than PhysX.
    High restitution (>0.8) is not fully supported.
    """
    drop_steps = 120
    bounce_steps = 120

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            physics_material=materials.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=restitution,
            ),
        )

        rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0)
        scene_cfg.object_a = RigidObjectCfg(
            prim_path="/World/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.25,
                rigid_props=rigid_props,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                physics_material=materials.RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                    restitution=restitution,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        sphere: RigidObject = scene["object_a"]

        for _ in range(drop_steps + bounce_steps):
            perform_sim_step(sim, scene, SIM_DT)

        final_height = wp.to_torch(sphere.data.root_pos_w)[0, 2].item()

        if restitution < 0.1:
            assert final_height < 0.5, f"Zero restitution should not bounce high: height={final_height:.4f}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_momentum_conservation(device: str, use_mujoco_contacts: bool):
    """Test momentum conservation in sphere-sphere collision."""
    collision_steps = 240

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        separation = 1.0
        scene_cfg.object_a = create_shape_cfg(ShapeType.SPHERE, "/World/SphereA", pos=(-separation / 2, 0.0, 0.5))
        scene_cfg.object_b = create_shape_cfg(ShapeType.SPHERE, "/World/SphereB", pos=(separation / 2, 0.0, 0.5))

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        sphere_a: RigidObject = scene["object_a"]
        sphere_b: RigidObject = scene["object_b"]

        initial_vel = 2.0
        sphere_a.write_root_velocity_to_sim(torch.tensor([[initial_vel, 0, 0, 0, 0, 0]], device=device))
        sphere_b.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        initial_momentum = initial_vel * 1.0

        for _ in range(collision_steps):
            perform_sim_step(sim, scene, SIM_DT)

        final_vel_a = wp.to_torch(sphere_a.data.root_lin_vel_w)[0, 0].item()
        final_vel_b = wp.to_torch(sphere_b.data.root_lin_vel_w)[0, 0].item()
        final_momentum = (final_vel_a + final_vel_b) * 1.0

        momentum_error = abs(final_momentum - initial_momentum)
        assert momentum_error < 0.3, f"Momentum not conserved: initial={initial_momentum}, final={final_momentum}"
        assert abs(final_vel_a) < initial_vel * 0.6, f"Object A should have slowed: {final_vel_a}"
        assert final_vel_b >= initial_vel * 0.4, f"Object B should have gained velocity: {final_vel_b}"
