# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest
import warp as wp
from isaaclab_newton.assets.deformable_object import DeformableObject
from isaaclab_newton.assets.deformable_object.deformable_object import DeformableRegistryEntry
from isaaclab_newton.cloner.replicate import NewtonReplicateContext
from isaaclab_newton.sim.spawners.materials import NewtonDeformableMaterialCfg


class _FakeBuilder:
    def __init__(self):
        self.particle_count = 0
        self.cloth_meshes = []

    def add_cloth_mesh(self, **kwargs) -> None:
        self.cloth_meshes.append(kwargs)
        self.particle_count += len(kwargs["vertices"])


def _make_surface_entry() -> DeformableRegistryEntry:
    half_sqrt = math.sqrt(0.5)
    return DeformableRegistryEntry(
        prim_path="{ENV_REGEX_NS}/cloth",
        planned_worlds=(0, 1),
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        indices=[0, 1, 2],
        init_pos=(1.0, 0.0, 0.0),
        init_rot=(0.0, 0.0, half_sqrt, half_sqrt),
        deformable_type="surface",
    )


def _vec3_as_tuple(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def test_deformable_package_exports_public_symbols():
    """Test that deformable symbols are exported from the package root."""
    assert DeformableObject.__name__ == "DeformableObject"


def test_newton_material_defaults_match_registry_defaults():
    """Test that Newton material cfg defaults match the deformable registry defaults."""
    material_cfg = NewtonDeformableMaterialCfg()

    assert material_cfg.density == DeformableRegistryEntry.density
    assert material_cfg.particle_radius == DeformableRegistryEntry.particle_radius


def test_builder_hook_applies_env_quaternion_to_deformable_entry():
    """Test that deformable builder placement honors the environment quaternion."""
    entry = _make_surface_entry()
    builder = _FakeBuilder()
    half_sqrt = math.sqrt(0.5)

    entry.add_to_builder(
        builder,
        env_idx=0,
        env_position=[10.0, 20.0, 30.0],
        env_rotation=[0.0, 0.0, half_sqrt, half_sqrt],
    )

    mesh = builder.cloth_meshes[0]
    rotated_x_axis = wp.quat_rotate(mesh["rot"], wp.vec3(1.0, 0.0, 0.0))

    assert _vec3_as_tuple(mesh["pos"]) == pytest.approx((10.0, 21.0, 30.0))
    assert _vec3_as_tuple(rotated_x_axis) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_builder_hook_resets_entry_offsets_on_first_environment():
    """Test that repeated model rebuilds do not accumulate stale particle offsets."""
    entry = _make_surface_entry()
    builder = _FakeBuilder()
    identity = [0.0, 0.0, 0.0, 1.0]

    entry.add_to_builder(builder, 0, [0.0, 0.0, 0.0], identity)
    entry.add_to_builder(builder, 1, [1.0, 0.0, 0.0], identity)

    assert entry.particle_offsets == [0, 3]

    rebuilt_builder = _FakeBuilder()
    entry.add_to_builder(rebuilt_builder, 0, [0.0, 0.0, 0.0], identity)

    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_builder_hook_uses_only_exact_clone_plan_destinations():
    """Partial clone-plan coverage controls the simulated worlds."""
    entry = _make_surface_entry()
    entry.planned_worlds = (1, 3)
    builder = _FakeBuilder()
    identity = [0.0, 0.0, 0.0, 1.0]

    for world in range(4):
        entry.add_to_builder(builder, world, [float(world), 0.0, 0.0], identity)

    assert entry.particle_offsets == [0, 3]
    assert builder.particle_count == 6


def test_newton_cloner_exports_only_the_context_type():
    """The simulation registry replaces the backend-global context alias."""
    import isaaclab_newton.cloner as cloner

    assert cloner.NewtonReplicateContext is NewtonReplicateContext
    assert not hasattr(cloner, "PHYSICS_CONTEXT")
