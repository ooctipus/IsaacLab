# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The world layout expands a finalized seed to the model of every replicated world."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonWorldLayout
from newton._src.solvers.mujoco import ActuatorTarget, add_actuator_general
from newton.actuators import DrivePD, InputProcessorDelay
from newton.solvers import SolverMuJoCo


def _prototype(kind: int) -> newton.ModelBuilder:
    """A floating robot with a static post, or a fixed-base arm with particles and a cloth."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    base = builder.add_link(xform=wp.transform((0.1, 0.0, 0.5), wp.quat_rpy(0.1, 0.0, 0.2)), label="Robot/base")
    builder.add_shape_box(body=base, hx=0.1, hy=0.2, hz=0.3, label="Robot/base/box")
    tip = builder.add_link(xform=wp.transform((0.1, 0.0, 1.0), wp.quat_identity()), label="Robot/tip")
    builder.add_shape_sphere(body=tip, radius=0.05, label="Robot/tip/sphere")
    if kind == 0:
        root = builder.add_joint_free(child=base, label="Robot/root")
        builder.add_shape_capsule(body=-1, xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), label="Post")
    else:
        root = builder.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform((0.0, 0.3, 0.0), wp.quat_rpy(0.0, 0.0, 0.4))
        )
        builder.add_particle((0.5, 0.0, 0.2), (0.0, 0.1, 0.0), mass=1.0)
        builder.add_particle((0.6, 0.0, 0.2), (0.0, 0.0, 0.0), mass=1.0)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.5, 0.4),
            rot=wp.quat_rpy(0.2, 0.0, 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            add_springs=True,
        )
    hinge = builder.add_joint_revolute(parent=base, child=tip, axis=(0.0, 1.0, 0.0), label="Robot/hinge")
    builder.add_articulation([root, hinge], label="Robot")
    add_actuator_general(builder, ActuatorTarget.joint(hinge), gear=(2.0,))
    dof = builder.joint_qd_start[hinge]
    builder.add_actuator(
        DrivePD,
        index=dof,
        pos_index=builder.joint_q_start[hinge],
        kp=10.0 + kind,
        input_processors=[(InputProcessorDelay, {"delay_steps": 1 + kind})],
    )
    return builder


def _shared(builder: newton.ModelBuilder) -> None:
    SolverMuJoCo.register_custom_attributes(builder)
    frequency = newton.Model.AttributeFrequency.WORLD
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute("wind", float, frequency, default=0.5, namespace="t")
    )
    builder.add_ground_plane()
    builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.1, label="/World/Table")


def _arrays(model: newton.Model) -> dict[str, np.ndarray]:
    arrays = {name: value.numpy() for name, value in vars(model).items() if isinstance(value, wp.array)}
    arrays.pop("bvh_shape_enabled"), arrays.pop("_bvh_shape_active_leaves")  # scratch beyond the enabled count
    if not model.particle_color_groups:
        arrays.pop("particle_colors")  # uninitialized without a coloring
    return arrays


def _build(layout_ids: np.ndarray, prototypes: list[newton.ModelBuilder]) -> tuple[newton.Model, newton.Model]:
    """Return the model built from one world per prototype and the model of every replicated world."""
    worlds = len(layout_ids)
    rng = np.random.default_rng(0)
    xforms = np.zeros((worlds, 7), dtype=np.float32)
    xforms[:, :3] = rng.uniform(-5.0, 5.0, (worlds, 3))
    xforms[:, 6] = 1.0
    xforms[1, 3:] = np.asarray(wp.quat_rpy(0.0, 0.0, 0.7), dtype=np.float32)
    xforms[-1] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    prefixes = [f"/World/envs/env_{world}" for world in range(worlds)]

    full = newton.ModelBuilder()
    _shared(full)
    bounds = np.r_[0, np.flatnonzero(np.diff(layout_ids)) + 1, worlds]
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        full.replicate(
            prototypes[layout_ids[start]],
            int(stop - start),
            xforms=xforms[start:stop],
            label_prefixes=prefixes[start:stop],
        )
    # A shape added after the worlds; its default palette color would follow its builder index.
    full.add_shape_sphere(body=-1, radius=0.2, color=(0.5, 0.5, 0.5), label="/World/LateSite")
    expected = full.finalize(device="cpu")

    seed = newton.ModelBuilder()
    _shared(seed)
    used, first = np.unique(layout_ids, return_index=True)
    used = used[np.argsort(first)]
    for prototype_id, world in zip(used, np.sort(first), strict=True):
        seed.begin_world()
        seed.add_builder(prototypes[prototype_id], label_prefix=prefixes[world])
        seed.end_world()
    seed.add_shape_sphere(body=-1, radius=0.2, color=(0.5, 0.5, 0.5), label="/World/LateSite")
    seed_of = np.zeros(2, dtype=np.int32)
    seed_of[used] = np.arange(len(used))
    layout = NewtonWorldLayout()
    layout.assign(seed_of[layout_ids], xforms, prefixes, tuple(prefixes[w] for w in np.sort(first)), np.arange(worlds))
    return layout.build_model(seed, device="cpu"), expected


@pytest.mark.parametrize("layout_ids", [[0, 0, 0, 0], [1, 0, 0, 1, 1, 0]], ids=["one-prototype", "interleaved"])
def test_seed_expands_to_the_replicated_model(layout_ids):
    """Expanding one finalized world per prototype equals finalizing every replicated world, bit for bit."""
    model, expected = _build(np.asarray(layout_ids), [_prototype(0), _prototype(1)])

    assert model.world_count == expected.world_count
    for name in ("body_label", "joint_label", "shape_label", "articulation_label"):
        assert getattr(model, name) == getattr(expected, name)
    assert model.body_shapes == expected.body_shapes
    np.testing.assert_array_equal(
        model._shape_collision_filter_pairs._packed, expected._shape_collision_filter_pairs._packed
    )
    np.testing.assert_array_equal(model._shape_contact_pair_data.counts, expected._shape_contact_pair_data.counts)
    np.testing.assert_array_equal(model.t.wind.numpy(), expected.t.wind.numpy())
    for name, value in vars(expected.mujoco).items():
        if isinstance(value, wp.array):
            assert getattr(model.mujoco, name).numpy().tobytes() == value.numpy().tobytes(), name
    assert len(model.actuators) == len(expected.actuators)
    for actuator, reference in zip(model.actuators, expected.actuators, strict=True):
        for name in ("indices", "pos_indices", "actuator_indices"):
            assert getattr(actuator, name).numpy().tolist() == getattr(reference, name).numpy().tolist(), name
        for part, wanted_part in (
            (actuator.drive, reference.drive),
            *zip(actuator.input_processors, reference.input_processors),
        ):
            for name, value in vars(wanted_part).items():
                if isinstance(value, wp.array):
                    assert getattr(part, name).numpy().tobytes() == value.numpy().tobytes(), name
    for name, value in vars(expected.soft_mesh_adjacency_device).items():
        if isinstance(value, wp.array):
            assert getattr(model.soft_mesh_adjacency_device, name).numpy().tobytes() == value.numpy().tobytes(), name
    actual, wanted = _arrays(model), _arrays(expected)
    assert actual.keys() == wanted.keys()
    for name, value in wanted.items():
        assert actual[name].tobytes() == value.tobytes(), name


def test_expanded_coloring_is_valid():
    """Each world keeps its prototype's coloring; together they color every world as validly as one coloring pass.

    The coloring graph is the one :meth:`newton.ModelBuilder.color` builds: particles of a triangle and the opposite
    vertices of a bending edge are neighbors, and so are the parent and child bodies of a joint. The worlds' groups
    are paired as replication pairs them, so the expanded coloring may group colors differently from a coloring of all
    worlds at once, with as many colors.
    """
    prototypes = [_prototype(0), _prototype(1)]
    for prototype in prototypes:
        prototype.color(include_bending=True)
    model, expected = _build(np.asarray([1, 0, 0, 1, 1, 0]), prototypes)

    tris = model.tri_indices.numpy()
    edges = model.edge_indices.numpy()
    joints = np.stack((model.joint_parent.numpy(), model.joint_child.numpy()), axis=1)
    for kind, neighbors in (
        ("particle", np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]], edges[:, [0, 1]]))),
        ("body", joints[np.all(joints >= 0, axis=1)]),
    ):
        colors = getattr(model, f"{kind}_colors").numpy()
        groups = [group.numpy() for group in getattr(model, f"{kind}_color_groups")]
        assert len(groups) == len(getattr(expected, f"{kind}_color_groups"))
        assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(colors)))
        for color, group in enumerate(groups):
            assert np.all(colors[group] == color)
        neighbors = neighbors[np.all(neighbors >= 0, axis=1)]
        assert len(neighbors) and not np.any(colors[neighbors[:, 0]] == colors[neighbors[:, 1]]), kind
