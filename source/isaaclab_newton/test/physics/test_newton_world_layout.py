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


def _prototype(kind: int) -> newton.ModelBuilder:
    """A floating robot with a static post, or a fixed-base arm with particles."""
    builder = newton.ModelBuilder()
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
    hinge = builder.add_joint_revolute(parent=base, child=tip, axis=(0.0, 1.0, 0.0), label="Robot/hinge")
    builder.add_articulation([root, hinge], label="Robot")
    return builder


def _shared(builder: newton.ModelBuilder) -> None:
    builder.add_ground_plane()
    builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.1, label="/World/Table")


def _arrays(model: newton.Model) -> dict[str, np.ndarray]:
    arrays = {name: value.numpy() for name, value in vars(model).items() if isinstance(value, wp.array)}
    arrays.pop("bvh_shape_enabled"), arrays.pop("_bvh_shape_active_leaves")  # scratch beyond the enabled count
    if not model.particle_color_groups:
        arrays.pop("particle_colors")  # uninitialized without a coloring
    return arrays


@pytest.mark.parametrize("layout_ids", [[0, 0, 0, 0], [1, 0, 0, 1, 1, 0]], ids=["one-prototype", "interleaved"])
def test_seed_expands_to_the_replicated_model(layout_ids):
    """Expanding one finalized world per prototype equals finalizing every replicated world, bit for bit."""
    layout_ids = np.asarray(layout_ids)
    prototypes = [_prototype(0), _prototype(1)]
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
    model = layout.build_model(seed, device="cpu")

    assert model.world_count == expected.world_count
    for name in ("body_label", "joint_label", "shape_label", "articulation_label"):
        assert getattr(model, name) == getattr(expected, name)
    assert model.body_shapes == expected.body_shapes
    np.testing.assert_array_equal(
        model._shape_collision_filter_pairs._packed, expected._shape_collision_filter_pairs._packed
    )
    np.testing.assert_array_equal(model._shape_contact_pair_data.counts, expected._shape_contact_pair_data.counts)
    actual, wanted = _arrays(model), _arrays(expected)
    assert actual.keys() == wanted.keys()
    for name, value in wanted.items():
        assert actual[name].tobytes() == value.tobytes(), name
