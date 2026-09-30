# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-only binding shape tests that have no PhysX equivalent."""

from __future__ import annotations

import pytest
import warp as wp

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet  # noqa: E402

wp.init()


def test_mock_binding_set_rigid_object_shapes():
    pytest.importorskip("isaaclab_ov.tensor_types").RIGID_BODY_POSE  # gates on wheel
    from isaaclab_ov import tensor_types as TT
    from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet

    bindings = MockOvPhysxBindingSet(
        num_instances=4,
        num_joints=0,
        num_bodies=1,
        asset_kind="rigid_object",
    )
    assert bindings.bindings[TT.RIGID_BODY_POSE].shape == (4, 7)
    assert bindings.bindings[TT.RIGID_BODY_VELOCITY].shape == (4, 6)
    assert bindings.bindings[TT.RIGID_BODY_WRENCH].shape == (4, 9)
    assert bindings.bindings[TT.RIGID_BODY_MASS].shape == (4,)
    assert bindings.bindings[TT.RIGID_BODY_INERTIA].shape == (4, 9)
    # Articulation-only bindings must be absent
    assert TT.DOF_POSITION not in bindings.bindings
    assert TT.LINK_WRENCH not in bindings.bindings


def test_mock_binding_read_preserves_structured_warp_dtype():
    """Mock bindings should read flat component data into structured Warp arrays."""
    from isaaclab_ov import tensor_types as TT

    bindings = MockOvPhysxBindingSet(num_instances=4, num_joints=1, num_bodies=2)
    destination = wp.empty((4, 2), dtype=wp.spatial_vectorf, device="cpu")

    bindings.bindings[TT.LINK_VELOCITY].read(destination)

    assert destination.numpy().shape == (4, 2, 6)
