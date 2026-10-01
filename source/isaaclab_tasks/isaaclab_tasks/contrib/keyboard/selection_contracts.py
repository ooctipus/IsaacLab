# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP selection invariants over explicit model-world and prototype domains."""

from __future__ import annotations

import numpy as np

from .mujoco_selection import MuJoCoSelection
from .newton_selection import JOINT_COORD, JOINT_DOF, NewtonSelection, NewtonSelectionGroup


def require_count_per_world(selection, count: int) -> None:
    """Require an authored cardinality, including every currently empty prototype.

    Episode masks do not change topology cardinality. A MuJoCo prototype count
    describes every possible instance; a Newton count describes one model world.
    Neither vector is exposed as though the two index domains were interchangeable.
    """
    if isinstance(selection, MuJoCoSelection):
        counts = selection.prototype_selection_counts
    elif isinstance(selection, (NewtonSelection, NewtonSelectionGroup)):
        counts = selection.world_selection_counts
    else:
        raise TypeError("Unsupported task selection representation.")
    if any(value != count for value in counts):
        raise ValueError(f"Selection requires exactly {count} entries per world before participation masking.")


def require_same_world_domain(left, right) -> None:
    """Require identical logical-world placement without reading any GPU array."""
    if type(left) is not type(right):
        raise ValueError("Selections must use the same world domain.")
    if isinstance(left, (NewtonSelection, MuJoCoSelection)):
        same = left.owner is right.owner
    elif isinstance(left, NewtonSelectionGroup):
        same = left.world_bindings == right.world_bindings
    else:
        raise TypeError("Unsupported task selection representation.")
    if not same:
        raise ValueError("Selections must use the same world domain and bindings.")


def _require_scalar_joints(selection):
    model, joints = selection.owner.model, selection.joint_ids.numpy()
    if np.any(np.diff(model.joint_q_start.numpy())[joints] != 1) or np.any(
        np.diff(model.joint_qd_start.numpy())[joints] != 1
    ):
        raise ValueError("Scalar joint selectors cannot select components of multi-coordinate or multi-DOF joints.")


def require_scalar_joint_pair(coords, dofs) -> None:
    """Require corresponding scalar joints in the same world and policy-slot order.

    This is a preparation check. Model identity and placement must agree before
    comparing joint indices: equal integers from different models are unrelated.
    """
    if coords.index_domain != JOINT_COORD or dofs.index_domain != JOINT_DOF:
        raise ValueError("Scalar joint pairs require coordinate and DOF selectors.")
    require_same_world_domain(coords, dofs)
    if coords.width != dofs.width:
        raise ValueError("Scalar joint pairs require equal policy widths.")
    if isinstance(coords, MuJoCoSelection):
        # Every prepared prototype is checked, even if it has no live instances.
        for q, qd in zip(coords.parts, dofs.parts, strict=True):
            require_scalar_joint_pair(q, qd)
        return
    if isinstance(coords, NewtonSelectionGroup):

        def rows(selection):
            result = {}
            for part, worlds in selection.native_bindings:
                joints = part.joint_ids.numpy().reshape(len(part.world_selection_counts), part.width)
                for row, world in enumerate(worlds.tolist()):
                    result[world] = joints[row]
            return result

        qrows, drows = rows(coords), rows(dofs)
        if any(not np.array_equal(joints, drows[world]) for world, joints in qrows.items()):
            raise ValueError("Coordinate and DOF selectors must select the same ordered scalar joints per world.")
        for part, _ in coords.native_bindings:
            _require_scalar_joints(part)
        return
    if not isinstance(coords, NewtonSelection):
        raise TypeError("Unsupported task selection representation.")
    if coords.world_selection_counts != dofs.world_selection_counts or not np.array_equal(
        coords.joint_ids.numpy(), dofs.joint_ids.numpy()
    ):
        raise ValueError("Coordinate and DOF selectors must select the same ordered scalar joints per world.")
    _require_scalar_joints(coords)
