# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantics tests for the standalone symmetry reducer.

Asset/cfg/orbit-sampling tests are host-side and always run. The Warp
``reduce_orientation`` behavior (cyclic / discrete / semantic / heterogeneous) is
gated on CUDA availability, mirroring the rest of the warp-backed multi_task
tests. Run one file per invocation::

    ./isaaclab.sh -p -m pytest source/isaaclab_tasks/isaaclab_tasks/core/multi_task/tests/test_symmetry.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.factory.assembly_profile import SymmetryOrbit
from isaaclab_tasks.core.multi_task.factory.assembly_profile_cfg import SymmetryOrbitCfg
from isaaclab_tasks.core.multi_task.utils.symmetry.asset_symmetry import (
    KIND_CYCLIC,
    KIND_GENERAL,
    AssetSymmetry,
    AxisSymmetry,
    SymmetryElement,
    SymmetryElementTable,
)
from isaaclab_tasks.core.multi_task.utils.symmetry.symmetry_cfg import (
    AssetSymmetryCfg,
    AxisSymmetryCfg,
    SemanticSymmetryCfg,
)

_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# config classes + table compilation (host-side, always run)
# ---------------------------------------------------------------------------


def test_axis_symmetry_cfg_constructs_single_axis_element():
    cfg = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=4)])
    assert len(cfg.elements) == 1
    assert isinstance(cfg.elements[0], AxisSymmetryCfg)
    assert cfg.elements[0].order == 4


def test_table_entry_classifies_cyclic_vs_general():
    continuous = AssetSymmetry(AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=0)])).table
    discrete = AssetSymmetry(AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=4)])).table
    semantic = AssetSymmetry(
        AssetSymmetryCfg(elements=[SemanticSymmetryCfg(offsets=[(0, 0, 0, 1), (0, 1, 0, 0)])]),
    ).table

    assert continuous.kind == KIND_CYCLIC
    assert continuous.order == 0
    assert discrete.kind == KIND_CYCLIC
    assert discrete.order == 4
    assert semantic.kind == KIND_GENERAL
    assert semantic.offset_quat.shape[1] == 4


def test_symmetry_aggregates_cfg_table_entries():
    import warp as wp

    called = []
    cfg = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=1)])

    class RecordingAssetSymmetry(AssetSymmetry):
        def __init__(self, inner_cfg):
            called.append(inner_cfg)
            super().__init__(inner_cfg)

    from isaaclab_tasks.core.multi_task.utils.symmetry import Symmetry

    wp.init()
    cfg.class_type = RecordingAssetSymmetry
    symmetry = Symmetry([cfg], "cpu")

    assert called[0] is cfg
    assert symmetry.num_types == 1
    assert symmetry._single_cyclic
    assert symmetry._offset_quat.shape[0] == 0


def test_axis_element_uses_class_type_hook():
    called = []
    axis = AxisSymmetryCfg(order=4)

    class RecordingAxisSymmetry(AxisSymmetry):
        def __init__(self, inner_cfg):
            called.append(inner_cfg)
            super().__init__(inner_cfg)

    axis.class_type = RecordingAxisSymmetry
    entry = AssetSymmetry(AssetSymmetryCfg(elements=[axis])).table

    assert isinstance(called[0], AxisSymmetryCfg)
    assert called[0].order == 4
    assert entry.kind == KIND_CYCLIC
    assert entry.order == 4


def test_custom_element_cfg_compiles_without_asset_dispatch():
    called = []

    class CustomSymmetryElement(SymmetryElement):
        def __init__(self, element):
            called.append(element)
            self.table = SymmetryElementTable(
                kind=KIND_GENERAL,
                axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                order=0,
                offset_quat=np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            )

    class CustomSymmetryElementCfg:
        def __init__(self):
            self.class_type = CustomSymmetryElement

    custom = CustomSymmetryElementCfg()
    entry = AssetSymmetry(AssetSymmetryCfg(elements=[custom])).table

    assert isinstance(called[0], CustomSymmetryElementCfg)
    assert entry.kind == KIND_GENERAL
    assert isinstance(entry.offset_quat, np.ndarray)
    assert entry.offset_quat.shape == (2, 4)


def test_continuous_axis_cannot_mix_with_other_elements():
    bad = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=0), AxisSymmetryCfg(order=4)])
    with pytest.raises(ValueError, match="continuous"):
        AssetSymmetry(bad)


def test_empty_elements_rejected():
    with pytest.raises(ValueError, match="empty"):
        AssetSymmetry(AssetSymmetryCfg(elements=[]))


@pytest.mark.parametrize(
    "cfg,match",
    [
        (AssetSymmetryCfg(elements=[AxisSymmetryCfg(axis=(0.0, 0.0, 0.0), order=0)]), "axis"),
        (AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=-1)]), "order"),
        (AssetSymmetryCfg(elements=[SemanticSymmetryCfg(offsets=[(0.0, 0.0, 0.0, 0.0)])]), "quaternions"),
    ],
)
def test_invalid_symmetry_entries_rejected(cfg: AssetSymmetryCfg, match: str):
    with pytest.raises(ValueError, match=match):
        AssetSymmetry(cfg)


def test_symmetry_orbit_continuous_is_z_only_and_unit():
    _, q = SymmetryOrbit(SymmetryOrbitCfg(symmetry=AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=0)])))(4096, "cpu")
    assert torch.allclose(q[:, :2], torch.zeros_like(q[:, :2]), atol=1e-6)  # x,y == 0 (z-axis only)
    assert torch.allclose(q.norm(dim=-1), torch.ones(q.shape[0]), atol=1e-5)


@pytest.mark.parametrize("order,members", [(4, 4), (2, 2), (1, 1)])
def test_symmetry_orbit_discrete_member_count(order: int, members: int):
    _, q = SymmetryOrbit(SymmetryOrbitCfg(symmetry=AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=order)])))(
        4096, "cpu"
    )
    uniq = torch.unique((q * 1e4).round() / 1e4, dim=0)
    assert uniq.shape[0] == members


# ---------------------------------------------------------------------------
# symmetry reduction (warp, CUDA-gated)
# ---------------------------------------------------------------------------


def _reduce(symmetry, held, target, type_id):
    import warp as wp

    n = held.shape[0]
    err = torch.zeros(n, device="cuda:0")
    near = torch.zeros(n, 4, device="cuda:0")
    symmetry.reduce_orientation(
        wp.from_torch(held.contiguous(), dtype=wp.quatf),
        wp.from_torch(target.contiguous(), dtype=wp.quatf),
        wp.from_torch(type_id.contiguous(), dtype=wp.int32),
        wp.from_torch(err),
        wp.from_torch(near, dtype=wp.quatf),
    )
    wp.synchronize_device("cuda:0")
    return err


def _rz(theta: float, n: int) -> torch.Tensor:
    q = torch.zeros(n, 4, device="cuda:0")
    q[:, 2] = math.sin(theta / 2)
    q[:, 3] = math.cos(theta / 2)
    return q


@pytest.mark.skipif(not _CUDA, reason="symmetry reduce kernel requires CUDA + warp")
def test_reduce_cyclic_and_semantic_and_heterogeneous():
    import warp as wp

    import isaaclab.utils.math as mu

    from isaaclab_tasks.core.multi_task.utils.symmetry import Symmetry

    wp.init()
    n = 4096
    # types: 0 continuous, 1 4-fold, 2 2-fold, 3 semantic(identity+Ry180)
    symmetry = Symmetry(
        [
            AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=0)]),
            AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=4)]),
            AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=2)]),
            AssetSymmetryCfg(elements=[SemanticSymmetryCfg(offsets=[(0, 0, 0, 1), (0, 1, 0, 0)])]),
        ],
        "cuda:0",
    )
    t = torch.randn(n, 4, device="cuda:0")
    t = t / t.norm(dim=-1, keepdim=True)
    ry = torch.zeros(n, 4, device="cuda:0")
    ry[:, 1] = 1.0  # 180 about y

    def deg(held, tid):
        return math.degrees(
            _reduce(symmetry, held, t, torch.full((n,), tid, dtype=torch.int32, device="cuda:0")).mean()
        )

    assert deg(mu.quat_mul(t, _rz(2.39, n)), 0) < 0.5  # continuous: any yaw equivalent
    assert deg(mu.quat_mul(t, _rz(math.pi / 2, n)), 1) < 0.5  # 4-fold accepts Rz(90)
    assert abs(deg(mu.quat_mul(t, _rz(math.pi / 4, n)), 1) - 45.0) < 1.0  # 4-fold rejects Rz(45)
    assert deg(mu.quat_mul(t, _rz(math.pi, n)), 2) < 0.5  # 2-fold accepts Rz(180)
    assert abs(deg(mu.quat_mul(t, _rz(math.pi / 2, n)), 2) - 90.0) < 1.0  # 2-fold rejects Rz(90)
    assert deg(mu.quat_mul(t, ry), 3) < 0.5  # semantic accepts Ry(180)

    # heterogeneous flat batch: one launch, mixed type_id
    held = torch.cat([mu.quat_mul(t[: n // 2], _rz(math.pi / 2, n // 2)), mu.quat_mul(t[n // 2 :], ry[n // 2 :])])
    tid = torch.cat([torch.full((n // 2,), 1), torch.full((n // 2,), 3)]).to(torch.int32).to("cuda:0")
    err = _reduce(symmetry, held, t, tid)
    assert math.degrees(err[: n // 2].mean()) < 0.5  # 4-fold half aligned
    assert math.degrees(err[n // 2 :].mean()) < 0.5  # semantic half aligned
