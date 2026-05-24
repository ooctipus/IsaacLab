# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`isaaclab.cloner.replicate_registry`.

Pure Python; the ``SimulationContext`` singleton is monkeypatched with a stub so the
registry can be exercised without launching Isaac Sim.
"""

from __future__ import annotations

import pytest

from isaaclab.cloner import replicate_registry
from isaaclab.cloner.replicate_registry import get_replicate_ctx, replicate
from isaaclab.sim.simulation_context import SimulationContext


class _StubLayout:
    """Sentinel passed through to context constructors."""


class _StubSim:
    """Just enough of :class:`SimulationContext` for the registry to construct contexts."""

    def __init__(self, layout: object | None, stage: object) -> None:
        self._layout = layout
        self.stage = stage

    def get_stage_layout(self):
        return self._layout


class _RecordingCtx:
    """Context that records its construction args and a per-instance replicate() call count."""

    instances: list[_RecordingCtx] = []

    def __init__(self, layout, *, stage):
        self.layout = layout
        self.stage = stage
        self.replicate_calls = 0
        _RecordingCtx.instances.append(self)

    def replicate(self) -> None:
        self.replicate_calls += 1


class _OtherCtx(_RecordingCtx):
    """Distinct ctx class so we can assert per-class singletons."""


class _RaisingCtx:
    def __init__(self, layout, *, stage):
        pass

    def replicate(self) -> None:
        raise RuntimeError("ctx replicate boom")


@pytest.fixture(autouse=True)
def _clean_registry_and_sim():
    """Swap in a fresh stub sim per test and reset the module-level registry."""
    saved_instance = SimulationContext._instance
    SimulationContext._instance = _StubSim(_StubLayout(), stage=object())

    replicate_registry._replicate_ctxs.clear()
    _RecordingCtx.instances.clear()
    try:
        yield
    finally:
        replicate_registry._replicate_ctxs.clear()
        _RecordingCtx.instances.clear()
        SimulationContext._instance = saved_instance


def test_get_replicate_ctx_constructs_once_with_layout_and_stage():
    sim = SimulationContext._instance
    first = get_replicate_ctx(_RecordingCtx)
    second = get_replicate_ctx(_RecordingCtx)

    assert first is second
    assert first.layout is sim._layout
    assert first.stage is sim.stage
    assert len(_RecordingCtx.instances) == 1


def test_get_replicate_ctx_separates_singletons_per_class():
    a = get_replicate_ctx(_RecordingCtx)
    b = get_replicate_ctx(_OtherCtx)

    assert a is not b
    assert isinstance(a, _RecordingCtx) and not isinstance(a, _OtherCtx)
    assert isinstance(b, _OtherCtx)


def test_replicate_invokes_each_ctx_in_registration_order_then_clears():
    a = get_replicate_ctx(_RecordingCtx)
    b = get_replicate_ctx(_OtherCtx)

    replicate()

    assert a.replicate_calls == 1
    assert b.replicate_calls == 1
    assert replicate_registry._replicate_ctxs == {}

    fresh = get_replicate_ctx(_RecordingCtx)
    assert fresh is not a
    assert len(_RecordingCtx.instances) == 3  # a, b, fresh


def test_replicate_clears_registry_even_when_ctx_raises():
    get_replicate_ctx(_RaisingCtx)

    with pytest.raises(RuntimeError, match="ctx replicate boom"):
        replicate()

    assert replicate_registry._replicate_ctxs == {}


def test_get_replicate_ctx_without_simulation_context_raises():
    SimulationContext._instance = None
    with pytest.raises(RuntimeError, match="without an active SimulationContext"):
        get_replicate_ctx(_RecordingCtx)


def test_get_replicate_ctx_without_stage_layout_raises():
    SimulationContext._instance = _StubSim(layout=None, stage=object())
    with pytest.raises(RuntimeError, match="before a StageLayout was published"):
        get_replicate_ctx(_RecordingCtx)
