# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the visualization marker registry."""

from __future__ import annotations

import copy
import gc
from types import SimpleNamespace

import pytest

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry

pytestmark = pytest.mark.unit


class _Owner:
    """Minimal debug-visualization owner; a real class so it can be weak-referenced."""

    def __init__(self) -> None:
        self.calls = 0

    def _debug_vis_callback(self, event) -> None:
        self.calls += 1


def test_add_and_clear_debug_vis_callback():
    """Registering returns an id, and clearing removes it and resets the owner's handle."""
    registry = VisMarkerRegistry()
    owner = _Owner()

    owner._debug_vis_handle = registry.add_debug_vis_callback(owner)
    assert isinstance(owner._debug_vis_handle, str)

    registry.dispatch_callbacks()
    assert owner.calls == 1

    registry.clear_debug_vis_callback(owner)
    assert owner._debug_vis_handle is None

    registry.dispatch_callbacks()
    assert owner.calls == 1


def test_dispatch_drops_callbacks_whose_owner_was_collected():
    """A collected owner must not abort dispatch for the callbacks that are still live.

    Callbacks hold a weak proxy, so an owner freed without deregistering leaves an entry
    that raises ``ReferenceError`` when invoked.
    """
    registry = VisMarkerRegistry()
    live = _Owner()
    dead = _Owner()

    registry.add_debug_vis_callback(live)
    registry.add_debug_vis_callback(dead)

    del dead
    gc.collect()

    registry.dispatch_callbacks()
    assert live.calls == 1

    # the stale entry is gone, so later dispatches keep working
    registry.dispatch_callbacks()
    assert live.calls == 2


def test_registry_resolves_only_structural_copies_of_planned_cfgs():
    """Cfg copies share plan-authored state, while changed copies remain unplanned."""

    class _MarkerState:
        def __init__(self, cfg):
            self.cfg = cfg

    cfg = VisualizationMarkersCfg(prim_path="/Visuals/Test", markers={"sphere": SimpleNamespace(radius=1.0)})
    registry = VisMarkerRegistry()
    registry.prepare((cfg,), (_MarkerState,))

    planned_cfg, (state,) = registry.get(copy.deepcopy(cfg))
    assert planned_cfg is cfg
    assert state.cfg is cfg

    changed = copy.deepcopy(cfg)
    changed.markers["sphere"].radius = 2.0
    with pytest.raises(ValueError, match="not covered by the clone plan"):
        registry.get(changed)


def test_registry_returns_inert_state_only_when_explicitly_configured_without_backends():
    """An empty backend selection is distinct from an unconfigured clone lifecycle."""
    cfg = VisualizationMarkersCfg(prim_path="/Visuals/Test", markers={"sphere": SimpleNamespace(radius=1.0)})
    registry = VisMarkerRegistry()

    with pytest.raises(RuntimeError, match="not been prepared"):
        registry.get(cfg)

    registry.prepare((), ())
    assert registry.get(cfg) == (cfg, ())


def test_registry_backend_selection_is_fixed_once():
    """Marker backend ownership cannot change inside one clone lifecycle."""
    registry = VisMarkerRegistry()
    registry.prepare((), ())

    with pytest.raises(RuntimeError, match="already prepared"):
        registry.prepare((), ())
