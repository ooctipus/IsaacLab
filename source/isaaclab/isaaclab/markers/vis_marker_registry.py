# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry for visualization marker state."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any


class VisMarkerRegistry:
    """Tracks visualization marker callbacks and active marker groups."""

    def __init__(self):
        self._callbacks: dict[str, Callable[[Any], None]] = {}
        self._groups: list[tuple[Any, tuple[Any, ...]]] = []
        self._marker_types: tuple[type, ...] | None = None

    def prepare(self, cfgs: tuple[Any, ...], marker_types: tuple[type, ...]) -> None:
        """Construct marker backend state for one clone lifecycle.

        Args:
            cfgs: Plan-owned marker configurations.
            marker_types: Backend marker implementations selected by the active visualizers.
        """
        if self._marker_types is not None:
            raise RuntimeError("Visualization marker backends are already prepared.")
        self._marker_types = marker_types
        self._groups = [(cfg, tuple(marker_type(cfg) for marker_type in marker_types)) for cfg in cfgs]

    def add_callback(self, name: str, callback: Callable[[Any], None]) -> str:
        """Register a callback invoked before marker-capable visualizers step each render tick."""
        self._callbacks[name] = callback
        return name

    def add_debug_vis_callback(self, owner: Any) -> str:
        """Register an owner's debug visualization callback.

        Args:
            owner: Object implementing ``_debug_vis_callback(event)``.

        Returns:
            Callback identifier that can be passed to :meth:`remove_callback`.
        """
        callback_id = f"visualization_marker:{type(owner).__name__}:{id(owner)}"
        owner_ref = weakref.proxy(owner)
        return self.add_callback(callback_id, lambda event: owner_ref._debug_vis_callback(event))

    def clear_debug_vis_callback(self, owner: Any) -> None:
        """Clear an owner's registered debug visualization callback, if any."""
        callback_id = getattr(owner, "_debug_vis_handle", None)
        if callback_id is not None:
            self.remove_callback(callback_id)
            owner._debug_vis_handle = None

    def remove_callback(self, callback_id: str) -> None:
        """Remove a visualization marker callback if it exists."""
        self._callbacks.pop(callback_id, None)

    def dispatch_callbacks(self, event: Any = None) -> None:
        """Invoke all registered visualization marker callbacks.

        Callbacks hold a weak proxy to their owner. An owner collected without
        deregistering leaves a stale entry whose proxy raises on use, so drop those
        rather than letting one dead owner abort the whole dispatch.
        """
        for callback_id, callback in list(self._callbacks.items()):
            try:
                callback(event)
            except ReferenceError:
                self._callbacks.pop(callback_id, None)

    def get(self, cfg: Any) -> tuple[Any, tuple[Any, ...]]:
        """Return planned backend states, or an inert state when no marker backend was selected."""
        if self._marker_types is None:
            raise RuntimeError("Visualization marker backends have not been prepared by a clone lifecycle.")
        for planned_cfg, groups in self._groups:
            if planned_cfg == cfg:
                return planned_cfg, groups
        if self._marker_types == ():
            return cfg, ()
        raise ValueError(f"VisualizationMarkersCfg at {cfg.prim_path!r} is not covered by the clone plan.")

    def get_groups(self) -> tuple[Any, ...]:
        """Return all planned visualization marker backend states."""
        return tuple(group for _, groups in self._groups for group in groups)

    def close(self) -> None:
        """Close marker backend states and clear the registry."""
        for group in self.get_groups():
            group.close()
        self._callbacks.clear()
        self._groups.clear()
        self._marker_types = None
