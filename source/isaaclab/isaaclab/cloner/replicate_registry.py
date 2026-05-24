# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy backend-replication context registry.

A backend-specific replication context (e.g. ``UsdReplicateContext``,
``PhysxReplicateContext``, ``NewtonReplicateContext``) collects the work queued by per-cfg
``replicate`` callables during :meth:`AssetBase.__init__` and applies it in one shot when
the scene calls :func:`replicate`.

The registry holds at most one instance per context class; it is populated on first
:func:`get_replicate_ctx` lookup and cleared by :func:`replicate`.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from isaaclab.sim.simulation_context import SimulationContext


class _ReplicateContext(Protocol):
    """Structural contract every backend replication context must satisfy.

    The constructor takes the active :class:`~isaaclab.layout.StageLayout` and the active
    USD stage; the stage is keyword-only so backends that do not need it (e.g. Newton site
    queues) can declare ``stage=None`` defaults. ``replicate()`` is the single drain entry
    invoked by :func:`replicate` after every per-cfg queue call has run.
    """

    def __init__(self, layout: Any, *, stage: Any) -> None: ...

    def replicate(self) -> None: ...


T = TypeVar("T", bound=_ReplicateContext)

_replicate_ctxs: dict[type, _ReplicateContext] = {}


def get_replicate_ctx(ctx_cls: type[T]) -> T:
    """Return the singleton context of ``ctx_cls``, constructing it on first use.

    On first call the context is constructed via ``ctx_cls(layout, stage=stage)`` where
    ``layout`` is the active :class:`~isaaclab.layout.StageLayout` and ``stage`` is the
    active USD stage, both pulled from the active
    :class:`~isaaclab.sim.SimulationContext`. Subsequent calls return the cached instance
    until :func:`replicate` clears the registry.

    Args:
        ctx_cls: Backend-specific context class.

    Returns:
        The cached singleton instance of ``ctx_cls``.

    Raises:
        RuntimeError: If no :class:`~isaaclab.sim.SimulationContext` is active or no
            :class:`~isaaclab.layout.StageLayout` has been published.
    """
    if ctx_cls in _replicate_ctxs:
        return _replicate_ctxs[ctx_cls]

    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError(f"cloner.get_replicate_ctx({ctx_cls.__name__}) called without an active SimulationContext.")
    layout = sim.get_stage_layout()
    if layout is None:
        raise RuntimeError(
            f"cloner.get_replicate_ctx({ctx_cls.__name__}) called before a StageLayout was published; "
            "InteractiveScene publishes one in _setup_scene before constructing assets."
        )

    ctx = ctx_cls(layout, stage=sim.stage)
    _replicate_ctxs[ctx_cls] = ctx
    return ctx


def replicate() -> None:
    """Run replication on every registered backend context, then clear the registry.

    Each context is finalized in registration order by calling its ``replicate()`` method.
    The registry is cleared even if a context raises so the next scene build starts clean.
    """
    try:
        for ctx in _replicate_ctxs.values():
            ctx.replicate()
    finally:
        _replicate_ctxs.clear()
