# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-cfg USD replication: :class:`UsdReplicateContext` + :func:`usd_replicate`.

The context collects ``Sdf.CopySpec`` requests queued by every cfg whose
:attr:`AssetBaseCfg.replicate` tuple includes :func:`usd_replicate`, then applies them
in path-depth order inside a single :class:`pxr.Sdf.ChangeBlock` when the scene calls
:func:`isaaclab.cloner.replicate`.
"""

from __future__ import annotations

from typing import Any

import torch

from pxr import Sdf, Usd

from isaaclab.layout import StageLayout

from .replicate_registry import get_replicate_ctx


class UsdReplicateContext:
    """Backend context that batches per-env USD ``CopySpec`` work into one ``ChangeBlock``.

    Lifecycle:

    * ``__init__`` opens an :class:`pxr.Sdf.ChangeBlock` so subsequent queue calls do not
      pay per-spec ``TfNotice`` cost.
    * :meth:`queue_copy_spec` records ``(source, dest)`` pairs while assets initialize.
    * :meth:`replicate` walks the queue in path-depth order (parents before children),
      authors each destination spec, copies the source spec when ``source != dest``, and
      closes the ``ChangeBlock``.

    The depth-sorted ordering reproduces the legacy :func:`isaaclab.cloner.usd_replicate`
    behaviour for nested namespaces -- ancestor specs land before descendants so each
    ``CopySpec`` resolves against an existing parent in the layer.
    """

    def __init__(self, layout: StageLayout, *, stage: Usd.Stage) -> None:
        self.layout = layout
        self.stage = stage
        self._copy_specs: list[tuple[str, str]] = []
        self._cb: Sdf.ChangeBlock = Sdf.ChangeBlock()
        self._cb.__enter__()

    def queue_copy_spec(self, source: str, dest: str) -> None:
        """Record a single ``Sdf.CopySpec`` request. Drained in :meth:`replicate`."""
        self._copy_specs.append((source, dest))

    def replicate(self) -> None:
        """Author every queued ``(source, dest)`` pair and close the ``ChangeBlock``."""
        try:
            root_layer = self.stage.GetRootLayer()
            # Sort by destination path depth so ancestors land before descendants. We count
            # ``/`` rather than reading ``Sdf.Path.pathElementCount`` because the latter has
            # an ``int | None`` typing surface in some USD stubs while the depth is fully
            # determined by the canonical path string.
            queue = sorted(self._copy_specs, key=lambda r: r[1].count("/"))
            for source, dest in queue:
                Sdf.CreatePrimInLayer(root_layer, dest)
                if source != dest:
                    Sdf.CopySpec(root_layer, Sdf.Path(source), root_layer, Sdf.Path(dest))
        finally:
            self._cb.__exit__(None, None, None)


def usd_replicate(cfg: Any, layout: StageLayout, cfg_idx: int) -> None:
    """Queue USD ``CopySpec`` work for every per-env slot ``cfg`` occupies.

    Resolves ``cfg``'s slot list from :attr:`StageLayout.source_ids`, picks the first
    env-scoped slot as the source-of-truth env, and queues a copy from that env's
    destination path to every other env's destination path. Shared-scope cfgs (first
    world is ``-1``) are a no-op: the spawn step already authored the prim at its single
    destination.

    Args:
        cfg: The cfg whose USD subtree should be replicated. Identity-keyed; never
            inspected here beyond locating its slot in ``layout``.
        layout: Active :class:`StageLayout` published by the scene.
        cfg_idx: Index of ``cfg`` in :attr:`StageLayout.sources`. The caller
            (``AssetBase.__init__``) supplies it to avoid an O(N) lookup.
    """
    ctx = get_replicate_ctx(UsdReplicateContext)

    slots = (layout.source_ids == cfg_idx).nonzero(as_tuple=False).flatten()
    if slots.numel() == 0:
        return

    # CSR decode: searchsorted(world_start, slot, right=True) - 2 is the env id, with -1
    # encoding the shared scope (world 0). See StageLayout for the full convention.
    worlds = torch.searchsorted(layout.world_start, slots, right=True) - 2
    src_world = int(worlds[0])
    if src_world < 0:
        return

    dest_template = layout.destinations[int(layout.destination_ids[slots[0]])]
    source_path = dest_template.format(src_world)
    for w in worlds.tolist():
        if w >= 0:
            ctx.queue_copy_spec(source_path, dest_template.format(w))
