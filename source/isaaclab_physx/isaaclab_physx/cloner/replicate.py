# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

from isaaclab import cloner


class PhysxReplicateContext:
    """Apply one clone-plan mapping through the PhysX replicator."""

    replicate_priority = 0
    clones_whole_env = True

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to register with the PhysX replicator.
        """
        self.stage = stage
        cache = UsdUtils.StageCache.Get()
        cached_id = cache.GetId(stage)
        self._stage_id = cached_id.ToLongInt() if cached_id.IsValid() else cache.Insert(stage).ToLongInt()
        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)
        self._replicator = None
        self._registered = False

    def replicate(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
        exclude_self_replication: bool = True,
    ) -> None:
        """Register the PhysX replicator for one flat clone mapping.

        Args:
            sources: Source prim paths.
            destinations: Destination path templates with ``"{}"`` for env id.
            env_ids: Environment indices.
            mapping: Bool/int mask selecting envs per source.
            positions: Optional per-environment world positions [m], unused by PhysX.
            quaternions: Optional per-environment orientations, unused by PhysX.
            exclude_self_replication: Whether to skip replicating a source prim onto itself
                when it also maps to other environments.
        """
        del positions, quaternions
        physx_queue: list[tuple[str, str, tuple[int, ...]]] = []

        if mapping.size(1) <= 1:
            return

        for i, src in enumerate(sources):
            worlds = env_ids[mapping[i].to(dtype=torch.bool)].tolist()
            if exclude_self_replication:
                matched = cloner.path.match(src, destinations[i])
                if matched is not None and matched.instance.isdigit():
                    filtered = [w for w in worlds if w != int(matched.instance)]
                    worlds = filtered if filtered else worlds
            physx_queue.append((src, destinations[i], tuple(map(int, worlds))))

        # Fully-heterogeneous 1:1 layouts have every source mapped only to its own
        # environment (no cross-env replication needed). Calling rep.replicate() once
        # per source with a single self-target is known to trigger intermittent native
        # heap corruption (double-free / SIGABRT) under mGPU, likely due to per-call
        # PhysX-internal allocations summing to a problematic total across processes.
        # For these layouts the source prims are already in their correct env positions
        # and PhysX can parse them from the stage without any replicator registration.
        def _is_self_only(src: str, destination: str, target_envs: tuple[int, ...]) -> bool:
            if len(target_envs) != 1:
                return False
            pre, suf = cloner.path.split(destination)
            return src == f"{pre}{target_envs[0]}{suf}"

        if all(_is_self_only(src, dst, envs) for src, dst, envs in physx_queue):
            return

        current_worlds: list[int] = []
        current_template: str = ""

        def attach_fn(_stage_id: int):
            return ["/World/template", "/World/envs"]

        def rename_fn(_replicate_path: str, i: int):
            return current_template.format(current_worlds[i])

        def attach_end_fn(_stage_id: int):
            nonlocal current_template
            for src, destination, target_envs in physx_queue:
                current_template = destination
                current_worlds[:] = target_envs
                if not current_worlds:
                    continue
                self._replicator.replicate(
                    _stage_id,
                    src,
                    len(current_worlds),
                    # TODO: envIds needs to support heterogeneous setup. for now, we rely on USD collision filtering
                    useEnvIds=False,
                    useFabricForReplication=False,
                )

        self._replicator = get_physx_replicator_interface()
        self._replicator.register_replicator(self._stage_id, attach_fn, attach_end_fn, rename_fn)
        self._registered = True

    def clear(self) -> None:
        """Unregister this stage's native PhysX replicator."""
        if self._registered:
            self._replicator.unregister_replicator(self._stage_id)
            self._registered = False
            self._replicator = None
