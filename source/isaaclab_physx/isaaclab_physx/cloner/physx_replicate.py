# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import torch

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

logger = logging.getLogger(__name__)


def _set_scene_partitions(
    stage: Usd.Stage,
    env_prim_paths: list[str],
    use_fabric: bool = False,
) -> None:
    """Set ``primvars:omni:scenePartition`` on each environment prim.

    The PhysX replicator uses this primvar to discover environment IDs
    implicitly, enabling GPU-accelerated per-environment collision isolation
    without requiring explicit collision groups.

    For env_0 the primvar is always written via USD (the source prim lives in
    USD even when Fabric cloning is active).  For env_1+ the primvar is written
    via Fabric/USDRT when ``use_fabric`` is True, since those prims may only
    exist in Fabric after replication.

    Args:
        stage: USD stage containing the environment prims.
        env_prim_paths: Ordered list of environment prim paths
            (e.g. ``["/World/envs/env_0", ..., "/World/envs/env_N"]``).
        use_fabric: Whether Fabric cloning was used for env_1+.
    """
    attr_name = "primvars:omni:scenePartition"

    if env_prim_paths:
        prim = stage.GetPrimAtPath(env_prim_paths[0])
        if prim.IsValid():
            prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Token).Set("env_partition_0")

    if use_fabric and len(env_prim_paths) > 1:
        import omni.usdrt

        usdrt_stage = omni.usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().Insert(stage).ToLongInt())
        for i, path in enumerate(env_prim_paths[1:], start=1):
            usdrt_prim = usdrt_stage.GetPrimAtPath(path)
            if usdrt_prim.IsValid():
                attr = usdrt_prim.CreateAttribute(attr_name, omni.usdrt.Sdf.ValueTypeNames.Token)
                attr.Set(f"env_partition_{i}")
    else:
        for i, path in enumerate(env_prim_paths[1:], start=1):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Token).Set(f"env_partition_{i}")


def physx_replicate(
    stage: Usd.Stage,
    sources: list[str],  # e.g. ["/World/Template/A", "/World/Template/B"]
    destinations: list[str],  # e.g. ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"]
    env_ids: torch.Tensor,  # env_ids
    mapping: torch.Tensor,  # (num_sources, num_envs) bool; True -> place sources[i] into world=j
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    use_fabric: bool = False,
    device: str = "cpu",
    exclude_self_replication: bool = True,
) -> None:
    """Replicate prims via PhysX replicator with per-row mapping.

    Builds per-source destination lists from ``mapping`` and calls PhysX ``replicate``.

    On GPU (``device != "cpu"``), scene partitions
    (``primvars:omni:scenePartition``) are authored on each env prim and
    ``useEnvIds`` is enabled for GPU-accelerated per-environment collision
    isolation.  This works for both homogeneous and heterogeneous setups — the
    PhysX replicator discovers env boundaries from the primvar regardless of
    which sources map to which environments.

    The replicator is registered for the call and then unregistered.

    ``attach_fn`` excludes ``/World/template`` and ``/World/envs`` so that PhysX does
    not independently parse prims that the replicator will handle.  The source prim
    receives its physics body as a side-effect of ``rep.replicate()`` (which always
    parses the source internally), so every source must appear in at least one
    ``replicate`` call.

    When ``exclude_self_replication`` is True (default), each source environment is
    removed from its own replication targets so the replicator only creates bodies at
    non-self destinations.  If removing self would leave the world list empty (i.e. the
    source maps only to its own environment), self is kept so that ``rep.replicate()``
    is still called and the source prim gets its physics body.

    Args:
        stage: USD stage.
        sources: Source prim paths (``S``).
        destinations: Destination templates (``S``) with ``"{}"`` for env index.
        env_ids: Environment indices (``[E]``).
        mapping: Bool/int mask (``[S, E]``) selecting envs per source.
        positions: Optional positions (unused, for API compatibility).
        quaternions: Optional orientations (unused, for API compatibility).
        use_fabric: Use Fabric for replication.
        device: Torch device for determining replication mode.
        exclude_self_replication: If True, skip replicating a source prim onto itself
            when the source also maps to other environments.  Default is True.
            Self-only sources always keep self so that ``rep.replicate()`` fires.

    Returns:
        None
    """
    del positions, quaternions

    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    current_worlds: list[int] = []
    current_template: str = ""
    num_envs = mapping.size(1)

    if num_envs > 1:
        env_id_list = env_ids.tolist()
        env_prim_paths = [f"/World/envs/env_{i}" for i in env_id_list]

        # Lift replication to env-root level.  The replicator discovers
        # env IDs from ``primvars:omni:scenePartition`` on the replicated
        # prim itself — so the source passed to ``replicate()`` must be the
        # env root (e.g. ``/World/envs/env_0``), not a child prim.
        #
        # For each (source, destination) pair we extract the env-root
        # source and the env-root template, then merge world lists from
        # all sub-prim sources that share the same env-root source.
        env_root_worlds: dict[str, set[int]] = {}
        env_root_template = f"/World/envs/env_{{}}"

        for i, src in enumerate(sources):
            worlds = env_ids[mapping[i]].tolist()
            if exclude_self_replication:
                pre, _, suf = destinations[i].partition("{}")
                self_id = src.removeprefix(pre).removesuffix(suf)
                if self_id.isdigit():
                    filtered = [w for w in worlds if w != int(self_id)]
                    worlds = filtered if filtered else worlds

            # Derive env-root source path from the sub-prim source.
            # e.g. "/World/envs/env_0/Object" → "/World/envs/env_0"
            parts = src.split("/")
            # /World/envs/env_X is always depth 3 (indices 0="", 1="World", 2="envs", 3="env_X")
            root_src = "/".join(parts[:4]) if len(parts) > 4 else src
            env_root_worlds.setdefault(root_src, set()).update(worlds)

        # Sort worlds for deterministic ordering
        effective_env_roots: list[tuple[str, list[int]]] = [
            (root, sorted(wset)) for root, wset in env_root_worlds.items()
        ]

        # Set partitions BEFORE replicate so the replicator can discover
        # env IDs from the primvar during physics body creation.
        _set_scene_partitions(stage, env_prim_paths, use_fabric=use_fabric)

        def attach_fn(_stage_id: int):
            return ["/World/template", "/World/envs"]

        def rename_fn(_replicate_path: str, i: int):
            return current_template.format(current_worlds[i])

        def attach_end_fn(_stage_id: int):
            nonlocal current_template
            rep = get_physx_replicator_interface()

            for root_src, worlds in effective_env_roots:
                current_template = env_root_template
                current_worlds[:] = worlds
                if not current_worlds:
                    continue
                rep.replicate(
                    _stage_id,
                    root_src,
                    len(current_worlds),
                    useEnvIds=False,
                    useFabricForReplication=use_fabric,
                )

            # Re-set partitions AFTER replicate so that homogeneous cloning
            # (which copies env_0 including its primvar) gets corrected.
            _set_scene_partitions(stage, env_prim_paths, use_fabric=use_fabric)

            rep.unregister_replicator(_stage_id)

        get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)
