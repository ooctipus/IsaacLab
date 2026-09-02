# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The :class:`ClonePlan` value type and the constructors that build one.

A plan is the whole description of a replication layout: which prototypes exist, where each
one is cloned to, and which envs each one populates. It is built once, queried through
:mod:`~isaaclab.cloner.query`, and executed by :func:`~isaaclab.cloner.replicate`.

Two internal constructors cover how a layout is specified:

* :func:`_make_clone_plan` — the layout is derived from the scene's asset cfgs, expanding
  multi-asset spawners into per-variant prototypes.
* :func:`_make_valid_clone_combinations` — restricts which variant combinations
  :func:`_make_clone_plan` may draw from, weighted per combination.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.utils.string import string_to_callable

from .cloner_cfg import DEFAULT_ENV_TEMPLATE, CloneCfg, InclusionSet, expand_env_regex_ns
from .path import match, relative_to


def _plan_cfgs(
    cfgs: Iterable[Any],
) -> tuple[
    tuple[Any, ...],
    dict[int, str],
    CloneCfg,
    dict[int, tuple[type[object], ...] | None],
    tuple[type[object], ...],
]:
    """Collect declared prim authors, their scene bindings, clone policy, and context routing."""
    cfgs = tuple(cfgs)
    clone_cfgs = tuple(cfg for cfg in cfgs if isinstance(cfg, CloneCfg))
    if len(clone_cfgs) != 1:
        raise ValueError("One clone lifecycle requires one CloneCfg as a top-level input.")

    found: list[Any] = []
    names: dict[int, str] = {}
    scene_consumers: list[Any] = []
    clone_cfg = clone_cfgs[0]
    visited: set[int] = set()

    def visit(value: Any, binding: str | None, naming_boundary: bool = False) -> None:
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)) or callable(value):
            return
        identity = id(value)
        if isinstance(value, CloneCfg):
            raise TypeError("CloneCfg must be supplied directly, not discovered inside another cfg.")
        if identity in visited:
            if binding is not None and identity in names and names[identity] != binding:
                raise ValueError(f"One cfg cannot be bound as both {names[identity]!r} and {binding!r}.")
            return
        visited.add(identity)
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, str(name) if naming_boundary else binding)
            return
        elif isinstance(value, (list, tuple)):
            children = value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            child_items = list(vars(value).items())
            instance_fields = dict(child_items)
            if "prim_path" in instance_fields:
                found.append(value)
                if binding is not None:
                    names[identity] = binding
            elif "cloning_contexts" in instance_fields:
                scene_consumers.append(value)
            for name, child in child_items:
                visit(child, name if naming_boundary else binding)
            return
        else:
            return
        for child in children:
            visit(child, binding)

    def visit_participant(value: Any, binding: str | None = None) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for name, child in value.items():
                visit_participant(child, str(name) if binding is None else binding)
            return
        if not dataclasses.is_dataclass(value) or isinstance(value, type):
            raise TypeError(f"Clone participants must be prim-authoring cfgs, got {type(value).__name__}.")
        fields = vars(value)
        if "prim_path" not in fields and "cloning_contexts" not in fields:
            raise TypeError(f"Clone participants must be prim-authoring cfgs, got {type(value).__name__}.")
        visit(value, binding, naming_boundary=binding is None)

    for cfg in cfgs:
        if cfg is not clone_cfg:
            visit_participant(cfg)

    cfg_contexts: dict[int, tuple[type[object], ...] | None] = {}
    for cfg in found:
        instance_fields = vars(cfg)
        if "cloning_contexts" not in instance_fields:
            continue
        references = instance_fields["cloning_contexts"]
        if references is None:
            cfg_contexts[id(cfg)] = None
            continue
        contexts = tuple(string_to_callable(value) if isinstance(value, str) else value for value in references)
        if any(not isinstance(context, type) for context in contexts):
            raise TypeError(f"{type(cfg).__name__}.cloning_contexts must contain only context classes.")
        cfg_contexts[id(cfg)] = contexts
    scene_contexts = tuple(
        dict.fromkeys(
            string_to_callable(reference) if isinstance(reference, str) else reference
            for cfg in scene_consumers
            for reference in vars(cfg)["cloning_contexts"]
        )
    )
    if any(not isinstance(context, type) for context in scene_contexts):
        raise TypeError("Whole-scene cloning_contexts must contain only context classes.")
    return tuple(found), names, clone_cfg, cfg_contexts, scene_contexts


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Description of one flat per-asset replication layout."""

    sources: tuple[str, ...]
    """Source prim paths, one per replication row."""

    destinations: tuple[str, ...]
    """Destination path templates with ``"{}"`` for the env id, one per row."""

    clone_mask: torch.Tensor
    """Bool tensor ``[len(sources), num_clones]``; ``True`` if env ``j`` comes from row ``i``."""

    env_ids: torch.Tensor
    """Long tensor ``[num_clones]`` of target env ids."""

    positions: torch.Tensor
    """Per-env world positions [m], shape ``[num_clones, 3]``."""

    replicate_physics: bool
    """Whether clone contexts may replicate one physics prototype across environments."""

    filter_collisions: bool = True
    """Whether PhysX collision filtering is applied after stage replication."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the nearest replication rows that cover it."""

    context_rows: dict[type[object], tuple[int, ...]] = field(default_factory=dict)
    """Registered or explicitly requested clone-context types to their routed rows."""

    collision_paths: tuple[str, ...] = ()
    """Plan-owned prim roots whose cfg declares ``collision_group=-1``."""

    env_template: str = DEFAULT_ENV_TEMPLATE
    """Environment path template used to resolve cfg namespace macros."""


def _grid_positions(num_instances: int, spacing: float = 1.0, up_axis: str = "z", device="cpu"):
    """Create a centered grid of positions for ``num_instances`` instances.

    Computes ``(x, y)`` coordinates in a roughly square grid centered at the origin
    with the provided spacing and places the third coordinate according to ``up_axis``.
    This matches the grid layout used by
    :class:`isaaclab.terrains.TerrainImporter` for consistent environment positioning.

    Args:
        num_instances: Number of instances.
        spacing: Distance between neighboring grid positions [m].
        up_axis: Up axis for positions ("z", "y", or "x").
        device: Torch device for returned tensors.

    Returns:
        Positions [m], shape ``(num_instances, 3)``.
    """
    # Match terrain_importer._compute_env_origins_grid layout for consistency
    num_rows = int(math.ceil(num_instances / math.sqrt(num_instances)))
    num_cols = int(math.ceil(num_instances / num_rows))

    # Create meshgrid matching terrain's "ij" indexing
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, device=device, dtype=torch.float32),
        torch.arange(num_cols, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Flatten and take first num_instances elements
    ii = ii.flatten()[:num_instances]
    jj = jj.flatten()[:num_instances]

    # Match terrain's coordinate system: X from rows (negated), Y from cols
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    z0 = torch.zeros(num_instances, device=device)

    # place on plane based on up_axis
    if up_axis.lower() == "z":
        pos = torch.stack([x, y, z0], dim=1)
    elif up_axis.lower() == "y":
        pos = torch.stack([x, z0, y], dim=1)
    else:  # up_axis == "x"
        pos = torch.stack([z0, x, y], dim=1)

    return pos


def _make_valid_clone_combinations(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    clone_combinations: Sequence[InclusionSet] | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Build the valid clone-combination variant tensor.

    Each combination contributes rows in proportion to its weight, split evenly
    across its spawn variants and interleaved round-robin, so any prefix of the
    tensor samples every combination.

    Args:
        asset_names: Clone-planned scene asset names, one per tensor column.
        variant_counts: Number of spawn variants per clone-planned asset.
        clone_combinations: Legal clone combinations; assets not mentioned by
            any combination are active in every row. ``None`` uses the full
            cartesian product of variants.
        device: Torch device for the output tensor. Defaults to ``"cpu"``.
    Returns:
        A ``[num_valid_combinations, num_assets]`` tensor of source variant
        indices, ``-1`` where an asset is absent.

    Raises:
        ValueError: If the inputs are inconsistent or no valid rows result.
    """
    if len(asset_names) != len(variant_counts):
        raise ValueError(f"Expected one variant count per asset, got {len(variant_counts)} and {len(asset_names)}.")
    if not asset_names:
        raise ValueError("Expected at least one asset name.")
    if any(count <= 0 for count in variant_counts):
        raise ValueError("Variant counts must be positive.")

    if not clone_combinations:
        rows = itertools.product(*[range(count) for count in variant_counts])
        return torch.tensor(list(rows), dtype=torch.long, device=device)

    clone_asset_names = set(asset_names)
    combination_assets: list[set[str]] = []
    for combination in clone_combinations:
        if combination.weight < 0:
            raise ValueError("Clone combination weights must be non-negative.")
        unknown_assets = sorted(set(combination.assets) - clone_asset_names)
        if unknown_assets:
            raise ValueError(f"Unknown assets in clone combination: {unknown_assets}.")
        combination_assets.append(set(combination.assets) & clone_asset_names)

    claimed_assets = set().union(*combination_assets) if combination_assets else set()

    expanded: list[tuple[int, list[tuple[int, ...]]]] = []
    for combination, active_assets in zip(clone_combinations, combination_assets):
        if combination.weight == 0:
            continue
        variant_ranges = []
        for asset_name, count in zip(asset_names, variant_counts):
            is_active = asset_name not in claimed_assets or asset_name in active_assets
            variant_ranges.append(range(count) if is_active else (-1,))
        expanded.append((combination.weight, list(itertools.product(*variant_ranges))))

    if not expanded:
        raise ValueError("Clone combinations produced no valid clone rows.")

    # A combination's share is its weight, split evenly across its spawn variants.
    # Integer multiplicities require a common denominator across variant counts.
    # Rows are emitted round-robin across combinations so a truncated prefix
    # (fewer environments than rows) still samples every combination.
    common_multiple = math.lcm(*[len(variants) for _, variants in expanded])
    rows = []
    cursors = [0] * len(expanded)
    for _ in range(common_multiple):
        for index, (weight, variants) in enumerate(expanded):
            for _ in range(weight):
                rows.append(variants[cursors[index] % len(variants)])
                cursors[index] += 1
    return torch.tensor(rows, dtype=torch.long, device=device)


def _make_clone_plan(  # noqa: C901
    cfgs: tuple[Any, ...],
    cfg_names: dict[int, str],
    clone_cfg: CloneCfg,
    cfg_contexts: dict[int, tuple[type[object], ...] | None],
    scene_contexts: tuple[type[object], ...],
    num_clones: int,
    env_spacing: float,
    device: str,
    context_roles: dict[type[object], set[str]],
) -> ClonePlan:
    """Build a :class:`ClonePlan` from asset cfgs.

    Iterates ``cfgs``, identifies env-scoped cfgs with a spawn, expands
    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` / :class:`~isaaclab.sim.MultiUsdFileCfg`
    into per-variant prototype rows, groups equivalent environments for batched cloning,
    and returns a self-contained :class:`ClonePlan` with ``cfg_rows`` populated.

    A cfg outside the environment namespace gets a row that nothing copies, so the plan
    names it even when the cfg authors that prim without a spawner. An environment-scoped
    cfg without a spawner only references an existing planned asset and gets no row. The
    input cfgs remain unchanged.

    Args:
        cfgs: Collected prim-authoring cfgs.
        cfg_names: Scene binding names for collected cfgs.
        clone_cfg: Sole declarative clone policy.
        cfg_contexts: Resolved per-cfg context requests.
        scene_contexts: Contexts requested by non-prim scene consumers.
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        device: Torch device for plan tensors.
        context_roles: Clone roles already registered by context type.

    Returns:
        A :class:`ClonePlan` whose ``sources``/``destinations``/``clone_mask`` describe
        the flat prototype-to-env mapping and whose ``cfg_rows`` maps each cfg to its nearest
        covering rows. Rows without clone slots name shared scene assets.
    """

    env_template = clone_cfg.clone_template

    # 1) Build per-group records. A cfg outside the environment namespace is declared
    # once and shared, so it becomes a global row instead.
    records: list[tuple[Any, Any, str, int, bool]] = []
    references: list[tuple[Any, str]] = []
    owned_paths: set[str] = set()
    for cfg in cfgs:
        prim_path = expand_env_regex_ns(cfg.prim_path, env_template)
        matched = match(prim_path, env_template)
        destination = prim_path if matched is None else env_template + matched.suffix
        instance_fields = vars(cfg)
        spawn_cfg = instance_fields.get("spawn")
        if spawn_cfg is None and "collision_group" not in instance_fields:
            references.append((cfg, prim_path))
            continue
        if matched is None:
            valid_path = Sdf.Path.IsValidPathString(prim_path)
            if not valid_path or not Sdf.Path(prim_path).IsPrimPath():
                raise ValueError(
                    f"Global authoring cfg {prim_path!r} must use an exact prim path or the session env template."
                )
        if destination in owned_paths:
            raise ValueError(f"Multiple cfgs author the same clone-plan destination: {destination!r}.")
        owned_paths.add(destination)
        if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
            count = len(spawn_cfg.assets_cfg)
        elif isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg) and not isinstance(spawn_cfg.usd_path, str):
            count = len(spawn_cfg.usd_path)
        else:
            count = 1
        if count <= 0:
            raise ValueError(f"Spawner at '{prim_path}' must have at least one variant.")
        records.append((cfg, spawn_cfg, destination, count, matched is None))

    # A parent row already copies every declared child subtree. Map child cfgs onto the nearest
    # parent's rows so constructors still resolve every exact prototype without asking any backend
    # to copy the same prims again.
    parent_by_id: dict[int, tuple[Any, Any, str, int, bool]] = {}
    for record in records:
        cfg, _spawn, destination, count, _is_global = record
        parents = [
            candidate
            for candidate in records
            if candidate is not record and relative_to(destination, candidate[2]) not in (None, "")
        ]
        if not parents:
            continue
        if count != 1:
            raise ValueError(f"Nested clone-plan cfg {destination!r} cannot declare multiple spawn variants.")
        parent_by_id[id(cfg)] = max(parents, key=lambda candidate: len(candidate[2]))

    groups: list[tuple[Any, Any, str, int]] = []
    global_cfgs: list[tuple[Any, str]] = []
    covered_cfgs: list[tuple[Any, Any]] = []
    for cfg, spawn_cfg, destination, count, is_global in records:
        parent = parent_by_id.get(id(cfg))
        if parent is not None:
            while id(parent[0]) in parent_by_id:
                parent = parent_by_id[id(parent[0])]
            covered_cfgs.append((cfg, parent[0]))
        elif is_global:
            global_cfgs.append((cfg, destination))
        else:
            groups.append((cfg, spawn_cfg, destination, count))
    env_ids = torch.arange(num_clones, dtype=torch.long, device=device)
    positions = _grid_positions(num_clones, env_spacing, device=device)

    clone_combinations = clone_cfg.clone_combinations
    sources_list: list[str] = []
    destinations_list: list[str] = []
    cfg_rows: dict[int, tuple[int, ...]] = {}

    # 2) No environment-scoped cfgs: the plan still describes the globals.
    if not groups:
        if clone_combinations:
            raise ValueError("Clone combinations require at least one independently cloned asset.")
        clone_mask = torch.zeros((0, num_clones), dtype=torch.bool, device=device)
    else:
        # 3) Enumerate prototype combinations and build the per-row mask. A homogeneous
        # scene is not special-cased: every asset always retains its own row.
        group_sizes = [count for _, _, _, count in groups]
        if clone_combinations:
            try:
                group_names = [cfg_names[id(cfg)] for cfg, _spawn, _destination, _count in groups]
            except KeyError as exc:
                raise ValueError("Every asset in clone_combinations must have one scene binding name.") from exc
            combos = _make_valid_clone_combinations(group_names, group_sizes, clone_combinations, device)
        else:
            combos = torch.tensor(
                list(itertools.product(*[range(size) for size in group_sizes])), dtype=torch.long, device=device
            )
        chosen = combos[torch.arange(num_clones, device=device) % len(combos)]

        group_offsets = torch.tensor(
            [0] + list(itertools.accumulate(group_sizes[:-1])), dtype=torch.long, device=device
        )
        active = chosen >= 0
        rows = (chosen + group_offsets).view(-1)
        cols = torch.arange(num_clones, device=device).view(-1, 1).expand(-1, len(group_sizes)).reshape(-1)
        active_flat = active.view(-1)

        clone_mask = torch.zeros((sum(group_sizes), num_clones), dtype=torch.bool, device=device)
        if active_flat.any():
            clone_mask[rows[active_flat], cols[active_flat]] = True

        row = 0
        for cfg, _spawn_cfg, destination, count in groups:
            cfg_rows[id(cfg)] = tuple(range(row, row + count))
            group_mask = clone_mask[row : row + count]
            source_env_ids = group_mask.to(torch.int).argmax(dim=1).tolist()
            active = group_mask.any(dim=1).tolist()
            for i, (source_env_id, is_active) in enumerate(zip(source_env_ids, active)):
                destinations_list.append(destination)
                # An active prototype stays at its first destination so another row cannot
                # overwrite it before every backend consumes the plan. Inactive rows retain a
                # stable slot, reported as ``None`` by ``query._cfg_source_paths``.
                sources_list.append(destination.format(source_env_id if is_active else i))
            row += count

    # 4) Shared assets and cfgs covered by a parent row join the same final plan.
    for cfg, prim_path in global_cfgs:
        cfg_rows[id(cfg)] = (len(sources_list),)
        sources_list.append(prim_path)
        destinations_list.append(prim_path)
    if global_cfgs:
        shared = torch.zeros((len(global_cfgs), num_clones), dtype=torch.bool, device=device)
        clone_mask = torch.cat([clone_mask, shared])
    for cfg, owner in covered_cfgs:
        cfg_rows[id(cfg)] = cfg_rows[id(owner)]
    for cfg, prim_path in references:
        owners = [
            record
            for record in records
            if (
                match(prim_path, record[2]) is not None
                if "{}" in record[2]
                else relative_to(prim_path, record[2]) is not None
            )
        ]
        if owners:
            owner = max(owners, key=lambda record: len(record[2]))
            cfg_rows[id(cfg)] = cfg_rows[id(owner[0])]
            continue
        if match(prim_path, env_template) is not None:
            continue
        if not Sdf.Path.IsValidPathString(prim_path) or not Sdf.Path(prim_path).IsPrimPath():
            raise ValueError(f"Global cfg {prim_path!r} must use an exact prim path.")
        try:
            row = destinations_list.index(prim_path)
        except ValueError:
            row = len(destinations_list)
            sources_list.append(prim_path)
            destinations_list.append(prim_path)
            clone_mask = torch.cat([clone_mask, torch.zeros((1, num_clones), dtype=torch.bool, device=device)])
        cfg_rows[id(cfg)] = (row,)

    cfg_destinations = {id(cfg): destination for cfg, _spawn, destination, _count, _global in records}
    collision_paths: list[str] = []
    for cfg in cfgs:
        instance_fields = vars(cfg)
        if instance_fields.get("collision_group") != -1 or id(cfg) not in cfg_rows:
            continue
        destination = cfg_destinations[id(cfg)]
        if "{}" not in destination:
            collision_paths.append(destination)
            continue
        for row in cfg_rows[id(cfg)]:
            columns = clone_mask[row].nonzero(as_tuple=False).flatten().to(env_ids.device)
            collision_paths.extend(destination.format(int(env_id)) for env_id in env_ids[columns].tolist())

    populated_rows = {
        row for row, destination in enumerate(destinations_list) if "{}" in destination and bool(clone_mask[row].any())
    }
    physics_contexts = tuple(context_type for context_type, roles in context_roles.items() if "physics" in roles)
    routed_rows: dict[type[object], set[int]] = {}
    for cfg in cfgs:
        rows = cfg_rows.get(id(cfg))
        if rows is None or id(cfg) not in cfg_contexts:
            continue
        contexts = physics_contexts if cfg_contexts[id(cfg)] is None else cfg_contexts[id(cfg)]
        for context_type in contexts:
            routed_rows.setdefault(context_type, set()).update(rows)
    for context_type, roles in context_roles.items():
        if "scene" in roles:
            routed_rows.setdefault(context_type, set()).update(populated_rows)
        elif "physics" in roles:
            routed_rows.setdefault(context_type, set())
    for context_type in scene_contexts:
        routed_rows.setdefault(context_type, set()).update(populated_rows)
    context_rows = {context_type: tuple(sorted(rows & populated_rows)) for context_type, rows in routed_rows.items()}

    return ClonePlan(
        sources=tuple(sources_list),
        destinations=tuple(destinations_list),
        clone_mask=clone_mask,
        env_ids=env_ids,
        positions=positions,
        replicate_physics=clone_cfg.replicate_physics,
        filter_collisions=clone_cfg.filter_collisions,
        cfg_rows=cfg_rows,
        context_rows=context_rows,
        collision_paths=tuple(dict.fromkeys(collision_paths)),
        env_template=env_template,
    )
