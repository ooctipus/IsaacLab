# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The :class:`ClonePlan` value type and the constructors that build one.

A plan is the whole description of a replication layout: which prototypes exist, where each
one is cloned to, and which envs each one populates. It is built once, queried through
:mod:`~isaaclab.cloner.query`, and executed by :class:`~isaaclab.cloner.ReplicateSession`.

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

from .cloner_cfg import DEFAULT_ENV_TEMPLATE, CloneCfg, InclusionSet, expand_env_regex_ns
from .path import match, relative_to


def _plan_cfgs(cfgs: Iterable[Any]) -> tuple[tuple[Any, ...], dict[int, str], CloneCfg | None]:
    """Collect declared prim authors, their scene bindings, and the sole clone configuration."""
    found: list[Any] = []
    names: dict[int, str] = {}
    clone_cfg: CloneCfg | None = None
    visited: set[int] = set()

    def visit(value: Any, binding: str | None, naming_boundary: bool = False) -> None:
        nonlocal clone_cfg
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)) or callable(value):
            return
        identity = id(value)
        if isinstance(value, CloneCfg):
            if clone_cfg is not None and clone_cfg is not value:
                raise ValueError("One clone lifecycle cannot contain multiple CloneCfg instances.")
            clone_cfg = value
            return
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
            field_names = tuple(field.name for field in dataclasses.fields(value))
            declared = set(field_names)
            spawn = value.spawn if "spawn" in declared else None
            if "prim_path" in declared and (
                spawn is not None or "collision_group" in declared or "markers" in declared
            ):
                found.append(value)
                if binding is not None:
                    names[identity] = binding
            child_items = [(name, getattr(value, name)) for name in field_names]
            child_items.extend((name, child) for name, child in vars(value).items() if name not in declared)
            owns_clone_cfg = any(isinstance(child, CloneCfg) for _, child in child_items)
            debug_vis = value.debug_vis if "debug_vis" in declared else True
            for name, child in child_items:
                if not debug_vis and name.endswith("visualizer_cfg"):
                    continue
                visit(child, name if naming_boundary or owns_clone_cfg else binding)
            return
        else:
            return
        for child in children:
            visit(child, binding)

    for cfg in cfgs:
        visit(cfg, None, naming_boundary=True)
    return tuple(found), names, clone_cfg


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

    positions: torch.Tensor | None = None
    """Per-env world positions [m], shape ``[num_clones, 3]``, or ``None``."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the nearest replication rows that cover it."""

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


def grid_transforms(num_instances: int, spacing: float = 1.0, up_axis: str = "z", device="cpu"):
    """Return centered grid positions and identity xyzw orientations."""
    positions = _grid_positions(num_instances, spacing, up_axis, device)
    orientations = torch.nn.functional.one_hot(torch.full((num_instances,), 3, device=device), num_classes=4).float()
    return positions, orientations


def _num_spawn_variants(spawn_cfg: Any) -> int:
    """Return the number of spawn variants declared by one spawner configuration.

    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` declares one variant per asset
    configuration and :class:`~isaaclab.sim.MultiUsdFileCfg` one per USD path;
    every other spawner declares a single variant.

    Args:
        spawn_cfg: Spawner configuration to inspect.

    Returns:
        The number of spawn variants the configuration expands into.
    """
    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        return len(spawn_cfg.assets_cfg)
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
    return 1


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


def _make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    device: str,
    *,
    env_template: str = DEFAULT_ENV_TEMPLATE,
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
        cfgs: Cloneable asset cfgs with resolved env-scoped ``prim_path`` and ``spawn``.
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        device: Torch device for plan tensors.

    Returns:
        A :class:`ClonePlan` whose ``sources``/``destinations``/``clone_mask`` describe
        the flat prototype-to-env mapping and whose ``cfg_rows`` maps each cfg to its nearest
        covering rows. Rows without clone slots name shared scene assets.
    """

    roots = tuple(cfgs)
    cfgs, cfg_names, clone_cfg = _plan_cfgs(roots)

    # 1) Build per-group records. A cfg outside the environment namespace is declared
    # once and shared, so it becomes a global row instead.
    records: list[tuple[Any, Any, str, int, bool]] = []
    owned_paths: set[str] = set()
    for cfg in cfgs:
        prim_path = expand_env_regex_ns(cfg.prim_path, env_template)
        matched = match(prim_path, env_template)
        destination = prim_path if matched is None else env_template + matched.suffix
        field_names = {field.name for field in dataclasses.fields(cfg)}
        spawn_cfg = cfg.spawn if "spawn" in field_names else None
        authors_without_spawn = "markers" in field_names
        if matched is None:
            valid_path = Sdf.Path.IsValidPathString(prim_path)
            if not valid_path or not Sdf.Path(prim_path).IsPrimPath():
                if spawn_cfg is not None:
                    raise ValueError(
                        f"Global authoring cfg {prim_path!r} must use an exact prim path or the session env template."
                    )
                continue
        if spawn_cfg is None and matched is not None and not authors_without_spawn:
            continue
        if destination in owned_paths:
            raise ValueError(f"Multiple cfgs author the same clone-plan destination: {destination!r}.")
        owned_paths.add(destination)
        count = 1 if spawn_cfg is None else _num_spawn_variants(spawn_cfg)
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

    def plan_with_globals(
        sources: list[str], destinations: list[str], mask: torch.Tensor, cfg_rows: dict[int, tuple[int, ...]]
    ) -> ClonePlan:
        """Append one row per global asset so the plan names every declared asset."""
        for cfg, prim_path in global_cfgs:
            cfg_rows[id(cfg)] = (len(sources),)
            sources.append(prim_path)
            destinations.append(prim_path)
        if global_cfgs:
            shared = torch.zeros((len(global_cfgs), num_clones), dtype=torch.bool, device=device)
            mask = torch.cat([mask, shared])
        for cfg, owner in covered_cfgs:
            cfg_rows[id(cfg)] = cfg_rows[id(owner)]
        cfg_destinations = {id(cfg): destination for cfg, _spawn, destination, _count, _global in records}
        collision_paths: list[str] = []
        for cfg in cfgs:
            field_names = {field.name for field in dataclasses.fields(cfg)}
            if "collision_group" not in field_names or cfg.collision_group != -1 or id(cfg) not in cfg_rows:
                continue
            destination = cfg_destinations[id(cfg)]
            if "{}" not in destination:
                collision_paths.append(destination)
                continue
            for row in cfg_rows[id(cfg)]:
                columns = mask[row].nonzero(as_tuple=False).flatten().to(env_ids.device)
                collision_paths.extend(destination.format(int(env_id)) for env_id in env_ids[columns].tolist())
        return ClonePlan(
            sources=tuple(sources),
            destinations=tuple(destinations),
            clone_mask=mask,
            env_ids=env_ids,
            positions=positions,
            cfg_rows=cfg_rows,
            collision_paths=tuple(dict.fromkeys(collision_paths)),
            env_template=env_template,
        )

    clone_combinations = None if clone_cfg is None else clone_cfg.clone_combinations

    # 2) No environment-scoped cfgs: the plan still describes the globals.
    if not groups:
        if clone_combinations:
            raise ValueError("Clone combinations require at least one independently cloned asset.")
        return plan_with_globals([], [], torch.zeros((0, num_clones), dtype=torch.bool, device=device), {})

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
    combos, counts = torch.unique(combos, dim=0, return_counts=True)
    combos = torch.repeat_interleave(combos, counts, dim=0)
    chosen = combos[torch.arange(num_clones, device=device) * len(combos) // num_clones]

    group_offsets = torch.tensor([0] + list(itertools.accumulate(group_sizes[:-1])), dtype=torch.long, device=device)
    active = chosen >= 0
    rows = (chosen + group_offsets).view(-1)
    cols = torch.arange(num_clones, device=device).view(-1, 1).expand(-1, len(group_sizes)).reshape(-1)
    active_flat = active.view(-1)

    num_rows = sum(group_sizes)
    clone_mask = torch.zeros((num_rows, num_clones), dtype=torch.bool, device=device)
    if active_flat.any():
        clone_mask[rows[active_flat], cols[active_flat]] = True

    sources_list: list[str] = []
    destinations_list: list[str] = []
    cfg_rows: dict[int, tuple[int, ...]] = {}
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
            # stable slot, reported as ``None`` by ``query.cfg_source_paths``.
            sources_list.append(destination.format(source_env_id if is_active else i))
        row += count

    return plan_with_globals(sources_list, destinations_list, clone_mask, cfg_rows)
