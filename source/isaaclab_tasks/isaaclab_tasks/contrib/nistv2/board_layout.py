# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset selection and physical layout for the NIST board."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.assembly_variants import (
    ASSEMBLY_VARIANTS as NIST_ASSEMBLY_VARIANTS,
)
from isaaclab_tasks.contrib.nist.assembly_variants import (
    AssemblyVariant,
)
from isaaclab_tasks.utils import PresetCfg, preset

_BOARD_VARIANTS = tuple(variant for variant in NIST_ASSEMBLY_VARIANTS if variant.name != "nut_thread_m4")

_ASSET_LABELS = {
    "nut_thread_m8": "N8",
    "nut_thread_m12": "N12",
    "nut_thread_m16": "N16",
    "gear_mesh_small": "Gs",
    "gear_mesh_medium": "Gm",
    "gear_mesh_large": "Gl",
    "rod_insert_4mm": "R4",
    "rod_insert_8mm": "R8",
    "rod_insert_12mm": "R12",
    "rod_insert_16mm": "R16",
    "peg_insert_4mm": "P4",
    "peg_insert_8mm": "P8",
    "peg_insert_12mm": "P12",
    "peg_insert_16mm": "P16",
    "usba": "USB",
    "waterproof": "WP",
    "bnc": "BNC",
    "dsub": "DS",
    "rj45": "RJ",
}


@configclass
class AssemblySetCfg:
    """Select the assembly catalog and physical scene capacity."""

    include: str = ".*"
    exclude: str | None = None
    num_slots: int | None = 1
    spawn_all_sockets: bool | PresetCfg = preset(default=False, all_sockets=True)


@dataclass(frozen=True, slots=True)
class BoardLayout:
    """Concrete assembly topology shared by scene construction and runtime terms."""

    variants: tuple[AssemblyVariant, ...]
    variant_names: tuple[str, ...]
    held_asset_names: tuple[str, ...]
    fixed_asset_names: tuple[str, ...]
    fixture_variant_indices: tuple[int, ...]
    fixture_index_by_variant: tuple[int, ...]
    asset_labels: tuple[str, ...]
    spawn_all_sockets: bool
    fixed_assets_are_variant_banks: bool

    @property
    def num_variants(self) -> int:
        """Number of assembly variants available to every held-asset slot."""
        return len(self.variants)

    @property
    def num_slots(self) -> int:
        """Number of physical held-asset slots in each environment."""
        return len(self.held_asset_names)

    @property
    def num_fixtures(self) -> int:
        """Number of distinct fixed fixtures in the selected catalog."""
        return len(self.fixture_variant_indices)

    @property
    def num_fixed_slots(self) -> int:
        """Number of physical fixed-asset bodies in each environment."""
        return len(self.fixed_asset_names)

    def held_variant_rows(self) -> list[list[int]]:
        """Return homogeneous prototype rows for the held-asset mesh banks."""
        return [
            [(row + slot) % self.num_variants for slot in range(self.num_slots)] for row in range(self.num_variants)
        ]

    def clone_rows(self) -> list[list[int]]:
        """Return fixed- and held-asset variants for homogeneous prototype worlds."""
        held_rows = self.held_variant_rows()
        if self.fixed_assets_are_variant_banks:
            fixed_rows = held_rows
        else:
            fixed_rows = [[0] * self.num_fixed_slots for _ in range(self.num_variants)]
        return [fixed + held for fixed, held in zip(fixed_rows, held_rows, strict=True)]


def board_layout(
    selection: AssemblySetCfg | Sequence[str] | None = None,
    num_slots: int | None = None,
    spawn_all_sockets: bool | None = None,
) -> BoardLayout:
    """Resolve an asset selection while preserving canonical catalog order."""
    if selection is None:
        selection = AssemblySetCfg()
    if isinstance(selection, AssemblySetCfg):
        variants = _select_variants(selection)
        if num_slots is None:
            num_slots = selection.num_slots
        if spawn_all_sockets is None:
            spawn_all_sockets = selection.spawn_all_sockets
            if isinstance(spawn_all_sockets, PresetCfg):
                spawn_all_sockets = spawn_all_sockets.default
    else:
        names = tuple(selection)
        if len(names) != len(set(names)):
            raise ValueError("Assembly names must be unique.")
        unknown = set(names).difference(variant.name for variant in _BOARD_VARIANTS)
        if unknown:
            raise ValueError(f"Unknown assembly name(s): {', '.join(sorted(unknown))}.")
        selected = set(names)
        variants = tuple(variant for variant in _BOARD_VARIANTS if variant.name in selected)
        if spawn_all_sockets is None:
            spawn_all_sockets = False
    if not variants:
        raise ValueError("Assembly selection must contain at least one asset pair.")
    if len(variants) == 1:
        raise ValueError(
            "Board variant banks require at least two assembly variants. Use IsaacContrib-Factory-Franka with "
            "the assembly preset for a single assembly."
        )
    if num_slots is None:
        num_slots = len(variants)
    if num_slots < 1:
        raise ValueError(f"Held-asset slot count must be positive, got {num_slots}.")
    if num_slots > len(variants):
        raise ValueError(f"Held-asset slot count {num_slots} exceeds the {len(variants)} selected variants.")

    fixture_keys: list[tuple[str, tuple[float, ...]]] = []
    fixture_variant_indices: list[int] = []
    fixture_index_by_variant: list[int] = []
    for variant_index, variant in enumerate(variants):
        key = (variant.fixed_asset.spawn.usd_path, variant.board_offset.pose)
        if key not in fixture_keys:
            fixture_keys.append(key)
            fixture_variant_indices.append(variant_index)
        fixture_index_by_variant.append(fixture_keys.index(key))

    fixture_counts = [fixture_index_by_variant.count(index) for index in range(len(fixture_keys))]
    every_fixture_is_selected = all(len(variants) - count < num_slots for count in fixture_counts)
    fixed_assets_are_variant_banks = not spawn_all_sockets and not every_fixture_is_selected
    num_fixed_slots = min(num_slots, len(fixture_keys)) if fixed_assets_are_variant_banks else len(fixture_keys)
    variant_names = tuple(variant.name for variant in variants)
    return BoardLayout(
        variants=variants,
        variant_names=variant_names,
        held_asset_names=tuple(f"held_{index:02d}" for index in range(num_slots)),
        fixed_asset_names=tuple(f"fixed_{index:02d}" for index in range(num_fixed_slots)),
        fixture_variant_indices=tuple(fixture_variant_indices),
        fixture_index_by_variant=tuple(fixture_index_by_variant),
        asset_labels=tuple(_ASSET_LABELS[name] for name in variant_names),
        spawn_all_sockets=bool(spawn_all_sockets),
        fixed_assets_are_variant_banks=fixed_assets_are_variant_banks,
    )


def _select_variants(selection: AssemblySetCfg) -> tuple[AssemblyVariant, ...]:
    try:
        include = re.compile(selection.include)
        exclude = re.compile(selection.exclude) if selection.exclude is not None else None
    except re.error as error:
        raise ValueError(f"Invalid assembly selection regex: {error}.") from error

    variants = tuple(
        variant
        for variant in _BOARD_VARIANTS
        if include.search(variant.name) is not None and (exclude is None or exclude.search(variant.name) is None)
    )
    return variants
