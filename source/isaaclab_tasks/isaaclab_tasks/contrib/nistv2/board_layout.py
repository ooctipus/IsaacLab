# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical entities in the full NIST board."""

from pathlib import Path

from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANTS

NUM_ASSEMBLIES = len(ASSEMBLY_VARIANTS)
HELD_ASSET_NAMES = tuple(f"held_{index:02d}" for index in range(NUM_ASSEMBLIES))

FIXED_ASSET_NAME_BY_VARIANT = tuple(
    f"fixed_{Path(variant.fixed_asset.spawn.usd_path).stem}" for variant in ASSEMBLY_VARIANTS
)
UNIQUE_FIXED_ASSET_NAMES = tuple(dict.fromkeys(FIXED_ASSET_NAME_BY_VARIANT))
UNIQUE_FIXED_VARIANT_INDICES = tuple(FIXED_ASSET_NAME_BY_VARIANT.index(name) for name in UNIQUE_FIXED_ASSET_NAMES)


def clone_variant_rows() -> list[list[int]]:
    """Return compact clone rows for the full board scene."""
    singleton_groups = 3 + len(UNIQUE_FIXED_ASSET_NAMES)  # table, board, robot, fixed fixtures
    rows = []
    for row in range(NUM_ASSEMBLIES):
        held_variants = [(row + slot) % NUM_ASSEMBLIES for slot in range(NUM_ASSEMBLIES)]
        rows.append([0] * singleton_groups + held_variants)
    return rows
