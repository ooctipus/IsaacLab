# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered procedural keyboard configurations for SO101 episode resets."""

from .keyboard_gen_cfg import KeyboardSpawnerCfg

_BASE_SPAWNER = KeyboardSpawnerCfg(
    family="ansi_full",
    partition_mode="fixed_dof",
    partition_dof=6,
    topology_mode="global_padded",
    max_slots=108,
    uniform_key_shapes=True,
)

# Fixed capacity; geometry and physical properties vary even between the two 108-key entries.
TYPING_KEYBOARD_VARIANTS = tuple(
    _BASE_SPAWNER.replace(seed=seed, key_count=count) for seed, count in enumerate((108, *range(6, 108, 6), 108))
)
