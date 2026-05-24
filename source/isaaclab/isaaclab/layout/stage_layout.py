# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StageLayout:
    """Pure-data occupancy map of distinct sources across the shared scope and per-env worlds.

    Replaces :class:`~isaaclab.cloner.ClonePlan`. Six fields, zero methods. Layout queries
    live in :mod:`isaaclab.layout.layout_util`.

    The occupancy is encoded as a CSR (compressed sparse row) over ``num_envs + 1`` worlds:
    world index ``0`` is shared scope, world index ``1 + i`` is env ``i``. Each world holds
    zero or more *slots*; each slot points at one source in :attr:`sources` (via
    :attr:`source_ids`) and one destination template in :attr:`destinations` (via
    :attr:`destination_ids`).

    For the common case of one destination per source, :attr:`destination_ids` mirrors
    :attr:`source_ids`. The decoupling exists so a single source can be authored under
    multiple destination namespaces (shared scope + per-env copies) and so that identical
    destination templates can be deduplicated.
    """

    sources: tuple[Any, ...]
    """Unique source registry (typically asset / sensor cfgs).

    ``source_ids[k]`` indexes into this. Identity-keyed (``source is source``), not
    value-keyed.
    """

    destinations: tuple[str, ...]
    """Unique destination-template registry.

    Each template uses ``"{}"`` as the env-id placeholder, e.g.
    ``"/World/envs/env_{}/Robot"``.
    """

    source_ids: torch.Tensor
    """Per-slot source index, shape ``[N_slots]``, dtype ``torch.long``.

    ``source_ids[k]`` is the index of the source occupying slot ``k``.
    """

    destination_ids: torch.Tensor
    """Per-slot destination-template index, shape ``[N_slots]``, dtype ``torch.long``.

    Parallel to :attr:`source_ids`; ``destination_ids[k]`` is the index of the
    destination template for slot ``k``.
    """

    world_start: torch.Tensor
    """CSR row offsets, shape ``[num_envs + 2]``, dtype ``torch.long``.

    World ``w`` owns slots in ``source_ids[world_start[w]:world_start[w + 1]]``.
    ``world_start[0] == 0`` and ``world_start[-1] == N_slots``.
    """

    env_pose: torch.Tensor
    """Per-env pose, shape ``[num_envs, 7]``, dtype ``torch.float32``.

    ``env_pose[i, :3]`` is the position [m] of env ``i`` and ``env_pose[i, 3:]`` is
    its quaternion in xyzw.
    """
