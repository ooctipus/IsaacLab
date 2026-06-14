# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flexible rotational symmetry.

Defines a held asset's set of indistinguishable goal orientations (continuous or
N-fold about an axis, plus manually-authored "semantic" equivalents) and reduces
a flat batch of asset instances to the nearest symmetry-equivalent of each target
-- the single source of truth shared by the success criterion, the command
observation, and the assembly sampler.

Layered per house convention: :mod:`.symmetry_cfg` is pure definition
dataclasses, :mod:`.asset_symmetry` compiles one asset symmetry, and
:mod:`.symmetry` owns the batch Warp reducer. Exports lazy-load from the
``.pyi`` stub so config classes are importable pre-launch without pulling
the Warp implementation.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
