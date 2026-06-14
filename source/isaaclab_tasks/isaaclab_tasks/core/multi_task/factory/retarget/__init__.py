# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline factory Newton-IK reset-state pipeline.

The nut-first / sub-world-batched analog of the terrain foot-sampling pipeline:

    1. **Place** the board+bolt assembly group, then the nut per sub-world -- on
       the bolt at an assembly fraction, on the board, or free in the air
       (:class:`NutPlacementSampler`). The assets are sampler data, not model bodies.
    2. **Sample** antipodal contact pairs on the held-asset mesh -- mesh-general,
       no annotated grasp keypoint (:class:`GraspPairSampler`).
    3. **Solve** one batched analytic-Jacobian Newton IK over per-fingertip
       targets, fingers pinned to the pair aperture (mimic-consistent).
    4. **Filter** by fingertip reachability + tag-gated model-internal collision
       queries (:mod:`.criteria`), replacing ``CollisionAnalyzer`` +
       ``RigidObjectHasher``.

:meth:`FactoryIKPipeline.build_balanced_table` returns the accepted rows.

All exports lazy-load from the ``.pyi`` stub: the implementation modules pull
``newton`` (and transitively ``pxr``), which MUST NOT be imported before the Kit
app launches -- env-cfg modules reference :class:`FactoryIKPipelineCfg` and are
imported by the task registry pre-launch.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
