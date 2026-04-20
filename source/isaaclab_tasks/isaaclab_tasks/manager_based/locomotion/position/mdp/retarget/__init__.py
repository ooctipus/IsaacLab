# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry-constrained articulation retargeting pipeline.

Stages:
    1. **Sample** contact points on geometry (terrain mesh).
    2. **Retarget** -- caller builds IK solver + objectives, fills targets
       from buffer, calls ``solver.step``.
    3. **Validate** -- user-defined acceptance criteria.
    4. **Select** -- FPS for spatial uniformity.
"""

from .buffer import RetargetBuffer
from .cfg import IKCfg, RetargetPipelineCfg, SupportSamplingCfg
from .pipeline import ObjectivesFactory, RetargetPipeline
from .sample import sample_contacts
from .validate import CriterionFn, validate_results
