# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass


@configclass
class RslRlValueShiftCfg:
    """Configuration for the value-shift augmentation.

    Wires an externally-owned observation cache and two buffers onto the
    :class:`~rsl_rl.extensions.ValueShift` extension (a post-update critic
    value-drift signal that adds no loss or gradient). Each field is a Python
    expression ``eval``-ed once after construction against ``{env, alg, self}``
    -- ``self`` is the extension -- so the bind can reach buffers that only
    exist at runtime (e.g. on a curriculum sampler). Expressions are used
    instead of callables because the cfg must survive ``to_dict()`` /
    hydra / OmegaConf serialization.
    """

    observation_bind: str = ""
    """Expression setting ``self.obs_cache`` -- the cached critic-group observations, one row per state."""

    current_value_bind: str = ""
    """Expression setting ``self.cur_val`` -- the previous-update per-state value buffer, shape ``[num_states]``."""

    value_diff_bind: str = ""
    """Expression setting ``self.diff_val`` -- the per-state ``|V_new - V_prev|`` buffer the consumer reads,
    shape ``[num_states]``."""
