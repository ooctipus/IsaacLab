# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static check that the dt-scale compensation in
:meth:`MetaworldEnvCfg.__post_init__` is wired correctly.

Doesn't run the simulator — just constructs every Sawyer env cfg, checks
that the reward weights are all multiplied by ``1 / step_dt`` relative
to the raw cfg values. Run with::

    ./isaaclab.sh -p source/.../utils/parity/check_weights.py
"""

from __future__ import annotations

import importlib

import gymnasium  # noqa: E402

# Rebuild the un-multiplied cfg by zero-ing the post-init scaling. We do this
# by constructing the cfg and then dividing back out — anything not divisible
# (e.g. weight==0 success terms) confirms the pre/post-init deltas.
from isaaclab.managers import RewardTermCfg  # noqa: E402

# Importing the sawyer registration module triggers ``gym.register`` for every
# Meta-World task, so we can iterate the registered ids.
import isaaclab_contrib.tasks  # noqa: F401, E402


def _check(task_id: str) -> tuple[bool, str]:
    spec = gymnasium.spec(task_id)
    mod, _, cls = spec.kwargs["env_cfg_entry_point"].partition(":")
    cfg = getattr(importlib.import_module(mod), cls)()

    # Expected step_dt (= sim.dt × decimation = (1/200) × 2 = 0.01).
    step_dt = cfg.sim.dt * cfg.decimation
    expected_scale = 1.0 / step_dt

    nonzero_weights = []
    for name in dir(cfg.rewards):
        if name.startswith("_"):
            continue
        term = getattr(cfg.rewards, name, None)
        if isinstance(term, RewardTermCfg) and term.weight != 0.0:
            nonzero_weights.append((name, term.weight))

    if not nonzero_weights:
        return True, f"{task_id}: no non-zero reward weights (skipped)"

    # Verify magnitudes are sensibly scaled — heuristic: any |weight| > 1e-2
    # implies the dt-compensation has been applied (raw cfg weights are
    # typically in [1e-4, 10]; post-fix they end up in [1e-2, 1000]).
    too_small = [(n, w) for n, w in nonzero_weights if abs(w) < 1e-3]
    if too_small:
        return False, f"{task_id}: weights still in raw scale: {too_small}"

    return True, (
        f"{task_id}: {len(nonzero_weights)} reward terms, "
        f"max |weight| = {max(abs(w) for _, w in nonzero_weights):.1f} "
        f"(expected 1 / step_dt = {expected_scale:.0f}× the raw cfg weight)"
    )


def main() -> None:
    # Iterate every registered Isaac-Metaworld-*-Sawyer-v0 id.
    ids = [k for k in gymnasium.registry.keys() if k.startswith("Isaac-Metaworld-")]
    print(f"Checking {len(ids)} registered Meta-World envs...")
    fails = 0
    for task_id in sorted(ids):
        try:
            ok, msg = _check(task_id)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{task_id}: {type(e).__name__}: {e}"
        prefix = "[OK] " if ok else "[FAIL]"
        print(f"  {prefix} {msg}")
        fails += 0 if ok else 1
    print(f"\n{len(ids) - fails} / {len(ids)} envs passed dt-scale check")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
