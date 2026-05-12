# Phase 7 — env_cfgs.py / multi_task_env_cfg.py refactor plan

`env_cfgs.py` and `multi_task_env_cfg.py` are 1907 + 1010 lines today. Most
of the bulk is per-task plumbing (env cfg + commands + rewards + events)
copied into the same file. This plan splits them by archetype with no
behavioural change.

## Target layout

```
config/sawyer/
├── _helpers.py                    # shared factories (~140 lines moved out of env_cfgs.py)
├── env_cfgs/
│   ├── __init__.py                # re-exports every Env*Cfg class
│   ├── articulated/
│   │   ├── drawer.py              # MetaworldDrawerOpen/Close (mw_drawer)
│   │   ├── button.py              # Button-Press-Topdown / Coffee-Button (mw_button)
│   │   ├── window.py              # Window-Open / Window-Close (mw_window)
│   │   ├── faucet.py              # Faucet-Open/Close + Dial-Turn + Lever-Pull
│   │   ├── door.py                # Door-Open/Close/Lock/Unlock
│   │   ├── peg.py                 # Peg-Insert-Side
│   │   ├── handle.py              # Handle-Press/Pull/(side)
│   │   ├── plate.py               # Plate-Slide × 4
│   │   ├── box.py                 # Box-Close
│   │   ├── peg_unplug.py          # Peg-Unplug-Side
│   │   ├── hammer.py              # Hammer
│   │   └── button_front.py        # Button-Press / Button-Press-Wall
│   └── cube/
│       ├── push.py                # Push / Push-Back / Push-Wall
│       ├── reach.py               # Reach (already in reach_env_cfg.py — move here)
│       ├── pick_place.py          # Pick-Place / Pick-Place-Wall
│       ├── basketball.py          # Basketball
│       ├── shelf.py               # Shelf-Place
│       ├── soccer.py              # Soccer
│       ├── sweep.py               # Sweep / Sweep-Into
│       ├── coffee.py              # Coffee-Push / Coffee-Pull
│       ├── stick.py               # Stick-Push / Stick-Pull
│       ├── bin.py                 # Bin-Picking
│       ├── assembly.py            # Assembly / Disassemble
│       ├── hand_insert.py         # Hand-Insert
│       └── pick_out_of_hole.py    # Pick-Out-Of-Hole
├── multi_task/
│   ├── __init__.py                # re-exports MetaworldMT*SawyerEnvCfg
│   ├── _base.py                   # MetaworldMultiTaskSceneCfg + commands + obs + helpers
│   ├── _rewards.py                # _MultiTaskRewardsCfg (15-task) + parameterised helpers
│   ├── _events.py                 # _MultiTaskEventCfg + task_indexed_joint_reset
│   ├── mt15.py                    # MetaworldMultiTaskSawyerEnvCfg (existing, just relocated)
│   ├── mt5.py                     # MetaworldMT5SawyerEnvCfg + scene/rewards/events
│   ├── mt10.py                    # MetaworldMT10SawyerEnvCfg + scene/rewards/events
│   └── mt3.py                     # MetaworldMT3SawyerEnvCfg (the new one)
├── __init__.py                    # gym registrations only — no logic
└── agents/                        # unchanged
```

## Constraints / things to preserve

1. **Gym IDs unchanged.** The user-facing `Isaac-Metaworld-*-Sawyer-v0` IDs
   stay the same; only entry-point module paths change in `__init__.py`.
2. **Single source of truth for sampling ranges.** `metaworld_specs.py`
   stays as-is — `_paired_from_spec` lives in `_helpers.py` and reads from
   it.
3. **Backwards-compatible imports.** Anything currently doing
   `from .env_cfgs import MetaworldDrawerOpenSawyerEnvCfg` keeps working
   via the package-level re-export in `env_cfgs/__init__.py`.

## Plan to execute (estimated 2-3 hour session)

1. Create `_helpers.py` containing `_ObsCfg`, `_fixed_paired`,
   `_paired_from_spec`, `_reset_robot`, `_reset_joint_to`,
   `_reset_joint_from_spec`, plus the reward-shape factories
   (`_DRAWER_OPEN_*`, `_PEG_INSERT_*`, etc.). Verify a single env (e.g.
   drawer-open) still loads.
2. Move articulated env cfgs in alphabetical order, one PR per asset
   family, running `verify_real_envs.py` and the parity sweep after each
   move.
3. Move cube env cfgs (lower priority — mostly copy-paste from
   push_env_cfg.py / pick_place_env_cfg.py / reach_env_cfg.py).
4. Split `multi_task_env_cfg.py` mirroring the same archetype split.
5. Squash the old `env_cfgs.py` and `multi_task_env_cfg.py` once everything
   re-exports cleanly. Run the full parity sweep + smoke-test each
   `Isaac-Metaworld-*-MT*-v0` id (with `--num_envs ≥ N_tasks`).

## Out of scope for this plan

- Behavioural changes (reward formulae, sampling ranges, etc.) — they should
  travel in their own commits.
- The MT25 / MT50 multi-task scenes — those need new TASK_NAMES / scene
  assets, not just a refactor. Address after this split lands so the new
  files have a clean home.
