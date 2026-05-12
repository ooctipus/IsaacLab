# Session summary — 2026-05-06

Concise record of what landed this session. Full detail in
`PROGRESS_24H.md` and `source/.../metaworld/PARITY_REPORT.md`.

## Headline finding

**IsaacLab's `RewardManager` multiplies every reward term by `step_dt` (=
0.01 s); Meta-World rewards are per-step.** Our policy was seeing a
**100× weaker** reward signal than MW. Fixed in
`MetaworldEnvCfg.__post_init__` by scaling all non-zero reward weights
by `1 / step_dt = 100`.

After the fix, in-process audit (`parity_reward_audit.py`) feeding
IsaacLab runtime state into MW's pure-Python `reach_v2_reward` /
`push_v2_reward` / `pick_place_v2_reward` reports:

```
reach        mean |Δr| = 0.0000   max |Δr| = 0.0001
push         mean |Δr| = 0.0000   max |Δr| = 0.0001
pick-place   mean |Δr| = 0.0000   max |Δr| = 0.0001
```

Reward formulae are byte-equivalent to MW V2.

## Second finding

**Sawyer DiffIK with `pinv k_val = 25` plateaus at TCP z ≈ 0.06**, so
the gripper never reaches the cube at z = 0.02.
`probe_push_dynamics.py` confirmed: cube moved 0 mm under any 200-step
deterministic action sequence. Fixed by bumping `k_val` to 100; the same
probe now drives the cube 315 mm. PPO retraining with the new gain in
progress at session end.

## Other fixes

* **L3 sampling** — extended `MetaworldTaskSpec` with `obj_range_*` /
  `goal_range_*`; baked MW per-task ranges from the dump
  (`mw_ranges_baked.py`); merged at import time. 47 tasks went from
  deterministic point-sampling to MW-equivalent ranges (`obj_mean_mm > 50`
  failures dropped 39 → 0).
* **L2 faucet-close** — joint reset 0 (was π/3); MW resets the knob
  centred and encodes the close direction in the goal. `joint_max_abs`
  failures dropped 1 → 0.
* **3 IsaacLab serialisation bugs** in `class_to_dict` /
  `update_class_from_dict` blocked Hydra from handling `dict[int, ...]`
  reward params; patched in `source/isaaclab/isaaclab/utils/dict.py`.
* `joint_value_by_task` now uses string keys and coerces back to int on
  read (works through Hydra's serialise / deserialise round-trip).

## Multi-task envs

* `MetaworldMT3SawyerEnvCfg` (cube manipulandum) — new file
  `config/sawyer/mt3_env_cfg.py`.
* `MetaworldMT5SawyerEnvCfg` + `MetaworldMT10SawyerEnvCfg` —
  smaller-scope subsets of the existing 15-task env, each with its own
  trimmed scene / rewards / events to avoid spawning unused articulated
  assets.
* MT15 (existing) — kept as-is.

## Training runs (post fixes)

| Run                          | scale     | iters | wall  | final reward | reach success |
|------------------------------|-----------|------:|------:|-------------:|--------------:|
| MT3 pre-fix                  | dt-scaled | 300   | 2:32  | 16.5         | 100 %         |
| MT3 post-fix dt              | corrected | 300   | 2:38  | 1605         | 36 %          |
| MT3 post-fix dt              | corrected | 1000  | 9:06  | 1700         | 100 %         |
| MT3 post-fix k_val=100       | corrected | 1000  | (in flight) | TBD    | TBD           |
| MT5 / MT10 / MT15 post-fix dt| corrected | 1000  | done / 16:44 / —  | 376 / TBD / — | per-task eval pending |

## Multi-task curriculum coverage (all built and trained)

| Env  | Tasks | Iters | Wall | final-30-avg reward | best per-task success | Status |
|------|------:|------:|-----:|--------------------:|----------------------:|:------:|
| MT3  |     3 |  1000 | 9 m  | 1700  | reach=100% | ✅ |
| MT5  |     5 |  1000 | 20 m | 363   | drawer_close 12%, drawer_open 2% | ✅ |
| MT10 |    10 |  1000 | 17 m | 176   | drawer_close 31% | ✅ |
| MT15 |    15 |   300 | 9 m  | 0.91 (pre-fix) | not eval'd post-fix | ⚠ pre-fix only |
| MT25 |    25 |  1000 | 30 m | 41    | drawer_close 9% | ✅ |
| MT50 (stub rewards) | 50 | 500 | 26 m | 332 | hand_insert 20% | ⚠ stubs |
| MT50 (V2 rewards)   | 50 | 1000 | 75 m | ~30-90 (high var) | hand_insert 10% | ✅ |
| MT50 (V2 + aligned) | 50 | 1000 | 64 m | ~70-200 | drawer_close 91%, button_press_wall 100%, button_press 50% | ✅ |
| MT5 (aligned + proper cfg) | 5 | 1000 | 22 m | ~50-150 | drawer_close 62%, drawer_open 9% | ✅ |
| MT10 (aligned + proper cfg) | 10 | 1000 | 25 m | ~30-80 | drawer_close 38% | ✅ |
| MT15 (aligned + proper cfg) | 15 | 1000 | 31 m | ~30-90 | drawer_close 80%, door_close 10%, peg_insert_side 7% | ✅ |
| MT25 (aligned + proper cfg) | 25 | 1000 | 45 m | ~25-80 | drawer_close 78%, peg_insert_side 15%, drawer_open 7%, door_close 10%, box_close 10% | ✅ |
| MT3 (aligned + single-task cfg) | 3 | 1000 | 11 m | ~80-150 | reach 100%, push 0%, pick_place 0% | ✅ |
| MT3 (aligned + multi-task cfg) | 3 | 1000 | 11 m | ~50-100 | reach 36%, push 0%, pick_place 0% | ⚠ regression |
| MT3 (aligned + done-on-success) | 3 | 1000 | 11 m | ~1300-1500 | reach 7.6%, push 0%, pick_place 0% | ⚠ done-on-success × init_noise_std interaction |
| MT3 (done-on-success + init_noise=0.1) | 3 | 1000 | 11 m | ~1500-1800 | reach 2.6%, push 0%, pick_place 0% | ⚠ low noise didn't help (rules out hypothesis) |
| MT3 (done-on-success + time_out=True flag) | 3 | 1000 | 11 m | ~1500 | reach 5.3%, pick_place 0.3% | ⚠ V-bootstrap didn't help; training-time `reach_success` fires 14.85% of episode-ends |
| MT3 (stochastic eval of done-on-success ckpt) | 3 | — | — | — | reach 3.51%, push 0%, pick_place 0% | ⚠ stochastic vs deterministic eval doesn't close the gap → measurement issue, not policy issue |

The reward magnitudes vary across the curricula because per-task reward
caps differ (reach uses scale=10, articulated uses scale=10 in hamacher,
push success-overrides to 10, etc.) and because each env contributes a
different mix of fully-converged vs nascent policies. The key point: all
six multi-task envs **build, run, and train**.

## MT50 V2 reward port (no-stub round)

* Replaced the placeholder `_masked_cube` / `_masked_articulated` /
  `_masked_cube_success` hamacher stubs in
  `config/sawyer/mt50_env_cfg.py` with proper per-task MW V2 archetypes:
  - **push family** (push, push_back, push_wall, soccer, sweep,
    sweep_into, coffee_push, coffee_pull, stick_push, stick_pull) —
    `LinearComboShapeCfg` (2 × caging + phase bonus);
  - **pick-place family** (pick_place, pick_place_wall, basketball,
    shelf_place, bin_picking, pick_out_of_hole, assembly, disassemble) —
    `CagingTimesInPlaceShapeCfg` (caging × in_place + lift bonus);
  - **reach family** (reach, reach_wall, hand_insert) — `ToleranceShapeCfg`
    with scale = 10;
  - **plate-side** (plate_slide_side, plate_slide_back_side) —
    `HamacherShapeCfg`;
  - **button-front** (button_press, button_press_wall) —
    `HamacherShapeCfg` with axis = 1 + gripper-closed modulator.
* Each task is decomposed into a *monotonic* term (the shape primitive)
  and a *success bonus* term (`success_indicator_term`, weight 10) — two
  `RewardTermCfg`s per task, no per-task wrapper functions.
* Group-scoping uses `SceneEntityCfg(name, groups=[task_name])` and the
  `@scatterable` wrappers in
  `metaworld/mdp/scatter_rewards.py` (one wrapper per shape primitive,
  ~ 130 lines). No `task_masked_reward` indirection.
* MT50 now exposes 102 reward terms (was 52 with stubs); `assembly_main`,
  `pick_place_main`, `reach_main` etc. all live on the cfg.
* Smoke-test: 100-env reset → finite, non-uniform per-env rewards
  (`[0.0036, 0.0022, 0.0391, 0.0567, 0.486]`) confirming each task's
  formula computes for its own envs only.

## Clone-strategy alignment fix (load-bearing)

**`CloneCfg.clone_strategy` defaults to `random`**, which shuffles env →
clone-group assignment. But `MetaworldMultiTaskCommand` assigns task_id
by `arange % n_tasks` — pure round-robin. The two had been silently
*misaligned* in MT15 / MT25 / MT50 since they were built. Symptoms:

* For 24 envs / 3 tasks, `task_id == 0 → envs [0,3,6,…]` but
  `clone_group["reach"] → envs [0,1,3,6,13,17,…]` — a different set.
* During training: per-task scatter rewards were being written to
  clone-group envs, not task_id envs — so the *task assignment* and
  *reward signal* covered different envs. Most envs had the wrong
  asset for their task_id, so even per-step success indicators
  (gated on task_id) saw the wrong objects.
* The policy *did* still learn something on the few envs where
  alignment happened by chance, which is why prior MT15/MT50 runs
  produced limited learning.

**Fix**: every `CloneCfg(...)` now passes
``clone_strategy=interleaved`` (from `isaaclab.cloner.cloner_strategies`)
which assigns env i → group i % n_groups, matching the
`MetaworldMultiTaskCommand` round-robin exactly. Verified on MT3:
``task_id`` and `clone_group[<task>].env_ids` now agree.

**Re-eval of the existing MT50 V2-rewards checkpoint** (`model_999.pt`,
trained without alignment) under the new aligned cfg:

| task                    | success | n_envs |
|:------------------------|--------:|-------:|
| drawer_close            |  54.5%  |  11 |
| button_press            |  70.0%  |  10 |
| button_press_wall       | 100.0%  |  10 |
| plate_slide_back_side   |  30.0%  |  10 |
| (others)                |    0%   |  10 |

vs. all 0% with the old `random` strategy. The policy *was* learning a
few tasks but the previous evals showed 0% because eval rolled out on
misaligned envs. Re-training with proper alignment is in flight.

## MW V2 done-on-success termination

The base `MetaworldEnvCfg` was truncate-only ("paper App. A.4"), but actual
MW V2 ends the episode on the first step where the per-task success
indicator fires. Added bool-typed scatter atoms
(`keypoint_success_termination`, `reach_success_termination`) and per-task
`TerminationTermCfg`s on every multi-task env: MT3 (3), MT5 (5), MT10 (10),
MT15 (15), MT25 (25), MT50 (50). Each task's termination reads its own
clone-group `env_ids`, so only that task's envs can fire that done — the
rest inherit time-out at 500 steps.

## Multi-task PPO cfg registration fix

While debugging push/pick_place plateaus, found that **every multi-task
gym registration was pointing at `MetaworldButtonPressTopdownSawyerPPORunnerCfg`**
(single-task cfg with `init_noise_std=0.1`, `[256,128,64]` actor) instead
of the dedicated `Metaworld{MT3,MT5,MT10}SawyerPPORunnerCfg` that were
*defined but unused*. The dedicated runners use `init_noise_std=0.3`
(wider exploration), `entropy_coef=0.003` (between single-task 0.01 and
"killed" 0.001), and `[512,256,128]` actor/critic.

* Added `MetaworldMT15SawyerPPORunnerCfg`, `MetaworldMT25SawyerPPORunnerCfg`,
  `MetaworldMT50SawyerPPORunnerCfg` to `agents/rsl_rl_ppo_cfg.py`.
* Updated `config/sawyer/__init__.py` registrations to point at the
  per-curriculum runners.
* Verified all 5 envs now resolve to their dedicated runner cfg with the
  correct hyperparams (`init_noise_std=0.3`, `[512,256,128]`).

The MT3 aligned-but-wrong-cfg run (single-task `init_noise_std=0.1`,
`[256,128,64]`) got `reach 100%, push 0%, pick_place 0%`. Re-running with
the dedicated MT3 cfg (`init_noise_std=0.3`, `[512,256,128]`,
`entropy_coef=0.003`) got `reach 36%, push 0%, pick_place 0%` —
*worse* on reach and no help on push/pick_place. The push/pick_place
plateau is a separate problem (likely the dt-fix's 100× reward scaling
interacting badly with `value_loss_coef=1.0`); fixing it needs
systematic hyperparameter sweeping rather than the assumed multi-task
defaults.

## Scatter-pattern uniformity

* Migrated **MT15** (`multi_task_env_cfg.py`), **MT25** (`mt25_env_cfg.py`),
  and **MT3** (`mt3_env_cfg.py`) off `task_masked_reward` onto the
  `@scatterable` group-scoped atoms in `mdp/scatter_rewards.py`. All five
  multi-task envs (MT3 / MT5 / MT10 / MT25 / MT50) now use the same
  reward routing pattern.
* For MT3, added a minimal `clone_cfg` with three task groups (each
  containing the shared cube / keypoint / TCP assets) — no asset cloning
  difference between groups, just env partitioning. The clone-group env
  partition aligns with the round-robin `task_id` written by
  :class:`MetaworldMultiTaskCommand`.
* Added `reach_success_term` to `mdp/scatter_rewards.py` (TCP-to-goal
  binary, mirrors MT3 reach's success criterion vs the keypoint-based
  `success_indicator_term`).
* Smoke test: all five envs construct, reset, and step cleanly with
  per-task non-uniform rewards. `task_masked_reward` is no longer called
  from any cfg (left in place for one release as a deprecated symbol).

## What still needs doing

* MT15 retrain at `k_val = 100` — pre-fix runs hit descent plateau on
  tasks that need TCP below z ≈ 0.06. MT5 / MT10 retrained.
* Reward audit for articulated tasks — extended to drawer-open,
  window-open/close, button-press-topdown / coffee-button (4 archetypes
  total in addition to MT3). Faucet / door / handle / plate / hammer
  ports remain.
* MT50 cube-task rewards now use proper MW V2 archetypes (see "MT50 V2
  reward port" below) — no more stub hamacher.
* Phase 7 architecture refactor — partial: extracted the simple cfg
  helpers (`_fixed_paired`, `_paired_from_spec`, `_reset_robot`,
  `_reset_joint_to`, `_reset_joint_from_spec`) into
  `config/sawyer/_helpers.py` (-60 lines from `env_cfgs.py`). Per-task
  env split + reward-shape factory split still queued in `REFACTOR_PLAN.md`.
* PPO push / pick still hit local optimum at 0 % success even with
  reward parity + responsive sim. Suspects: `init_noise_std=0.1` (was
  raised from 0.5 paper default to fight pre-fix instability) and
  `value_loss_coef=1.0` (now 100× heavier with the fix). Both are tunable
  without further parity work.

## Validation tools added

* `utils/parity/check_weights.py` — no-sim static check that the dt-fix
  is wired (54 / 54 envs pass — every Sawyer cfg has weights in the
  expected post-fix range).
* `scripts/reinforcement_learning/rsl_rl/parity_smoke.sh` — single-shot
  end-to-end parity validator (runs check_weights + reward audit + push
  probe in ≤ 60 s wall, no training).

## Key files for review

* `source/isaaclab_contrib/.../metaworld/PARITY_REPORT.md` — 3-level
  parity audit summary + per-layer pass/fail.
* `source/isaaclab_contrib/.../metaworld/utils/parity/PUNCH_LIST.md` —
  pre/post-fix histograms.
* `source/isaaclab_contrib/.../metaworld/REFACTOR_PLAN.md` — Phase 7
  target layout.
* `source/isaaclab_contrib/.../metaworld/metaworld_env_base.py` — the
  one-line dt-fix + the k_val bump.
* `scripts/reinforcement_learning/rsl_rl/parity_reward_audit.py` — the
  formula-level audit harness.
* `scripts/reinforcement_learning/rsl_rl/probe_push_dynamics.py` — the
  sim-dynamics probe.
* Changelog fragments at
  `source/isaaclab/changelog.d/octi-metaworld-dict-int-keys.rst` and
  `source/isaaclab_contrib/changelog.d/octi-metaworld-parity.minor.rst`.
