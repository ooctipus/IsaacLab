# MT50 Parity Punch List

Generated 2026-05-06 from a full 50/50 sweep of `parity_compare.py` against the
MW-side dump in `data/`. See per-task reports under `reports/` and the
formatted summary from `aggregate.py`.

## Headline counts

| Layer | Metric                | Tolerance     | Pre-fix       | Post-fix (Cat-A + E) |
|-------|-----------------------|---------------|---------------|----------------------|
| L1    | tcp_mm                | 50 mm         | 45 / 50       | 45 / 50 (reset noise) |
| L1    | goal_mm               | 30 mm         | 49 / 50       | 32 / 50 (sample noise) |
| L1    | keypoint_mm           | 50 mm         | 42 / 50       | 36 / 50              |
| L2    | joint_max abs         | 0.05 m or rad | 1 / 50        | **0 / 50**           |
| L3    | isaac_obj_std         | ≥ 0.005       | 47 / 50 below | 8 / 50 below (= MW's 8 deterministic tasks — match) |
| L3    | obj_mean_mm           | 50 mm         | 39 / 50       | **0 / 50**           |
| L3    | tgt_mean_mm           | 30 mm         | 45 / 50       | 3 / 50               |

## Category A — sampling determinism (47/50, highest leverage)

Every articulated-asset task uses `_fixed_paired(obj, obj, goal, goal)` so
`std == 0`. MW samples per task with σ ≈ 0.06 m. **Fix**: extend
`MetaworldTaskSpec` with `obj_range_low / obj_range_high / goal_range_low /
goal_range_high`, default to a small ±5 cm box, and switch
`_paired_from_spec` to use those instead of the point.

MT3 (reach/push/pick-place) already samples — std=0.06–0.08 — keep as is.

## Category B — coordinate offsets (per-task, 45/50)

Sorted by `tgt_mean_mm` (target-coordinate divergence vs MW per-task mean):

```
stick-pull-v3                424 mm
door-open-v3                 391 mm
plate-slide-side-v3          374 mm
door-unlock-v3               311 mm
lever-pull-v3                299 mm
handle-pull-side-v3          291 mm
coffee-button-v3             278 mm
hammer-v3                    252 mm
door-lock-v3                 252 mm
stick-push-v3                241 mm
peg-unplug-side-v3           233 mm
handle-press-v3              222 mm
handle-pull-v3               222 mm
faucet-close-v3              210 mm
handle-press-side-v3         207 mm
peg-insert-side-v3           ...
```

Door / handle / faucet / lever / stick / hammer / coffee-button all show
≥200 mm goal divergence — these are the highest-priority numerical
corrections. Per-task fix: read MW's reset `target_pos` from the dump
JSON, paste into `TASK_SPECS[<task>].goal`.

Categorically the offsets cluster into:

* **Door family** — 4 tasks at 250–390 mm, suggesting a shared frame
  mismatch in `door_keypoint` or `mw_door` mounting.
* **Handle family** — 4 tasks at 220–290 mm, likewise pointing at the
  `handle_press` / `handle_press_side` USDs being placed at the wrong
  world coordinate.
* **Stick / Coffee** — 220–425 mm, indicating the stick / mug spec coords
  drifted from MW's defaults.

## Category C — Sawyer reset noise (45/50, TCP > 50 mm)

`reset_joints_by_offset(±0.05 m)` causes ~50 mm TCP scatter. MW IK-resets
to a deterministic TCP. Two options:

1. Lower the offset to 0.0 — exact MW match, zero exploration noise.
2. Keep ±0.02 — modest randomization, ~20 mm TCP scatter.

Option 2 is closer to PPO best practice (Octi feedback: "vectorization is
our advantage"). Either way, the **base** Sawyer joint init pose may not
match MW's per-task reset (some tasks override `hand_init_pos`). Audit:
extract MW's `init_tcp` per task from the dump and align Isaac's
`SAWYER_METAWORLD_CFG.init_state.joint_pos` (or a per-task IK reset) so
that with offset=0 we hit MW's TCP within 5 mm.

## Category D — reward magnitude (top 10)

```
faucet-close-v3             |Δr|=2.37
faucet-open-v3              |Δr|=2.24
box-close-v3                |Δr|=1.99
reach-v3                    |Δr|=1.34
reach-wall-v3               |Δr|=1.33
drawer-open-v3              |Δr|=1.21
door-lock-v3                |Δr|=1.08
door-unlock-v3              |Δr|=0.98
plate-slide-v3              |Δr|=0.87
hammer-v3                   |Δr|=0.84
```

Faucet rewards differ by factor ~10× (Isaac ~0.02, MW ~2.4). Likely
hamacher `scale=10.0` × in-place sigmoid mismatch. Reach-family rewards
also off by ~1, which is the entire scale of the tolerance shape.

Approach: run the per-component reward audit (`info_dict` from MW vs the
Isaac reward-term breakdown) on faucet/box/reach first — those three
cover 4 archetypes (hamacher, linear-combo, tolerance-shape).

## Category E — joint reset divergence

```
faucet-close-v3   joint_max=1.047 rad ≈ π/3
drawer-close-v3   joint_max=0.010    (tiny)
button-press-...  joint_max=0.005    (tiny)
```

faucet-close: `TASK_SPECS["faucet_close"].joint_reset_value=π/3` but MW
resets to 0. Spot fix: set to 0.0 in TASK_SPECS, swap `goal` to π/3 if the
task requires turning to that angle. Confirm against MW's
`obj_init_angle`.

## Fix sequence

1. **A → sampling ranges** (1 spec change, code-level helper, 47-task win).
2. **E → faucet-close joint reset** (1 line).
3. **B-door / B-handle / B-faucet** — coordinate corrections in
   `TASK_SPECS` (per-task) using MW dump's `_target_pos` and `obj_init_pos`
   as ground truth.
4. **D → reward audit** (per-archetype: faucet hamacher, box hamacher,
   reach tolerance).
5. **C → Sawyer init pose** (later; depends on whether we keep reset noise).
