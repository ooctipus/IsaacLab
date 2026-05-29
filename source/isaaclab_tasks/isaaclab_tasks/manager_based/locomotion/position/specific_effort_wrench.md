# Wrench-based specific-effort success criterion

Investigation notes from 2026-05-27 on why the position-locomotion success
criterion was accepting unnatural splayed-foot stances, what we measured, and
what changed.

## TL;DR

- The previous criterion used `applied_torque` (motor command) to compute
  per-joint "specific effort". When a joint sits at a hard URDF limit, the
  joint stop's constraint reaction absorbs the load and `applied_torque` reads
  near zero — so the metric silently undercounted mechanical load and let the
  policy farm "success" by parking joints at their limits.
- We empirically confirmed this masking using the
  `JointWrenchSensor` (Newton's actual joint reaction wrench, including
  constraint reactions). On HAA-pinned splay events, motor reads near zero
  while the joint-axis reaction torque is 30–58 N·m.
- The wrench-honest specific-effort cleanly separates good stance (≤ 0.144)
  from splay stance (≥ 0.153). The motor-based metric does not separate them
  at all (and actually scores splay slightly *lower* than good stance).
- Switched the criterion to use the joint-axis component of the wrench, with
  threshold `0.6 / N_support_feet` (= 0.15 for a quadruped). Updated the
  diagnostic snapshot to mirror the same source.

## Background and the problem

The previous success gate in `RelativeStateCommand.get_task_done` was

```
success ≡ timer ∧ settled ∧ (max_j |τ_motor_j| / (m·g·L_ref) < threshold)
```

with `τ_motor` from `Articulation.data.applied_torque` and threshold
`success_effort_multiplier / N_support_feet` (= `1.5/4 = 0.375` for a
quadruped, Biewener-style static-support reasoning).

A trained policy was reaching success with a "frog-squat" / "splayed-foot"
posture — all four HAAs pinned at ±0.491 rad (the URDF abduction soft limit),
knees deeply bent, base lowered to ~52 % of the characteristic limb length.
This passed the criterion because every per-joint `|τ_motor|` reading was
small. Intuition said the splay should require lots of HAA torque; the
criterion said otherwise.

## What `applied_torque` reports at a hard limit

`Articulation.data.applied_torque` is documented in this branch (see
`base_articulation_data.py`) as "torques applied from the actuator model
… set into the simulation". It is the post-clipping motor command. It does
**not** include the constraint reaction at a joint stop.

When the joint sits at its mechanical limit and the policy commands the same
position as the limit, PD has zero error and the motor commands ≈ 0 N·m. Any
external load that would push the joint further into the stop is absorbed by
the stop's constraint solver, not by the motor. Hence
`applied_torque[HAA] ≈ 0` even when the leg is mechanically loaded.

We can verify this with the `JointWrenchSensor` in this branch, which reads
Newton's `body_parent_f` (the world-frame wrench transmitted from parent to
child through the joint) and re-expresses it in the child-side joint frame at
the joint anchor. This *does* include constraint reactions.

## Empirical measurement

Wired `JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")` into the
position `SceneCfg` and extended `diagnostic_success_snapshot` to print, per
joint per success event:
- `q`, `dlo = q − q_lower_limit`, `dhi = q_upper_limit − q`
- `|τ_motor| = |applied_torque|`
- `‖τ_react‖`, `τ_react(x, y, z)`, `‖F_react‖` from the wrench sensor

We then captured 9 success snapshots from the splay-stance policy and
18 success snapshots from an earlier good-stance policy (trained with the
joint-position-near-default criterion) for direct comparison.

### Sanity check: motor matches wrench on free joints

For every joint not at a hard limit, the joint-axis component of the wrench
torque (`τ_react.x` in the incoming-joint-frame) equals the motor reading to
two decimals. Examples from event 12 of the splay run:

| Joint  | `|τ_motor|` | `τ_react.x` |
|--------|-------------|-------------|
| LF_KFE | 18.66       | +18.66      |
| LH_KFE | 24.81       | −24.81      |
| LF_HFE |  3.20       |  −3.20      |
| RF_HFE |  2.98       |  +2.97      |

So `|τ_react.x|` is a strict generalisation of `|τ_motor|`: it equals the
motor reading on free joints and exposes the constraint reaction on pinned
joints.

### Splay run: HAAs pinned, masking confirmed

In every splay snapshot, all four HAAs sit within 0.003 rad of their URDF
limit (`q = ±0.491`). For these joints the motor reading is small or moderate
while the wrench-axis component is large:

| Event | HAA       | `|τ_motor|` (N·m) | `τ_react.x` (N·m) | Constraint share |
|-------|-----------|-------------------|-------------------|------------------|
|  12   | LH_HAA    |  1.20             |  −32.79           | ~97 % constraint |
|  12   | RH_HAA    |  0.53             |  +31.04           | ~98 % constraint |
|  18   | LF_HAA    |  7.52             |  −68.37           | ~89 % constraint |
|  19   | RF_HAA    |  0.04             |  +37.68           | ~99 % constraint |
|  20   | RF_HAA    | 13.62             |  +58.14           | ~77 % constraint |

Across all 36 HAA samples (4 HAAs × 9 events) the joint-axis reaction is
3–60× the motor reading, with a median undercount factor of ~10× on the
masked joint and ~1.8× on the worst-joint specific effort. So the masking is
real, substantial, and consistent.

### Specific-effort comparison (worst joint per event)

```
                       Good stance       Splay stance
spec_eff (motor)       0.072 – 0.144     0.087 – 0.113
spec_eff (wrench)      0.072 – 0.144     0.153 – 0.220
median (motor)         0.099             0.094
median (wrench)        0.099             0.169
```

Two findings:

1. **Motor-based metric cannot discriminate.** Splay's `spec_eff_motor` is
   if anything *lower* than good-stance's, because masking. No threshold
   choice on the motor metric correctly classifies both regimes.
2. **Wrench-based metric cleanly discriminates.** Good-stance worst is
   0.144; splay best is 0.153. Distributions don't overlap. Any threshold in
   [0.145, 0.152] would correctly classify every observed event with no
   false positives or false negatives across these 27 events.

### Joint-load hierarchy as a side observation

Counting which joint type is the worst-loaded per event:

| Joint type | Good stance (free joints) | Splay (HAA at limit) |
|------------|---------------------------|----------------------|
| KFE        | worst in 12 / 18 events   | worst in 0 / 9 events |
| HAA        | worst in 5 / 18 events    | worst in 9 / 9 events |
| HFE        | worst in 1 / 18 events    | worst in 0 / 9 events |
| ratio HAA-max / KFE-max | 0.40 – 1.17 (median 0.70) | 1.62 – 2.29 (median 1.81) |

In good-stance posture the knee carries the most load (long lever arm in a
bent-knee stance), while HAA is typically 30–60 % of KFE; HAA only spikes
when the COM is laterally offset. In the splay, every HAA is mechanically
overloaded via its joint stop, and the HAA/KFE ratio reverses. This is a
clean fingerprint of the bad regime that the wrench-axis metric exposes and
the motor metric hides.

## Why the splay was even efficient at the wrench level

Even after correcting for the constraint masking, the splay is
mechanically a low-effort stance: maximum honest specific-effort observed is
~0.22, well under the original `1.5 / 4 = 0.375` design threshold. The
crouched, splayed geometry actually keeps total mechanical loading below
half of the static-support ceiling. Two independent reasons made the splay
work:

- The geometry itself is efficient for static support (short lever arms
  from foot reaction to each joint, especially on the bent knee).
- The HAA constraint stops absorb the residual lateral load that the leg
  geometry can't fully cancel, with no actuator effort cost.

Both of these matter. The metric fix alone would not have rejected the
splay if the original threshold (0.375) had been kept. The empirical 0.15
threshold is what makes the criterion actually useful.

## Change

Three edits, kept narrow:

1. `commands_cfg.py`: `success_effort_multiplier` default `1.5 → 0.6`.
   For a quadruped this lands the gate at exactly `0.6 / 4 = 0.15`. New
   `joint_wrench_sensor_name: str = "joint_wrench"` field points the cfg
   at the sensor.
2. `state_command.py`: `RelativeStateCommand.__init__` resolves the wrench
   sensor (fail fast with a clear message if missing), and
   `get_task_done` reads
   ```
   τ_axis_max = max_j |wrench_sensor.data.torque[..., 0]|
   ```
   instead of `|applied_torque|`. `[..., 0]` is the incoming-joint-frame
   x-component, which by Newton's URDF-derived convention aligns with the
   revolute joint axis on legged robots; the sanity check above (motor ==
   `τ_react.x` on every free joint across both runs) confirms the
   alignment for ANYmal-C.
3. `terminations.py`: `diagnostic_success_snapshot` now derives its
   per-joint `spec_eff` column from the same wrench-axis source so its
   "worst joint" report matches the criterion. Falls back to motor torque
   if a joint cannot be matched to a sensor channel, which should not
   happen on revolute-only articulations.

The `success_body_lin_speed_thresh` and `success_body_ang_speed_thresh`
"settled" gates and the `timer_done` countdown are unchanged.

## Predicted outcome on the existing splay policy

If we evaluated the new criterion (wrench-axis, threshold 0.15) against the
already-collected snapshots:

- Good-stance run (18 events): all pass (worst event 10 at 0.144, ~4 %
  margin under threshold).
- Splay run (9 events): all fail (best event 15 at 0.153, ~2 % over
  threshold; most events at 0.16 – 0.22).

For a fresh policy trained against the new criterion the diagnostic remains
in place; new exploits (e.g. unloading one foot to lower
`N_support_feet`-equivalent, or pressing the knee into its limit instead of
HAA) would show up the same way they did this time — as obvious patterns
in the snapshot output.

## Notes for follow-ups

- The joint-axis assumption (`incoming-joint-frame x = joint axis`) is an
  empirical match for ANYmal-C in this codebase. If we ever bring up a
  robot where Newton's `joint_X_c` orients differently, project the wrench
  torque vector onto the explicit `joint_axis` from the Newton model
  instead of indexing `[..., 0]`.
- The original `success_effort_multiplier / N_support_feet` shape is kept
  so the gate scales sensibly across N (biped → 0.30, hexapod → 0.10), but
  the threshold was empirically calibrated for a quadruped. Calibrate per
  morphology before claiming the metric works elsewhere.
- The unrelated `mean_per_body_shock` and a couple of reward-side helpers
  in this package still reference the removed
  `ArticulationData.body_incoming_joint_wrench_b`. They were not touched
  here but should be migrated to the `JointWrenchSensor` API when their
  call sites are revisited.
