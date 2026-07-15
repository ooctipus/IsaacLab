<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Motion Retargeting Architecture and Acceptance Contract

This document is normative for motion retargeting under `core/multi_task`. The
release target is one clean pipeline supporting LAFAN -> G1, LAFAN -> SMPL,
CMU -> SMPL, and CMU -> G1. Solver convergence, visual plausibility, or parity
with released coordinates is not sufficient: every route must pass the same
independent, full-corpus certificate.

## Pinned Primary References

The references define algorithm precedents, not this package's module layout
and not ground-truth output.

- [cuMotion at ac115fdb][cumotion] is the production-oriented reference for
  collision-aware IK and trajectory-optimization contracts. It incorporates
  hardened descendants of algorithms introduced by cuRobo. It does not provide
  a human-retargeting outcome oracle.
- [cuRobo MotionRetargeter at cb00c2d6][curobo] owns the solve-staging
  precedent: broad global IK for the first frame followed by previous-solution,
  velocity-limited local IK or MPC. Its returned result is not an independent
  certificate, so its success handling must not be copied as an acceptance
  policy.
- [ProtoMotions/PyRoki at 49fe5ad6][protomotions] owns the whole-trajectory
  objective precedent: local and global landmark alignment, raw adjacent-frame
  root/joint smoothness, a separate physical-time soft velocity-limit term, and
  source-derived foot contact terms. Its fixed-length trim/pad workflow and
  visual-only validation are not part of this contract.
- [NVIDIA SOMA Retargeter at b3ef2708][soma-pipeline] owns the concrete G1
  correspondence, translation/rotation weighting, and 24-step analytic Newton
  warm-start precedent. Its initialization frames, [G1 mapping][soma-map],
  [foot stabilizer][soma-feet], and [joint clamp][soma-clamp] are references,
  not correctness oracles. Any post-process or clamp must be followed by the
  complete independent certificate.

When a reference is upgraded, pin the new revision here and record which
measured behavior justified the upgrade.

## Composition Root and Ownership

`../motion_env_cfg.py` is the composition root. A preset selects independent
axes for source evidence, target morphology, and the retargeting pipeline.
There must be no LAFAN-G1, LAFAN-SMPL, CMU-G1, or CMU-SMPL product module,
alias, fallback, or composite preset.

The dependency direction is:

```text
motion_env_cfg
  -> data/sources + robots/{g1,smpl}
  -> retarget.py
  -> mdp/commands/{commands_cfg,motion_trajectory,motion_task_table_builder}
  -> mdp/commands/motion_task_table
```

File ownership is strict:

- `motion_env_cfg.py` composes readable presets and wires the same acceptance
  policy for every source, target, and solver family.
- `motion/data/sources/*.py` decodes source evidence or released comparison
  data. A source decoder does not know the target robot or choose a solver.
- `motion/robots/{g1,smpl}/` owns target kinematics, morphology, limits,
  support geometry, calibration, and the initializer policy required by that
  morphology. Both G1 and SMPL select batched frame IK. SMPL's full-body
  rotation map supplies a target-owned seed; it never bypasses feasibility.
  A target module does not know the source dataset.
- `motion/retarget.py` owns only semantic correspondence and projection from
  source anatomy to target morphology. It carries the target-selected
  initializer policy without choosing it. It produces targets; it does not
  solve, certify, select a backend, or own route-specific policy.
- `motion/mdp/commands/commands_cfg.py` owns one explicit algorithm config and
  one shared acceptance policy. Exact, analytic, and optimized inputs receive
  the same intrinsic certificate.
- `motion/mdp/commands/motion_trajectory.py` owns staged solve orchestration,
  fixed scratch, backend conversion, and solver-independent semantic, contact,
  and collision metrics. Algorithm-specific kernels may live under an
  `impl/` boundary, but generic code must not import source- or robot-specific
  modules.
- `motion/mdp/commands/motion_task_table_builder.py` owns offline
  materialization, canonical velocity reconstruction, certification, and
  fail-closed corpus assembly.
- `motion/mdp/commands/motion_task_table.py` owns the immutable table and
  quality/coverage schema, not build policy.
- `tests/` owns structural absence gates as well as numerical gates.
- `scripts/reinforcement_learning/forward_backward/phase3/benchmark_motion_retargeting.py`
  owns the four-route acceptance report. A limited `inspection_limit` run is a
  bring-up diagnostic and must be labeled non-accepting.

Do not add a helper, class, factory, adapter, or file for a one-use tensor
operation. Fix an ownership boundary instead of adding a compatibility layer.

## Mathematical Contract

For frame `t`, the source decoder supplies native-time world-space anatomy,
physical timestep `dt`, and source-derived contact state. Decoding never
resamples, trims, pads, or changes clip boundaries. The target declares all
optimization evidence and the smaller subset that is independently meaningful
for publication:

- all position roles `L_all` and required position roles `L_req`;
- all distal roles `D_all`, independently declared contact distal roles
  `D_contact`, and publication-required distal roles `D_req`;
- root orientation and, when calibrated source rotations exist, diagnostic
  non-root orientations;
- target-owned support geometry and source contact channels ordered by
  `D_contact`.

`D_contact` and `D_req` are separate selections from `D_all`: contact rows own
source-contact state and support terms, while required rows own the publication
fidelity gate. Every contact role must also be required, but the two selections
do not need to be equal or have the same ordering.

The root frame is anatomical rather than copied from a source body frame. Let
`B_s(t)` be the orthonormal source anatomy basis and `B_t,rest` the
target-rest anatomy basis. The required target root is

```text
R_root*(t) = B_s(t) B_t,rest^T .
```

For a calibrated body rotation `j`, rest-relative transport first produces

```text
R_bar_j(t) = R_s,j(t) R_s,j,rest^-1 R_t,j,rest .
```

One world gauge then aligns every transported row to the anatomical root while
preserving all transported relative rotations:

```text
Delta(t) = R_root*(t) R_bar_root(t)^-1
R_j*(t) = Delta(t) R_bar_j(t) .
```

An `anatomical_root` source exposes only the root rotation row. A
`calibrated_body` source exposes the root plus calibrated non-root rows.
Unknown cross-rig body frames are never treated as interchangeable.

Position targets preserve source edge directions but use target-owned segment
lengths. Distal targets use an explicit target-owned law:

- `between_positions` transports a source position direction and applies the
  target endpoint length; G1 uses it for its physical +X hand endpoint rows;
- G1 `wrist_forward` uses anatomy-forward projected orthogonal to the forearm
  for raw anatomical sources, or a rest-calibrated wrist-local forward axis for
  calibrated sources. These existing +Z Proto wrist cues remain soft
  objectives and diagnostics.

G1 contact owns only the left/right foot rows. Its publication-required distal
rows are the two feet plus the physical left/right +X hand endpoints; the +Z
`wrist_forward` rows are not publication evidence. SMPL declares only its two
foot distal rows, which are both contact and required rows.

Let `q[t]` contain the target free root and joints. Target forward kinematics
provides positions `P_i(q[t])`, distal points `E_e(q[t])`, rotations
`R_j(q[t])`, support points, and collision distances. Optimization may use
all declared evidence:

```text
min_Q sum_t (
    sum_i in L_all w_pos[i] rho(||P_i(q[t]) - p_i*(t)||^2)
  + sum_e in D_all w_dist[e] rho(||E_e(q[t]) - e_e*(t)||^2)
  + sum_j w_rot[j] ||Log(R_j*(t)^T R_j(q[t]))||^2
  + w_default ||joints(q[t]) - joints_default||^2
) + J_time(Q, dt) + J_contact(Q, c*, dt) + J_collision(Q)
```

Position and distal coefficients are the global objective multiplier times a
target-role multiplier divided by the corresponding target-owned length.
Rotation rows carry their target-role multiplier directly. This lets each
robot express meaningful evidence priorities without creating source-specific
solver configs.

`J_time` uses branch-safe differences divided by the native physical
timestep. `J_contact` is gated only by source contact states and measures
support gap, normal tilt, slip speed, and stable-interval drift.
`J_collision` uses target-owned signed-distance geometry.

For a `fixed` source-root policy, used by SMPL, source refinement runs in a
reduced tangent space: free-root translation and rotation tangent DOFs
`[0, 1, 2, 3, 4, 5]` are frozen in every solve direction and their Jacobian
columns are zero. The authored root coordinates `q[:7]` are restored after
each source solve as an exactness guard. This is not a residual equality.
Frame-local IK is proposal-only for fixed-root targets and also has its root
restored. Contact refinement is a separate transaction: after contact-height
alignment it regains full root freedom.

Publication source fidelity is deliberately narrower than the optimization
cost. It gates the maximum required-position error, required-distal point
error, required-distal direction error, and root-rotation error. Maxima over
all positions, all landmark directions, all distal rows, and non-root
rotations remain diagnostics. A soft cost or diagnostic improvement is never
evidence that a hard contract was met.

Required distal directions receive one bounded raw-unit objective without
becoming trajectory inequalities. For normalized robot/source directions `u`
and `t`, the residual is
`max(0, ||u - t|| - 2 sin((theta_max - delta) / 2))`, where
`delta = min(0.01 rad, theta_max / 2)`. Its precision is 100, matching the raw
orientation scale used by the reference cuRobo retargeter; unlike the rejected
normalized chord-energy constraint, neither residual nor Jacobian divides by a
small acceptance tolerance. The solver-free publication certificate still
evaluates the exact angular maximum against `theta_max`.

Every retained candidate is also subject to these hard contracts:

- finite target coordinates, velocities, and forward kinematics;
- a normalized free-root quaternion;
- target joint position and canonical native-time velocity limits;
- declared ground, obstacle, and self-collision clearance;
- source-stable contact support;
- exact clip identity, frame order, boundaries, and timebase.

## Staged Algorithm

The implementation proceeds from one target-owned composition root:

1. Decode and validate the complete source clip at its native frame count,
   boundaries, order, and timestep.
2. Project source anatomy to the selected target morphology, including the
   anatomical root, optional calibrated-body gauge, target segment lengths,
   distal laws, and contact probes.
3. Construct the target-owned seed. G1 supplies its anatomy-derived root and
   default non-root coordinates. SMPL converts the 22 mapped global rotations
   into topological local ordered-XYZ coordinates, selects continuous Euler
   branches, and projects the mapped candidate into coordinate limits as a
   seed, not as a certificate.
4. Execute batched frame IK for both targets. The broad deterministic solve on
   each true first frame starts from the target-owned seed, then advances in
   native frame order from the previous solved frame. Clips remain parallel in
   lockstep while the mathematically dependent time axis remains serial. The
   24-step local solve normalizes the root quaternion and projects target
   coordinates into their position limits. It deliberately does not impose a
   greedy per-frame velocity-reachable interval: native-time velocity,
   acceleration, and jerk are coupled properties owned once by the whole-clip
   solver. The global/local continuation follows cuRobo and SOMA while the
   offline temporal ownership follows ProtoMotions. The whole-clip solver owns
   temporal feasibility, acceptance, and publication.
5. Restore hard source feasibility before objective refinement. Initializers
   already inside the source envelope skip restoration; all others run the
   feasibility-only trajectory solve and checkpoint the first certified
   trajectory. The objective phase then relinearizes from that checkpoint.
   Fixed-root targets use the reduced tangent space above, and bounded
   direction activation remains an ordinary objective. Capture
   `source_attempt` before rollback. Objective failure restores the certified
   feasible checkpoint rather than the raw initializer; restoration failure
   fails closed. Only objective-converged trajectories satisfying source
   acceptance can enter contact refinement or publication.
6. Record the retained result as `post_source` before contact-height
   alignment. If applicable contact still fails, align the height gauge,
   snapshot the aligned state, and attempt contact/collision refinement with
   full root freedom and clearance inequalities. Capture
   `contact_attempt` before rollback; commit only when convergence, required
   source fidelity, contact, and geometry all pass.
7. Reconstruct canonical target velocities once from the retained coordinates,
   run the solver-free certificate, and publish only complete accepted
   manifests.

The inspection stages are `frame_seed`, `source_attempt`, `post_source`,
`contact_seed`, `contact_attempt`, and `final`. Unattempted attempt stages are `NaN`;
they are never filled with retained values. `frame_seed` names the selected
target initializer result, whether direct or fitted. Source and contact
attempts are transactional per clip, so a failed phase cannot contaminate a
neighboring clip or masquerade as final output.

Released coordinates are comparison data or seeds only. They do not bypass
the same intrinsic constraints. Internal chunking is allowed only when state
crosses chunk boundaries and reassembly preserves exact source identity.

## Padding-Free GPU Execution Plan

The trajectory mathematics above is independent of its execution schedule.
Execution uses flat, offset-indexed storage so complete variable-length clips
share the GPU without padding, truncation, duplicated tail frames, or valid-row
masks:

```text
q_flat[sum_i T_i, Q]                 trajectory_offsets[B + 1]
candidate_q[sum_f K_f, Q]            candidate_offsets[F + 1]
temporal_edge_src[E], edge_dst[E]     edge_segment[E]
```

`trajectory_offsets[i + 1] - trajectory_offsets[i]` is the exact native frame
count of clip `i`. `candidate_offsets[f + 1] - candidate_offsets[f]` is the
number of real proposals retained for frame `f`; rescue proposals therefore do
not allocate rows for already-unambiguous frames. Temporal velocity and
acceleration kernels consume explicit real edges or derive neighbors inside
one offset interval. No stencil can observe workspace capacity or cross a clip
boundary. A reusable allocation may have unused capacity after the active flat
prefix, but the tail is neither initialized as data nor sampled by a kernel.

The production transaction is:

1. Generate deterministic frame proposals in one flat `[frame, candidate]`
   problem list. Use four ordinary proposals initially; add rescue proposals
   only for failed or branch-ambiguous frames.
2. Solve frame IK in GPU batches chosen from live free memory. Accumulate
   `J^T J` and `J^T r` directly while traversing target kinematics; do not
   materialize a global dense Jacobian, residual matrix, or full FK history.
3. Select a clip-consistent proposal path with a ragged second-order dynamic
   program when acceleration participates in the score. The selected path is
   immediately certified and becomes `q_incumbent`.
4. Refine complete trajectories for source fidelity, native-time smoothness,
   contact, and collision. The incumbent is always feasible. Trial states are
   implicit and are committed only after exact source-envelope, bounds,
   contact, collision, finiteness, and objective checks. A rejected trial is a
   no-write operation; iteration or time limits return the best certified
   incumbent, never the final attempted state.
5. Reconstruct canonical velocities and run the independent publication
   certificate once more. This last certificate is authoritative even when an
   optimization kernel reports success.

The source envelope is evaluated per target-required observable and per clip,
not as a weighted sum. A trial must remain within the configured required
position, distal-position, distal-direction, and root-rotation limits. The
line search evaluates `alpha = 1` first. Only segments rejecting that trial
enter a flat list of smaller alphas, evaluated concurrently. FK, objective,
contact, collision, bound checks, source-envelope checks, and partial
reductions should be fused when their intermediate values have no other
consumer. Deterministic segment reduction, selection, and commit remain a
separate boundary so acceptance is auditable.

### Frame IK kernel selection

Frame IK has two implementations of the same numerical contract:

- **monolithic fused:** one cooperative problem owns kinematic traversal,
  normal accumulation, damping, factorization, update, projection, and the
  convergence summary;
- **chunked fused:** residual/body/coordinate tiles accumulate into compact
  normal-equation blocks before one solve/update when a monolithic kernel would
  lose occupancy to registers or shared memory.

The memory planner chooses active problem rows from current free memory and
the exact per-row live set; it has no fixed 20 GiB ceiling. The same planner
therefore scales from a 24 GiB workstation to 80 GiB H100 and 288+ GiB GB200
devices. Capacity is the largest balanced batch that preserves the minimum
launch count after a safety reserve. Kernel choice depends on `(Q, D, R)`,
compiled register/shared-memory use, and measured occupancy, not on source
dataset or route name. G1-sized systems are expected to favor monolithic or
small-coordinate tiles. SMPL-sized systems may favor residual/coordinate
chunking, but that is a benchmark result rather than policy.

The first implementation is monolithic for both current targets. Their useful
fixed-root tangent dimensions are 29 and 69, so packed normal storage is small
enough to test one-block-per-candidate execution before introducing tiling.
Chunking is admitted only when generated-code inspection shows spills or when
measured occupancy, shared-memory pressure, or a small rescue pool makes it
faster. The planner derives the chunk count from live candidates, tangent
dimension, SM count, and device limits.

For the current target shapes, eliminating dense per-row intermediates changes
the dominant frame-IK live storage approximately as follows:

| Target shape | Existing per problem | Fused normal form per problem |
|---|---:|---:|
| `Q=36, D=35, R=63` | 12,792 bytes | 276 bytes plus cooperative scratch |
| `Q=76, D=75, R=138` | 47,672 bytes | 572 bytes plus cooperative scratch |

These figures are liveness estimates, not acceptance thresholds. Benchmarking
must report registers, shared memory, occupancy, launches, wall time, and peak
allocated bytes for both implementations before deleting the loser.

### Trajectory linear solver selection

Trajectory refinement exposes one structured normal-operator contract with two
candidate backends:

- **direct block-band cyclic reduction** assembles only real diagonal and
  temporal off-diagonal blocks and eliminates independent blocks in parallel;
- **structured matrix-free PCG** applies frame-local and temporal terms without
  assembling the full band matrix and uses a compact block preconditioner.

Velocity, acceleration, and jerk make the current temporal normal operator
block bandwidth three. Ordinary block-tridiagonal cyclic reduction is therefore
not a valid implementation. The direct candidate must use generalized banded
reduction, nested dissection, or an explicitly costed three-frame supernode.
Supernodes scale the factor work with `3D`, so larger memory alone does not make
them efficient for high-dimensional targets.

The existing one-block-per-clip forward/backward factorization is not called
cyclic reduction because it is serial along time. It remains a comparison
implementation only until true parallel elimination exists. The production
selector compares exact live-memory estimates and representative solve time at
the actual `(Q, D, R, T_i)` shapes. A backend is eligible only after dense
float64 conformance on small ragged systems, residual reduction, deterministic
repeatability, and publication-envelope parity. Among eligible backends it
chooses the faster option that fits the current memory budget and records the
decision in run evidence. Source and target names never participate. The
expected tendency is direct cyclic reduction for smaller coordinate blocks and
matrix-free PCG or chunked direct elimination for larger SMPL blocks, but the
measurement decides.

Both backends consume `trajectory_offsets`; neither allocates `[B, T_max, ...]`.
Direct assembly stores only live band blocks. Matrix-free PCG stores a bounded
number of flat `F x D` vectors and compact preconditioner blocks. Solver
iterations stop per segment without host synchronization, and finished
segments no longer participate in operator work.

### Kernel liveness and fusion gates

Every proposed kernel must have a liveness table before implementation. An
intermediate is global only when another kernel genuinely consumes it or when
it is required evidence. In particular:

| Stage | Persist | Fuse or recompute |
|---|---|---|
| frame proposal | candidate coordinates, compact cost/status | FK transforms, residual rows, dense Jacobian |
| proposal selection | selected coordinates, predecessor/path summary | pairwise transition tiles |
| trajectory linearization | compact operator/preconditioner state | full Jacobian and inactive objective rows |
| trial evaluation | per-segment maxima, cost, accept code | trial coordinates, FK, contact/collision samples |
| commit | incumbent coordinates and certificate | rejected trials and rollback copy |

Fusion is rejected when it makes register pressure slower than the measured
unfused path, obscures the independent acceptance reduction, or prevents
native Warp capture. No hot kernel allocates, synchronizes the host, calls
Torch, emits debug output, or returns a dynamically sized tensor.

### Implementation and evidence order

Work proceeds in vertical slices; passing broad legacy tests is not a reason to
advance a mathematically incomplete slice:

1. Freeze four-route numerical and visual baselines, worst-clip identities,
   profiler traces, and the independent certificate.
2. Add flat candidate/trajectory contracts and negative tests forbidding
   padded `[B, T_max, ...]` ownership, duplicated tail frames, or route-specific
   backend branches.
3. Implement ragged proposal selection and certified-incumbent rollback with
   small dense CPU oracles. Verify each regression fails against the old path.
4. Benchmark monolithic and chunked fused frame IK on both target shapes; keep
   only numerically conformant winners or a dimension-driven split.
5. Implement true block cyclic reduction and compact matrix-free PCG behind one
   structured operator contract. Compare both at short, median, and long real
   clip lengths and under 24, 80, and 288+ GiB memory budgets.
6. Fuse staged-alpha trial evaluation, source-envelope certification, and
   transactional commit; prove a rejected or capped iteration returns the
   original certified incumbent bit-for-bit.
7. Run targeted numerical tests, four-route limited clips, known worst clips,
   dynamic visual comparisons, then the complete corpus. Only after these pass
   run repository-wide formatting and architecture audits.
8. Delete the superseded serial, padded, generic-IPM, unused-backend, alias,
   fallback, and temporary benchmark paths. The final existence audit is part
   of completion, not follow-up cleanup.

The July 2026 v65 LAFAN-to-G1 profile is the starting performance baseline:
3,000 frames in 36.55 seconds (82.1 frames/s), about 585.3 MiB incremental
Torch CUDA memory, 895,938 launches, and a 12.96-second 200-iteration
trajectory phase for a representative 300-frame profile. The first stable
GPU redesign targets 10--25 minutes per million frames; the mature target is
2--8 minutes per million frames, subject to every hard four-route quality gate.
Speedups that change clip coverage or relax the source envelope do not count.

## Route and Oracle Matrix

| Route | Source evidence | Released outcome oracle | Authority |
|---|---|---|---|
| LAFAN -> G1 | Grounded LAFAN BVH | Released BFM G1 29-DoF rows for matching clips | Regression comparison; identical coordinate decoding uses the existing `2e-6` tolerance. Intrinsic gates still dominate. |
| LAFAN -> SMPL | Grounded LAFAN BVH | None | Source semantics plus the solver-free target certificate are authoritative. |
| CMU -> SMPL | Raw CMU AMASS/SMPL-H | HumEnv SMPL rows for matching clips | Compare target FK and semantics. Raw Euler equality is not authoritative, and invalid released rows must be repaired rather than exempted. |
| CMU -> G1 | Raw CMU AMASS/SMPL-H | None | Source semantics plus the solver-free target certificate are authoritative. |

Reference ownership is stage-specific: cuMotion supplies generic status and
collision-aware solver contracts; cuRobo supplies global/local staging and
ProtoMotions supplies whole-clip objective and temporal-coupling precedents;
SOMA supplies G1 mapping, weights, and 24-step warm Newton. They are not output
oracles because their sources, mappings, models, and time grids differ. No
route may borrow another route's released coordinates as ground truth.

## Full-Corpus Acceptance Policy

Acceptance runs the declared train and evaluation manifests for all four
routes. Thresholds apply to maxima over the entire route, not means or
percentiles. P99 and per-role values are diagnostics only.

The target owns which source observables are contact-bearing and which are
required for publication. Contact rows must be included in the required rows,
but the declarations remain independent:

| Target | Required positions | Contact distal directions | Required distal directions | Diagnostic-only source evidence |
|---|---|---|---|---|
| G1 | pelvis, left/right ankle, left/right wrist | left/right foot | left/right foot and physical left/right +X hand endpoint | remaining positions, all landmark directions, both +Z `wrist_forward` hand rows, non-root rotations when present |
| SMPL | pelvis, head, left/right ankle, left/right toe, left/right wrist | left/right foot | left/right foot | remaining positions, all landmark directions, non-root rotations |

Both targets use the same bounds for their required rows:

| Criterion | Hard threshold |
|---|---:|
| Finite coordinates, velocities, and FK | Every value |
| Root quaternion norm error | `<= 32 * eps(dtype)` |
| Target joint-position violation | `<= 32 * eps(dtype) * max(1, abs(bound))` |
| Canonical joint-velocity/target-limit ratio | `<= 1 + 32 * eps(dtype)` |
| Required mapped position error | `<= 0.02 m` |
| Required distal endpoint position error | `<= 0.03 m` |
| Required distal direction error | `<= 0.10 rad` |
| Root orientation error | `<= 0.10 rad` |
| Stable-contact mean support gap | `<= 0.01 m` |
| Stable-contact support tilt | `<= 0.10 rad` |
| Stable-contact slip speed | `<= 0.05 m/s` |
| Stable-contact cumulative drift | `<= 0.02 m` |
| Target collision-probe penetration below world ground | `<= 0.002 m` |
| Self-collision overlap, when certified geometry exists | `<= 0.002 m` |

The quality schema keeps the distinction mechanical. The hard source columns
are, in order:

```text
source_required_position_max_m
source_required_distal_position_max_m
source_required_distal_direction_max_rad
source_root_rotation_max_rad
```

The following columns are evidence for diagnosis and algorithm comparison, not
publication gates:

```text
source_all_position_max_m
source_all_distal_position_max_m
source_all_landmark_direction_max_rad
source_all_distal_direction_max_rad
source_nonroot_rotation_max_rad
```

`source_nonroot_rotation_max_rad` is `NaN` when the source exposes no
non-root rotation evidence. This is honest unavailability, not zero error.

Contact criteria apply only when a clip has at least one strict source-stable
frame/channel sample. A no-contact clip reports contact metrics as `NaN`,
false applicability, and zero stable-sample count. It must not report zeros.
A manifest with no applicable samples is a configuration error unless it is
explicitly declared no-contact.

`source_contact_confidence_mean` is diagnostic. It must be finite and lie in
`[0, 1]`, but it cannot substitute for applicability or a strict stable
sample count. Missing certified self-collision geometry is unavailable, never
zero overlap. Solver convergence and feasibility are required when a phase is
attempted, but solver status cannot override an output metric.

Coverage is a hard invariant:

- output clips and frames equal input clips and frames;
- clip order, identity, boundaries, and physical timebase are unchanged;
- rejected clips equal zero;
- there is no silent filtering, resampling, trimming, padding, segmentation,
  route-specific fallback, or accepted-only selector;
- acceptance never uses `inspection_limit`;
- seed, dtype, revisions, assets, target identities, and solver settings are
  recorded, and repeat runs preserve pass/fail.

If any clip fails, corpus assembly raises with its identity and failed maxima.
Released-oracle comparisons remain separate and cannot waive an intrinsic
source, limit, contact, collision, coverage, or finiteness failure.

## Reference Adoption and Evidence

Pinned libraries are sources of candidate ideas, not specifications to clone.
A reference decomposition, offset, target, time grid, or success flag carries
no authority merely because it exists upstream. The implementation may differ
whenever the alternative is clearer and has stronger evidence.

An idea is admitted by one of two routes:

1. **Analytical superiority.** State the property before implementation and
   prove it in the relevant contract: coordinate-frame invariance, exact
   native-time preservation, topology-complete initialization, bounded target
   coordinates, transactional rollback, or another mechanically testable
   property. Add a regression gate for that property.
2. **Empirical superiority.** Predeclare the metric and keep thresholds, then
   compare on the complete four-route train/evaluation matrix. Improvement must
   exceed repeat variance, preserve full coverage, pass every hard gate, and
   avoid material runtime, memory, or diagnostic regression.

Reference parity alone is neither route. Small-clip ablations may select the
next experiment, but they cannot publish a winner. The permanent report
contains route maxima, p99 diagnostics, attempted-stage evidence,
accepted/rejected counts, exact clip/frame coverage, runtime, peak memory,
revisions, assets, dtype, and solver configuration.

After a winner is established, delete losing implementations, temporary flags,
aliases, fallbacks, and one-use abstractions. If neither an analytical contract
nor full-matrix empirical evidence distinguishes a semantic change, it remains
experimental.

Architecture tests must mechanically reject:

- source-target product modules and composite aliases;
- source imports from target modules or target imports from source modules;
- robot or dataset names in the generic trajectory implementation;
- solver or objective ownership in `retarget.py`;
- family-specific threshold exemptions;
- acceptance over all-row diagnostics instead of target-required evidence;
- accepted-only corpus filtering or altered native time;
- post-hoc clamp/filter paths without complete recertification;
- solver-status-only acceptance;
- compatibility aliases and hidden fallback branches.

[cumotion]: https://github.com/nvidia-isaac/cumotion/blob/ac115fdb7737a4da07251ea85d15e4063be1ce6e/README.md
[curobo]: https://github.com/NVlabs/curobo/blob/cb00c2d60210c4ceae2ff0fb68d46d87878b0867/curobo/_src/motion/motion_retargeter.py
[protomotions]: https://github.com/NVlabs/ProtoMotions/blob/49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c/pyroki/batch_retarget_to_g1_from_keypoints.py
[soma-pipeline]: https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/pipelines/newton_pipeline.py
[soma-map]: https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/configs/unitree_g1/soma_to_g1_retargeter_config.json
[soma-feet]: https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/pipelines/feet_stabilizer.py
[soma-clamp]: https://github.com/NVIDIA/soma-retargeter/blob/b3ef2708d84bfd1314ddb52d0db6c9c211df1f57/soma_retargeter/pipelines/joint_limit_clamper.py
