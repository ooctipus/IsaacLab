# FPGS architecture follow-on: results and limitations

This engineering report continues the preserved handoff and the first
[structural study](STRUCTURAL_STUDY_20260910.md). It is an experimental snapshot,
not a production-readiness, matched-solver-accuracy or policy-quality claim.
The additional 2–4× whole-physics target has **not** been achieved.

## Source lineage and scope

The starting points are Isaac Lab `1d8feb82d17dbfab8f0772de56f84deae2cb7974`
and Newton `a2ca01b14a540c50248765641419eef2672bf8a5`, both on the `ooctipus`
forks' `ooctipus/fpgs-opt-20260910` branch. They retain the original handoff
commits `2129e5628596f9908cc307482a01700f45b555d7` and
`31cf87f4694f873a027e41e2ca5e9ad441234456`, respectively.

The measured Newton checkpoint is published as
[`d60528895e154950e0fcf165c953528ce7d1edee`](https://github.com/ooctipus/newton/commit/d60528895e154950e0fcf165c953528ce7d1edee)
on `ooctipus/fpgs-architecture-20260911`, from the `ooctipus` fork. Its committed
tree exactly matches the tested assembled tree. This report's separate Isaac
Lab branch has the same name and still pins starting Newton `a2ca01b14`:
the required inherited-test and P25 sustained-grasp gates are not cleared.
An experimental source checkout is not a validated parent dependency pin.
Original worktrees and historical reports are unchanged. Isaac Lab runtime
and benchmark-harness source are not modified; new tooling belongs to Newton.

## Measurement contract

RTX PRO 6000 Blackwell **Max-Q Workstation Edition** is the primary target;
GB300 is measured concurrently in a separate process, with no two jobs sharing
one selected GPU. Driver 610.43.03, Warp 1.17.0, Torch 2.11.0+cu130,
MuJoCo/MuJoCo Warp 3.12.0, Nsight Systems 2025.6.3, Python 3.12.14.
The installed MuJoCo versions differ from Newton's declared `~=3.11.0`
requirement; that inherited runtime warning is retained, not suppressed.

Unless stated otherwise: 16,384 environments, seed zero, 200 warmup steps,
40 unprofiled synchronized steps and 40 profiled steps, three alternating
A/B rounds. Values are medians of the three capture means. Physics time sums
unique root CUDA-graph durations for the entire batch per environment step;
conditional MJWarp work remains inside those roots. Auxiliary graphs and
synchronized environment wall time are separate. These are not training FPS.
No iteration count, stopping rule, timestep, collision cadence or physics
parameter is tuned to improve a ratio. Backend recipes are different and are
not certified to have equal convergence, contacts or trajectories.

## Validated candidate measurements

These measurements used separately frozen candidate sources before final
composition. They do not yet certify the assembled branch as a whole.

| Newton change | RTX whole-physics speedup | GB300 speedup | Scope |
|---|---:|---:|---|
| Row-parallel RHS + cached dense row state, Ant | 1.142× | 1.142× | Same ordered dense solve |
| Row-parallel RHS + cached dense row state, Humanoid | 1.156× | 1.151× | Same ordered dense solve |
| Register-resident whitening, AnymalD | 1.059× | 1.018× | Same existing parallel formulation |
| Collision pair-participant preparation, Franka | 1.053× | 1.031× | Restricted explicit-pair geometry preparation |

GB300 Humanoid compares the previous best recipe (dense row budgets off) with
the candidate recipe (budgets and row registers on). Its 1.151× is a combined
code/configuration result, not isolated same-flags attribution.

RHS accumulation keeps each row's original DOF addition order and is admitted
only under proven single-articulation response ownership. Cached dense row
state preserves the original GS sweep/reduction order, friction dependencies,
padding and fallback behavior. It requires both
`FEATHER_PGS_DENSE_ROW_BUDGETS=1` and `FEATHER_PGS_DENSE_ROW_REGISTERS=1`.
Register whitening requires `FEATHER_PGS_REGISTER_WHITENING=1` and narrow
shape/formulation/hardware eligibility. Both optimization modes default off;
the guarded RHS change is automatic.

Pair-participant preparation requires `NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP=1`
and an explicit fixed-pair pipeline. Only prepared participants have meaningful
prepared geometry; unsupported pipeline combinations raise. Its preparation
boundary passed exact gates, but contact ordering/counts varied even in
baseline-to-baseline whole-run controls. No whole-trajectory parity claim is
made for this collision candidate.

Native full-buffer, alias-owner, eager/captured and live-state gates support
the individual solver candidates. The assembled branch's 32 focused tests
passed on each GPU. Fresh full-suite default-mode comparison ran 245 baseline
and 261 candidate tests per GPU, with no skips. All 16 added tests passed;
the same two failures and four errors remain, with identical full tracebacks
after normalizing only checkout roots. This is regression evidence, not a
passing suite. The real constructor selection test also fails on the baseline
and passes on the candidate on both GPUs. Cached-mode full discovery also ran
261 tests per GPU with no skips and the same six inherited outcomes; normalized
tracebacks match both the fresh default baseline and the archived matching-flag
baseline. Do not describe the test suite or long-horizon grasp validation as
passing.

### Final assembled-source checks

Fresh paired whole-physics measurements of the assembled Newton source tree
reproduce the individual gains. These are a separate capture set from the
MJWarp comparison below; MJWarp has not been remeasured in this set.

| Task | RTX baseline / assembled ms | Speedup | GB300 baseline / assembled ms | Speedup |
|---|---:|---:|---:|---:|
| Ant | 1.659 / 1.457 | 1.139× | 1.661 / 1.457 | 1.139× |
| Humanoid | 7.122 / 6.173 | 1.154× | 6.662 / 5.787 | 1.151× |
| AnymalD | 9.782 / 9.252 | 1.057× | 10.165 / 9.953 | 1.021× |
| Franka | 6.226 / 5.842 | 1.066× | 5.790 / 5.678 | 1.020× |

Each completed task has twelve audited captures, using the same three-round
protocol and unchanged baseline/candidate flags as the individual comparisons.
The GB300 Humanoid code/configuration and Franka contact-order caveats still
apply. All four assembled-source comparisons are complete. The tested staged Git tree is
`d39304f40b1cff28821e53ac1de18ee4cfbd3ed7`; this is a tree identity, not a
dependency pin; it is the tree of the published Newton checkpoint above.
Synchronized host time remains separately
recorded and variable; these physics gains are not training-throughput claims.

## Fresh MJWarp comparison

| Task | RTX FPGS / MJWarp ms | Ratio | GB300 FPGS / MJWarp ms | Ratio |
|---|---:|---:|---:|---:|
| AnymalD | 9.242 / 38.942 | 4.21× | 9.954 / 43.085 | 4.33× |
| Allegro | 19.200 / 65.193 | 3.40× | 21.183 / 81.658 | 3.85× |
| Franka, warning-heavy | 5.842 / 51.156 | 8.76× | 5.673 / 12.623 | 2.23× |
| Ant | 1.452 / 2.487 | 1.71× | 1.454 / 2.533 | 1.74× |
| Humanoid | 6.164 / 9.230 | 1.50× | 5.786 / 9.273 | 1.60× |
| Cartpole | 0.288 / 0.635 | 2.20× | 0.323 / 0.678 | 2.10× |

All 72 captures passed independent raw-graph/recipe/source/exit/finite-state
audits. FPGS uses the measured candidates above for Ant/Humanoid, AnymalD and
Franka; Allegro and Cartpole retain published Newton `a2ca01b14`. MJWarp uses
that same published baseline for every task. Ratios are not multiplied by
historical RTX 5090 measurements.

Each Franka MJWarp run emitted approximately 725,000 line-search-limit
warnings across the whole run. Their device execution/printing cost is
included. Similar warning totals do not explain the RTX/GB300 asymmetry, and
the 8.76× figure is not evidence of a clean near-10× solver advantage.
RTX Ant synchronized wall medians were 12.507 ms FPGS versus 11.060 ms MJWarp,
despite faster FPGS physics; no training-throughput gain follows from that row.

### SO101 keyboard: 4,096 worlds, workload mismatch

The Newton benchmark alias `keyboard-so101` selects the unchanged
`IsaacContrib-Keyboard-SO101` robot typing task, not keyboard teleoperation.
It adds no solver attributes or task-specific optimization flags. Both arms
use published Newton `a2ca01b14`, with the same three-round sampling protocol.

| Device | FPGS physics ms | MJWarp physics ms | Raw recipe ratio |
|---|---:|---:|---:|
| RTX PRO 6000 | 9.921614 | 77.623700 | 7.82× |
| GB300 | 9.761955 | 31.422919 | 3.22× |

**This is not a comparable-contact-workload solver benchmark.** Both metadata
snapshots in all six FPGS captures show zero contacts and broad-phase pairs;
MJWarp uses Newton collision here and reports approximately 99,000/101,000
contacts before/after. MJWarp also emits approximately 985,000 line-search
warnings per full run. These snapshots do not reveal the complete timed
contact history. The cause is under investigation; finite states do not
establish equal task behavior or solver quality.

A preceding 32-world, 20-warmup smoke had about 546 FPGS contacts and reached
the clamped 192-row capacity. Raw reservation counts alone cannot establish
absence of rejected contacts because allocation rolls back on overflow.
Separate rollback-aware telemetry diagnostics completed on both devices, but
their high-water fields are internally inconsistent: raw/clamped maxima are
zero while the recorded overflow excess reaches 423. The instrumented rollout
also has substantially more contact/row work than the uninstrumented captures.
The cause is not yet isolated. Separately, the unchanged telemetry kernels
passed all 24 fresh-buffer integer/raw-output cases per device, including
eager and repeated captured execution; the isolated kernels do not reproduce
the live inconsistency. These diagnostics cannot establish admission
parity or replace the benchmark timings, and zero high-water fields must not
be interpreted as absence of overflow. No 16K SO101 capture has been completed.
All twelve 4K captures passed independent timing/source
audits: 1,920 unique physics roots, four roots/eight solver substeps per step,
and no auxiliary roots in these captures.

A separate read-only MJWarp diagnostic finds mean Newton iteration counts of
1.224/1.228 before/after on RTX and 1.226/1.234 on GB300, with maximum counts
4/5, not the configured limit of 100. These are last-solve snapshots, not
timed-window averages. Approximately 7,400–7,700 detected contacts have a
nonnegative constraint address, distinct from the approximately 100,000
detected/converted contacts. All 4,096 worlds carry sticky bit 1024; installed
MuJoCo Warp source identifies it as the line-search iteration limit, **not**
row/contact capacity overflow. It does not identify when or how often a world
hit that limit. These findings reinforce the solver-quality caveat rather
than explain the hardware timing asymmetry.

## Why the north star is not yet realized

Fresh five-step node profiles of current Ant/Humanoid show substantial work
in both dynamics/mass/prediction and constraint response/GS. Explicit
copy/clear payload is 171.875/506.875 MiB per batched environment step.
Those are API payload bytes, not total DRAM traffic or proven dead storage.
The largest copy bundle snapshots reusable kinematic/inverse-dynamics state;
safe removal needs a complete last-reader/next-writer and graph-replay proof.

Even crediting all selected summed node work as critical, making response and
GS free gives only about 1.5–1.8× further whole-physics speedup. Node work
overlaps, so this is an optimistic planning counterfactual, not an achievable
bound. Uncovered node-envelope time is only 1.1–2.8% and is not all launch
overhead. Fewer launches alone cannot provide the required improvement.

At today's fixed MJWarp recipe, achieving 10× on RTX requires another 5.84×
FPGS acceleration on Ant, 6.68× on Humanoid, 2.37× on AnymalD and 2.95× on
Allegro. No validated universal path to those numbers exists. The larger
opportunity is coherent world/articulation ownership across topology,
kinematics, mass action, rows and solve, eliminating computation and private
materialization while preserving all physical and persistent-state consumers.

Byte-exact physical outputs do not require preserving demonstrably dead
private intermediate layouts. Conversely, merely omitting private bytes from
an oracle is not a liveness proof. Existing GS and parallel formulations are
not interchangeable: their inherited solver-parity limitation remains.

Rejected experiments are retained in the study record: scalar-world dense
execution, packed world-interleaved C, half-warp dense mapping, naive
response→GS fusion, and cooperative GJK were exact after applicable repairs
but slower. A topology-specialized FK prototype changed floating-point
results; its exact tables-only variant was slower. These failures do not
prove that the architectural north star is impossible, nor that fusion or
world-lane execution will automatically be faster.

A repaired 48-row one-warp parallel solver also passed native and live
byte-exact gates, including restart/early-exit histories, but was 18–36%
slower than current register whitening across the two layouts/devices tested.
All 800 paired timing samples were retained and independently audited. This
variant is rejected, not included in the candidate gains above.

The composite-inertia→factor ownership prototype keeps full composite inertias
on-chip through the unchanged tiled factorization, initially retaining a
diagnostic global mirror. Two actual Ant/Humanoid fixtures per device passed
original-control and full typed/raw eager/captured output gates at both 512
and 16K worlds. Complete serial producer-plus-factor component timings were
slower: original/resident ratios approximately 0.77/0.67 on Ant RTX/GB300 and
0.53/0.47 on Humanoid. All 800 alternating timing samples retained exact
post-checks. This mirror variant is rejected and not integrated.

A separate no-global-Ic variant removes only the proven-private mirror and its
argument. Its eight 16K real boundaries passed the original→mirror→retired-owner
bridge, including all retained typed/raw owners and an unchanged poisoned
retired buffer. It remains slower: original/resident approximately 0.840/0.717
on Ant RTX/GB300 and 0.565/0.496 on Humanoid, with all 800 alternating timing
samples and final exact checks retained. This mapping is also rejected. Sharing
the factor with later stages and an independent shared-wrench torque owner
remain unproven experiments. Component timings do not recreate original stream
overlap and cannot establish whole-physics gains.

## Reproduction and evidence retention

Newton's `tools/fpgs_bench/compare_backends.py` uses an explicitly selected,
unchanged Isaac Lab harness. It supports paired hardware, alternating backend
order, isolated flags, exact source/driver hashes and owned-process cleanup.
Its adjacent README gives runnable commands, including SO101 smoke guidance.
The portable driver has 31 passing CPU tests and has completed the SO101
paired GPU measurements. To reproduce from the published checkpoint, select
the unchanged Lab harness at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`, use a
separate Newton checkout at `d60528895e154950e0fcf165c953528ce7d1edee` for FPGS,
and Newton `a2ca01b14a540c50248765641419eef2672bf8a5` for MJWarp. Pass those
explicit directories using the driver's `--isaaclab`, `--fpgs` and `--mjwarp`
arguments; the driver checks actual imports without changing Lab's dependency
pin or synchronizing its environment. The runtime versions above must also
match. This produces fresh measurements, not byte-identical stochastic runs.

Use the adjacent driver README's paired 16K/200/40/40 three-round command.
Enable dense row budgets and row registers on both GPUs for Ant/Humanoid;
enable `FEATHER_PGS_REGISTER_WHITENING=1` on both for AnymalD; enable
`NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP=1` on both for Franka. Run these task groups
separately so flags do not leak across recipes. Allegro and Cartpole receive
no additional flags. For SO101 explicitly select `--task keyboard-so101
--num-envs 4096` and no additional flags, retaining the workload caveat above.

The assembled checkpoint has 32 focused GPU tests per device, full default and
cached 261-test discovery with the six unchanged inherited outcomes, and 48
independently audited whole-comparison captures. Large captures and negative
prototype evidence remain local; they are not bundled into the repositories.
