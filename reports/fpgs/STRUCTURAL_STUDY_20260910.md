# FPGS structural optimization study — 2026-09-10

This study targets an additional 2–4× improvement over the handed-off FPGS implementation, not its existing
speedup over MJWarp. No such improvement has been established. The evidence favors broad changes to execution
and data flow; it does not justify a large claim from sweep synchronization or launch fusion alone.

The original [handoff](HANDOFF_fpgs_20260911.md), [design history](fpgs_solver_shape_and_spirit_20260909.md), and
[validated snapshot](README.md) remain historical records. Their older RTX 5090 timings are not measurements of
the two devices in this study. In particular, the archived Franka 5.12× comparison was not reproduced by the
handoff validation, which reported approximately 3.7–3.9×.

## Contract and evidence

- Reference: Isaac Lab `2129e5628596f9908cc307482a01700f45b555d7`, Newton
  `31cf87f4694f873a027e41e2ca5e9ad441234456`, both from the `ooctipus` forks.
- Primary device: NVIDIA RTX PRO 6000 Blackwell **Max-Q Workstation Edition**, GPU 0. Secondary: NVIDIA GB300,
  GPU 1. Driver 610.43.03; Warp 1.17.0; Torch 2.11.0+cu130; Nsight Systems 2025.6.3.
- Paired runs execute concurrently on the two GPUs, with separate results. The baseline/synchronization study
  has three alternating A/B rounds; the extra row-cache probe has one round and is not a repeatability result.
- The initial AnymalD/Allegro comparisons use 16,384 environments, seed 0, 200 warmup steps, a 40-step timing batch,
  and three profiled steps. Their handoff recipes are unchanged, with eight solver substeps and four physics-graph
  launches per environment step. A graph time covers the entire environment batch, not one individual environment.
- The initial independent SQLite reconstruction matched all 32 earlier reports within 1e-6 µs, but a later audit
  found both analyzers shared a blind spot: graph memory nodes can have `graphNodeId` without a `graphId` column.
  Both analyzers were corrected and all 32 captures rechecked without modifying the originals. Graph spans and
  kernel sums were unchanged, but memory operations previously classified as idle added 182–284 µs/step to graph
  busy unions. The table below includes that correction. Agreement between analyzers was not sufficient proof
  of complete graph accounting. Direct whole-graph tracing was separately validated on Ant/Humanoid on both GPUs.
- Stage values below come from the actual median-graph baseline capture for each task/device. They are sums of
  kernel durations, not causal critical-path allocations: overlapping work cannot simply be added as savings.
  The projection kernels already contain preparation and response work; their whole duration is not sweep time.

Keep substeps, collision cadence, contacts, row ordering/ownership, projection, iteration/stopping rules, and
physics parameters fixed. Do not skip Jacobian clearing while partial row writers still depend on it. Compare
actual per-substep kernel outputs on identical inputs; contact counts and finiteness alone do not establish parity.
The inherited parallel projection and legacy Gauss–Seidel algorithms have different fixed points. This study does
not resolve or conceal that difference. A model-changing algorithm needs its own accuracy study and acceptance.

## Measured budget

Baseline microseconds per solver substep, averaged across collision cadence:

| Work | RTX AnymalD | GB300 AnymalD | RTX Allegro | GB300 Allegro |
|---|---:|---:|---:|---:|
| Parallel projection, including fused rows/response | 514.8 | 550.0 | 619.7 | 685.3 |
| External rows/setup | 141.9 | 148.8 | 489.4 | 416.6 |
| Remaining response | 28.0 | 35.1 | 62.3 | 71.4 |
| Legacy fallback | 10.3 | 9.8 | 42.8 | 78.7 |
| Collision | 127.6 | 143.2 | 733.2 | 984.1 |
| Dynamics, mass/factor, predictor, publication, integration | 293.3 | 291.1 | 427.3 | 359.6 |
| Sensors/contact forces | 63.3 | 65.3 | 0.0 | 0.0 |
| Other device work | 12.9 | 10.3 | 21.7 | 19.6 |
| Graph memory operations, summed durations | 24.8 | 27.6 | 40.5 | 39.8 |
| **Observed graph span** | **1227.1** | **1299.5** | **2409.9** | **2653.8** |
| Graph idle gaps, corrected busy union | 22.8 | 29.5 | 35.5 | 42.5 |

Median graph times per environment step: RTX AnymalD **9.817 ms**, Allegro **19.279 ms**; GB300 AnymalD
**10.396 ms**, Allegro **21.231 ms**. These are not end-to-end training times or new MJWarp comparisons.

The graph contains 39 kernel plus 20.5 memory nodes/substep on AnymalD, and 57.5 kernel plus 27.5 memory nodes
on Allegro. Removing every measured idle gap would save only 1.9%/1.5% on RTX and 2.3%/1.6% on GB300.
Fusion must remove data movement or computational
work to matter substantially; launch count alone is not a sufficient performance model.

Allegro's external-row budget is distributed: Jacobian population, contact metadata, velocity-limit pre-scaling,
active Jacobian clearing, velocity-limit slots, and contact slots. Collapsing only the contact prelude does not
remove this whole group. On GB300, GJK/MPR take a larger share while hand dynamics and published FK are faster;
there is no justification for pooling the hardware results.

## What a 2–4× result requires

For a fraction `p` accelerated by `k`, the fixed-other-work estimate is `1 / (1 - p + p/k)`. These fractions use
summed stage work divided by graph span and are optimistic because some work overlaps.

| Scope | Device | Fraction | Scope speedup needed for 2× graph | For 4× graph |
|---|---|---:|---:|---:|
| AnymalD: all constraints plus dynamics/publication/integration | RTX | 80.5% | 2.64× | 14.55× |
| Same | GB300 | 79.6% | 2.69× | 17.19× |
| Allegro: all constraints plus collision | RTX | 80.8% | 2.62× | 13.91× |
| Same | GB300 | 84.3% | 2.46× | 9.10× |

Even infinitely fast parallel-projection kernels cap the RTX graph improvement at approximately 1.72× on
AnymalD and 1.35× on Allegro. Infinitely fast collision caps Allegro at 1.44× RTX/1.59× GB300. Neither isolated
direction can deliver 2×. A 4× whole-graph proposal must also address substantial portions of the remaining work.

## Structural experiments and gates

1. **Subgroup-owned world solve.** Evaluate 16 and 8 lanes per world, multiple rows per lane, and row data kept
   in registers. This differs from the previously unsuccessful experiment that merely packed existing 32-lane
   worlds into a block. First test complete projected iterations on staged real inputs, including friction pairs,
   limits, momentum restart, convergence voting, and velocity reconstruction. Measure the cost of exchanging
   register-held row data; removing shared loads does not make those exchanges free. Check register/shared
   allocation and spills, numerical results, and 16K-world throughput before attempting pipeline integration.
2. **World-owned row/response flow.** Establish complete-row ownership across contact, drive, position-limit,
   velocity-limit, and bilateral producers, including two-articulation contacts and fallback worlds. Then test
   compact world-local records that remove redundant full Jacobian/metadata round trips and clearing. Simply
   enabling the existing in-kernel row builder or reviving the rejected one-thread-per-row builder is not a new
   structural hypothesis. Count the work and bytes that actually disappear.
3. **Topology-specialized dynamics integration.** Reduce runtime topology lookup and repeated state/factor
   materialization, then compose with the winning row/solve mapping. A larger fused kernel must earn back its
   increased live state and occupancy cost. The old estimate of 1.5–2 ms from dynamics fusion alone was corrected
   in the handoff to roughly 0.3–0.5 ms; do not reuse the withdrawn estimate.
4. **Cooperative convex support on Allegro.** If collision remains limiting, prototype 8/16/32-lane support
   searches with controller-owned simplex state. Preserve each vertex's arithmetic and the original argmax tie
   rule (earliest vertex), sentinel behavior, and full contact output. Assess actual hull sizes and support cost
   before replacing the production narrow phase. Geometry bucketing also needs preserved solver-visible contact
   ordering. Contact caching, decimation, hull simplification, or changed termination are separate math studies.
5. **Integration and release.** Require same-input kernel validation, complete-pipeline state checks, unchanged
   contact/row contracts, and repeated dual-GPU graph A/B. Report solver-stage, graph, and wall results separately.
   Keep unsupported shapes on their existing path. Preserve both original worktrees and publish Newton to the
   `ooctipus` fork before updating Isaac Lab's exact dependency pin.

These are research gates, not predicted speedups. A stage-only result must be discounted by its measured share
of the full graph. Do not integrate a large rewrite on the strength of a synthetic-only microbenchmark.

### Subgroup experiment: rejected

A staged corpus contains 19,852 actual post-warmup AnymalD worlds admitted to the 32-row tier. Both devices
replayed the same RTX-captured inputs. The staging diagnostic reproduced the original full kernel's velocity,
impulses, row metadata, response, and diagonal buffers bitwise. The isolated 32-lane reference then reproduced
exported final row state, velocity, executed iterations, and restart summaries exactly on both devices.

Median sweep times in microseconds across seven alternating rounds of 100 identical-state graph replays:

| Mapping | RTX | GB300 |
|---|---:|---:|
| Original 32-lane reference | 128.15 | 135.10 |
| 16 lanes, tree reduction | 130.61 | 129.58 |
| 8 lanes, tree reduction | 136.06 | 152.64 |
| 16 lanes, ordered accumulation | 308.30 | 334.23 |
| 8 lanes, ordered accumulation | 195.36 | 245.37 |

These are isolated sweeps, not whole production-kernel or graph times. Including their common reconstruction
kernel does not change the conclusion: no candidate wins on RTX. Tree variants change restart histories, and
the 8-lane variant changes one stopping decision. Ordered variants still change one stopping decision despite
very small typical velocity differences; their remaining arithmetic-boundary discrepancy was not resolved.
No variant is eligible for integration. All compiled without spills, but register counts increased substantially.

### Matched phase split: insufficient gain

A second bounded experiment retained the original 32-lane arithmetic and separated preparation, sweep, and
reconstruction to shorten resource lifetimes. All three stages were compared against the complete original
kernel on identical full-capacity live inputs, including inactive worlds and all new global traffic. Four
post-warmup launches per device passed full-buffer bitwise checks; two were timed with 100 alternating A/B
pairs each. All seven mutable original output buffers were restored before every replay, outside the timed
compute graph. Each device used its own live trajectory, so A/B was matched within a GPU, not across hardware.

| Full 16K-world AM32 launch | RTX | GB300 |
|---|---:|---:|
| Original kernel, medians for two live states | 256.0 / 256.8 µs | 252.4 / 252.4 µs |
| Complete three-stage replacement | 250.3 / 251.3 µs | 234.0 / 236.0 µs |
| Matched speedup | 1.023× / 1.022× | 1.079× / 1.069× |

This is not a whole-physics graph result. The small RTX gain is insufficient to justify integration of three
kernels and new staging buffers toward the structural target. Exporting and rereading `Z` alone adds at least
54.7 MB per launch at the earlier observed active counts. At 512 environments the split was slower on both GPUs.
The compiler resource improvement was real, but faster isolated sweeps did not translate into a large gain
for the full preparation/solve/reconstruction pipeline. This direction is set aside without production changes.

### Dense working-set ownership and fused response

The broader handoff survey found another structural opportunity on Ant and Humanoid: their original dense C64
Gauss–Seidel route reserves a full-capacity shared working set even when a world has at most 32 active rows.
The experiment partitions whole worlds into two launches: a 32-row local-storage kernel and the original
64-row-capacity fallback. World identities, global C/vector strides, row ordering, the 32-lane reduction tree,
projection, iteration counts, and output padding remain unchanged. No queue or row reordering is introduced.
Worlds whose friction metadata reads outside the active prefix retain the full-capacity path. The implementation
mirrors the original denominator branch, including NaN behavior, before deciding whether reduced storage is safe.
Full-capacity fallback preserves baseline behavior; it is not a validator for truly out-of-capacity row counts
or parent indices.

On RTX, static shared allocation falls from 9,984 to 3,008 bytes for the small-world kernel, with registers
38 → 32 per thread; GB300 registers fall from 32 to 29. The fallback retains full storage. A three-tier prototype
also reserved 16 rows, but both 16- and 32-row versions hit RTX's block residency limit. Removing that unnecessary
launch substantially improved the result. The layout is an explicit, default-off
`FEATHER_PGS_DENSE_ROW_BUDGETS=1` experiment, limited to CUDA, C64, and the existing `tiled_row` route. Small
batches and hardware/task combinations must be measured before enabling it.

The fused dense response has a separate ownership bottleneck: one thread publishes all diagonal entries after
the tile matrix operations. Assigning each diagonal entry to a distinct thread preserves the exact matrix solve,
matrix product, and CFM addition but shortens register lifetimes. For Ant's D14 shape, compiled registers fall
from 166 → 95 on RTX and 168 → 96 on GB300; for Humanoid's D27 shape, 254 → 96 and 248 → 96. All compile without
spills. These are compiler reports, not fresh hardware-counter measurements. The original serial publication is
retained when a non-divisible constraint capacity selects a tile wider than the thread block.

Same-input full-capacity response replay, 100 alternating A/B pairs at each of two live checkpoints:

| Fused response kernel | RTX | GB300 |
|---|---:|---:|
| Ant speedup | 1.495× / 1.498× | 1.620× / 1.589× |
| Humanoid speedup | 1.525× / 1.523× | 1.897× / 1.891× |

All typed buffers and overlapping raw allocations matched bitwise, including inactive bytes, for eager,
non-default-stream, and CUDA graph replay. Mutable outputs were reset before every replay outside the timed
compute graph. Each hardware A/B used its own baseline trajectory. These are stage timings, not whole-graph
speedups. Production tests additionally cover every dense active count, warm impulses, delayed friction,
cross-boundary metadata, nonpositive/NaN diagonals, arbitrary world mappings, and response capacities 5 and 65.

The final two-tier production solve passed all four live checkpoints per task/device (262,144 world inputs,
including inactive worlds). Its matched isolated speedups were Ant **2.02–2.04× RTX / 1.26–1.27× GB300**, and
Humanoid **1.37–1.39× RTX / 0.974–1.006× GB300**. Keep the dense layout disabled on GB300 Humanoid.
Actual production response kernels also passed all eight full replay fixtures. The eight new tests passed on
both devices and under RTX Compute Sanitizer synchronization checking, with zero reported synchronization errors.
The four required legacy test modules reproduced 16 passes and the same one inherited failure on baseline and
production on both GPUs. No claim of a completely passing suite or long-horizon grasp validation is made.

## Combined whole-graph validation

Candidate Newton: `a2ca01b14a540c50248765641419eef2672bf8a5`, published to
[`ooctipus/newton`, branch `ooctipus/fpgs-opt-20260910`](https://github.com/ooctipus/newton/tree/ooctipus/fpgs-opt-20260910).
This is a direct descendant of the handoff. The optimization branch excludes the synchronization-only,
row-cache, subgroup, phase-split, and collision prototypes.

Three alternating A/B rounds, original 16K-world task recipes, direct whole-graph software tracing.
Values below are median microseconds per environment step; speedup is **handoff / candidate**, not MJWarp / FPGS.

| Task | RTX baseline → candidate | RTX speedup | GB300 baseline → candidate | GB300 speedup |
|---|---:|---:|---:|---:|
| Ant | 2254.1 → 1654.8 | **1.362×** | 1930.5 → 1656.9 | **1.165×** |
| Humanoid | 8501.8 → 7097.3 | **1.198×** | 7581.9 → 6670.5 | **1.137×** |
| Franka | 6156.0 → 6139.3 | 1.003× | 5751.2 → 5865.0 | 0.981× |
| AnymalD | 9733.5 → 9733.1 | 1.000× | 10135.3 → 10116.1 | 1.002× |
| Allegro | 18866.1 → 18840.7 | 1.001× | 21190.9 → 20587.5 | 1.029× |

Ant enables the dense storage layout on both GPUs. Humanoid enables it only on RTX; GB300 keeps its original
dense solve and benefits from the response change. The other recipes retain their solver paths; the RTX flag
does not make them eligible for the C64 dense layout. Ant/Humanoid graph spreads are at most 1.15% across rounds.
The RTX regression tasks are effectively flat. Do not attribute GB300 Allegro's short-window difference to a
new solver improvement: fallback participation and independent contact workloads vary.

The approximately 2% GB300 Franka slowdown in the original short graph windows is retained as an unresolved
observation, not dismissed as noise. A separate 40-step **node-mode diagnostic** measured RTX 6413.9 → 6412.3 µs
and GB300 6047.0 → 6051.2 µs. All 171 full recorded kernel names, all 19,520 graph-kernel launch descriptors,
all 9,440 graph-memory descriptors, and per-stream operation order matched within each hardware A/B pair.
Neither modified solver kernel executed on Franka. This longer, differently instrumented capture did not reproduce
the gap, but it does not causally resolve the original observation; the trajectories were not identical.

Unprofiled wall-step medians are a separate result:

| Task | RTX baseline → candidate | GB300 baseline → candidate |
|---|---:|---:|
| Ant | 7.246 → 6.987 ms | 7.041 → 6.990 ms |
| Humanoid | 16.793 → 15.654 ms | 16.118 → 14.761 ms |

Ant wall samples have large reset/event outliers: RTX baseline spans 7.189–18.636 ms and candidate
6.303–12.301 ms; GB300 candidate spans 6.885–17.769 ms. These samples do not establish a large application
throughput gain. Humanoid wall ranges are 16.678–16.936 → 15.654–15.660 ms on RTX and
15.711–16.415 → 14.295–15.093 ms on GB300. No RL training throughput or policy-quality claim is made.

All 60 repeated-comparison captures passed independent source/configuration checks, finite-state checks, and
direct SQLite timing reconstruction. All 576 graph records were uniquely contained in `sim.step`, so none of
these five task results include the auxiliary sensor-graph issue found subsequently on G1. Kernel-level parity
comes from the separate same-input replay gates, not from comparing contact counts between independent rollouts.

The final G1/Kuka/Cartpole smoke covered both revisions on both GPUs, with the same 16K-world recipes and one
A/B round. All 12 captures passed finite-state, configuration/source-freeze, and independent direct SQLite
checks (120 physics records). These are coverage checks, not repeated performance measurements:

| Task | RTX physics µs/step, baseline → candidate | GB300 physics µs/step, baseline → candidate |
|---|---:|---:|
| G1 | 40961.3 → 41076.0 | 67864.7 → 67928.0 |
| Kuka / Allegro | 17826.0 → 17968.5 | 16953.6 → 16921.0 |
| Cartpole | 290.6 → 289.9 | 321.9 → 324.7 |

G1 launches four physics graphs and one separate observation/sensor graph per step. The old global Warp-launch
wrapper mislabeled the sensor graph as physics; the initial smoke failed closed rather than accepting that
mixed scope. The corrected instrumentation identifies the Newton step graph by object identity, and the analyzer
requires physics launches to be contained in `sim.step`. Auxiliary graph time is reported separately: G1 RTX
1037.9 → 1036.0 µs/step and GB300 1053.5 → 1052.9 µs/step. Kuka and Cartpole have no auxiliary graph launches.
Node-mode structural analysis rejects mixed physics/auxiliary captures; use graph mode for G1. Nine portable
tests cover instrumentation, graph ownership, malformed correlations, and backward-compatible physics-only
captures. The corrected analyzer reproduced all 60 earlier accepted captures and the four Franka diagnostics
without changing their physics timings.

The additional **2–4× whole-physics target is not achieved**. The largest demonstrated isolated-stage gain is
the approximately 2.04× Ant dense solve. Further work should return to complete world-owned row/response flow
and topology-specialized dynamics, with the measured scope-level ceilings above; scattered micro-tuning or
switching legacy fallback worlds to a different projection algorithm cannot substantiate the target.

## Earlier probes and measurement hazards

The synchronization-only probe preserves bitwise outputs in synthetic real-kernel replay and live same-input
task checks. It shows a modest kernel improvement but no convincing RTX Allegro whole-step win. The row-cache
probe is likewise not a validated whole-step improvement. These are separate from the structural effort.

Allegro's legacy fallback usually takes about 10 µs/call in the profiles but occasionally takes roughly
800–860 µs. A separate read-only observer checked all 320 solver substeps over 40 post-warmup environment steps
on each GPU. All 19 RTX and 14 GB300 slow launches admitted exactly one world above the 128-row parallel ownership
limit; every other world returned early. Active row counts reached 141/152, although endpoint checks never
exceeded 123/122. Matrix-free rows, deferred response, and capacity overflow were absent. Every launch above
300 µs had such a world; no admitted world lacked a spike. These event-instrumented eager runs diagnose the
cause; they are not replacement benchmark timings or identical-trajectory hardware comparisons.

In the RTX row-cache probe, approximately 248 µs/step saved in parallel kernels was offset by 265 µs/step of extra
fallback work. Short profile windows can therefore mislead comparisons. Compacting empty worlds does not remove
the single-world sequential tail. Raising the parallel row threshold would change numerical ownership and is
not a free fix; an exact fallback optimization needs fixtures containing these 129+ row worlds.

A previous register-oriented GJK experiment removed generated PTX local arrays but improved GB300 GJK by only
about 6.7% and failed CUDA wrapper parity, including a near-contact classification. It is not included. Fresh Nsight
Compute hardware counters are unavailable because of GPU performance-counter permissions; archived spill counters
are not fresh measurements. Nsight Systems timings and compiler resource reports remain available.

A separate full 8-lane cooperative GJK attempt also failed the exact-output gate and was closed without timing
or integration. The original replay first reproduced all captured outputs exactly on each GPU. The candidate
added two accepted RTX pairs (88,000 → 88,002) and changed witness fields on both devices, even though GB300's
accepted count stayed 88,062. Replicating the controller across lanes also increased resources: every thread kept
a 352-byte local stack, with roughly unchanged or higher registers per thread. Matching only the support argmax
or accepted count is insufficient to establish complete narrow-phase equivalence.

The inherited `test_dense_warmstart_preserves_projected_closure` failure reproduces on baseline, the
synchronization candidate, and the final production candidate (`cache_peak` approximately 1.312e-5 versus a
required value above 1e-4). It is not a new
optimization regression, and the affected test suite must not be described as entirely passing.

## Reproduction

The Lab dependency pin and lockfile select the published `ooctipus/newton` commit
`a2ca01b14a540c50248765641419eef2672bf8a5`. A fresh locked installation reproduced the validated solver source
hash and passed all eight new tests on both GPUs. To compare against the original handoff, create separate clean
Newton checkouts at that commit and `31cf87f4694f873a027e41e2ca5e9ad441234456`; use their absolute paths below.
GPU 0 is RTX PRO 6000 and GPU 1 is GB300 on this machine. Each command launches both devices concurrently,
alternates revision order, and records three rounds with the original task recipes:

```bash
uv run --no-sync python scripts/benchmarks/fpgs_profile/compare_gpus.py \
    --newton baseline=/path/to/newton-handoff --newton optimized=/path/to/newton-optimized \
    --gpus 0 1 --task ant --repeats 3 --trace-mode graph \
    --gpu-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1 --gpu-env 1:FEATHER_PGS_DENSE_ROW_BUDGETS=1

uv run --no-sync python scripts/benchmarks/fpgs_profile/compare_gpus.py \
    --newton baseline=/path/to/newton-handoff --newton optimized=/path/to/newton-optimized \
    --gpus 0 1 --task humanoid --repeats 3 --trace-mode graph \
    --gpu-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1
```

The final repeated-comparison artifacts remain local under `outputs/fpgs_profile/`:

| Directory | Contents |
|---|---|
| `compare_gpus_zopdsb8s` | Ant, three rounds, both GPUs |
| `compare_gpus_bkzeh7e9` | Humanoid, three rounds, both GPUs |
| `compare_gpus_701s3g7z` | Franka, AnymalD, Allegro, three rounds, both GPUs |
| `compare_gpus_0_pn0j7e` | Franka, 40-step node-mode diagnostic, one round |
| `compare_gpus_r0rvbs8o` | G1, Kuka, Cartpole, final graph-mode smoke, one round, both GPUs |

The final smoke used the same command shape with `--task g1 --task kuka --task cartpole --repeats 1`, graph mode,
and only `--gpu-env 0:FEATHER_PGS_DENSE_ROW_BUDGETS=1`. Its Lab profiler commit was
`c3451215357d82db0c82177bc5428598d48a9c51`; only the dependency pins, release fragment, and reports were dirty.
The final documentation update does not change runtime code. Main five-task comparisons used profiler commit
`076f9453ed12e9e57cb2682cca2ecd4c2aa33e11`, before the separately tested auxiliary-graph labeling correction.

Large same-input fixtures and independent replay/audit tooling remain local in `/tmp/fpgs_opt_validation/`
and `/tmp/fpgs-response-replay-AX2fTV/`. They are not prerequisites for the maintained synthetic regression tests,
but the original live-fixture parity results cannot be reproduced from this branch alone without recapturing
those inputs. No large reference captures or private experiment worktrees are included in either commit.

See the [profiling harness](../../scripts/benchmarks/fpgs_profile/README.md) for dual-GPU comparisons and the
same-input parity checker. The baseline/synchronization captures remain local in
`outputs/fpgs_profile/compare_gpus__z8na145`; the one-round cache probe is in `compare_gpus_4nfist89`.
Manifests retain exact hardware, commands, source hashes, and recipes. Raw captures are not committed.

```bash
uv run --no-sync python scripts/benchmarks/fpgs_profile/analyze_structure.py \
    outputs/fpgs_profile/compare_gpus__z8na145 \
    outputs/fpgs_profile/compare_gpus_4nfist89 \
    --analysis-root /tmp/fpgs-memory-audit-dxek9aip \
    --output outputs/fpgs_profile/structural_audit_new.json
```

The analyzer performs no GPU work, refuses to overwrite its output, and reports separate task/device budgets,
kernel-duration distributions, direct SQLite checks, and optimistic scope-level speedup estimates.
The local `--analysis-root` contains corrected reports in a separate tree; the original captures remain unchanged.
To rebuild that tree elsewhere, follow the archived-analysis instructions in the profiling harness README.
