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
- Each task uses 16,384 environments, seed 0, 200 warmup steps, a 40-step timing batch, and three profiled steps.
  The handoff recipes are unchanged. Every environment step has eight solver substeps and four physics-graph
  launches. A graph time covers the entire environment batch, not one individual environment.
- `analyze_structure.py` independently reconstructed all 32 saved graph spans and busy unions from read-only
  SQLite traces, including graph memcpy/memset activity, and matched the existing analysis within 1e-6 µs.
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
| **Observed graph span** | **1227.1** | **1299.5** | **2409.9** | **2653.8** |
| Graph idle gaps | 46.1 | 56.2 | 66.9 | 77.8 |

Median graph times per environment step: RTX AnymalD **9.817 ms**, Allegro **19.279 ms**; GB300 AnymalD
**10.396 ms**, Allegro **21.231 ms**. These are not end-to-end training times or new MJWarp comparisons.

The graph contains 39 kernel nodes/substep on AnymalD and 57.5 on Allegro. Removing every measured idle gap
would save only 3.8%/2.8% on RTX and 4.3%/2.9% on GB300. Fusion must remove data movement or computational
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

## Earlier probes and measurement hazards

The synchronization-only probe preserves bitwise outputs in synthetic real-kernel replay and live same-input
task checks. It shows a modest kernel improvement but no convincing RTX Allegro whole-step win. The row-cache
probe is likewise not a validated whole-step improvement. These are separate from the structural effort.

Allegro's legacy fallback usually takes about 10 µs/call but occasionally takes roughly 800–860 µs. Beginning/end
metadata do not identify the responsible substep. In the RTX row-cache probe, approximately 248 µs/step saved in
parallel kernels was offset by 265 µs/step of extra fallback work. Investigate the actual row counts and ownership
conditions per substep. Raising the parallel row threshold would change numerical ownership and is not a free fix.

A previous register-oriented GJK experiment removed generated PTX local arrays but improved GB300 GJK by only
about 6.7% and failed CUDA wrapper parity, including a near-contact classification. It is not included. Fresh Nsight
Compute hardware counters are unavailable because of GPU performance-counter permissions; archived spill counters
are not fresh measurements. Nsight Systems timings and compiler resource reports remain available.

The inherited `test_dense_warmstart_preserves_projected_closure` failure reproduces on both baseline and the
synchronization candidate (`cache_peak` approximately 1.312e-5 versus a required value above 1e-4). It is not a new
optimization regression, and the affected test suite must not be described as entirely passing.

## Reproduction

See the [profiling harness](../../scripts/benchmarks/fpgs_profile/README.md) for dual-GPU comparisons and the
same-input parity checker. The baseline/synchronization captures remain local in
`outputs/fpgs_profile/compare_gpus__z8na145`; the one-round cache probe is in `compare_gpus_4nfist89`.
Manifests retain exact hardware, commands, source hashes, and recipes. Raw captures are not committed.

```bash
uv run --no-sync python scripts/benchmarks/fpgs_profile/analyze_structure.py \
    outputs/fpgs_profile/compare_gpus__z8na145 \
    outputs/fpgs_profile/compare_gpus_4nfist89 \
    --output outputs/fpgs_profile/structural_audit_new.json
```

The analyzer performs no GPU work, refuses to overwrite its output, and reports separate task/device budgets,
kernel-duration distributions, direct SQLite checks, and optimistic scope-level speedup estimates.
