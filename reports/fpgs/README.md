# FPGS verification and cross-machine handoff — 2026-09-10

Start here. This report supersedes the numerical claims in the archived handoff below.
This is an experimental development snapshot, not a claim of production readiness or physics parity.

## Checkouts and pins

Both repositories use branch `zhengyuz/fpgs-handoff-20260910`:

- Isaac Lab: https://github.com/ooctipus/IsaacLab/tree/zhengyuz/fpgs-handoff-20260910
- Newton: https://github.com/ooctipus/newton/tree/zhengyuz/fpgs-handoff-20260910

The Isaac Lab commit containing this report is the Isaac Lab pin (`git rev-parse HEAD`).
The exact Newton commit is `31cf87f4694f873a027e41e2ca5e9ad441234456`, pinned in the
root `pyproject.toml` override and `uv.lock`.
The branch retains the existing dependency pins, plus the experimental source changes.
Original worktrees and branches were left untouched.

```bash
git clone --branch zhengyuz/fpgs-handoff-20260910 https://github.com/ooctipus/IsaacLab.git IsaacLab-fpgs
cd IsaacLab-fpgs
unset PYTHONPATH  # Do not accidentally import a different Newton checkout.
uvx --from uv==0.11.26 uv sync --locked
git rev-parse HEAD
uv run python -c 'import newton; print(newton.__file__)'
```

For Newton development, also clone the Newton branch above and set
`PYTHONPATH=/absolute/path/to/your/newton-checkout` when running Isaac Lab.
Keep that checkout at the pinned commit until you intentionally start new experiments.

## Fresh measurements

RTX 5090, physical GPU 1 (170 SMs), driver 580.173.02; 16,384 environments.
These are **physics CUDA-graph microseconds per environment step**, not FPS,
not microseconds per solver substep, and not training throughput. Lower is better.
The speedup is MJWarp graph time divided by FPGS graph time.

| Task | Previously quoted FPGS | Fresh FPGS | Fresh MJWarp | Fresh speedup |
|---|---:|---:|---:|---:|
| AnymalD | 8,990 | 8,899 | 34,952 | 3.93× |
| Allegro | 17,605 | 17,497 | 59,564 | 3.40× |
| Kuka / Allegro | 17,634 | 17,167 | 69,424 | 4.04× |
| Franka | 5,854 | 5,763 | 21,120 | 3.66× |
| G1 | 38,931 | 39,015 | 50,002 | 1.28× |
| Cartpole | 248 | 249 | 542 | 2.18× |
| Ant | 2,072 | 2,037 | 2,254 | 1.11× |
| Humanoid | 7,920 | 7,855 | 8,513 | 1.08× |

All FPGS times are within 2.7% of the quoted values. States sampled before and after
the benchmark were finite for both backends. This is a short performance check,
not a long-horizon stability or policy-quality evaluation.

Franka's earlier 5.12× is **not reproduced**. Two additional MJWarp captures gave
22,575 and 21,586 µs, yielding 3.66–3.92× against the fresh FPGS measurement.
The original baseline was 29,984 µs. An independent trace audit localized the change
to MJWarp line-search kernels: 19,267 → 10,365 µs per environment step, while other
graph kernels stayed approximately constant. Both runs emitted over 631,000
GPU-side line-search-limit warnings. Convergence-dependent work and GPU printf cost
are included; the captures do not isolate their individual contributions.

Full-step FPS is a separate metric, retained in [results.json](results.json).
For example, AnymalD's full-step ratio was 2.90×, while Ant's was 0.46× despite
its 1.11× physics-graph ratio. No RL training speedup was measured.

## Reproduce

Install Nsight Systems separately (the verification used 2026.1.1).
With an otherwise idle GPU, from the Isaac Lab root:

```bash
GPU=1 bash scripts/benchmarks/fpgs_profile/reproduce.sh
```

This runs all eight FPGS/MJWarp pairs, with a fresh output directory by default.
It clears inherited experimental solver flags and refuses to overwrite previous captures.
It checks GPU compute occupancy before each run; avoid starting competing jobs during a run.
Raw Nsight reports can contain process environment data: keep them private and
review/redact them before sharing.

Measurement contract: seed 0, 200 warmup steps, one 40-step synchronized FPS batch,
then three profiled environment steps; random actions, CUDA graphs enabled.
Nsight flags: `--capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx
--cuda-graph-trace=node --cuda-event-trace=false`.
Graph span sums the first-to-last device operation of each physics graph launch.
For AnymalD there are four launches per environment step and two solver substeps
per launch. The reported environment-step time therefore covers eight solver substeps.

Exact optimized recipe:

| Task | Extra environment variables | Solver attributes |
|---|---|---|
| All | `FEATHER_PGS_GROUP_LANES=16 FEATHER_PGS_ROWS_MASKED=1 NEWTON_NARROW_PHASE_THREADS_X=4` | Task defaults |
| AnymalD | `FEATHER_PGS_INK=1 FEATHER_PGS_MF_EXACT_ROWSUM=1 FEATHER_PGS_WORLD_ROWS=1` | `grouped_dynamics=True mf_gs_parallel_rows=48 mf_gs_parallel_matrix_free=True lazy_kinematics=True` |
| Allegro | `FEATHER_PGS_INK=1 FEATHER_PGS_TIER_BLOCKS=16384` | `grouped_dynamics=True mf_gs_parallel_rows=128 mf_gs_parallel_matrix_free=True lazy_kinematics=True` |
| G1 | None | `grouped_dynamics=True` |
| Kuka, Franka, Cartpole, Ant, Humanoid | None | Legacy task paths |

Normal training remains:

```bash
CUDA_VISIBLE_DEVICES=1 uv run isaaclab train --task=Isaac-Velocity-Flat-AnymalD physics=feather_pgs --num_envs=16384
```

The fpgs-dev launcher equivalent is
`./isaaclab train --task=Isaac-Velocity-Flat-AnymalD physics=feather_pgs --num_envs=16384`.
**That plain training command does not enable the optimized AnymalD/Allegro recipe**
and does not measure the table's graph-time metric. Use the benchmark command above
for the reported numbers.

## Physics and section 5.1

No task physics scalar settings were intentionally changed for verification.
Recorded historical fields matched the currently resolved configurations; historical
artifacts did not preserve every collision/CFM/gap field, so complete historical
parameter identity cannot be established. Current resolved configurations are saved
in [resolved_physics.json](resolved_physics.json).

However, **similar contact statistics do not imply unchanged solver physics**.
The opt-in parallel projection uses a different update/projection metric from
legacy row Gauss–Seidel. Its eligible worlds use up to 24 parallel sweeps with
Nesterov acceleration and tolerance `1e-4`, not simply the task's legacy 8/12
PGS iterations executed faster.

Independent CPU recalculation of stored 16,384-world captures confirmed:

| Parallel 400 vs legacy 400, relative velocity error | Median | p90 |
|---|---:|---:|
| AnymalD | 1.766664% | 5.531140% |
| Allegro | 0.332075% | 5.155287% |

These corroborate archived handoff section 5.1, including distinct converged results.
The archived truth script's “24 vs 24 (today)” label is not the actual benchmark
contract: captured metadata used legacy `iterations=8` for AnymalD and `12` for Allegro. Do not promote that label
into a parity claim. Never set `FEATHER_PGS_SKIP_J_CLEAR`; the archived 16.6 ms
Allegro result with that setting is invalid.

## Read next / continuation priorities

1. [Original agent handoff](HANDOFF_fpgs_20260911.md), especially sections 5.1–5.5.
2. [Benchmark optimization/design guide](fpgs_solver_shape_and_spirit_20260909.md),
   especially sections 2, 12.7, 17–19.
3. [Profiling harness](../../scripts/benchmarks/fpgs_profile/README.md).
4. First choose the required mathematical target (legacy parity versus a deliberately
   changed solver). Validate per-substep velocities using real kernels before
   making further “same physics” performance claims.
5. For Allegro, the archived evidence points to GJK register pressure/spills as a
   remaining opportunity. Treat proposed dual active-set / acceleration ideas as
   untested, not established improvements.

The two historical Markdown files are preserved verbatim at the user's request.
Their old absolute paths, “nothing is committed” statement, and standing table are
historical context; this report and branch pins supersede them.

## Provenance and validation

The full 16-run sweep and two Franka repeats were measured on the original dirty
development snapshots based on Isaac Lab `53dd865aa37978c0a5e8cdce6d9f45e656fec29d`
and Newton `5631a64a373d1c73b4157ad4124eaf34914480a2`.
Hashes of Newton sources, Isaac Lab sources and the harness were unchanged across
the complete sweep and repeats. Commit preparation subsequently applied repository
lint/format cleanup in isolated worktrees; see validation notes below for checks
on the final snapshot. Do not confuse the original base SHAs with the experimental
source snapshot committed on these handoff branches.

Local raw evidence (not committed):
`/home/zhengyuz/Projects/IsaacLab.wt/fpgs-results-20260906/verify_20260910_0dciAR`.
Only compact numeric results and resolved configuration metadata are published.
Nsight SQLite/report files, huge warning logs, credentials, caches, and training
outputs are deliberately excluded.

Known analyzer limitation: graph memcpy/memset nodes keyed only by `graphNodeId`
are not fully represented in the current busy-time/node-count breakdown.
Independent SQL checks including those nodes left AnymalD/Franka graph spans
unchanged. Use the graph-span metric for this comparison, not an inferred breakdown
of exact occupancy from those auxiliary fields.

Final packaging validation is recorded in [VALIDATION.md](VALIDATION.md).
The post-format optimized-path checks measured 8,884 µs for AnymalD and
17,505 µs for Allegro, with finite sampled states. One Newton warm-start test
fails identically on the original and packaged snapshots; details are recorded
there rather than treating this experimental branch as a passing release.
