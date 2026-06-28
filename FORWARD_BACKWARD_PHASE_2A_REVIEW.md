<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Forward–Backward Unification: Phase 2A Review

Status: **preflight evidence complete; local BFM hardware gate failed; Phase 2B not started**

RSL-RL candidate: **`af2105f58184331a68ae7d419233ea9277f19ed2`**

Persistent artifact root: **`/home/zhengyuz/Projects/forward_backward_phase2_runs`**

## Outcome

Phase 2A verified the five repository identities, three Python environments,
all frozen Meta/BFM datasets, checkpoints, and numerical oracles. Native Meta
and BFM checkpoints load, both native environments reset and step, and their
reference evaluation paths execute without writing into the frozen source
artifacts.

The systems gate produced a decisive result. The native 1,024-environment BFM
simulator adds **8.213 GiB** beyond its starting CUDA context. Combined with the
Phase 1 exact learner peak of **25.2188 GiB**, the projected co-resident footprint
is **33.4319 GiB**. Each local RTX 5090 exposes **31.3566 GiB**, so the full
candidate cannot fit physically and exceeds the device by **2.0753 GiB** before
the required 15% safety margin.

A 48 GiB device projects to **30.35% headroom** at the same measured footprint.
The local GPUs remain suitable for Meta, component parity, adapter development,
and reduced BFM smoke tests, but not for a valid 1,024-environment BFM learning
comparison. No capacity, precision, model, replay, or environment-count fallback
was introduced.

## Exit-gate decision

| Phase 2A requirement | Result | Evidence |
|---|---|---|
| Frozen source and local-state identity | Pass | Five repository snapshots with tracked-diff and untracked-path fingerprints |
| Exact dependency-freeze command and output | Pass | IsaacLab, MetaMotivo, and BFM `.venv` manifests |
| Dataset/checkpoint/oracle hashes | Pass with two documented Phase 0 transcription corrections | 14 streaming SHA-256 checks |
| Native environment import/reset/step | Pass | Meta 50-env and BFM 1/1,024-env probes |
| Native checkpoint/evaluator smoke | Pass | One complete Meta motion; BFM motion 25 context and 100 policy steps |
| BFM exact-shape simulator measurement | Pass | 1,024 environments, 120 control steps, all 862 clips loaded |
| Co-resident learner + simulator, 15% headroom | **Fail on local 32 GiB** | 33.4319 GiB projected versus 31.3566 GiB usable |
| Persistent storage budget | Pass for rolling/final-checkpoint policy | 139.64 GiB available; 80 GiB minimum declared |
| Exact 2D/2E transition matrix fixed before outcomes | Pass, pending user review | `FORWARD_BACKWARD_PHASE_2_RUN_MATRIX.yaml` |

The Phase 2A measurement work is complete, but its strict hardware exit gate is
not satisfied by this machine. Phase 2B requires an explicit review decision:
either authorize adapter/parity work locally while provisioning a 48+ GiB GPU
before BFM learning, or stop until eligible hardware is available.

## Immutable provenance snapshot

The generated snapshot is:

```text
/home/zhengyuz/Projects/forward_backward_phase2_runs/preflight/phase2a_snapshot.json
```

### Repository identity

| Repository | Branch / commit | Tracked state relative to Phase 0 |
|---|---|---|
| IsaacLab | `feature/unify` / `063250f5` | Same tracked diff hash; new Phase 1/2 documents and tools extend the untracked manifest |
| RSL-RL | `octi/forward-backward-unification` / `af2105f5` | Tracked worktree clean; 18 pre-existing untracked paths recorded |
| MetaMotivo | `main` / `ff8dcc55` | Same tracked diff hash as Phase 0; 14 untracked paths recorded |
| HumEnv | `main` / `0548761a` | Same tracked diff hash as Phase 0; 15 untracked paths recorded |
| BFM-Zero | `main` / `b87916f5` | Same tracked diff hash as Phase 0; materialized LFS data and 133 untracked paths recorded |

No worktree was cleaned or reset. Unrelated user changes remain untouched.

### Environment identity

| Environment | Python | Torch / CUDA | Clean RSL-RL resolution | Freeze SHA-256 |
|---|---:|---:|---|---|
| IsaacLab `.venv` | 3.12.12 | 2.10.0+cu128 / 12.8 | `/home/zhengyuz/Projects/rsl_rl` | `93e4497a59ff3933c6bd62755fe7984da50d300a1ebae9fcc53d8d9e022a08bb` |
| MetaMotivo `.venv` | 3.10.19 | 2.9.1+cu128 / 12.8 | `/home/zhengyuz/Projects/rsl_rl` | `055785b2f9a655d9a9f2b3b74d46ff1f5b5a269ebacf52fa0a4ac8ad71a41715` |
| BFM `.venv` | 3.10.19 | 2.7.0+cu128 / 12.8 | Installed package by default; explicit local path import succeeds | `a106988a853e6c14fd992396cae6bfed9de21f6835811041f7bd526bf8e59dcb` |

The BFM environment does not currently select the candidate RSL-RL checkout by
default. Phase 2B must select `/home/zhengyuz/Projects/rsl_rl` explicitly and
refreeze the environment before any candidate evidence is accepted.

The preflight also exposed an interpreter-isolation requirement. A process
launched through `isaaclab.sh` exports Isaac Sim `PYTHONHOME`/`PYTHONPATH` state;
launching a Python 3.10 child `.venv` without scrubbing those variables mixes
Python 3.12 and 3.10 standard libraries. The manifest tool now clears only the
inherited interpreter-selection variables for cross-environment probes. Native
Meta/BFM commands continue to run directly in their repository-local `.venv`.

## Frozen artifact verification

All declared datasets and numerical evidence match their actual files:

| Artifact family | Verified objects |
|---|---|
| Meta split | 1,638-train and 182-test list; exact frozen hashes |
| Meta checkpoint | 168.38 MB model and 106.81 MB optimizer state |
| Meta oracle | generator, compactor, manifest, 11.86 MB semantic tensor |
| BFM data | 209.66 MB 40-motion source and 205.12 MB 862-clip corpus |
| BFM checkpoint | 3.385 GB released model |
| BFM oracle | generator, manifest, semantic tensor |
| BFM curve | 19,826 rows = 862 motions × 23 evaluation checkpoints |

### Phase 0 digest transcription corrections

The files are stable; two hashes were transcribed incorrectly into the Phase 0
review prose:

| Artifact | Phase 0 prose | Independently verified value |
|---|---|---|
| Meta oracle compactor | omitted `a2` | `899c3cdd0e641684e7ed0a4a2b9688e1e0432f43af472a2e8d5153958f3269b7` |
| BFM released model | omitted `d` | `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361` |

The BFM corrected value is also stored inside its independently generated
`phase0_oracle/oracle.json`. These are provenance corrections, not changed
files, equations, metrics, or acceptance thresholds. The immutable Phase 2A
snapshot records both corrections explicitly.

## Native smoke evidence

### MetaMotivo/HumEnv

The smoke used GPU 1, the frozen checkpoint, and one evenly selected motion from
the 182-motion test list through the native `TrackingEvaluation`:

```text
checkpoint load       pass
model architecture    simple, hidden 1024 × 2, obs 358, action 69
environment reset     pass
motion evaluation     pass, 1/1 complete
EMD                    1.4328
PHC-L-infinity         1.0
output                 preflight/meta_checkpoint_smoke/eval_results_minimal_1motions.json
```

The checkpoint was exposed through a read-only symlink inside the Phase 2 root,
so the native evaluator's output write did not modify the archived run.

### BFM-Zero

The smoke used GPU 1, the released 3.385 GB checkpoint, motion 25, headless Isaac
Sim, and the native inference path:

```text
checkpoint load       pass
motion corpus         862 clips discovered
Isaac Sim G1 build    pass
motion-25 reset       pass
policy rollout        100 control steps complete
context artifact      preflight/bfm_checkpoint_smoke/model/tracking_inference/zs_25.pkl
```

The model/config were read-only symlinks. ONNX/context outputs were written only
under the Phase 2 root. The process exited cleanly and released the GPU.

## Final-shape environment measurements

### Meta, 50 environments

| Quantity | Measurement |
|---|---:|
| Full 1,638-motion load + build | 12.04 s |
| Control throughput | 172.62 vector steps/s |
| Transition throughput | 8,630.80 edge positions/s |
| Maximum host RSS | 161.8 MiB |
| Observation | `obs[50,358]`, `time[50,1]` |
| Boundary exercise | 50 truncations in 350 measured steps; zero terminations |

The immutable reference run reached 4.975M logged transitions in 157.60 minutes
(about 2.627 hours) including native evaluation. Tracking evaluation itself was
approximately 16.6–43.1 seconds per checkpoint; the native reward evaluator was
approximately 205–213 seconds per checkpoint.

### BFM, 1,024 environments

| Quantity | Measurement |
|---|---:|
| Build and all-862-motion load | 14.44 s |
| Control throughput | 23.08 vector steps/s |
| Transition throughput | 23,631.54 edge positions/s |
| Used device memory after steady steps | 8.723 GiB |
| Increment beyond starting CUDA context | **8.213 GiB** |
| Torch reserved / physical process use | 1.135 GiB / 8.723 GiB |
| Maximum host RSS | 6.02 GiB |
| Observations | state 64, privileged 463, last action 29, history 372, time 1 |

The large difference between Torch reservation and physical device use is why
the simulator cannot be capacity-planned from Torch tensors alone.

### Co-resident capacity law

```text
Phase 1 exact learner peak       25.2188 GiB
Native simulator increment       8.2131 GiB
Projected combined              33.4319 GiB
Local usable device             31.3566 GiB
Physical deficit                 2.0753 GiB
Projected 48 GiB headroom          30.35%
```

Two 32 GiB cards do not solve this without a new distributed learner/simulator
design. Phase 1 intentionally does not provide that design, and adding it during
validation would change the system under test.

## Proposed transition and compute budget

The review-frozen proposal lives in
`FORWARD_BACKWARD_PHASE_2_RUN_MATRIX.yaml`.

### Meta

| Gate | Pairing | Horizon | Approximate reference evidence |
|---|---|---:|---:|
| 2D | seed 0 source/candidate | 0.5M | one native interval |
| 2E | seeds 0/1/2 source/candidate | 1.5M each | three intervals |
| 2F | missing source 1/2 + candidate 0/1/2 | 5M each | 2.627 h/reference run |

The candidate budget will be updated from the first 2D integrated measurement;
Meta is not the resource risk in Phase 2.

### BFM

The candidate estimate combines the measured simulator step cost with 16 Phase 1
updates at 0.215079 s/update. It is optimistic until the integrated 2D run adds
native evaluation overhead; the budget therefore adds 20% candidate contingency.

| Gate | Horizon | Source evidence | Candidate estimate | Paired budget |
|---|---:|---:|---:|---:|
| 2D | 9.6M | 6.791 h | 9.074 h raw / 10.889 h budget | 17.680 GPU-h |
| 2E | 28.8M × 3 seeds | 20.342 h/seed | 27.223 h raw / 32.668 h budget per seed | **159.030 GPU-h** |
| 2F | 211.2M × one pair | 149.580 h | 199.638 h raw / 239.566 h budget | **389.146 GPU-h** |

The released BFM log is direct evidence: its 211.2M row records 8,974.77 minutes
(149.58 hours). The full corrected-terminal pair is therefore a roughly
16.2-GPU-day budget even before scheduling inefficiency.

## Storage policy

The filesystem currently has 139.64 GiB free. The run matrix requires at least
80 GiB free before BFM training and uses this retention policy:

- per-motion numeric evaluation rows are persisted at every frozen grid point;
- full learner/replay checkpoints are rolling recovery scratch while a run is
  incomplete;
- only the final full checkpoint/model, metrics, manifests, and resource trace
  become immutable when `COMPLETE` is written; and
- no complete reference/candidate artifact is overwritten or silently dropped.

This avoids multiplying a 22+ GiB learner checkpoint by every 9.6M evaluation
point while preserving final state and the entire learning curve.

## Files added by Phase 2A

```text
FORWARD_BACKWARD_PHASE_2A_REVIEW.md
FORWARD_BACKWARD_PHASE_2_RUN_MATRIX.yaml
scripts/reinforcement_learning/forward_backward/phase2/
  preflight.py
  meta_env_probe.py
  bfm_env_probe.py
```

Generated external artifacts:

```text
/home/zhengyuz/Projects/forward_backward_phase2_runs/
  registry.jsonl
  preflight/
    phase2a_snapshot.json
    phase2a_validation.json
    meta_env_50.json
    bfm_env_1024.json
    meta_checkpoint_smoke/
    bfm_checkpoint_smoke/
```

## Verification and repository health

Completed:

```text
preflight manifest compile/execute             passed
14 frozen artifact hashes                      passed
Meta checkpoint + native evaluator smoke       passed
BFM checkpoint + Isaac Sim rollout smoke       passed
Meta 50-env final-shape probe                   passed
BFM 1,024-env final-shape probe                passed
YAML validation                                 passed
license insertion / codespell / key checks      passed
```

The mandated repository-wide `./isaaclab.sh -f` was run. It formatted the new
preflight script, then failed on unrelated pre-existing repository state:

- undefined import names in `cloner_utils.py` and the PhysX contact sensor;
- an unused variable in the pre-existing Octi CRL implementation; and
- numerous golden image LFS pointers not covered by the current attributes.

The hook also removed a final newline from tracked `imgui.ini`; that unrelated
hook change was restored exactly. No unrelated source change was modified by
Phase 2A. A focused hook run over the Phase 2A files is recorded separately.

## Review decision requested

Recommended decision:

1. approve the Phase 2A provenance, smoke, run matrix, and cost findings;
2. allow Phase 2B adapter/evaluator implementation and Phase 2C parity work on
   the local machine; and
3. require a single 48+ GiB CUDA device before authorizing BFM 2D learning, then
   rerun the integrated co-resident memory probe on that device.

Phase 2B has not started. Stop here for user review.
