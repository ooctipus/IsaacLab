<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Forward–Backward Unification: Phase 2B Review

Status: **translation adapters and native evaluators complete; stopped before 2C**

Reviewed implementation commits:

- RSL-RL: `c1d76564b` on `octi/forward-backward-unification`, pushed to `ooctipus/rsl_rl`.
- MetaMotivo: `853cee7` on local branch `octi/phase2-adapter`.
- BFM-Zero: `94267c8` on `octi/phase2-adapter`, pushed to `ooctipus/BFM-Zero`.

Persistent evidence root:

```text
/home/zhengyuz/Projects/forward_backward_phase2_runs/remote_4g/phase2b
```

## Decision

Phase 2B passes its frozen exit gate. Both adapters transport native facts into
the Phase 1 RSL-RL contracts without implementing learner targets, losses,
relabeling, rewards, or EMD. Complete native evaluator outputs contain exactly
182 Meta motions and 862 BFM motions, every numeric row is finite, the released
BFM checkpoint reproduces sub-1 EMD, and exact same-step final capture is both
complete and inexpensive.

No Phase 2C parity work or candidate learning run has started. Approval is
required before proceeding.

## Code delivered

### Shared RSL-RL boundary

```text
rsl_rl/algorithms/forward_backward.py
rsl_rl/modules/distribution.py
rsl_rl/runners/off_policy_runner.py
tests/runners/test_off_policy_runner.py
```

The existing generic runner now supports a source-compatible random-action
seed phase. It counts attempted environment transitions, calls the learner's
bounded uniform behavior path before the declared threshold, delays updates by
the same source-loop boundary, and checkpoints the transition clock. A separate
behavior generator is checkpointed so seed actions neither perturb the learner
context RNG nor break exact resume. Existing configurations that omit
`random_action_steps` retain their previous behavior.

This correction was found during the final cadence audit: Meta uses 50,000
uniform-random transitions, while BFM uses 10,240. Starting updates as soon as
one replay batch existed would not have been a valid source comparison.

### MetaMotivo adapter

```text
../../metamotivo/phase2_adapter/
  __init__.py
  candidate.py
  environment.py
  evaluate_all_motions.py
  evaluation.py
  expert.py
  test_adapter.py
  test_candidate.py
```

The adapter exposes the unchanged HumEnv distribution, a motion-safe immutable
expert corpus, the native `TrackingEvaluation`, and the unified RSL-RL
candidate. The final cadence is 10 vector steps over 50 environments followed
by 50 updates, 50,000 random seed transitions, and one checkpoint every 1,000
iterations = 0.5M transitions.

### BFM-Zero adapter

```text
../../BFM-Zero/phase2_adapter/
  __init__.py
  benchmark_final_capture.py
  candidate.py
  environment.py
  evaluate_all_motions.py
  evaluation.py
  expert.py
  test_adapter.py
  test_candidate.py
```

The adapter exposes the native G1 environment and expert motion library,
asymmetric routes, compact history reconstruction, eight named auxiliary reward
facts, explicit native-reference/correct-terminal profiles, the unchanged
`HumanoidVerseIsaacTrackingEvaluation`, and the unified candidate. The cadence
is one 1,024-environment vector step followed by 16 updates, 10,240 random seed
transitions, and a checkpoint every 9,375 iterations = 9.6M transitions.

The BFM work is available at:

```text
https://github.com/ooctipus/BFM-Zero/tree/octi/phase2-adapter
```

Only `phase2_adapter/*.py` was committed. Pre-existing LFS materialization,
`uv.lock`, caches, checkpoints, reproduction outputs, and research changes were
not staged.

### Common evidence tooling

```text
scripts/reinforcement_learning/forward_backward/phase2/evaluation_records.py
scripts/reinforcement_learning/forward_backward/phase2/test_evaluation_records.py
```

The frozen 11-column record, complete rectangular-cardinality validation,
no-drop exact join, deterministic 10,000-replicate paired bootstrap, normalized
trapezoid AUC, no-overwrite writer, and manifest validation are shared by both
tracks.

## Schema and action trace

| Contract | MetaMotivo/HumEnv | BFM-Zero/G1 |
|---|---|---|
| Control rate | 30 Hz | 50 Hz |
| Native action | 69, passed unchanged | 29, passed unchanged |
| Actor route | `policy[358]` | `state[64] + last_action[29] + history_actor[372] = 465` |
| Forward/value route | `policy[358]` | `state[64] + privileged_state[463] + last_action[29] + history_actor[372] = 928` |
| Backward/discriminator route | `policy[358]` | `state[64] + privileged_state[463] = 527` |
| History | Native flat policy observation | Four frames × 93 facts = 372; reconstructed by replay from action/state slices |
| Reward facts | Native environment reward; discriminator is recomputed by learner | Native environment reward plus eight named auxiliary evidence channels; discriminator recomputed by learner |
| Autoreset | Gymnasium next-step | Same-step |
| Expert windows | Contiguous length 8, never cross motion offsets | Contiguous length 8; length 257 also available for 250-step rollout tracking plus context prefix |

No adapter owns an FB target, discriminator objective, value target, actor
objective, context mixture, normalization update, auxiliary reward combination,
or optimizer. Candidate presets refer to the single
`rsl_rl.algorithms.forward_backward:ForwardBackward` implementation.

## Boundary proof

### Meta next-step

HumEnv emits the terminal successor on the done step and performs reset at the
next call. The adapter records the done edge normally, then marks the following
reset-only row `action_applied=False`; replay therefore cannot create a
pseudo-transition from an action the simulator did not apply. Tests also prove
the 69-dimensional action array reaches the native environment unchanged.

### BFM exact same-step

Released BFM resets inside `_post_physics_step` before producing its returned
observation. The correct-terminal profile installs one narrow hook around the
native `reset_envs_idx` method:

1. identify the exact done indices supplied by the source environment;
2. recompute the already-reached observation before reset;
3. copy only those rows of the four emitted observation fields and physical
   `qpos[36]`/`qvel[35]` evidence;
4. call the original reset method unchanged; and
5. return the normal post-reset observation plus an explicit validity mask.

The extra observation call runs inside a forked Torch RNG context, so observation
noise cannot advance the behavior RNG stream. On non-terminal steps the hook is
never entered; action, environment step, returned observation, reward, done, and
auxiliary evidence stay on the native path. The fake-environment contract test
proves pre-reset value 1 versus post-reset value 0 and exact action identity.

The 1,024-environment native benchmark produced 1,024 done rows, 1,024 exact
final rows, zero missing finals, and a terminal/post-reset state L-infinity
difference of 20.1808, directly proving that the captured state is not the reset
state.

## Exact-final systems cost

The accepted performance comparison is an A-B-A sequence on the same RTX 6000
Ada GPU, with 1,024 environments, 25 warm-up steps, and 600 measured control
steps per leg:

| Profile | Duration | Throughput | Torch allocation |
|---|---:|---:|---:|
| Native A1 | 23.9776 s | 25,623.9 edges/s | 1.074891 GiB |
| Correct terminal B | 24.4287 s | 25,150.7 edges/s | 1.079112 GiB |
| Native A2 | 24.7314 s | 24,842.9 edges/s | 1.074891 GiB |
| Native A1/A2 mean | 24.3545 s | 25,233.4 edges/s | 1.074891 GiB |

Relative to the bracketed native mean, exact capture costs **0.33% throughput**
and **0.30% duration**. Additional Torch allocation is **4.322 MiB**. The
explicit dense payload lower bound is 3.903 MiB:

```text
1024 × (928 observation + 36 qpos + 35 qvel) × 4 bytes + 1024 validity bits/bytes
```

The measured excess over that payload is approximately 0.42 MiB. This is close
enough to the theoretical storage limit that a Warp rewrite is not justified in
Phase 2B.

The earlier simultaneous cross-GPU run measured a misleading 4.90% difference;
it is retained as concurrency evidence but is not used for the accepted cost.

## Native evaluator evidence

The adapters call the native evaluators and only flatten their returned scalar
metrics. Neither adapter copies or reimplements EMD.

### MetaMotivo

Accepted artifact:

```text
meta_reference_all_motions_retry4_seeded_gpu0/
```

| Quantity | Result |
|---|---:|
| Checkpoint transition | 4,975,000 |
| Training/evaluation seed | 0 / 0 |
| Motions | 182 |
| Scalar metrics per motion | 9 |
| Long-form rows | 1,638 |
| Native evaluator duration | 43.668 s |
| Mean EMD | 1.694893 |
| Mean PHC L-infinity success | 0.648352 |
| Mean PHC mean success | 0.967033 |

The output exactly matches the expected motion/metric rectangle and contains no
nonfinite value. The native worker assumes logical CUDA device 0, so a retained
GPU-1 attempt failed clearly with a device mismatch; the accepted protocol uses
GPU 0.

### BFM-Zero

Accepted artifact:

```text
bfm_released_all_motions_retry2_seeded/
```

| Quantity | Result |
|---|---:|
| Checkpoint transition | 211,200,000 |
| Training/evaluation seed | 4728 / 4728 |
| Motions | 862 |
| Scalar metrics per motion | 9 |
| Long-form rows | 7,758 |
| Native evaluator duration | 67.470 s |
| Released CSV mean EMD | 0.735207 |
| Seeded rerun mean EMD | **0.741493** |
| Mean delta | +0.006286 |
| Mean absolute per-motion delta | 0.025603 |
| Per-motion correlation | 0.934604 |

The seeded rerun is below 1 EMD and satisfies the frozen native-stochastic final
gate of 0.7453. The stochastic curve is not treated as a deterministic paired
gate; that remains a later correct-terminal comparison.

## Test and quality evidence

| Scope | Result |
|---|---|
| Shared record schema/statistics | 5 passed |
| Meta adapter/candidate, local | 4 passed |
| Meta adapter/candidate, remote `.venv` | 4 passed |
| BFM adapter/candidate, remote `.venv` | 4 passed; 3 upstream dependency warnings |
| RSL-RL full focused FB suite, local | 185 passed, 16 external-oracle skips |
| RSL-RL focused synced suite, remote `.venv` | 185 passed |
| RSL-RL pre-commit, all files before and after commit | all hooks passed |
| BFM adapter Ruff | passed; 9 files formatted |
| Meta adapter Ruff | passed; 8 files formatted |

## Retained failure and correction ledger

Failed and invalid runs remain under the evidence root. They were not deleted or
silently overwritten.

| Attempt | Finding | Correction |
|---|---|---|
| Initial Meta 60-worker run | Remote soft file limit 1,024 caused `Too many open files` | Raised only the process soft limit to 65,536; evaluator unchanged |
| Initial BFM startup | Simulator received unindexed `cuda` | Public CLI now defaults to indexed `cuda:0`; multi-GPU runs pass explicit indices |
| First exact benchmark | Used nonexistent Tensor `maximum_` | Replaced with `torch.maximum(..., out=...)` |
| Meta retry 1 | NumPy `int64` rejected by JSON writer after evaluation | Translation writer now handles NumPy scalars; native metrics unchanged |
| BFM retry 1 | Complete 862-motion output but seed was recorded, not applied; mean EMD 2.659 | Call source `set_seed_everywhere(4728)` before model/environment construction |
| Meta seeded GPU-1 attempt | Native multiprocessing workers used logical CUDA 0 | Accepted seeded run uses logical CUDA 0 |
| Final cadence audit | Candidate would update after one batch rather than after source random seed phase | Shared RSL-RL random behavior/update gate; Meta 50k and BFM 10,240 presets |

## Exit-gate audit

| Frozen Phase 2B requirement | Result |
|---|---|
| All native adapter tests pass | Pass |
| Non-terminal source behavior unchanged | Pass by direct adapter path, action-identity tests, and reset-hook scope |
| All-motion outputs complete and finite | Pass: Meta 182 × 9; BFM 862 × 9 |
| Final capture matches pre-reset physical state and reports cost | Pass: 1,024/1,024 valid, zero missing, 0.33% throughput cost |
| No duplicated learner mathematics or EMD | Pass |
| Source/candidate cadence represented | Pass after shared random-action correction |

## Requested review decision

Please review the adapter boundaries, exact-final hook, source cadence, complete
evaluator outputs, and 0.33% systems cost. If approved, Phase 2C will begin with
fixed-batch value/gradient/one-Adam-step parity. It will not start stochastic
learning.
