# Section 5.1 reference replay

`truth.py` reruns the real legacy row Gauss-Seidel and parallel projection kernels on captured inputs.
It prints median, p90 and maximum relative per-world velocity differences for 24 and 400 iterations.
The **parallel 400 vs legacy 400** line is the comparison behind section 5.1; 400 is a reference iteration
budget, not a proof of convergence in every world. The 24-vs-24 line is a controlled reference comparison,
not the current task configuration: these captures record legacy budgets of 8 (AnymalD) and 12 (Allegro).

## Run

Use the Newton FPGS source supplied with this handoff and its compatible uv environment. The original environment
had Newton `1.6.0.dev0`, Warp `1.17.0`, and NumPy `2.5.1`. The recorded Newton HEAD was
`5631a64a373d1c73b4157ad4124eaf34914480a2` **with local changes**; that commit alone is insufficient.
This script imports private Newton kernel factories and depends on their matching argument layout.

Transfer the capture pairs described below, preserving their basenames. From the Isaac Lab checkout, with
`FPGS_NEWTON` set to the matching Newton checkout and `FPGS_CAPTURE_ROOT` to writable working copies:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$FPGS_NEWTON" FEATHER_PGS_WR_STOP=0 \
  uv run python reports/fpgs/reference/truth.py "$FPGS_CAPTURE_ROOT/anymald_seq/mfgs_step1608_phase0" 48
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$FPGS_NEWTON" FEATHER_PGS_WR_STOP=0 \
  uv run python reports/fpgs/reference/truth.py "$FPGS_CAPTURE_ROOT/allegro_seq/mfgs_step1608_phase0" 128
```

The script uses logical `cuda:0`; `CUDA_VISIBLE_DEVICES` selects the physical GPU. The captures were made on an
RTX 5090 and record architecture 120. Other GPU architectures have not been validated. Each invocation writes
`<capture_base>_truth.npz`, replacing any existing file there; retain archived outputs separately.
The two scripts need no Isaac Lab simulation launch, assets, or Nsight capture for this offline replay.
`oracle.py` supplies input loading and launch ordering. Its separate `--new` and `--reps > 1` experiments
need `newpath.py`, which is intentionally outside this reference toolkit.

## Inputs remain local

No NPZ inputs, stored truth outputs, or raw Nsight/SQLite captures are included here.
The **12 input NPZ/JSON pairs total 15.985 GiB** and are necessary to replay all six stored substeps for both
tasks. Replaying just the first snapshot for each task still requires 2.664 GiB of input NPZs.
Existing `_truth.npz` files are optional comparison outputs, not substitutes for the input pairs.

[CAPTURES.json](CAPTURES.json) records the exact original capture root, sequence filenames and sizes, and SHA256s
for both first snapshots and their stored truth outputs. The remaining steps are 1609 through 1613 with the same
`mfgs_step<step>_phase0.npz/.json` pattern. The two small metadata JSONs are bundled here under descriptive names;
restore each beside its input NPZ as `mfgs_step1608_phase0.json` when transferring an input.

The snapshots live under a temporary directory on the original machine; they must be transferred separately
before that directory is cleaned. This toolkit alone cannot rerun the stored comparison.

## Provenance and changes

Original script SHA256s:

- `formulation/truth.py`: `ed69dd3f98c1165c2bf0746311fa895ddf95748ff72ae9248e1dbe82a5cf7384`
- `k2real/oracle.py`: `33100f7740f57242332164ffa5477f053d874ad5c768273b479cfac3b29e8ec7`

Both originals were siblings of the capture root in the same scratchpad. The oracle also matches the archived
`profile_artifacts_20260908/k2/oracle.py` byte for byte. Portable changes add repository license headers, use the
sibling oracle import, apply mechanical formatting/import cleanup, close loaded files with context managers,
and correct the historical "today" label. Kernel factories, launch inputs, equations, numerical defaults,
iteration counts, tolerances, world selection and error calculations are unchanged.

Only syntax, lint and source-equivalence checks were run while packaging. This packaging step did not rerun
the CUDA truth calculation. See the [handoff](../HANDOFF_fpgs_20260911.md) and
[formulation study](../fpgs_solver_shape_and_spirit_20260909.md) for interpretation and prior observations.
