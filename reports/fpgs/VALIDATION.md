# Packaging validation

This is an experimental WIP handoff with a known inherited test failure,
not a fully passing release.

## Final optimized-path checks

After formatting/import cleanup, the final source snapshot was profiled again
with the same 16,384-environment recipe, physical GPU 1, seed 0, 200 warmup steps,
40 timed steps and three profiled steps:

| Task | Final snapshot graph µs / env step | Before packaging |
|---|---:|---:|
| AnymalD | 8,884.115 | 8,899.315 |
| Allegro | 17,504.578 | 17,497.078 |

Both before/after sampled states were finite. Final Newton source commit:
`31cf87f4694f873a027e41e2ca5e9ad441234456`.
Isaac Lab source was frozen before these checks; subsequent changes were only
dependency pins and standalone handoff/reference documents.

The full original 16-run sweep plus two Franka repeats had all recorded states
finite. Independent GPU-process audit found all 18 expected benchmark PIDs and
no foreign GPU 1 compute PIDs at approximately one-second sampling intervals.
This does not exclude interference shorter than the sampling interval.

## Isaac Lab

- Eight targeted launcher tests passed.
- Twenty-three selected Newton-manager configuration/control tests passed.
- FeatherPGS selector and disabled-by-default experimental configuration checks passed.
- Required `uv run isaaclab -f` workflow passed for the source snapshot.
- Shell syntax checks passed for `nsys_run.sh`, `ncu_run.sh`, and `reproduce.sh`.
- A stubbed command-expansion check verified all 16 reproduction invocations,
  correct task-specific flags, and no flag leakage between runs.
- Lockfile regenerated with uv 0.11.26; only the two Newton git-reference entries
  changed. All other package versions were preserved.

To avoid modifying the original working environment, the actual formatting
invocation used `UV_PROJECT_ENVIRONMENT=<original-venv> uv run --no-sync
--with pre-commit isaaclab -f`, with isolated source packages on `PYTHONPATH`.
The optional tool lived in a uv cache overlay. The changelog checker was pointed
at a temporary local ref for the actual base `53dd865a`; the ref was removed.
No remote validation branch was created. Standalone reports are outside the
Sphinx documentation tree; no Sphinx source changed.

## Newton

`uvx pre-commit run -a`: all hooks passed.

`uv build --wheel` passed for the committed Newton source. All six new modules
were included, and all nine changed source files inside the wheel matched their
checkout SHA256 hashes. Nothing was installed or changed in the original environment.

Required targeted tests:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m unittest \
    newton.tests.test_feather_pgs_mimic \
    newton.tests.test_feather_pgs_connect \
    newton.tests.test_feather_pgs_springs \
    newton.tests.test_feather_pgs_preelim
```

Result: **16 passed, one failed** (17 tests).
Failure:
`TestFeatherPGSPreelimination.test_dense_warmstart_preserves_projected_closure`.
The test expects `cache_peak > 1e-4`; observed `1.3120733e-05`.
Rerunning this single test against the untouched original Newton worktree
produced the identical value and failure. No solver fix or test relaxation was
made. Investigate this inherited condition before calling the branch release-ready.

The actual runs reused the existing Isaac Lab interpreter through
`uv run --no-project --python <original-venv>/bin/python` and selected the
appropriate Newton checkout via `PYTHONPATH`; no shared environment mutation.

## Snapshot audit

Newton includes the three previously dirty source files and all six previously
untracked experimental modules. Isaac Lab includes the five dirty source/demo
files and the nine existing profiling files, plus the portable reproduction
script, requested guides, dependency pin, and two package changelog fragments.

No migration or architecture redesign was performed. Existing experimental
decomposition is preserved as WIP so the next agent can resume it.

Isaac Lab source cleanup preserves all original Python ASTs except import order
and the explicit convergence caveat in a config docstring. Newton formatting
preserves the equations/defaults/dispatch conditions intentionally; differences
include eager internal imports, an unused debug-only clone removed, explicit
truncating zip behavior, native-code local renames, and equivalent syntax cleanup.
See [newton_source_inventory.json](newton_source_inventory.json) for per-file hashes
and changed-function inventory.

Historical guides remain verbatim. Raw captures and full logs remain local.
The [reference toolkit](reference/README.md) explains the separate large input
captures needed to rerun the exact archived section 5.1 comparisons.
