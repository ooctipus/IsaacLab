# mjwarp solver NaN-debug instrumentation

The Newton NaN debug utilities (`isaaclab_newton.physics.debug_state_buffer.DebugStateBuffer` and the
`scripts/tools/replay_*` tools) optionally use a GPU-side snapshot of the MuJoCo-Warp constraint solve.
That snapshot is produced by an env-gated patch to mujoco_warp's `solver.py`. Because mujoco_warp is
installed into the (git-ignored) virtualenv, the patched file cannot be committed directly — apply
`mjwarp_nan_debug.patch` to your install instead.

- **Target:** `mujoco_warp == 3.8.1` (`mujoco_warp/_src/solver.py`). Other versions may need a 3-way merge.
- The Newton-side buffer + replay tools work **without** this patch; it only adds the GPU solve snapshot
  (`snap_h` / `mjw_solve_debug_*`) and the pre-solve dump that the lowest-level replays consume.

## Apply

```bash
# from the directory that contains the mujoco_warp package (i.e. site-packages):
cd "$(python -c 'import mujoco_warp, os; print(os.path.dirname(os.path.dirname(mujoco_warp.__file__)))')"
git apply -p1 /path/to/scripts/tools/mjwarp_nan_debug.patch     # or: patch -p1 < .../mjwarp_nan_debug.patch
```

Revert with `git apply -R -p1 ...` (or `patch -R -p1 < ...`).

## Env-gated entry points (all default OFF — no effect unless set)

| Variable | Effect |
|---|---|
| `FACTORY_SOLVE_SNAPSHOT=1` | Latch the first NaN iteration's Hessian + iterate GPU-side (consumed by `DebugStateBuffer` → `mjw_solve_debug_*`). |
| `FACTORY_PRESOLVE_DUMP=<path.npz>` | On the first NaN `qacc`, dump the complete consistent pre-solve state for that world to `<path>` (consumed by `replay_solver_hessian.py` / `replay_mjw_direct_solve.py`). |
| `FACTORY_ITER_DUMP=<path.npz>` | Dump the first NaN Cholesky iteration's assembled Hessian + inputs. |
| `FACTORY_SOLVE_TRACE=1` | Per-iteration NaN tripwire to pinpoint which solver op first goes non-finite (forces the Python solve loop). |
| `FACTORY_HESSIAN_DIAG_REG=<eps>` | Add `eps * max(diag)` to the constraint Hessian before factorization (candidate-fix test for indefinite Hessians). |

The `SOLVE_SNAPSHOT` path is all-GPU on the hot loop; the `PRESOLVE_DUMP` / `ITER_DUMP` paths only do
host work on the first NaN. None affect numerics unless their variable is set.
