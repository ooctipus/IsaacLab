---
name: debug-newton-physics
description: Diagnose and investigate Isaac Lab Newton and MuJoCo Warp physics incidents using strict capture, custom triggers, transient operation probes, archive validation, comparison, and replay. Use when Codex needs to configure or analyze Newton/MJWarp debugging for NaNs or infinities, solver/Hessian failures, collision or contact errors, finite joint explosions, post-step pre-reset failures, multi-world attribution, debugger schema changes, custom operation providers, or physics incident archives.
---

<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Debug Newton Physics

Use the branch-owned physics incident recorder as the source of evidence. Prefer a focused, reproducible capture over speculative solver changes or temporary edits to installed Newton or MuJoCo Warp code.

## Establish availability

1. Work from the Isaac Lab repository root.
2. Confirm these branch-owned files exist:
   - `source/isaaclab_newton/isaaclab_newton/physics/_incident_recorder.py`
   - `source/isaaclab_newton/isaaclab_newton/physics/mjwarp_debug.py`
   - `scripts/tools/physics_debug.py`
   - `docs/source/overview/core-concepts/physical-backends/newton/debugging-incidents.rst`
3. If they are absent, report that the private debugger branch has not been merged into this worktree. Do not reconstruct the deleted `DebugStateBuffer`, specialized replay scripts, or MJWarp source patch.
4. Read the canonical debugging guide above when exact configuration, trigger, provider, or replay-adapter examples are needed. Treat the installed code and the archive manifest as authoritative when upstream Newton or MJWarp versions differ from remembered field names.

## Choose the evidence path

State the physical invariant that failed, the expected scope, and the earliest observable phase before changing configuration.

- For a retained NaN or infinity, scan `state`, `solver`, or another recorded provider with `detect_nonfinite_in`.
- For a finite but invalid result, register a named `NewtonManager.set_debug_incident_trigger()` callback before solver initialization.
- For a transient MJWarp local, enable operation recording. Use the built-in provider for final solver and collision contexts, or configure first-iteration scanning for a deep workspace.
- For a transition or reset-boundary failure, enable replay and compare pre/post values. Rely on synchronous post-step capture to preserve the failing state before the next task reset.
- For a multi-world failure, keep `failed_worlds_only=True` unless cross-world coupling is itself under investigation.

Keep selection patterns strict. Every include, exclude, and non-finite scan pattern must match; treat a mismatch as an upstream schema change, not as a reason to add a fallback.

## Configure a focused baseline

Start from this shape and adjust providers to the hypothesis:

```python
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonDebugCaptureCfg,
    NewtonDebugReplayCfg,
)

physics_cfg = NewtonCfg(
    use_cuda_graph=False,
    solver_reset=NewtonCfg.SolverResetCfg(enabled=False),
    debug_capture=NewtonDebugCaptureCfg(
        output_dir="./physics_debug",
        history_length=200,
        failed_worlds_only=True,
        max_incidents=5,
        halt_on_incident=True,
        fail_on_capture_error=True,
        record_scene=False,
        record_model=True,
        record_control=True,
        record_solver=True,
        record_contacts=False,
        record_collision_pipeline=False,
        record_operations=False,
        detect_nonfinite_in=("state", "solver"),
        replay=NewtonDebugReplayCfg(
            enabled=True,
            record_state=True,
            record_control=True,
            record_solver=True,
        ),
    ),
)
```

Apply these rules:

- Keep `fail_on_capture_error=True`. Do not catch and suppress initialization, schema, provider, or export failures.
- Keep `record_scene=False` unless static USD geometry is necessary; captured arrays are the authoritative dynamic state.
- Set `use_cuda_graph=False` when replay, per-substep capture, or operation recording is enabled.
- Size `max_gpu_bytes` deliberately. Let initialization report the exact field allocation instead of dropping fields.
- Enable `record_contacts` and `record_collision_pipeline` only for a finalized external Newton collision pipeline.
- For internal MJWarp collision, leave those options off and use `record_solver=True` for retained `mjw_data` plus `record_operations=True` for transient solver and collision contexts.
- Treat `solver_reset` as independent. Enable it only when stale solver-owned reset state is relevant; capture still works without it.
- Keep `halt_on_incident=True` by default. Continue only when collecting multiple incidents is intentional and safe.

## Probe finite failures

Register custom triggers after installing `NewtonCfg.debug_capture` and before solver initialization. Return `NewtonManager.DebugTriggerResult` with exact `world_ids`, or use `global_scope=True` only for genuinely world-independent evidence.

Make trigger extraction solver-aware and fail if its expected schema is unavailable. Do not use `getattr(..., default)`, broad exception handling, or guessed field aliases. Inspect `context.state`, `context.control`, `context.contacts`, `context.collision_pipeline`, `context.solver`, `context.operations`, `context.phase`, and `context.substep_idx` as appropriate.

Use finite triggers for conditions such as:

- joint velocity or acceleration exceeding a physical bound;
- penetration depth growing while a contact pair remains expected;
- a broad-phase pair disappearing before narrow phase;
- contact rows or constraints disappearing between collision and solve;
- impulses, residuals, or iteration counts exceeding a known valid range.

## Probe MJWarp without source patches

For exactly one MJWarp-backed `SolverMuJoCo`, including one selected from a coupled solver mapping, set incident or replay `record_operations=True`. The manager automatically installs `MJWarpDebugOperationProvider`, validates the live private call path, discovers the installed `Data`, `SolverContext`, and `CollisionContext` schemas, and restores all interposed functions on close.

For first-bad-iteration evidence, register an explicit provider before solver initialization:

```python
from isaaclab_newton.physics import MJWarpDebugOperationProvider, NewtonManager

NewtonManager.set_debug_operation_provider(
    MJWarpDebugOperationProvider(
        first_nonfinite_include_fields=(
            "mjwarp_solver_context.h",
            "mjwarp_solver_context.hfactor",
        ),
    )
)
```

Also enable incident `record_operations`, add `"operations"` to `detect_nonfinite_in`, and use strict detection paths under `operations.first_nonfinite_context`. Set `capture_per_substep=True` when later substeps could overwrite the operation snapshot.

Do not keep or inject a local MJWarp kernel diff. A value created and overwritten entirely within one compiled kernel is the one boundary the host-side provider cannot observe. In that case, add an explicit retained upstream diagnostic or a narrowly scoped custom hook, test it in a standalone reproduction, and remove any `wp.printf` before committing.

## Follow the symptom

- For a convex hull sinking into a box, trace broad-phase pair generation, narrow-phase geometry, active contact rows, solver constraints, and applied impulses in order. Compare the last finite pre-state with the failing state; do not assume the first array dimension is a world dimension.
- For a Hessian or factorization NaN, scan the narrow workspace per iteration and preserve `pre_solve_data`, `first_nonfinite_context`, and `first_nonfinite_data`.
- For a finite joint-velocity explosion, trigger on the physical threshold and retain applied control, model parameters, solver state, and replay pre/post history.
- For a post-step pre-reset NaN, inspect the incident post-state and solver snapshot first, then compare history or replay pre/post arrays. Keep solver reset independent so it cannot become a prerequisite for capture.

## Inspect and compare artifacts

Validate an artifact before drawing conclusions:

```bash
./isaaclab.sh -p scripts/tools/physics_debug.py validate physics_incident_....npz
./isaaclab.sh -p scripts/tools/physics_debug.py inspect physics_incident_....npz --json
./isaaclab.sh -p scripts/tools/physics_debug.py diff left.npz right.npz
```

Use `--allowed_status partial` only after reading the manifest error. It does not make an incomplete replay capability complete. Use `replay` only when the archive declares a complete capability and an explicitly trusted adapter is registered.

Analyze in this order:

1. Confirm archive status, checksums, dependencies, runtime provenance, schema fingerprint, failed world IDs, phase, and substep.
2. Read the complete selected, ignored, and unallocated field inventories.
3. Compare `pre__*`, chronological `history__*`, and `incident__*`.
4. Compare `replay__pre__*` and `replay__post__*` when replay is enabled.
5. Correlate active-row metadata rather than unused capacity.
6. Follow causality from control and model state through collision, contacts, constraints, and solver output.

## Preserve strictness when extending

- Add a custom operation provider with exact `bind(solver)`, `snapshot()`, and `close()` semantics when the built-in provider cannot represent a solver.
- Keep snapshots non-`None`, discoverable, stable in shape and dtype, and independently cloned.
- Preserve transactional cleanup and make failed cleanup retryable.
- Add regression coverage that fails without the fix, then passes with it.
- Run focused tests through `./isaaclab.sh -p -m pytest`, rebuild public API documentation with `./isaaclab.sh -d`, and run `./isaaclab.sh -f` before committing.
- Never edit installed MJWarp source, silently omit requested data, infer world ownership from coincidental array shapes, enable pickle loading, or add production `wp.printf`.
