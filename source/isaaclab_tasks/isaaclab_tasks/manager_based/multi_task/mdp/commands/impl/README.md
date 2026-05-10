# Multi-Task Command Dispatch Implementations

Private implementation boundary for the Warp-backed multi-task command term.
The public surface lives one level up:

```text
commands/multi_task_command.py       # base command term and factory switch
commands/multi_task_command_warp.py  # public Warp-backed subclass
```

Backends are selectable via ``MultiTaskCfg.dispatch_backend``. Each subfolder
owns the full ``read -> execute -> rotate -> compose`` pipeline for one
strategy; the shared pieces (output buffers, compose-kernel selection,
schedule plans) sit at this level.

## Folder Layout

```text
impl/
  backend.py              # backend factory and shared phase glue
  outputs.py              # shared output buffer struct
  schedules.py            # static schedule plan helpers
  compose_select.py       # adaptive switch between scalar and parallel composers
  mega_kernel/            # dense (env, slot) baseline; default for small k_max
  schedule_ordered_mega/  # dense backend with schedule-ordered slots
  packed_scatter/         # fused-pipeline-sorted queue with legacy scatter
  primitive_queue_local/  # per-primitive-family queues with local grouped outputs
  primitive_graph_local/  # local grouped outputs with shared primitive nodes
```

All hot paths are CUDA-graph captured. Plan construction allocates fixed-size
scratch in ``__init__`` so capture replay is safe.

## Backend Roles

- ``mega_kernel`` — one fused dispatch over ``(env, slot)`` that branches on
  ``state_kernel_id``. Default for small ``k_max`` where launch fusion beats
  branch divergence. Owns the canonical ``read``, ``rotate``, and ``compose``
  phases that other backends reuse.
- ``schedule_ordered_mega`` — same dense shape as ``mega_kernel`` with slots
  re-ordered so each warp sees a coherent state-kernel region. Useful when the
  per-env assignment is heterogeneous but stable.
- ``packed_scatter`` — fused-pipeline-sorted flat queue feeding a branch-light
  packed dispatch that scatters into the legacy output tensors. Compatibility
  bridge; preserves the public output layout.
- ``primitive_queue_local`` — per-primitive-family queues feeding grouped
  outputs. Better cache behaviour when work is heterogeneous and downstream
  consumers can read grouped layouts.
- ``primitive_graph_local`` — local grouped outputs with shared primitive
  producer nodes. Wins as fanout grows (multiple subtasks consuming the same
  body-frame producer) and is the production winner for high-fanout presets
  after the dispatch+compose fusion pass.

## Phase Composition

A backend exposes the four phase entry points:

```python
plan = build_backend_plan(command)

read_backend(command, plan)
execute_backend(command, plan)
rotate_backend(command, plan)
compose_backend(command, plan)
```

If a backend only replaces ``execute``, it should reuse the canonical phases
exported from ``mega_kernel``:

```python
from ..mega_kernel import (
    compose_warp,
    fill_unified_buffer_warp,
    rotate_canonical_slots_to_body_frame_warp,
)
```

Backend selection is explicit at construction. Do not add hidden fallbacks
inside hot kernels; if a backend cannot satisfy its tensor/layout contract,
fail at plan-build time.

## Definition Of Done For A New Backend

### Public Interface

- Selectable by ``MultiTaskCfg.dispatch_backend``.
- Exercised by ``MultiTaskCommandWarp(cfg, env)``, not just a private bench.
- No caller-specific compatibility branches inside the backend.

### Correctness

- Matches the ``reference`` (Torch) backend across the active presets.
- Matches ``mega_kernel`` for the Warp output layout where applicable.
- Preserves body-frame command semantics.
- Handles padded slots without leaking output lanes.
- Fails clearly at construction for unsupported state kernels.

### Graphability

- Fixed launch topology for a fixed backend plan.
- Scratch and output buffers allocated in ``__init__`` or plan construction.
- Stable ``wp.array`` handles for every captured stage.
- Warmup-before-capture smoke and replay-after-input-mutation smoke.
- No Python or GPU-data branching inside the captured hot path.

### Performance

Benchmark output should report per-phase timing (read, execute, rotate,
compose, full step) at the active preset scale and at the synthetic
high-diversity scale. Use ``benchmark/bench_tile_fusion_testbed.py`` and
``benchmark/bench_dispatch_homogeneity.py`` as the empirical reference.

### Cleanup Gate

A backend that does not meet its keep threshold is removed or downgraded to a
benchmark-only experiment. Temporary benchmark scripts may stay only when they
are intentionally part of the study artifacts.

## Testing

Every backend must pass:

```text
test_multi_task_command_mock.py
test_multi_task_warp_equivalence.py
```

Backend-specific tests should cover all-same / random / skewed / grouped
assignments, padded slots, body-frame rotated command slots, and a
CUDA-graph-capture smoke.
