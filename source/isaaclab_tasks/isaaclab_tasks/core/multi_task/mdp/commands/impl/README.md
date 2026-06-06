# Multi-Task Command Dispatch Implementations

Private implementation boundary for the multi-task command term. The public
surface lives one level up:

```text
commands/__init__.py            # exports MultiTaskCfg, MinMaxSampler, MultiTaskCommand
commands/multi_task_command.py  # base command term and Torch/Warp factory switch
```

Backends, the Warp/Torch subclasses, the config classes, and the Warp and
Torch kernel registries all live under ``impl/`` so the public surface stays
just the package-level re-exports and the factory in ``multi_task_command``.

Backends are selectable via ``MultiTaskCfg.dispatch_backend``. Each subfolder
owns the full ``read -> execute -> rotate -> compose`` pipeline for one
strategy; the only pieces shared across backends are the kernel registries,
schedule lowering, and compose-kernel selection.

## Folder Layout

```text
impl/
  multi_task_cfg.py            # MultiTaskCfg, MinMaxSampler (config dataclasses)
  multi_task_command_torch.py  # PyTorch reference subclass (dispatch_backend="torch")
  multi_task_command_warp.py   # Warp switchboard subclass
  backend.py                   # Warp backend factory and CommandBackend protocol
  schedules.py                 # static schedule plan helpers
  compose_select.py            # adaptive switch between scalar and parallel composers
  kernel_ids.py                # kernel ID enums shared by Torch and Warp paths
  kernels_torch.py             # Torch state/delta/metric/activation kernel registry
  kernels_wp.py                # Warp kernels (dispatch, compose, rotate, slab fills)
  kernels_viz.py               # debug visualization: VIZ_REGISTRY + per-step update fns
  mega_kernel/                 # dense (env, slot) baseline; default for small k_max.
                               # Supports ``slot_order="schedule"`` mode, exposed
                               # publicly as ``dispatch_backend="schedule_ordered_mega"``.
  packed_scatter/              # fused-pipeline-sorted queue with legacy scatter
  primitive_queue_local/       # per-primitive-family queues with local grouped outputs
  primitive_graph_local/       # dense-kernel signature-graph with always-materialize
                               # producer launches
```

All hot paths are CUDA-graph captured. Plan construction allocates fixed-size
scratch in ``__init__`` so capture replay is safe.

## Backend Roles

- ``mega_kernel`` — one fused dispatch over ``(env, slot)`` that branches on
  ``state_kernel_id``. Default for small ``k_max`` where launch fusion beats
  branch divergence. The ``slot_order="schedule"`` mode (selected publicly via
  ``dispatch_backend="schedule_ordered_mega"``) sorts slot tables on resample so
  each warp sees a coherent state-kernel region — useful when per-env
  assignment is heterogeneous but stable.
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

Each backend owns its own ``bindings.py`` / ``read.py`` / ``execute.py`` /
``rotation.py`` / ``compose.py``. Cross-backend imports of phase
implementations are not allowed — if two backends would share the same
phase code, duplicate it. The agreement boundary is ``MultiTaskCommandWarp``;
beyond that each backend describes its own data management.

Backend selection is explicit at construction. Do not add hidden fallbacks
inside hot kernels; if a backend cannot satisfy its tensor/layout contract,
fail at plan-build time.

## Definition Of Done For A New Backend

### Public Interface

- Selectable by ``MultiTaskCfg.dispatch_backend``.
- Exercised by ``MultiTaskCommandWarp(cfg, env)``, not just a private bench.
- No caller-specific compatibility branches inside the backend.

### Correctness

- Matches the ``"torch"`` reference backend across the active presets.
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
