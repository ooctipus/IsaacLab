# Octi multi_task project practices

These practices apply to the Octi-style code under
`source/isaaclab_tasks/isaaclab_tasks/core/multi_task/`. They do **not**
relax the expectations for IsaacLab core modules.

### Compatibility policy

- **Keep public boundaries small and stable.** Public wrappers, config classes,
  manager term signatures, presets, and test-facing APIs should remain stable
  unless the caller set is fully migrated in the same change.
- **Internals may move when the public contract stays intact.** Move code only
  when the new location makes the data layout or backend boundary clearer.
  Backend files and kernel modules are useful; helper modules/classes that just
  hide a few tensor slices or branches are not.
- **Avoid compatibility-by-fallback inside hot or controlled paths.** If Octi
  owns all call sites, prefer a direct required contract over optional fallback
  branches, legacy aliases, or broad defensive checks.
- **Put shims at the boundary when compatibility is actually needed.** If old
  configs or external call sites must keep working, normalize them once at the
  public wrapper/config boundary and keep the hot implementation strict.
- **Do not silently accept unknown states.** Missing required config, bad tensor
  shape, wrong dtype, or unsupported backend should fail clearly near the public
  boundary. Do not let kernels or vectorized code continue with ambiguous data.
- **Do not carry unused generality.** Delete adapters, stale aliases, dead
  properties, cached counters, compatibility branches, diagnostic side
  channels, hooks, and helper classes that are no longer pulling their weight.
  State should have a clear temporal role across calls; otherwise keep it local.
  In Octi multi_task, a short direct block is better than a generic framework
  that obscures the data layout.

### Warp and CUDA practice

- **Use Warp only for GPU hot paths where it materially helps.** Keep ordinary
  CPU/control logic in Python or Torch. Benchmark before and after when changing
  kernels or replacing Torch vectorization.
- **Keep Warp implementations pure Warp.** Backend files such as
  `impl/*_warp.py` should not depend on Torch tensors. The public wrapper should
  own `torch` allocation, `wp.from_torch(...)` conversion, and backend selection.
- **Preallocate graph-critical state in `__init__`.** Scratch buffers, changed-id
  buffers, counters, and sort storage should be fixed-size and caller-sized
  using explicit capacities such as `max_updates`.
- **Do not allocate, sync, or return dynamic tensors in the hot path.** Avoid
  `torch.unique`, `torch.argsort`, `.item()`, `tolist()`, implicit `.reshape()`
  allocations, or Python list construction in update paths intended for CUDA
  graph capture.
- **Use fixed output buffers for dynamic results.** Prefer `(values,
  num_values)` tensors such as `changed_ids` and `num_changed` over returning a
  newly-sized tensor from each call.
- **Be explicit about tensor contracts.** For graphable paths, require
  contiguous tensors, known dtypes, fixed payload shape, and batch sizes within
  `max_updates`. Do not call `.contiguous()` in the hot path to hide a caller
  contract violation.
- **Treat Warp utility calls as suspect until capture-tested.** Some helper
  functions may allocate internally. Verify graph capture with Warp capture APIs
  (`wp.capture_begin`, `wp.capture_end`, `wp.capture_launch`) rather than
  assuming `torch.cuda.CUDAGraph` captures Warp work correctly.
- **Use the right scratch sizes.** For `wpu.radix_sort_pairs`, allocate
  double-width key/value storage when required by Warp. Keep group/count scratch
  in `int32` when values are bounded by the update batch size; keep stream ids
  `int64` when indexing Torch-origin ids.
- **Avoid pathological per-thread loops.** Benchmark duplicate-heavy,
  all-same-id, all-unique, and random-id cases. If grouping sorted ids, use
  bounded or logarithmic searches where a linear scan can explode on skewed
  distributions.
- **Do not leave debug prints in kernels.** `wp.printf` is only for standalone
  reproduction scripts and must be removed from production code.

### What is elegant in Octi multi_task

- **Simple data layout beats abstraction.** Prefer visible tensor layout and
  short direct code over adapter hierarchies, registries, or generalized
  lifecycle objects when the domain is controlled.
- **Let the code read like data moving through tensors.** Keep append cursors,
  masks, indices, rates, counts, and slot tables visible at the use site unless
  they are genuinely reused. Inline small loops and one-use reductions instead
  of hiding them behind generic helpers.
- **Prefer stateless functions and kernels over stateful classes.** If a module
  only groups related tensor operations, expose module-level functions/kernels
  and let the caller own scratch, counters, and buffers. Use a class only when
  it owns necessary cross-call state.
- **Keep temporal state explicit and minimal.** A field is justified when it
  carries information across calls, such as the slot assigned to each env for
  the next success update. Do not keep mirrors of values already owned by a
  monitor, sampler, buffer, or caller.
- **Backend composition is acceptable when it is a real boundary.** A thin
  public wrapper may choose Torch or Warp implementations under `impl/`, but do
  not build a backend abstraction around a few lines of caller-owned tensor
  logic.
- **Configuration should be intentional.** Add config knobs only when they
  represent a real operational choice. Do not add knobs to preserve code paths
  that are no longer desired.
- **Domain-specific policy belongs at the caller/config boundary.** Reusable
  combinators should expose data or a narrow hook, but should not own factory-
  specific visualization panels, logging schemas, estimator paths, or preset
  policy.
- **No hidden diagnostics in hot paths.** Periodic logs, debug counters, sampler
  bin dumps, and visualization side channels should be explicit caller-owned
  behavior. Delete temporary diagnostic helpers once their call sites are gone.
- **Tests should cover semantics and performance-sensitive shapes.** For
  streamer/monitor-style utilities, test Torch/Warp parity, duplicate ids,
  all-same ids, non-bool payloads if supported, fixed external state tensors,
  and graph-capture smoke when relevant.
- **Benchmark scripts may be temporary.** Use standalone scripts or
  `./isaaclab.sh -p -c "..."` for timing, report the key numbers, then remove
  temporary benchmark files from the workspace.
- **No multi_task changelog churn unless requested.** For Octi multi_task
  experimental work in this branch, do not create or update changelog fragments
  unless the user explicitly asks for one.

### Contrast with IsaacLab core

- IsaacLab core code should remain broadly reusable, conservative, documented,
  and backwards-compatible. Do not apply Octi multi_task's reduced-defense,
  controlled-caller style to core packages.
- Core public APIs need deprecation paths and migration guidance. Octi
  multi_task internals may be simplified aggressively when tests and known
  presets are migrated together.
- Core code should favor maintainability for unknown downstream users. Octi
  multi_task may favor performance, graphability, and directness when the caller
  contract is explicit and locally enforced.
