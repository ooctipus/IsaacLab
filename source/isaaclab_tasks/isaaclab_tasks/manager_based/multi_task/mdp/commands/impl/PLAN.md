# Multi-Task Command Dispatch Implementations

This folder is the private implementation boundary for the Warp-backed
multi-task command term. The public IsaacLab command term stays in:

```text
commands/multi_task_command.py       # base command term and factory switch
commands/multi_task_command_warp.py  # public Warp-backed subclass
```

The implementation strategies live here:

```text
commands/impl/
  mega_kernel/             # dense env-slot reference Warp backend
  packed_scatter/          # fused-pipeline-sorted flat queue with legacy scatter
  primitive_graph_local/   # local grouped outputs plus shared primitive nodes
  primitive_queue_local/   # local grouped outputs by primitive-family queues
  schedule_ordered_mega/   # dense env-slot backend with schedule-ordered slots
```

All folders are selectable through `MultiTaskCfg.dispatch_backend`. Future
backend folders should only be added with real execution code and public
benchmark/test coverage.

## Motivation

The current command system is semantic at authoring time:

```text
task -> subtask -> state kernel -> metric -> activation -> command/reward output
```

That is the right user-facing model. A task author should think in terms of
`BODY_POS`, `BODY_QUAT`, `BODY_LIN_VEL`, contact predicates, joint power, and
similar task concepts.

The GPU does not care about those semantic names. Many semantic state kernels
share a smaller set of fused computation pipelines. Those pipelines are still
larger than true primitives such as threshold, reduce, subtract, norm, metric,
and activation:

```text
BODY_POS, BODY_LIN_VEL, BODY_ANG_VEL, OBJECT_POS
  -> direct_vec3_delta
     = gather vec3 -> subtract target vec3 -> L2

JOINT_POS, JOINT_VEL, BODY_POS_Z
  -> direct_scalar_delta
     = gather scalar -> subtract target scalar -> abs

BODY_QUAT
  -> direct_quat_delta
     = gather quat -> quaternion delta -> angle

BODY_CONTACT
  -> vec3_threshold_vector_delta
     = gather contact-force vec3s -> threshold each lane -> vector delta

BODY_CONTACT_COUNT
  -> vec3_threshold_sum_delta
     = gather contact-force vec3s -> threshold each lane -> sum -> scalar delta

BODY_CONTACT_COUNT_DIFF
  -> vec3_threshold_pair_diff_delta
     = gather contact-force vec3s -> threshold -> pair-count diff -> scalar delta

JOINT_MECH_POWER
  -> scalar_sum_delta
     = gather scalar joint lanes -> signed/absolute reduction -> scalar delta
```

The expected scaling regime is:

```text
task instances      ~1M
unique subtasks     ~1k-10k
state kernels       ~100-200
pipeline families   much smaller
true primitives     smaller still, but not good standalone kernel boundaries
```

The design goal is to preserve semantic authoring while lowering runtime work
into fused-pipeline execution where that improves GPU occupancy, branch
coherence, and memory locality. True primitives are the vocabulary used to
describe and lower pipelines; fused pipelines are the actual dispatch boundary
unless a future graph fuses multiple primitive stages without extra global
memory traffic.

## Current Decomposition Reality

The current production state kernels do not yet contain enough repeated
lower-level work for a primitive graph to win everywhere. They split into two
groups:

```text
Mostly terminal kernels:
  JOINT_POS, JOINT_VEL, BODY_POS_Z
    -> scalar read -> subtract -> abs

  BODY_QUAT
    -> quat read -> quat delta -> angle

  BODY_POS, BODY_LIN_VEL, BODY_ANG_VEL
    -> vec3 read -> subtract -> norm

Potential shared primitive producers:
  BODY_CONTACT, BODY_CONTACT_COUNT, BODY_CONTACT_COUNT_DIFF
    -> same contact-force threshold predicate, then different consumers

  JOINT_MECH_POWER
    -> reduction that could be shared if future tasks reuse the same
       joint-power aggregate
```

So `primitive_queue_local` is a **primitive-family** backend: it removes the
large semantic branch tree and writes local grouped outputs, but most kernels
still fuse gather, delta, metric, activation, and output materialization.

`primitive_graph_local` is the first true shared-primitive backend. Today its
strong real sharing is contact predicate reuse plus conservatively gated
producer nodes for direct vec3/scalar/quat and scalar-sum schedules. That is why
it is close to, but not consistently faster than, `primitive_queue_local` on
current locomotion presets. It should become valuable as the task set grows
lower-level reuse: body-frame transforms, shared quaternion inverse/frame reads,
repeated vec3 deltas, contact masks, reductions, power sums, and shared
activation/materialize nodes.

The synthetic benchmark now models that future explicitly by sharing
target-independent producer nodes: current vec3, current quat, reduce8,
reduce32, contact mask, and frame basis. With graph fanout 4 on 1M random work
items, `primitive_graph_local_synth` moved from a contact-only marginal win to
a meaningful shared-IR result:

```text
mega_kernel                  0.3313 ms
primitive_queue_local_synth  0.2119 ms
primitive_graph_local_synth  0.1538 ms
graph_packed_local_synth     0.1186 ms
packed_scatter               0.1416 ms
packed_local                 0.1193 ms
```

Fanout matters:

```text
graph_fanout=1   graph_packed_local_synth 0.2842 ms  # graph overhead with no reuse
graph_fanout=2   graph_packed_local_synth 0.1712 ms
graph_fanout=4   graph_packed_local_synth 0.1186 ms
graph_fanout=8   graph_packed_local_synth 0.0880 ms
graph_fanout=16  graph_packed_local_synth 0.0831 ms
graph_fanout=32  graph_packed_local_synth 0.0801 ms
```

Conclusion: the idea is sound only when the lowering exposes real reusable
producers. A graph of terminal one-consumer nodes is just extra launches and
global memory traffic. The useful target is:

```text
shared producer nodes
  -> parallel consumer materialization
  -> packed/local outputs
```

That shape preserves consumer parallelism and removes the dense scatter cost. In
the synthetic benchmark it reaches `0.1186 ms` at fanout 4 and improves to
`0.0801 ms` at fanout 32.

Production command-level check:

```text
preset=shared_direct, 16384 envs:
primitive_queue_local  0.0547 ms
primitive_graph_local  0.0547 ms

preset=locomotion, 16384 envs:
primitive_queue_local  0.0576 ms
primitive_graph_local  0.0593 ms
```

This is the current line: the backend is structurally ready for producer reuse
and graph capture, but present command presets do not yet expose enough heavy
fanout for a clear production win. The next useful optimization is not "more
nodes"; it is better graph layout for high-fanout heavy producers while keeping
low-fanout direct projections fused.

## Design Philosophy

### Keep The Public Surface Small

`multi_task_command_warp.py` is the public switchboard. It should stay small:

```python
self._backend = build_command_backend(self, cfg.dispatch_backend)

self._backend.dispatch(self, valid_slots)
self._backend.compose(self, valid_slots)
```

New backend experiments should not turn the public class into a large branch
tree. Put strategy-specific code under a strategy folder.

### Split By Implementation Strategy First

The top-level folder boundary is the backend strategy:

```text
mega_kernel/
packed_scatter/
```

Inside a strategy folder, split by phase if needed:

```text
bindings.py
read.py
execute.py
rotation.py
compose.py
```

This makes it obvious what implementation is selected and where future
alternatives should go.

### Do Not Grow The Mega-Kernel Forever

`mega_kernel` is useful and should remain the correctness baseline. It is also
not the architecture we want to grow indefinitely. A large `if/elif` tree over
100-200 semantic kernels becomes hard to reason about and can become expensive
when nearby GPU threads take different branches.

### Make Warp Hot Paths Graphable By Construction

Warp hot paths should avoid:

```text
torch.unique
torch.argsort in Python-managed hot paths
.item()
.tolist()
dynamic tensor allocation
dynamic return sizes
per-step Python list construction
```

Use fixed-size buffers, explicit capacities, stable `wp.array` handles, and
`(values, count)` style device outputs.

## Current Production Backend: `mega_kernel`

`mega_kernel` is the current implementation. It preserves the existing public
output contract:

```text
_buf_error        [num_envs, k_max]
_buf_activation   [num_envs, k_max]
_command_reach    [num_envs, reach_width]
_command_track    [num_envs, track_width]
```

Structure:

```text
mega_kernel/
  bindings.py   # long-lived Torch/Warp views and Warp structs
  read.py       # scene buffer slabs -> unified buffer
  execute.py    # dispatch_mega(env, slot)
  rotation.py   # body-frame command vec3 rotation
  compose.py    # reward/progress/success composition
```

Runtime:

```text
read scene slabs
  -> dispatch_mega over (env, slot)
  -> rotate policy-facing vec3 command slots
  -> compose reward/progress/success
```

The main execute kernel is still:

```python
@wp.kernel
def dispatch_mega(env_slots, spec, state, outputs):
    env, slot = wp.tid()
    if slot >= env_slots.slot_count[env]:
        return

    sid = env_slots.subtask_ids[env, slot]
    skid = spec.state_kernel_id[sid]

    if skid == STATE_BODY_POS or skid == STATE_BODY_LIN_VEL:
        ...
    elif skid == STATE_BODY_QUAT:
        ...
    elif skid == STATE_BODY_CONTACT:
        ...
```

This is simple and has only one execute launch. It is good when work is already
locally homogeneous. It is worse when a warp contains many different state
kinds.

## Alternative Backend 1: `primitive_pipeline_queue`

Shape:

```text
semantic state kernels
  -> fused primitive-pipeline queues
  -> branch-light pipeline kernels
  -> legacy output tensors
```

Example:

```text
queue[direct_vec3_delta]                = work items for BODY_POS/BODY_LIN_VEL/...
queue[direct_scalar_delta]              = work items for JOINT_POS/BODY_POS_Z/...
queue[direct_quat_delta]                = work items for BODY_QUAT
queue[vec3_threshold_vector_delta]      = work items for BODY_CONTACT
queue[vec3_threshold_sum_delta]         = work items for BODY_CONTACT_COUNT
queue[vec3_threshold_pair_diff_delta]   = work items for BODY_CONTACT_COUNT_DIFF
queue[scalar_sum_delta]                 = work items for JOINT_MECH_POWER
```

Sketch:

```python
@wp.kernel
def direct_vec3_delta_kernel(work_ids, state, targets, outputs):
    q = wp.tid()
    i = work_ids[q]
    # No semantic branch. The queue already says this is direct-vec3 work.
    current = read_vec3(state, i)
    target = read_vec3_target(targets, i)
    delta = target - current
    scatter_legacy(outputs, i, delta, wp.length(delta))
```

Why it can win:

```text
fewer divergent branches
one launch per fused pipeline, not per semantic state kernel
legacy output layout preserved
```

Why it can lose:

```text
queue construction has a cost
indexed reads through work_ids reduce memory locality
8 launches can be worse than 1 launch when work is already homogeneous
```

Benchmark result for 1M random work items:

```text
mega_kernel       0.3332 ms
primitive_pipeline_queue   0.2088 ms
```

That is about 1.60x faster than random mega. It is not the best measured path,
but it is the easiest conceptual step after the current backend.

## Alternative Backend 2: `packed_scatter`

Shape:

```text
semantic state kernels
  -> fused-pipeline-sorted flat work queue
  -> one branch-light packed dispatch
  -> scatter into legacy output tensors
```

This keeps the current public output contract but improves input locality.

Sketch:

```python
@wp.kernel
def packed_vec3_scatter_kernel(packed_src, packed_tgt, original_ids, outputs):
    q = wp.tid()
    original = original_ids[q]
    delta = packed_tgt[q] - packed_src[q]
    scatter_legacy(outputs, original, delta, wp.length(delta))
```

Why it can win:

```text
coherent branch regions by fused-pipeline id
simple legacy compatibility boundary
existing command/reward output tensors can remain
```

Why it can lose:

```text
current wired backend packs metadata, not per-step source values
scatter writes still have poor locality
composer still consumes legacy layout
```

Benchmark result for 1M random work items:

```text
mega_kernel      0.3332 ms
packed_scatter   0.1389 ms
```

That is about 2.40x faster than random mega in the synthetic benchmark.

This is the current compatibility bridge. It is useful only if it remains a
stepping stone toward local grouped outputs; it should not become a second
long-term architecture beside `mega_kernel`.

## Alternative Backend 3: `primitive_queue_local`

Shape:

```text
semantic state kernels
  -> fused primitive-pipeline queues
  -> branch-light pipeline kernels
  -> packed local outputs
  -> grouped composer / grouped observation path
```

Sketch:

```python
@wp.kernel
def direct_vec3_delta_local_kernel(packed_src, packed_tgt, packed_outputs):
    q = wp.tid()
    delta = packed_tgt[q] - packed_src[q]
    packed_outputs.delta[q] = delta
    packed_outputs.error[q] = wp.length(delta)
```

Why it can win:

```text
contiguous reads
contiguous writes
no scatter back to legacy layout
downstream kernels can stay grouped
```

Why it is harder:

```text
not drop-in compatible with current public tensors
needs a grouped composer
needs an observation path that can consume packed/grouped outputs
debug/readback paths need a boundary conversion or separate view
```

Benchmark result for 1M random work items:

```text
mega_kernel             0.3332 ms
primitive_queue_local   0.1174 ms  # synthetic benchmark label: packed_local
```

That is about 2.84x faster than random mega. This is the long-term upper-bound
direction, not the first compatibility-preserving migration.

## Theoretical Analysis

A GPU dispatch strategy balances four costs:

```text
launch count
branch divergence
memory locality
queue/order construction
```

### Launch Count

One launch is cheap and simple:

```text
mega_kernel: 1 execute launch
```

One launch per semantic state kernel is usually too many:

```text
kind_queue: up to 64 launches in the benchmark, potentially 100-200 later
```

One launch per fused pipeline is a better compromise:

```text
primitive_pipeline_queue: 8 launches in the benchmark
```

Without graph capture, launch overhead dominates queue-style backends. With
graph capture, launch count matters less, but many tiny kernels are still not
free.

### Branch Divergence

Mega-kernel branch cost depends on local work order:

```text
random heterogeneous work -> bad branch locality
grouped homogeneous work  -> good branch locality
```

This is why the benchmark showed:

```text
1M random:
  mega_kernel 0.3332 ms

1M already grouped by kind:
  mega_kernel 0.0940 ms
```

If work is naturally grouped, a mega-kernel is hard to beat.

### Memory Locality

Pipeline queues remove semantic branches but still gather through `work_ids`.
Packed layouts improve memory coalescing:

```text
primitive_pipeline_queue  -> indexed reads and legacy scatter writes
packed_scatter            -> sorted metadata and legacy scatter writes
primitive_queue_local     -> contiguous/local reads and contiguous writes
```

The benchmark ranking followed this locality model:

```text
primitive_pipeline_queue  0.2088 ms
packed_scatter            0.1389 ms
primitive_queue_local     0.1174 ms  # synthetic label: packed_local
```

### Queue/Order Construction

Queue construction can dominate dispatch if rebuilt every step.

Measured 1M random work items:

```text
64 torch nonzero queues      2.0763 ms
8 torch nonzero queues       0.3320 ms
Warp primitive atomic queue  0.1428 ms
Warp radix sort              0.1024 ms
```

So:

```text
stable assignment:
  precompute queues/sorted ids at resample time

rapidly changing assignment:
  use graphable sort/segment or atomic queue construction

already grouped layout:
  avoid building queues and use the grouped order directly
```

## Recommended Migration Path

### Step 1: Keep `mega_kernel` As The Correctness Baseline

Do not delete the current backend. It is the simplest implementation and the
best baseline for tests and regressions.

### Step 2: Add `schedule_ordered_mega`

The Warp/Newton study changes the next implementation target. Before building
more queue machinery, test the cheaper idea: keep one dispatch launch, but make
native `(env, slot)` work order schedule-coherent.

Shape:

```text
task/subtask authoring order
  -> TaskSpec lowering with schedule ids
  -> env-local slot order grouped by fused schedule
  -> one branchy dispatch launch with much better local branch coherence
  -> existing dense output store
```

This backend answers the most important question from the synthetic benchmark:

```text
Can a better native slot layout recover most of the queue win without queue
construction, indexed reads, or multiple execute launches?
```

### Step 3: Keep `packed_scatter` As The Fused-Schedule-Sorted Bridge

Compile semantic subtasks into fused-pipeline queues. Preserve legacy output
tensors first. The wired `packed_scatter` backend uses a flat queue sorted by
pipeline family so one execute launch can run exact work count with coherent
branch regions.

This replaces only the execute phase:

```text
read.py from mega_kernel
packed_scatter execute
rotation.py from mega_kernel
compose.py from mega_kernel
```

### Step 4: Add Packed Schedule Inputs

The current `packed_scatter` backend packs work metadata, not per-step primitive
input values. The next locality step is to build packed pipeline input buffers
and keep legacy output tensors, then measure whether the extra packing bandwidth
pays off on real command data.

### Step 5: Implement `primitive_queue_local`

Only do this when composer and observation consumers can stay grouped.

This is the most invasive path, but it is the one that best matches future
large heterogeneous workloads.

## Repository-Study Execution Plan

This is the detailed plan derived from the Warp/Newton study. It is deliberately
acceptance-driven: if a backend does not satisfy its criteria, delete it or keep
it as an explicitly named benchmark artifact, not as production architecture.

### Terminology Contract

Use these names consistently:

| Term | Meaning |
|---|---|
| Semantic state kernel | User/task-facing state kind such as `BODY_POS`, `BODY_QUAT`, `BODY_CONTACT_COUNT_DIFF`. |
| Primitive op | Small reusable `@wp.func` operation: gather, threshold, reduce, delta, metric, activation, scatter/materialize. |
| Fused schedule | Kernel-level primitive DAG such as `direct_vec3_delta`, `direct_quat_delta`, `contact_pair_count_delta`. |
| Backend layout | Concrete storage/execution shape: dense `(env, slot)`, schedule-ordered dense, flat sorted work ids, or local grouped outputs. |

Acceptance criteria:

- No backend names a fused schedule a "primitive".
- Primitive ops live as `@wp.func` helpers, not public kernels.
- Fused schedules are the smallest production launch boundary unless a
  benchmark proves a different boundary is faster.

### Phase 0: Lock The Backend Contract

Goal: make backend ownership explicit before adding more implementations.

Required work:

- Keep public selection through `MultiTaskCfg.dispatch_backend`.
- Keep `MultiTaskCommand` / `MultiTaskCommandWarp` as thin public wrappers.
- Make backend-owned output stores the only place that knows whether data is
  dense, sorted, queued, or local grouped.
- Keep unsupported backend/state-kernel combinations as construction-time
  errors. Do not put fallback branches in hot kernels.

Acceptance criteria:

- `dispatch_backend` supports `reference`, `mega_kernel`, and every new backend
  by string through the public command config.
- A new backend can be selected in the public benchmark without importing its
  private kernels directly.
- No new backend contains `NotImplementedError`, silent fallback to
  `mega_kernel`, or broad "try one layout then another" hot-path logic.
- Current public tests still construct commands through the public command
  class, not private backend classes.

### Phase 1: Build Schedule Lowering

Goal: create one explicit semantic-to-schedule lowering table shared by all
Warp backends.

Required work:

- Define schedule ids for the current state-kernel set.
- Define a small `@wp.func` primitive vocabulary for current schedules.
- Lower semantic subtasks to schedule ids at construction.
- Expose enough plan metadata for both dense and queued backends:

```text
subtask_id -> schedule_id
subtask_id -> source gather metadata
subtask_id -> target offset/stride
subtask_id -> canonical output offset
subtask_id -> activation spec
```

Acceptance criteria:

- Every current preset state kernel has exactly one schedule lowering.
- Unsupported semantic kernels fail during backend plan construction with a
  clear message naming the missing state kernel.
- Lowering is tested against all current presets:
  `simple_pos_vel`, `pose_vel`, `velocity`, `position`, `locomotion`.
- Schedule metadata is stable after construction and requires no per-step Python
  list rebuild.

### Phase 2: Implement `schedule_ordered_mega`

Goal: test "layout first" before adding queue-local complexity.

Required work:

- Add `commands/impl/schedule_ordered_mega/`.
- Reorder env-local slots by fused schedule at task assignment/build time.
- Keep one execute launch and the dense output store.
- Keep semantics identical to `mega_kernel`.
- Add benchmark output for `schedule_ordered_mega`.

Acceptance criteria:

- Public config string: `dispatch_backend="schedule_ordered_mega"`.
- Tests prove numerical parity with `reference` and `mega_kernel` for all
  current presets.
- Graph-capture smoke passes after warmup and replay with mutated input buffers.
- Benchmark reports:
  - current presets at 16k envs,
  - current presets at 131k envs where feasible,
  - synthetic high-diversity random/skew/grouped assignment.
- Keep threshold:
  - no worse than `mega_kernel` by more than 5% on current real presets,
  - at least 1.20x faster than `mega_kernel` on the 1M random high-diversity
    synthetic schedule benchmark, or the result proves native ordering already
    makes `mega_kernel` equivalent and the backend should be deleted.

Implementation result:

```text
Current public presets, 16k envs:
  simple_pos_vel: mega 0.1428 ms, schedule_ordered 0.1397 ms
  pose_vel:       mega 0.1735 ms, schedule_ordered 0.1747 ms
  velocity:       mega 0.1360 ms, schedule_ordered 0.1354 ms
  position:       mega 0.1252 ms, schedule_ordered 0.1207 ms
  locomotion:     mega 0.1780 ms, schedule_ordered 0.1724 ms

Locomotion, 131k envs:
  mega 0.4456 ms, schedule_ordered 0.4455 ms

Synthetic 1M random high-diversity:
  slots_per_env=8:  mega 0.3305 ms, schedule_ordered 0.3071 ms  (1.08x)
  slots_per_env=32: mega 0.6147 ms, schedule_ordered 0.2044 ms  (3.01x)
```

Decision: keep `schedule_ordered_mega` as a non-default layout experiment. It
meets the current-preset non-regression gate and proves the layout idea for
wider local slot rows, but it does **not** replace queue/local work for the
future 1M random, `k=8` regime.

### Phase 3: Make The Existing Warp Read Path Graph-Clean

Goal: remove Python reader churn from the graphable hot path before judging new
execute backends.

Required work:

- Prebind stable scene-backed slab source handles in backend construction.
- Keep dynamic reader patching for reference/tests at the boundary, not inside
  production Warp hot paths.
- Move computed slabs such as joint mechanical power into explicit Warp fill
  kernels or backend-owned buffers.
- Keep debug visualization and non-stable Torch computations outside captured
  backend stages.

Acceptance criteria:

- No per-step `wp.from_torch(...)` for stable slab handles in the production
  Warp hot path.
- No per-step Torch compute for slab values required by captured dispatch.
- Graph capture replays with changed source tensor contents and produces changed
  outputs.
- Slab benchmark reports dynamic-reader, prebound-eager, and prebound-captured
  timings.
- The change does not alter reference-path test mocking behavior.

### Phase 4: Re-evaluate `packed_scatter`

Goal: decide whether `packed_scatter` is a useful bridge or dead weight.

Required work:

- Benchmark `packed_scatter` against `mega_kernel` and
  `schedule_ordered_mega` after Phase 2/3.
- Report dispatch-only and full read/execute/rotate/compose timings.
- Keep dense output materialization cost separate from schedule dispatch cost.

Acceptance criteria:

- Public benchmark includes `reference`, `mega_kernel`,
  `schedule_ordered_mega`, and `packed_scatter`.
- If `packed_scatter` is not at least 1.10x faster than the best dense backend
  on high-diversity random workloads, keep it only as a benchmark artifact or
  delete it.
- If it wins dispatch-only but loses full pipeline due to scatter/materialize
  costs, document that explicitly and do not treat it as the long-term backend.

### Phase 5: Implement `primitive_queue_local`

Goal: test the true grouped/local upper-bound path.

Required work:

- Add `commands/impl/primitive_queue_local/`.
- Build schedule queues at task assignment/resample time when assignments are
  stable.
- Use backend-local grouped output buffers for error, activation, delta, and
  command materialization.
- Compose reward/progress from grouped local outputs without first scattering
  to dense `(env, slot)` tensors.
- Materialize dense command tensors only at an explicit observation/debug
  boundary.

Acceptance criteria:

- Public config string: `dispatch_backend="primitive_queue_local"`.
- Tests prove parity with reference outputs:
  reward, done/progress flags, `command`, `command_reach`, `command_track`, and
  body-frame command rotation.
- Tests cover all-same, random, skewed, grouped, empty/padded slots, and current
  terrain presets.
- Graph capture passes with fixed capacities and stable pointers.
- Queue construction is not performed every physics step for stable task
  assignments.
- Benchmark reports queue-build/resample cost separately from per-step cost.
- Keep threshold:
  - at least 1.20x faster than the best dense backend on the 1M random
    high-diversity benchmark when dense materialization is not required,
  - no worse than 10% on current real presets, or documented as future-scale
    only and excluded from default configs.

### Phase 6: Optional Graphable Queue Builder

Goal: support workloads where assignment really changes every step.

Required work:

- Prototype graphable sort/segment and tile-compaction queue builders.
- Compare against atomic queues under random, skewed, grouped, and all-same
  schedule ids.
- Keep queue buffers fixed-capacity and report overflow as construction-time or
  boundary-time failure, not dynamic resize.

Acceptance criteria:

- Queue builder is capture-tested with `wp.ScopedCapture`.
- No `.item()`, `.tolist()`, `torch.unique`, `torch.argsort`, or dynamic tensor
  return in the hot path.
- For 1M work items, builder plus dispatch beats the best no-queue backend on
  the target changing-assignment workload. Otherwise keep it as a benchmark
  note, not production code.

## Definition Of Done For Any New Backend

Every backend added after `mega_kernel` must satisfy all items below before it
is considered production, not just "wired".

### Public Interface

- Selectable by `MultiTaskCfg.dispatch_backend`.
- Exercised by `MultiTaskCommand(cfg, env)` or `MultiTaskCommandWarp(cfg, env)`,
  never by direct private-kernel benchmark only.
- No private caller-specific compatibility path in the implementation.

### Correctness

- Matches `reference` for current presets.
- Matches `mega_kernel` for Warp-specific output layout where applicable.
- Preserves body-frame command semantics.
- Handles padded slots without leaking output lanes.
- Fails clearly at construction for unsupported state kernels or capacity
  violations.

### Graphability

- Fixed launch topology for a fixed backend plan.
- Scratch and output buffers allocated in `__init__` or plan construction.
- Stable `wp.array` handles for captured stages.
- Warmup-before-capture smoke test.
- Replay-after-input-mutation smoke test.
- No dynamic Python/GPU-data branching in the captured hot path.

### Performance

Benchmark output must include:

```text
reference
mega_kernel
schedule_ordered_mega
packed_scatter
primitive_queue_local  # once implemented
```

Benchmark rows must separate:

```text
queue/order build
read/slab fill
execute
rotation
compose
dense materialization
full step
launch count
```

Required benchmark scenarios:

```text
current terrain presets at 16384 envs
current terrain presets at larger env count where feasible
synthetic 1M high-diversity random assignment
synthetic 1M skewed assignment
synthetic 1M grouped assignment
changing-assignment queue-builder stress, if queue building is part of the backend
wide command materialization stress, e.g. width 256 and 1024
```

### Cleanup Gate

- A backend that does not meet its keep threshold is removed or renamed as a
  benchmark-only experiment.
- `packed_scatter` cannot become a permanent second dense architecture unless
  full-pipeline benchmarks justify it.
- Temporary benchmark scripts can stay only if they are intentionally part of
  the study artifacts; otherwise remove them before commit.

## Implementation Contract For Future Backends

A backend should expose a small set of phase functions. A full backend can look
like:

```python
plan = build_backend_plan(command)

read_backend(command, plan)
execute_backend(command, plan)
rotate_backend(command, plan)
compose_backend(command, plan)
```

If a backend only replaces execute, reuse the current phases:

```python
from commands.impl.mega_kernel import (
    compose_warp,
    fill_unified_buffer_warp,
    rotate_canonical_slots_to_body_frame_warp,
)
from commands.impl.some_backend import build_some_backend_plan, dispatch_some_backend_warp

plan = build_some_backend_plan(command)
fill_unified_buffer_warp(command, plan)
dispatch_some_backend_warp(command, plan)
rotate_canonical_slots_to_body_frame_warp(command, plan)
compose_warp(command, plan)
```

Keep backend selection explicit. Do not add hidden fallbacks inside hot kernels.
If a backend cannot satisfy its tensor/layout contract, fail at construction.

## Testing Requirements

Every new backend should pass:

```text
test_multi_task_command_mock.py
test_multi_task_warp_equivalence.py
```

It should also include backend-specific tests for:

```text
all-same task assignment
random task assignment
skewed task assignment
grouped task assignment
empty/padded slots
body-frame rotated command slots
CUDA graph capture smoke
```

Benchmarks should report:

```text
number of launches
dispatch-only time
queue/order build time
full read -> execute -> rotate -> compose time
graph-captured and non-captured timing when relevant
```

## Practical Rule Of Thumb

Use this decision table:

| Workload shape | Best candidate |
|---|---|
| Small current presets | `mega_kernel` |
| Already grouped by state kind | `mega_kernel` |
| Random heterogeneous, stable assignment | `indexed_mega` or `packed_scatter` |
| Random heterogeneous, changing assignment | graphable sort + indexed dispatch |
| Many semantic kernels sharing few fused pipelines | `primitive_pipeline_queue` |
| Need legacy tensors preserved | `packed_scatter` |
| Composer/obs can consume grouped layout | `primitive_queue_local` |

The main idea is simple:

```text
semantic for authoring
primitive vocabulary for lowering
fused pipelines for GPU execution
packed/grouped for long-term throughput
```
