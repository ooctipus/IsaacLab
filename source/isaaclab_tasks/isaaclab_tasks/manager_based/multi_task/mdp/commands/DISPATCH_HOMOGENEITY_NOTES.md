# Dispatch Homogeneity Notes

Exploratory notes for replacing the current heterogeneous `dispatch_mega`
shape with a more GPU-friendly execution plan.

## Question

The current Warp command path uses one mega-kernel over `(env, slot)` and
branches on `state_kernel_id`. This is good for launch-count reduction, but it
does not scale cleanly if the command system grows from a small preset to:

```text
task instances      ~1M
unique subtasks     ~1k-10k
state kernels       ~100-200
primitive families  much smaller
```

Hypothesis: the useful GPU workload is not "one dynamic kernel per task slot";
it is "many task slots sharing a small number of fused primitive pipelines."
The pipelines are built from smaller true primitives such as gather, threshold,
reduce, subtract, metric, and activation, but the fused pipeline is the better
kernel boundary.

## Synthetic Benchmark

Artifact:

```bash
./isaaclab.sh -p -m isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_dispatch_homogeneity
```

The benchmark compares:

| Variant | Shape |
|---|---|
| `mega` | One launch, one thread per work item, 64-way branch tree. |
| `kind_queue` | One queue per synthetic state kernel. Branch-free, but up to 64 launches. |
| `primitive_queue_local_synth` | Synthetic proxy for production queue-local dispatch. |
| `primitive_graph_local_synth` | Synthetic proxy for production graph-local dispatch. |
| `graph_packed_local_synth` | Future graph-local target with packed local outputs. |
| `packed_scatter` | Eight fused-pipeline queues with packed contiguous inputs, scattered outputs. |
| `packed_local` | Eight fused-pipeline queues with packed inputs and packed outputs. |
| `idx_mega_primitive` | One branchy launch over work ids sorted by pipeline family. |
| `idx_mega_kind` | One branchy launch over work ids sorted by exact state-kernel kind. |
| `sort_idx_primitive` | Graph-captured primitive sort followed by indexed mega. |
| `sort_idx_kind` | Graph-captured exact-kind sort followed by indexed mega. |

The expanded synthetic state kernels are intentionally closer to the expected
future command mix:

| Fused Pipeline | Shape |
|---|---|
| `vec3_delta_l2` | Position/velocity style 3D delta and L2 error. |
| `scalar_delta_abs` | Joint/scalar command delta and absolute error. |
| `quat_delta_angle` | Quaternion delta and angle error. |
| `reduce8` | Small reduction such as compact contact or power terms. |
| `reduce32` | Larger joint/mechanical-power style reduction. |
| `contact_any16` | Predicate over contact-like lanes. |
| `contact_diff16` | Left/right contact-count difference. |
| `local_frame_vec3` | Vec3 error rotated by a quaternion frame. |

All timings below were CUDA graph captured on an RTX 5090 with Warp 1.12.0 and
Torch 2.10.0+cu128. Lower is better.

### 1M Work Items, 64 Synthetic Kernels

Randomly interleaved work:

| Variant | Time |
|---|---:|
| `mega` | 0.3332 ms |
| `idx_mega_primitive` | 0.1520 ms |
| `idx_mega_kind` | 0.1939 ms |
| `sort_idx_primitive` | 0.2812 ms |
| `sort_idx_kind` | 0.3242 ms |
| `kind_queue` | 0.3696 ms |
| `primitive_queue_local_synth` | 0.2088 ms |
| `primitive_graph_local_synth` | 0.1556 ms |
| `graph_packed_local_synth` | 0.1192 ms |
| `packed_scatter` | 0.1389 ms |
| `packed_local` | 0.1174 ms |

Skewed work distribution:

| Variant | Time |
|---|---:|
| `mega` | 0.1868 ms |
| `idx_mega_primitive` | 0.1388 ms |
| `idx_mega_kind` | 0.1389 ms |
| `sort_idx_primitive` | 0.2622 ms |
| `sort_idx_kind` | 0.2746 ms |
| `kind_queue` | 0.3864 ms |
| `primitive_queue_local_synth` | 0.1419 ms |
| `packed_scatter` | 0.1062 ms |
| `packed_local` | 0.0959 ms |

Already grouped-by-kind work:

| Variant | Time |
|---|---:|
| `mega` | 0.0940 ms |
| `kind_queue` | 0.2917 ms |
| `primitive_queue_local_synth` | 0.1356 ms |
| `packed_scatter` | 0.1298 ms |
| `packed_local` | 0.1257 ms |

Interpretation:

- Queueing only by exact kernel id is not attractive by itself. Even under graph
  capture, too many tiny launches lose.
- Primitive queues help when work is randomly interleaved or skewed because they
  remove branch divergence without exploding launch count.
- Shared primitive graphs help only when the lowering exposes real fanout.
  Sharing target-independent producers such as current vec3, current quat,
  reductions, contact masks, and frame bases moved the random 64-kernel case
  from `primitive_queue_local_synth` at `0.2119 ms` to
  `primitive_graph_local_synth` at `0.1538 ms`.
- Combining shared producers with packed/local consumer outputs is the useful
  graph direction: `graph_packed_local_synth` reached `0.1186 ms` at fanout 4,
  slightly faster than `packed_local` while preserving consumer parallelism.
- If work is already grouped by kind, the mega-kernel is excellent: warps are
  already homogeneous and it has one launch. This is a useful warning.
- Packed input layout matters. `packed_scatter` is consistently better than
  indexed fused-pipeline queues for random/skewed work.
- `packed_local` is the real upper bound: once downstream consumers can stay in
  grouped layout, the expanded random workload is roughly 3x faster than
  random mega.
- Indexed sorted-mega is a serious alternative: it fixes branch locality while
  preserving one launch and the original output layout. It still needs a sorted
  work-id list, so queue/order construction cost decides whether it works every
  step or only when assignment is stable.
- Sorting and indexed mega in the same graph is only worthwhile for random
  high-divergence work. For skewed work, dynamic sorting loses to the unsorted
  mega-kernel; prebuilt sorted ids still win.

### Active-Kernel Sweep, 1M Random Work Items

| Active kernels | Active pipelines | `mega` | `primitive_queue_local_synth` | `packed_scatter` | `packed_local` |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.0936 ms | 0.0630 ms | 0.0553 ms | 0.0588 ms |
| 16 | 2 | 0.1141 ms | 0.1082 ms | 0.0809 ms | 0.0586 ms |
| 32 | 4 | 0.1483 ms | 0.1075 ms | 0.0812 ms | 0.0654 ms |
| 64 | 8 | 0.3544 ms | 0.2240 ms | 0.1474 ms | 0.1200 ms |

Interpretation:

- Divergence cost grows with active branch count.
- Primitive queue benefit grows as semantic state-kernel diversity rises, but
  the best result comes when queues also improve data layout.
- The 64-kernel jump is much larger than the 32-kernel result because the
  branch tree now includes heavier reductions, predicate loops, and a
  local-frame transform path. This is a more useful stress case for future
  command kernels than the original four-light-primitive toy benchmark.

### Shared-Primitive Graph Fanout

The synthetic `primitive_graph_local_synth` path now materializes shared
producer nodes for:

```text
current vec3
current quat
reduce8
reduce32
contact predicate
frame basis
```

The benchmark intentionally rewrites selected source/producer input rows so all
consumers of a node share the same current feature while keeping target values
target-specific. This isolates the question we care about: if future task
lowering exposes target-independent shared producers, does the graph shape pay
for itself?

1M random work items, 64 state kernels:

| Graph fanout | `primitive_queue_local_synth` | `primitive_graph_local_synth` | `graph_packed_local_synth` | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 0.2179 ms | 0.3254 ms | 0.2842 ms | No reuse; graph overhead dominates. |
| 2 | 0.2104 ms | 0.2071 ms | 0.1712 ms | Local output helps, but reuse is still weak. |
| 4 | 0.2119 ms | 0.1538 ms | 0.1186 ms | Shared producers plus local output beats `packed_local`. |
| 8 | 0.2094 ms | 0.1230 ms | 0.0880 ms | Reuse is strong and consumers remain parallel. |
| 16 | 0.2118 ms | 0.1177 ms | 0.0831 ms | Higher fanout keeps improving. |
| 32 | 0.2117 ms | 0.1149 ms | 0.0801 ms | Best measured path in this synthetic workload. |

Interpretation:

- Going "one level down" is useful only when the lower level creates actual
  fanout. Decomposing terminal one-consumer work makes the graph slower.
- The graph variant that matters is shared producers with parallel consumer
  materialization into packed local outputs. This preserves the parallel width
  of the graph and removes the dense scatter cost.
- The current production command kernels have strong sharing in contact
  predicate work; broader wins require future state kernels that reuse frame
  bases, reductions, contact masks, current vec3 features, current quats, or
  other target-independent producer nodes.
- `primitive_graph_local` should be treated as the architecture for reusable
  primitive producers, not as a blanket replacement for every fused schedule.
- The next production-shaped target is therefore concrete: shared producer
  nodes, parallel consumer materialization, and packed/local output layout.

Production follow-up:

- `primitive_graph_local` now has planner support for direct vec3/scalar/quat
  producer nodes and scalar-sum producer nodes, not only contact predicates.
- The production planner is conservative. Cheap direct projections remain fused
  unless fanout is very high; scalar-sum can materialize a producer row and feed
  parallel consumers.
- Command-level benchmarks through `MultiTaskCommand` show this is currently a
  safe/parity architecture, not yet a broad win. On the `shared_direct` mock
  workload at 16k envs, `primitive_queue_local` and `primitive_graph_local`
  both measured about `0.055 ms/update`; on the locomotion preset,
  `primitive_graph_local` was about `0.059 ms/update` vs.
  `primitive_queue_local` at about `0.058 ms/update`.
- The lesson is important: materializing a producer is not enough. The producer
  must be heavy enough and fanout must be high enough, or the extra graph
  memory traffic cancels the saved projection work.
- Producer-side launch fusion is useful. The dense graph path now computes
  direct vec3/scalar/quat, scalar-sum, and contact producer rows in one
  signature-grouped Warp launch instead of five tiny launches. On
  `future_synthetic_heavy_interleaved`, the producer phase dropped from about
  `0.093 ms` to about `0.039 ms`, and command-level graph dispatch improved to
  about `0.140 ms`.
- Consumer-side split experiments were rejected. Per-schedule dense consumers
  and precomputed per-slot schedule/node ids both made dispatch slower; the
  dense graph consumer remains one `(env, slot)` launch until compose/output
  layout changes give it a better contract.

### 16k Env Scale

For 131k work items, roughly `16384 envs * 8 slots`:

| Variant | Time |
|---|---:|
| `mega` | 0.0452 ms |
| `idx_mega_primitive` | 0.0164 ms |
| `idx_mega_kind` | 0.0205 ms |
| `sort_idx_primitive` | 0.0661 ms |
| `sort_idx_kind` | 0.0662 ms |
| `primitive_queue_local_synth` | 0.0329 ms |
| `packed_scatter` | 0.0308 ms |
| `packed_local` | 0.0267 ms |

Interpretation:

- At today's likely per-step command scale, the richer workload already shows a
  small primitive-queue win, but exact-kernel queueing is still much worse. The
  refactor should target primitive homogeneity, not "one queue per semantic
  state kernel."
- The queue design becomes interesting when command/task count grows or when we
  can keep packed grouped layout through composition.

### Queue Build Cost

Artifacts:

```bash
./isaaclab.sh -p /tmp/octi_queue_build_timing.py
./isaaclab.sh -p -m isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_dispatch_queue_build
```

Standalone Torch queue construction for 1M random work items on the same GPU:

| Operation | Time |
|---|---:|
| 64 `torch.nonzero(kind == i)` queues | 2.0763 ms |
| 8 primitive `torch.nonzero(primitive == i)` queues | 0.3320 ms |
| `torch.argsort(kind)` | 0.0798 ms |
| `torch.bincount(kind)` | 0.0516 ms |

Interpretation:

- Rebuilding exact-kernel queues in Python/Torch every step is a non-starter;
  queue construction would dominate the dispatch work.
- Primitive queues are much cheaper but still too expensive for a per-step
  rebuild if the dispatch kernels are only tens of microseconds.
- Production queues should be built when task assignment changes, not every
  physics step. If queues must change every step, build counts and offsets in a
  fixed-shape device path and capture-test it.
- Sorting can be competitive with many `nonzero` passes. If the layout can
  tolerate sorted/grouped slot order, sort-then-segment deserves a real
  prototype.

Graph-captured Warp atomic queue construction for 1M work items:

| Pattern | Primitive counts | Primitive queues | Kind queues |
|---|---:|---:|---:|
| random | 0.1301 ms | 0.1428 ms | 0.2127 ms |
| skew | 0.2788 ms | 0.2911 ms | 0.2320 ms |
| grouped | 0.4755 ms | 0.4825 ms | 0.5015 ms |

Interpretation:

- A simple Warp atomic fused-pipeline queue is much better than Torch `nonzero` for
  random work, but it is not free; it is about the same scale as the richer
  dispatch kernels.
- Atomics degrade when ids are skewed or grouped because many threads contend
  on the same counters. In skewed work, exact-kind queues can beat primitive
  queues because contention is split across more counters.
- If assignment is stable across many steps, prebuilding queues at resample time
  is still best.
- If assignment changes every step and the distribution is skewed, sort/segment
  or an assignment layout that is already grouped may beat atomic queues.

Graph-captured Warp radix-sort queue/order construction for 1M work items:

| Pattern | Primitive sort | Primitive sort + segment | Kind sort |
|---|---:|---:|---:|
| random | 0.1024 ms | 0.1107 ms | 0.1026 ms |
| skew | 0.1024 ms | 0.1107 ms | 0.1031 ms |
| grouped | 0.0991 ms | 0.1070 ms | 0.0996 ms |

Interpretation:

- Sort cost is stable across distributions, unlike atomics.
- For random 1M work, the combined captured `sort_idx_primitive` path is
  `0.2812 ms`, faster than random `mega` at `0.3332 ms` but slower than
  prebuilt `idx_mega_primitive` at `0.1520 ms`.
- For skewed 1M work, the combined captured `sort_idx_primitive` path is
  `0.2622 ms`, slower than skewed `mega` at `0.1868 ms`. If sorted order is
  reused across steps, indexed mega becomes faster (`0.1388 ms`).
- This points to a two-mode plan: stable assignments should precompute sorted
  work ids; rapidly changing random assignments can use graphable sort; rapidly
  changing skewed assignments should probably stay with mega or use a grouped
  task layout.

### Launch Overhead Without Graph Capture

For the expanded 1M random workload without CUDA graph capture:

| Variant | Time |
|---|---:|
| `mega` | 0.3515 ms |
| `kind_queue` | 3.7448 ms |
| `primitive_queue_local_synth` | 0.4729 ms |
| `packed_scatter` | 0.4726 ms |
| `packed_local` | 0.4649 ms |

Interpretation:

- Queue-based dispatch only makes sense for a graphable/captured path or for
  very large kernels where launch overhead is irrelevant.
- Exact-kernel queues are especially sensitive to launch overhead; a 64-launch
  path is unacceptable unless it is captured and each launch has substantial
  work.
- This supports the Octi rule: graphability is not a knob for Warp hot paths;
  the design should be graphable by construction.

## Dense Command Buffer Cost

Artifact:

```bash
./isaaclab.sh -p -m isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_command_pipeline_layout
```

The real command path does more than dispatch projection/error:

```text
zero buf_error / buf_activation / command_reach / command_track
  -> dispatch scatter
  -> compose_reward
```

The benchmark simulates this dense layout and separates the costs.

### Current Scale

For `16384 envs * 8 slots`, command width 256:

| Operation | Time |
|---|---:|
| `torch zero all` | 0.0171 ms |
| `warp zero all` | 0.0226 ms |
| `warp zero slots` | 0.0043 ms |
| `dispatch only` | 0.0063 ms |
| `compose env loop` | 0.0043 ms |
| `pipeline full zero` | 0.0289 ms |
| `pipeline slot zero` | 0.0125 ms |
| `pipeline no zero` | 0.0089 ms |

For the same env/slot count, command width 1024:

| Operation | Time |
|---|---:|
| `torch zero all` | 0.0824 ms |
| `warp zero all` | 0.0817 ms |
| `pipeline full zero` | 0.0885 ms |
| `pipeline slot zero` | 0.0123 ms |
| `pipeline no zero` | 0.0090 ms |

Interpretation:

- Dense command zeroing scales with `num_envs * command_width`, not active
  work. Once command width grows, the clear dominates dispatch and compose.
- Torch zero and graph-captured Warp zero are essentially the same bandwidth
  operation. Moving the clear into Warp does not solve the layout cost.
- Slot-buffer clearing is cheap because it scales with `num_envs * k_max`.

### 1M Active Slots

For `131072 envs * 8 slots`, command width 256:

| Operation | Time |
|---|---:|
| `torch zero all` | 0.1674 ms |
| `warp zero all` | 0.1663 ms |
| `dispatch only` | 0.0246 ms |
| `compose env loop` | 0.0144 ms |
| `pipeline full zero` | 0.2558 ms |
| `pipeline slot zero` | 0.0451 ms |
| `pipeline no zero` | 0.0382 ms |

For `131072 envs * 8 slots`, command width 1024:

| Operation | Time |
|---|---:|
| `torch zero all` | 0.6538 ms |
| `warp zero all` | 0.6489 ms |
| `pipeline full zero` | 0.7487 ms |
| `pipeline slot zero` | 0.0451 ms |
| `pipeline no zero` | 0.0382 ms |

Selective command-row clearing (`clear_envs=8192`) at the same scale and width:

| Operation | Time |
|---|---:|
| `warp zero cmd rows` | 0.0410 ms |
| `pipeline row zero` | 0.1341 ms |

Interpretation:

- At future scale, dense per-step command clearing can dominate the whole
  command path by an order of magnitude.
- Current terrain preset command widths are small (`locomotion`: reach 8,
  track 8; `pose_vel`: reach 7, track 6), so the dense-clear issue is mostly
  future-facing for wider command dictionaries or many task families.
- The safe target is not "never clear"; it is "clear command rows only when
  task assignment changes, then overwrite active command columns every step."
- `compose_reward`'s one-thread-per-env loop is not the bottleneck for current
  `k_max` regimes. A slot-atomic progress reducer was not faster enough to
  justify the added complexity, and it does not cover the real product/latch
  semantics anyway.

### Repaired Real Trace Benchmark

I also repaired the existing mock trace benchmark:

- Added a mock reader for `BUFFER_KIND.JOINT_MECH_POWER_ABS`.
- Resolved `MultiTaskEnvCfg().commands.goal_point` presets before constructing
  `MultiTaskCommand`.
- Forced `debug_vis=False` in mock mode so it does not try to create USD
  visualization markers.
- Added mock `asset.data.root_quat_w` so body-frame command rotation can run.

Run:

```bash
TRACE=source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/mdp/commands
TRACE=$TRACE/benchmark/trace_multi_task_command.py
./isaaclab.sh -p "$TRACE" --mode mock --use_warp --num_envs 16384 \
  --num_steps 80 --warmup_steps 20 --top_n 12 \
  --output /tmp/octi_multitask_trace
```

Key current-scale observations:

- `dispatch_mega` CUDA kernel: ~5.96 us/step.
- `compose_reward` CUDA kernel: ~2.16 us/step.
- Slab fill kernels together are comparable to dispatch.
- Body-frame rotation still runs as Torch ops after dispatch and shows up
  prominently in the trace (`quat_apply_inverse`, `aten::linalg_cross`,
  fused elementwise kernels).

Interpretation:

- For today's small command widths, command clearing is not the main bottleneck.
  The dense-clear change matters more for future wider command layouts.
- If optimizing current `locomotion`, the next likely target is moving the
  canonical body-frame rotation into Warp/fusing it with dispatch, not queueing.

### Body-Frame Rotation Follow-Up

I replaced the Warp subclass's post-dispatch Torch rotation with a cached Warp
binding plus `rotate_canonical_vec3_pair`:

```text
dispatch_mega                              # world-frame canonical vec3 deltas
rotate_canonical_vec3_pair(reach + track)  # root-frame policy obs
compose_reward                             # still uses rotation-invariant errors
```

The reference path stays unchanged, so the existing Warp-equivalence test still
compares Warp output against the Torch reference.

Targeted validation:

| Test | Result |
|---|---|
| `test_multi_task_command_mock.py` + `test_multi_task_warp_equivalence.py` | 36 passed |

Trace after the change (`16384` envs, mock Warp path):

| Kernel | Avg CUDA time |
|---|---:|
| `dispatch_mega` | 5.98 us |
| `rotate_canonical_vec3_pair` | 1.54 us |
| `compose_reward` | 2.20 us |

The Torch `quat_apply_inverse` / `aten::linalg_cross` kernels disappeared from
the top CUDA events. The current preset now uses one rotation launch per root
asset instead of separate reach/track launches.

Launch count in the profiled dispatch window dropped from 480 to 420
`cuLaunchKernel` calls over 60 recorded steps after fusing reach+track rotation.

Isolated benchmark:

```bash
./isaaclab.sh -p -m \
  isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_body_frame_rotation \
  --num_envs 16384 131072 --num_offsets 1 2 4 8 --repeat 200
```

| Envs | Offsets | Torch | Warp | Speedup |
|---:|---:|---:|---:|---:|
| 16384 | 1 | 0.1945 ms | 0.0110 ms | 17.72x |
| 16384 | 2 | 0.0843 ms | 0.0110 ms | 7.66x |
| 16384 | 4 | 0.1838 ms | 0.0105 ms | 17.48x |
| 16384 | 8 | 0.3579 ms | 0.0103 ms | 34.61x |
| 131072 | 1 | 0.1973 ms | 0.0107 ms | 18.52x |
| 131072 | 2 | 0.1023 ms | 0.0105 ms | 9.73x |
| 131072 | 4 | 0.1832 ms | 0.0103 ms | 17.81x |
| 131072 | 8 | 0.3451 ms | 0.0105 ms | 32.81x |

Max error versus Torch was `<= 1.2e-6`.

Graph-capture smoke:

```text
rotate_canonical_vec3_pair capture ok
```

Remaining graphability issue in the live Warp path:

- `_fill_unified_buffer_warp()` still calls Python readers, reshapes, and
  `wp.from_torch(...)` for each slab each step.
- Most readers are stable zero-copy views and could be bound once.
- `JOINT_MECH_POWER_ABS` is not a stable view today; it computes
  `abs(applied_torque * joint_vel)` with Torch each step.
- The test-facing `BUFFER_KIND_READERS` patch API also allows readers to return
  new tensors after command construction, so blindly prebinding reader outputs
  would change test semantics.

The cleaner production direction is to split the read contract:

```text
stable scene-backed slabs:
  prebind source arrays once in the Warp command binding

computed slabs:
  implement explicit Warp fill kernels, e.g.
  fill_slab_joint_mech_power_abs(applied_torque, joint_vel, unified)

test/mocking boundary:
  keep dynamic reader patching for the reference path
  use stable tensor handles for Warp-path tests
```

That removes per-step Python reader dispatch from the graphable Warp path
without pretending every buffer kind is a view.

Slab binding overhead benchmark:

```bash
./isaaclab.sh -p -m \
  isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_slab_binding_overhead \
  --num_envs 16384 --repeat 500

./isaaclab.sh -p -m \
  isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_slab_binding_overhead \
  --num_envs 131072 --repeat 200
```

For six copy slabs with sizes `[36, 48, 36, 36, 16, 16]`:

| Envs | Path | Wall ms/step | CUDA event ms/step |
|---:|---|---:|---:|
| 16384 | dynamic reader + `wp.from_torch` | 0.0916 | 0.0901 |
| 16384 | prebound `wp.array` handles | 0.0465 | 0.0466 |
| 16384 | prebound graph replay | 0.0145 | 0.0143 |
| 131072 | dynamic reader + `wp.from_torch` | 0.1848 | 0.1849 |
| 131072 | prebound `wp.array` handles | 0.1849 | 0.1846 |
| 131072 | prebound graph replay | 0.1801 | 0.1797 |

Interpretation:

- At the 16k-env scale, slab copy is launch/setup dominated; prebinding and
  graph replay are worthwhile.
- At 131k envs, the same copies are memory-bandwidth dominated; graph replay
  helps little unless the algorithm reduces bytes moved or fuses more work into
  the slab copy.
- This supports a two-tier plan: prebind stable slab handles for graphability,
  but do not expect it to solve future 1M-task memory traffic by itself.

## 2D Env-Slot Launch Ordering

Artifact:

```bash
./isaaclab.sh -p -m isaaclab_tasks.manager_based.multi_task.mdp.commands.benchmark.bench_dispatch_2d_order
```

Real `dispatch_mega` launches `dim=(num_envs, k_max)`, so flat random ordering
is not the whole story. This benchmark compares branch patterns in a 2D
env-slot launch.

For `1048576 envs * 8 slots`:

| Pattern | 2D launch | Flat launch |
|---|---:|---:|
| slot-homogeneous | 0.0603 ms | 0.0569 ms |
| env-homogeneous | 0.0431 ms | 0.0410 ms |
| random task slots | 0.0604 ms | 0.0567 ms |
| task slots, envs sorted by task | 0.0492 ms | 0.0459 ms |
| fully random | 0.0607 ms | 0.0572 ms |

Interpretation:

- Env-homogeneous work is faster than slot-homogeneous work, which suggests the
  relevant neighboring threads in this launch shape tend to see slots within an
  env rather than the same slot across many envs.
- This makes per-task subtask ordering a real optimization lever. If each task
  stores subtasks ordered by primitive/state family, the existing 2D mega
  launch can get some of the sorted-work benefit without a global work-list
  sort.
- Sorting envs by task also helps, but less than making each env's local slot
  sequence homogeneous.
- Current terrain presets are already mostly locally grouped by primitive:
  examples include `velocity = [vec3, vec3, scalar, scalar]`,
  `pose = [vec3, quat, scalar, scalar]`, and most locomotion tasks as
  `[vec3/quat, scalar, scalar, ...]`. This is good. The recommendation is to
  preserve/enforce this invariant as new task families are added, not to
  reorder today's presets aggressively.

## Warp/Newton Repository Study

Sources checked:

- Installed Warp 1.12.0 examples/tests and public Warp 1.13.0 docs.
- Local `isaaclab_newton` integration under `source/isaaclab_newton`.
- Upstream Newton repository/docs at `newton-physics/newton`.

### Warp Patterns That Matter Here

- CUDA graph capture records launches/copies/zeroes, not arbitrary Python
  control flow. Python-side layout decisions, queue construction, tensor
  wrapping, and backend selection must happen before capture or through a fixed
  device-side launch sequence.
- Captured graphs depend on stable array lifetimes and pointer identity. The
  installed experimental Warp env keeps persistent input buffers and copies new
  actions into them instead of re-wrapping tensors each step; this is the right
  model for command slabs too.
- `wp.ScopedCapture` is the normal path, but the local Newton integration shows
  that capture sometimes needs an eager warmup first to flush one-time
  allocations. This matches the experimental `WarpGraphCache`: warm up once,
  capture the steady-state launches, then replay.
- Conditional graph nodes (`wp.capture_if`, `wp.capture_while`) exist on recent
  CUDA stacks, but Warp tests gate them on CUDA/driver support. Newton's P-ADMM
  config also keeps an unrolled-loop alternative for graph environments where
  conditional nodes are not desired. For Octi command dispatch, fixed unrolled
  launch sequences are simpler than graph conditionals.
- Warp supports runtime kernel/function specialization through closures and
  static expressions. That is useful for a small fixed set of schedule kernels,
  but not for creating kernels dynamically from every task preset in the hot
  path: adding kernels mutates a module and can force recompilation.
- New tile primitives are interesting for queue construction, compaction, and
  block-local reductions. They are not automatically better for the command
  projection path, because each `(env, slot)` mostly owns independent scalar or
  small-vector work with little inter-thread reuse.
- `wp.indexedarray` is a lightweight Warp-side view, but PyTorch has no
  equivalent and conversion requires either a copy or sharing data/index buffers
  separately. If we use indexed/sorted work ids, keep that path Warp-native
  through dispatch and composition.

### Newton Patterns That Matter Here

- Newton exposes high-level solver choices, but inside each solver it uses
  fixed physics phases and many semantically named kernels. It does not try to
  solve physics heterogeneity with one giant "everything" dispatcher.
- Primitive math is factored as `@wp.func` helpers and called inside larger
  phase kernels. The VBD rigid kernels are a good example: joint projection,
  force, and Hessian helper functions are reused inside solver kernels rather
  than launched as tiny kernels.
- The solver data contract is persistent state: `Model`, `State`, `Control`,
  `Contacts`, solver scratch, and output state all live across steps. This is
  closer to our desired backend-owned output store than to per-call temporary
  tensors.
- Local Newton graph capture explicitly excludes non-capturable work such as
  Fabric/USD sync and performs only pure physics launches inside the graph.
  Octi command should treat debug visualization, dynamic reader patching, and
  non-stable Torch computations the same way: outside the graphable backend
  contract.
- Local Newton write APIs repeatedly note that mask-based full-shape writes are
  required for graphed pipelines, even when index-based methods exist. This is
  a useful warning for command reset/update paths: fixed masks or fixed buffers
  are more graph-friendly than dynamically sized env-id lists.
- Newton accepts multiple launches when they are real solver phases. It avoids
  excessive launch count by graph capture and phase fusion, not by forcing all
  semantics into one divergent kernel.

### Implications For Octi Command Backends

The repository study changes the priority order:

1. **Keep true primitives as `@wp.func`, not kernels.** Examples: gather
   scalar/vec3/quat, threshold predicate, sum/pair-diff reduction, delta,
   metric, activation, scatter/materialize.
2. **Use fused schedule kernels as the launch boundary.** A schedule is a
   repeated primitive DAG such as `direct_vec3_delta` or
   `contact_pair_diff_delta`. Schedule kernels call the primitive functions
   inline.
3. **Optimize native ordering before dynamic queues.** The synthetic benchmark
   showed grouped mega can beat queue backends. Warp/Newton patterns agree:
   fixed phase order and coherent data layout are the first-class design tools.
   If task/subtask assignment can be stored in schedule-coherent slot order, a
   simple dispatch kernel may remain the best production backend.
4. **Build queues at assignment/resample time when possible.** Per-step queue
   construction costs are the same order as dispatch. A dynamic queue path is
   only attractive when assignment changes every step and the divergence win
   exceeds the queue-build cost.
5. **Make graphability required, not optional.** The public backend can still
   run eagerly for debugging, but the implementation must have a fixed launch
   topology, fixed scratch allocations, stable arrays, and no Python/GPU-data
   decisions in the hot path.
6. **Move the data contract to backend-owned stores.** Dense command tensors,
   local packed outputs, schedule queues, sorted work ids, and masks should be
   backend internals. The public command interface should consume the resulting
   reward/progress/observation tensors, not dictate `(env, slot)` layout.
7. **Use tile programming selectively.** It is worth trying for stream
   compaction or schedule queue construction, especially if we need a graphable
   per-resample builder. It is not the default answer for per-slot projection
   kernels unless a benchmark shows block-local reuse.

This points to two practical next backends:

| Backend | Purpose |
|---|---|
| `schedule_ordered_mega` | Keep one dispatch launch, but enforce schedule-coherent slot ordering and remove unnecessary branch work. This tests the "layout first" path. |
| `primitive_queue_local` | Prebuilt schedule queues plus backend-local outputs. This tests the upper-bound path when downstream composition can stay grouped/local. |

`packed_scatter` is useful as an intermediate experiment, but it is not the
final target unless dense output materialization is truly required every step.

## Current Real Preset Structure

Using a mock scene to build `TaskSpec` for current terrain presets:

| Preset | Tasks | Unique subtasks | `k_max` | Read groups |
|---|---:|---:|---:|---:|
| `simple_pos_vel` | 2 | 4 | 3 | 4 |
| `pose_vel` | 2 | 6 | 4 | 6 |
| `velocity` | 1 | 4 | 4 | 4 |
| `position` | 1 | 3 | 3 | 3 |
| `locomotion` | 8 | 12 | 4 | 8 |

For `locomotion`, 8 tasks deduplicate to 12 subtasks:

```text
BODY_LIN_VEL             1
BODY_ANG_VEL             1
BODY_CONTACT_COUNT       2
BODY_POS                 3
BODY_QUAT                2
BODY_CONTACT_COUNT_DIFF  2
JOINT_MECH_POWER         1
```

The existing spec builder already exposes useful structure:

- `read_group_member_sids`
- `read_group_state_kernel_id`
- `subtask_gather_offset`
- `subtask_gather_count`
- `state_stride`
- `canonical_offset`

But the Warp runtime flattens this back into per-slot dynamic branching.

## Shared Structure to Extract

The future-friendly hierarchy should not be:

```text
task -> subtask -> state-kernel branch
```

It should be:

```text
task -> subtask -> primitive projection -> metric -> activation -> scatter/compose
```

The important split is that "state kernel" is often semantic, while "primitive"
is computational:

| Layer | Examples | GPU implication |
|---|---|---|
| Semantic state kernel | `BODY_POS`, `BODY_LIN_VEL`, `BODY_ANG_VEL` | Different meaning, same vec3 skeleton. |
| Primitive projection | vec3/scalar/quat/contact gather or reduce | Good queue key. |
| Metric | L2, absolute scalar, quaternion angle | Often shared across many semantic kernels. |
| Activation | tanh/saturating/instant | Can be fused after metric if activation set stays small. |
| Scatter/compose | canonical offset, stride, instant flag, reward slot | Controls whether packed layout survives. |

This suggests a `DispatchPlan` with at least two granularities:

```text
semantic_subtasks
  state_kernel_id
  gather metadata
  target offset / stride
  canonical scatter metadata

primitive_work
  primitive_id
  subtask_id or packed metadata row
  env id / slot id
  packed input offsets
```

The compiler should then choose among three execution forms:

1. **Grouped mega**: if assignment/layout is already homogeneous by state kind,
   one branchy launch can be fastest.
2. **Indexed sorted mega**: one launch over sorted work ids; strong first
   prototype when preserving the current output layout.
3. **Primitive queues with scatter**: best when semantic kernels share heavy
   primitive work and sorted mega still pays too much branch cost.
4. **Packed primitive pipeline**: best long-term form if reward/observation
   consumers can read grouped outputs directly.

## Recommended Direction

Do not grow `dispatch_mega`.

The code layout now mirrors the implementation strategy boundary:

```text
multi_task_command_warp.py          # public Warp command term
impl/
  mega_kernel/                      # wired backend: 2D (env, slot) execution
    bindings.py                     # long-lived Torch/Warp views
    read.py                         # scene slabs -> unified buffer
    execute.py                      # dispatch_mega launch
    rotation.py                     # body-frame command rotation
    compose.py                      # compose_reward launch
  packed_scatter/                   # wired backend: fused-pipeline-sorted flat queue
```

Wired backends are selected through `MultiTaskCfg.dispatch_backend`.
Measured future alternatives should become folders only when their required
plans/data layouts and public command-level benchmarks exist.

Instead, split the migration into two separate changes. The dense-clear change
is lower-risk and likely pays off immediately when command width grows; the
dispatch-order change attacks branch divergence.

### Step 1: Clear Less

Current per-step shape in `MultiTaskCommand._update_command`:

```text
_buf_error.zero_()
_buf_activation.zero_()
_command_reach.zero_()
_command_track.zero_()
dispatch
compose
```

Target shape:

```text
on task assignment/resample for env_ids:
  _command_reach[env_ids].zero_()
  _command_track[env_ids].zero_()
  _buf_error[env_ids].zero_()        # only if debug/readback needs inactive slots zero
  _buf_activation[env_ids].zero_()   # only if debug/readback needs inactive slots zero

each step:
  dispatch overwrites all active slot outputs and active command columns
  compose reads only slots < slot_count
```

The row clear already exists in `MultiTaskCommand._resample_command`:

```text
self._command_reach[env_ids] = 0.0
self._command_track[env_ids] = 0.0
```

So the likely redundant part is the per-step full command clear, not the
resample-time row clear.

I applied this minimal production experiment:

```text
keep:
  _buf_error.zero_()
  _buf_activation.zero_()

remove from per-step update:
  _command_reach.zero_()
  _command_track.zero_()
```

Targeted validation:

| Test | Result |
|---|---|
| `test_multi_task_command_mock.py` + `test_multi_task_warp_equivalence.py` | 36 passed |

The first plain run exposed a separate initialization issue: the mega backend's
plan construction called `wp.from_torch` before Warp runtime initialization in
the test process. Adding an explicit idempotent `wp.init()` in plan construction
fixed the plain test path.

Safety requirements:

- Every active subtask must overwrite all command lanes it owns every step.
- Rows must be cleared whenever an env changes task assignment, because inactive
  canonical columns are visible to observations.
- Composer must continue masking by `slot_count`; padded slot data is undefined.
- If any public/debug consumer expects inactive `_buf_error` or
  `_buf_activation` slots to be zero every step, either keep slot clears or move
  that guarantee to the read boundary.

### Step 2: Compile Dispatch Order

Add a compiled dispatch plan:

```text
TaskSpec
  -> DispatchPlan
       sorted_work_ids
       primitive_groups
       static per-subtask metadata
       fixed-capacity work queues
       queue counts
       optional packed input/output slabs
```

Runtime shape:

```text
resample/update env slots
  -> build or refresh sorted work ids / fused-pipeline queues
  -> launch indexed mega or fixed pipeline kernels
  -> scatter to existing outputs, or keep packed outputs for a queued composer
```

Initial fused pipeline families should be smaller than state-kernel families.
The names below describe computation shape, not semantic task meaning:

| Fused pipeline | Current state kernels |
|---|---|
| `direct_vec3_delta` | `BODY_POS`, `BODY_LIN_VEL`, `BODY_ANG_VEL` |
| `direct_scalar_delta` | `JOINT_POS`, `JOINT_VEL`, `BODY_POS_Z` |
| `direct_quat_delta` | `BODY_QUAT` |
| `vec3_threshold_vector_delta` | `BODY_CONTACT` |
| `vec3_threshold_sum_delta` | `BODY_CONTACT_COUNT` |
| `vec3_threshold_pair_diff_delta` | `BODY_CONTACT_COUNT_DIFF` |
| `scalar_sum_delta` | `JOINT_MECH_POWER` |

The first production prototype should probably be `idx_mega_primitive` or
`idx_mega_kind`, because it preserves the existing output contract and only
changes work order:

```text
buf_error[env, slot]
buf_activation[env, slot]
command_reach[env, canonical_offset]
command_track[env, canonical_offset]
```

That gives a manageable migration:

1. Build sorted work ids at resample time.
2. Keep current output tensors.
3. Replace only `dispatch_mega` with indexed sorted-mega dispatch.
4. Keep `compose_reward` unchanged.
5. Benchmark against `dispatch_mega` with and without the dense-clear change.

Then, if results justify it or branch cost still dominates, go deeper:

1. Split indexed sorted-mega into primitive kernels with scatter.
2. Keep packed grouped outputs.
3. Write a queued/grouped composer.
4. Stop scattering intermediate slot data unless an observation/debug path
   actually needs it.

## Primitive-Queue-Local Precheck

The follow-up benchmark pass intentionally avoided current locomotion presets;
they are too small and too homogeneous to answer the future architecture
question. The relevant stress case is synthetic command-like work:

```text
1,048,576 work items
64 semantic state kernels
8 fused schedule families
graph-captured Warp timing
```

Older benchmark runs used shorter synthetic labels. In this section:

- `primitive_queue_local_synth` / older `primitive_queue` means one queue per
  fused schedule family, with indexed reads and dense/global writes.
- `packed_scatter` means packed source/target rows by fused schedule, with
  dense/global writes.
- `packed_local` is the synthetic upper bound for `primitive_queue_local`:
  packed source/target rows by fused schedule with packed local outputs and no
  dense materialization.

### Dispatch Stress Results

| Work order | `mega` | `idx_mega_primitive` | `primitive_queue` | `packed_scatter` | `packed_local` |
|---|---:|---:|---:|---:|---:|
| random | 0.3317 ms | 0.1521 ms | 0.2124 ms | 0.1418 ms | 0.1200 ms |
| skew | 0.1871 ms | 0.1368 ms | 0.1399 ms | 0.1039 ms | 0.0934 ms |
| grouped | 0.0927 ms | 0.0948 ms | 0.1223 ms | 0.1231 ms | 0.1317 ms |
| random, pipeline-sorted native order | 0.1089 ms | 0.1113 ms | 0.1192 ms | 0.1176 ms | 0.1169 ms |
| skew, pipeline-sorted native order | 0.1064 ms | 0.1205 ms | 0.1792 ms | 0.1905 ms | 0.2003 ms |

Key result: `primitive_queue_local` is not automatically the right answer.
When native work order is random, the packed-local upper bound is about
2.76x faster than mega. When native work order is already pipeline-coherent,
mega is faster than queue/local variants. This means the first optimization
question is layout/order, not queueing.

### State-Kernel Diversity Scaling

Random 1M work items:

| Semantic kernels | `mega` | `packed_scatter` | `packed_local` | `packed_local / mega` |
|---:|---:|---:|---:|---:|
| 8 | 0.0903 ms | 0.0594 ms | 0.0605 ms | 1.49x |
| 16 | 0.1104 ms | 0.0704 ms | 0.0585 ms | 1.89x |
| 32 | 0.1376 ms | 0.0782 ms | 0.0667 ms | 2.06x |
| 64 | 0.3317 ms | 0.1418 ms | 0.1200 ms | 2.76x |

The queue/local architecture becomes compelling only as semantic diversity
rises and native ordering is not already coherent.

### Queue Build Costs

Dynamic queue construction is expensive enough to erase many dispatch wins:

| Operation, 1M random work | Time |
|---|---:|
| Torch kind `nonzero` queues | 1.8126 ms |
| Torch fused-schedule `nonzero` queues | 0.3039 ms |
| Torch `argsort(kind)` | 0.0798 ms |
| Warp fused-schedule atomic queues | 0.1331 ms |
| Warp fused-schedule sort | 0.1025 ms |
| Warp fused-schedule segment | 0.1108 ms |

For skewed work, Warp atomic fused-schedule queues were worse:
0.2744 ms. A production backend should therefore build queue/order state at
resample time when possible. Per-step graphable sort/segment is acceptable
only when assignment truly changes every step and the dispatch win is larger
than roughly 0.10 ms per 1M work items.

### Layout Costs

Dense materialization and clearing are also first-order costs when command
width grows:

| Shape | Full dense zero | Slot-only zero | Dispatch | Compose | Pipeline no zero |
|---|---:|---:|---:|---:|---:|
| 16k envs, `k=8`, width 256 | 0.0287 ms pipeline | 0.0123 ms pipeline | 0.0062 ms | 0.0041 ms | 0.0090 ms |
| 16k envs, `k=32`, width 1024 | 0.1131 ms pipeline | 0.0328 ms pipeline | 0.0144 ms | 0.0152 ms | 0.0287 ms |

This supports the output-store refactor: dense command tensors must not remain
the hot-path truth for high-width task sets. But it also says local outputs
alone are not enough; if observations require dense materialization every step,
the local backend loses much of its reason to exist.

### Revised Implementation Bar

Do not implement `primitive_queue_local` as the next production backend unless
the design satisfies these constraints:

1. Fused schedules are built from reusable `@wp.func` primitive ops. Do not
   launch one kernel per tiny primitive op.
2. Native task/slot ordering by fused schedule is considered first. If the
   `(env, slot)` layout can be made pipeline-coherent, a simple mega-style
   kernel may be the fastest backend.
3. Queue/order state is built at resample time for stable assignments.
4. Local outputs avoid dense materialization before compose.
5. Dense command materialization is measured separately and should become an
   explicit observation/debug boundary, not mandatory hot-path work.
6. A production `primitive_queue_local` benchmark must show a clear win on
   random or high-diversity command-like workloads, not just parity on current
   locomotion presets.

### Producer-Sharing Upper Bound

The graph backend's remaining dispatch question is whether shared producers are
worth materializing. A tiled block-local experiment was tried and rejected:
keeping producer values inside one env block avoids global producer rows, but
Warp tile extraction plus 256-wide blocks cost more than the saved repeated
loads/reductions. The failed tiled/env-local kernels were removed from the
production backend.

Public `MultiTaskCommand` benchmark, 16,384 envs, interleaved future synthetic
workload:

| Backend | Time |
|---|---:|
| `mega_kernel` | 0.4751 ms |
| `schedule_ordered_mega` | 0.3705 ms |
| `primitive_graph_local` | 0.3776 ms |

This workload has high fanout, but the producers are cheap. Recomputing a few
unified-buffer loads per slot is about as good as materializing producer rows.

Benchmark-only heavy synthetic variant (`future_synthetic_heavy_interleaved`)
uses 128 mock joints so the shared scalar-sum producer has real arithmetic. In
that regime the graph model reaches the expected upper-bound behavior:

| Backend | Time |
|---|---:|
| `mega_kernel` | 0.7380 ms |
| `schedule_ordered_mega` | 0.5214 ms |
| `primitive_graph_local` | 0.4077 ms |

After producer launch fusion, the same public benchmark measured
`primitive_graph_local` dispatch at about `0.140 ms` on this heavy synthetic
workload. A deep phase profile showed `dense_graph_producers` at about
`0.039 ms` and `dense_graph_consumer` at about `0.110 ms`; the next bottleneck
is therefore the dense consumer/output contract, not producer computation.

Conclusion: graph sharing is the right direction when producer work is
substantial or reused many times. For cheap direct projections, slot ordering is
the dominant optimization and graph materialization should remain conditional.

## Important Caveats

- If work can be ordered so each warp is homogeneous, the mega-kernel is hard to
  beat. A cheaper alternative may be canonical slot ordering by state family
  within each env or by task assignment.
- Queue build cost was not included in the synthetic benchmark. For graphable
  production code, queue storage must be preallocated and queue counts must be
  fixed-capacity device tensors. Rebuilding queues every step with Python/Torch
  would erase the win.
- The biggest measured win came from data layout, not merely from replacing
  branches with queues. The long-term design should focus on carrying grouped
  layout into downstream kernels.
- Current production presets are small enough that this is future-facing. It is
  valuable for the 1M-task regime, not necessarily for today's 8-task locomotion
  preset.

## Validation Status

Targeted checks that passed after the production changes:

```text
./isaaclab.sh -p -m py_compile \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/mdp/commands/*.py \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/mdp/commands/benchmark/trace_multi_task_command.py

./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/tests/test_multi_task_command_mock.py \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/tests/test_multi_task_warp_equivalence.py

./isaaclab.sh -p -m pre_commit run ruff --files <touched command files>
./isaaclab.sh -p -m pre_commit run ruff-format --files <touched command files>
./isaaclab.sh -p -m pre_commit run codespell --files <touched command files>
```

Results:

| Check | Result |
|---|---|
| Targeted command tests | 36 passed |
| Targeted Ruff | passed |
| Targeted Ruff format | passed |
| Targeted codespell | passed |
| `rotate_canonical_vec3_pair` Warp capture smoke | passed |

Global `./isaaclab.sh -f` still fails on unrelated pre-existing files:
`scripts/reinforcement_learning/crl/*`, `source/isaaclab/isaaclab/utils/wandb.py`,
factory `rsl_rl_ppo_cfg.py`, terrain retarget/mesh files, and an existing
changelog spelling complaint. I did not fix those in this pass.
