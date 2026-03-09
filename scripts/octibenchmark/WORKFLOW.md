# octibenchmark — Workflow & Architecture

## Overview

octibenchmark profiles IsaacLab environment step internals using NVIDIA's
`nsys` profiler with NVTX annotations. It **does not modify any IsaacLab
source files** — all instrumentation is injected at runtime via monkey-patching.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User entry point                             │
│  run_matrix.py  (or sweep.py / CLI)                                 │
│  Discovers example configs, executes selected matrices              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ generates BenchmarkRun list
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  bench_cfg.py  — BenchmarkMatrix.execute()                          │
│  For each valid run:                                                │
│    1. Build nsys command                                            │
│    2. Launch subprocess: nsys profile → benchmark.py / _train.py    │
│    3. Collect .nsys-rep → analyze.py                                │
│    4. Aggregate results → wandb                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ subprocess per run
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  benchmark.py  or  benchmark_train.py   (runs inside nsys)          │
│    1. Create env via gym.make()                                     │
│    2. install_nvtx_hooks(env.unwrapped)  ← monkey-patches methods   │
│    3. Warmup frames  (outside nsys capture window)                  │
│    4. cudaProfilerStart()                                           │
│    5. Benchmark loop  (nsys captures CUDA + NVTX here)              │
│    6. cudaProfilerStop()                                            │
│    7. env.close()                                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ produces .nsys-rep file
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  analyze.py                                                         │
│    1. nsys export --type=sqlite  →  .sqlite file                    │
│    2. SQL queries on NVTX_EVENTS table  → per-range stats           │
│    3. SQL queries on CUPTI_ACTIVITY_KIND_KERNEL  → GPU kernel stats  │
│    4. Regex aggregation on kernel names (optional)                   │
│    5. Step anatomy: map each GPU kernel to its NVTX code section    │
│    6. Return structured dict {nvtx_ranges, gpu_kernels, ...}        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Modules

### 1. `nvtx_hooks.py` — NVTX Injection (no timing code)

**Purpose**: Wraps environment methods with `nvtx.range_push(label)` /
`nvtx.range_pop()` so nsys can group GPU work under human-readable labels.

**How it works**: `install_nvtx_hooks(env)` detects the env type and
monkey-patches methods. Each wrapped call becomes a named range in the
nsys timeline.

**Overhead**: ~100 ns per push/pop pair. No Python timers, no
`torch.cuda.synchronize()`, no `time.perf_counter()`. nsys handles all
measurement at the driver level.

**Hooks installed**:

| Label                              | What it wraps                           | Env type       |
|------------------------------------|-----------------------------------------|----------------|
| `env.step`                         | `env.step()`                            | Both           |
| `sim.step`                         | `env.sim.step()`                        | Both           |
| `sim.render`                       | `env.sim.render()`                      | Both           |
| `scene.write_data_to_sim`          | `env.scene.write_data_to_sim()`         | Both           |
| `scene.update`                     | `env.scene.update()`                    | Both           |
| `env._reset_idx`                   | `env._reset_idx()`                      | Both           |
| `direct.pre_physics_step`          | `env._pre_physics_step()`               | DirectRLEnv    |
| `direct.apply_action`              | `env._apply_action()`                   | DirectRLEnv    |
| `direct.get_rewards`               | `env._get_rewards()`                    | DirectRLEnv    |
| `direct.get_dones`                 | `env._get_dones()`                      | DirectRLEnv    |
| `direct.get_observations`          | `env._get_observations()`               | DirectRLEnv    |
| `vision.compute_image_obs`         | `env._compute_image_observations()`     | Vision envs    |
| `vision.compute_proprio_obs`       | `env._compute_proprio_observations()`   | Vision envs    |
| `vision.feature_extractor.step`    | `env.feature_extractor.step()`          | Vision envs    |
| `action.process`, `action.apply`   | action_manager methods                  | ManagerBased   |
| `reward.compute`                   | reward_manager.compute()                | ManagerBased   |
| `observation.compute`              | observation_manager.compute()           | ManagerBased   |
| `termination.compute`              | termination_manager.compute()           | ManagerBased   |
| `command.compute`                  | command_manager.compute()               | ManagerBased   |
| `event.apply`                      | event_manager.apply()                   | ManagerBased   |
| `reward.term:<name>`               | Individual reward term functions         | ManagerBased   |
| `observation.term[<group>]:<name>` | Individual observation term functions    | ManagerBased   |
| `termination.term:<name>`          | Individual termination term functions    | ManagerBased   |
| `event.term[<mode>]:<name>`        | Individual event term functions           | ManagerBased   |

For RL training (`benchmark_train.py`), additional hooks are installed on
the RSL-RL runner's algorithm:

| Label                       | What it wraps                |
|-----------------------------|------------------------------|
| `runner.alg.act`            | Policy forward pass          |
| `runner.alg.process_env_step` | Transition storage         |
| `runner.alg.update`         | PPO gradient update          |
| `runner.alg.compute_returns`| GAE / return computation     |

---

### 2. `benchmark.py` — Single-Run Non-RL Benchmark

**Purpose**: Steps the environment with random actions under nsys profiling.
No neural network — isolates pure environment + physics + rendering cost.

**Typical runtime**: Depends on num_envs, physics backend, and whether
rendering is enabled. Rough order of magnitude:

| Configuration                              | ~Time per run |
|--------------------------------------------|---------------|
| State-only, Newton, 4096 envs, 100 frames  | 30–60 s       |
| Vision, Newton, 2048 envs, 128×128, 100 fr | 60–120 s      |
| Vision, Newton, 8192 envs, 256×256, 100 fr | 3–10 min      |

(Most time is environment creation + warmup. The profiled portion is shorter.)

**Supports two phases** (via `--phase` CLI arg):

- **`step`** (default): Profile the stepping loop only. NVTX hooks are
  installed *after* warmup to avoid slowing down startup.
- **`startup`**: Profile environment creation and first reset only.
  nsys capture covers `gym.make()` + `env.reset()`.

**Key steps (step phase)**:
1. Parse CLI args + Hydra overrides (presets, resolution)
2. `launch_simulation(env_cfg, args_cli)` — auto-detects Kit vs Newton
3. `gym.make()` → create the environment (no hooks yet)
4. Warmup loop (no hooks, outside nsys capture)
5. `install_nvtx_hooks(env.unwrapped)` → inject NVTX labels after warmup
6. `torch.cuda.cudart().cudaProfilerStart()`
7. Benchmark loop with random actions
8. `torch.cuda.cudart().cudaProfilerStop()`

---

### 3. `benchmark_train.py` — Single-Run RSL-RL Training Benchmark

**Purpose**: Full RL training loop (RSL-RL `OnPolicyRunner`) with NVTX hooks
on both the environment and the PPO algorithm. Captures the combined cost of
env stepping, policy inference, GAE computation, and gradient updates.

**Typical runtime**:

| Configuration                                    | ~Time per run |
|--------------------------------------------------|---------------|
| State-only, Newton, 4096 envs, 5 iterations      | 1–3 min       |
| Vision, Newton, 2048 envs, 128×128, 5 iterations | 3–8 min       |

---

### 4. `analyze.py` — nsys Result Parser

**Purpose**: Converts `.nsys-rep` → SQLite, then runs SQL queries to extract
NVTX range statistics and GPU kernel statistics.

**Pipeline**:
```
.nsys-rep  ──nsys export──▶  .sqlite  ──SQL──▶  structured dict
```

**What it extracts**:

1. **NVTX ranges** (from `NVTX_EVENTS` table, `eventType=59` = push/pop):
   - `name`: the label string (e.g. `sim.step`)
   - `count`: number of invocations
   - `total_ns`: total wall time across all calls
   - `avg_ns`: average wall time per call

2. **GPU kernels** (from `CUPTI_ACTIVITY_KIND_KERNEL` table):
   - `name`: demangled CUDA kernel name
   - `count`, `total_ns`, `avg_ns`

3. **Kernel pattern aggregations** (optional) — see dedicated section below.

4. **Step anatomy** (`--anatomy`): Dissects a single `env.step` and maps
   every GPU kernel to the NVTX-labeled code section that launched it.
   See the dedicated section below.

**Schema auto-detection**: Handles multiple nsys versions (newer uses
`start`/`end` columns with inline `text`; older uses
`startTimestamp`/`endTimestamp` with `textId` → `StringIds` join).

---

### 5. `bench_cfg.py` — Declarative Benchmark Matrix

**Purpose**: Define multi-dimensional benchmark sweeps as Python dataclasses.
Generates the cross-product of all axes, orchestrates execution, and
logs to wandb. No task-specific logic — all Hydra overrides are
user-defined.

**Classes**:

- `Launcher` enum: `NON_RL` (random actions) or `RSL_RL` (training)
- `BenchmarkRun`: A single resolved benchmark invocation
- `BenchmarkMatrix`: Defines sweep axes, generates all runs

**Design principle**: Nothing is hardcoded to a specific task or config
schema. Presets, resolutions, physics backends — they're all just Hydra
overrides. You define which ones to sweep and which stay constant.

**Example configuration**:

```python
from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher

matrix = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
    num_envs=[2048, 4096, 8192],

    # Sweep axes — the matrix generates the cross-product of all of these
    hydra_sweeps={
        "preset": [
            "presets=newton,newton_renderer,rgb",
            "presets=newton,newton_renderer,depth",
        ],
        "resolution": [
            "env.tiled_camera.width=64 env.tiled_camera.height=64",
            "env.tiled_camera.width=128 env.tiled_camera.height=128",
            "env.tiled_camera.width=256 env.tiled_camera.height=256",
        ],
    },

    # Constant overrides — applied to every run, not swept
    hydra_args=["env.decimation=4"],

    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
)

# Preview all runs
for run in matrix.runs():
    print(run.tag)

# Execute
matrix.execute(
    output_dir="/tmp/my_bench",
    wandb_project="my-benchmarks",
    kernel_patterns=["index", "ccd", "cholesky", "elementwise"],
)
```

This generates `1 task × 3 num_envs × 2 presets × 3 resolutions = 18 runs`.

**BenchmarkMatrix fields**:

| Field             | Type                        | Description                                    |
|-------------------|-----------------------------|-------------------------------------------------|
| `tasks`           | `list[str]`                 | Task names to sweep                             |
| `num_envs`        | `list[int]`                 | Environment counts to sweep                     |
| `hydra_sweeps`    | `dict[str, list[str]]`      | Named sweep axes of Hydra overrides (see below) |
| `hydra_args`      | `list[str]`                 | Constant Hydra overrides for every run          |
| `launcher`        | `Launcher`                  | `NON_RL` or `RSL_RL`                            |
| `num_frames`      | `int`                       | Env steps per run (non-RL)                      |
| `warmup_frames`   | `int`                       | Steps before nsys capture                       |
| `max_iterations`  | `int`                       | Training iterations (RSL_RL)                    |

**`hydra_sweeps` in detail**:

Keys are axis names (used in tags and wandb grouping). Values are lists
of Hydra override strings. Each string can contain **multiple
space-separated overrides** that belong together:

```python
hydra_sweeps={
    # Each entry is one Hydra override string
    "preset": [
        "presets=newton,newton_renderer,rgb",    # one option
        "presets=newton,newton_renderer,depth",   # another option
    ],
    # Multiple overrides in one string — width and height go together
    "resolution": [
        "env.tiled_camera.width=128 env.tiled_camera.height=128",
        "env.tiled_camera.width=256 env.tiled_camera.height=256",
    ],
    # You can sweep over anything — it's just Hydra args
    "physics": [
        "presets=newton",
        "presets=physx",
    ],
}
```

The matrix computes: `tasks × num_envs × axis_1 × axis_2 × ...`

Each run gets one choice from each axis, plus all `hydra_args`.

**Why this works for any task**: There is no hardcoded `_VISION_TASKS`
set or task-specific validation. If you benchmark a task that doesn't
use presets, just omit that axis. If your task has a different config
path for camera resolution, use whatever Hydra path is correct:

```python
# Custom task with different camera config
BenchmarkMatrix(
    tasks=["My-Custom-Task-v0"],
    num_envs=[512, 1024],
    hydra_sweeps={
        "camera": [
            "env.front_cam.width=320 env.front_cam.height=240",
            "env.front_cam.width=640 env.front_cam.height=480",
        ],
    },
)
```

**Generated nsys command** (example):
```
nsys profile -t cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o /tmp/bench/Shadow-Vision__envs4096__newton_newton_renderer_rgb__128_128__non_rl \
    --force-overwrite=true \
    python scripts/octibenchmark/benchmark.py \
    --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 \
    --num_envs 4096 --headless \
    --num_frames 100 --warmup_frames 10 \
    presets=newton,newton_renderer,rgb \
    env.tiled_camera.width=128 env.tiled_camera.height=128 \
    env.decimation=4
```

---

### 6. `sweep.py` — Simple num_envs Sweep (CLI-driven)

**Purpose**: Quick CLI-driven sweep over `num_envs` values for a single
task/preset combination. Simpler than `BenchmarkMatrix` — one axis only.

```bash
python scripts/octibenchmark/sweep.py \
    --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 \
    --num_envs 64 256 1024 \
    --num_frames 100 \
    presets=newton,newton_renderer,rgb
```

---

### 7. `examples/shadow_hand.py` — Pre-built Config

**Purpose**: Pure data definitions for Shadow Hand benchmark matrices.
5 predefined matrices covering vision/state-only × non-RL/training.
No CLI logic — just `BenchmarkMatrix` instances exposed via `ALL_MATRICES`.

**Matrices defined**:

| Name                         | Task                        | num_envs        | Sweep axes     | Runs |
|------------------------------|-----------------------------|-----------------|----------------|------|
| `shadow_vision_nonrl`        | Shadow-Vision               | 2048/4096/8192  | preset × res   | 18   |
| `shadow_vision_bench_nonrl`  | Shadow-Vision-Benchmark     | 2048/4096/8192  | preset × res   | 18   |
| `shadow_state_nonrl`         | Shadow (state-only)         | 2048–16384      | preset         | 4    |
| `shadow_vision_train`        | Shadow-Vision               | 2048/4096       | preset × res   | 4    |
| `shadow_state_train`         | Shadow (state-only)         | 4096/8192       | preset         | 2    |

---

### 8. `run_matrix.py` — Generic Matrix Runner

**Purpose**: CLI entry point that discovers example modules from
`octibenchmark/examples/` and runs their matrices. Each example module
just needs to define an `ALL_MATRICES` dict — no CLI boilerplate needed.

```bash
# List all available examples and their matrices
python scripts/octibenchmark/run_matrix.py --list

# Dry run — print all Shadow Hand commands
python scripts/octibenchmark/run_matrix.py \
    --example shadow_hand --dry_run

# Run specific matrices
python scripts/octibenchmark/run_matrix.py \
    --example shadow_hand \
    --matrices shadow_vision_nonrl shadow_state_nonrl

# Full sweep with kernel analysis
python scripts/octibenchmark/run_matrix.py \
    --example shadow_hand \
    --kernel_patterns "index" "mask" "scatter" "copy"
```

To add a new benchmark suite, create a new file in `examples/` with an
`ALL_MATRICES` dict — `run_matrix.py` will auto-discover it.

---

## Step Anatomy — Mapping GPU Kernels to Code Sections

### The problem

When looking at an nsys profile, you see hundreds of GPU kernels but
it's hard to tell which code section launched each one. "Where do those
3 mystery kernels between `sim.step` and `reward.compute` come from?"

The aggregate NVTX breakdown tells you *how much time* each section
takes, but not the *exact sequence* of GPU work within a single step.

### The solution: `--anatomy`

Step anatomy dissects one `env.step` and maps **every GPU kernel** to the
NVTX-labeled code section that launched it, in execution order.

```bash
# Dissect the first step
python scripts/octibenchmark/analyze.py /tmp/bench.nsys-rep --anatomy

# Dissect the 5th step (0-indexed)
python scripts/octibenchmark/analyze.py /tmp/bench.nsys-rep --anatomy --step_index 4
```

**Example output**:
```
Step #0 anatomy — 1174 kernel launches, 18.432 ms total GPU time

  sim.step
      ccd_kernel_6ac72ff8_cuda_kernel_forward               0.698 ms
      ccd_kernel_d1d91ad7_cuda_kernel_forward               0.639 ms
      update_gradient_JTCJ_dense_fad13704_forward           0.034 ms
      linesearch_iterative_kernel_22d13d8d_forward          0.024 ms
                                                   ────────── 8.234 ms (420 kernels)
  UNATTRIBUTED
      vectorized_elementwise_kernel                          0.001 ms
                                                   ────────── 0.001 ms (1 kernels)
  sim.render
      ray_trace_kernel_ab12cd34_forward                      1.203 ms
                                                   ────────── 4.567 ms (312 kernels)
  direct.get_observations
      torch_cat_kernel                                       0.012 ms
      ...
                                                   ────────── 3.210 ms (200 kernels)
  direct.get_rewards
      elementwise_kernel                                     0.008 ms
      ...
```

Kernels that fall between NVTX ranges (launched by code not covered by
any hook) appear as `UNATTRIBUTED` — these are the "mystery kernels".

### How it works internally

Uses the **CUDA correlation chain** in the nsys sqlite database:

```
NVTX range (CPU time)
  └── contains RUNTIME API call (CPU time + correlationId)
        └── launched KERNEL execution (GPU time + correlationId)
```

1. Find the Nth `env.step` NVTX range
2. Find all RUNTIME API calls within that time window
3. Join RUNTIME → KERNEL via `correlationId` to get GPU kernel times
4. LEFT JOIN against all sub-NVTX ranges, picking the **innermost**
   (narrowest) range per kernel via `ROW_NUMBER(...ORDER BY duration ASC)`
5. Return results in CPU execution order (`rt_start`)

This correctly handles asynchronous GPU execution — even though the
kernel runs on GPU *after* the CPU-side launch, the correlation chain
traces it back to the exact NVTX range that issued the launch.

---

## Kernel Pattern Aggregation — `kernel_patterns` Explained

### The problem

A single benchmark run produces hundreds or thousands of distinct GPU kernel
names. For example, a Shadow Hand Vision run (Newton, 64 envs, 10 frames)
launches kernels like:

```
  27.948 ms  x40    ccd_kernel_builder__locals__ccd_kernel_6ac72ff8_cuda_kernel_forward
  25.548 ms  x40    ccd_kernel_builder__locals__ccd_kernel_d1d91ad7_cuda_kernel_forward
  12.562 ms  x372   update_gradient_JTCJ_dense_fad13704_cuda_kernel_forward
   9.622 ms  x332   linesearch_iterative__locals__kernel_22d13d8d_cuda_kernel_forward
   2.679 ms  x372   update_gradient_cholesky__locals__kernel_09e6d52b_cuda_kernel_forward
   1.749 ms  x40    eval_articulation_fk_2fc6eb37_cuda_kernel_forward
   1.183 ms  x636   index_elementwise_kernel
   0.648 ms  x870   vectorized_elementwise_kernel
   0.584 ms  x80    _tile_cholesky_factorize_solve__locals__cholesky_factorize_solve_94a9d168...
   0.516 ms  x587   elementwise_kernel
   0.116 ms  x40    _nxn_broadphase__locals__kernel_ced280db_cuda_kernel_forward
   ... (100+ more)
```

These names are auto-generated by Warp / PyTorch and include hashes. You
can't compare them across runs by exact name. But you often want to answer
questions like:

- "How much total GPU time is spent in **CCD** (continuous collision detection)?"
- "How much time goes to **Cholesky** factorization vs **elementwise** ops?"
- "When comparing Newton vs PhysX, how do **indexed** writes compare to
  **masked** writes?"

### The solution: regex patterns

`kernel_patterns` is a list of **regex strings**. Each pattern is matched
(case-insensitive) against every GPU kernel name. All matching kernels'
times are summed together into one aggregate.

```python
kernel_patterns=["ccd", "cholesky", "index", "elementwise", "broadphase"]
```

This produces:

```
Pattern:              'ccd'   total=  64.034 ms   calls=  120   matching_kernels=3
    - ccd_kernel_builder__locals__ccd_kernel_6ac72ff8_cuda_kernel_forward
    - ccd_kernel_builder__locals__ccd_kernel_d1d91ad7_cuda_kernel_forward
    - ccd_kernel_builder__locals__ccd_kernel_9d89d5cf_cuda_kernel_forward

Pattern:         'cholesky'   total=   3.432 ms   calls=  532   matching_kernels=3
    - update_gradient_cholesky__locals__kernel_09e6d52b_cuda_kernel_forward
    - _tile_cholesky_factorize_solve__locals__cholesky_factorize_solve_94a9d168...
    - _tile_cholesky_factorize_solve__locals__cholesky_factorize_solve_58b9f0a0...

Pattern:            'index'   total=   1.244 ms   calls=  706   matching_kernels=5
    - index_elementwise_kernel
    - reset_wrench_composer_index_0a64ddcb_cuda_kernel_forward
    - write_joint_vel_data_index_69f1049d_cuda_kernel_forward
    ...

Pattern:      'elementwise'   total=   2.671 ms   calls= 2493   matching_kernels=5
    - index_elementwise_kernel
    - vectorized_elementwise_kernel
    - elementwise_kernel
    ...

Pattern:       'broadphase'   total=   0.116 ms   calls=   40   matching_kernels=1
    - _nxn_broadphase__locals__kernel_ced280db_cuda_kernel_forward
```

### How it works internally

`analyze.py` → `aggregate_by_patterns()`:

```python
for pattern in kernel_patterns:
    compiled = re.compile(pattern, re.IGNORECASE)
    for kernel in all_gpu_kernels:
        if compiled.search(kernel["name"]):
            # Add this kernel's total_ns and count to the pattern's aggregate
```

- Each pattern is a Python regex (supports `|`, `.*`, character classes, etc.)
- Matching is **case-insensitive** and uses `re.search` (substring match)
- A single kernel **can match multiple patterns** (e.g. `index_elementwise_kernel`
  matches both `"index"` and `"elementwise"`)
- The output per pattern: `total_ns`, `total_ms`, `count`, `num_unique_kernels`,
  and the list of matched kernel names

### Common use cases

| Question                                    | Patterns                                    |
|---------------------------------------------|---------------------------------------------|
| CCD vs broadphase vs solver cost            | `["ccd", "broadphase", "linesearch", "cholesky"]` |
| Indexed vs masked memory access             | `["index\|scatter\|index_put", "mask\|where"]`    |
| Physics solver breakdown                    | `["ccd", "cholesky", "gradient", "constraint"]`  |
| Rendering cost                              | `["ray_trace\|render", "rasterize"]`              |
| PyTorch overhead vs Warp kernels            | `["elementwise\|reduce\|copy", "cuda_kernel_forward"]` |

### Where patterns are passed

```python
# In BenchmarkMatrix.execute()
matrix.execute(kernel_patterns=["ccd", "index", "cholesky"])

# In sweep.py CLI
python sweep.py --kernel_patterns "ccd" "index" "cholesky" ...

# In analyze.py CLI (single-file analysis)
python analyze.py result.nsys-rep --kernel_patterns "ccd" "index"
```

When using wandb, each pattern becomes a logged metric per run:
`kernel_ms/ccd`, `kernel_ms/index`, etc., enabling charts that compare
kernel categories across num_envs or across physics backends.

---

## How nsys Is Involved

nsys (NVIDIA Nsight Systems) is the **only** measurement tool. octibenchmark
does not use Python timers or `torch.cuda.synchronize()`.

### Capture control

The benchmark scripts use the **cudaProfilerApi** capture mode:

```
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
```

This means nsys starts in a paused state and only records between:
```python
torch.cuda.cudart().cudaProfilerStart()   # start recording
# ... benchmark loop ...
torch.cuda.cudart().cudaProfilerStop()     # stop recording
```

This excludes environment creation, JIT compilation, and warmup from the
profiled data.

### What nsys captures

- `-t cuda,nvtx` tells nsys to capture:
  - **CUDA API calls** and **GPU kernel executions** (timing, grid size, etc.)
  - **NVTX ranges** (the push/pop labels injected by `nvtx_hooks.py`)

### nsys output format

1. `.nsys-rep` — binary report file (viewable in Nsight Systems GUI)
2. `.sqlite` — exported by `analyze.py` via `nsys export --type=sqlite`

### SQLite tables used

| Table                              | What it contains                              |
|------------------------------------|-----------------------------------------------|
| `NVTX_EVENTS`                     | Push/pop ranges with timestamps and labels    |
| `CUPTI_ACTIVITY_KIND_KERNEL`       | Every GPU kernel launch with start/end times  |
| `CUPTI_ACTIVITY_KIND_RUNTIME`      | CUDA runtime API calls (correlationId links to kernels) |
| `StringIds`                        | String interning table (maps IDs to names)    |

---

## How Metrics Are Gathered

### Step 1: NVTX labels are injected at runtime

```python
env = gym.make(task, cfg=env_cfg)
install_nvtx_hooks(env.unwrapped)  # monkey-patches ~20 methods
```

Each patched method now does:
```python
def wrapped_step(*args, **kwargs):
    nvtx.range_push("env.step")       # ~50 ns
    result = original_step(*args, **kwargs)
    nvtx.range_pop()                   # ~50 ns
    return result
```

### Step 2: nsys records GPU activity within NVTX ranges

When the benchmark loop runs, nsys sees:
```
NVTX push "env.step"
    NVTX push "sim.step"
        CUDA kernel: wp_simulate_particles (0.8 ms)
        CUDA kernel: wp_broad_phase (0.3 ms)
    NVTX pop
    NVTX push "sim.render"
        CUDA kernel: wp_ray_trace (1.2 ms)
    NVTX pop
    NVTX push "direct.get_observations"
        CUDA kernel: torch_cat (0.1 ms)
    NVTX pop
NVTX pop
```

### Step 3: analyze.py queries the SQLite DB

```sql
-- NVTX range stats
SELECT text AS name,
       COUNT(*) AS count,
       SUM(end - start) AS total_ns,
       AVG(end - start) AS avg_ns
FROM NVTX_EVENTS
WHERE eventType = 59        -- push/pop ranges only
  AND end IS NOT NULL
  AND text IS NOT NULL
GROUP BY text
ORDER BY total_ns DESC
```

```sql
-- GPU kernel stats (with StringIds join)
SELECT s.value AS name,
       COUNT(*) AS count,
       SUM(k.end - k.start) AS total_ns,
       AVG(k.end - k.start) AS avg_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
GROUP BY s.value
ORDER BY total_ns DESC
```

### Step 4: Results are formatted and logged

**Console output** (from `format_nvtx_table`):
```
NVTX Range                                    Calls    Total ms       Avg ms  % of step
------------------------------------------------------------------------------------------
env.step                                        100     2450.123      24.501     100.0%
sim.step                                        100     1180.456      11.805      48.2%
sim.render                                      100      815.234       8.152      33.3%
direct.get_observations                         100      342.567       3.426      14.0%
vision.compute_image_obs                        100      310.890       3.109      12.7%
direct.get_rewards                              100       45.678       0.457       1.9%
...
```

**wandb logging** (from `BenchmarkMatrix.execute` or `sweep.py`):
- Flat table with all dimensions as columns (task, num_envs, presets,
  resolution, launcher, nvtx_range, total_ms, avg_ms, pct_of_step)
- Per-group line charts: x = num_envs, y = avg_ms per component
- Effective FPS: `num_envs / avg_step_time_seconds`
- Kernel pattern aggregations (e.g., total ms for "index" vs "mask" kernels)

---

## Time Estimates for a Full Benchmark Suite

Based on the Shadow Hand example config (`shadow_hand.py`):

| Matrix                     | Runs | Est. time per run | Est. total |
|----------------------------|------|--------------------|------------|
| `shadow_vision_nonrl`      | 18   | 1–5 min            | 20–90 min  |
| `shadow_vision_bench_nonrl`| 18   | 1–5 min            | 20–90 min  |
| `shadow_state_nonrl`       | 4    | 30–60 s            | 2–4 min    |
| `shadow_vision_train`      | 4    | 3–8 min            | 12–32 min  |
| `shadow_state_train`       | 2    | 1–3 min            | 2–6 min    |
| **Total**                  | **46** |                  | **~1–4 hr** |

Times vary significantly with GPU model, num_envs, and resolution. Large
configurations (8192 envs, 256×256) take much longer than small ones.

---

## File Structure

```
scripts/octibenchmark/
├── __init__.py              # Package marker
├── nvtx_hooks.py            # NVTX injection via monkey-patching
├── benchmark.py             # Non-RL benchmark script (runs under nsys)
├── benchmark_train.py       # RSL-RL training benchmark (runs under nsys)
├── analyze.py               # nsys .nsys-rep → SQLite → structured results + step anatomy
├── bench_cfg.py             # BenchmarkMatrix / BenchmarkRun dataclasses
├── sweep.py                 # Simple CLI-driven num_envs sweep
├── run_matrix.py            # Generic CLI entry point for example matrices
├── test_nvtx_hooks.py       # 10 unit tests for nvtx_hooks
├── test_analyze.py          # 25 unit tests for analyze (incl. step anatomy)
├── test_bench_cfg.py        # 26 unit tests for bench_cfg
├── WORKFLOW.md              # This file
└── examples/
    ├── __init__.py
    └── shadow_hand.py  # Shadow Hand matrix definitions (data only)
```
