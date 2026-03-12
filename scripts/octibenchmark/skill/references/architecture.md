# octibenchmark Architecture

## Execution Flow

```
User entry point (run_matrix.py / sweep.py / CLI)
  │
  │ generates BenchmarkRun list
  ▼
bench_cfg.py — BenchmarkMatrix.execute()
  For each run:
    1. Build nsys command via BenchmarkRun.nsys_command()
    2. Launch subprocess: nsys profile → benchmark.py / benchmark_train.py
    3. Collect .nsys-rep file
    4. Parse via analyze.py
    5. Log to wandb
  │
  │ subprocess per run
  ▼
benchmark.py or benchmark_train.py (runs inside nsys)
    1. Create env via gym.make()
    2. install_nvtx_hooks(env.unwrapped) — monkey-patches methods
    3. Warmup loop (outside capture)
    4. cudaProfilerStart() → benchmark loop → cudaProfilerStop()
    5. Close environment
  │
  │ .nsys-rep output
  ▼
analyze.py
    1. nsys export --type=sqlite → .sqlite
    2. SQL queries on NVTX_EVENTS, CUPTI_ACTIVITY_KIND_KERNEL tables
    3. Structured results (NVTX ranges, GPU kernels, kernel patterns, step anatomy)
```

## Core Classes (bench_cfg.py)

### Enums

- `Launcher.NON_RL` — Random action stepping (`benchmark.py`)
- `Launcher.RSL_RL` — RSL-RL training loop (`benchmark_train.py`)
- `Phase.STEP` — Profile stepping loop only (default)
- `Phase.STARTUP` — Profile environment/runner creation only

### BenchmarkRun

A single resolved benchmark invocation. Key attributes:
- `task`, `num_envs`, `hydra_overrides` (dict), `launcher`, `phase`
- `num_frames`, `warmup_frames`, `max_iterations`

Key methods:
- `tag` property — Compact unique filename/wandb label
- `all_hydra_args` property — Merges sweep overrides + constant hydra_args
- `script_path` property — Returns benchmark.py or benchmark_train.py
- `nsys_command(output_path)` — Builds full nsys + python command list

### BenchmarkMatrix

Declarative specification of a benchmark sweep. Key fields:
- `tasks` — List of task names
- `num_envs` — List of environment counts (always swept)
- `hydra_sweeps` — Dict of named axes → lists of Hydra override strings
- `hydra_args` — Constant overrides applied to every run
- `launcher`, `phase`, `num_frames`, `warmup_frames`, `max_iterations`
- `extra_nvtx_hooks` — Task-specific NVTX instrumentation

Key methods:
- `runs()` — Generates cross-product of all axes as BenchmarkRun list
- `execute(output_dir, wandb_project, ...)` — Orchestrates all runs, logs results

## NVTX Hook Injection (nvtx_hooks.py)

Monkey-patches environment methods with NVTX push/pop ranges. ~100 ns overhead per pair.

### Common hooks (all env types)
- `env.step`, `sim.step`, `sim.render`
- `scene.write_data_to_sim`, `scene.update`
- `env._reset_idx`

### Manager-based environment hooks
- Action, reward, observation, termination, command, event, recorder, curriculum managers
- **Per-term hooks**: Individual terms get separate NVTX ranges
  (e.g., `reward.term:reach_distance`, `observation.term[proprioceptive]:joint_positions`)

### Direct RL environment hooks
- `_pre_physics_step`, `_apply_action`, `_get_rewards`, `_get_dones`, `_get_observations`

### RSL-RL algorithm hooks (benchmark_train.py only)
- `runner.alg.act`, `runner.alg.process_env_step`, `runner.alg.update`, `runner.alg.compute_returns`

## Analysis Pipeline (analyze.py)

### Data extraction from nsys SQLite
1. **NVTX ranges** — Per range: name, count, total_ns, avg_ns
2. **GPU kernels** — Per kernel: demangled name, count, total_ns, avg_ns
3. **Kernel pattern aggregations** — Regex matching on kernel names (case-insensitive `re.search`)
4. **Step anatomy** — Maps GPU kernels to NVTX code sections using CUDA correlation chain:
   ```
   NVTX range (CPU time)
     └── contains RUNTIME API call (CPU time + correlationId)
           └── launched KERNEL execution (GPU time + correlationId)
   ```

### Schema auto-detection
- Newer nsys: `start`/`end` columns with inline `text`
- Older nsys: `startTimestamp`/`endTimestamp` with `textId` → `StringIds` join

## wandb Integration

- One wandb **group** per matrix execution
- One wandb **run** per config (unique task/hydra_overrides/launcher combo)
- Metrics: per-NVTX total_ms/avg_ms/pct_of_step, per-pattern total_ms, effective FPS
