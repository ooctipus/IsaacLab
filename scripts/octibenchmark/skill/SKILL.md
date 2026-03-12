---
name: octibenchmark
description: >
  IsaacLab GPU performance benchmarking and nsys profiling framework.
  Use when user wants to: benchmark IsaacLab environments, profile GPU performance,
  measure env step timing, create benchmark matrices, analyze nsys reports, add NVTX hooks,
  generate performance plots, run scaling analysis, compare branches, investigate CPU vs GPU
  time breakdown, map CUDA kernels to source code lines, or generate a full performance report.
  Also use when working on files in scripts/octibenchmark/.
metadata:
  author: IsaacLab
  version: 2.0.0
---

# octibenchmark Skill

You are an expert in the octibenchmark framework — an NVIDIA nsys-based performance
profiling system for IsaacLab environments. All instrumentation is injected at runtime
via monkey-patching; **no IsaacLab source files are modified**.

## Critical Context

Before working on any octibenchmark task, understand the architecture by consulting
`references/architecture.md`. For code conventions, see `references/conventions.md`.
For the analysis utilities API, see `references/analysis_utils.md`.

## Environment Setup

**Always use the project's Python environment**, not system Python:

```bash
# Preferred: explicit env_isaaclab Python with PYTHONPATH
PYTHONPATH=scripts:$PYTHONPATH env_isaaclab/bin/python <script.py>

# Alternative: isaaclab.sh wrapper
./isaaclab.sh -p scripts/octibenchmark/<script.py> [args]
```

**IMPORTANT**: `./isaaclab.sh -p` may resolve to the wrong Python if not configured.
For ad-hoc analysis scripts, always use `env_isaaclab/bin/python` directly.

## Key Files

| File | Purpose |
|---|---|
| `nvtx_hooks.py` | NVTX annotation injection via monkey-patching |
| `benchmark.py` | Non-RL benchmark (random actions under nsys) |
| `benchmark_train.py` | RSL-RL training benchmark (under nsys) |
| `analyze.py` | nsys `.nsys-rep` parser; SQLite queries for metrics |
| `analysis_utils.py` | Composable analysis: tree, scaling, comparison, decomposition, source mapping |
| `report.py` | Auto-generate 6-level performance report from benchmark sqlite files |
| `bench_cfg.py` | Declarative benchmark matrix config (`BenchmarkMatrix`, `BenchmarkRun`) |
| `run_matrix.py` | Generic CLI entry point for example matrices |
| `examples/` | Pre-built benchmark matrices (e.g., `kuka_allegro.py`) |

## Instructions

### Step 1: Run benchmarks

```bash
# Dry-run first to verify the matrix
./isaaclab.sh -p scripts/octibenchmark/run_matrix.py --example kuka_allegro --dry_run

# Run for real
./isaaclab.sh -p scripts/octibenchmark/run_matrix.py \
    --example kuka_allegro --output_dir /tmp/kuka_bench
```

Output: `.nsys-rep` + `.sqlite` + `matrix_results.json` per run.
Some preset combinations may fail — proceed with successful runs.

### Step 2: Generate full report (one command)

```bash
PYTHONPATH=scripts:$PYTHONPATH env_isaaclab/bin/python \
    scripts/octibenchmark/report.py /tmp/kuka_bench --step_index 5 --top_terms 5
```

This auto-generates a 5-level plot hierarchy:

```
plots/
├── 1_scaling/           # Scaling loglog, step timeseries, stacked breakdown
├── 2_step_overview/     # CPU/GPU breakdown, utilization, hotpath, kernel density
├── 3_manager_breakdown/ # Term decomposition, gap analysis, per-manager timelines
├── 4_term_ops/          # Op-by-op timeline for top N terms
└── 5_kernel_micro/      # Raw microsecond dispatch patterns
```

### Step 3: Display and interpret

Show plots from each level to the user using the Read tool on the PNG files.
Walk through the 5 levels in order:

1. **Scaling**: Any ranges with exponent > 1.0? (super-linear = bottleneck)
2. **Step overview**: CPU vs GPU split, GPU utilization %, hot path
3. **Manager breakdown**: Which terms are CPU-dominated? Dispatch patterns?
4. **Term ops**: What PyTorch/Warp ops does each term execute?
5. **Kernel micro**: Raw dispatch pattern — many tiny ops or few large ones?

### Step 4: Graph-to-code mapping (semi-automated)

When the user wants to map CUDA kernel dispatches back to Python source lines:

```python
import isaaclab_tasks  # registers gym envs
from octibenchmark.analysis_utils import (
    resolve_term_source, draft_source_template, format_draft_template,
    query_term_decomposition, align_dispatches_to_source, SourceMapEntry,
)

# Phase 1-3: Fully automated
src = resolve_term_source("Isaac-Task-v0", "term_name", manager="reward")
decomp = query_term_decomposition(db, step_index=5,
    range_name="reward.term:term_name", include_dispatches=True)
draft = draft_source_template(src.source_code, src.line_number)
print(format_draft_template(draft, decomp.dispatches))
```

The draft output shows predicted vs actual dispatches and flags function calls
that need expansion. **The agent then:**

1. Reads the draft — notes the mismatch count
2. Finds flagged function calls and reads their source
3. Spots repeating patterns in dispatch sequence (e.g., 4× sensor loop)
4. Builds final `list[SourceMapEntry]` with one entry per dispatch
5. Runs `align_dispatches_to_source()` — target >90% match
6. Generates color-coded timeline plot with matplotlib

See `references/graph_to_code.md` for the full workflow with a worked example.

### Step 5: Ad-hoc deep dives

For specific questions, use `analysis_utils` functions directly.
See `references/analysis_utils.md` for the full API and `references/vocabulary.md`
for a mapping of user questions to function calls.

## Adding New Benchmark Matrices

1. Create `examples/my_task.py` with `BenchmarkMatrix` instances.
2. Export in `ALL_MATRICES` dict at module level.
3. Verify: `run_matrix.py --example my_task --dry_run`.
4. See `references/conventions.md` for naming and tagging rules.

## Adding New NVTX Hooks

1. Framework-level: add to `nvtx_hooks.py` using `_wrap(obj, attr, label)`.
2. Task-specific: use `extra_nvtx_hooks` in the matrix definition.
3. Always add tests in `test_nvtx_hooks.py`.

## Running Tests

```bash
./isaaclab.sh -p -m pytest scripts/octibenchmark/ -v
```

## Troubleshooting

**nsys not found**: Ensure NVIDIA Nsight Systems is installed and `nsys` is on PATH.

**Empty NVTX ranges**: Check that `install_nvtx_hooks()` was called on the unwrapped env.

**Schema mismatch**: The analyzer auto-detects nsys SQLite schema versions.
If you get column errors, check which nsys version produced the report.

**Wrong Python**: Use `env_isaaclab/bin/python` directly instead of `./isaaclab.sh -p`.

**Preset failures**: Not all preset combinations work for all tasks.
E.g., `presets=cube` alone fails for Kuka Allegro — use `presets=newton,cube`.

## Examples

### Example 1: Full performance investigation

User says: "Benchmark Kuka Allegro and show me where time is spent"

1. Run matrix: `run_matrix.py --example kuka_allegro --output_dir /tmp/bench`
2. Generate report: `report.py /tmp/bench --step_index 5 --top_terms 5`
3. Walk through 5-level plots, summarize findings at each level

### Example 2: Compare two branches

User says: "What got faster between old and new code?"

1. Build NVTX trees from both sqlite files
2. `compare_trees(tree_old, tree_new, "v1", "v2")`
3. Show wall/GPU time deltas sorted by absolute change

### Example 3: Map kernel to code

User says: "Show me which code line in fingers_to_object causes the most CPU time"

1. `resolve_term_source("Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "fingers_to_object")`
2. `query_term_decomposition(db, 5, "reward.term:fingers_to_object", include_dispatches=True)`
3. `draft_source_template()` → agent refines → `align_dispatches_to_source()`
4. Generate annotated timeline plot

### Example 4: Scaling analysis

User says: "What doesn't scale with num_envs?"

1. `analyze_scaling({64: db_64, 256: db_256, 1024: db_1024})`
2. Flag ranges with exponent > 1.0 (super-linear)
3. Drill into bottlenecks with `compare_trees` at different scale points
