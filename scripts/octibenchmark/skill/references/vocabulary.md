# User Question → Function Call Mapping

## What the user says → what to call

| User says | What to call |
|---|---|
| "show me the tree" / "step breakdown" | `query_nvtx_tree(db, step_index)` + `format_tree(tree)` |
| "compare branches" / "what changed" | `compare_trees(tree_a, tree_b)` + `format_comparison(diff)` |
| "what scales poorly" / "scaling analysis" | `analyze_scaling({64: db1, 256: db2, ...})` + `format_scaling(result)` |
| "GPU utilization" / "is GPU busy" | `query_nvtx_tree(db)` then `gpu_utilization(tree)` |
| "hottest path" / "critical path" | `find_hotpath(tree)` |
| "too many kernels" / "launch overhead" | Tree nodes with high `kernel_count`, low avg GPU ns |
| "is step stable" / "warmup" / "outliers" | `query_step_timeseries(db)` + `query_range_distribution(db, name)` |
| "what kernels are in X" / "drill into X" | `query_nvtx_kernel_map(db, step_index)` filtered by path |
| "kernel hardware" / "grid size" | `query_kernel_details(db, name_pattern)` |
| "where is CPU time spent" / "why is X slow" | `query_term_decomposition(db, step, name, include_dispatches=True)` |
| "decompose all terms" / "CPU breakdown" | `query_all_term_decompositions(db, step)` + `format_term_decomposition(...)` |
| "what ops does X run" / "dispatch sequence" | `query_term_decomposition(...)` then iterate `.dispatches` |
| "full report" / "generate all plots" | `report.py /path/to/bench_dir` (auto 5-level report) |
| "flatten to table" | `flatten_tree(tree, max_depth)` |
| "map kernels to code" / "graph-to-code" | `resolve_term_source` → `draft_source_template` → agent refines → `align_dispatches_to_source` |

## Graphing

No pre-built graph functions. Dataclasses are designed to be plot-ready.
Always use `matplotlib.use("Agg")` (no display server needed).

### Proven plot recipes

| Plot | Data source | matplotlib call | Notes |
|---|---|---|---|
| **Scaling log-log** | `ScalingEntry.num_envs` + `.times_ns` | `ax.loglog(num_envs, times_ms, "o-")` | One line per NVTX range, include O(1)/O(n) reference lines |
| **Step timeseries** | `query_step_timeseries(db)` | `ax.plot(steps, times_ms, "o-")` | One subplot per num_envs |
| **GPU utilization bar** | `gpu_utilization(tree)` | `ax.barh(names, ratios)` | Horizontal bars sorted by utilization |
| **Breakdown stacked** | `matrix_results.json` | `ax.bar(x, vals, bottom=bottom)` | Stack top-level ranges |
| **Hot path waterfall** | `find_hotpath(tree)` | `ax.barh(y, wall_ms)` | One bar per depth level |
| **Distribution histogram** | `RangeDistribution.durations_ns` | `ax.hist(durations_ms, bins=30)` | Add lines for mean/p50/p99 |
| **Cross-run comparison** | `TreeDiffEntry` list | `ax.barh(paths, wall_delta_pct)` | Red=slower, green=faster |
| **Source-code map** | `SourceAlignment` + `TermDecomposition` | Timeline bars + code annotations | Color by source region |

### Script pattern for ad-hoc plots

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from octibenchmark.analysis_utils import ...

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 7))
# ... plot ...
fig.savefig("/tmp/bench_dir/plots/name.png", dpi=150, bbox_inches="tight")
```
