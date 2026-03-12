# Analysis Utilities Architecture

## Module: `analysis_utils.py`

Composable Python functions for deep nsys profiling analysis. All functions
return dataclasses for programmatic use and visualization.

## Three-Layer Design

### Layer 1: Single-profile extraction (from one sqlite)

| Function | Returns | Purpose |
|---|---|---|
| `query_nvtx_tree(db, step_index, attribute_kernels)` | `NvtxNode` | Build hierarchical NVTX range tree within one step |
| `query_step_timeseries(db, range_names)` | `list[StepSnapshot]` | Per-step timing across all captured steps |
| `query_range_distribution(db, range_name)` | `RangeDistribution` | Statistical distribution of one range |
| `query_kernel_details(db, name_pattern, step_index)` | `list[KernelDetail]` | Per-launch kernel data with hardware info |
| `query_nvtx_kernel_map(db, step_index, full_path)` | `dict[str, list[KernelEntry]]` | NVTX path → attributed kernel list |
| `query_term_decomposition(db, step, name, include_dispatches)` | `TermDecomposition` | CPU time breakdown of one NVTX range |
| `query_all_term_decompositions(db, step, min_depth, max_depth)` | `list[TermDecomposition]` | Batch decompose all ranges at given depth |
| `flatten_tree(tree, max_depth, min_wall_ns)` | `list[FlatRow]` | Tree → flat rows with slash-separated paths |
| `align_dispatches_to_source(template, dispatches)` | `SourceAlignment` | Match dispatch sequence to source code template |

### Layer 2: Cross-profile analysis (from multiple sqlites)

| Function | Returns | Purpose |
|---|---|---|
| `compare_trees(tree_a, tree_b, label_a, label_b)` | `TreeDiffResult` | Match nodes by path, compute diffs |
| `analyze_scaling(profiles, range_names)` | `ScalingResult` | Per-range scaling exponent across num_envs |

### Layer 3: Derived metrics

| Function | Returns | Purpose |
|---|---|---|
| `find_hotpath(tree, metric)` | `list[NvtxNode]` | Follow heaviest child at each level |
| `gpu_utilization(tree)` | `dict[str, float]` | GPU/wall ratio per node |
| `scaling_bottlenecks(scaling, threshold)` | `list[ScalingEntry]` | Filter for super-linear scaling |

## Core Data Structures

### NvtxNode (NVTX tree node)

```python
NvtxNode:
    name: str                   # "sim.step"
    start_ns, end_ns: int       # absolute timestamps
    wall_ns: int                # end - start
    depth: int                  # 0 = root
    children: list[NvtxNode]
    kernel_count: int           # directly attributed kernels
    direct_kernel_gpu_ns: int   # GPU time of direct kernels

    # Computed properties:
    subtree_kernel_gpu_ns: int  # GPU time including all descendants
    subtree_kernel_count: int   # kernel count including all descendants
```

### TreeDiffEntry (cross-profile comparison)

```python
TreeDiffEntry:
    path: str                   # "env.step/sim.step"
    status: str                 # "matched" | "only_a" | "only_b"
    a_wall_ns, b_wall_ns: int
    a_gpu_ns, b_gpu_ns: int
    a_kernel_count, b_kernel_count: int
    wall_delta_pct: float       # positive = B slower
    gpu_delta_pct: float
```

### ScalingEntry (scaling analysis)

```python
ScalingEntry:
    range_name: str
    num_envs: list[int]         # [64, 256, 1024]
    times_ns: list[float]       # corresponding avg wall times
    exponent: float             # log-log slope
    r_squared: float            # fit quality
    classification: str         # "sub-linear" | "linear" | "super-linear" | "constant"
```

### TermDecomposition (CPU time breakdown)

```python
TermDecomposition:
    name: str                   # NVTX range label
    depth: int                  # nesting depth
    wall_ns: int                # total wall clock
    n_dispatches: int           # number of CUDA kernel dispatches
    pre_python_ns: int          # Python before first dispatch
    between_python_ns: int      # Python between dispatches (usually dominant)
    api_overhead_ns: int        # CUDA API call overhead
    post_python_ns: int         # Python after last dispatch
    gpu_kernel_ns: int          # total GPU execution time
    avg_gap_ns: float           # average gap between dispatches
    max_gap_ns: int             # largest gap
    dispatches: list[DispatchEvent]  # per-dispatch details (if include_dispatches=True)

    # Properties:
    python_total_ns: int        # pre + between + post
    python_pct: float           # python overhead as % of wall time
    between_pct: float          # between-dispatch overhead as % of wall time
```

### DispatchEvent (single CUDA dispatch)

```python
DispatchEvent:
    op_name: str                # "torch: cross (cross product)" or "warp: extract_position"
    category: str               # "torch_cross", "warp", "torch_reduce", etc.
    kernel_name: str            # raw demangled CUDA kernel name
    api_start_ns: int           # relative to range start
    api_end_ns: int
    gpu_duration_ns: int        # actual GPU execution time
    gap_before_ns: int          # Python gap before this dispatch
```

### classify_kernel (kernel name → op mapping)

`classify_kernel(kernel_name)` → `(op_name, category)` maps demangled CUDA kernel
names to human-readable op names. Categories:
- `warp` — IsaacLab/Warp custom kernels (`*_cuda_kernel_forward`, `fused_*`)
- `torch_reduce` — sum, mean, norm reductions
- `torch_binary` — arithmetic (+, -, *, /)
- `torch_ewise` — elementwise ops (clamp, where)
- `torch_cross` — cross product
- `torch_cat` — tensor concatenation
- `torch_index` — index_select, scatter, gather
- `torch_compare` — comparison ops (>, <, ==)
- `torch_copy`, `torch_fill`, `torch_math`, `torch_unary`, `torch_bool`, `other`

### SourceMapEntry + SourceAlignment (graph-to-code mapping)

```python
SourceMapEntry:
    category_pattern: str   # regex matched against DispatchEvent.category
    source_label: str       # "L60: wp.to_torch(body_pos_w)"
    group: str              # coloring key ("data_extract", "contact", etc.)

SourceAlignment:
    entries: list[tuple[SourceMapEntry | None, int]]  # (template_entry, dispatch_index)
    matched: int            # dispatches matched
    total: int              # total dispatches
    match_pct: float        # percentage matched
```

**Workflow for graph-to-code:**
1. Read the term's source function
2. Build a `list[SourceMapEntry]` template — one entry per expected CUDA dispatch,
   annotated with the source line that triggers it
3. Call `align_dispatches_to_source(template, decomp.dispatches)`
4. Use the alignment to generate color-coded dispatch-to-source plots

For function calls (e.g., `contacts()` inside a reward term), expand the called
function inline in the template. The greedy alignment handles minor mismatches.

## Tree Building Algorithm

`query_nvtx_tree` builds the NVTX range hierarchy using timestamp containment:

1. Query all NVTX ranges within the target `env.step`
2. Sort by `(start ASC, duration DESC)` — wider ranges first at same start
3. Stack-based tree construction:
   - For each range: pop nodes that don't contain it, current top is parent
   - Containment check: `parent.start <= range.start AND parent.end >= range.end`
4. If `attribute_kernels=True`: query correlation chain, walk each kernel to innermost tree node

## Composition Patterns

```python
# Find worst step, drill into it
series = query_step_timeseries(db)
worst = max(series, key=lambda s: s.wall_ns)
tree = query_nvtx_tree(db, step_index=worst.step_index)
print(format_tree(tree))

# Cross-branch comparison
tree_old = query_nvtx_tree(db_old, step_index=5)
tree_new = query_nvtx_tree(db_new, step_index=5)
diff = compare_trees(tree_old, tree_new, "v1.0", "v2.0")
print(format_comparison(diff))

# Scaling analysis
result = analyze_scaling({64: db_64, 256: db_256, 1024: db_1024})
print(format_scaling(result))
bottlenecks = scaling_bottlenecks(result, exponent_threshold=1.2)

# GPU utilization
tree = query_nvtx_tree(db, attribute_kernels=True)
util = gpu_utilization(tree)
for name, ratio in sorted(util.items(), key=lambda x: x[1]):
    print(f"{name}: {ratio:.1%} GPU utilization")

# Cross-scale comparison (what grew most from small→large)
tree_small = query_nvtx_tree(db_small, step_index=5)
tree_large = query_nvtx_tree(db_large, step_index=5)
diff = compare_trees(tree_small, tree_large, "64envs", "1024envs")
# Sort by wall_delta_pct to find what grew fastest
for e in sorted(diff.entries, key=lambda e: abs(e.wall_delta_pct), reverse=True)[:10]:
    print(f"{e.path}: {e.wall_delta_pct:+.1f}%")
```

## Practical Notes

### DB path patterns

After `run_matrix.py`, sqlite files follow the naming pattern:
```
{task}__{envs{N}}__{preset_values}__{launcher}.sqlite
```
Build the `{num_envs: path}` dict by globbing:
```python
import glob
dbs = {}
for f in glob.glob("/tmp/bench_dir/*.sqlite"):
    # Extract num_envs from filename
    for part in f.split("__"):
        if part.startswith("envs"):
            dbs[int(part[4:])] = f
```

### Step index selection

- `step_index=0` is often a warmup outlier — prefer `step_index=5` or later for tree analysis
- For stability analysis, use `query_step_timeseries` to see all steps first

### matrix_results.json

This file contains pre-aggregated per-run NVTX range metrics (count, total_ns, avg_ns).
Useful for quick stacked bar charts without loading sqlite databases. Keys are run tags.
