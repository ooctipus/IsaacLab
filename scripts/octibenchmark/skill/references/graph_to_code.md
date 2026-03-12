# Graph-to-Code: Mapping CUDA Dispatches to Python Source

## Overview

The graph-to-code pipeline maps individual CUDA kernel dispatches from an nsys
profile back to the exact Python source line that triggered them. This is the
deepest level of performance analysis — understanding not just *what* is slow,
but *which line of code* causes each GPU operation.

## Architecture

The pipeline has 5 phases. Phases 1-3 are fully automated. Phase 4 requires
agent reasoning. Phase 5 is automated.

```
Phase 1: resolve_term_source()     → TermSource (source code + metadata)
Phase 2: query_term_decomposition() → TermDecomposition (dispatch sequence)
Phase 3: draft_source_template()    → list[DraftTemplateLine] (predicted ops)
Phase 4: Agent builds template      → list[SourceMapEntry] (refined mapping)
Phase 5: align_dispatches_to_source() → SourceAlignment + plot
```

## Phase 1: Resolve term to source

```python
import isaaclab_tasks  # must import to register gym envs
from octibenchmark.analysis_utils import resolve_term_source

src = resolve_term_source(
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    "fingers_to_object",
    manager="reward",
)
# src.func_name = "object_ee_distance"
# src.source_file = ".../dexsuite/mdp/rewards.py"
# src.line_number = 34
# src.source_code = "def object_ee_distance(..."
# src.params = {"std": 0.4, "thumb_name": "thumb_link_3_object_s", ...}
```

This uses the gymnasium registry → env config → term config → `inspect` chain.
No environment instantiation needed.

## Phase 2: Get dispatch sequence

```python
from octibenchmark.analysis_utils import query_term_decomposition

decomp = query_term_decomposition(
    db_path, step_index=5,
    range_name="reward.term:fingers_to_object",
    include_dispatches=True,
)
# decomp.dispatches = [DispatchEvent(...), ...]
# Each has: op_name, category, kernel_name, gap_before_ns, gpu_duration_ns
```

## Phase 3: Generate draft template

```python
from octibenchmark.analysis_utils import draft_source_template, format_draft_template

draft = draft_source_template(src.source_code, src.line_number)
print(format_draft_template(draft, decomp.dispatches))
```

This outputs:
1. Source lines with predicted dispatch categories per line
2. The actual dispatch sequence
3. A MISMATCH warning if predicted ≠ actual count
4. Instructions for the agent

## Phase 4: Agent refines the template

**This is the manual bridge.** The agent reads the draft and actual dispatches,
then builds the final `list[SourceMapEntry]`.

### What the agent must do

1. **Identify the mismatch source.** Usually function calls that expand to many
   dispatches. In the `fingers_to_object` example: `contacts()` on line 63
   expands to 24 dispatches (4× sensor reads).

2. **Read called functions.** Follow the call chain:
   - `contacts(env, ...)` → calls `_contact_force_mag()` per sensor
   - `_contact_force_mag()` → `wp.to_torch(...)` + `torch.linalg.norm()`

3. **Spot repeating patterns.** Look for N× identical dispatch sequences:
   ```
   warp: copy_from_sim → warp: sensor_update → warp: outdated_update → torch: reduce → torch: ewise
   ```
   This 5-dispatch pattern repeating 4× = 4 contact sensors in a loop.

4. **Build `SourceMapEntry` list.** One entry per expected dispatch:
   ```python
   template = [
       SourceMapEntry("warp", "L60: wp.to_torch(body_pos_w)", "data"),
       SourceMapEntry("torch_index", "L60: [:, body_ids]", "data"),
       SourceMapEntry("warp", "L61: wp.to_torch(root_pos_w)", "data"),
       SourceMapEntry("torch_ewise", "L62: sub (broadcast)", "distance"),
       SourceMapEntry("torch_reduce", "L62: linalg.norm()", "distance"),
       SourceMapEntry("torch_reduce", "L62: .max()", "distance"),
       # contacts() → thumb
       SourceMapEntry("warp", "L85→69: thumb: copy_from_sim", "contact"),
       SourceMapEntry("warp", "L85→69: thumb: sensor_update", "contact"),
       SourceMapEntry("warp", "L85→69: thumb: outdated_update", "contact"),
       SourceMapEntry("torch_reduce", "L85→70: thumb: norm()", "contact"),
       SourceMapEntry("torch_ewise", "L90: thumb_mag > threshold", "contact"),
       # ... repeat for 3 fingers ...
       # Line 63 continued
       SourceMapEntry("torch_ewise", "L63: .float()", "type_conv"),
       SourceMapEntry("torch_ewise", "L63: .clamp()", "type_conv"),
       # Line 64
       SourceMapEntry("torch_ewise", "L64: distance / std", "reward"),
       SourceMapEntry("torch_ewise", "L64: torch.tanh()", "reward"),
       SourceMapEntry("torch_ewise", "L64: 1 - tanh", "reward"),
       SourceMapEntry("torch_ewise", "L64: * contact_bonus", "reward"),
   ]
   ```

5. **Validate alignment:**
   ```python
   result = align_dispatches_to_source(template, decomp.dispatches)
   print(f"Match: {result.match_pct:.0f}%")  # Target: >90%
   ```

## Phase 5: Generate the plot

The agent writes a matplotlib script that:
1. Draws a dispatch timeline (left panel) with bars colored by source group
2. Shows source code annotations (right panel) linked to dispatch ranges
3. Uses consistent colors per group (`data`, `distance`, `contact`, `reward`)

### Color scheme convention

```python
GROUP_COLORS = {
    "data_extract": "#4CAF50",  # green — wp.to_torch, indexing
    "distance":     "#2196F3",  # blue — norm, max, sub
    "contact":      "#FF9800",  # orange — sensor reads, comparisons
    "type_conv":    "#9C27B0",  # purple — float cast, clamp
    "reward":       "#E91E63",  # pink — tanh, div, mul (final computation)
}
```

## Worked Example: fingers_to_object

The `object_ee_distance` function (reward.term:fingers_to_object) produces
37 dispatches at 1024 envs. The source-to-dispatch mapping reveals:

| Source region | Lines | Dispatches | % of total |
|---|---|---|---|
| Data extraction (wp.to_torch) | L60-61 | #0-#2 (3) | 8% |
| Distance (sub + norm + max) | L62 | #3-#5 (3) | 8% |
| **Contact sensing (4× loop)** | L63→L85-92 | **#6-#29 (24)** | **65%** |
| Type conversion | L63 | #30-#31 (2) | 5% |
| Final reward (tanh kernel) | L64 | #32-#36 (5) | 14% |

**Key insight:** The `contacts()` loop over 4 sensors dominates dispatch count.
Each iteration triggers 5-6 dispatches (3 warp sensor reads + 1 norm + 1-2 comparisons).
Fusing the contact loop into a single batched operation would cut dispatch count
from 37 to ~16.

## Why full automation is hard

The core challenge is **predicting how many CUDA kernels each Python expression
dispatches**. This depends on:

- **Runtime state**: `wp.to_torch()` may trigger 0-3 warp kernels depending on
  data cache state and lazy evaluation
- **Operator fusion**: `torch.linalg.norm(x, dim=-1)` could be 1 kernel (fused)
  or 3 (square→sum→sqrt)
- **Function calls**: `contacts(env, ...)` expands to 24 dispatches because it
  loops over 4 sensors — this can't be determined without reading the called function

The semi-automated approach is the right trade-off: the automated phases handle
everything mechanical, and the agent handles pattern matching (which it excels at).
Building a template takes ~2-3 minutes once you have the source.
