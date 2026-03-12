# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composable analysis utilities for octibenchmark nsys profiling data.

Provides structured data extraction, cross-profile comparison, and scaling
analysis on nsys sqlite databases. All functions return dataclasses for
easy programmatic use and visualization.

The utilities are organized in three layers:

**Layer 1 — Single-profile extraction:**
    :func:`query_nvtx_tree`, :func:`query_step_timeseries`,
    :func:`query_range_distribution`, :func:`query_kernel_details`,
    :func:`query_nvtx_kernel_map`, :func:`flatten_tree`,
    :func:`query_term_decomposition`, :func:`query_all_term_decompositions`

**Layer 2 — Cross-profile analysis:**
    :func:`compare_trees`, :func:`analyze_scaling`

**Layer 3 — Derived metrics:**
    :func:`find_hotpath`, :func:`gpu_utilization`, :func:`scaling_bottlenecks`
"""

from __future__ import annotations

import dataclasses
import math
import re
import sqlite3
import statistics

from octibenchmark.analyze import query_nvtx_ranges

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NvtxNode:
    """A node in the NVTX range hierarchy tree.

    Each node represents a single NVTX push/pop range occurrence within one
    ``env.step``. Children are ranges whose ``[start, end]`` interval is
    strictly contained within this node's interval.
    """

    name: str
    """NVTX range label (e.g. ``"sim.step"``)."""
    start_ns: int
    """Absolute start timestamp [ns]."""
    end_ns: int
    """Absolute end timestamp [ns]."""
    wall_ns: int
    """Wall-clock duration [ns] (``end_ns - start_ns``)."""
    depth: int
    """Nesting depth (0 = root)."""
    children: list[NvtxNode] = dataclasses.field(default_factory=list)
    """Child NVTX ranges contained within this range."""
    kernel_count: int = 0
    """Number of GPU kernels directly attributed to this range (not descendants)."""
    direct_kernel_gpu_ns: int = 0
    """Total GPU time [ns] of kernels directly attributed to this range."""

    @property
    def subtree_kernel_gpu_ns(self) -> int:
        """Total GPU time [ns] of all kernels in this subtree."""
        return self.direct_kernel_gpu_ns + sum(c.subtree_kernel_gpu_ns for c in self.children)

    @property
    def subtree_kernel_count(self) -> int:
        """Total number of GPU kernels in this subtree."""
        return self.kernel_count + sum(c.subtree_kernel_count for c in self.children)


@dataclasses.dataclass
class StepSnapshot:
    """Timing data for a single ``env.step`` occurrence."""

    step_index: int
    """Zero-based step index."""
    wall_ns: int
    """Wall-clock duration of the ``env.step`` range [ns]."""
    start_ns: int
    """Absolute start timestamp [ns]."""
    end_ns: int
    """Absolute end timestamp [ns]."""
    range_times: dict[str, int] = dataclasses.field(default_factory=dict)
    """Map of NVTX range name to total wall time [ns] within this step."""


@dataclasses.dataclass
class RangeDistribution:
    """Statistical distribution of a single NVTX range across all occurrences."""

    name: str
    """NVTX range label."""
    count: int
    """Number of occurrences."""
    mean_ns: float
    """Mean duration [ns]."""
    stddev_ns: float
    """Standard deviation [ns]."""
    min_ns: int
    """Minimum duration [ns]."""
    max_ns: int
    """Maximum duration [ns]."""
    p50_ns: int
    """50th percentile (median) [ns]."""
    p90_ns: int
    """90th percentile [ns]."""
    p99_ns: int
    """99th percentile [ns]."""
    durations_ns: list[int] = dataclasses.field(default_factory=list)
    """Raw per-invocation durations [ns] in occurrence order."""


@dataclasses.dataclass
class KernelDetail:
    """Per-invocation GPU kernel data with hardware info."""

    name: str
    """Demangled kernel name."""
    start_ns: int
    """GPU execution start timestamp [ns]."""
    end_ns: int
    """GPU execution end timestamp [ns]."""
    duration_ns: int
    """GPU execution duration [ns]."""
    grid: tuple[int, int, int] = (0, 0, 0)
    """Grid dimensions (x, y, z)."""
    block: tuple[int, int, int] = (0, 0, 0)
    """Block dimensions (x, y, z)."""
    registers_per_thread: int = 0
    """Registers used per thread."""
    static_shared_memory_bytes: int = 0
    """Static shared memory [B]."""
    dynamic_shared_memory_bytes: int = 0
    """Dynamic shared memory [B]."""
    stream_id: int = 0
    """CUDA stream ID."""
    device_id: int = 0
    """GPU device ID."""


@dataclasses.dataclass
class KernelEntry:
    """Lightweight kernel record for NVTX-to-kernel mapping."""

    name: str
    """Demangled kernel name."""
    duration_ns: int
    """GPU execution duration [ns]."""


@dataclasses.dataclass
class FlatRow:
    """One row of a flattened NVTX tree, suitable for table display."""

    path: str
    """Full slash-separated NVTX path (e.g. ``"env.step/sim.step"``)."""
    name: str
    """Leaf name of this range."""
    depth: int
    """Nesting depth (0 = root)."""
    wall_ns: int
    """Wall-clock duration [ns]."""
    kernel_count: int
    """Directly attributed GPU kernels."""
    direct_kernel_gpu_ns: int
    """GPU time of directly attributed kernels [ns]."""
    subtree_kernel_gpu_ns: int
    """GPU time of all kernels in this subtree [ns]."""
    pct_of_parent: float
    """Wall time as percentage of parent's wall time."""


@dataclasses.dataclass
class TreeDiffEntry:
    """One NVTX path's data in a cross-profile tree comparison."""

    path: str
    """Full slash-separated NVTX path."""
    status: str
    """One of ``"matched"``, ``"only_a"``, ``"only_b"``."""
    a_wall_ns: int = 0
    """Wall time [ns] in profile A."""
    b_wall_ns: int = 0
    """Wall time [ns] in profile B."""
    a_gpu_ns: int = 0
    """Subtree GPU time [ns] in profile A."""
    b_gpu_ns: int = 0
    """Subtree GPU time [ns] in profile B."""
    a_kernel_count: int = 0
    """Subtree kernel count in profile A."""
    b_kernel_count: int = 0
    """Subtree kernel count in profile B."""
    wall_delta_pct: float = 0.0
    """Wall time percentage change relative to A. Positive = B slower."""
    gpu_delta_pct: float = 0.0
    """GPU time percentage change relative to A. Positive = B slower."""


@dataclasses.dataclass
class TreeDiffResult:
    """Diff of two nsys profiles via NVTX tree comparison."""

    label_a: str
    """Label for profile A."""
    label_b: str
    """Label for profile B."""
    entries: list[TreeDiffEntry] = dataclasses.field(default_factory=list)
    """Per-path comparisons, sorted by absolute wall delta descending."""


@dataclasses.dataclass
class ScalingEntry:
    """Scaling data for a single NVTX range across ``num_envs`` values."""

    range_name: str
    """NVTX range label."""
    num_envs: list[int]
    """Environment counts (x-axis)."""
    times_ns: list[float]
    """Corresponding average wall times [ns] (y-axis)."""
    exponent: float
    """Slope of ``log(time)`` vs ``log(num_envs)`` fit."""
    r_squared: float
    """Coefficient of determination of the log-log fit."""
    classification: str
    """One of ``"sub-linear"``, ``"linear"``, ``"super-linear"``, ``"constant"``."""


@dataclasses.dataclass
class ScalingResult:
    """Scaling analysis across multiple ``num_envs`` values."""

    entries: list[ScalingEntry] = dataclasses.field(default_factory=list)
    """Per-range scaling data, sorted by exponent descending (worst first)."""


@dataclasses.dataclass
class DispatchEvent:
    """A single CUDA kernel dispatch within an NVTX range."""

    op_name: str
    """Human-readable operation name (e.g. 'torch: cross product')."""
    category: str
    """Op category (e.g. 'warp', 'torch_reduce', 'torch_ewise')."""
    kernel_name: str
    """Demangled CUDA kernel name."""
    api_start_ns: int
    """CUDA API call start timestamp relative to range start [ns]."""
    api_end_ns: int
    """CUDA API call end timestamp relative to range start [ns]."""
    gpu_duration_ns: int
    """GPU kernel execution duration [ns]."""
    gap_before_ns: int
    """Python gap before this dispatch [ns] (time since previous dispatch ended or range start)."""


@dataclasses.dataclass
class TermDecomposition:
    """CPU time breakdown of a single NVTX range using CUDA API timestamps."""

    name: str
    """NVTX range label."""
    depth: int
    """Nesting depth within the step."""
    wall_ns: int
    """Total wall-clock duration [ns]."""
    n_dispatches: int
    """Number of CUDA kernel dispatches."""
    pre_python_ns: int
    """Python time before first dispatch [ns]."""
    between_python_ns: int
    """Python time between dispatches [ns]."""
    api_overhead_ns: int
    """Total CUDA API call overhead [ns]."""
    post_python_ns: int
    """Python time after last dispatch [ns]."""
    gpu_kernel_ns: int
    """Total GPU kernel execution time [ns]."""
    avg_gap_ns: float
    """Average gap between dispatches [ns]. 0 if < 2 dispatches."""
    max_gap_ns: int
    """Largest gap between dispatches [ns]. 0 if < 2 dispatches."""
    dispatches: list[DispatchEvent] = dataclasses.field(default_factory=list)
    """Ordered list of dispatch events (populated if include_dispatches=True)."""

    @property
    def python_total_ns(self) -> int:
        """Total Python overhead [ns] (pre + between + post)."""
        return self.pre_python_ns + self.between_python_ns + self.post_python_ns

    @property
    def python_pct(self) -> float:
        """Python overhead as percentage of wall time."""
        return self.python_total_ns / self.wall_ns * 100 if self.wall_ns > 0 else 0.0

    @property
    def between_pct(self) -> float:
        """Between-dispatch Python time as percentage of wall time."""
        return self.between_python_ns / self.wall_ns * 100 if self.wall_ns > 0 else 0.0


@dataclasses.dataclass
class SourceMapEntry:
    """Maps a dispatch category pattern to a source code location.

    Used to build a "dispatch template" that aligns CUDA dispatches to
    the Python source lines that triggered them.
    """

    category_pattern: str
    """Regex matched against ``DispatchEvent.category`` (e.g. ``"warp"``, ``"torch_reduce"``)."""
    source_label: str
    """Human-readable label (e.g. ``"L60: wp.to_torch(body_pos_w)"``)."""
    group: str
    """Grouping key for coloring (e.g. ``"data_extract"``, ``"contact"``)."""


@dataclasses.dataclass
class SourceAlignment:
    """Result of aligning a dispatch template to actual dispatches."""

    entries: list[tuple[SourceMapEntry | None, int]]
    """``(template_entry, dispatch_index)`` pairs. ``None`` for unmatched."""
    matched: int
    """Number of dispatches that matched a template entry."""
    total: int
    """Total number of actual dispatches."""

    @property
    def match_pct(self) -> float:
        """Percentage of dispatches matched."""
        return self.matched / self.total * 100 if self.total > 0 else 0.0


def align_dispatches_to_source(
    template: list[SourceMapEntry],
    dispatches: list[DispatchEvent],
) -> SourceAlignment:
    """Align a dispatch template to actual CUDA dispatches using greedy forward matching.

    Walks the dispatch sequence and the template in parallel. For each dispatch,
    checks if its category matches the current template entry's pattern. On match,
    both advance; on mismatch, only the dispatch index advances (marked unmatched).

    Args:
        template: Expected dispatch sequence with source annotations.
        dispatches: Actual dispatch events from :func:`query_term_decomposition`.

    Returns:
        Alignment result with per-dispatch source mapping.
    """
    aligned: list[tuple[SourceMapEntry | None, int]] = []
    t_idx = 0

    for d_idx, d in enumerate(dispatches):
        if t_idx < len(template):
            tmpl = template[t_idx]
            if re.search(tmpl.category_pattern, d.category):
                aligned.append((tmpl, d_idx))
                t_idx += 1
                continue
        aligned.append((None, d_idx))

    matched = sum(1 for t, _ in aligned if t is not None)
    return SourceAlignment(entries=aligned, matched=matched, total=len(dispatches))


@dataclasses.dataclass
class TermSource:
    """Resolved source code for a manager term."""

    term_name: str
    """NVTX term name (e.g. ``"fingers_to_object"``)."""
    func_name: str
    """Python function name (e.g. ``"object_ee_distance"``)."""
    source_file: str
    """Absolute path to the source file."""
    line_number: int
    """Line number where the function is defined."""
    source_code: str
    """Full source code of the function."""
    params: dict
    """Parameters from the term config."""


@dataclasses.dataclass
class DraftTemplateLine:
    """One line from an auto-generated draft dispatch template."""

    line_no: int
    """Source line number."""
    code: str
    """The Python source line."""
    predicted_ops: list[tuple[str, str]]
    """List of (category_pattern, note) — predicted CUDA dispatches."""


# Patterns for detecting torch/warp ops in source lines.
# Each: (regex, category_pattern, note)
_OP_DETECT_PATTERNS = [
    (r"wp\.to_torch\(", "warp", "warp→torch conversion"),
    (r"torch\.linalg\.norm\(|\.norm\(", "torch_reduce", "norm"),
    (r"torch\.sum\(|\.sum\(", "torch_reduce", "sum"),
    (r"torch\.mean\(|\.mean\(", "torch_reduce", "mean"),
    (r"\.max\(", "torch_reduce", "max"),
    (r"\.min\(", "torch_reduce", "min"),
    (r"torch\.tanh\(", "torch_ewise", "tanh"),
    (r"torch\.exp\(", "torch_ewise", "exp"),
    (r"torch\.where\(", "torch_ewise", "where"),
    (r"torch\.clamp\(|\.clamp\(", "torch_ewise", "clamp"),
    (r"torch\.square\(|\.square\(", "torch_ewise", "square"),
    (r"torch\.abs\(|\.abs\(", "torch_ewise", "abs"),
    (r"\.float\(\)", "torch_ewise", "float cast"),
    (r"torch\.cross\(", "torch_cross", "cross product"),
    (r"torch\.cat\(", "torch_cat", "cat"),
    (r"torch\.index_select\(", "torch_index", "index_select"),
    (r"\[:.*,.*\]", "torch_index", "fancy indexing"),
]


def _find_term_cfg(cfg_obj, term_name: str, max_depth: int = 4):
    """Recursively search a config tree for a term with the given name.

    Observations nest terms multiple levels deep (variant → group → term),
    while rewards/terminations are flat. This walks up to ``max_depth`` levels.
    """
    # Direct attribute check
    candidate = getattr(cfg_obj, term_name, None)
    if candidate is not None and hasattr(candidate, "func"):
        return candidate

    if max_depth <= 0:
        return None

    # Recurse into sub-configs (groups, variants)
    for attr_name in dir(cfg_obj):
        if attr_name.startswith("_"):
            continue
        child = getattr(cfg_obj, attr_name, None)
        if child is None or isinstance(child, (str, int, float, bool, list, dict, type)):
            continue
        if callable(child) and not hasattr(child, "__dict__"):
            continue
        result = _find_term_cfg(child, term_name, max_depth - 1)
        if result is not None:
            return result
    return None


def resolve_term_source(task_id: str, term_name: str, manager: str = "reward") -> TermSource:
    """Resolve an NVTX term name to its Python source code.

    Imports the task config, finds the term's function reference, and extracts
    its source code using ``inspect``.

    Args:
        task_id: Gymnasium task ID (e.g. ``"Isaac-Dexsuite-Kuka-Allegro-Lift-v0"``).
        term_name: Term attribute name (e.g. ``"fingers_to_object"``).
        manager: Manager type (``"reward"``, ``"observation"``, ``"termination"``).

    Returns:
        Resolved source information.

    Raises:
        ValueError: If the term or task cannot be resolved.
    """
    import inspect

    import gymnasium as gym

    # Create a minimal spec to get the config class
    spec = gym.spec(task_id)
    if spec is None or spec.entry_point is None:
        raise ValueError(f"Task {task_id!r} not found in gymnasium registry")

    # Get the config from kwargs
    env_cfg_entry = spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_entry is None:
        raise ValueError(f"Task {task_id!r} has no env_cfg_entry_point")

    # Import the config
    if callable(env_cfg_entry):
        cfg = env_cfg_entry()
    else:
        module_path, class_name = env_cfg_entry.rsplit(":", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cfg = getattr(mod, class_name)()

    # Navigate to the manager config
    manager_cfg = getattr(cfg, manager + "s", None)
    if manager_cfg is None:
        raise ValueError(f"Config has no {manager}s attribute")

    # Find the term — search recursively through the config tree.
    # Observations are nested: observations → variant (state) → group (proprio) → term.
    # Rewards/terminations are flat: rewards → term.
    term_cfg = _find_term_cfg(manager_cfg, term_name)
    if term_cfg is None:
        raise ValueError(
            f"Term {term_name!r} not found in {manager}s config. "
            f"Available: {[a for a in dir(manager_cfg) if not a.startswith('_')]}"
        )

    func = term_cfg.func
    if isinstance(func, type):
        # Class-based term — get __call__
        func_for_source = func.__call__
        func_name = func.__name__
    else:
        func_for_source = func
        func_name = func.__name__

    source_file = inspect.getfile(func_for_source)
    source_lines, line_number = inspect.getsourcelines(func_for_source)
    source_code = "".join(source_lines)

    params = getattr(term_cfg, "params", {}) or {}

    return TermSource(
        term_name=term_name,
        func_name=func_name,
        source_file=source_file,
        line_number=line_number,
        source_code=source_code,
        params=params,
    )


def draft_source_template(source_code: str, start_line: int) -> list[DraftTemplateLine]:
    """Auto-detect torch/warp operations in source code lines.

    Scans each line for known patterns and predicts which CUDA dispatch
    categories it will produce. This is a **draft** — function calls are
    not expanded, and dispatch counts per line are approximate.

    Args:
        source_code: The function source code.
        start_line: Line number of the first line in the source file.

    Returns:
        List of draft template lines with predicted ops.
    """
    result = []
    in_docstring = False
    in_signature = False
    for i, line in enumerate(source_code.splitlines()):
        stripped = line.strip()

        # Skip docstrings
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # Skip function signature lines (def ... through closing paren)
        if stripped.startswith("def "):
            in_signature = not stripped.endswith(":")
            continue
        if in_signature:
            in_signature = not stripped.endswith(":") and not stripped.endswith("),")
            if stripped.endswith(":"):
                in_signature = False
            continue

        if not stripped:
            continue

        code_part = line.split("#")[0].strip()
        ops: list[tuple[str, str]] = []
        for pattern, category, note in _OP_DETECT_PATTERNS:
            for _ in re.finditer(pattern, code_part):
                ops.append((category, note))

        # Detect arithmetic on tensors (rough heuristic)
        # Only flag if line has other tensor ops
        if ops:
            for op_char, op_name in [("-", "sub"), ("+", "add"), ("*", "mul"), ("/", "div")]:
                count = code_part.count(f" {op_char} ")
                for _ in range(count):
                    ops.append(("torch_ewise", op_name))

        result.append(
            DraftTemplateLine(
                line_no=start_line + i,
                code=line.rstrip(),
                predicted_ops=ops,
            )
        )

    return result


def format_draft_template(draft: list[DraftTemplateLine], dispatches: list[DispatchEvent]) -> str:
    """Format a draft template alongside actual dispatches for agent review.

    Produces a side-by-side view that helps the agent build the final
    ``SourceMapEntry`` template.

    Args:
        draft: Auto-generated draft from :func:`draft_source_template`.
        dispatches: Actual dispatch events from the nsys trace.

    Returns:
        Formatted text for agent review.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("DRAFT SOURCE TEMPLATE — Review and refine")
    lines.append("=" * 80)
    lines.append("")

    # Part 1: Source lines with predicted ops
    lines.append("── Source lines with predicted dispatches ──")
    total_predicted = 0
    for entry in draft:
        code = entry.code.strip()
        if entry.predicted_ops:
            ops_str = ", ".join(f"{cat}({note})" for cat, note in entry.predicted_ops)
            lines.append(f"  L{entry.line_no}: {code}")
            lines.append(f"         → {len(entry.predicted_ops)} dispatch(es): {ops_str}")
            total_predicted += len(entry.predicted_ops)
        elif code and "(" in code and not code.startswith(("@", "if ", "elif ", "for ")):
            # Likely a function call — flag it for expansion
            # Skip default args, type annotations, and decorators
            if "=" in code and code.index("=") < code.index("("):
                # Assignment — check if RHS is a function call with tensor ops
                rhs = code[code.index("=") + 1 :].strip()
                if any(kw in rhs for kw in ("torch.", "wp.", "env.", "self.")):
                    lines.append(f"  L{entry.line_no}: {code}")
                    lines.append("         → ⚠ FUNCTION CALL — expand inline to predict dispatches")
            elif not any(kw in code for kw in ("SceneEntityCfg", "RigidObject", ":")):
                lines.append(f"  L{entry.line_no}: {code}")
                lines.append("         → ⚠ FUNCTION CALL — expand inline to predict dispatches")

    lines.append(f"\n  Total predicted: {total_predicted} | Actual: {len(dispatches)}")
    if total_predicted != len(dispatches):
        lines.append(f"  ⚠ MISMATCH: {abs(len(dispatches) - total_predicted)} dispatch(es) unaccounted for")
        lines.append("  Common causes: function call expansion, fused ops, lazy warp evaluation")

    # Part 2: Actual dispatch sequence
    lines.append("")
    lines.append("── Actual dispatch sequence ──")
    for i, d in enumerate(dispatches):
        lines.append(f"  #{i:2d}  {d.category:<16}  {d.op_name:<40}  gap={d.gap_before_ns / 1000:.0f}μs")

    # Part 3: Instructions for the agent
    lines.append("")
    lines.append("── Agent instructions ──")
    lines.append("1. Match predicted ops to actual dispatches sequentially")
    lines.append("2. For function calls: expand inline (read the called function,")
    lines.append("   predict its dispatches, insert them at the call site)")
    lines.append("3. For loops: look for repeating dispatch patterns in the actual")
    lines.append("   sequence and match them to loop iterations")
    lines.append("4. Build final template as list[SourceMapEntry] with one entry")
    lines.append("   per actual dispatch, choosing a group color per code region")
    lines.append("5. Run align_dispatches_to_source() and verify match_pct > 90%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_nvtx_schema(cursor) -> tuple[str, str, bool]:
    """Detect NVTX_EVENTS schema.

    Args:
        cursor: Open sqlite cursor with NVTX_EVENTS table available.

    Returns:
        Tuple of (start_col, end_col, has_inline_text). ``has_inline_text``
        is True if the ``text`` column exists (modern nsys), False if
        ``textId`` + ``StringIds`` join is needed (legacy nsys).

    Raises:
        RuntimeError: If schema cannot be determined.
    """
    col_names = {row[1] for row in cursor.execute("PRAGMA table_info(NVTX_EVENTS)").fetchall()}

    if "start" in col_names and "end" in col_names:
        ts_start, ts_end = "start", "end"
    elif "startTimestamp" in col_names and "endTimestamp" in col_names:
        ts_start, ts_end = "startTimestamp", "endTimestamp"
    else:
        raise RuntimeError("NVTX_EVENTS: cannot determine timestamp columns")

    has_inline_text = "text" in col_names
    return ts_start, ts_end, has_inline_text


def _get_tables(cursor) -> set[str]:
    """Return the set of table names in the database."""
    return {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _has_anatomy_tables(cursor, tables: set[str]) -> bool:
    """Check if the required tables for kernel correlation exist."""
    if not {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}.issubset(tables):
        return False
    return "CUPTI_ACTIVITY_KIND_RUNTIME" in tables or "CUPTI_ACTIVITY_KIND_DRIVER" in tables


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Compute percentile from a sorted list using nearest-rank method."""
    if not sorted_values:
        return 0
    idx = int(math.ceil(pct / 100.0 * len(sorted_values))) - 1
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def _pct_change(old: int | float, new: int | float) -> float:
    """Compute percentage change from old to new."""
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return (new - old) / old * 100.0


def classify_kernel(kernel_name: str) -> tuple[str, str]:
    """Classify a CUDA kernel name into a human-readable op and category.

    Args:
        kernel_name: Demangled CUDA kernel name from nsys.

    Returns:
        Tuple of (op_name, category). Categories: ``"warp"``, ``"torch_reduce"``,
        ``"torch_index"``, ``"torch_cat"``, ``"torch_cross"``, ``"torch_math"``,
        ``"torch_copy"``, ``"torch_fill"``, ``"torch_compare"``, ``"torch_binary"``,
        ``"torch_unary"``, ``"torch_ewise"``, ``"torch_bool"``, ``"other"``.
    """
    kl = kernel_name.lower()

    # Warp / IsaacLab custom kernels (identifiable by _cuda_kernel_forward suffix or fused_ prefix)
    if "_cuda_kernel_forward" in kl or kl.startswith("fused_"):
        # Extract meaningful name
        if "split_transform_to_pos" in kl:
            return "warp: extract_position", "warp"
        if "split_transform_to_quat" in kl:
            return "warp: extract_quaternion", "warp"
        if "copy_from_newton" in kl or "copy_from_" in kl:
            return "warp: copy_from_sim", "warp"
        if "update_contact_sensor" in kl:
            return "warp: contact_sensor_update", "warp"
        if "update_outdated_envs" in kl:
            return "warp: outdated_env_update", "warp"
        if kl.startswith("fused_"):
            short = kernel_name.split("(")[0]
            if len(short) > 40:
                short = short[:37] + "..."
            return f"warp: {short}", "warp"
        # Generic warp kernel
        short = kernel_name.split("_cuda_kernel")[0]
        if len(short) > 40:
            short = short[-37:]
        return f"warp: {short}", "warp"

    # PyTorch ops (identifiable by at::native:: namespace)
    if "reduce_kernel" in kl:
        return "torch: reduce (sum/mean/norm)", "torch_reduce"
    if "index_elementwise" in kl or "index_select" in kl:
        return "torch: index_select / gather", "torch_index"
    if "scatter" in kl:
        return "torch: scatter", "torch_index"
    if "catarraybatchedcopy" in kl:
        return "torch: cat (concatenate)", "torch_cat"
    if "cross_kernel" in kl:
        return "torch: cross (cross product)", "torch_cross"
    if "tanh_kernel" in kl:
        return "torch: tanh", "torch_math"
    if "unrolled_elementwise" in kl and "copy" in kl:
        return "torch: type_cast / copy", "torch_copy"
    if "fillfunctor" in kl or ("fill" in kl and "elementwise" in kl):
        return "torch: fill / zeros", "torch_fill"
    if "compare" in kl and "elementwise" in kl:
        return "torch: comparison (>, <, ==)", "torch_compare"
    if "binaryfunctor" in kl or "binaryf" in kl:
        return "torch: binary_op (+, -, *, /)", "torch_binary"
    if "aunaryfunctor" in kl or "aunary" in kl:
        return "torch: unary_op (neg, abs)", "torch_unary"
    if "bunaryfunctor" in kl or "bunary" in kl:
        return "torch: bool_op (and, or)", "torch_bool"
    if "cudafunctor" in kl or "cudaf" in kl:
        return "torch: elementwise (clamp, where)", "torch_ewise"
    if "gpu_kernel_impl" in kl or "elementwise_kernel" in kl:
        return "torch: elementwise (generic)", "torch_ewise"
    if "vectorized_elementwise" in kl:
        return "torch: elementwise (vectorized)", "torch_ewise"

    # Unknown
    short = kernel_name[:50]
    return f"other: {short}", "other"


# ---------------------------------------------------------------------------
# Layer 1: Single-profile data extraction
# ---------------------------------------------------------------------------


def query_nvtx_tree(
    db_path: str,
    step_index: int = 0,
    attribute_kernels: bool = True,
) -> NvtxNode | None:
    """Build the hierarchical NVTX range tree within one ``env.step``.

    Reconstructs the full nesting structure via timestamp containment.
    Optionally attributes GPU kernels to their innermost containing range
    using the CUDA correlation chain.

    Args:
        db_path: Path to the nsys sqlite database.
        step_index: Which ``env.step`` occurrence to dissect (0-based).
        attribute_kernels: If True, attribute GPU kernels to tree nodes
            using the CUDA correlation chain.

    Returns:
        Root :class:`NvtxNode` of the tree, or None if the step does not
        exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    if "NVTX_EVENTS" not in tables:
        conn.close()
        return None

    try:
        ts_start, ts_end, has_text = _get_nvtx_schema(cursor)
    except RuntimeError:
        conn.close()
        return None

    # Find the target env.step range
    if has_text:
        step = cursor.execute(
            f"SELECT {ts_start}, {ts_end} FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text = 'env.step' ORDER BY {ts_start} LIMIT 1 OFFSET ?",
            (step_index,),
        ).fetchone()
    else:
        step = cursor.execute(
            f"SELECT e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 AND s.value = 'env.step' ORDER BY e.{ts_start} LIMIT 1 OFFSET ?",
            (step_index,),
        ).fetchone()

    if step is None:
        conn.close()
        return None

    step_start, step_end = step

    # Query all NVTX ranges within this step (including env.step itself)
    if has_text:
        ranges = cursor.execute(
            f"SELECT text, {ts_start}, {ts_end} FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text IS NOT NULL "
            f"AND {ts_start} >= ? AND {ts_end} <= ? "
            f"ORDER BY {ts_start} ASC, ({ts_end} - {ts_start}) DESC",
            (step_start, step_end),
        ).fetchall()
    else:
        ranges = cursor.execute(
            f"SELECT s.value, e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 "
            f"AND e.{ts_start} >= ? AND e.{ts_end} <= ? "
            f"ORDER BY e.{ts_start} ASC, (e.{ts_end} - e.{ts_start}) DESC",
            (step_start, step_end),
        ).fetchall()

    if not ranges:
        conn.close()
        return None

    # Build tree via stack-based algorithm
    root_name, root_start, root_end = ranges[0]
    root = NvtxNode(
        name=root_name,
        start_ns=root_start,
        end_ns=root_end,
        wall_ns=root_end - root_start,
        depth=0,
    )
    stack: list[NvtxNode] = [root]

    for name, start, end in ranges[1:]:
        # Pop nodes that do not contain this range
        while stack and not (stack[-1].start_ns <= start and stack[-1].end_ns >= end):
            stack.pop()

        if not stack:
            break

        parent = stack[-1]
        node = NvtxNode(
            name=name,
            start_ns=start,
            end_ns=end,
            wall_ns=end - start,
            depth=parent.depth + 1,
        )
        parent.children.append(node)
        stack.append(node)

    # Attribute kernels to tree nodes via correlation chain
    if attribute_kernels and _has_anatomy_tables(cursor, tables):
        _attribute_kernels_to_tree(cursor, tables, root, step_start, step_end)

    conn.close()
    return root


def _attribute_kernels_to_tree(
    cursor,
    tables: set[str],
    root: NvtxNode,
    step_start: int,
    step_end: int,
) -> None:
    """Attribute GPU kernels to tree nodes using the CUDA correlation chain.

    For each kernel launched within the step, finds the innermost tree node
    that contains the CUDA API call's timestamp and increments that node's
    kernel count and GPU time.
    """
    # Build kernel list via correlation chain (same pattern as analyze.py)
    api_tables = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api_tables.append("CUPTI_ACTIVITY_KIND_RUNTIME")
    if "CUPTI_ACTIVITY_KIND_DRIVER" in tables:
        api_tables.append("CUPTI_ACTIVITY_KIND_DRIVER")

    if not api_tables:
        return

    api_selects = []
    params: list[int] = []
    for api_table in api_tables:
        api_selects.append(
            f"SELECT api.start AS rt_start, api.end AS rt_end, "
            f"(k.end - k.start) AS kernel_ns "
            f"FROM {api_table} api "
            f"JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId "
            f"WHERE api.start >= ? AND api.end <= ?"
        )
        params.extend([step_start, step_end])

    query = " UNION ALL ".join(api_selects) + " ORDER BY rt_start"
    kernels = cursor.execute(query, params).fetchall()

    # For each kernel, walk the tree to find innermost containing node
    for rt_start, rt_end, kernel_ns in kernels:
        node = _find_innermost_node(root, rt_start, rt_end)
        if node is not None:
            node.kernel_count += 1
            node.direct_kernel_gpu_ns += kernel_ns


def _find_innermost_node(root: NvtxNode, rt_start: int, rt_end: int) -> NvtxNode | None:
    """Find the deepest tree node whose interval contains [rt_start, rt_end]."""
    if not (root.start_ns <= rt_start and root.end_ns >= rt_end):
        return None

    # Try to descend into children
    for child in root.children:
        result = _find_innermost_node(child, rt_start, rt_end)
        if result is not None:
            return result

    return root


def flatten_tree(
    tree: NvtxNode,
    max_depth: int | None = None,
    min_wall_ns: int = 0,
    _parent_path: str = "",
    _parent_wall_ns: int = 0,
) -> list[FlatRow]:
    """Flatten an NVTX tree into rows with full paths.

    Args:
        tree: Root :class:`NvtxNode`.
        max_depth: Stop recursing beyond this depth. None = unlimited.
        min_wall_ns: Skip nodes with wall time below this threshold [ns].

    Returns:
        List of :class:`FlatRow` in DFS order.
    """
    path = f"{_parent_path}/{tree.name}" if _parent_path else tree.name
    parent_wall = _parent_wall_ns if _parent_wall_ns > 0 else tree.wall_ns
    pct = (tree.wall_ns / parent_wall * 100.0) if parent_wall > 0 else 100.0

    rows: list[FlatRow] = []

    if tree.wall_ns >= min_wall_ns:
        rows.append(
            FlatRow(
                path=path,
                name=tree.name,
                depth=tree.depth,
                wall_ns=tree.wall_ns,
                kernel_count=tree.kernel_count,
                direct_kernel_gpu_ns=tree.direct_kernel_gpu_ns,
                subtree_kernel_gpu_ns=tree.subtree_kernel_gpu_ns,
                pct_of_parent=pct,
            )
        )

    if max_depth is not None and tree.depth >= max_depth:
        return rows

    for child in tree.children:
        rows.extend(
            flatten_tree(
                child,
                max_depth=max_depth,
                min_wall_ns=min_wall_ns,
                _parent_path=path,
                _parent_wall_ns=tree.wall_ns,
            )
        )

    return rows


def query_step_timeseries(
    db_path: str,
    range_names: list[str] | None = None,
) -> list[StepSnapshot]:
    """Extract per-step timing across all captured ``env.step`` occurrences.

    Args:
        db_path: Path to the nsys sqlite database.
        range_names: If provided, only include these NVTX ranges in
            ``range_times``. If None, include all sub-ranges.

    Returns:
        List of :class:`StepSnapshot` in chronological order.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    if "NVTX_EVENTS" not in tables:
        conn.close()
        return []

    try:
        ts_start, ts_end, has_text = _get_nvtx_schema(cursor)
    except RuntimeError:
        conn.close()
        return []

    # Get all env.step ranges
    if has_text:
        steps = cursor.execute(
            f"SELECT {ts_start}, {ts_end} FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text = 'env.step' ORDER BY {ts_start}",
        ).fetchall()
    else:
        steps = cursor.execute(
            f"SELECT e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 AND s.value = 'env.step' ORDER BY e.{ts_start}",
        ).fetchall()

    if not steps:
        conn.close()
        return []

    snapshots: list[StepSnapshot] = []

    for idx, (s_start, s_end) in enumerate(steps):
        # Query sub-ranges within this step
        if has_text:
            sub_ranges = cursor.execute(
                f"SELECT text, SUM({ts_end} - {ts_start}) AS total_ns "
                f"FROM NVTX_EVENTS "
                f"WHERE eventType = 59 AND text IS NOT NULL AND text != 'env.step' "
                f"AND {ts_start} >= ? AND {ts_end} <= ? "
                f"GROUP BY text",
                (s_start, s_end),
            ).fetchall()
        else:
            sub_ranges = cursor.execute(
                f"SELECT s.value, SUM(e.{ts_end} - e.{ts_start}) AS total_ns "
                f"FROM NVTX_EVENTS e JOIN StringIds s ON e.textId = s.id "
                f"WHERE e.eventType = 59 AND s.value != 'env.step' "
                f"AND e.{ts_start} >= ? AND e.{ts_end} <= ? "
                f"GROUP BY s.value",
                (s_start, s_end),
            ).fetchall()

        range_times = {}
        for name, total_ns in sub_ranges:
            if range_names is None or name in range_names:
                range_times[name] = total_ns

        snapshots.append(
            StepSnapshot(
                step_index=idx,
                wall_ns=s_end - s_start,
                start_ns=s_start,
                end_ns=s_end,
                range_times=range_times,
            )
        )

    conn.close()
    return snapshots


def query_range_distribution(
    db_path: str,
    range_name: str,
) -> RangeDistribution | None:
    """Compute statistical distribution of a single NVTX range.

    Args:
        db_path: Path to the nsys sqlite database.
        range_name: NVTX range label to analyze (e.g. ``"sim.step"``).

    Returns:
        :class:`RangeDistribution` or None if the range was not found.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    if "NVTX_EVENTS" not in tables:
        conn.close()
        return None

    try:
        ts_start, ts_end, has_text = _get_nvtx_schema(cursor)
    except RuntimeError:
        conn.close()
        return None

    if has_text:
        rows = cursor.execute(
            f"SELECT ({ts_end} - {ts_start}) AS duration_ns FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text = ? AND {ts_end} IS NOT NULL AND {ts_end} > {ts_start} "
            f"ORDER BY {ts_start}",
            (range_name,),
        ).fetchall()
    else:
        rows = cursor.execute(
            f"SELECT (e.{ts_end} - e.{ts_start}) AS duration_ns FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 AND s.value = ? "
            f"AND e.{ts_end} IS NOT NULL AND e.{ts_end} > e.{ts_start} "
            f"ORDER BY e.{ts_start}",
            (range_name,),
        ).fetchall()

    conn.close()

    if not rows:
        return None

    durations = [row[0] for row in rows]
    sorted_d = sorted(durations)
    count = len(durations)
    mean = sum(durations) / count
    stddev = statistics.stdev(durations) if count > 1 else 0.0

    return RangeDistribution(
        name=range_name,
        count=count,
        mean_ns=mean,
        stddev_ns=stddev,
        min_ns=sorted_d[0],
        max_ns=sorted_d[-1],
        p50_ns=_percentile(sorted_d, 50),
        p90_ns=_percentile(sorted_d, 90),
        p99_ns=_percentile(sorted_d, 99),
        durations_ns=durations,
    )


def query_kernel_details(
    db_path: str,
    name_pattern: str | None = None,
    step_index: int | None = None,
) -> list[KernelDetail]:
    """Extract per-invocation GPU kernel data with hardware info.

    Args:
        db_path: Path to the nsys sqlite database.
        name_pattern: Optional regex to filter kernel names.
        step_index: If provided, restrict to kernels within this ``env.step``.

    Returns:
        List of :class:`KernelDetail` sorted by start time.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    # Find kernel table
    kernel_table = None
    for candidate in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]:
        if candidate in tables:
            kernel_table = candidate
            break

    if kernel_table is None or "StringIds" not in tables:
        conn.close()
        return []

    # Detect available columns
    col_info = cursor.execute(f"PRAGMA table_info({kernel_table})").fetchall()
    col_names = {row[1] for row in col_info}

    # Determine name column
    if "shortName" in col_names:
        name_col = "shortName"
    elif "demangledName" in col_names:
        name_col = "demangledName"
    else:
        conn.close()
        return []

    # Build select with available hardware columns
    hw_cols = []
    for col, default in [
        ("gridX", 0),
        ("gridY", 0),
        ("gridZ", 0),
        ("blockX", 0),
        ("blockY", 0),
        ("blockZ", 0),
        ("registersPerThread", 0),
        ("staticSharedMemory", 0),
        ("dynamicSharedMemory", 0),
        ("streamId", 0),
        ("deviceId", 0),
    ]:
        if col in col_names:
            hw_cols.append(f"k.{col}")
        else:
            hw_cols.append(str(default))

    hw_select = ", ".join(hw_cols)

    # Determine time window
    time_filter = ""
    params: list[int] = []

    if step_index is not None and "NVTX_EVENTS" in tables:
        try:
            ts_start, ts_end, has_text = _get_nvtx_schema(cursor)
        except RuntimeError:
            conn.close()
            return []

        if has_text:
            step = cursor.execute(
                f"SELECT {ts_start}, {ts_end} FROM NVTX_EVENTS "
                f"WHERE eventType = 59 AND text = 'env.step' ORDER BY {ts_start} LIMIT 1 OFFSET ?",
                (step_index,),
            ).fetchone()
        else:
            step = cursor.execute(
                f"SELECT e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
                f"JOIN StringIds s ON e.textId = s.id "
                f"WHERE e.eventType = 59 AND s.value = 'env.step' ORDER BY e.{ts_start} LIMIT 1 OFFSET ?",
                (step_index,),
            ).fetchone()

        if step is not None:
            time_filter = "AND k.start >= ? AND k.end <= ? "
            params.extend([step[0], step[1]])

    query = (
        f"SELECT s.value, k.start, k.end, (k.end - k.start), {hw_select} "
        f"FROM {kernel_table} k "
        f"JOIN StringIds s ON k.{name_col} = s.id "
        f"WHERE 1=1 {time_filter}"
        f"ORDER BY k.start"
    )

    results: list[KernelDetail] = []
    compiled = re.compile(name_pattern, re.IGNORECASE) if name_pattern else None

    for row in cursor.execute(query, params):
        name = row[0]
        if compiled and not compiled.search(name):
            continue
        results.append(
            KernelDetail(
                name=name,
                start_ns=row[1],
                end_ns=row[2],
                duration_ns=row[3],
                grid=(row[4], row[5], row[6]),
                block=(row[7], row[8], row[9]),
                registers_per_thread=row[10],
                static_shared_memory_bytes=row[11],
                dynamic_shared_memory_bytes=row[12],
                stream_id=row[13],
                device_id=row[14],
            )
        )

    conn.close()
    return results


def query_nvtx_kernel_map(
    db_path: str,
    step_index: int = 0,
    full_path: bool = True,
) -> dict[str, list[KernelEntry]]:
    """Map NVTX range paths to their attributed GPU kernels.

    Args:
        db_path: Path to the nsys sqlite database.
        step_index: Which ``env.step`` to dissect (0-based).
        full_path: If True, use slash-separated paths (e.g.
            ``"env.step/sim.step"``). If False, use innermost name only.

    Returns:
        Dict mapping NVTX path (or name) to list of :class:`KernelEntry`.
    """
    tree = query_nvtx_tree(db_path, step_index=step_index, attribute_kernels=False)
    if tree is None:
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    if not _has_anatomy_tables(cursor, tables):
        conn.close()
        return {}

    # Get kernels via correlation chain
    api_tables = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api_tables.append("CUPTI_ACTIVITY_KIND_RUNTIME")
    if "CUPTI_ACTIVITY_KIND_DRIVER" in tables:
        api_tables.append("CUPTI_ACTIVITY_KIND_DRIVER")

    api_selects = []
    params: list[int] = []
    for api_table in api_tables:
        api_selects.append(
            f"SELECT api.start AS rt_start, api.end AS rt_end, "
            f"s.value AS kernel_name, (k.end - k.start) AS kernel_ns "
            f"FROM {api_table} api "
            f"JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId "
            f"JOIN StringIds s ON k.shortName = s.id "
            f"WHERE api.start >= ? AND api.end <= ?"
        )
        params.extend([tree.start_ns, tree.end_ns])

    query = " UNION ALL ".join(api_selects) + " ORDER BY rt_start"
    kernel_rows = cursor.execute(query, params).fetchall()
    conn.close()

    # Build path lookup from tree
    path_map: dict[int, str] = {}  # node id -> path
    _build_path_map(tree, "", path_map, full_path)

    # Map kernels to paths
    result: dict[str, list[KernelEntry]] = {}
    for rt_start, rt_end, kernel_name, kernel_ns in kernel_rows:
        node = _find_innermost_node(tree, rt_start, rt_end)
        if node is not None:
            key = path_map.get(id(node), node.name)
        else:
            key = "(unattributed)"

        if key not in result:
            result[key] = []
        result[key].append(KernelEntry(name=kernel_name, duration_ns=kernel_ns))

    return result


def _build_path_map(
    node: NvtxNode,
    parent_path: str,
    path_map: dict[int, str],
    full_path: bool,
) -> None:
    """Recursively build a mapping from node id to path string."""
    if full_path:
        path = f"{parent_path}/{node.name}" if parent_path else node.name
    else:
        path = node.name
    path_map[id(node)] = path
    for child in node.children:
        _build_path_map(child, path if full_path else "", path_map, full_path)


def query_term_decomposition(
    db_path: str,
    step_index: int = 0,
    range_name: str | None = None,
    include_dispatches: bool = False,
) -> TermDecomposition | None:
    """Decompose CPU time within one NVTX range using CUDA API timestamps.

    Breaks down the range's wall time into: Python before first dispatch,
    Python between dispatches, CUDA API call overhead, and Python after
    last dispatch.

    Args:
        db_path: Path to the nsys sqlite database.
        step_index: Which ``env.step`` occurrence (0-based).
        range_name: NVTX range label to decompose. If None, decomposes
            the ``env.step`` itself.
        include_dispatches: If True, populate the ``dispatches`` list
            with per-dispatch details including classified op names.

    Returns:
        :class:`TermDecomposition`, or None if the range is not found.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    if "NVTX_EVENTS" not in tables:
        conn.close()
        return None

    try:
        ts_start, ts_end, has_text = _get_nvtx_schema(cursor)
    except RuntimeError:
        conn.close()
        return None

    # Find the env.step boundaries
    if has_text:
        step = cursor.execute(
            f"SELECT {ts_start}, {ts_end} FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text = 'env.step' ORDER BY {ts_start} LIMIT 1 OFFSET ?",
            (step_index,),
        ).fetchone()
    else:
        step = cursor.execute(
            f"SELECT e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 AND s.value = 'env.step' ORDER BY e.{ts_start} LIMIT 1 OFFSET ?",
            (step_index,),
        ).fetchone()

    if step is None:
        conn.close()
        return None

    step_start, step_end = step

    # Find the target range and its depth
    if range_name is None or range_name == "env.step":
        rng_start, rng_end, rng_depth = step_start, step_end, 0
    else:
        # Get all ranges to compute depth
        if has_text:
            all_ranges = cursor.execute(
                f"SELECT text, {ts_start}, {ts_end} FROM NVTX_EVENTS "
                f"WHERE eventType = 59 AND text IS NOT NULL "
                f"AND {ts_start} >= ? AND {ts_end} <= ? "
                f"ORDER BY {ts_start} ASC, ({ts_end} - {ts_start}) DESC",
                (step_start, step_end),
            ).fetchall()
        else:
            all_ranges = cursor.execute(
                f"SELECT s.value, e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
                f"JOIN StringIds s ON e.textId = s.id "
                f"WHERE e.eventType = 59 "
                f"AND e.{ts_start} >= ? AND e.{ts_end} <= ? "
                f"ORDER BY e.{ts_start} ASC, (e.{ts_end} - e.{ts_start}) DESC",
                (step_start, step_end),
            ).fetchall()

        rng_start = rng_end = rng_depth = None
        stack: list[tuple[str, int, int]] = []
        for name, s, e in all_ranges:
            while stack and not (stack[-1][1] <= s and stack[-1][2] >= e):
                stack.pop()
            depth = len(stack)
            if name == range_name and rng_start is None:
                rng_start, rng_end, rng_depth = s, e, depth
            stack.append((name, s, e))

        if rng_start is None:
            conn.close()
            return None

    result = _decompose_range(
        cursor,
        tables,
        range_name or "env.step",
        rng_start,
        rng_end,
        rng_depth,
        include_dispatches,
    )
    conn.close()
    return result


def query_all_term_decompositions(
    db_path: str,
    step_index: int = 0,
    min_depth: int = 1,
    max_depth: int | None = None,
    include_dispatches: bool = False,
) -> list[TermDecomposition]:
    """Decompose CPU time for all NVTX ranges within a step.

    Args:
        db_path: Path to the nsys sqlite database.
        step_index: Which ``env.step`` occurrence (0-based).
        min_depth: Minimum nesting depth to include (1 = top-level children
            of ``env.step``).
        max_depth: Maximum nesting depth to include. None = unlimited.
        include_dispatches: If True, populate per-dispatch details.

    Returns:
        List of :class:`TermDecomposition`, sorted by wall time descending.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = _get_tables(cursor)

    if "NVTX_EVENTS" not in tables:
        conn.close()
        return []

    try:
        ts_start, ts_end, has_text = _get_nvtx_schema(cursor)
    except RuntimeError:
        conn.close()
        return []

    # Find step boundaries
    if has_text:
        step = cursor.execute(
            f"SELECT {ts_start}, {ts_end} FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text = 'env.step' ORDER BY {ts_start} LIMIT 1 OFFSET ?",
            (step_index,),
        ).fetchone()
    else:
        step = cursor.execute(
            f"SELECT e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 AND s.value = 'env.step' ORDER BY e.{ts_start} LIMIT 1 OFFSET ?",
            (step_index,),
        ).fetchone()

    if step is None:
        conn.close()
        return []

    step_start, step_end = step

    # Get all NVTX ranges and compute depths
    if has_text:
        all_ranges = cursor.execute(
            f"SELECT text, {ts_start}, {ts_end} FROM NVTX_EVENTS "
            f"WHERE eventType = 59 AND text IS NOT NULL "
            f"AND {ts_start} >= ? AND {ts_end} <= ? "
            f"ORDER BY {ts_start} ASC, ({ts_end} - {ts_start}) DESC",
            (step_start, step_end),
        ).fetchall()
    else:
        all_ranges = cursor.execute(
            f"SELECT s.value, e.{ts_start}, e.{ts_end} FROM NVTX_EVENTS e "
            f"JOIN StringIds s ON e.textId = s.id "
            f"WHERE e.eventType = 59 "
            f"AND e.{ts_start} >= ? AND e.{ts_end} <= ? "
            f"ORDER BY e.{ts_start} ASC, (e.{ts_end} - e.{ts_start}) DESC",
            (step_start, step_end),
        ).fetchall()

    # Build depth via stack, collect target ranges
    targets: list[tuple[str, int, int, int]] = []
    stack: list[tuple[str, int, int]] = []
    for name, s, e in all_ranges:
        while stack and not (stack[-1][1] <= s and stack[-1][2] >= e):
            stack.pop()
        depth = len(stack)
        if depth >= min_depth and (max_depth is None or depth <= max_depth):
            targets.append((name, s, e, depth))
        stack.append((name, s, e))

    # Decompose each range
    results: list[TermDecomposition] = []
    for name, s, e, depth in targets:
        decomp = _decompose_range(cursor, tables, name, s, e, depth, include_dispatches)
        results.append(decomp)

    conn.close()
    results.sort(key=lambda d: d.wall_ns, reverse=True)
    return results


def _decompose_range(
    cursor,
    tables: set[str],
    name: str,
    rng_start: int,
    rng_end: int,
    depth: int,
    include_dispatches: bool,
) -> TermDecomposition:
    """Internal: decompose one NVTX range into CPU time categories."""
    wall_ns = rng_end - rng_start

    # Check if we have runtime API tables
    has_runtime = "CUPTI_ACTIVITY_KIND_RUNTIME" in tables
    has_driver = "CUPTI_ACTIVITY_KIND_DRIVER" in tables
    has_kernel = "CUPTI_ACTIVITY_KIND_KERNEL" in tables
    has_strings = "StringIds" in tables

    if not (has_runtime or has_driver) or not has_kernel:
        # No API call data — entire range is opaque
        return TermDecomposition(
            name=name,
            depth=depth,
            wall_ns=wall_ns,
            n_dispatches=0,
            pre_python_ns=wall_ns,
            between_python_ns=0,
            api_overhead_ns=0,
            post_python_ns=0,
            gpu_kernel_ns=0,
            avg_gap_ns=0,
            max_gap_ns=0,
        )

    # Query CUDA API calls + kernel info within this range
    api_tables: list[str] = []
    if has_runtime:
        api_tables.append("CUPTI_ACTIVITY_KIND_RUNTIME")
    if has_driver:
        api_tables.append("CUPTI_ACTIVITY_KIND_DRIVER")

    # Detect kernel name column (shortName vs demangledName)
    if has_kernel:
        col_info = cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)").fetchall()
        col_names = {row[1] for row in col_info}
        if "shortName" in col_names:
            kernel_name_col = "shortName"
        elif "demangledName" in col_names:
            kernel_name_col = "demangledName"
        else:
            kernel_name_col = None
    else:
        kernel_name_col = None

    selects: list[str] = []
    params: list[int] = []
    for api_table in api_tables:
        if include_dispatches and has_strings and kernel_name_col:
            selects.append(
                f"SELECT api.start, api.end, (k.end - k.start), s.value "
                f"FROM {api_table} api "
                f"JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId "
                f"JOIN StringIds s ON k.{kernel_name_col} = s.id "
                f"WHERE api.start >= ? AND api.end <= ?"
            )
        else:
            selects.append(
                f"SELECT api.start, api.end, (k.end - k.start), '' "
                f"FROM {api_table} api "
                f"JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId "
                f"WHERE api.start >= ? AND api.end <= ?"
            )
        params.extend([rng_start, rng_end])

    query = " UNION ALL ".join(selects) + " ORDER BY 1"  # ORDER BY api.start
    rows = cursor.execute(query, params).fetchall()

    if not rows:
        return TermDecomposition(
            name=name,
            depth=depth,
            wall_ns=wall_ns,
            n_dispatches=0,
            pre_python_ns=wall_ns,
            between_python_ns=0,
            api_overhead_ns=0,
            post_python_ns=0,
            gpu_kernel_ns=0,
            avg_gap_ns=0,
            max_gap_ns=0,
        )

    # Decompose timing
    n = len(rows)
    pre_ns = rows[0][0] - rng_start
    post_ns = rng_end - rows[-1][1]
    total_api_ns = sum(row[1] - row[0] for row in rows)
    total_gpu_ns = sum(row[2] for row in rows)

    gaps: list[int] = []
    for i in range(1, n):
        gap = rows[i][0] - rows[i - 1][1]
        gaps.append(gap)
    total_between_ns = sum(gaps)
    avg_gap = total_between_ns / len(gaps) if gaps else 0.0
    max_gap = max(gaps) if gaps else 0

    # Build dispatch events if requested
    dispatch_list: list[DispatchEvent] = []
    if include_dispatches:
        prev_end = rng_start
        for api_s, api_e, gpu_ns, kname in rows:
            op_name, category = classify_kernel(kname) if kname else ("unknown", "other")
            dispatch_list.append(
                DispatchEvent(
                    op_name=op_name,
                    category=category,
                    kernel_name=kname,
                    api_start_ns=api_s - rng_start,
                    api_end_ns=api_e - rng_start,
                    gpu_duration_ns=gpu_ns,
                    gap_before_ns=api_s - prev_end,
                )
            )
            prev_end = api_e

    return TermDecomposition(
        name=name,
        depth=depth,
        wall_ns=wall_ns,
        n_dispatches=n,
        pre_python_ns=pre_ns,
        between_python_ns=total_between_ns,
        api_overhead_ns=total_api_ns,
        post_python_ns=post_ns,
        gpu_kernel_ns=total_gpu_ns,
        avg_gap_ns=avg_gap,
        max_gap_ns=max_gap,
        dispatches=dispatch_list,
    )


# ---------------------------------------------------------------------------
# Layer 2: Cross-profile analysis
# ---------------------------------------------------------------------------


def compare_trees(
    tree_a: NvtxNode,
    tree_b: NvtxNode,
    label_a: str = "A",
    label_b: str = "B",
) -> TreeDiffResult:
    """Compare two NVTX trees by matching nodes on their full paths.

    Args:
        tree_a: NVTX tree from profile A.
        tree_b: NVTX tree from profile B.
        label_a: Label for profile A.
        label_b: Label for profile B.

    Returns:
        :class:`TreeDiffResult` with per-path comparisons.
    """
    flat_a = {r.path: r for r in flatten_tree(tree_a)}
    flat_b = {r.path: r for r in flatten_tree(tree_b)}

    all_paths = set(flat_a.keys()) | set(flat_b.keys())
    entries: list[TreeDiffEntry] = []

    for path in all_paths:
        a = flat_a.get(path)
        b = flat_b.get(path)

        if a and b:
            entries.append(
                TreeDiffEntry(
                    path=path,
                    status="matched",
                    a_wall_ns=a.wall_ns,
                    b_wall_ns=b.wall_ns,
                    a_gpu_ns=a.subtree_kernel_gpu_ns,
                    b_gpu_ns=b.subtree_kernel_gpu_ns,
                    a_kernel_count=a.kernel_count,
                    b_kernel_count=b.kernel_count,
                    wall_delta_pct=_pct_change(a.wall_ns, b.wall_ns),
                    gpu_delta_pct=_pct_change(a.subtree_kernel_gpu_ns, b.subtree_kernel_gpu_ns),
                )
            )
        elif a:
            entries.append(
                TreeDiffEntry(
                    path=path,
                    status="only_a",
                    a_wall_ns=a.wall_ns,
                    a_gpu_ns=a.subtree_kernel_gpu_ns,
                    a_kernel_count=a.kernel_count,
                )
            )
        else:
            assert b is not None
            entries.append(
                TreeDiffEntry(
                    path=path,
                    status="only_b",
                    b_wall_ns=b.wall_ns,
                    b_gpu_ns=b.subtree_kernel_gpu_ns,
                    b_kernel_count=b.kernel_count,
                )
            )

    entries.sort(key=lambda e: abs(e.a_wall_ns - e.b_wall_ns), reverse=True)
    return TreeDiffResult(label_a=label_a, label_b=label_b, entries=entries)


def analyze_scaling(
    profiles: dict[int, str],
    range_names: list[str] | None = None,
) -> ScalingResult:
    """Analyze how NVTX range times scale with ``num_envs``.

    Fits ``log(time) = exponent * log(num_envs) + intercept`` for each
    range to determine the scaling behavior.

    Args:
        profiles: Mapping of ``num_envs`` to sqlite database path.
        range_names: If provided, only analyze these ranges. If None,
            analyze all ranges found in any profile.

    Returns:
        :class:`ScalingResult` with per-range scaling data.
    """
    # Collect per-range timing at each num_envs
    range_data: dict[str, dict[int, float]] = {}  # {name: {num_envs: avg_ns}}

    for num_envs, db_path in sorted(profiles.items()):
        nvtx_ranges = query_nvtx_ranges(db_path)
        for r in nvtx_ranges:
            name = r["name"]
            if range_names is not None and name not in range_names:
                continue
            if name not in range_data:
                range_data[name] = {}
            range_data[name][num_envs] = r["avg_ns"]

    entries: list[ScalingEntry] = []

    for name, data_points in range_data.items():
        if len(data_points) < 2:
            continue

        envs = sorted(data_points.keys())
        times = [data_points[e] for e in envs]

        # Skip ranges with zero or negative times
        if any(t <= 0 for t in times):
            continue

        exponent, r_squared = _fit_log_log(envs, times)

        if exponent < 0.1:
            classification = "constant"
        elif exponent < 1.2:
            if exponent < 0.8:
                classification = "sub-linear"
            else:
                classification = "linear"
        else:
            classification = "super-linear"

        entries.append(
            ScalingEntry(
                range_name=name,
                num_envs=envs,
                times_ns=times,
                exponent=exponent,
                r_squared=r_squared,
                classification=classification,
            )
        )

    entries.sort(key=lambda e: e.exponent, reverse=True)
    return ScalingResult(entries=entries)


def _fit_log_log(x_values: list[int], y_values: list[float]) -> tuple[float, float]:
    """Fit a log-log linear regression and return (slope, r_squared).

    Uses manual least-squares to avoid a numpy dependency.
    """
    n = len(x_values)
    log_x = [math.log(v) for v in x_values]
    log_y = [math.log(v) for v in y_values]

    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xy = sum(xi * yi for xi, yi in zip(log_x, log_y))
    sum_x2 = sum(xi * xi for xi in log_x)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R²
    y_pred = [slope * xi + intercept for xi in log_x]
    mean_y = sum_y / n
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(log_y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in log_y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return slope, r_squared


# ---------------------------------------------------------------------------
# Layer 3: Derived metrics
# ---------------------------------------------------------------------------


def find_hotpath(
    tree: NvtxNode,
    metric: str = "wall_ns",
) -> list[NvtxNode]:
    """Find the critical path by following the heaviest child at each level.

    Args:
        tree: Root :class:`NvtxNode`.
        metric: Which metric to maximize. One of ``"wall_ns"`` or
            ``"subtree_kernel_gpu_ns"``.

    Returns:
        List of nodes from root to the heaviest leaf.
    """
    path = [tree]
    current = tree
    while current.children:
        if metric == "subtree_kernel_gpu_ns":
            heaviest = max(current.children, key=lambda c: c.subtree_kernel_gpu_ns)
        else:
            heaviest = max(current.children, key=lambda c: c.wall_ns)
        path.append(heaviest)
        current = heaviest
    return path


def gpu_utilization(tree: NvtxNode) -> dict[str, float]:
    """Compute GPU utilization ratio for each node in the tree.

    The ratio is ``subtree_kernel_gpu_ns / wall_ns``. A low ratio means
    the code section is CPU-bound or has pipeline stalls.

    Args:
        tree: Root :class:`NvtxNode`.

    Returns:
        Dict mapping node name to GPU utilization ratio (0.0 to 1.0+).
        Values > 1.0 indicate overlapping GPU work across streams.
    """
    result: dict[str, float] = {}
    _gpu_util_walk(tree, result)
    return result


def _gpu_util_walk(node: NvtxNode, result: dict[str, float]) -> None:
    """Recursively compute GPU utilization for each node."""
    if node.wall_ns > 0:
        result[node.name] = node.subtree_kernel_gpu_ns / node.wall_ns
    for child in node.children:
        _gpu_util_walk(child, result)


def scaling_bottlenecks(
    scaling: ScalingResult,
    exponent_threshold: float = 1.2,
) -> list[ScalingEntry]:
    """Filter scaling results to ranges with worse-than-linear scaling.

    Args:
        scaling: Output from :func:`analyze_scaling`.
        exponent_threshold: Minimum exponent to flag as a bottleneck.

    Returns:
        List of :class:`ScalingEntry` exceeding the threshold.
    """
    return [e for e in scaling.entries if e.exponent >= exponent_threshold]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_tree(tree: NvtxNode, max_depth: int | None = None) -> str:
    """Format an NVTX tree as an indented text tree with timing info.

    Args:
        tree: Root :class:`NvtxNode`.
        max_depth: Maximum depth to display.

    Returns:
        Formatted string.
    """
    lines: list[str] = []
    _format_tree_walk(tree, lines, max_depth)
    return "\n".join(lines)


def _format_tree_walk(node: NvtxNode, lines: list[str], max_depth: int | None) -> None:
    """Recursively format tree nodes."""
    indent = "  " * node.depth
    wall_ms = node.wall_ns / 1e6
    gpu_ms = node.subtree_kernel_gpu_ns / 1e6
    k_count = node.subtree_kernel_count

    if k_count > 0:
        lines.append(f"{indent}{node.name}  wall={wall_ms:.3f}ms  gpu={gpu_ms:.3f}ms  kernels={k_count}")
    else:
        lines.append(f"{indent}{node.name}  wall={wall_ms:.3f}ms  (cpu-only)")

    if max_depth is not None and node.depth >= max_depth:
        if node.children:
            lines.append(f"{indent}  ... ({len(node.children)} children)")
        return

    for child in node.children:
        _format_tree_walk(child, lines, max_depth)


def format_comparison(diff: TreeDiffResult) -> str:
    """Format a tree diff as a table with deltas.

    Args:
        diff: Output from :func:`compare_trees`.

    Returns:
        Formatted string.
    """
    lines: list[str] = []
    lines.append(f"Comparison: {diff.label_a} vs {diff.label_b}")
    lines.append("")
    lines.append(f"  {'Path':<50} {'Status':<10} {diff.label_a + ' ms':>10} {diff.label_b + ' ms':>10} {'Delta %':>10}")
    lines.append(f"  {'─' * 50} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    for e in diff.entries:
        a_ms = e.a_wall_ns / 1e6
        b_ms = e.b_wall_ns / 1e6
        path = e.path if len(e.path) <= 50 else "..." + e.path[-47:]

        if e.status == "matched":
            delta_str = f"{e.wall_delta_pct:>+9.1f}%"
        else:
            delta_str = f"{'—':>10}"

        lines.append(f"  {path:<50} {e.status:<10} {a_ms:>10.3f} {b_ms:>10.3f} {delta_str}")

    return "\n".join(lines)


def format_scaling(scaling: ScalingResult) -> str:
    """Format scaling analysis as a table with exponents.

    Args:
        scaling: Output from :func:`analyze_scaling`.

    Returns:
        Formatted string.
    """
    lines: list[str] = []
    lines.append(
        f"  {'NVTX Range':<45} {'Exponent':>10} {'R²':>8} {'Classification':>16} {'Min ms':>10} {'Max ms':>10}"
    )
    lines.append(f"  {'─' * 45} {'─' * 10} {'─' * 8} {'─' * 16} {'─' * 10} {'─' * 10}")

    for e in scaling.entries:
        min_ms = min(e.times_ns) / 1e6
        max_ms = max(e.times_ns) / 1e6
        name = e.range_name if len(e.range_name) <= 45 else e.range_name[:42] + "..."
        lines.append(
            f"  {name:<45} {e.exponent:>10.3f} {e.r_squared:>8.3f} "
            f"{e.classification:>16} {min_ms:>10.3f} {max_ms:>10.3f}"
        )

    return "\n".join(lines)


def format_distribution(dist: RangeDistribution) -> str:
    """Format a range distribution as a compact stats block.

    Args:
        dist: Output from :func:`query_range_distribution`.

    Returns:
        Formatted string.
    """
    mean_ms = dist.mean_ns / 1e6
    stddev_ms = dist.stddev_ns / 1e6
    cv = dist.stddev_ns / dist.mean_ns if dist.mean_ns > 0 else 0.0

    lines = [
        f"Distribution: {dist.name} ({dist.count} occurrences)",
        f"  mean   = {mean_ms:.3f} ms",
        f"  stddev = {stddev_ms:.3f} ms  (CV = {cv:.1%})",
        f"  min    = {dist.min_ns / 1e6:.3f} ms",
        f"  p50    = {dist.p50_ns / 1e6:.3f} ms",
        f"  p90    = {dist.p90_ns / 1e6:.3f} ms",
        f"  p99    = {dist.p99_ns / 1e6:.3f} ms",
        f"  max    = {dist.max_ns / 1e6:.3f} ms",
    ]
    return "\n".join(lines)


def format_timeseries(snapshots: list[StepSnapshot]) -> str:
    """Format step timeseries as a compact table.

    Args:
        snapshots: Output from :func:`query_step_timeseries`.

    Returns:
        Formatted string.
    """
    if not snapshots:
        return "(no steps found)"

    lines: list[str] = []

    # Collect all range names across all steps
    all_ranges = sorted({name for s in snapshots for name in s.range_times})
    range_cols = all_ranges[:5]  # Limit to 5 columns for readability

    header = f"  {'Step':>6} {'Wall ms':>10}"
    for name in range_cols:
        col_name = name if len(name) <= 15 else name[:12] + "..."
        header += f" {col_name:>15}"
    lines.append(header)

    lines.append(f"  {'─' * 6} {'─' * 10}" + "".join(f" {'─' * 15}" for _ in range_cols))

    for s in snapshots:
        row = f"  {s.step_index:>6} {s.wall_ns / 1e6:>10.3f}"
        for name in range_cols:
            t = s.range_times.get(name, 0)
            row += f" {t / 1e6:>15.3f}"
        lines.append(row)

    return "\n".join(lines)


def format_term_decomposition(decompositions: list[TermDecomposition]) -> str:
    """Format term decompositions as a table showing CPU time categories.

    Args:
        decompositions: Output from :func:`query_all_term_decompositions`.

    Returns:
        Formatted string.
    """
    if not decompositions:
        return "(no terms found)"

    lines: list[str] = []
    lines.append(
        f"  {'NVTX Range':<45} {'Wall ms':>8} {'#Disp':>6} "
        f"{'Pre%':>5} {'Between%':>9} {'API%':>5} {'Post%':>6} "
        f"{'AvgGap':>8} {'MaxGap':>8}"
    )
    lines.append(f"  {'─' * 45} {'─' * 8} {'─' * 6} {'─' * 5} {'─' * 9} {'─' * 5} {'─' * 6} {'─' * 8} {'─' * 8}")

    for d in decompositions:
        name = d.name if len(d.name) <= 45 else d.name[:42] + "..."
        indent = "  " * max(0, d.depth - 1)
        display_name = f"{indent}{name}"
        if len(display_name) > 45:
            display_name = display_name[:42] + "..."

        wall_ms = d.wall_ns / 1e6
        pre_pct = d.pre_python_ns / d.wall_ns * 100 if d.wall_ns > 0 else 0
        bet_pct = d.between_pct
        api_pct = d.api_overhead_ns / d.wall_ns * 100 if d.wall_ns > 0 else 0
        post_pct = d.post_python_ns / d.wall_ns * 100 if d.wall_ns > 0 else 0
        avg_gap_us = d.avg_gap_ns / 1000
        max_gap_us = d.max_gap_ns / 1000

        lines.append(
            f"  {display_name:<45} {wall_ms:>8.3f} {d.n_dispatches:>6} "
            f"{pre_pct:>4.0f}% {bet_pct:>8.0f}% {api_pct:>4.0f}% {post_pct:>5.0f}% "
            f"{avg_gap_us:>7.1f}\u03bc {max_gap_us:>7.0f}\u03bc"
        )

    return "\n".join(lines)
