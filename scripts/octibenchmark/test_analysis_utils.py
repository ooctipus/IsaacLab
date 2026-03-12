# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the octibenchmark analysis_utils module.

These tests create synthetic nsys sqlite databases and verify that the
analysis utilities extract correct results without requiring an actual
nsys run or GPU.

Run with::

    python -m pytest scripts/octibenchmark/test_analysis_utils.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

import pytest

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from octibenchmark.analysis_utils import (
    DispatchEvent,
    NvtxNode,
    SourceMapEntry,
    align_dispatches_to_source,
    analyze_scaling,
    classify_kernel,
    compare_trees,
    find_hotpath,
    flatten_tree,
    format_comparison,
    format_distribution,
    format_scaling,
    format_term_decomposition,
    format_timeseries,
    format_tree,
    gpu_utilization,
    query_all_term_decompositions,
    query_kernel_details,
    query_nvtx_kernel_map,
    query_nvtx_tree,
    query_range_distribution,
    query_step_timeseries,
    query_term_decomposition,
    scaling_bottlenecks,
)


def _create_base_tables(cursor):
    """Create the base nsys tables shared by most fixtures."""
    cursor.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cursor.execute("""
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER, text TEXT,
            rangeId INTEGER, category INTEGER, color INTEGER,
            globalTid INTEGER, endGlobalTid INTEGER,
            textId INTEGER, domainId INTEGER,
            uint64Value INTEGER, int64Value INTEGER, doubleValue REAL,
            uint32Value INTEGER, int32Value INTEGER, floatValue REAL,
            jsonTextId INTEGER, jsonText TEXT, binaryData TEXT
        )
    """)


def _create_kernel_tables(cursor):
    """Create kernel and API tables for correlation chain tests."""
    cursor.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, shortName INTEGER,
            correlationId INTEGER,
            deviceId INTEGER, contextId INTEGER, streamId INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            registersPerThread INTEGER, staticSharedMemory INTEGER,
            dynamicSharedMemory INTEGER, localMemoryPerThread INTEGER,
            localMemoryTotal INTEGER
        )
    """)


@pytest.fixture
def rich_db():
    """Create a synthetic DB with 4-level NVTX nesting and kernel attribution.

    Layout (step 0, env.step at 0-20ms):

        env.step [0 - 20ms]
        ├─ sim.step [1ms - 8ms]
        │  └─ physics.solve [2ms - 6ms]  → 2 kernels (corr 1, 2)
        ├─ reward.compute [9ms - 14ms]
        │  └─ reward.term:reach [10ms - 12ms]  → 1 kernel (corr 3)
        ├─ observation.compute [15ms - 18ms]  → 1 kernel (corr 4)
        └─ (unattributed kernel between ranges, corr 5)

    Step 1 (env.step at 30-50ms, slightly different timing):

        env.step [30ms - 50ms]
        ├─ sim.step [31ms - 38ms]
        │  └─ physics.solve [32ms - 36ms]  → 2 kernels (corr 6, 7)
        ├─ reward.compute [39ms - 44ms]
        │  └─ reward.term:reach [40ms - 42ms]  → 1 kernel (corr 8)
        └─ observation.compute [45ms - 48ms]  → 1 kernel (corr 9)

    Step 2 (env.step at 60-82ms, slower):

        env.step [60ms - 82ms]
        ├─ sim.step [61ms - 70ms]
        │  └─ physics.solve [62ms - 68ms]  → 2 kernels (corr 10, 11)
        ├─ reward.compute [71ms - 76ms]
        │  └─ reward.term:reach [72ms - 74ms]  → 1 kernel (corr 12)
        └─ observation.compute [77ms - 80ms]  → 1 kernel (corr 13)
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    _create_base_tables(c)
    _create_kernel_tables(c)

    # StringIds for kernel names
    c.execute("INSERT INTO StringIds VALUES (1, 'kernel_physics_a')")
    c.execute("INSERT INTO StringIds VALUES (2, 'kernel_physics_b')")
    c.execute("INSERT INTO StringIds VALUES (3, 'kernel_reward')")
    c.execute("INSERT INTO StringIds VALUES (4, 'kernel_obs')")
    c.execute("INSERT INTO StringIds VALUES (5, 'kernel_mystery')")

    # === Step 0 ===
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (0, 20000000, 59, 'env.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (1000000, 8000000, 59, 'sim.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (2000000, 6000000, 59, 'physics.solve')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (9000000, 14000000, 59, 'reward.compute')")
    c.execute(
        "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (10000000, 12000000, 59, 'reward.term:reach')"
    )
    c.execute(
        "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (15000000, 18000000, 59, 'observation.compute')"
    )

    # Kernels for step 0
    # corr 1: physics kernel A (100us)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (2500000, 2600000, 1)")
    c.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
        "VALUES (3000000, 3100000, 1, 1, 128, 1, 1, 256, 1, 1, 32, 4096, 0, 1, 0)"
    )
    # corr 2: physics kernel B (200us)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (4000000, 4100000, 2)")
    c.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
        "VALUES (4500000, 4700000, 2, 2, 256, 1, 1, 128, 1, 1, 48, 8192, 0, 1, 0)"
    )
    # corr 3: reward kernel (150us)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (10500000, 10600000, 3)")
    c.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
        "VALUES (11000000, 11150000, 3, 3, 64, 1, 1, 512, 1, 1, 24, 2048, 0, 2, 0)"
    )
    # corr 4: obs kernel (80us)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (15500000, 15600000, 4)")
    c.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
        "VALUES (16000000, 16080000, 4, 4, 32, 1, 1, 256, 1, 1, 16, 1024, 0, 1, 0)"
    )
    # corr 5: unattributed kernel (50us, between sim.step and reward.compute)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (8500000, 8600000, 5)")
    c.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
        "VALUES (8700000, 8750000, 5, 5, 16, 1, 1, 128, 1, 1, 8, 512, 0, 1, 0)"
    )

    # === Step 1 (30-50ms) ===
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (30000000, 50000000, 59, 'env.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (31000000, 38000000, 59, 'sim.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (32000000, 36000000, 59, 'physics.solve')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (39000000, 44000000, 59, 'reward.compute')")
    c.execute(
        "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (40000000, 42000000, 59, 'reward.term:reach')"
    )
    c.execute(
        "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (45000000, 48000000, 59, 'observation.compute')"
    )

    # Kernels for step 1
    for corr, rt_s, rt_e, k_s, k_e, sname in [
        (6, 32500000, 32600000, 33000000, 33100000, 1),
        (7, 34000000, 34100000, 34500000, 34700000, 2),
        (8, 40500000, 40600000, 41000000, 41150000, 3),
        (9, 45500000, 45600000, 46000000, 46080000, 4),
    ]:
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (?, ?, ?)",
            (rt_s, rt_e, corr),
        )
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
            "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
            "VALUES (?, ?, ?, ?, 64, 1, 1, 256, 1, 1, 32, 4096, 0, 1, 0)",
            (k_s, k_e, sname, corr),
        )

    # === Step 2 (60-82ms, slower) ===
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (60000000, 82000000, 59, 'env.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (61000000, 70000000, 59, 'sim.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (62000000, 68000000, 59, 'physics.solve')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (71000000, 76000000, 59, 'reward.compute')")
    c.execute(
        "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (72000000, 74000000, 59, 'reward.term:reach')"
    )
    c.execute(
        "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (77000000, 80000000, 59, 'observation.compute')"
    )

    for corr, rt_s, rt_e, k_s, k_e, sname in [
        (10, 62500000, 62600000, 63000000, 63100000, 1),
        (11, 64000000, 64100000, 64500000, 64700000, 2),
        (12, 72500000, 72600000, 73000000, 73150000, 3),
        (13, 77500000, 77600000, 78000000, 78080000, 4),
    ]:
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (?, ?, ?)",
            (rt_s, rt_e, corr),
        )
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
            "registersPerThread, staticSharedMemory, dynamicSharedMemory, streamId, deviceId) "
            "VALUES (?, ?, ?, ?, 64, 1, 1, 256, 1, 1, 32, 4096, 0, 1, 0)",
            (k_s, k_e, sname, corr),
        )

    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def scaling_dbs():
    """Create multiple synthetic DBs simulating different num_envs.

    - DB at 'num_envs=64': sim.step avg ~2ms, reward.compute avg ~0.5ms
    - DB at 'num_envs=256': sim.step avg ~4ms (linear), reward.compute avg ~4ms (quadratic)
    - DB at 'num_envs=1024': sim.step avg ~8ms (linear), reward.compute avg ~32ms (quadratic)
    """
    dbs = {}
    paths = []

    configs = [
        (64, 2_000_000, 500_000),
        (256, 4_000_000, 4_000_000),
        (1024, 8_000_000, 32_000_000),
    ]

    for num_envs, sim_ns, reward_ns in configs:
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        paths.append(path)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        _create_base_tables(c)

        # Create 3 env.step ranges with the target timing
        step_ns = sim_ns + reward_ns + 1_000_000  # +1ms overhead
        for i in range(3):
            base = i * (step_ns + 5_000_000)
            c.execute(
                "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, ?, 59, 'env.step')",
                (base, base + step_ns),
            )
            c.execute(
                "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, ?, 59, 'sim.step')",
                (base + 100_000, base + 100_000 + sim_ns),
            )
            c.execute(
                "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, ?, 59, 'reward.compute')",
                (base + 200_000 + sim_ns, base + 200_000 + sim_ns + reward_ns),
            )

        conn.commit()
        conn.close()
        dbs[num_envs] = path

    yield dbs

    for path in paths:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: query_nvtx_tree
# ---------------------------------------------------------------------------


class TestNvtxTree:
    """Tests for :func:`query_nvtx_tree`."""

    def test_root_is_env_step(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0)
        assert tree is not None
        assert tree.name == "env.step"
        assert tree.depth == 0

    def test_correct_children(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0)
        child_names = [c.name for c in tree.children]
        assert "sim.step" in child_names
        assert "reward.compute" in child_names
        assert "observation.compute" in child_names

    def test_nested_depth(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0)
        sim = next(c for c in tree.children if c.name == "sim.step")
        assert sim.depth == 1
        assert len(sim.children) == 1
        physics = sim.children[0]
        assert physics.name == "physics.solve"
        assert physics.depth == 2

    def test_wall_times(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0)
        assert tree.wall_ns == 20_000_000  # 20ms
        sim = next(c for c in tree.children if c.name == "sim.step")
        assert sim.wall_ns == 7_000_000  # 7ms

    def test_kernel_attribution(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        sim = next(c for c in tree.children if c.name == "sim.step")
        physics = sim.children[0]
        # physics.solve should have 2 kernels directly
        assert physics.kernel_count == 2
        assert physics.direct_kernel_gpu_ns == 300_000  # 100us + 200us

    def test_subtree_kernel_counts(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        # Root should have all 5 kernels in subtree
        assert tree.subtree_kernel_count == 5

    def test_no_kernel_attribution(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=False)
        assert tree.subtree_kernel_count == 0

    def test_step_index_1(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=1)
        assert tree is not None
        assert tree.wall_ns == 20_000_000  # 20ms

    def test_step_index_out_of_range(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=99)
        assert tree is None

    def test_reward_term_nesting(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        reward = next(c for c in tree.children if c.name == "reward.compute")
        assert len(reward.children) == 1
        term = reward.children[0]
        assert term.name == "reward.term:reach"
        assert term.kernel_count == 1  # kernel_reward attributed here


# ---------------------------------------------------------------------------
# Tests: flatten_tree
# ---------------------------------------------------------------------------


class TestFlattenTree:
    """Tests for :func:`flatten_tree`."""

    def test_correct_paths(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=False)
        rows = flatten_tree(tree)
        paths = [r.path for r in rows]
        assert "env.step" in paths
        assert "env.step/sim.step" in paths
        assert "env.step/sim.step/physics.solve" in paths
        assert "env.step/reward.compute/reward.term:reach" in paths

    def test_max_depth(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=False)
        rows = flatten_tree(tree, max_depth=1)
        max_d = max(r.depth for r in rows)
        assert max_d == 1  # Only root and direct children

    def test_pct_of_parent(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=False)
        rows = flatten_tree(tree)
        root = rows[0]
        assert root.pct_of_parent == pytest.approx(100.0)
        sim = next(r for r in rows if r.name == "sim.step")
        expected_pct = 7_000_000 / 20_000_000 * 100
        assert sim.pct_of_parent == pytest.approx(expected_pct)

    def test_subtree_gpu_ns(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        rows = flatten_tree(tree)
        sim_row = next(r for r in rows if r.name == "sim.step")
        # sim.step subtree GPU = physics.solve kernels (100us + 200us)
        assert sim_row.subtree_kernel_gpu_ns == 300_000

    def test_min_wall_ns(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=False)
        rows = flatten_tree(tree, min_wall_ns=5_000_000)
        # Only ranges >= 5ms should appear
        for r in rows:
            assert r.wall_ns >= 5_000_000


# ---------------------------------------------------------------------------
# Tests: query_step_timeseries
# ---------------------------------------------------------------------------


class TestStepTimeseries:
    """Tests for :func:`query_step_timeseries`."""

    def test_returns_all_steps(self, rich_db):
        snapshots = query_step_timeseries(rich_db)
        assert len(snapshots) == 3

    def test_wall_times(self, rich_db):
        snapshots = query_step_timeseries(rich_db)
        assert snapshots[0].wall_ns == 20_000_000
        assert snapshots[1].wall_ns == 20_000_000
        assert snapshots[2].wall_ns == 22_000_000  # step 2 is slower

    def test_range_times(self, rich_db):
        snapshots = query_step_timeseries(rich_db)
        s0 = snapshots[0]
        assert "sim.step" in s0.range_times
        assert s0.range_times["sim.step"] == 7_000_000

    def test_filter_range_names(self, rich_db):
        snapshots = query_step_timeseries(rich_db, range_names=["sim.step"])
        s0 = snapshots[0]
        assert "sim.step" in s0.range_times
        assert "reward.compute" not in s0.range_times

    def test_chronological_order(self, rich_db):
        snapshots = query_step_timeseries(rich_db)
        starts = [s.start_ns for s in snapshots]
        assert starts == sorted(starts)

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (x INTEGER)")
        conn.commit()
        conn.close()
        assert query_step_timeseries(db_path) == []


# ---------------------------------------------------------------------------
# Tests: query_range_distribution
# ---------------------------------------------------------------------------


class TestRangeDistribution:
    """Tests for :func:`query_range_distribution`."""

    def test_env_step_count(self, rich_db):
        dist = query_range_distribution(rich_db, "env.step")
        assert dist is not None
        assert dist.count == 3

    def test_mean(self, rich_db):
        dist = query_range_distribution(rich_db, "env.step")
        # Steps: 20ms, 20ms, 22ms → mean ≈ 20.67ms
        expected_mean = (20_000_000 + 20_000_000 + 22_000_000) / 3
        assert dist.mean_ns == pytest.approx(expected_mean)

    def test_min_max(self, rich_db):
        dist = query_range_distribution(rich_db, "env.step")
        assert dist.min_ns == 20_000_000
        assert dist.max_ns == 22_000_000

    def test_stddev(self, rich_db):
        dist = query_range_distribution(rich_db, "env.step")
        assert dist.stddev_ns > 0  # Not all same duration

    def test_nonexistent_range(self, rich_db):
        dist = query_range_distribution(rich_db, "nonexistent.range")
        assert dist is None

    def test_single_occurrence(self, tmp_path):
        db_path = str(tmp_path / "single.sqlite")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        _create_base_tables(c)
        c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (0, 5000000, 59, 'test')")
        conn.commit()
        conn.close()
        dist = query_range_distribution(db_path, "test")
        assert dist is not None
        assert dist.count == 1
        assert dist.stddev_ns == 0.0


# ---------------------------------------------------------------------------
# Tests: query_kernel_details
# ---------------------------------------------------------------------------


class TestKernelDetails:
    """Tests for :func:`query_kernel_details`."""

    def test_returns_all_kernels(self, rich_db):
        details = query_kernel_details(rich_db)
        assert len(details) > 0
        names = {d.name for d in details}
        assert "kernel_physics_a" in names
        assert "kernel_reward" in names

    def test_hardware_info(self, rich_db):
        details = query_kernel_details(rich_db)
        phys_a = next(d for d in details if d.name == "kernel_physics_a")
        assert phys_a.grid == (128, 1, 1)
        assert phys_a.block == (256, 1, 1)
        assert phys_a.registers_per_thread == 32
        assert phys_a.static_shared_memory_bytes == 4096

    def test_name_pattern_filter(self, rich_db):
        details = query_kernel_details(rich_db, name_pattern="physics")
        assert all("physics" in d.name for d in details)

    def test_step_index_filter(self, rich_db):
        details_s0 = query_kernel_details(rich_db, step_index=0)
        details_s1 = query_kernel_details(rich_db, step_index=1)
        # Step 0 has 5 kernels (including unattributed), step 1 has 4
        assert len(details_s0) == 5
        assert len(details_s1) == 4

    def test_duration(self, rich_db):
        details = query_kernel_details(rich_db)
        phys_a = next(d for d in details if d.name == "kernel_physics_a")
        assert phys_a.duration_ns == 100_000  # 100us


# ---------------------------------------------------------------------------
# Tests: query_nvtx_kernel_map
# ---------------------------------------------------------------------------


class TestNvtxKernelMap:
    """Tests for :func:`query_nvtx_kernel_map`."""

    def test_full_paths(self, rich_db):
        kmap = query_nvtx_kernel_map(rich_db, step_index=0, full_path=True)
        assert "env.step/sim.step/physics.solve" in kmap
        assert len(kmap["env.step/sim.step/physics.solve"]) == 2

    def test_short_names(self, rich_db):
        kmap = query_nvtx_kernel_map(rich_db, step_index=0, full_path=False)
        assert "physics.solve" in kmap

    def test_reward_term_kernels(self, rich_db):
        kmap = query_nvtx_kernel_map(rich_db, step_index=0, full_path=True)
        reward_path = "env.step/reward.compute/reward.term:reach"
        assert reward_path in kmap
        assert len(kmap[reward_path]) == 1
        assert kmap[reward_path][0].name == "kernel_reward"

    def test_unattributed_to_env_step(self, rich_db):
        """Kernels between sub-ranges should be attributed to env.step."""
        kmap = query_nvtx_kernel_map(rich_db, step_index=0, full_path=True)
        # The mystery kernel at 8.5ms is between sim.step [1-8ms] and reward.compute [9-14ms]
        # It falls within env.step but no sub-range → attributed to env.step
        env_kernels = kmap.get("env.step", [])
        mystery = [k for k in env_kernels if k.name == "kernel_mystery"]
        assert len(mystery) == 1


# ---------------------------------------------------------------------------
# Tests: compare_trees
# ---------------------------------------------------------------------------


class TestCompareTrees:
    """Tests for :func:`compare_trees`."""

    def test_matched_paths(self, rich_db):
        tree_a = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        tree_b = query_nvtx_tree(rich_db, step_index=2, attribute_kernels=True)
        diff = compare_trees(tree_a, tree_b, "step0", "step2")
        matched = [e for e in diff.entries if e.status == "matched"]
        paths = {e.path for e in matched}
        assert "env.step" in paths
        assert "env.step/sim.step" in paths

    def test_wall_delta(self, rich_db):
        tree_a = query_nvtx_tree(rich_db, step_index=0)
        tree_b = query_nvtx_tree(rich_db, step_index=2)
        diff = compare_trees(tree_a, tree_b)
        root = next(e for e in diff.entries if e.path == "env.step")
        # step0: 20ms, step2: 22ms → +10%
        assert root.wall_delta_pct == pytest.approx(10.0)

    def test_structural_divergence(self):
        """Trees with different structure should show only_a / only_b."""
        tree_a = NvtxNode(
            "root",
            0,
            100,
            100,
            0,
            [
                NvtxNode("child_a", 10, 50, 40, 1),
            ],
        )
        tree_b = NvtxNode(
            "root",
            0,
            100,
            100,
            0,
            [
                NvtxNode("child_b", 10, 50, 40, 1),
            ],
        )
        diff = compare_trees(tree_a, tree_b)
        statuses = {e.path: e.status for e in diff.entries}
        assert statuses["root/child_a"] == "only_a"
        assert statuses["root/child_b"] == "only_b"
        assert statuses["root"] == "matched"

    def test_sorted_by_delta(self, rich_db):
        tree_a = query_nvtx_tree(rich_db, step_index=0)
        tree_b = query_nvtx_tree(rich_db, step_index=2)
        diff = compare_trees(tree_a, tree_b)
        deltas = [abs(e.a_wall_ns - e.b_wall_ns) for e in diff.entries]
        assert deltas == sorted(deltas, reverse=True)


# ---------------------------------------------------------------------------
# Tests: analyze_scaling
# ---------------------------------------------------------------------------


class TestAnalyzeScaling:
    """Tests for :func:`analyze_scaling`."""

    def test_returns_entries(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        assert len(result.entries) > 0

    def test_reward_scales_worse(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        by_name = {e.range_name: e for e in result.entries}
        # reward.compute scales quadratically (0.5ms → 4ms → 32ms)
        # sim.step scales linearly (2ms → 4ms → 8ms)
        assert by_name["reward.compute"].exponent > by_name["sim.step"].exponent

    def test_classification(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        by_name = {e.range_name: e for e in result.entries}
        assert by_name["reward.compute"].classification == "super-linear"

    def test_sorted_worst_first(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        exponents = [e.exponent for e in result.entries]
        assert exponents == sorted(exponents, reverse=True)

    def test_filter_range_names(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs, range_names=["sim.step"])
        assert len(result.entries) == 1
        assert result.entries[0].range_name == "sim.step"

    def test_r_squared(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        for entry in result.entries:
            assert 0.0 <= entry.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Tests: Layer 3 (derived metrics)
# ---------------------------------------------------------------------------


class TestFindHotpath:
    """Tests for :func:`find_hotpath`."""

    def test_follows_heaviest_child(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=False)
        path = find_hotpath(tree, metric="wall_ns")
        names = [n.name for n in path]
        assert names[0] == "env.step"
        # sim.step (7ms) is heaviest child of env.step
        assert names[1] == "sim.step"
        # physics.solve (4ms) is the only child of sim.step
        assert names[2] == "physics.solve"

    def test_gpu_metric(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        path = find_hotpath(tree, metric="subtree_kernel_gpu_ns")
        names = [n.name for n in path]
        assert "env.step" in names


class TestGpuUtilization:
    """Tests for :func:`gpu_utilization`."""

    def test_returns_ratios(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        util = gpu_utilization(tree)
        assert "env.step" in util
        assert "sim.step" in util
        # All ratios should be between 0 and 1 (no overlapping streams in test)
        for name, ratio in util.items():
            assert ratio >= 0.0

    def test_cpu_only_range_has_low_util(self):
        tree = NvtxNode("test", 0, 10_000_000, 10_000_000, 0)
        util = gpu_utilization(tree)
        assert util["test"] == 0.0


class TestScalingBottlenecks:
    """Tests for :func:`scaling_bottlenecks`."""

    def test_filters_by_threshold(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        bottlenecks = scaling_bottlenecks(result, exponent_threshold=1.2)
        for entry in bottlenecks:
            assert entry.exponent >= 1.2

    def test_returns_empty_with_high_threshold(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        bottlenecks = scaling_bottlenecks(result, exponent_threshold=100.0)
        assert bottlenecks == []


# ---------------------------------------------------------------------------
# Tests: Formatters
# ---------------------------------------------------------------------------


class TestClassifyKernel:
    """Tests for kernel name classification."""

    def test_warp_kernel(self):
        op, cat = classify_kernel("split_transform_to_pos_1d_7b86c500_cuda_kernel_forward")
        assert cat == "warp"
        assert "extract_position" in op

    def test_warp_fused(self):
        op, cat = classify_kernel("fused_neg_cat")
        assert cat == "warp"

    def test_torch_reduce(self):
        op, cat = classify_kernel("void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float>>")
        assert cat == "torch_reduce"

    def test_torch_elementwise(self):
        op, cat = classify_kernel("void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor>")
        assert cat == "torch_ewise"

    def test_torch_cross(self):
        op, cat = classify_kernel("void at::native::cross_kernel<float, OffsetCalculator<(int)3>>")
        assert cat == "torch_cross"

    def test_torch_cat(self):
        op, cat = classify_kernel("void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<>")
        assert cat == "torch_cat"

    def test_unknown(self):
        op, cat = classify_kernel("some_totally_unknown_kernel")
        assert cat == "other"


class TestTermDecomposition:
    """Tests for term-level CPU time decomposition."""

    def test_decompose_named_range(self, rich_db):
        """Decompose physics.solve which has 2 kernels."""
        decomp = query_term_decomposition(rich_db, step_index=0, range_name="physics.solve")
        assert decomp is not None
        assert decomp.name == "physics.solve"
        assert decomp.n_dispatches == 2
        assert decomp.wall_ns == 4_000_000  # 6ms - 2ms
        assert decomp.pre_python_ns > 0
        assert decomp.between_python_ns > 0
        assert decomp.gpu_kernel_ns > 0
        # Total should roughly equal wall time
        total = decomp.pre_python_ns + decomp.between_python_ns + decomp.api_overhead_ns + decomp.post_python_ns
        assert abs(total - decomp.wall_ns) < 1000  # within 1μs rounding

    def test_decompose_with_dispatches(self, rich_db):
        """Check that include_dispatches populates the dispatch list."""
        decomp = query_term_decomposition(
            rich_db,
            step_index=0,
            range_name="physics.solve",
            include_dispatches=True,
        )
        assert decomp is not None
        assert len(decomp.dispatches) == 2
        assert decomp.dispatches[0].gpu_duration_ns > 0
        assert decomp.dispatches[0].gap_before_ns > 0

    def test_decompose_single_kernel(self, rich_db):
        """Decompose reward.term:reach which has 1 kernel."""
        decomp = query_term_decomposition(rich_db, step_index=0, range_name="reward.term:reach")
        assert decomp is not None
        assert decomp.n_dispatches == 1
        assert decomp.between_python_ns == 0  # only 1 dispatch, no "between"
        assert decomp.avg_gap_ns == 0
        assert decomp.max_gap_ns == 0

    def test_decompose_nonexistent(self, rich_db):
        """Nonexistent range returns None."""
        decomp = query_term_decomposition(rich_db, step_index=0, range_name="does.not.exist")
        assert decomp is None

    def test_decompose_properties(self, rich_db):
        decomp = query_term_decomposition(rich_db, step_index=0, range_name="physics.solve")
        assert decomp.python_total_ns == decomp.pre_python_ns + decomp.between_python_ns + decomp.post_python_ns
        assert 0 <= decomp.python_pct <= 100
        assert 0 <= decomp.between_pct <= 100


class TestAllTermDecompositions:
    """Tests for batch term decomposition."""

    def test_all_depth1(self, rich_db):
        """Get all depth-1 ranges (sim.step, reward.compute, observation.compute)."""
        decomps = query_all_term_decompositions(rich_db, step_index=0, min_depth=1, max_depth=1)
        assert len(decomps) == 3
        names = {d.name for d in decomps}
        assert names == {"sim.step", "reward.compute", "observation.compute"}
        # Sorted by wall_ns descending
        assert decomps[0].wall_ns >= decomps[1].wall_ns >= decomps[2].wall_ns

    def test_all_depth1_2(self, rich_db):
        """Get all depth 1-2 ranges."""
        decomps = query_all_term_decompositions(rich_db, step_index=0, min_depth=1, max_depth=2)
        assert len(decomps) == 5  # sim.step, physics.solve, reward.compute, reward.term:reach, observation.compute
        names = {d.name for d in decomps}
        assert "physics.solve" in names
        assert "reward.term:reach" in names

    def test_all_with_dispatches(self, rich_db):
        decomps = query_all_term_decompositions(
            rich_db,
            step_index=0,
            min_depth=2,
            max_depth=2,
            include_dispatches=True,
        )
        for d in decomps:
            if d.n_dispatches > 0:
                assert len(d.dispatches) == d.n_dispatches

    def test_different_step(self, rich_db):
        """Step 1 should also work."""
        decomps = query_all_term_decompositions(rich_db, step_index=1, min_depth=1, max_depth=1)
        assert len(decomps) == 3

    def test_format_term_decomposition(self, rich_db):
        decomps = query_all_term_decompositions(rich_db, step_index=0, min_depth=1, max_depth=2)
        text = format_term_decomposition(decomps)
        assert "NVTX Range" in text
        assert "Between%" in text
        assert "sim.step" in text


class TestFormatters:
    """Tests for formatting functions."""

    def test_format_tree(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0, attribute_kernels=True)
        text = format_tree(tree)
        assert "env.step" in text
        assert "sim.step" in text
        assert "physics.solve" in text
        assert "wall=" in text

    def test_format_tree_max_depth(self, rich_db):
        tree = query_nvtx_tree(rich_db, step_index=0)
        text = format_tree(tree, max_depth=1)
        assert "physics.solve" not in text
        assert "children" in text

    def test_format_comparison(self, rich_db):
        tree_a = query_nvtx_tree(rich_db, step_index=0)
        tree_b = query_nvtx_tree(rich_db, step_index=2)
        diff = compare_trees(tree_a, tree_b, "step0", "step2")
        text = format_comparison(diff)
        assert "step0" in text
        assert "step2" in text
        assert "matched" in text

    def test_format_scaling(self, scaling_dbs):
        result = analyze_scaling(scaling_dbs)
        text = format_scaling(result)
        assert "Exponent" in text
        assert "sim.step" in text

    def test_format_distribution(self, rich_db):
        dist = query_range_distribution(rich_db, "env.step")
        text = format_distribution(dist)
        assert "mean" in text
        assert "stddev" in text
        assert "p50" in text

    def test_format_timeseries(self, rich_db):
        snapshots = query_step_timeseries(rich_db)
        text = format_timeseries(snapshots)
        assert "Step" in text
        assert "Wall ms" in text

    def test_format_timeseries_empty(self):
        text = format_timeseries([])
        assert "no steps found" in text


class TestAlignDispatchesToSource:
    """Tests for align_dispatches_to_source()."""

    @staticmethod
    def _make_dispatch(category: str, op: str = "") -> DispatchEvent:
        return DispatchEvent(
            op_name=op or f"op:{category}",
            category=category,
            kernel_name="",
            api_start_ns=0,
            api_end_ns=100,
            gpu_duration_ns=50,
            gap_before_ns=200,
        )

    def test_perfect_alignment(self):
        template = [
            SourceMapEntry("warp", "L1: extract", "data"),
            SourceMapEntry("torch_reduce", "L2: norm", "compute"),
            SourceMapEntry("torch_ewise", "L3: tanh", "compute"),
        ]
        dispatches = [
            self._make_dispatch("warp"),
            self._make_dispatch("torch_reduce"),
            self._make_dispatch("torch_ewise"),
        ]
        result = align_dispatches_to_source(template, dispatches)
        assert result.matched == 3
        assert result.total == 3
        assert result.match_pct == 100.0
        for tmpl, d_idx in result.entries:
            assert tmpl is not None

    def test_extra_dispatches(self):
        template = [
            SourceMapEntry("warp", "L1: extract", "data"),
            SourceMapEntry("torch_reduce", "L2: norm", "compute"),
        ]
        dispatches = [
            self._make_dispatch("warp"),
            self._make_dispatch("torch_ewise"),  # extra — no match
            self._make_dispatch("torch_reduce"),
        ]
        result = align_dispatches_to_source(template, dispatches)
        assert result.matched == 2
        assert result.total == 3
        # dispatch 1 should be unmatched
        assert result.entries[1][0] is None

    def test_empty_dispatches(self):
        template = [SourceMapEntry("warp", "L1", "data")]
        result = align_dispatches_to_source(template, [])
        assert result.matched == 0
        assert result.total == 0
        assert result.match_pct == 0.0

    def test_empty_template(self):
        dispatches = [self._make_dispatch("warp")]
        result = align_dispatches_to_source([], dispatches)
        assert result.matched == 0
        assert result.total == 1
        assert result.entries[0][0] is None

    def test_regex_pattern(self):
        template = [
            SourceMapEntry("torch_.*", "L1: any torch op", "compute"),
        ]
        dispatches = [self._make_dispatch("torch_reduce")]
        result = align_dispatches_to_source(template, dispatches)
        assert result.matched == 1
