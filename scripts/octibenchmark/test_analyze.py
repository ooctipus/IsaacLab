# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the octibenchmark analyze module.

These tests create a synthetic nsys sqlite database and verify that the
analysis functions extract the expected results without requiring an
actual nsys run or GPU.

Run with::

    python -m pytest scripts/octibenchmark/test_analyze.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

import pytest

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from octibenchmark.analyze import (
    aggregate_by_patterns,
    format_kernel_table,
    format_nvtx_table,
    format_step_anatomy,
    format_step_anatomy_averaged,
    query_gpu_kernels,
    query_nvtx_ranges,
    query_step_anatomy,
    query_step_anatomy_averaged,
)


@pytest.fixture
def synthetic_db():
    """Create a temporary sqlite database mimicking nsys export schema."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Create StringIds table
    cursor.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cursor.execute("INSERT INTO StringIds VALUES (1, 'env.step')")
    cursor.execute("INSERT INTO StringIds VALUES (2, 'sim.step')")
    cursor.execute("INSERT INTO StringIds VALUES (3, 'reward.compute')")
    cursor.execute("INSERT INTO StringIds VALUES (4, 'observation.compute')")
    cursor.execute("INSERT INTO StringIds VALUES (5, 'env._reset_idx')")
    cursor.execute("INSERT INTO StringIds VALUES (10, 'my_index_kernel')")
    cursor.execute("INSERT INTO StringIds VALUES (11, 'my_scatter_kernel')")
    cursor.execute("INSERT INTO StringIds VALUES (12, 'my_mask_select_kernel')")
    cursor.execute("INSERT INTO StringIds VALUES (13, 'matmul_kernel')")

    # Create NVTX_EVENTS table with start/end schema (modern nsys)
    cursor.execute("""
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER,
            rangeId INTEGER, category INTEGER, color INTEGER,
            text TEXT, globalTid INTEGER, endGlobalTid INTEGER,
            textId INTEGER, domainId INTEGER,
            uint64Value INTEGER, int64Value INTEGER, doubleValue REAL,
            uint32Value INTEGER, int32Value INTEGER, floatValue REAL,
            jsonTextId INTEGER, jsonText TEXT, binaryData TEXT
        )
    """)

    # Insert NVTX push/pop ranges (eventType=59)
    # env.step: 3 calls, each 10ms
    for i in range(3):
        base = i * 20_000_000  # 20ms apart
        cursor.execute(
            "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, ?, 59, 'env.step')",
            (base, base + 10_000_000),
        )
    # sim.step: 6 calls (2 per env.step), each 2ms
    for i in range(6):
        base = 1_000_000 + i * 5_000_000
        cursor.execute(
            "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, ?, 59, 'sim.step')",
            (base, base + 2_000_000),
        )
    # reward.compute: 3 calls, each 1ms
    for i in range(3):
        base = 5_000_000 + i * 20_000_000
        cursor.execute(
            "INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, ?, 59, 'reward.compute')",
            (base, base + 1_000_000),
        )

    # Create GPU kernel table
    cursor.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, shortName INTEGER,
            deviceId INTEGER, contextId INTEGER, streamId INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            registersPerThread INTEGER, staticSharedMemory INTEGER,
            dynamicSharedMemory INTEGER, localMemoryPerThread INTEGER,
            localMemoryTotal INTEGER
        )
    """)

    # Insert kernel executions
    # my_index_kernel: 10 calls, 500us each
    for i in range(10):
        cursor.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName) VALUES (?, ?, 10)",
            (i * 1_000_000, i * 1_000_000 + 500_000),
        )
    # my_scatter_kernel: 5 calls, 200us each
    for i in range(5):
        cursor.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName) VALUES (?, ?, 11)",
            (i * 2_000_000, i * 2_000_000 + 200_000),
        )
    # my_mask_select_kernel: 8 calls, 300us each
    for i in range(8):
        cursor.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName) VALUES (?, ?, 12)",
            (i * 1_500_000, i * 1_500_000 + 300_000),
        )
    # matmul_kernel: 2 calls, 1ms each
    for i in range(2):
        cursor.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName) VALUES (?, ?, 13)",
            (i * 5_000_000, i * 5_000_000 + 1_000_000),
        )

    conn.commit()
    conn.close()

    yield path

    os.unlink(path)


class TestQueryNvtxRanges:
    """Tests for :func:`query_nvtx_ranges`."""

    def test_returns_all_ranges(self, synthetic_db):
        results = query_nvtx_ranges(synthetic_db)
        names = {r["name"] for r in results}
        assert "env.step" in names
        assert "sim.step" in names
        assert "reward.compute" in names

    def test_correct_counts(self, synthetic_db):
        results = query_nvtx_ranges(synthetic_db)
        by_name = {r["name"]: r for r in results}
        assert by_name["env.step"]["count"] == 3
        assert by_name["sim.step"]["count"] == 6
        assert by_name["reward.compute"]["count"] == 3

    def test_correct_total_time(self, synthetic_db):
        results = query_nvtx_ranges(synthetic_db)
        by_name = {r["name"]: r for r in results}
        # env.step: 3 calls * 10ms = 30ms = 30_000_000 ns
        assert by_name["env.step"]["total_ns"] == 30_000_000
        # sim.step: 6 calls * 2ms = 12ms
        assert by_name["sim.step"]["total_ns"] == 12_000_000

    def test_sorted_by_total_time_desc(self, synthetic_db):
        results = query_nvtx_ranges(synthetic_db)
        totals = [r["total_ns"] for r in results]
        assert totals == sorted(totals, reverse=True)

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (x INTEGER)")
        conn.commit()
        conn.close()
        results = query_nvtx_ranges(db_path)
        assert results == []


class TestQueryGpuKernels:
    """Tests for :func:`query_gpu_kernels`."""

    def test_returns_all_kernels(self, synthetic_db):
        results = query_gpu_kernels(synthetic_db)
        names = {r["name"] for r in results}
        assert "my_index_kernel" in names
        assert "my_scatter_kernel" in names
        assert "my_mask_select_kernel" in names
        assert "matmul_kernel" in names

    def test_correct_counts(self, synthetic_db):
        results = query_gpu_kernels(synthetic_db)
        by_name = {r["name"]: r for r in results}
        assert by_name["my_index_kernel"]["count"] == 10
        assert by_name["my_scatter_kernel"]["count"] == 5
        assert by_name["matmul_kernel"]["count"] == 2

    def test_correct_total_time(self, synthetic_db):
        results = query_gpu_kernels(synthetic_db)
        by_name = {r["name"]: r for r in results}
        # my_index_kernel: 10 * 500us = 5ms = 5_000_000 ns
        assert by_name["my_index_kernel"]["total_ns"] == 5_000_000

    def test_sorted_by_total_time_desc(self, synthetic_db):
        results = query_gpu_kernels(synthetic_db)
        totals = [r["total_ns"] for r in results]
        assert totals == sorted(totals, reverse=True)


class TestAggregateByPatterns:
    """Tests for :func:`aggregate_by_patterns`."""

    def test_single_pattern(self, synthetic_db):
        kernels = query_gpu_kernels(synthetic_db)
        agg = aggregate_by_patterns(kernels, ["index"])
        assert "index" in agg
        assert agg["index"]["count"] == 10  # my_index_kernel

    def test_multiple_patterns(self, synthetic_db):
        kernels = query_gpu_kernels(synthetic_db)
        agg = aggregate_by_patterns(kernels, ["index|scatter", "mask"])
        # index|scatter matches my_index_kernel (10) + my_scatter_kernel (5)
        assert agg["index|scatter"]["count"] == 15
        # mask matches my_mask_select_kernel (8)
        assert agg["mask"]["count"] == 8

    def test_no_match(self, synthetic_db):
        kernels = query_gpu_kernels(synthetic_db)
        agg = aggregate_by_patterns(kernels, ["nonexistent_pattern"])
        assert agg["nonexistent_pattern"]["count"] == 0
        assert agg["nonexistent_pattern"]["total_ns"] == 0


class TestFormatting:
    """Tests for table formatting functions."""

    def test_format_nvtx_table_nonempty(self, synthetic_db):
        ranges = query_nvtx_ranges(synthetic_db)
        table = format_nvtx_table(ranges)
        assert "env.step" in table
        assert "sim.step" in table
        assert "% of step" in table

    def test_format_nvtx_table_empty(self):
        table = format_nvtx_table([])
        assert "no NVTX ranges found" in table

    def test_format_kernel_table_nonempty(self, synthetic_db):
        kernels = query_gpu_kernels(synthetic_db)
        table = format_kernel_table(kernels, top_n=2)
        assert "my_index_kernel" in table
        # Should show "and X more"
        assert "more kernels" in table

    def test_format_kernel_table_empty(self):
        table = format_kernel_table([])
        assert "no GPU kernels found" in table


@pytest.fixture
def anatomy_db():
    """Create a synthetic DB with the full correlation chain for step anatomy.

    Layout for step 0 (env.step at 0-10ms):
      sim.step      [1ms - 4ms]   -> 2 kernels (corr 1, 2)
      reward.compute [5ms - 7ms]  -> 1 kernel  (corr 3)
      obs.compute    [7ms - 9ms]  -> 1 kernel  (corr 4)
      (unattributed)              -> 1 kernel  (corr 5, between ranges)
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute("INSERT INTO StringIds VALUES (1, 'kernel_physics')")
    c.execute("INSERT INTO StringIds VALUES (2, 'kernel_physics2')")
    c.execute("INSERT INTO StringIds VALUES (3, 'kernel_reward')")
    c.execute("INSERT INTO StringIds VALUES (4, 'kernel_obs')")
    c.execute("INSERT INTO StringIds VALUES (5, 'kernel_mystery')")

    c.execute("""
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

    # env.step: 0 - 10ms
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (0, 10000000, 59, 'env.step')")
    # apply.action: 0.2ms - 0.8ms (CPU-only, no GPU kernels launched)
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (200000, 800000, 59, 'apply.action')")
    # sim.step: 1ms - 4ms (inside env.step)
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (1000000, 4000000, 59, 'sim.step')")
    # reward.compute: 5ms - 7ms
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (5000000, 7000000, 59, 'reward.compute')")
    # obs.compute: 7ms - 9ms
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (7000000, 9000000, 59, 'obs.compute')")

    # A second env.step for step_index=1 tests
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (20000000, 30000000, 59, 'env.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (21000000, 24000000, 59, 'sim.step')")

    c.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER
        )
    """)

    c.execute("""
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

    # Correlation chain for step 0:
    # corr 1: rt in sim.step -> kernel_physics (100us GPU)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (1500000, 1600000, 1)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (2000000, 2100000, 1, 1)")

    # corr 2: rt in sim.step -> kernel_physics2 (200us GPU)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (2500000, 2600000, 2)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (3000000, 3200000, 2, 2)")

    # corr 3: rt in reward.compute -> kernel_reward (150us GPU)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (5500000, 5600000, 3)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (6000000, 6150000, 3, 3)")

    # corr 4: rt in obs.compute -> kernel_obs (80us GPU)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (7500000, 7600000, 4)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (8000000, 8080000, 4, 4)")

    # corr 5: rt between ranges (4.5ms, inside env.step but outside sub-ranges) -> kernel_mystery
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (4500000, 4600000, 5)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (4800000, 4850000, 5, 5)")

    # Correlation chain for step 1:
    # corr 6: rt in sim.step of step 1 -> kernel_physics
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (21500000, 21600000, 6)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (22000000, 22100000, 1, 6)")

    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


class TestStepAnatomy:
    """Tests for :func:`query_step_anatomy` and :func:`format_step_anatomy`."""

    def test_returns_all_entries_in_step(self, anatomy_db):
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=0)
        # 5 kernel entries + 1 CPU-only entry (apply.action)
        assert len(anatomy) == 6

    def test_execution_order(self, anatomy_db):
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=0)
        ranges = [e["nvtx_range"] for e in anatomy]
        # CPU-only apply.action first, then sim.step kernels, unattributed, reward, obs
        assert ranges[0] == "apply.action"
        assert ranges[1] == "sim.step"
        assert ranges[2] == "sim.step"
        assert ranges[3] is None  # unattributed
        assert ranges[4] == "reward.compute"
        assert ranges[5] == "obs.compute"

    def test_cpu_only_entry(self, anatomy_db):
        """CPU-only NVTX ranges appear with kernel_name=None and wall_ns > 0."""
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=0)
        cpu_entry = anatomy[0]
        assert cpu_entry["nvtx_range"] == "apply.action"
        assert cpu_entry["kernel_name"] is None
        assert cpu_entry["kernel_ns"] == 0
        assert cpu_entry["wall_ns"] == 600000  # 0.8ms - 0.2ms

    def test_kernel_names(self, anatomy_db):
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=0)
        kernel_entries = [e for e in anatomy if e["kernel_name"]]
        assert kernel_entries[0]["kernel_name"] == "kernel_physics"
        assert kernel_entries[1]["kernel_name"] == "kernel_physics2"
        assert kernel_entries[4]["kernel_name"] == "kernel_obs"

    def test_innermost_range_attribution(self, anatomy_db):
        """Kernels in sim.step are attributed to sim.step, not env.step."""
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=0)
        for entry in anatomy:
            if entry["kernel_name"] in ("kernel_physics", "kernel_physics2"):
                assert entry["nvtx_range"] == "sim.step"

    def test_step_index_1(self, anatomy_db):
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=1)
        assert len(anatomy) == 1
        assert anatomy[0]["nvtx_range"] == "sim.step"
        assert anatomy[0]["kernel_name"] == "kernel_physics"

    def test_step_index_out_of_range(self, anatomy_db):
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=99)
        assert anatomy == []

    def test_missing_tables(self, tmp_path):
        db_path = str(tmp_path / "no_runtime.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, text TEXT)")
        conn.commit()
        conn.close()
        anatomy, _ = query_step_anatomy(db_path)
        assert anatomy == []

    def test_wall_times_returned(self, anatomy_db):
        """query_step_anatomy returns per-range wall times."""
        _, wall_times = query_step_anatomy(anatomy_db, step_index=0)
        assert "sim.step" in wall_times
        assert wall_times["sim.step"] == 3_000_000  # 4ms - 1ms
        assert "reward.compute" in wall_times
        assert wall_times["reward.compute"] == 2_000_000  # 7ms - 5ms

    def test_format_nonempty(self, anatomy_db):
        anatomy, wall_times = query_step_anatomy(anatomy_db, step_index=0)
        text = format_step_anatomy(anatomy, step_index=0, range_wall_times=wall_times)
        assert "Step #0 anatomy" in text
        assert "5 kernel launches" in text
        assert "sim.step" in text
        assert "reward.compute" in text
        assert "UNATTRIBUTED" in text
        assert "apply.action" in text

    def test_format_cpu_only_verbose(self, anatomy_db):
        """CPU-only ranges show wall time and 'CPU only' label in verbose mode."""
        anatomy, wall_times = query_step_anatomy(anatomy_db, step_index=0)
        text = format_step_anatomy(anatomy, step_index=0, verbose=True, range_wall_times=wall_times)
        assert "CPU only" in text
        assert "apply.action" in text

    def test_format_cpu_only_collapsed(self, anatomy_db):
        """CPU-only ranges show '(cpu)' label in collapsed mode."""
        anatomy, wall_times = query_step_anatomy(anatomy_db, step_index=0)
        text = format_step_anatomy(anatomy, step_index=0, verbose=False, range_wall_times=wall_times)
        assert "(cpu)" in text
        assert "apply.action" in text

    def test_format_wall_time_shown(self, anatomy_db):
        """Collapsed format shows wall time alongside GPU time."""
        anatomy, wall_times = query_step_anatomy(anatomy_db, step_index=0)
        text = format_step_anatomy(anatomy, step_index=0, verbose=False, range_wall_times=wall_times)
        assert "Wall ms" in text
        assert "GPU ms" in text

    def test_format_empty(self):
        text = format_step_anatomy([])
        assert "no step anatomy data" in text

    def test_format_collapsed(self, anatomy_db):
        """verbose=False should not show individual kernel names."""
        anatomy, _ = query_step_anatomy(anatomy_db, step_index=0)
        text = format_step_anatomy(anatomy, step_index=0, verbose=False)
        assert "sim.step" in text
        assert "kernel_physics" not in text


class TestStepAnatomyAveraged:
    """Tests for :func:`query_step_anatomy_averaged`."""

    def test_averages_across_steps(self, anatomy_db):
        averaged, num_steps = query_step_anatomy_averaged(anatomy_db)
        # Step 0 has structure: sim.step, None, reward.compute, obs.compute
        # Step 1 has structure: sim.step
        # They differ, so only matching steps are averaged.
        # Step 0 structure is unique (1 occurrence), step 1 is unique (1 occurrence).
        # The most common wins — both have count 1, so whichever comes first.
        assert num_steps >= 1
        assert len(averaged) > 0

    def test_range_names_preserved(self, anatomy_db):
        averaged, _ = query_step_anatomy_averaged(anatomy_db)
        names = [r["nvtx_range"] for r in averaged]
        assert "sim.step" in names

    def test_avg_ns_positive(self, anatomy_db):
        averaged, _ = query_step_anatomy_averaged(anatomy_db)
        for r in averaged:
            assert r["avg_ns"] >= 0
            assert r["avg_kernel_count"] >= 0

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (x INTEGER)")
        conn.commit()
        conn.close()
        averaged, num_steps = query_step_anatomy_averaged(db_path)
        assert averaged == []
        assert num_steps == 0

    def test_format_averaged(self, anatomy_db):
        averaged, num_steps = query_step_anatomy_averaged(anatomy_db)
        text = format_step_anatomy_averaged(averaged, num_steps)
        assert "averaged" in text
        assert "sim.step" in text
        assert "Wall ms" in text
        assert "GPU ms" in text

    def test_format_averaged_empty(self):
        text = format_step_anatomy_averaged([], 0)
        assert "no step anatomy data" in text


@pytest.fixture
def driver_api_db():
    """Create a DB where physics kernels are dispatched via CUDA Driver API.

    Layout for step 0 (env.step at 0-10ms):
      sim.step      [1ms - 4ms]  -> 2 kernels via DRIVER API (corr 101, 102)
      reward.compute [5ms - 7ms] -> 1 kernel via RUNTIME API (corr 3)
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute("INSERT INTO StringIds VALUES (1, 'kernel_physics')")
    c.execute("INSERT INTO StringIds VALUES (2, 'kernel_physics2')")
    c.execute("INSERT INTO StringIds VALUES (3, 'kernel_reward')")

    c.execute("""
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

    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (0, 10000000, 59, 'env.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (1000000, 4000000, 59, 'sim.step')")
    c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (5000000, 7000000, 59, 'reward.compute')")

    c.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
            start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER
        )
    """)

    c.execute("""
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

    # Physics kernels dispatched via DRIVER API (not Runtime)
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER (start, end, correlationId) VALUES (1500000, 1600000, 101)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (2000000, 2100000, 1, 101)")

    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER (start, end, correlationId) VALUES (2500000, 2600000, 102)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (3000000, 3200000, 2, 102)")

    # Reward kernel via RUNTIME API
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, correlationId) VALUES (5500000, 5600000, 3)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (6000000, 6150000, 3, 3)")

    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


class TestDriverApiAnatomy:
    """Tests for anatomy with CUDA Driver API kernels."""

    def test_driver_api_kernels_attributed(self, driver_api_db):
        """Physics kernels dispatched via Driver API are attributed to sim.step."""
        anatomy, _ = query_step_anatomy(driver_api_db, step_index=0)
        kernel_entries = [e for e in anatomy if e["kernel_name"]]
        assert len(kernel_entries) == 3

        sim_kernels = [e for e in kernel_entries if e["nvtx_range"] == "sim.step"]
        assert len(sim_kernels) == 2
        assert {e["kernel_name"] for e in sim_kernels} == {"kernel_physics", "kernel_physics2"}

    def test_mixed_api_paths(self, driver_api_db):
        """Both Driver API and Runtime API kernels appear in correct order."""
        anatomy, _ = query_step_anatomy(driver_api_db, step_index=0)
        kernel_entries = [e for e in anatomy if e["kernel_name"]]
        ranges = [e["nvtx_range"] for e in kernel_entries]
        assert ranges == ["sim.step", "sim.step", "reward.compute"]

    def test_driver_only_db(self, tmp_path):
        """Anatomy works with only DRIVER table (no RUNTIME table)."""
        db_path = str(tmp_path / "driver_only.sqlite")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        c.execute("INSERT INTO StringIds VALUES (1, 'kernel_a')")
        c.execute("""CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER, text TEXT,
            rangeId INTEGER, category INTEGER, color INTEGER,
            globalTid INTEGER, endGlobalTid INTEGER,
            textId INTEGER, domainId INTEGER,
            uint64Value INTEGER, int64Value INTEGER, doubleValue REAL,
            uint32Value INTEGER, int32Value INTEGER, floatValue REAL,
            jsonTextId INTEGER, jsonText TEXT, binaryData TEXT)""")
        c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (0, 10000000, 59, 'env.step')")
        c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (1000000, 4000000, 59, 'sim.step')")
        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
            start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER, nameId INTEGER)""")
        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, shortName INTEGER, correlationId INTEGER,
            deviceId INTEGER, contextId INTEGER, streamId INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            registersPerThread INTEGER, staticSharedMemory INTEGER,
            dynamicSharedMemory INTEGER, localMemoryPerThread INTEGER,
            localMemoryTotal INTEGER)""")
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER (start, end, correlationId) VALUES (1500000, 1600000, 1)")
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId) VALUES (2000000, 2100000, 1, 1)")
        conn.commit()
        conn.close()

        anatomy, _ = query_step_anatomy(db_path, step_index=0)
        kernel_entries = [e for e in anatomy if e["kernel_name"]]
        assert len(kernel_entries) == 1
        assert kernel_entries[0]["nvtx_range"] == "sim.step"
        os.unlink(db_path)
