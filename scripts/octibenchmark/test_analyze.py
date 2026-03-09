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
    query_gpu_kernels,
    query_nvtx_ranges,
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
