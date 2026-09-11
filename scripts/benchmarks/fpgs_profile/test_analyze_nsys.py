# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable CPU tests for strict physics/auxiliary CUDA-graph accounting."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from analyze_nsys import analyze_graph_trace, validate_graph_correlations


class GraphScopeTests(unittest.TestCase):
    """Build small synthetic Nsight exports without requiring Nsight or a GPU."""

    def setUp(self):
        """Give each test an isolated, disposable database and metadata file."""
        temporary = tempfile.TemporaryDirectory(prefix="fpgs-graph-scope-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.database = self.directory / "capture.sqlite"
        self.metadata = self.directory / "capture.json"

    def fixture(self, auxiliary=True, nodes=False):
        """Create two steps, each with two physics graphs and optionally one sensor graph."""
        pid = 7 << 24
        tid = pid + 1
        with sqlite3.connect(self.database) as connection:
            connection.executescript("""
                create table StringIds (id integer, value text);
                insert into StringIds values (1,'cudaGraphLaunch_v10000');
                create table CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
                    start integer, end integer, deviceId integer, contextId integer,
                    streamId integer, correlationId integer, globalPid integer,
                    graphId integer, graphExecId integer);
                create table CUPTI_ACTIVITY_KIND_RUNTIME (
                    start integer, end integer, correlationId integer, globalTid integer,
                    nameId integer, returnValue integer);
                create table NVTX_EVENTS (
                    start integer, end integer, globalTid integer,
                    text text, textId integer, eventType integer);
                create table CUPTI_ACTIVITY_KIND_KERNEL (
                    globalPid integer, deviceId integer, correlationId integer,
                    graphId integer, graphNodeId integer);
            """)
            for step in range(2):
                base = step * 10000
                connection.execute(
                    "insert into NVTX_EVENTS values (?,?,?,?,null,59)",
                    (base, base + 9000, tid, f"env_step:{step}"),
                )
                for index in range(3 if auxiliary else 2):
                    launch = base + 1000 + index * 2000
                    sensor = index == 2
                    corr = step * 10 + index + 1
                    graph_id, exec_id = (2, 20) if sensor else (1, 10)
                    connection.execute(
                        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,1,0)",
                        (launch, launch + 10, corr, tid),
                    )
                    connection.execute(
                        "insert into CUPTI_ACTIVITY_KIND_GRAPH_TRACE values (?,?,0,1,1,?,?,?,?)",
                        (launch + 20, launch + (70 if sensor else 120), corr, pid, graph_id, exec_id),
                    )
                    for label, before, after in (
                        ("observation" if sensor else "sim.step", 200, 200),
                        ("auxiliary_graph" if sensor else "physics_graph", 100, 50),
                    ):
                        connection.execute(
                            "insert into NVTX_EVENTS values (?,?,?,?,null,59)",
                            (launch - before, launch + after, tid, label),
                        )
                    if nodes:
                        connection.execute(
                            "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,0,?,?,1)",
                            (pid, corr, graph_id),
                        )
        metadata = {
            "task": "synthetic",
            "num_envs": 1,
            "profile_steps": 2,
            "cuda_graph": True,
            "model": {"state_finite": True},
            "model_after": {"state_finite": True},
            "host_calls_per_step": {"physics_graph": 2.0},
        }
        if auxiliary:
            metadata["host_calls_per_step"]["auxiliary_graph"] = 1.0
        self.metadata.write_text(json.dumps(metadata))

    def mutate(self, sql):
        """Change only this test's disposable SQLite export."""
        with sqlite3.connect(self.database) as connection:
            connection.executescript(sql)

    def analyze(self):
        """Run the maintained direct-graph validator."""
        return analyze_graph_trace(self.database, self.metadata)

    def test_physics_only_backward_compatible(self):
        """Older physics-only metadata needs no auxiliary counter."""
        self.fixture(auxiliary=False)
        result = self.analyze()
        self.assertEqual(result["summary"]["graph_span_us_per_step"], 0.2)
        self.assertEqual(result["summary"]["graph_launches_per_step"], 2)
        self.assertEqual(result["summary"]["auxiliary_graph_span_us_per_step"], 0)
        self.assertEqual(result["summary"]["auxiliary_graph_launches_per_step"], 0)
        self.assertEqual(result["graph_identity"][-2:], [1, 10])

    def test_auxiliary_time_is_not_physics(self):
        """Account for every launch while separating sensor time from physics."""
        self.fixture()
        result = self.analyze()
        self.assertEqual(result["summary"]["graph_scope"], "physics")
        self.assertEqual(result["summary"]["graph_span_us_per_step"], 0.2)
        self.assertEqual(result["summary"]["auxiliary_graph_span_us_per_step"], 0.05)
        self.assertEqual(result["graph_membership_audit"]["physics_launches"], 4)
        self.assertEqual(result["graph_membership_audit"]["auxiliary_launches"], 2)
        for step in result["per_step"]:
            self.assertEqual(len(step["graph_launches"]), 2)
            self.assertEqual(len(step["auxiliary_graph_launches"]), 1)

    def test_distinct_auxiliary_identities_allowed(self):
        """Multiple auxiliary graphs do not weaken the single physics identity rule."""
        self.fixture()
        self.mutate("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set graphId=3,graphExecId=30 where correlationId=13;")
        self.assertEqual(len(self.analyze()["auxiliary_graph_identities"]), 2)

    def test_old_sensor_mislabeled_as_physics_rejected(self):
        """Global capture-launch labeling cannot silently include observation work."""
        self.fixture()
        self.mutate("update NVTX_EVENTS set text='physics_graph' where text='auxiliary_graph';")
        with self.assertRaisesRegex(ValueError, "inside sim.step"):
            self.analyze()

    def test_bad_device_or_host_records_rejected(self):
        """Reject missing, unmatched, duplicated, failed, or cross-scope correlations."""
        self.fixture()
        cases = (
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set correlationId=0 where correlationId=1", "Missing"),
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set correlationId=999 where correlationId=1", "uniquely matched"),
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set correlationId=1 where correlationId=2", "uniquely matched"),
            ("update CUPTI_ACTIVITY_KIND_RUNTIME set returnValue=1 where correlationId=1", "Successful host"),
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set deviceId=1 where correlationId=1", "one CUDA"),
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set globalPid=0 where correlationId=1", "one CUDA"),
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set contextId=2 where correlationId=1", "one CUDA"),
            ("delete from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=1", "Successful host"),
            ("update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set graphExecId=11 where correlationId=1", "single fixed physics"),
            (
                "update CUPTI_ACTIVITY_KIND_GRAPH_TRACE set graphId=1,graphExecId=10 where graphId=2",
                "both physics and auxiliary",
            ),
            ("delete from NVTX_EVENTS where text='auxiliary_graph'", "ranges are incomplete"),
            ("update NVTX_EVENTS set text='auxiliary_graph' where text='physics_graph'", "single fixed physics"),
        )
        for sql, error in cases:
            with self.subTest(sql=sql):
                self.mutate(sql)
                try:
                    with self.assertRaisesRegex(ValueError, error):
                        self.analyze()
                finally:
                    self.database.unlink()
                    self.fixture()

    def test_auxiliary_metadata_count_must_match(self):
        """The unprofiled counter must agree with each profiled step."""
        self.fixture()
        metadata = json.loads(self.metadata.read_text())
        metadata["host_calls_per_step"]["auxiliary_graph"] = 2
        self.metadata.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "auxiliary_graph launch count"):
            self.analyze()

    def test_node_physics_only_is_valid(self):
        """Node work budgets retain their physics-only scope."""
        self.fixture(auxiliary=False, nodes=True)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(validate_graph_correlations(connection)["graph_scope"], "physics")

    def test_node_auxiliary_or_mislabeled_graphs_rejected(self):
        """Node analysis must not silently fold a sensor into solver stage budgets."""
        self.fixture(nodes=True)
        with sqlite3.connect(self.database) as connection:
            with self.assertRaisesRegex(ValueError, "use graph mode"):
                validate_graph_correlations(connection)
        self.mutate("update NVTX_EVENTS set text='physics_graph' where text='auxiliary_graph';")
        with sqlite3.connect(self.database) as connection:
            with self.assertRaisesRegex(ValueError, "inside sim.step"):
                validate_graph_correlations(connection)


if __name__ == "__main__":
    unittest.main()
