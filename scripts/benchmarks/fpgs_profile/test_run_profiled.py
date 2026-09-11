# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check launch instrumentation without importing or starting the simulator."""

import ast
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestGraphInstrumentation(unittest.TestCase):
    def test_scope_identity_arguments_and_exception(self):
        """Preserve launch semantics and distinguish physics from auxiliary graphs."""
        path = Path(__file__).with_name("run_profiled.py")
        # The profiling entry point imports simulation dependencies at module scope.
        # Load just its instrumentation function against observable fake launch APIs.
        source = ast.parse(path.read_text())
        function = next(
            node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_instrument"
        )
        physics, auxiliary = object(), object()
        manager = SimpleNamespace(_graph=physics)
        labels, calls = [], []

        def launch(graph, *args, **kwargs):
            calls.append((graph, args, kwargs))
            if kwargs.get("fail"):
                raise RuntimeError("expected")
            return "unchanged return"

        warp = SimpleNamespace(capture_launch=launch)
        counters = defaultdict(int)
        globals_ = {
            "_wrap": lambda *args: None,
            "wp": warp,
            "_NVTX": True,
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    nvtx=SimpleNamespace(
                        range_push=lambda label: labels.append(label), range_pop=lambda: labels.append("pop")
                    )
                )
            ),
            "HOST": defaultdict(float),
            "COUNTS": counters,
            "time": time,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), globals_)
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene=None, sim=None))
        with patch.dict(sys.modules, {"isaaclab_newton.physics": SimpleNamespace(NewtonManager=manager)}):
            globals_["_instrument"](env)
        self.assertEqual(warp.capture_launch(physics, 7, stream="stream"), "unchanged return")
        self.assertEqual(warp.capture_launch(auxiliary), "unchanged return")
        manager._graph = auxiliary
        with self.assertRaisesRegex(RuntimeError, "expected"):
            warp.capture_launch(auxiliary, fail=True)
        self.assertEqual(labels, ["physics_graph", "pop", "auxiliary_graph", "pop", "physics_graph", "pop"])
        self.assertEqual(dict(counters), {"physics_graph": 2, "auxiliary_graph": 1})
        self.assertEqual(calls[0], (physics, (7,), {"stream": "stream"}))


if __name__ == "__main__":
    unittest.main()
