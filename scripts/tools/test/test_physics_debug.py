# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified strict physics-debug command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import isaaclab_newton.physics.debug_replay as debug_replay
import numpy as np
import pytest
from isaaclab_newton.physics._debug_archive import write_archive

from scripts.tools import physics_debug

_CREATED_AT = "2026-06-27T12:34:56+00:00"


@pytest.fixture(autouse=True)
def _isolated_replay_registry(monkeypatch):
    """Keep replay adapter registrations local to each test."""
    monkeypatch.setattr(debug_replay, "_REPLAY_ADAPTERS", {})


def _capability(
    capability_id: str = "state_replay",
    *,
    stage: str = "state",
    status: str = "complete",
    provider: str | None = "fake.state",
    fields: list[str] | None = None,
    adapter: str | None = "fake.adapter",
    reason: str | None = None,
) -> dict[str, dict[str, object]]:
    return {
        capability_id: {
            "stage": stage,
            "status": status,
            "provider": provider,
            "fields": ["state"] if fields is None else fields,
            "adapter": adapter,
            "reason": reason,
        }
    }


def _write(
    path: Path,
    values: np.ndarray | None = None,
    *,
    status: str = "complete",
    capabilities: dict[str, dict[str, object]] | None = None,
) -> None:
    arrays = {"state": np.arange(4, dtype=np.float32) if values is None else values}
    metadata = {} if capabilities is None else {"capabilities": capabilities}
    write_archive(
        path,
        arrays,
        status=status,
        required_keys={"state"},
        metadata=metadata,
        dependency_names=(),
        created_at=_CREATED_AT,
    )


def _register_success_adapter(callback=None) -> None:
    if callback is None:

        def callback(_request):
            return {"replayed": True}

    debug_replay.register_replay_adapter(
        debug_replay.ReplayAdapter(
            adapter_id="fake.adapter",
            stages=frozenset({"state"}),
            providers=frozenset({"fake.state"}),
            required_fields=("state",),
            callback=callback,
        )
    )


def test_validate_accepts_complete_and_requires_explicit_partial_opt_in(tmp_path, capsys):
    """Validation defaults to complete-only and exposes an explicit partial opt-in."""
    complete = tmp_path / "complete.npz"
    partial = tmp_path / "partial.npz"
    _write(complete)
    _write(partial, status="partial")

    assert physics_debug.main(["validate", str(complete), "--json"]) == 0
    complete_output = json.loads(capsys.readouterr().out)
    assert complete_output["ok"] is True
    assert complete_output["status"] == "complete"

    assert physics_debug.main(["validate", str(partial), "--json"]) == 2
    partial_error = json.loads(capsys.readouterr().err)
    assert partial_error["ok"] is False
    assert "status 'partial' is not accepted" in partial_error["error"]

    assert physics_debug.main(["validate", str(partial), "--allowed_status", "partial", "--json"]) == 0
    partial_output = json.loads(capsys.readouterr().out)
    assert partial_output["status"] == "partial"


def test_inspect_json_reports_inventory_and_capabilities(tmp_path, capsys):
    """JSON inspection exposes stable archive, array, and replay-capability metadata."""
    path = tmp_path / "incident.npz"
    _write(path, capabilities=_capability())

    assert physics_debug.main(["inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["archive"] == str(path)
    assert payload["status"] == "complete"
    assert payload["array_count"] == 1
    assert payload["arrays"][0]["key"] == "state"
    assert payload["arrays"][0]["shape"] == [4]
    assert payload["capabilities"] == [
        {
            "adapter": "fake.adapter",
            "capability_id": "state_replay",
            "fields": ["state"],
            "provider": "fake.state",
            "reason": None,
            "stage": "state",
            "status": "complete",
        }
    ]


def test_diff_returns_zero_for_exact_match_and_one_for_value_mismatch(tmp_path, capsys):
    """Exact diff has stable zero/mismatch-one return semantics."""
    left = tmp_path / "left.npz"
    same = tmp_path / "same.npz"
    changed = tmp_path / "changed.npz"
    _write(left)
    _write(same)
    _write(changed, values=np.array([9, 1, 2, 3], dtype=np.float32))

    assert physics_debug.main(["diff", str(left), str(same), "--json"]) == 0
    matching = json.loads(capsys.readouterr().out)
    assert matching["match"] is True
    assert matching["mismatches"] == []

    assert physics_debug.main(["diff", str(left), str(changed), "--json"]) == 1
    different = json.loads(capsys.readouterr().out)
    assert different["match"] is False
    assert [entry["path"] for entry in different["mismatches"]] == ["arrays.state.sha256"]


def test_replay_rejects_archive_without_declared_capability(tmp_path, capsys):
    """Replay never infers a capability from legacy or coincidental array keys."""
    path = tmp_path / "no_capability.npz"
    _write(path)

    assert physics_debug.main(["replay", str(path), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "no declared capability" in error["error"]
    assert "No legacy-key inference is permitted" in error["error"]


def test_replay_rejects_missing_execution_adapter_without_implicit_import(tmp_path, capsys, monkeypatch):
    """An archive-controlled adapter name is never imported or executed implicitly."""
    path = tmp_path / "missing_adapter.npz"
    _write(path, capabilities=_capability(adapter="not.registered"))
    imported: list[str] = []
    monkeypatch.setattr(physics_debug.importlib, "import_module", lambda name: imported.append(name))

    assert physics_debug.main(["replay", str(path), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "has no registered execution adapter" in error["error"]
    assert imported == []


def test_replay_rejects_non_complete_capability(tmp_path, capsys):
    """A partial capability cannot be executed even from a complete archive."""
    path = tmp_path / "partial_capability.npz"
    _write(path, capabilities=_capability(status="partial", reason="solver inputs missing"))
    _register_success_adapter()

    assert physics_debug.main(["replay", str(path), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "status is 'partial', not 'complete'" in error["error"]
    assert "solver inputs missing" in error["error"]


def test_replay_rejects_non_complete_archive(tmp_path, capsys):
    """Replay defaults to rejecting an archive whose overall capture is partial."""
    path = tmp_path / "partial_archive.npz"
    _write(path, status="partial", capabilities=_capability())
    _register_success_adapter()

    assert physics_debug.main(["replay", str(path), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "status 'partial' is not accepted" in error["error"]


def test_replay_accepts_partial_archive_only_with_explicit_status_opt_in(tmp_path, capsys):
    """An opted-in partial archive may execute a capability still declared complete."""
    path = tmp_path / "partial_archive.npz"
    _write(path, status="partial", capabilities=_capability())
    _register_success_adapter()

    assert physics_debug.main(["replay", str(path), "--allowed_status", "partial", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capability"] == "state_replay"
    assert payload["result"] == {"replayed": True}


def test_partial_archive_opt_in_does_not_override_capability_status(tmp_path, capsys):
    """Archive status opt-in never weakens the selected capability contract."""
    path = tmp_path / "partial_capability_archive.npz"
    _write(
        path,
        status="partial",
        capabilities=_capability(status="partial", reason="state fields missing"),
    )
    _register_success_adapter()

    assert physics_debug.main(["replay", str(path), "--allowed_status", "partial", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "metadata.capabilities.state_replay.status is" in error["error"]
    assert "not" in error["error"]
    assert "state fields missing" in error["error"]


def test_replay_rejects_ambiguous_executable_capabilities(tmp_path, capsys):
    """Auto replay refuses to choose between multiple executable capabilities."""
    path = tmp_path / "ambiguous.npz"
    capabilities = _capability("first")
    capabilities.update(_capability("second"))
    _write(path, capabilities=capabilities)
    _register_success_adapter()

    assert physics_debug.main(["replay", str(path), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "ambiguous" in error["error"]
    assert "first via fake.adapter" in error["error"]
    assert "second via fake.adapter" in error["error"]

    assert physics_debug.main(["replay", str(path), "--capability", "second", "--json"]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected["capability"] == "second"


def test_replay_executes_registered_adapter_and_emits_json(tmp_path, capsys):
    """A complete declared capability executes only its trusted registered adapter."""
    path = tmp_path / "replay.npz"
    _write(path, capabilities=_capability())
    requests: list[debug_replay.ReplayRequest] = []

    def replay(request: debug_replay.ReplayRequest):
        requests.append(request)
        return {"sum": float(request.arrays["state"].sum())}

    _register_success_adapter(replay)

    assert physics_debug.main(["replay", str(path), "--stage", "state", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "adapter": "fake.adapter",
        "archive": str(path),
        "capability": "state_replay",
        "ok": True,
        "provider": "fake.state",
        "result": {"sum": 6.0},
        "stage": "state",
    }
    assert len(requests) == 1
    assert requests[0].archive_path == path
    assert requests[0].capability.capability_id == "state_replay"


def test_replay_imports_explicit_adapter_module_before_selection(tmp_path, capsys, monkeypatch):
    """An explicit trusted module can register the adapter used for replay."""
    path = tmp_path / "replay.npz"
    _write(path, capabilities=_capability())
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        _register_success_adapter()
        return object()

    monkeypatch.setattr(physics_debug.importlib, "import_module", import_module)

    assert physics_debug.main(["replay", str(path), "--adapter_module", "trusted.physics_adapters", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"] == "fake.adapter"
    assert imported == ["trusted.physics_adapters"]


def test_replay_rejects_duplicate_adapter_modules_before_import(tmp_path, capsys, monkeypatch):
    """Duplicate trusted module names fail before any import-time side effects."""
    path = tmp_path / "replay.npz"
    _write(path, capabilities=_capability())
    imported: list[str] = []
    monkeypatch.setattr(physics_debug.importlib, "import_module", lambda name: imported.append(name))

    assert (
        physics_debug.main(
            [
                "replay",
                str(path),
                "--adapter_module",
                "trusted.physics_adapters",
                "--adapter_module",
                "trusted.physics_adapters",
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert "--adapter_module values must be unique" in error["error"]
    assert "trusted.physics_adapters" in error["error"]
    assert imported == []


def test_replay_reports_explicit_adapter_module_import_failure(tmp_path, capsys, monkeypatch):
    """Import failures name the user-supplied trusted module and original error."""
    path = tmp_path / "replay.npz"
    _write(path, capabilities=_capability())

    def import_module(name: str):
        raise ImportError("adapter dependency unavailable")

    monkeypatch.setattr(physics_debug.importlib, "import_module", import_module)

    assert physics_debug.main(["replay", str(path), "--adapter_module", "trusted.broken_adapters", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "Failed to import trusted replay adapter module 'trusted.broken_adapters'" in error["error"]
    assert "adapter dependency unavailable" in error["error"]


def test_register_replay_adapter_rejects_non_declaration():
    """The public registry rejects lookalike objects before reading attributes."""
    with pytest.raises(TypeError, match="ReplayAdapter instances"):
        debug_replay.register_replay_adapter(object())


def test_diff_compares_dependency_and_runtime_provenance():
    """Exact archive comparison includes dependency and runtime provenance."""
    arrays: dict[str, np.ndarray] = {}
    left_manifest = {
        "format_name": "physics_debug",
        "format_version": 1,
        "status": "complete",
        "required_keys": [],
        "metadata": {},
        "dependencies": {"newton": {"status": "present", "version": "1"}},
        "runtime": {"python": "3.12"},
        "arrays": {},
    }
    right_manifest = {
        **left_manifest,
        "dependencies": {"newton": {"status": "present", "version": "2"}},
        "runtime": {"python": "3.13"},
    }

    mismatches = physics_debug._compare_archives(arrays, left_manifest, arrays, right_manifest)

    assert [entry["path"] for entry in mismatches] == ["manifest.dependencies", "manifest.runtime"]


def test_replay_rejects_non_json_callback_result_in_human_mode(tmp_path, capsys):
    """Adapter results are strict JSON even when the CLI prints human output."""
    path = tmp_path / "replay.npz"
    _write(path, capabilities=_capability())
    _register_success_adapter(lambda request: {"nonfinite": float("nan")})

    assert physics_debug.main(["replay", str(path)]) == 2

    error = capsys.readouterr().err
    assert "not strict JSON-compatible" in error
    assert "Out of range float values" in error


def test_script_main_uses_shared_registry_from_real_adapter_module(tmp_path):
    """A subprocess script and a real external adapter module share one registry."""
    archive_path = tmp_path / "replay.npz"
    _write(archive_path, capabilities=_capability())
    module_name = "trusted_physics_debug_adapter"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """from isaaclab_newton.physics.debug_replay import ReplayAdapter, ReplayRequest, register_replay_adapter

def replay(request: ReplayRequest):
    return {"sum": float(request.arrays["state"].sum())}

register_replay_adapter(
    ReplayAdapter(
        adapter_id="fake.adapter",
        stages=frozenset({"state"}),
        providers=frozenset({"fake.state"}),
        required_fields=("state",),
        callback=replay,
    )
)
""",
        encoding="utf-8",
    )
    script_path = Path(physics_debug.__file__)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(tmp_path), env.get("PYTHONPATH")) if value)

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "replay",
            str(archive_path),
            "--adapter_module",
            module_name,
            "--json",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["adapter"] == "fake.adapter"
    assert payload["result"] == {"sum": 6.0}
