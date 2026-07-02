# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the reusable RSL-RL training callback and strict-load boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train_rsl_rl.py"
CONTRACT = (
    ROOT
    / "scripts"
    / "reinforcement_learning"
    / "forward_backward"
    / "phase3"
    / "fixtures"
    / "motion_training_smoke_contract_v2.json"
)


def _module():
    script_dir = ENTRYPOINT.parent.parent
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("phase3f_train_rsl_rl", ENTRYPOINT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


def test_checkpoint_validation_always_uses_strict_cpu_map_load() -> None:
    """Validation must avoid a second full CUDA checkpoint allocation."""

    class Runner:
        calls: list[tuple] = []

        def load(self, path: str, *, strict: bool, map_location: str, mmap: bool):
            self.calls.append((path, strict, map_location, mmap))

    runner = Runner()
    checkpoint = Path("/tmp/model.pt")

    _module()._validate_checkpoint(runner, checkpoint)

    assert runner.calls == [(str(checkpoint), True, "cpu", True)]


def test_training_callback_config_snapshot_precedes_constructor_normalization() -> None:
    """Environment construction must not rewrite the callback's declared semantic config."""
    module = _module()
    env_cfg = {"scene": {"robot": {"prim_path": "{ENV_REGEX_NS}/Robot"}}}

    configured_env_cfg = module._snapshot_config_for_callback(env_cfg)
    env_cfg["scene"]["robot"]["prim_path"] = "/World/envs/env_.*/Robot"

    assert configured_env_cfg["scene"]["robot"]["prim_path"] == "{ENV_REGEX_NS}/Robot"
    assert configured_env_cfg is not env_cfg


def test_training_callback_receives_only_declared_stages() -> None:
    """The generic CLI boundary should reject misspelled lifecycle stages."""
    module = _module()
    calls: list[str] = []

    def record_stage(**values: object) -> None:
        calls.append(values["stage"])

    module._invoke_training_callback(record_stage, "prepare", marker=0)

    module._invoke_training_callback(record_stage, "launch", marker=1)
    module._invoke_training_callback(record_stage, "complete", marker=2)
    module._invoke_training_callback(record_stage, "validate", marker=3)

    assert calls == ["prepare", "launch", "complete", "validate"]
    with pytest.raises(ValueError, match="callback stage"):
        module._invoke_training_callback(record_stage, "misspelled", marker=4)


def test_phase3f_contract_callbacks_resolve_from_the_train_import_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every frozen callback path must resolve from ``scripts/reinforcement_learning``."""
    from isaaclab.utils.string import string_to_callable

    monkeypatch.syspath_prepend(str(ENTRYPOINT.parent.parent))
    expected = "forward_backward.phase3.motion_training_receipt:training_callback"
    contract = json.loads(CONTRACT.read_text())
    for profile in contract["profiles"].values():
        for command_name in ("command", "validation_command_template"):
            command = profile[command_name]
            callback_path = command[command.index("--training_callback") + 1]

            assert callback_path == expected
            callback = string_to_callable(callback_path)

            assert callback.__name__ == "training_callback"
