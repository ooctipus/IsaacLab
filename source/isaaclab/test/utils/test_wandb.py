# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from types import SimpleNamespace

import pytest

import isaaclab.utils.wandb as wandb_utils

pytestmark = pytest.mark.unit


class _RemoteFile:
    def __init__(self, name: str):
        self.name = name
        self.download_count = 0

    def download(self, root: str, replace: bool):
        self.download_count += 1
        Path(root, self.name).write_bytes(b"checkpoint")


class _Run:
    def __init__(self, run_id: str, name: str, files: list[_RemoteFile]):
        self.id = run_id
        self.name = name
        self._files = files

    def files(self):
        return self._files


class _Api:
    def __init__(self, *, direct_run: _Run | None = None, named_runs: list[_Run] | None = None):
        self.direct_run = direct_run
        self.named_runs = named_runs or []
        self.run_calls: list[str] = []
        self.runs_calls: list[tuple[str, dict]] = []

    def run(self, path: str):
        self.run_calls.append(path)
        return self.direct_run

    def runs(self, path: str, filters: dict):
        self.runs_calls.append((path, filters))
        return self.named_runs


def _install_api(monkeypatch, api: _Api):
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)


def test_get_model_checkpoint_keeps_run_id_lookup(tmp_path, monkeypatch):
    model = _RemoteFile("model_12.pt")
    api = _Api(direct_run=_Run("abc123", "display-name", [model]))
    _install_api(monkeypatch, api)

    checkpoint = wandb_utils.get_model_checkpoint(
        run_id="abc123", project="factory_manager3", wandb_username="uw-lab", tmp_folder_dir=str(tmp_path)
    )

    assert api.run_calls == ["uw-lab/factory_manager3/abc123"]
    assert api.runs_calls == []
    assert Path(checkpoint) == tmp_path / "factory_manager3" / "abc123" / "model_12.pt"


def test_get_model_checkpoint_resolves_unique_display_name_and_caches_per_node(tmp_path, monkeypatch):
    older = _RemoteFile("model_9.pt")
    latest = _RemoteFile("model_120.pt")
    api = _Api(named_runs=[_Run("internal42", "screwing-rb1-c08-s2-0805a", [latest, older])])
    _install_api(monkeypatch, api)
    kwargs = {
        "run_name": "screwing-rb1-c08-s2-0805a",
        "project": "factory_manager3",
        "wandb_username": "uw-lab",
        "tmp_folder_dir": str(tmp_path),
    }

    first = wandb_utils.get_model_checkpoint(**kwargs)
    second = wandb_utils.get_model_checkpoint(**kwargs)

    assert first == second
    assert Path(first) == tmp_path / "factory_manager3" / "internal42" / "model_120.pt"
    assert api.run_calls == []
    assert api.runs_calls == [("uw-lab/factory_manager3", {"display_name": "screwing-rb1-c08-s2-0805a"})]
    assert older.download_count == 0
    assert latest.download_count == 1


def test_get_model_checkpoint_rejects_ambiguous_display_name(tmp_path, monkeypatch):
    runs = [_Run("first", "duplicate", []), _Run("second", "duplicate", [])]
    _install_api(monkeypatch, _Api(named_runs=runs))

    with pytest.raises(ValueError, match="matched 2 runs"):
        wandb_utils.get_model_checkpoint(
            run_name="duplicate", project="factory_manager3", wandb_username="uw-lab", tmp_folder_dir=str(tmp_path)
        )
