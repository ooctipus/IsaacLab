# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import subprocess
from pathlib import Path


def _write_python_probe(path: Path, owner: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "{owner}" >> "$ISAACLAB_INTERPRETER_PROBE"\n')
    path.chmod(0o755)


def test_shell_wrapper_prefers_repository_virtual_environment(tmp_path):
    """Ensure an inactive repository ``.venv`` wins over bundled Kit Python."""
    repository_root = Path(__file__).parents[4]
    wrapper = tmp_path / "isaaclab.sh"
    shutil.copy2(repository_root / "isaaclab.sh", wrapper)
    _write_python_probe(tmp_path / ".venv" / "bin" / "python", "environment")
    _write_python_probe(tmp_path / "_isaac_sim" / "python.sh", "kit")
    probe = tmp_path / "selected_interpreter.txt"
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    environment["ISAACLAB_INTERPRETER_PROBE"] = str(probe)

    result = subprocess.run(
        ["bash", str(wrapper), "-p", "-c", "pass"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert probe.read_text().splitlines() == ["environment", "environment"]
