# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Invariants asserted against a *built* container image.

The image under test is named by ``IMAGE_TAG``; the tests skip when it is unset so a plain
``pytest docker/test`` stays green on a machine with no image. Run one explicitly with::

    IMAGE_TAG=isaac-lab-base:latest pytest docker/test/test_image_invariants.py
"""

from __future__ import annotations

import os
import subprocess

import pytest

IMAGE_TAG = os.environ.get("IMAGE_TAG", "")


def _in_image(script: str) -> str:
    """Run ``script`` with bash inside the image under test and return its stdout."""
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", IMAGE_TAG, "-lc", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture(autouse=True)
def _require_image():
    if not IMAGE_TAG:
        pytest.skip("IMAGE_TAG is unset; no built image to assert against")


def test_no_prebundled_package_lost_its_entry_point():
    """A dangling ``__init__.py`` in a prebundle stops Isaac Sim extensions loading.

    Isaac Sim shares prebundled packages between extensions as per-file symlinks, so deleting
    or replacing one strands every symlink into it. #6329 added this invariant to the pip
    install path after nvbugs 6343978, where it cost 438 error lines and 14 failed extensions.
    The images now install with ``uv sync``, which never calls that code, so assert it here.

    Only ``__init__.py`` is fatal: the shipped image already carries dangling submodules and
    ``.pyi`` stubs that no extension imports - 41 of them, against develop's 48 - mostly
    generated protobuf stubs inside an Omniverse extension's own prebundle.
    """
    broken = _in_image('find / -path "*pip_prebundle*" -xtype l -name "__init__.py" 2>/dev/null || true').strip()
    assert not broken, "prebundled packages lost their entry point:\n" + broken


def test_wandb_logger_is_installed():
    """Training images support the W&B logger selected by cluster submissions."""
    assert _in_image(
        'cd /workspace/isaaclab && uv run --no-sync python -c "import wandb; print(wandb.__version__)"'
    ).strip()


def test_native_newton_bridge_matches_its_installed_source():
    """Native images ship the matching L40/Blackwell bridge without a host compiler path."""
    _in_image("""
"${VIRTUAL_ENV}/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
try:
    package = importlib.metadata.distribution("newton")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(0)
source = Path(package.locate_file("newton/_src/utils/cuda_graph.cu"))
if not source.is_file():
    raise SystemExit(0)
library = Path(os.environ["NEWTON_CUDA_GRAPH_LIBRARY"])
manifest = json.loads(library.with_suffix(".json").read_text())
assert manifest["architectures"] == [89, 120]
assert manifest["cuda_toolkit"] == "12.9.1"
assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
assert manifest["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
assert library.is_relative_to("/opt/newton-cuda-graph")
PY""")


def test_bundled_keyboard_assets_compose_without_remote_client():
    """The runtime user can read the complete USD trees without contacting Omniverse."""
    _in_image("""cd /workspace/isaaclab && uv run --no-sync python - <<'PY'
from unittest.mock import patch
from pxr import Usd
from isaaclab.utils.assets import retrieve_file_path
with patch("isaaclab.utils.assets._get_omni_client", side_effect=AssertionError("Unexpected asset client")):
    for path in ("/opt/isaaclab-assets/ground/default_ground_plane.usda",
                 "/opt/isaaclab-assets/so101/so101_new_calib.usda"):
        stage = Usd.Stage.Open(retrieve_file_path(path))
        assert stage and not stage.GetCompositionErrors(), path
PY""")
