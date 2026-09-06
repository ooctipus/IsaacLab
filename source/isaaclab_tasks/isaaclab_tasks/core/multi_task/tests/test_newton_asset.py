# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for simulator-free Newton asset resolution."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from pxr import Usd, UsdUtils

from isaaclab_tasks.core.multi_task.kinematics import newton_asset as newton_asset_module
from isaaclab_tasks.core.multi_task.kinematics import resolve_newton_asset_path


class _Response(io.BytesIO):
    """In-memory HTTP response used by the recursive-cache tests."""

    def __init__(self, url: str, content: bytes) -> None:
        super().__init__(content)
        self._url = url

    def geturl(self) -> str:
        """Return the request URL as the final response URL."""
        return self._url


def test_local_asset_path_passes_through_without_creating_cache(tmp_path: Path) -> None:
    """An existing local asset must be returned directly without cache writes."""
    asset = tmp_path / "robot.usda"
    asset.write_text("#usda 1.0\n")
    cache = tmp_path / "cache"

    assert resolve_newton_asset_path(asset, cache_dir=cache) == str(asset)
    assert not cache.exists()


def test_omniverse_asset_path_is_rejected_without_importing_omni(tmp_path: Path) -> None:
    """Omniverse URLs must fail at the simulator-free public boundary."""
    with pytest.raises(ValueError, match="omniverse:// URLs require Isaac Sim"):
        resolve_newton_asset_path("omniverse://localhost/NVIDIA/robot.usd", cache_dir=tmp_path)


def test_http_usd_and_recursive_dependencies_are_cached_and_rewritten(tmp_path: Path, monkeypatch) -> None:
    """HTTP layers and non-layer assets must form a self-contained local cache."""
    root_url = "https://assets.example.test/robots/root.usda"
    contents = {
        root_url: b"""#usda 1.0
(
    subLayers = [@layers/child.usda@]
)
def Xform "Root" (
    references = @geometry/reference.usda@</Reference>
)
{
    custom asset texture = @textures/albedo.bin@
}
""",
        "https://assets.example.test/robots/layers/child.usda": b"""#usda 1.0
def Xform "Child" (
    payload = @../geometry/payload.usda@</Payload>
)
{
}
""",
        "https://assets.example.test/robots/geometry/reference.usda": (b'#usda 1.0\ndef Xform "Reference" {}\n'),
        "https://assets.example.test/robots/geometry/payload.usda": b'#usda 1.0\ndef Xform "Payload" {}\n',
        "https://assets.example.test/robots/textures/albedo.bin": b"texture-bytes",
    }
    requests: list[str] = []

    def open_url(request, timeout: float):
        del timeout
        url = request.full_url
        requests.append(url)
        try:
            return _Response(url, contents[url])
        except KeyError as exc:
            raise URLError(f"missing fixture URL: {url}") from exc

    monkeypatch.setattr(newton_asset_module, "urlopen", open_url)
    cache = tmp_path / "cache"
    resolved = Path(resolve_newton_asset_path(root_url, cache_dir=cache))

    assert resolved.is_file()
    assert set(requests) == set(contents)
    assert Usd.Stage.Open(str(resolved)) is not None
    dependencies = tuple(path for group in UsdUtils.ExtractExternalReferences(str(resolved)) for path in group)
    assert len(dependencies) == 3
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in dependencies)

    request_count = len(requests)
    assert resolve_newton_asset_path(root_url, cache_dir=cache) == str(resolved)
    assert len(requests) == request_count


def test_missing_optional_presentation_asset_does_not_reject_valid_usd(tmp_path: Path, monkeypatch) -> None:
    """A missing material file must not hide otherwise valid robot geometry."""
    root_url = "https://assets.example.test/robot.usda"
    root = b"""#usda 1.0
def Xform "Robot"
{
    custom asset material = @Props/OmniPBR.mdl@
}
"""

    def open_url(request, timeout: float):
        del timeout
        if request.full_url == root_url:
            return _Response(root_url, root)
        raise HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(newton_asset_module, "urlopen", open_url)
    with pytest.warns(RuntimeWarning, match="Optional Newton presentation asset was not found"):
        resolved = resolve_newton_asset_path(root_url, cache_dir=tmp_path)

    assert Usd.Stage.Open(resolved) is not None


def test_missing_recursive_usd_layer_is_fatal(tmp_path: Path, monkeypatch) -> None:
    """A missing layer must fail rather than return an incomplete kinematic model."""
    root_url = "https://assets.example.test/robot.usda"
    root = b"""#usda 1.0
(
    subLayers = [@missing.usda@]
)
"""

    def open_url(request, timeout: float):
        del timeout
        if request.full_url == root_url:
            return _Response(root_url, root)
        raise HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(newton_asset_module, "urlopen", open_url)
    with pytest.raises(FileNotFoundError, match="missing.usda"):
        resolve_newton_asset_path(root_url, cache_dir=tmp_path)


def test_http_download_error_names_the_failed_asset(tmp_path: Path, monkeypatch) -> None:
    """Network failures must identify the unresolved Newton asset URL."""
    url = "https://assets.example.test/missing.usd"

    def fail_urlopen(request, timeout: float):
        del request, timeout
        raise URLError("offline")

    monkeypatch.setattr(newton_asset_module, "urlopen", fail_urlopen)
    with pytest.raises(RuntimeError, match="Unable to download Newton asset.*missing.usd"):
        resolve_newton_asset_path(url, cache_dir=tmp_path)
