# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free asset resolution for Newton kinematic models."""

from __future__ import annotations

import hashlib
import os
import posixpath
import shutil
import tempfile
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
from urllib.request import Request, url2pathname, urlopen

_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "asset_cache" / "newton_http"
_HTTP_TIMEOUT_SECONDS = 30.0
_USD_LAYER_SUFFIXES = frozenset((".usd", ".usda", ".usdc"))
_USD_ASSET_SUFFIXES = _USD_LAYER_SUFFIXES | {".usdz"}


def resolve_newton_asset_path(path: str | os.PathLike[str], cache_dir: str | os.PathLike[str] | None = None) -> str:
    """Return a local asset path that Newton/OpenUSD can open without Isaac Sim.

    Existing local files pass through without touching the cache. HTTP(S) assets
    are mirrored into a deterministic cache together with their external USD
    layers and available file-valued assets. Cached USD paths are rewritten to
    local dependency paths. Missing non-layer presentation assets are reported
    and left unresolved; missing USD layers fail the operation.

    Args:
        path: Existing local file, ``file://`` URL, or HTTP(S) asset URL.
        cache_dir: Optional HTTP cache root. Defaults to a shared temporary
            IsaacLab asset cache.

    Returns:
        Existing local path or the cached local path for an HTTP(S) asset.

    Raises:
        FileNotFoundError: If a local file or HTTP asset does not exist.
        RuntimeError: If an HTTP asset cannot be downloaded or a USD layer
            cannot be cached.
        ValueError: If the path uses Omniverse or another unsupported scheme.
    """
    raw_path = os.fspath(path)
    if not raw_path:
        raise ValueError("Newton asset path must be non-empty.")

    local_path = os.path.expanduser(raw_path)
    if os.path.isfile(local_path):
        return local_path

    parsed = urlsplit(raw_path)
    scheme = parsed.scheme.lower()
    if not scheme:
        raise FileNotFoundError(f"Newton asset file does not exist: {raw_path}")
    if scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Remote file URL hosts are unsupported for Newton assets: {raw_path}")
        local_path = url2pathname(unquote(parsed.path))
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Newton asset file does not exist: {local_path}")
        return local_path
    if scheme == "omniverse":
        raise ValueError(
            "omniverse:// URLs require Isaac Sim's omni.client and are unsupported by simulator-free Newton tools."
        )
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported Newton asset URL scheme {scheme!r}: {raw_path}")

    asset_url, _fragment = urldefrag(raw_path)
    root = Path(cache_dir).expanduser() if cache_dir is not None else _DEFAULT_CACHE_DIR
    return str(_cache_http_asset(asset_url, root, {}))


def _cache_http_asset(url: str, cache_dir: Path, active: dict[str, Path]) -> Path:
    """Cache one HTTP asset and its recursive dependencies."""
    cached = active.get(url)
    if cached is not None:
        return cached

    local_path = _http_cache_path(url, cache_dir)
    active[url] = local_path
    complete_path = local_path.with_name(f".{local_path.name}.complete")
    if local_path.is_file() and complete_path.is_file():
        return local_path

    effective_url = _download_http_asset(url, local_path) if not local_path.is_file() else url
    if local_path.suffix.lower() in _USD_LAYER_SUFFIXES:
        dependencies = _extract_external_references(local_path)
        replacements: dict[str, str] = {}
        for dependency in dependencies:
            try:
                replacements[dependency] = _cache_dependency(dependency, effective_url, cache_dir, active)
            except FileNotFoundError:
                if _is_usd_asset(dependency):
                    raise
                warnings.warn(
                    f"Optional Newton presentation asset was not found: {dependency}", RuntimeWarning, stacklevel=2
                )
        _rewrite_asset_paths(local_path, replacements)
    _write_atomic(complete_path, url.encode())
    return local_path


def _http_cache_path(url: str, cache_dir: Path) -> Path:
    """Map one HTTP URL to a traversal-safe deterministic cache path."""
    parsed = urlsplit(url)
    origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    origin_key = hashlib.sha256(origin.encode()).hexdigest()[:20]
    relative = posixpath.normpath(unquote(parsed.path)).lstrip("/")
    if relative in ("", ".") or relative == ".." or relative.startswith("../"):
        raise ValueError(f"HTTP Newton asset URL must name a file: {url}")
    local_path = cache_dir / origin_key / Path(relative)
    if parsed.query:
        query_key = hashlib.sha256(parsed.query.encode()).hexdigest()[:12]
        local_path = local_path.with_name(f"{local_path.stem}-{query_key}{local_path.suffix}")
    return local_path


def _download_http_asset(url: str, local_path: Path) -> str:
    """Download one URL atomically and return its final response URL."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "IsaacLab-NewtonAssetResolver/1"})
    temporary_path: str | None = None
    try:
        with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            with tempfile.NamedTemporaryFile(dir=local_path.parent, delete=False) as temporary:
                temporary_path = temporary.name
                shutil.copyfileobj(response, temporary)
            os.replace(temporary_path, local_path)
            return response.geturl()
    except HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"Newton asset was not found: {url}") from exc
        raise RuntimeError(f"Unable to download Newton asset {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to download Newton asset {url}: {exc.reason}") from exc
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _extract_external_references(local_path: Path) -> tuple[str, ...]:
    """Return stable unique dependencies authored by one cached USD layer."""
    from pxr import Tf, UsdUtils

    try:
        groups = UsdUtils.ExtractExternalReferences(str(local_path))
    except Tf.ErrorException as exc:
        raise RuntimeError(f"Downloaded Newton USD layer cannot be read: {local_path}") from exc
    return tuple(dict.fromkeys(reference for group in groups for reference in group if reference))


def _is_usd_asset(path: str) -> bool:
    """Return whether one authored dependency is a USD layer or package."""
    from pxr import Ar

    outer = Ar.SplitPackageRelativePathOuter(path) if Ar.IsPackageRelativePath(path) else path
    return Path(urlsplit(outer).path).suffix.lower() in _USD_ASSET_SUFFIXES


def _cache_dependency(
    dependency: str,
    parent_url: str,
    cache_dir: Path,
    active: dict[str, Path],
) -> str:
    """Resolve one authored USD dependency to a cached local path."""
    from pxr import Ar

    if Ar.IsPackageRelativePath(dependency):
        outer, inner = Ar.SplitPackageRelativePathInner(dependency)
        return Ar.JoinPackageRelativePath(_cache_dependency(outer, parent_url, cache_dir, active), inner)

    expanded = os.path.expanduser(dependency)
    if os.path.isabs(expanded):
        if not os.path.isfile(expanded):
            raise FileNotFoundError(f"Newton USD dependency does not exist: {dependency}")
        return expanded

    parsed = urlsplit(dependency)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        return resolve_newton_asset_path(dependency, cache_dir=cache_dir)
    if scheme == "omniverse":
        raise ValueError(f"Newton USD dependency {dependency!r} requires Isaac Sim's omni.client and cannot be cached.")
    if scheme and scheme not in ("http", "https"):
        raise ValueError(f"Unsupported Newton USD dependency scheme {scheme!r}: {dependency}")
    dependency_url, _fragment = urldefrag(dependency if scheme else urljoin(parent_url, dependency))
    return str(_cache_http_asset(dependency_url, cache_dir, active))


def _rewrite_asset_paths(local_path: Path, replacements: dict[str, str]) -> None:
    """Atomically rewrite one cached layer to absolute cached dependencies."""
    if not replacements:
        return

    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(str(local_path))
    if layer is None:
        raise RuntimeError(f"Downloaded Newton USD layer cannot be opened: {local_path}")
    UsdUtils.ModifyAssetPaths(layer, lambda asset_path: replacements.get(asset_path, asset_path))
    with tempfile.NamedTemporaryFile(dir=local_path.parent, suffix=local_path.suffix, delete=False) as temporary:
        temporary_path = temporary.name
    try:
        if not layer.Export(temporary_path):
            raise RuntimeError(f"Unable to write cached Newton USD layer: {local_path}")
        os.replace(temporary_path, local_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _write_atomic(path: Path, content: bytes) -> None:
    """Write one small cache marker atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = temporary.name
        temporary.write(content)
    try:
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
