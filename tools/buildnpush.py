# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build and push Isaac Lab Docker images.

This script is the implementation behind ``./buildnpush.sh``. The shell file is
kept as a tiny compatibility wrapper; all build state and command decisions live
here so the image roles stay explicit:

* ``isaac-lab-deps:<hash>`` stores dependency/base rebuilds.
* ``isaac-lab-prepared:<tag>`` stores the latest local prepared image for a tag.
* ``nvcr.io/nvidian/octi-isaac-lab:<tag>`` is the final pushed image tag.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KITLESS_BASE_IMAGE = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
FINAL_IMAGE_REPO = "nvcr.io/nvidian/octi-isaac-lab"
BASE_IMAGE = "isaac-lab-base"
MAX_LAYERS_FOR_DEPS_CACHE = 118
STATE_FILE = Path.home() / ".isaaclab-deps-cache.txt"
MAX_STATE_AGE_DAYS = 30
NGC = Path("/home/zhengyuz/ngc-cli/ngc")

ASSETS_DATA_DIR = Path("source/isaaclab_assets/data")
EXTRA_SYMLINKS = [Path("dep"), Path("dep/rsl_rl")]

# Extras the images are synced with. These mirror the ``ISAACLAB_UV_SYNC_ARGS`` defaults in
# the Dockerfiles and are only used to re-run ``uv sync --check`` against a built image;
# ``test_uv_sync_extras_match_the_dockerfiles`` fails if the two drift apart.
BASE_UV_SYNC_EXTRAS = "--extra all --extra ovrtx --extra ov"
KITLESS_UV_SYNC_EXTRAS = "--extra all --extra ovrtx"
# Emitted by ``uv sync --check`` when it did run and found the environment stale. Its absence on a
# failure means the check itself broke (missing uv, unreadable lock) rather than the image drifting.
UV_OUTDATED_MARKER = "The environment is outdated"


class BuildError(RuntimeError):
    """Raised for user-facing build failures."""


@dataclass(frozen=True)
class BuildArgs:
    """Parsed build options."""

    tag: str
    source: bool = False
    pip: bool = False
    deps: bool = False
    all: bool = False
    skip_push: bool = False
    kitless: bool = False
    kitless_base_image: str = DEFAULT_KITLESS_BASE_IMAGE


@dataclass
class BuildPlan:
    """Concrete build decision."""

    skip_deps: bool
    use_cache: bool
    run_pip_install: bool
    reason: str
    build_base_image: str
    # Force the dependency-sync layer to re-run while keeping the system layers cached.
    bust_deps_cache: bool = False


@dataclass
class BuildContext:
    """Computed image names and dependency hash for one build."""

    args: BuildArgs
    deps_hash: str
    deps_image: str
    prepared_image: str
    final_image: str


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    unset_env: Iterable[str] = (),
    capture: bool = False,
    check: bool = True,
) -> str:
    """Run a subprocess.

    Args:
        cmd: Command and arguments.
        env: Optional environment overrides.
        unset_env: Environment keys to remove before applying overrides.
        capture: Whether to capture stdout.
        check: Whether to raise on a non-zero return code.

    Returns:
        Captured stdout when ``capture`` is true, otherwise an empty string.
    """

    merged_env = os.environ.copy()
    for key in unset_env:
        merged_env.pop(key, None)
    if env:
        merged_env.update(env)
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=merged_env,
            check=False,
            text=True,
            capture_output=True,
        )
        if check and proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise BuildError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{stderr}")
        return proc.stdout
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=False)
    if check and proc.returncode != 0:
        raise BuildError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return ""


def docker(*args: str, capture: bool = False, check: bool = True, env: dict[str, str] | None = None) -> str:
    """Run a Docker command."""

    return run(["docker", *args], capture=capture, check=check, env=env)


def image_exists(image: str) -> bool:
    """Return whether a Docker image exists locally."""

    return (
        subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def image_created(image: str) -> str:
    """Return Docker's creation timestamp for an image."""

    if not image_exists(image):
        return ""
    return docker("image", "inspect", image, "--format", "{{.Created}}", capture=True).strip()


def image_layer_count(image: str) -> int:
    """Return the actual Docker RootFS layer count for an image."""

    if not image_exists(image):
        return 0
    out = docker("image", "inspect", image, "--format", "{{len .RootFS.Layers}}", capture=True, check=False)
    try:
        return int(out.strip())
    except ValueError:
        return 0


def image_within_layer_budget(image: str) -> bool:
    """Return whether an image has enough layer headroom for a cache role."""

    count = image_layer_count(image)
    return 0 < count < MAX_LAYERS_FOR_DEPS_CACHE


def list_images(reference: str | None = None) -> list[str]:
    """List local Docker images as ``repo:tag`` strings."""

    cmd = ["images"]
    if reference is not None:
        cmd += ["--filter", f"reference={reference}"]
    cmd += ["--format", "{{.Repository}}:{{.Tag}}"]
    return [line.strip() for line in docker(*cmd, capture=True).splitlines() if line.strip()]


def newest_image(images: Iterable[str]) -> str | None:
    """Return the newest existing image by Docker creation timestamp."""

    best: str | None = None
    best_created = ""
    for image in images:
        created = image_created(image)
        if created and (best is None or created > best_created):
            best = image
            best_created = created
    return best


def newest_within_budget_deps_image() -> str | None:
    """Return the newest ``isaac-lab-deps:*`` image with layer headroom."""

    return newest_image(img for img in list_images("isaac-lab-deps:*") if image_within_layer_budget(img))


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=value`` environment file."""

    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def hash_file(md5: hashlib._Hash, path: Path, label: str) -> None:
    """Add a file's bash-compatible marker and bytes to a dependency hash."""

    if not path.is_file():
        return
    md5.update(f"### {label}\n".encode())
    md5.update(path.read_bytes())


def compute_deps_hash(kitless: bool) -> str:
    """Compute the hash governing the dependency/base image."""

    md5 = hashlib.md5()
    if kitless:
        md5.update(b"### build-mode: kitless\n")
        build_recipe = Path("docker/Dockerfile.kitless")
    else:
        build_recipe = Path("docker/Dockerfile.base")

    # ``uv.lock`` pins the resolved dependency set (exact git revisions included), so a
    # lock-only bump must invalidate the deps image even when pyproject.toml is unchanged.
    for rel_path in [
        Path("docker/.env.base"),
        build_recipe,
        Path("docker/docker-compose.yaml"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("isaaclab.sh"),
    ]:
        hash_file(md5, REPO_ROOT / rel_path, rel_path.as_posix())

    source = REPO_ROOT / "source"
    source_files = [
        path for path in source.rglob("*") if path.is_file() and path.name in {"pyproject.toml", "extension.toml"}
    ]
    for path in sorted(source_files, key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        hash_file(md5, path, path.relative_to(REPO_ROOT).as_posix())
    return md5.hexdigest()[:12]


@contextmanager
def resolved_symlinks():
    """Temporarily replace selected directory symlinks with copied content."""

    restored: list[tuple[Path, Path]] = []

    def resolve_one(rel_path: Path) -> None:
        path = REPO_ROOT / rel_path
        if not path.is_symlink():
            return
        target = path.resolve()
        if not target.is_dir():
            return
        print(f"   resolving {rel_path} -> {target}")
        path.unlink()
        shutil.copytree(target, path, symlinks=False)
        restored.append((path, target))

    print("Resolving symlinks for Docker build context...")
    assets = REPO_ROOT / ASSETS_DATA_DIR
    if assets.is_dir():
        for item in sorted(assets.iterdir()):
            resolve_one(item.relative_to(REPO_ROOT))
    for item in EXTRA_SYMLINKS:
        resolve_one(item)

    try:
        yield
    finally:
        if restored:
            print("Restoring symlinks...")
        for path, target in reversed(restored):
            print(f"   restoring {path.relative_to(REPO_ROOT)} -> {target}")
            if path.exists() or path.is_symlink():
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            path.symlink_to(target, target_is_directory=True)


def make_context(args: BuildArgs) -> BuildContext:
    """Create immutable build context values."""

    deps_hash = "" if args.source else compute_deps_hash(args.kitless)
    return BuildContext(
        args=args,
        deps_hash=deps_hash,
        deps_image=f"isaac-lab-deps:{deps_hash}" if deps_hash else "",
        prepared_image=f"isaac-lab-prepared:{args.tag}",
        final_image=f"{FINAL_IMAGE_REPO}:{args.tag}",
    )


def prepared_checkpoint_image(ctx: BuildContext) -> str | None:
    """Return the local dependency checkpoint for this tag, with migration fallback."""

    if image_exists(ctx.prepared_image):
        return ctx.prepared_image
    return newest_image([ctx.final_image, BASE_IMAGE])


def resolve_pip_overlay_base(ctx: BuildContext) -> str:
    """Resolve the base image for ``-p`` overlay builds."""

    prepared = prepared_checkpoint_image(ctx)
    if prepared:
        print(f"   using last prepared image for pip overlay: {prepared} ({image_layer_count(prepared)} layers)")
        return prepared

    if image_exists(ctx.deps_image) and image_within_layer_budget(ctx.deps_image):
        print(
            f"   using cached deps image for pip overlay: {ctx.deps_image} ({image_layer_count(ctx.deps_image)} layers)"
        )
        return ctx.deps_image

    existing = newest_within_budget_deps_image()
    if existing:
        print(
            f"   using newest in-budget deps cache for pip overlay: {existing} ({image_layer_count(existing)} layers)"
        )
        return existing

    raise BuildError("No usable base image found for -p/--pip. Run -d/--deps to flatten and rebuild.")


def determine_plan(ctx: BuildContext) -> BuildPlan:
    """Determine the Docker build plan for the requested options."""

    args = ctx.args
    if args.source:
        prepared = prepared_checkpoint_image(ctx)
        if prepared is None:
            raise BuildError("No prepared image found for -s/--source. Run -p/--pip or -d/--deps first.")
        print(f"   using last prepared image for source overlay: {prepared} ({image_layer_count(prepared)} layers)")
        return BuildPlan(True, True, False, "source-only overlay on last prepared image (-s/--source)", prepared)

    if args.all:
        return BuildPlan(False, False, True, "full rebuild (-a/--all)", ctx.deps_image)

    if args.deps:
        return BuildPlan(False, True, True, "deps rebuild (-d/--deps)", ctx.deps_image, bust_deps_cache=True)

    if args.pip:
        return BuildPlan(
            True,
            True,
            True,
            "source + pip overlay on last prepared image (-p/--pip)",
            resolve_pip_overlay_base(ctx),
        )

    if image_exists(ctx.deps_image):
        if image_within_layer_budget(ctx.deps_image):
            print(f"   cached deps image found: {ctx.deps_image} ({image_layer_count(ctx.deps_image)} layers)")
            return BuildPlan(True, True, False, f"deps cached ({ctx.deps_image})", ctx.deps_image)
        print(
            f"   {ctx.deps_image} has {image_layer_count(ctx.deps_image)} layers "
            f"(budget {MAX_LAYERS_FOR_DEPS_CACHE}); rebuilding deps"
        )
        return BuildPlan(False, True, True, "cached deps image exceeds layer budget; rebuilding deps", ctx.deps_image)

    if newest_within_budget_deps_image() is None:
        candidate = newest_image(
            img for img in [ctx.prepared_image, ctx.final_image, BASE_IMAGE] if image_within_layer_budget(img)
        )
        if candidate:
            docker("tag", candidate, ctx.deps_image)
            print(f"   reusing {candidate} as first deps cache: {ctx.deps_image}")
            return BuildPlan(True, True, False, f"migrated existing image ({candidate})", ctx.deps_image)
        print(f"   no cached deps for hash {ctx.deps_hash}")
        return BuildPlan(False, True, True, "first build (no existing images)", ctx.deps_image)

    print("   deps hash changed; full deps rebuild required")
    print(f"   no cached deps for hash {ctx.deps_hash}")
    return BuildPlan(False, True, True, "deps changed, full rebuild required", ctx.deps_image)


def print_build_config(ctx: BuildContext, plan: BuildPlan) -> None:
    """Print the resolved build configuration."""

    print("\nBuild Configuration:")
    print(f"   Tag:           {ctx.args.tag}")
    print(f"   Deps Hash:     {ctx.deps_hash if ctx.deps_hash else 'SKIP (-s)'}")
    print(f"   Strategy:      {plan.reason}")
    print(f"   Pip Install:   {'YES' if (not plan.skip_deps or plan.run_pip_install) else 'SKIP (cached)'}")
    if plan.skip_deps:
        print(f"   Overlay Base:  {plan.build_base_image}")
    print(f"   Docker Cache:  {'YES' if plan.use_cache else 'NO'}")
    print(f"   Dep Re-sync:   {'FORCED' if plan.bust_deps_cache or not plan.use_cache else 'cache-permitting'}")
    print(f"   Push to NGC:   {'SKIP' if ctx.args.skip_push else 'YES'}\n")


def build_full_deps(ctx: BuildContext, plan: BuildPlan, docker_env: dict[str, str]) -> None:
    """Build the full dependency image."""

    print("Building full image with dependencies...")
    env = {"SKIP_PIP_INSTALL": "0"}
    unset_env: list[str] = []
    if not plan.use_cache:
        env["ISAACLAB_NOCACHE"] = "1"
    else:
        unset_env.append("ISAACLAB_NOCACHE")

    # A cached dependency-sync layer would otherwise keep the previously resolved
    # packages even when the lock changed; a fresh token forces that layer to re-run.
    if plan.bust_deps_cache:
        env["DEPS_CACHE_BUST"] = str(int(time.time()))
        print(f"   forcing dependency re-sync (DEPS_CACHE_BUST={env['DEPS_CACHE_BUST']})")
    else:
        unset_env.append("DEPS_CACHE_BUST")

    if ctx.args.kitless:
        cache_flag = [] if plan.use_cache else ["--no-cache"]
        print(f"   kitless build on {ctx.args.kitless_base_image}")
        docker(
            "build",
            *cache_flag,
            "-f",
            "docker/Dockerfile.kitless",
            "--build-arg",
            f"KITLESS_BASE_IMAGE_ARG={ctx.args.kitless_base_image}",
            "--build-arg",
            f"ISAACLAB_PATH_ARG={docker_env['DOCKER_ISAACLAB_PATH']}",
            "--build-arg",
            f"DOCKER_USER_HOME_ARG={docker_env['DOCKER_USER_HOME']}",
            "--build-arg",
            f"DEPS_CACHE_BUST={env.get('DEPS_CACHE_BUST', '0')}",
            "-t",
            BASE_IMAGE,
            ".",
            env={"DOCKER_BUILDKIT": "1", **env},
        )
    else:
        run(["./docker/container.py", "start", "--build"], env=env, unset_env=unset_env)

    if not image_exists(BASE_IMAGE):
        raise BuildError(f"Full deps build did not produce {BASE_IMAGE}; see the Docker build output above.")

    print(f"Caching deps image as {ctx.deps_image}")
    docker("tag", BASE_IMAGE, ctx.deps_image)


def uv_sync_extras(ctx: BuildContext) -> str:
    """Return the ``uv sync`` extras the image for this build is expected to carry."""

    return KITLESS_UV_SYNC_EXTRAS if ctx.args.kitless else BASE_UV_SYNC_EXTRAS


def build_overlay(ctx: BuildContext, plan: BuildPlan, docker_env: dict[str, str]) -> None:
    """Build a source-only or source+pip overlay image."""

    if plan.run_pip_install:
        print("Using prepared image, copying source and installing Python deps...")
    else:
        print("Using prepared image, copying source only...")

    docker(
        "build",
        "--no-cache",
        "-f",
        "docker/Dockerfile.source-only",
        "--build-arg",
        f"DEPS_BASE_IMAGE={plan.build_base_image}",
        "--build-arg",
        f"ISAACLAB_PATH_ARG={docker_env['DOCKER_ISAACLAB_PATH']}",
        "--build-arg",
        f"ISAACSIM_ROOT_PATH_ARG={docker_env['DOCKER_ISAACSIM_ROOT_PATH']}",
        "--build-arg",
        f"RUN_PIP_INSTALL={1 if plan.run_pip_install else 0}",
        "--build-arg",
        f"ISAACLAB_UV_SYNC_ARGS={uv_sync_extras(ctx)}",
        "-t",
        BASE_IMAGE,
        ".",
        env={"DOCKER_BUILDKIT": "1"},
    )

    if plan.run_pip_install:
        layers = image_layer_count(BASE_IMAGE)
        if layers < MAX_LAYERS_FOR_DEPS_CACHE:
            print(
                f"Promoting synced build to hash-matched deps cache: {ctx.deps_image} "
                f"({layers}/{MAX_LAYERS_FOR_DEPS_CACHE} layers)"
            )
            docker("tag", BASE_IMAGE, ctx.deps_image)
        else:
            print(f"Layer count {layers} >= {MAX_LAYERS_FOR_DEPS_CACHE}; skipping deps-cache promotion.")
            print("Run -d/--deps to flatten back to a low-layer dependency image.")


def out_of_sync_packages(check_output: str) -> list[str]:
    """Return the third-party packages ``uv sync --check`` wants to change.

    ``uv`` reports every editable workspace member as needing a reinstall on each check: their
    recorded metadata never matches what a fresh sync would produce, so all of ``source/`` shows
    up even when the environment is exactly right. Those entries always resolve to a local
    ``file://`` path, whereas a genuinely drifted dependency resolves to a registry or git URL
    (Newton, for instance, is pinned to a git revision). Keep only the latter.

    Args:
        check_output: Combined stdout/stderr of ``uv sync --check``.

    Returns:
        The plan lines uv printed for non-workspace packages, in the order they appeared.
    """
    offenders = []
    for line in check_output.splitlines():
        stripped = line.strip()
        if not re.match(r"^[-+] [A-Za-z0-9._-]+", stripped) or "file://" in stripped:
            continue
        offenders.append(stripped)
    return offenders


def verify_synced_deps(ctx: BuildContext, docker_env: dict[str, str]) -> None:
    """Fail when the freshly built image's environment does not match the repository's lock.

    A cached or partially applied dependency layer leaves an older resolved environment inside
    an image that is about to be tagged as current. That is invisible at tag time and only
    surfaces as a version mismatch at run time. Comparing ``uv.lock`` files would not catch it,
    because the final source copy overwrites the lock in the image regardless of what was
    installed, so ask uv whether the *installed* packages still satisfy the lock.

    Args:
        ctx: Build context, used to pick the extras the image was synced with.
        docker_env: Parsed ``docker/.env.base`` values, used to locate Isaac Lab in the image.

    Raises:
        BuildError: If a third-party package in the image is out of sync with the lock, or if the
            check could not be run at all.
    """
    if not (REPO_ROOT / "uv.lock").is_file():
        return
    isaaclab_path = docker_env["DOCKER_ISAACLAB_PATH"]
    extras = uv_sync_extras(ctx)
    # ``--check`` only inspects the environment against the lock; ``--frozen`` keeps it from
    # re-locking (which would need the network) and ``--offline`` makes that failure explicit.
    check_cmd = f"cd {isaaclab_path} && uv sync --check --frozen --offline {extras}"
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", BASE_IMAGE, "-lc", check_cmd],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        print("Verified image environment matches the repository's uv.lock.")
        return

    combined = f"{proc.stdout}\n{proc.stderr}".strip()
    offenders = out_of_sync_packages(combined)
    if offenders:
        listed = "\n".join(offenders)
        raise BuildError(
            f"{BASE_IMAGE} environment is out of sync with uv.lock:\n"
            f"{listed}\n"
            "The build reused a stale dependency layer, which a -s/--source overlay cannot update.\n"
            "Rebuild with './buildnpush.sh <tag>' to rebase onto the lock-matched deps image, or\n"
            "'./buildnpush.sh <tag> -d' to re-resolve the dependencies from scratch."
        )
    if UV_OUTDATED_MARKER not in combined:
        raise BuildError(f"could not verify {BASE_IMAGE} against uv.lock; 'uv sync --check' failed:\n{combined}")
    print("Verified image environment matches the repository's uv.lock (workspace members aside).")


def tag_and_push(ctx: BuildContext, plan: BuildPlan) -> None:
    """Record dependency checkpoints, tag the final image, and optionally push."""

    if plan.run_pip_install or not plan.skip_deps:
        print(f"Recording prepared dependency checkpoint as {ctx.prepared_image}")
        docker("tag", BASE_IMAGE, ctx.prepared_image)
    else:
        print(f"Preserving prepared dependency checkpoint: {ctx.prepared_image}")

    print(f"Tagging image as {ctx.final_image}")
    docker("tag", BASE_IMAGE, ctx.final_image)

    if ctx.args.skip_push:
        print("Skipping push (--skip-push)")
        return
    print(f"Pushing to NGC: {ctx.final_image}")
    run([str(NGC), "registry", "image", "push", ctx.final_image])
    print(f"Image pushed: {ctx.final_image}")


def update_state_and_cleanup(ctx: BuildContext) -> None:
    """Track recent dependency hashes and remove unmapped old deps caches."""

    print("\nCleaning up old deps images...")
    kitless_flag = "1" if ctx.args.kitless else "0"
    now = int(time.time())
    entries: list[tuple[str, str, int, str]] = []

    if STATE_FILE.exists():
        for line in STATE_FILE.read_text().splitlines():
            parts = line.split("|")
            if len(parts) == 4:
                tag, dep_hash, timestamp, flag = parts
            elif len(parts) == 3:
                tag, dep_hash, timestamp = parts
                flag = "0"
            else:
                continue
            if tag == ctx.args.tag and flag == kitless_flag:
                continue
            try:
                entries.append((tag, dep_hash, int(timestamp), flag))
            except ValueError:
                continue

    entries.append((ctx.args.tag, ctx.deps_hash, now, kitless_flag))

    keep = {ctx.deps_image}
    fresh_entries: list[tuple[str, str, int, str]] = []
    for tag, dep_hash, timestamp, flag in entries:
        age_days = (now - timestamp) // 86400
        if age_days <= MAX_STATE_AGE_DAYS:
            fresh_entries.append((tag, dep_hash, timestamp, flag))
            keep.add(f"isaac-lab-deps:{dep_hash}")
            print(f"   keeping tag '{tag}' deps:{dep_hash} kitless={flag} age={age_days}d")
        else:
            print(f"   expired tag '{tag}' mapping age={age_days}d")

    STATE_FILE.write_text(
        "".join(f"{tag}|{dep_hash}|{timestamp}|{flag}\n" for tag, dep_hash, timestamp, flag in fresh_entries)
    )

    removed = 0
    for image in list_images("isaac-lab-deps:*"):
        if image in keep:
            continue
        print(f"   removing {image} (not mapped to any tag)")
        proc = subprocess.run(
            ["docker", "rmi", image],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            removed += 1
        else:
            print("      in use, skipped")
    if removed:
        print(f"   removed {removed} old deps image(s)")
    docker("image", "prune", "-f", capture=True, check=False)


def clean_stale_egg_info() -> None:
    """Remove stale editable-install ``*.egg-info`` dirs from ``source/`` before building.

    They are regenerable build artifacts. When created by a local install they are owned by the host
    user, but the container build runs as uid 1000; setuptools then aborts with "Cannot update time stamp
    of directory '<pkg>.egg-info'". ``source/`` is bind-mounted into the container, so ``.dockerignore``
    cannot filter them out - clearing them here is the reliable fix.
    """
    removed = 0
    for egg_info in sorted((REPO_ROOT / "source").glob("*/*.egg-info")):
        if egg_info.is_dir():
            shutil.rmtree(egg_info, ignore_errors=True)
            removed += 1
    if removed:
        print(f"Removed {removed} stale *.egg-info dir(s) from source/")


def build_image(args: BuildArgs) -> None:
    """Run the requested build."""

    clean_stale_egg_info()
    ctx = make_context(args)
    print("Dependency Analysis...")
    print(f"   Hash: {ctx.deps_hash if ctx.deps_hash else 'SKIP (-s)'}")
    plan = determine_plan(ctx)
    print_build_config(ctx, plan)

    docker_env = parse_env_file(REPO_ROOT / "docker/.env.base")
    with resolved_symlinks():
        if plan.skip_deps:
            build_overlay(ctx, plan, docker_env)
        else:
            build_full_deps(ctx, plan, docker_env)
        # Every path needs this gate, including the source-only overlay: that overlay inherits its
        # environment from a prepared image that may have been built against an older lock, so it is
        # the path most likely to ship a stale dependency set.
        verify_synced_deps(ctx, docker_env)
        tag_and_push(ctx, plan)
    if ctx.deps_hash:
        update_state_and_cleanup(ctx)
    print("\nDone.")


def show_status() -> None:
    """Print local image status."""

    current_hash = compute_deps_hash(False)
    print("Isaac Lab Docker Images Status")
    print("================================")
    print(f"Current deps hash: {current_hash}\n")
    print("Images:")
    rows = docker("images", "--format", "  {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}", capture=True)
    shown = False
    for row in rows.splitlines():
        if re.match(r"^(isaac-lab|nvcr\.io/nvidian/octi-isaac-lab)", row):
            print(row)
            shown = True
    if not shown:
        print("  (none)")
    print("\nDisk Usage:")
    df = docker("system", "df", "--format", "  Images: {{.Size}}", capture=True, check=False)
    print(df.splitlines()[0] if df.splitlines() else "  unavailable")
    if image_exists(f"isaac-lab-deps:{current_hash}"):
        print(f"\nCurrent deps image exists: isaac-lab-deps:{current_hash}")
    else:
        print(f"\nNo deps image for current hash: {current_hash}")


def clean_old_deps() -> None:
    """Remove old dependency-cache images, preserving the current hash."""

    current_hash = compute_deps_hash(False)
    current_image = f"isaac-lab-deps:{current_hash}"
    deps_images = list_images("isaac-lab-deps:*")
    if not deps_images:
        print("No deps images found.")
        return
    print(f"Current hash: {current_hash}\n")
    removed = 0
    for image in deps_images:
        if image == current_image:
            print(f"  keeping: {image} (current)")
            continue
        print(f"  removing: {image}")
        proc = subprocess.run(
            ["docker", "rmi", image],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            removed += 1
        else:
            print("    in use, skipped")
    print(f"\nRemoved {removed} old deps image(s).")
    docker("image", "prune", "-f", check=False)


def clean_all() -> None:
    """Remove all local Isaac Lab build images after confirmation."""

    print("Removing ALL Isaac Lab Docker images.")
    print("This removes isaac-lab-base, isaac-lab-prepared:*, isaac-lab-deps:*, and final NGC tags.")
    reply = input("Are you sure? [y/N] ")
    if reply.lower() not in {"y", "yes"}:
        print("Cancelled.")
        return
    patterns = (
        "isaac-lab-base",
        "isaac-lab-prepared:",
        "isaac-lab-deps:",
        f"{FINAL_IMAGE_REPO}:",
    )
    for image in list_images():
        if image.startswith(patterns):
            print(f"  removing: {image}")
            docker("rmi", image, check=False)
    docker("image", "prune", "-f", check=False)


def tag_image(source_image: str, tag: str, *, skip_push: bool) -> None:
    """Retag an existing image and optionally push it."""

    if not image_exists(source_image):
        raise BuildError(f"source image not found locally: {source_image}")
    final_image = f"{FINAL_IMAGE_REPO}:{tag}"
    prepared_image = f"isaac-lab-prepared:{tag}"
    print("Retagging image")
    print(f"   Source:   {source_image}")
    print(f"   Prepared: {prepared_image}")
    print(f"   Final:    {final_image}")
    docker("tag", source_image, prepared_image)
    docker("tag", source_image, final_image)
    if skip_push:
        print("Skipping push (--skip-push)")
        return
    print(f"Pushing to NGC: {final_image}")
    run([str(NGC), "registry", "image", "push", final_image])
    print(f"Image pushed: {final_image}")


def add_cache_mount(
    docker_args: list[str], env_name: str, default_source: str, shared_root_suffix: str, target: str
) -> None:
    """Add a Docker volume mount for a cache directory if it exists."""

    source = os.environ.get(env_name, "")
    if not source and os.environ.get("HOST_ISAACLAB_CACHE_ROOT"):
        source = str(Path(os.environ["HOST_ISAACLAB_CACHE_ROOT"]) / shared_root_suffix)
    if not source:
        source = default_source
    path = Path(source).expanduser()
    if path.is_dir():
        docker_args += ["-v", f"{path.resolve()}:{target}:rw"]


def enter_container(tag: str, *, mount_local: bool) -> None:
    """Enter a built container image."""

    image = None
    for candidate in [
        f"{FINAL_IMAGE_REPO}:{tag}",
        f"isaac-lab-prepared:{tag}",
        f"isaac-lab-base:{tag}",
        BASE_IMAGE,
    ]:
        if image_exists(candidate):
            image = candidate
            break
    if image is None:
        print("Available images:")
        for img in list_images():
            if "isaac-lab" in img or "octi-isaac-lab" in img:
                print(f"  {img}")
        raise BuildError(f"no image found for tag '{tag}'")

    docker_args = [
        "--gpus",
        "all",
        "--network",
        "host",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "OMNI_KIT_ALLOW_ROOT=1",
        "-e",
        f"WANDB_USERNAME={os.environ.get('WANDB_USERNAME', 'nvidia')}",
    ]
    if os.environ.get("WANDB_API_KEY"):
        docker_args += ["-e", f"WANDB_API_KEY={os.environ['WANDB_API_KEY']}"]
    netrc = Path.home() / ".netrc"
    if netrc.is_file():
        docker_args += ["-v", f"{netrc}:/root/.netrc:ro"]

    add_cache_mount(
        docker_args,
        "HOST_ISAACSIM_KIT_CACHE_PATH",
        "_isaac_sim/kit/cache",
        "isaac-sim/kit/cache",
        "/isaac-sim/kit/cache",
    )
    add_cache_mount(
        docker_args,
        "HOST_OMNIVERSE_CACHE_PATH",
        str(Path.home() / ".cache/ov"),
        "ov",
        "/root/.cache/ov",
    )
    add_cache_mount(
        docker_args,
        "HOST_NVIDIA_GL_CACHE_PATH",
        str(Path.home() / ".cache/nvidia/GLCache"),
        "nvidia/GLCache",
        "/root/.cache/nvidia/GLCache",
    )
    add_cache_mount(
        docker_args,
        "HOST_NVIDIA_COMPUTE_CACHE_PATH",
        str(Path.home() / ".nv/ComputeCache"),
        "nv/ComputeCache",
        "/root/.nv/ComputeCache",
    )
    add_cache_mount(
        docker_args,
        "HOST_NVIDIA_OPTIX_CACHE_PATH",
        str(Path.home() / ".cache/NVIDIA/OptixCache"),
        "NVIDIA/OptixCache",
        "/root/.cache/NVIDIA/OptixCache",
    )

    # ``models_tmp`` is bind-mounted and written by the in-container user (uid 1000),
    # while host-side tools (e.g. the wandb checkpoint download in ``play``) write to
    # it as the host user. Those uids differ, so make the mount point world-writable
    # and self-heal a stale, non-writable one a previous container left behind. The
    # repo root is user-owned, so an *empty* foreign-owned ``models_tmp`` can be
    # replaced without sudo (deletion is governed by the parent directory's perms).
    models_tmp = REPO_ROOT / "models_tmp"
    if models_tmp.is_dir() and not os.access(models_tmp, os.W_OK):
        try:
            models_tmp.rmdir()  # only succeeds if empty; never touches foreign files
        except OSError:
            print(
                f"   WARNING: {models_tmp} is not writable and not empty. Host-side "
                f"checkpoint writes may fail; run 'sudo rm -rf {models_tmp}' to reset it."
            )
    models_tmp.mkdir(exist_ok=True)
    with contextlib.suppress(PermissionError):
        models_tmp.chmod(0o777)
    docker_args += ["-v", f"{models_tmp}:/workspace/isaaclab/models_tmp:rw"]
    if mount_local:
        docker_args += ["-v", f"{REPO_ROOT}:/local:rw"]
        print(f"Entering container: {image} (local mounted at /local)")
    else:
        print(f"Entering container: {image}")
    run(["docker", "run", "-it", "--rm", *docker_args, "--entrypoint", "/bin/bash", image])


def print_help() -> None:
    """Print command help."""

    print(
        """Usage:
  ./buildnpush.sh <tag> [options]
  ./buildnpush.sh tag <source-image> <tag> [--skip-push]
  ./buildnpush.sh enter <tag> [--mount]
  ./buildnpush.sh --status
  ./buildnpush.sh --clean
  ./buildnpush.sh --clean-all

Build Options:
  -s, --source      Copy source only on top of isaac-lab-prepared:<tag>
  -p, --pip         Copy source and run uv sync on top of isaac-lab-prepared:<tag>
  -d, --deps        Rebuild dependency/base image, then copy source
  -a, --all         Full no-cache dependency/base rebuild, then copy source
      --kitless     Build the Newton-only image from docker/Dockerfile.kitless
      --skip-push   Build/tag only, do not push to NGC
  -h, --help        Show this help

Image roles:
  isaac-lab-deps:<hash>       dependency cache
  isaac-lab-prepared:<tag>    latest local prepared image for fast -s/-p
  nvcr.io/nvidian/octi-isaac-lab:<tag> final pushed image
""".rstrip()
    )


def parse_build_args(argv: list[str]) -> BuildArgs:
    """Parse build-mode CLI arguments."""

    parser = argparse.ArgumentParser(prog="./buildnpush.sh", add_help=False)
    parser.add_argument("tag")
    parser.add_argument("-s", "--source", action="store_true")
    parser.add_argument("-p", "--pip", action="store_true")
    parser.add_argument("-d", "--deps", action="store_true")
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--kitless", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    ns = parser.parse_args(argv)
    if ns.help:
        print_help()
        raise SystemExit(0)
    depth_flags = [ns.source, ns.pip, ns.deps, ns.all]
    if sum(bool(flag) for flag in depth_flags) > 1:
        raise BuildError("choose only one of -s/--source, -p/--pip, -d/--deps, or -a/--all")
    return BuildArgs(
        tag=ns.tag,
        source=ns.source,
        pip=ns.pip,
        deps=ns.deps,
        all=ns.all,
        skip_push=ns.skip_push,
        kitless=ns.kitless,
        kitless_base_image=os.environ.get("KITLESS_BASE_IMAGE", DEFAULT_KITLESS_BASE_IMAGE),
    )


def main(argv: list[str]) -> int:
    """CLI entry point."""

    if not argv or argv[0] in {"-h", "--help"}:
        print_help()
        return 0

    try:
        if argv[0] == "tag":
            parser = argparse.ArgumentParser(prog="./buildnpush.sh tag")
            parser.add_argument("source_image")
            parser.add_argument("tag")
            parser.add_argument("--skip-push", action="store_true")
            ns = parser.parse_args(argv[1:])
            tag_image(ns.source_image, ns.tag, skip_push=ns.skip_push)
            return 0
        if argv[0] == "enter":
            parser = argparse.ArgumentParser(prog="./buildnpush.sh enter")
            parser.add_argument("tag")
            parser.add_argument("--mount", action="store_true")
            ns = parser.parse_args(argv[1:])
            enter_container(ns.tag, mount_local=ns.mount)
            return 0
        if argv[0] == "--status":
            show_status()
            return 0
        if argv[0] == "--clean":
            clean_old_deps()
            return 0
        if argv[0] == "--clean-all":
            clean_all()
            return 0
        build_image(parse_build_args(argv))
        return 0
    except BuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
