# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/buildnpush.py``."""

from __future__ import annotations

import contextlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import buildnpush as bp

# Real ``uv sync --check`` output shapes. Editable workspace members are reported on every check
# even when the image is correct, so only the git/registry entries indicate an actually stale image.
_UV_CHECK_WORKSPACE_ONLY = """Would download 14 packages
Would uninstall 14 packages
Would install 14 packages
 - isaaclab==13.2.1 (from file:///workspace/isaaclab/source/isaaclab)
 + isaaclab @ file:///workspace/isaaclab/source/isaaclab
 - isaaclab-newton==2.2.0 (from file:///workspace/isaaclab/source/isaaclab_newton)
 + isaaclab-newton @ file:///workspace/isaaclab/source/isaaclab_newton
The environment is outdated; run `uv sync` to update the environment
"""

_UV_CHECK_DRIFTED_DEPENDENCY = """Would install 1 package
 - isaaclab==13.2.1 (from file:///workspace/isaaclab/source/isaaclab)
 + isaaclab @ file:///workspace/isaaclab/source/isaaclab
 + newton @ git+https://github.com/newton-physics/newton.git@d7581b73
The environment is outdated; run `uv sync` to update the environment
"""


def _ctx(args: bp.BuildArgs) -> bp.BuildContext:
    return bp.BuildContext(
        args=args,
        deps_hash="abc123",
        deps_image="isaac-lab-deps:abc123",
        prepared_image=f"isaac-lab-prepared:{args.tag}",
        final_image=f"nvcr.io/nvidian/octi-isaac-lab:{args.tag}",
    )


class BuildnpushTest(unittest.TestCase):
    """Tests for build image role and strategy decisions."""

    def _mock_images(self, created: dict[str, str]):
        return mock.patch.multiple(
            bp,
            image_exists=lambda image: image in created,
            image_created=lambda image: created.get(image, ""),
            image_layer_count=lambda image: 80 if image in created else 0,
        )

    def test_source_overlay_uses_prepared_image_even_when_hash_deps_exist(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", source=True))
        with self._mock_images(
            {
                ctx.prepared_image: "2026-06-16T20:00:00Z",
                ctx.final_image: "2026-06-16T19:00:00Z",
                ctx.deps_image: "2026-06-16T18:00:00Z",
                bp.BASE_IMAGE: "2026-06-16T17:00:00Z",
            }
        ):
            plan = bp.determine_plan(ctx)

        self.assertTrue(plan.skip_deps)
        self.assertFalse(plan.run_pip_install)
        self.assertEqual(plan.build_base_image, ctx.prepared_image)

    def test_source_overlay_prefers_checkpoint_over_newer_final_image(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", source=True))
        with self._mock_images(
            {
                ctx.prepared_image: "2026-06-16T19:00:00Z",
                ctx.final_image: "2026-06-16T20:00:00Z",
                bp.BASE_IMAGE: "2026-06-16T18:00:00Z",
            }
        ):
            plan = bp.determine_plan(ctx)

        self.assertEqual(plan.build_base_image, ctx.prepared_image)

    def test_source_build_does_not_compute_dependency_hash(self) -> None:
        with mock.patch.object(bp, "compute_deps_hash", side_effect=AssertionError("hash called")):
            ctx = bp.make_context(bp.BuildArgs(tag="factory", source=True))

        self.assertEqual(ctx.deps_hash, "")
        self.assertEqual(ctx.deps_image, "")

    def test_source_overlay_requires_prepared_image(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", source=True))
        with self._mock_images({}):
            with self.assertRaisesRegex(bp.BuildError, "No prepared image"):
                bp.determine_plan(ctx)

    def test_source_only_build_defers_lock_reconciliation(self) -> None:
        """A source-only build leaves lock reconciliation to the cluster launch script."""

        def reject_deps_build(*args, **kwargs):
            raise AssertionError("a source-only build must not rebuild dependencies")

        verify_lockfile = mock.Mock()
        verify_synced_deps = mock.Mock()
        with mock.patch.multiple(
            bp,
            clean_stale_egg_info=lambda: None,
            verify_lockfile=verify_lockfile,
            determine_plan=lambda ctx: bp.BuildPlan(
                skip_deps=True,
                use_cache=True,
                run_pip_install=False,
                reason="source",
                build_base_image=ctx.prepared_image,
            ),
            print_build_config=lambda ctx, plan: None,
            parse_env_file=lambda path: {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab"},
            resolved_symlinks=contextlib.nullcontext,
            build_overlay=lambda ctx, plan, docker_env: None,
            build_full_deps=reject_deps_build,
            verify_synced_deps=verify_synced_deps,
            tag_and_push=lambda ctx, plan: None,
        ):
            bp.build_image(bp.BuildArgs(tag="factory", source=True, skip_push=True))

        verify_lockfile.assert_called_once_with()
        verify_synced_deps.assert_not_called()

    def test_pip_overlay_uses_prepared_image_before_deps_cache(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", pip=True))
        with self._mock_images(
            {
                ctx.prepared_image: "2026-06-16T20:00:00Z",
                ctx.deps_image: "2026-06-16T19:00:00Z",
            }
        ):
            plan = bp.determine_plan(ctx)

        self.assertTrue(plan.skip_deps)
        self.assertTrue(plan.run_pip_install)
        self.assertEqual(plan.build_base_image, ctx.prepared_image)

    def test_default_build_uses_matching_hash_deps_cache(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory"))
        with self._mock_images({ctx.deps_image: "2026-06-16T20:00:00Z"}):
            plan = bp.determine_plan(ctx)

        self.assertTrue(plan.skip_deps)
        self.assertFalse(plan.run_pip_install)
        self.assertEqual(plan.build_base_image, ctx.deps_image)

    def test_source_tagging_preserves_prepared_checkpoint(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", source=True, skip_push=True))
        plan = bp.BuildPlan(
            skip_deps=True,
            use_cache=True,
            run_pip_install=False,
            reason="source",
            build_base_image=ctx.prepared_image,
        )
        calls: list[tuple[str, ...]] = []
        with mock.patch.object(bp, "docker", lambda *args, **kwargs: calls.append(args) or ""):
            bp.tag_and_push(ctx, plan)

        self.assertEqual(calls, [("tag", bp.BASE_IMAGE, ctx.final_image)])

    def test_pip_tagging_updates_prepared_checkpoint(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", pip=True, skip_push=True))
        plan = bp.BuildPlan(
            skip_deps=True,
            use_cache=True,
            run_pip_install=True,
            reason="pip",
            build_base_image=ctx.prepared_image,
        )
        calls: list[tuple[str, ...]] = []
        with mock.patch.object(bp, "docker", lambda *args, **kwargs: calls.append(args) or ""):
            bp.tag_and_push(ctx, plan)

        self.assertEqual(
            calls,
            [
                ("tag", bp.BASE_IMAGE, ctx.prepared_image),
                ("tag", bp.BASE_IMAGE, ctx.final_image),
            ],
        )

    def test_tag_image_records_prepared_and_final_tags(self) -> None:
        calls: list[tuple[str, ...]] = []
        with mock.patch.object(bp, "image_exists", lambda image: image == "isaac-lab-base"):
            with mock.patch.object(bp, "docker", lambda *args, **kwargs: calls.append(args) or ""):
                bp.tag_image("isaac-lab-base", "factory", skip_push=True)

        self.assertEqual(
            calls,
            [
                ("tag", "isaac-lab-base", "isaac-lab-prepared:factory"),
                ("tag", "isaac-lab-base", "nvcr.io/nvidian/octi-isaac-lab:factory"),
            ],
        )

    def test_parse_build_args_rejects_multiple_depth_flags(self) -> None:
        with self.assertRaisesRegex(bp.BuildError, "choose only one"):
            bp.parse_build_args(["factory", "-s", "-p"])

    def test_deps_rebuild_forces_dependency_resync(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", deps=True))
        with self._mock_images({ctx.deps_image: "2026-06-16T20:00:00Z"}):
            plan = bp.determine_plan(ctx)

        self.assertFalse(plan.skip_deps)
        self.assertTrue(plan.run_pip_install)
        self.assertTrue(plan.bust_deps_cache)

    def test_cached_deps_build_does_not_force_dependency_resync(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory"))
        with self._mock_images({ctx.deps_image: "2026-06-16T20:00:00Z"}):
            plan = bp.determine_plan(ctx)

        self.assertFalse(plan.bust_deps_cache)

    def test_deps_hash_ignores_lock_changes(self) -> None:
        lock = bp.REPO_ROOT / "uv.lock"
        if not lock.is_file():
            self.skipTest("uv.lock is not present in this checkout")
        original = lock.read_bytes()
        before = bp.compute_deps_hash(False)
        try:
            lock.write_bytes(original + b"\n# deps-hash probe\n")
            after = bp.compute_deps_hash(False)
        finally:
            lock.write_bytes(original)

        self.assertEqual(before, after)

    def test_deps_hash_ignores_python_project_changes(self) -> None:
        for relative_path in ("pyproject.toml", "source/isaaclab_rl/pyproject.toml"):
            with self.subTest(path=relative_path):
                path = bp.REPO_ROOT / relative_path
                original = path.read_bytes()
                before = bp.compute_deps_hash(False)
                try:
                    path.write_bytes(original + b"\n# deps-hash probe\n")
                    after = bp.compute_deps_hash(False)
                finally:
                    path.write_bytes(original)
                self.assertEqual(before, after)

    def test_deps_hash_changes_with_the_foundation_recipe(self) -> None:
        recipe = bp.REPO_ROOT / "docker/Dockerfile.base"
        original = recipe.read_bytes()
        before = bp.compute_deps_hash(False)
        try:
            recipe.write_bytes(original + b"\n# foundation-hash probe\n")
            after = bp.compute_deps_hash(False)
        finally:
            recipe.write_bytes(original)

        self.assertNotEqual(before, after)

    def test_deps_hash_changes_with_the_apt_installer(self) -> None:
        installer = bp.REPO_ROOT / "tools/install_deps.py"
        original = installer.read_bytes()
        before = bp.compute_deps_hash(False)
        try:
            installer.write_bytes(original + b"\n# foundation-hash probe\n")
            after = bp.compute_deps_hash(False)
        finally:
            installer.write_bytes(original)

        self.assertNotEqual(before, after)

    def test_deps_hash_changes_with_extension_system_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extension = root / "source/example/config/extension.toml"
            extension.parent.mkdir(parents=True)
            extension.write_text("[package]\n")
            with mock.patch.object(bp, "REPO_ROOT", root):
                before = bp.compute_deps_hash(False)
                extension.write_text('[package]\n[isaac_lab_settings]\napt_deps = ["git"]\n')
                after = bp.compute_deps_hash(False)

        self.assertNotEqual(before, after)

    def test_deps_hash_ignores_extension_metadata_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extension = root / "source/example/config/extension.toml"
            extension.parent.mkdir(parents=True)
            extension.write_text('[package]\nversion = "1.0.0"\n')
            with mock.patch.object(bp, "REPO_ROOT", root):
                before = bp.compute_deps_hash(False)
                extension.write_text('[package]\nversion = "1.0.1"\n')
                after = bp.compute_deps_hash(False)

        self.assertEqual(before, after)

    def test_verify_lockfile_checks_without_installing(self) -> None:
        with mock.patch.object(bp, "run") as run:
            bp.verify_lockfile()

        run.assert_called_once_with(["uv", "lock", "--check", "--offline"])

    def _mock_uv_check(self, returncode: int, stderr: str = ""):
        """Patch the ``docker run`` that executes ``uv sync --check`` inside the built image."""
        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

        return mock.patch.object(bp.subprocess, "run", fake_run), captured

    def test_verify_synced_deps_rejects_an_environment_that_drifted_from_the_lock(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory"))
        docker_env = {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab"}
        patcher, _ = self._mock_uv_check(1, stderr=_UV_CHECK_DRIFTED_DEPENDENCY)
        with patcher:
            with self.assertRaisesRegex(bp.BuildError, "out of sync with uv.lock"):
                bp.verify_synced_deps(ctx, docker_env)

    def test_verify_synced_deps_accepts_an_environment_whose_only_diff_is_workspace_members(self) -> None:
        """uv always re-reports the editable ``source/`` packages, which must not fail the build."""
        ctx = _ctx(bp.BuildArgs(tag="factory"))
        docker_env = {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab"}
        patcher, _ = self._mock_uv_check(1, stderr=_UV_CHECK_WORKSPACE_ONLY)
        with patcher:
            bp.verify_synced_deps(ctx, docker_env)

    def test_verify_synced_deps_rejects_a_check_that_could_not_run(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory"))
        docker_env = {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab"}
        patcher, _ = self._mock_uv_check(127, stderr="bash: uv: command not found")
        with patcher:
            with self.assertRaisesRegex(bp.BuildError, "could not verify"):
                bp.verify_synced_deps(ctx, docker_env)

    def test_out_of_sync_packages_keeps_only_non_workspace_entries(self) -> None:
        self.assertEqual(bp.out_of_sync_packages(_UV_CHECK_WORKSPACE_ONLY), [])
        self.assertEqual(
            bp.out_of_sync_packages(_UV_CHECK_DRIFTED_DEPENDENCY),
            ["+ newton @ git+https://github.com/newton-physics/newton.git@d7581b73"],
        )

    def test_verify_synced_deps_accepts_an_environment_that_matches_the_lock(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory"))
        docker_env = {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab"}
        patcher, captured = self._mock_uv_check(0)
        with patcher:
            bp.verify_synced_deps(ctx, docker_env)

        self.assertIn("uv sync --check --frozen --offline " + bp.BASE_UV_SYNC_EXTRAS, captured[0][-1])

    def test_verify_synced_deps_checks_the_kitless_extras_for_a_kitless_build(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", kitless=True))
        docker_env = {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab"}
        patcher, captured = self._mock_uv_check(0)
        with patcher:
            bp.verify_synced_deps(ctx, docker_env)

        self.assertIn("uv sync --check --frozen --offline " + bp.KITLESS_UV_SYNC_EXTRAS, captured[0][-1])

    def test_base_uv_sync_extras_match_the_dockerfiles(self) -> None:
        """The base and source-overlay defaults must install the same extras."""
        for dockerfile in ("docker/Dockerfile.base", "docker/Dockerfile.source-only"):
            with self.subTest(dockerfile=dockerfile):
                text = (bp.REPO_ROOT / dockerfile).read_text()
                match = re.search(r'^ARG ISAACLAB_UV_SYNC_ARGS="([^"]*)"', text, re.MULTILINE)
                if match is None:
                    self.fail(f"{dockerfile} does not define ISAACLAB_UV_SYNC_ARGS")
                self.assertEqual(match.group(1), bp.BASE_UV_SYNC_EXTRAS)

    def test_kitless_overlay_receives_kitless_sync_extras(self) -> None:
        ctx = _ctx(bp.BuildArgs(tag="factory", kitless=True))
        plan = bp.BuildPlan(True, True, False, "cached", ctx.deps_image)
        calls: list[tuple[str, ...]] = []
        with mock.patch.object(bp, "docker", lambda *args, **kwargs: calls.append(args) or ""):
            bp.build_overlay(
                ctx, plan, {"DOCKER_ISAACLAB_PATH": "/workspace/isaaclab", "DOCKER_ISAACSIM_ROOT_PATH": ""}
            )

        self.assertIn(f"ISAACLAB_UV_SYNC_ARGS={bp.KITLESS_UV_SYNC_EXTRAS}", calls[0])


if __name__ == "__main__":
    unittest.main()
