# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/buildnpush.py``."""

from __future__ import annotations

import unittest
from unittest import mock

import buildnpush as bp


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


if __name__ == "__main__":
    unittest.main()
