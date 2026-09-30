# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloner path/query algebra.

These exercise :mod:`isaaclab.cloner.path` and :mod:`isaaclab.cloner.query`, which are pure
string/array operations over a :class:`~isaaclab.cloner.ClonePlan`. They need no stage, no
simulator and no USD, so they live outside ``test/sim/``.
"""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.cloner import ClonePlan, make_clone_plan

##
# Path primitives.
##


def test_path_split():
    """Split clone destination templates around their clone slot."""
    assert cloner.path.split("/World/envs/env_{}/Robot") == ("/World/envs/env_", "/Robot")
    assert cloner.path.split("/World/scenes/{}/") == ("/World/scenes/", "")
    # A second slot would survive into the suffix and break the later format call.
    with pytest.raises(ValueError, match="at most one"):
        cloner.path.split("/World/envs/env_{}/Robot/{}")


def test_path_split_without_a_clone_slot():
    """A destination with no clone slot names one prim every env shares, so it is all prefix.

    That the slot is absent is what tells a caller the row is not replicated, which is how a
    global asset is spelled: its destination is its source path.
    """
    assert cloner.path.split("/World/ground") == ("/World/ground", "")
    assert cloner.path.split("/World/ground/") == ("/World/ground", "")


def test_path_algebra_degenerates_for_a_global_destination():
    """Every path operation is identity-ish on a slotless destination, so callers need no branch."""
    ground = "/World/ground"
    # Matching is only "is this under it"; there is no instance to capture.
    assert cloner.path.match("/World/ground/Plane", ground) == ("", "/Plane")
    assert cloner.path.match(ground, ground) == ("", "")
    assert cloner.path.match("/World/envs/env_0/Robot", ground) is None
    # Source and destination are the same path, so rebasing a global is a no-op.
    assert cloner.path.rebase("/World/ground/Plane", ground, ground) == "/World/ground/Plane"
    assert cloner.path.relativize("/World/ground/Plane", ground) == "/Plane"


def test_path_relative_to():
    """relative_to strips a concrete root on a boundary, or returns None."""
    root = "/World/envs/env_0/Robot"
    assert cloner.path.relative_to("/World/envs/env_0/Robot/base", root) == "/base"
    assert cloner.path.relative_to("/World/envs/env_0/Robot", root) == ""
    assert cloner.path.relative_to("/World/envs/env_0/RobotArm", root) is None
    assert cloner.path.relative_to("/World/ground", root) is None


def test_path_rebase():
    """rebase swaps a boundary-aligned root prefix, not a substring."""
    assert (
        cloner.path.rebase("/World/envs/env_0/Robot/base", "/World/envs/env_0", "/World/envs/env_5")
        == "/World/envs/env_5/Robot/base"
    )


def test_expand_env_regex_ns_preserves_regex_quantifiers():
    """Macro expansion changes only the named macro, not braces owned by the regex."""
    path_expr = r"{ENV_REGEX_NS}/Robot/link_[0-9]{2}"

    assert cloner.expand_env_regex_ns(path_expr) == r"/World/envs/env_[^/]+/Robot/link_[0-9]{2}"
    assert cloner.expand_env_regex_ns(path_expr, "/World/scenes/scene_{}") == (
        r"/World/scenes/scene_[^/]+/Robot/link_[0-9]{2}"
    )
    # boundary-safe: str.replace would corrupt this, rebase leaves it unchanged
    assert (
        cloner.path.rebase("/World/envs/env_0X/Robot", "/World/envs/env_0", "/World/envs/env_5")
        == "/World/envs/env_0X/Robot"
    )


def test_make_clone_plan_retains_per_variant_cfg_semantics():
    """Semantic metadata follows each MultiAsset variant and includes its shared cfg tags."""
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(semantic_tags=[("class name", "cone shape")]),
                sim_utils.SphereCfg(semantic_tags=[("class name", "sphere shape")]),
            ],
            semantic_tags=[("group", "prop")],
        ),
    )

    plan = make_clone_plan((cfg,), num_clones=2, env_spacing=1.0)

    assert plan.semantic_tags == (
        (("class_name", "cone_shape"), ("group", "prop")),
        (("class_name", "sphere_shape"), ("group", "prop")),
    )


def test_cfg_source_paths_preserves_inactive_variant_slots_without_mutating_cfg():
    """Plan-owned spawn sources retain variant identity even when one row is inactive."""
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(radius=0.1, height=0.2),
                sim_utils.SphereCfg(radius=0.1),
                sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
            ]
        ),
    )
    before = cfg.spawn.to_dict()

    plan = make_clone_plan(
        (cfg,),
        num_clones=2,
        env_spacing=1.0,
        valid_set=np.asarray([[0], [1]], dtype=np.int64),
    )

    assert cloner.query.cfg_source_paths(plan, cfg) == (
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
        None,
    )
    assert cfg.spawn.to_dict() == before


def test_default_clone_strategy_assigns_combinations_sequentially():
    """The default strategy assigns legal combinations in contiguous balanced blocks."""
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.ConeCfg(), sim_utils.SphereCfg()]),
    )

    plan = make_clone_plan((cfg,), num_clones=6, env_spacing=1.0)

    assert plan.clone_mask.tolist() == [
        [True, True, True, False, False, False],
        [False, False, False, True, True, True],
    ]


def test_sequential_clone_combinations_preserve_weights():
    """Repeated legal rows preserve their configured share under sequential assignment."""
    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.ConeCfg(), sim_utils.SphereCfg()]),
    )

    plan = make_clone_plan(
        (cfg,), num_clones=10, env_spacing=1.0, valid_set=np.asarray([[0], [1], [1], [1]], dtype=np.int64)
    )

    assert plan.clone_mask.sum(axis=1).tolist() == [3, 7]
    assert plan.clone_mask[0].tolist() == [True, True, True, False, False, False, False, False, False, False]


def test_raycast_cfg_references_a_planned_body_without_disabled_debug_markers():
    """A ray caster consumes a frame without planning debug geometry it will not draw."""
    from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
    from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg

    body = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/SensorBody",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )
    raycast = RayCasterCfg(
        prim_path=body.prim_path,
        mesh_prim_paths=["/World/Ground"],
        pattern_cfg=GridPatternCfg(),
    )

    plan = make_clone_plan((body, raycast), num_clones=2, env_spacing=1.0)

    assert "spawn" not in RayCasterCfg.__dataclass_fields__
    assert plan.sources == ("/World/envs/env_0/SensorBody",)
    assert plan.cfg_rows == {id(body): (0,)}
    assert plan.geometry_requests == ("/World/Ground",)

    raycast.debug_vis = True
    plan = make_clone_plan((body, raycast, raycast.visualizer_cfg), num_clones=2, env_spacing=1.0)
    assert plan.sources == ("/World/envs/env_0/SensorBody", "/Visuals/RayCaster")
    assert plan.cfg_rows == {id(body): (0,), id(raycast.visualizer_cfg): (1,)}


def test_global_cfg_without_a_spawner_still_owns_its_authored_root():
    """A global cfg-authored subtree belongs to the plan even when its class authors it directly."""
    terrain = SimpleNamespace(prim_path="/World/Ground")

    plan = make_clone_plan((terrain,), num_clones=2, env_spacing=1.0)

    assert plan.sources == plan.destinations == ("/World/Ground",)
    assert plan.cfg_rows == {id(terrain): (0,)}
    assert plan.semantic_tags == ((),)
    assert not bool(plan.clone_mask.any())


def test_make_clone_plan_rejects_duplicate_scene_ownership():
    """Two authoring cfgs cannot claim the same physical destination."""
    cfgs = [
        SimpleNamespace(
            prim_path="/World/envs/env_[^/]+/Object",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        )
        for _ in range(2)
    ]

    with pytest.raises(ValueError, match="Multiple cfgs author the same clone-plan destination"):
        make_clone_plan(cfgs, num_clones=2, env_spacing=1.0)


# (path, root) pairs spanning the ordinary cases plus the stage root and trailing slashes.
_PATH_ROOT_CASES = [
    ("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot"),
    ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot"),
    ("/World/envs/env_0/RobotArm", "/World/envs/env_0/Robot"),
    ("/World/ground", "/World/envs/env_0/Robot"),
    ("/World/envs/env_0/Robot", "/World/envs/env_0/"),
    ("/World/envs/env_0", "/"),
    ("/", "/"),
    ("/World", "/World"),
]


@pytest.mark.parametrize("path, root", _PATH_ROOT_CASES)
def test_path_law_membership(path, root):
    """P1: under() holds exactly when relative_to() resolves."""
    assert cloner.path.under(path, root) == (cloner.path.relative_to(path, root) is not None)


@pytest.mark.parametrize("path, root", _PATH_ROOT_CASES)
@pytest.mark.parametrize("dst_root", ["/World/other", "/World/other/", "/"])
def test_path_law_rebase_swaps_only_the_root(path, root, dst_root):
    """P2: rebase is the destination root plus the tail, and rebasing onto the same root is a no-op."""
    tail = cloner.path.relative_to(path, root)
    rebased = cloner.path.rebase(path, root, dst_root)
    if tail is None:
        assert rebased == path
    else:
        assert rebased == (dst_root.rstrip("/") + tail) or "/"
        assert cloner.path.rebase(path, root, root) == path


@pytest.mark.parametrize("path, root", _PATH_ROOT_CASES)
def test_path_law_no_special_cases(path, root):
    """P3: "/" is the root of every absolute path, and a trailing slash is insignificant."""
    assert cloner.path.under(path, "/")
    assert cloner.path.relative_to(path, root) == cloner.path.relative_to(path, root.rstrip("/") or "/")
    assert cloner.path.rebase(path, root, "/World/x") == cloner.path.rebase(path, root + "/", "/World/x")


def test_path_match_captures_the_clone_slot():
    """match keeps the instance the template's slot captured, which relativize discards."""
    tmpl = "/World/envs/env_{}/Robot"
    assert cloner.path.match("/World/envs/env_3/Robot/base", tmpl) == ("3", "/base")
    assert cloner.path.match("/World/envs/env_[^/]+/Robot", tmpl) == ("[^/]+", "")
    assert cloner.path.match("/World/envs/env_3/RobotArm", tmpl) is None


@pytest.mark.parametrize(
    "path_expr, template",
    [
        ("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot"),
        ("/World/envs/env_12/Robot", "/World/envs/env_{}/Robot"),
        ("/World/scenes/0/Robot/link", "/World/scenes/{}/Robot"),
    ],
)
def test_path_law_template_split(path_expr, template):
    """P4: a match reassembles into the original path, and its suffix is what relativize returns."""
    matched = cloner.path.match(path_expr, template)
    assert matched is not None
    assert template.format(matched.instance) + matched.suffix == path_expr
    assert cloner.path.relativize(path_expr, template) == matched.suffix


def test_path_stage_root_is_not_a_segment():
    """Rebasing off and onto "/" does not introduce an empty segment."""
    assert cloner.path.relative_to("/World/envs/env_0", "/") == "/World/envs/env_0"
    assert cloner.path.relative_to("/", "/") == ""
    assert cloner.path.rebase("/World/Robot", "/", "/Scene") == "/Scene/World/Robot"
    assert cloner.path.rebase("/World/Robot", "/World", "/") == "/Robot"


##
# Clone plans used by the query tests. Each covers a distinct shape of the source/clone
# relation, and the law tests below run over all of them.
##


def _plan(sources, destinations, mask) -> ClonePlan:
    """A plan with the given rows, whose env ids are ``0..num_envs-1``."""
    mask_array = np.asarray(mask, dtype=np.bool_)
    return ClonePlan(
        sources=tuple(sources),
        destinations=tuple(destinations),
        clone_mask=mask_array,
        env_ids=np.arange(mask_array.shape[1], dtype=np.int64),
    )


def _robot_plan(mask_row=(True, True, True, True)) -> ClonePlan:
    """A single-asset (Robot) clone plan over 4 envs with the given mask row."""
    return _plan(("/World/envs/env_0/Robot",), ("/World/envs/env_{}/Robot",), [list(mask_row)])


def _wide_env_id_plan() -> ClonePlan:
    """Two variants of one asset over 12 envs, the second starting at a two-digit env id."""
    return _plan(
        ("/World/envs/env_0/Object", "/World/envs/env_10/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[env < 10 for env in range(12)], [env >= 10 for env in range(12)]],
    )


PLANS = {
    "homogeneous": _robot_plan(),
    "partial_coverage": _robot_plan((True, True, False, True)),
    "two_variants": _plan(
        ("/World/envs/env_0/Object", "/World/envs/env_2/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[True, True, False, True], [False, False, True, False]],
    ),
    "nested_prototype": _plan(
        ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot/wrist/Camera"),
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/wrist/Camera"),
        [[True, True, True, True], [True, True, True, True]],
    ),
    "distinct_env_root": _plan(
        ("/World/source/Robot",),
        ("/World/scenes/{}/Robot",),
        [[True, True]],
    ),
    "wide_env_ids": _wide_env_id_plan(),
}


##
# Query operations.
##


def test_path_env_ids():
    """Return exactly the environments reached by a source-space path."""
    assert cloner.query.path_env_ids(_robot_plan(), "/World/envs/env_0/Robot/base") == (0, 1, 2, 3)
    assert cloner.query.path_env_ids(_robot_plan((True, True, False, True)), "/World/envs/env_0/Robot/base") == (
        0,
        1,
        3,
    )
    assert cloner.query.path_env_ids(_robot_plan(), "/World/ground") == ()


def test_path_to_clone_respects_plan_ownership():
    """Resolve source paths only into environments their nearest row populates."""
    plan = _robot_plan((True, True, False, True))
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", 3) == "/World/envs/env_3/Robot/base"
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", 2) is None
    assert cloner.query.path_to_clone(plan, "/World/ground", 0) is None

    heterogeneous = PLANS["two_variants"]
    assert cloner.query.path_to_clone(heterogeneous, "/World/envs/env_0/Object/base", 2) is None
    assert (
        cloner.query.path_to_clone(heterogeneous, "/World/envs/env_2/Object/base", 2) == "/World/envs/env_2/Object/base"
    )


def test_path_to_clone_nested_prototype_uses_nearest_source():
    """Clone a nested camera through its own row instead of its ancestor row."""
    path = "/World/envs/env_0/Robot/wrist/Camera/lens"
    assert cloner.query.path_to_clone(PLANS["nested_prototype"], path, 1) == (
        "/World/envs/env_1/Robot/wrist/Camera/lens"
    )


def test_path_to_source_nested_templates_pick_most_specific():
    """A path owned by both an ancestor and a descendant template resolves to the descendant."""
    plan = _plan(
        ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot/ee_link/palm_link/Camera"),
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/ee_link/palm_link/Camera"),
        [[True, True], [True, True]],
    )

    # The camera path matches both templates; the more specific (longer-matching) one wins.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_0/Robot/ee_link/palm_link/Camera")
    assert resolved == (
        "/World/envs/env_0/Robot/ee_link/palm_link/Camera",
        "/World/envs/env_[^/]+/Robot/ee_link/palm_link/Camera",
        "",
    )

    # A path that only the ancestor template owns still resolves against it with its suffix.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_0/Robot/base")
    assert resolved == ("/World/envs/env_0/Robot", "/World/envs/env_[^/]+/Robot", "/base")


def test_path_to_source_ambiguous_templates_raise():
    """Two distinct, equally specific templates owning a path remain a genuine ambiguity."""
    # Both templates match "/World/envs/env_0/Robot" exactly, leaving no suffix to rank by.
    plan = _plan(
        ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot"),
        ("/World/envs/{}/Robot", "/World/{}/env_0/Robot"),
        [[True, True], [True, True]],
    )

    with pytest.raises(ValueError, match="matches multiple destination templates"):
        cloner.query.path_to_source(plan, "/World/envs/env_0/Robot")


def test_path_to_source_merges_same_template_rows():
    """Heterogeneous source rows sharing one destination template resolve through a single owner."""
    # One logical asset cloned from two source variants onto the same destination template.
    # Neither row alone covers all envs; row 0 -> envs (0, 2), row 1 -> envs (1, 3).
    plan = _plan(
        ("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[True, False, True, False], [False, True, False, True]],
    )

    # Without an env id, the first populated row represents the asset.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera")
    assert resolved == ("/World/envs/env_0/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")

    # With an env id, the variant that actually populates that env is reported.
    resolved = cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera", env_id=3)
    assert resolved == ("/World/envs/env_1/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")


def test_path_to_source_partial_coverage_returns():
    """Partial-env coverage is allowed: env 3 uncovered by either row does not raise."""
    # Row 0 -> envs (0, 2), row 1 -> env (1); env 3 is covered by neither row.
    plan = _plan(
        ("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[True, False, True, False], [False, True, False, False]],
    )

    resolved = cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera")
    assert resolved == ("/World/envs/env_0/Object", "/World/envs/env_[^/]+/Object", "/Body/Camera")

    # No row populates env 3, so resolving for that env reports nothing.
    assert cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body/Camera", env_id=3) is None


def test_path_to_source_inactive_rows_return_none():
    """A template whose every matching row populates no env resolves to ``None``."""
    plan = _plan(
        ("/World/envs/env_0/Object",),
        ("/World/envs/env_{}/Object",),
        [[False, False, False, False]],
    )

    assert cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Object/Body") is None


def test_iter_sources_yields_nearest_owner():
    """ClonePlan rows can be matched by destination path expression."""
    plan = _plan(
        ("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[True, True, False, False], [False, False, True, True]],
    )

    matches = list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Object/Body/Camera"))

    assert matches == [
        (
            "/World/envs/env_0/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_0/Object/Body/Camera",
            (0, 1),
        ),
        (
            "/World/envs/env_1/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_1/Object/Body/Camera",
            (2, 3),
        ),
    ]


def test_destination_paths_follow_partial_rows_and_env_ids():
    """Exact destination paths come only from the rows and ids carried by the plan."""
    plan = ClonePlan(
        sources=("/World/envs/env_4/Object",),
        destinations=("/World/envs/env_{}/Object",),
        clone_mask=np.asarray(((False, True, False, True),), dtype=np.bool_),
        env_ids=np.asarray((2, 4, 8, 9), dtype=np.int64),
    )

    assert cloner.query.destination_paths(plan, "/World/envs/env_[^/]+/Object/mesh") == {
        4: "/World/envs/env_4/Object/mesh",
        9: "/World/envs/env_9/Object/mesh",
    }


def test_iter_sources_reports_only_the_envs_a_row_populates():
    """A row covering part of the envs is a source for those envs only."""
    plan = _plan(
        ("/World/envs/env_2/Object",),
        ("/World/envs/env_{}/Object",),
        [[False, False, True, True]],
    )

    assert list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Object/Body/Camera")) == [
        (
            "/World/envs/env_2/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_2/Object/Body/Camera",
            (2, 3),
        )
    ]


def test_iter_sources_skips_rows_without_envs():
    """A nearer template populating no env does not hide the populated ancestor owning the path."""
    plan = _plan(
        ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot/wrist/Camera"),
        ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/wrist/Camera"),
        [[True, True, False, False], [False, False, False, False]],
    )

    assert list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Robot/wrist/Camera")) == [
        (
            "/World/envs/env_0/Robot",
            "/World/envs/env_{}/Robot",
            "/World/envs/env_0/Robot/wrist/Camera",
            (0, 1),
        )
    ]


def test_iter_sources_distinct_env_root():
    """The destination template need not sit under the default env root."""
    plan = PLANS["distinct_env_root"]

    assert list(cloner.query.iter_sources(plan, "/World/scenes/[^/]+/Robot/base")) == [
        ("/World/source/Robot", "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 1))
    ]


def test_iter_sources_ranks_variants_independently_of_env_id_width():
    """Regression: a variant is not ranked out because its first env id has more digits.

    Specificity is the suffix below the destination template, which does not depend on the
    env ids a row happens to populate. Ranking by the *formatted* template instead made
    ``env_10`` look more specific than ``env_0`` and silently dropped the first variant for
    any scene with more than ten envs.
    """
    plan = _wide_env_id_plan()

    matches = list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Object/Body"))

    assert [match[0] for match in matches] == ["/World/envs/env_0/Object", "/World/envs/env_10/Object"]
    assert [match[3] for match in matches] == [tuple(range(10)), (10, 11)]


##
# Laws, checked over every plan shape above.
##


def _owning_row(plan: ClonePlan, path: str, env_id: int) -> int | None:
    """Independent oracle for the ownership rule stated in :mod:`isaaclab.cloner.query`.

    The deepest source root containing ``path`` owns it; among rows tying at that depth
    (the variants of one asset), the row populating ``env_id`` wins.
    """
    rows = [row for row, source in enumerate(plan.sources) if cloner.path.under(path, source)]
    if not rows:
        return None
    deepest = max(len(plan.sources[row].rstrip("/")) for row in rows)
    rows = [row for row in rows if len(plan.sources[row].rstrip("/")) == deepest]
    return next((row for row in rows if bool(plan.clone_mask[row][env_id])), None)


def _probe_paths(plan: ClonePlan) -> list[str]:
    """Source-space paths to probe: every prototype root, and prims below it."""
    return [source + tail for source in plan.sources for tail in ("", "/base", "/link/child")]


@pytest.mark.parametrize("plan_name", sorted(PLANS))
def test_query_law_factorization_and_domain(plan_name):
    """A clone keeps its source-relative suffix and exists only where its row says."""
    plan = PLANS[plan_name]
    for path in _probe_paths(plan):
        reached = cloner.query.path_env_ids(plan, path)
        for env_id in map(int, plan.env_ids):
            clone = cloner.query.path_to_clone(plan, path, env_id)
            assert (clone is not None) == (env_id in reached)
            if clone is None:
                continue
            row = _owning_row(plan, path, env_id)
            assert row is not None
            tail = cloner.path.relative_to(path, plan.sources[row])
            assert clone == plan.destinations[row].format(env_id) + tail


@pytest.mark.parametrize("plan_name", sorted(PLANS))
def test_query_law_round_trip(plan_name):
    """Q3: resolving a clone path returns the prototype it was cloned from.

    The clone paths are built from the oracle rather than from the module under test, so the
    round trip is checked against the ownership rule itself. A clone path is concrete, so no
    env id has to be supplied: the clone slot names it.
    """
    plan = PLANS[plan_name]

    for path in _probe_paths(plan):
        for env_id in range(plan.clone_mask.shape[1]):
            row = _owning_row(plan, path, env_id)
            if row is None:
                continue
            clone = cloner.path.rebase(path, plan.sources[row], plan.destinations[row].format(env_id))

            resolved = cloner.query.path_to_source(plan, clone)
            assert resolved is not None, f"{clone} did not resolve back for env {env_id}"
            source, _glob, suffix = resolved
            assert source + suffix == path
            # Naming the env explicitly must agree with reading it out of the path.
            assert cloner.query.path_to_source(plan, clone, env_id=env_id) == resolved


def test_query_resolve_distinguishes_concrete_paths_from_wildcards():
    """A concrete clone path names its env; a wildcard expression stands for all of them.

    The clone slot is what separates the two: ``env_2`` selects the variant that populates
    env 2, while ``env_.*`` cannot and falls back to the first populated variant unless the
    caller names an env.
    """
    plan = PLANS["two_variants"]
    concrete = "/World/envs/env_2/Object/base"

    # Concrete: resolves to the variant env 2 was actually cloned from.
    source, _glob, suffix = cloner.query.path_to_source(plan, concrete)
    assert source + suffix == "/World/envs/env_2/Object/base"

    # Wildcard: one-to-many, so it reports a representative variant...
    wildcard = "/World/envs/env_[^/]+/Object/base"
    source, _glob, suffix = cloner.query.path_to_source(plan, wildcard)
    assert source + suffix == "/World/envs/env_0/Object/base"

    # ...unless the caller names the env it means.
    source, _glob, suffix = cloner.query.path_to_source(plan, wildcard, env_id=2)
    assert source + suffix == "/World/envs/env_2/Object/base"


def test_query_translates_env_ids_through_the_plan():
    """Mask columns are not env ids: a plan targeting envs (2, 5) reports 2 and 5.

    Replication formats destinations with ``env_ids[column]``, so the
    queries have to agree with it rather than reporting column indices.
    """
    plan = ClonePlan(
        sources=("/World/envs/env_2/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.asarray([[True, True]], dtype=np.bool_),
        env_ids=np.asarray([2, 5], dtype=np.int64),
    )
    path = "/World/envs/env_2/Robot/base"

    assert cloner.query.path_env_ids(plan, path) == (2, 5)
    assert cloner.query.path_to_clone(plan, path, 5) == "/World/envs/env_5/Robot/base"
    assert cloner.query.path_to_clone(plan, path, 1) is None
    assert next(iter(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Robot")))[3] == (2, 5)

    source, _glob, suffix = cloner.query.path_to_source(plan, "/World/envs/env_5/Robot/base")
    assert source + suffix == path
    # Column indices are not environments: env 1 is not targeted by this plan.
    assert cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Robot", env_id=1) is None


@pytest.mark.parametrize("env_id", [-1, 4, 99])
def test_query_rejects_env_ids_outside_the_plan(env_id):
    """Out-of-range and negative ids resolve to nothing instead of wrapping the mask."""
    assert cloner.query.path_to_source(_robot_plan(), "/World/envs/env_[^/]+/Robot", env_id=env_id) is None


def test_a_global_row_does_not_disturb_the_environment_queries():
    """A shared asset remains queryable without changing the environment roots."""
    plan = _plan(
        ["/World/envs/env_0/Robot", "/World/ground"],
        ["/World/envs/env_{}/Robot", "/World/ground"],
        [[True, True, True], [False, False, False]],
    )

    assert cloner.query.env_root_paths(plan) == [f"/World/envs/env_{i}" for i in range(3)]
    # It is still resolvable as itself -- that is the point of keeping it in the plan. Source and
    # destination are the same path, so resolution is the identity plus the asset suffix.
    assert cloner.query.path_to_source(plan, "/World/ground/Plane") == (
        "/World/ground",
        "/World/ground",
        "/Plane",
    )
    # Nothing copies it, but every environment has it, so the query answers with the one prim for
    # all of them -- which is what lets a caller resolve a global asset without asking the stage.
    assert list(cloner.query.iter_sources(plan, "/World/ground/Plane")) == [
        ("/World/ground", "/World/ground", "/World/ground/Plane", (0, 1, 2))
    ]


def test_env_root_paths_reads_the_environments_off_the_plan():
    """A backend authoring something per env asks the plan where they are, not how they are named."""
    plan = _plan(
        ["/Scene/worlds/world_0/Robot"],
        ["/Scene/worlds/world_{}/Robot"],
        [[True, True, True]],
    )

    assert cloner.query.env_root_paths(plan) == [
        "/Scene/worlds/world_0",
        "/Scene/worlds/world_1",
        "/Scene/worlds/world_2",
    ]
    assert cloner.query.env_root_paths(None) == []


def test_query_and_path_are_real_modules():
    """``cloner.path``/``cloner.query`` import as modules, not just package attributes."""
    import isaaclab.cloner.path  # noqa: PLC0415
    import isaaclab.cloner.query  # noqa: PLC0415
    from isaaclab.cloner.path import under  # noqa: PLC0415
    from isaaclab.cloner.query import path_to_source  # noqa: PLC0415

    assert isaaclab.cloner.path.__name__ == "isaaclab.cloner.path"
    assert isaaclab.cloner.query.__name__ == "isaaclab.cloner.query"
    assert under is cloner.path.under
    assert path_to_source is cloner.query.path_to_source


def test_cloner_imports_without_kit():
    """Importing the package in a clean interpreter must not drag in pxr.

    ``isaaclab.sim.utils.queries`` imports the cloner and the cloner's plan constructors
    import ``isaaclab.sim``, so this guards both against an import cycle and against pulling
    pxr in before Kit boots, which corrupts Kit's own USD runtime.
    """
    probe = (
        "from isaaclab.cloner import ClonePlan; import sys; "
        "print(any(n == 'pxr' or n.startswith('pxr.') for n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing isaaclab.cloner pulled in pxr"
