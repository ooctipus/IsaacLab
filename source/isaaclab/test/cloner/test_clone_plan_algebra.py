# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cloner path/query algebra.

These exercise :mod:`isaaclab.cloner.path` and :mod:`isaaclab.cloner.query`, which are pure
string/tensor operations over a :class:`~isaaclab.cloner.ClonePlan`. They need no stage, no
simulator and no USD, so they live outside ``test/sim/``.
"""

import subprocess
import sys
from dataclasses import dataclass

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import _make_clone_plan, _make_valid_clone_combinations


@dataclass
class _SpawnCfg:
    prim_path: str
    spawn: object | None


##
# Path primitives.
##


def test_path_split():
    """Split clone destination templates around their clone slot."""
    assert cloner.path.split("/World/envs/env_{}/Robot") == ("/World/envs/env_", "/Robot")
    assert cloner.path.split("/World/scenes/{}/") == ("/World/scenes/", "")
    with pytest.raises(ValueError, match="exactly one"):
        cloner.path.split("/World/envs/env_0/Robot")
    # A second slot would survive into the suffix and break the later format call.
    with pytest.raises(ValueError, match="exactly one"):
        cloner.path.split("/World/envs/env_{}/Robot/{}")


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


def test_make_clone_plan_gives_each_asset_one_flat_row():
    """A homogeneous scene keeps per-asset ownership instead of collapsing into one env row."""
    robot = _SpawnCfg(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )
    table = _SpawnCfg(
        prim_path="/World/envs/env_[^/]+/Table",
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.1)),
    )

    plan = _make_clone_plan([robot, table], num_clones=3, env_spacing=1.0, device="cpu")

    assert plan.sources == ("/World/envs/env_0/Robot", "/World/envs/env_0/Table")
    assert plan.destinations == ("/World/envs/env_{}/Robot", "/World/envs/env_{}/Table")
    assert plan.clone_mask.tolist() == [[True, True, True], [True, True, True]]
    assert plan.cfg_rows == {id(robot): (0,), id(table): (1,)}


def test_make_clone_plan_reads_nested_direct_cfg_and_expands_its_namespace():
    """The plan discovers an asset declared as data inside a direct-environment cfg."""

    @dataclass
    class DirectCfg:
        asset: object

    asset = _SpawnCfg(prim_path="{ENV_REGEX_NS}/Object", spawn=sim_utils.SphereCfg(radius=0.1))

    plan = _make_clone_plan([DirectCfg(asset)], num_clones=2, env_spacing=1.0, device="cpu")

    assert plan.sources == ("/World/envs/env_0/Object",)
    assert plan.destinations == ("/World/envs/env_{}/Object",)
    assert plan.cfg_rows == {id(asset): (0,)}


def test_cfg_source_paths_preserves_inactive_variant_slots_without_mutating_cfg():
    """Inactive variants stay addressable without rewriting their configuration."""
    cfg = _SpawnCfg(
        prim_path="/World/envs/env_[^/]+/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.ConeCfg(), sim_utils.SphereCfg(), sim_utils.CuboidCfg()]
        ),
    )
    before = cfg.spawn.to_dict()

    plan = _make_clone_plan([cfg], num_clones=2, env_spacing=1.0, device="cpu")

    assert cloner.query.cfg_source_paths(plan, cfg) == (
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
        None,
    )
    assert cfg.spawn.to_dict() == before
    with pytest.raises(ValueError, match="owns no row"):
        cloner.query.cfg_source_paths(plan, object())

    mismatched = _SpawnCfg(prim_path="/World/envs/env_[^/]+/Other", spawn=None)
    plan.cfg_rows[id(mismatched)] = plan.cfg_rows[id(cfg)]
    with pytest.raises(ValueError, match="does not match clone-plan destination"):
        cloner.query.cfg_source_paths(plan, mismatched)


def test_valid_clone_combinations_encode_presence_variants_and_weights():
    """Legal combinations remain declarative inputs to the flat row assignment."""
    combinations = _make_valid_clone_combinations(
        ["robot", "object"],
        [1, 2],
        [
            cloner.InclusionSet(assets=["robot"], weight=1),
            cloner.InclusionSet(assets=["object"], weight=2),
        ],
    )

    assert combinations.tolist() == [
        [0, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [-1, 0],
        [-1, 1],
    ]


def test_global_cfg_without_spawner_owns_one_non_cloned_row():
    """A directly authored shared subtree is plan-owned even though nothing copies it."""
    terrain = AssetBaseCfg(prim_path="/World/Ground")

    plan = _make_clone_plan([terrain], num_clones=2, env_spacing=1.0, device="cpu")

    assert plan.sources == plan.destinations == ("/World/Ground",)
    assert plan.cfg_rows == {id(terrain): (0,)}
    assert cloner.query.cfg_source_paths(plan, terrain) == ("/World/Ground",)
    assert not bool(plan.clone_mask.any())


def test_plan_derives_cross_environment_collision_roots_from_cfg_data():
    """Only cfgs marked collision_group=-1 contribute exact plan-owned collision roots."""
    ground = AssetBaseCfg(prim_path="/World/Ground", collision_group=-1)
    light = AssetBaseCfg(prim_path="/World/Light", collision_group=0)
    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        collision_group=-1,
    )

    plan = _make_clone_plan([ground, light, obstacle], num_clones=2, env_spacing=1.0, device="cpu")

    assert plan.collision_paths == (
        "/World/Ground",
        "/World/envs/env_0/Obstacle",
        "/World/envs/env_1/Obstacle",
    )


def test_plan_rejects_global_regex_authoring_and_ignores_regex_references():
    """Authoring needs an exact destination; a reference expression owns no independent row."""
    author = _SpawnCfg(
        prim_path="/World/Origin.*/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )
    reference = _SpawnCfg(prim_path="/World/Origin.*/Robot/base", spawn=None)

    with pytest.raises(ValueError, match="must use an exact prim path or the session env template"):
        _make_clone_plan([author], num_clones=2, env_spacing=1.0, device="cpu")

    plan = _make_clone_plan([reference], num_clones=2, env_spacing=1.0, device="cpu")
    assert plan.sources == plan.destinations == ()


def test_make_clone_plan_rejects_duplicate_scene_ownership():
    """Two authoring cfgs cannot claim the same physical destination."""
    cfgs = [
        _SpawnCfg(
            prim_path="/World/envs/env_[^/]+/Object",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        )
        for _ in range(2)
    ]

    with pytest.raises(ValueError, match="Multiple cfgs author the same clone-plan destination"):
        _make_clone_plan(cfgs, num_clones=2, env_spacing=1.0, device="cpu")


def test_nested_cfg_reuses_parent_rows_without_duplicate_copy():
    """A declared child resolves every parent prototype without adding clone rows."""
    robot = _SpawnCfg(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[sim_utils.ConeCfg(), sim_utils.SphereCfg()]),
    )
    camera = _SpawnCfg(
        prim_path="/World/envs/env_[^/]+/Robot/Camera",
        spawn=sim_utils.PinholeCameraCfg(),
    )

    plan = _make_clone_plan([camera, robot], num_clones=2, env_spacing=1.0, device="cpu")

    assert plan.destinations == ("/World/envs/env_{}/Robot",) * 2
    assert plan.cfg_rows[id(camera)] == plan.cfg_rows[id(robot)] == (0, 1)
    assert cloner.query.cfg_source_paths(plan, camera) == (
        "/World/envs/env_0/Robot/Camera",
        "/World/envs/env_1/Robot/Camera",
    )
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/Camera", 0) == ("/World/envs/env_0/Robot/Camera")
    assert cloner.query.path_to_clone(plan, "/World/envs/env_1/Robot/Camera", 1) == ("/World/envs/env_1/Robot/Camera")
    assert cloner.query.path_to_source(plan, "/World/envs/env_1/Robot/Camera") == (
        "/World/envs/env_1/Robot",
        "/World/envs/env_[^/]+/Robot",
        "/Camera",
    )


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


def _plan(sources, destinations, mask, env_ids=None, env_template="/World/envs/env_{}") -> ClonePlan:
    clone_mask = torch.tensor(mask, dtype=torch.bool)
    return ClonePlan(
        sources=tuple(sources),
        destinations=tuple(destinations),
        clone_mask=clone_mask,
        env_ids=torch.arange(clone_mask.shape[1]) if env_ids is None else torch.tensor(env_ids, dtype=torch.long),
        env_template=env_template,
    )


def _robot_plan(mask_row=(True, True, True, True)) -> ClonePlan:
    """A single-asset (Robot) clone plan over 4 envs with the given mask row."""
    return _plan(("/World/envs/env_0/Robot",), ("/World/envs/env_{}/Robot",), [list(mask_row)])


def _wide_env_id_plan() -> ClonePlan:
    """Two variants of one asset over 12 envs, the second starting at a two-digit env id."""
    mask = torch.zeros((2, 12), dtype=torch.bool)
    mask[0, :10] = True
    mask[1, 10:] = True
    return ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_10/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=mask,
        env_ids=torch.arange(12),
    )


PLANS = {
    "homogeneous": _robot_plan(),
    "partial_coverage": _robot_plan((True, True, False, True)),
    "two_variants": _plan(
        ("/World/envs/env_0/Object", "/World/envs/env_2/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[True, True, False, True], [False, False, True, False]],
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
    """path_env_ids returns the environments a source-space path reaches."""
    assert cloner.query.path_env_ids(_robot_plan(), "/World/envs/env_0/Robot/base") == (0, 1, 2, 3)
    partial = _robot_plan((True, True, False, True))
    assert cloner.query.path_env_ids(partial, "/World/envs/env_0/Robot/base") == (0, 1, 3)
    assert cloner.query.path_env_ids(_robot_plan(), "/World/ground") == ()


def test_path_to_clone():
    """path_to_clone returns a source path's clone, or None when unowned / not mapped."""
    plan = _robot_plan((True, True, False, True))
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", 3) == "/World/envs/env_3/Robot/base"
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", 2) is None
    assert cloner.query.path_to_clone(plan, "/World/ground", 0) is None


def test_path_to_clone_heterogeneous_selects_env_owning_row():
    """In a heterogeneous plan, a source path only reaches envs its own variant row populates."""
    plan = PLANS["two_variants"]
    # env 2 draws Object from the row-1 variant, so the row-0 prototype does not reach it.
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Object/base", 2) is None
    assert cloner.query.path_to_clone(plan, "/World/envs/env_2/Object/base", 2) == "/World/envs/env_2/Object/base"


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
    """Q1/Q2: a clone keeps everything below the prototype root, and exists only where the plan says."""
    plan = PLANS[plan_name]
    num_envs = plan.clone_mask.shape[1]

    for path in _probe_paths(plan):
        reached = cloner.query.path_env_ids(plan, path)
        assert set(reached) <= set(range(num_envs))

        for env_id in range(num_envs):
            clone = cloner.query.path_to_clone(plan, path, env_id)
            # Q2: a clone exists exactly for the environments the plan reaches.
            assert (clone is not None) == (env_id in reached)
            if clone is None:
                continue
            # Q1: the clone is the row's destination with the source-relative tail appended.
            row = _owning_row(plan, path, env_id)
            assert row is not None
            tail = cloner.path.relative_to(path, plan.sources[row])
            assert clone == plan.destinations[row].format(env_id) + tail
            assert clone == cloner.path.rebase(path, plan.sources[row], plan.destinations[row].format(env_id))


@pytest.mark.parametrize("plan_name", sorted(PLANS))
def test_query_law_round_trip(plan_name):
    """Q3: resolving a clone path returns the prototype it was cloned from.

    A clone path is concrete, so no env id has to be supplied: the clone slot names it.
    """
    plan = PLANS[plan_name]

    for path in _probe_paths(plan):
        for env_id in cloner.query.path_env_ids(plan, path):
            clone = cloner.query.path_to_clone(plan, path, env_id)
            assert clone is not None

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

    Clone dispatch formats destinations with ``env_ids[column]``, so queries must agree with
    it rather than reporting column indices.
    """
    plan = ClonePlan(
        sources=("/World/envs/env_2/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.tensor([[True, True]], dtype=torch.bool),
        env_ids=torch.tensor([2, 5], dtype=torch.long),
    )
    path = "/World/envs/env_2/Robot/base"

    assert cloner.query.path_env_ids(plan, path) == (2, 5)
    assert cloner.query.path_to_clone(plan, path, 5) == "/World/envs/env_5/Robot/base"
    # Column indices are not environments: env 1 is not targeted by this plan.
    assert cloner.query.path_to_clone(plan, path, 1) is None
    assert next(iter(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Robot")))[3] == (2, 5)

    source, _glob, suffix = cloner.query.path_to_source(plan, "/World/envs/env_5/Robot/base")
    assert source + suffix == path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_query_gathers_env_ids_across_devices():
    """Environment ids can remain on CPU when the clone mask is on CUDA."""
    plan = ClonePlan(
        sources=("/World/envs/env_2/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.tensor([[True, False, True]], dtype=torch.bool, device="cuda"),
        env_ids=torch.tensor([2, 5, 8], dtype=torch.long),
    )
    path = "/World/envs/env_2/Robot/base"

    assert cloner.query.path_env_ids(plan, path) == (2, 8)
    assert next(iter(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Robot")))[3] == (2, 8)


@pytest.mark.parametrize("env_id", [-1, 4, 99])
def test_query_rejects_env_ids_outside_the_plan(env_id):
    """Out-of-range and negative ids resolve to nothing instead of wrapping the mask."""
    plan = _robot_plan()
    assert cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", env_id) is None
    assert cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Robot", env_id=env_id) is None


def test_query_agrees_across_duplicate_source_rows():
    """An inactive variant may share a fallback source path with an active one.

    ``_make_clone_plan`` points an unused variant at ``destination.format(i)``, which can
    collide with another row's source. The two source-side queries must still agree about
    which envs that path reaches.
    """
    plan = _plan(
        # Row 1 is inactive and fell back to the same source path as active row 0.
        ("/World/envs/env_0/Object", "/World/envs/env_0/Object"),
        ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        [[True, False, True, False], [False, False, False, False]],
    )
    path = "/World/envs/env_0/Object/base"

    reached = cloner.query.path_env_ids(plan, path)
    assert reached == (0, 2)
    for env_id in range(plan.clone_mask.shape[1]):
        assert (cloner.query.path_to_clone(plan, path, env_id) is not None) == (env_id in reached)


def test_environment_queries_read_flat_rows_and_ignore_globals():
    """Environment collapse derives its roots from cloned rows, not shared world rows."""
    plan = _plan(
        ["/World/envs/env_0/Robot", "/World/envs/env_0/Table", "/World/Ground"],
        ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Table", "/World/Ground"],
        [[True, True, True], [True, True, True], [False, False, False]],
        env_ids=[0, 1, 2],
    )

    assert cloner.query.clone_rows(plan) == [0, 1]
    assert cloner.query.env_root_paths(plan) == [f"/World/envs/env_{env_id}" for env_id in range(3)]
    assert cloner.query.whole_env_copy(plan) == ("/World/envs/env_0", "/World/envs/env_{}")


def test_global_rows_resolve_without_a_clone_slot():
    """A shared plan-owned subtree resolves directly instead of entering template parsing."""
    plan = _plan(
        ["/World/Ground"],
        ["/World/Ground"],
        [[False, False]],
        env_ids=[0, 1],
    )

    assert cloner.query.path_to_source(plan, "/World/Ground/Collision") == (
        "/World/Ground",
        "/World/Ground",
        "/Collision",
    )
    assert list(cloner.query.iter_sources(plan, "/World/Ground/Collision")) == [
        ("/World/Ground", "/World/Ground", "/World/Ground/Collision", (0, 1))
    ]


@pytest.mark.parametrize(
    "sources, mask",
    [
        (["/World/envs/env_0/Robot", "/World/envs/env_0/Table"], [[True, True], [True, False]]),
        (["/World/envs/env_0/Robot", "/World/envs/env_1/Table"], [[True, True], [True, True]]),
    ],
)
def test_whole_env_copy_refuses_non_equivalent_flat_plans(sources, mask):
    """Partial coverage and prototypes outside the source env require per-asset copies."""
    destinations = ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Table"]
    plan = _plan(sources, destinations, mask, env_ids=[0, 1])

    assert cloner.query.whole_env_copy(plan) is None


def test_env_root_paths_respects_custom_template():
    """Consumers ask the plan for custom environment roots instead of assuming names."""
    plan = _plan(
        ["/Scene/worlds/world_4/Robot"],
        ["/Scene/worlds/world_{}/Robot"],
        [[True, True]],
        env_ids=[4, 7],
        env_template="/Scene/worlds/world_{}",
    )

    assert cloner.query.env_root_paths(plan) == ["/Scene/worlds/world_4", "/Scene/worlds/world_7"]


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
    probe = "import isaaclab.cloner, sys; print(any(n == 'pxr' or n.startswith('pxr.') for n in sys.modules))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing isaaclab.cloner pulled in pxr"
