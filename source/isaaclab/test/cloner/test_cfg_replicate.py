# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the per-cfg ``replicate`` field on :class:`AssetBaseCfg` and
:class:`SensorBaseCfg`.

Pure Python; importing the cfg modules does not boot Isaac Sim, so these tests run
in any standard ``./isaaclab.sh -p`` environment without the Kit application.
"""

from __future__ import annotations

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.cloner.usd_replicator import usd_replicate as _usd_replicate
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString


def _no_op_replicate(_cfg: object, _layout: object, _cfg_idx: int) -> None:
    """Sentinel callable used by override tests."""


def test_asset_base_cfg_replicate_default_resolves_to_usd_replicate():
    cfg = AssetBaseCfg(prim_path="/Foo")
    assert isinstance(cfg.replicate, tuple)
    assert len(cfg.replicate) == 1
    entry = cfg.replicate[0]
    assert isinstance(entry, ResolvableString)
    assert str(entry) == "isaaclab.cloner.usd_replicator:usd_replicate"
    assert entry._resolve() is _usd_replicate


def test_sensor_base_cfg_replicate_default_resolves_to_usd_replicate():
    cfg = SensorBaseCfg(prim_path="/Foo/Sensor")
    assert isinstance(cfg.replicate, tuple)
    assert len(cfg.replicate) == 1
    entry = cfg.replicate[0]
    assert isinstance(entry, ResolvableString)
    assert str(entry) == "isaaclab.cloner.usd_replicator:usd_replicate"
    assert entry._resolve() is _usd_replicate


def test_replicate_default_is_independent_per_instance():
    """Two cfg instances must not share a mutable replicate alias."""
    a = AssetBaseCfg(prim_path="/A")
    b = AssetBaseCfg(prim_path="/B")
    # Tuples themselves are immutable, but configclass wrap should not collapse them
    # into a single shared list-of-callables that overrides leak through.
    assert a.replicate == b.replicate
    assert tuple(str(x) for x in a.replicate) == ("isaaclab.cloner.usd_replicator:usd_replicate",)


def test_replicate_per_instance_override_with_callable():
    cfg = AssetBaseCfg(prim_path="/Foo", replicate=(_no_op_replicate,))
    assert cfg.replicate == (_no_op_replicate,)
    cfg.replicate[0](None, None, 0)


def test_replicate_per_instance_override_with_string_resolves():
    cfg = AssetBaseCfg(
        prim_path="/Foo",
        replicate=("isaaclab.cloner.usd_replicator:usd_replicate",),
    )
    entry = cfg.replicate[0]
    assert isinstance(entry, ResolvableString)
    assert entry._resolve() is _usd_replicate


def test_replicate_supports_multiple_entries_in_order():
    """Backend overlays compose by stacking entries; order must be preserved."""
    calls: list[str] = []

    def first(_cfg: object, _layout: object, _cfg_idx: int) -> None:
        calls.append("first")

    def second(_cfg: object, _layout: object, _cfg_idx: int) -> None:
        calls.append("second")

    cfg = AssetBaseCfg(prim_path="/Foo", replicate=(first, second))
    for fn in cfg.replicate:
        fn(cfg, None, 0)
    assert calls == ["first", "second"]


def test_replicate_subclass_override_changes_default():
    """Backend cfg subclasses (e.g. an ``ArticulationCfg``) override the default by
    redefining the field. New instances of the subclass pick up the new default;
    instances of the parent class continue to see the parent default.
    """

    @configclass
    class _SubCfg(AssetBaseCfg):
        replicate: tuple = (_no_op_replicate, "isaaclab.cloner.usd_replicator:usd_replicate")

    sub_cfg = _SubCfg(prim_path="/Foo")
    assert len(sub_cfg.replicate) == 2
    assert sub_cfg.replicate[0] is _no_op_replicate
    assert isinstance(sub_cfg.replicate[1], ResolvableString)

    parent_cfg = AssetBaseCfg(prim_path="/Foo")
    assert tuple(str(x) for x in parent_cfg.replicate) == ("isaaclab.cloner.usd_replicator:usd_replicate",)


def test_replicate_empty_tuple_is_legal_for_non_replicating_cfg():
    """Sensors with no USD prim of their own (e.g. PVA) override to ``()``."""
    cfg = SensorBaseCfg(prim_path="/Foo/Sensor", replicate=())
    assert cfg.replicate == ()
    for _ in cfg.replicate:
        raise AssertionError("empty replicate must yield no callables")
