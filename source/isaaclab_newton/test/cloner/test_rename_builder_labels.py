# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton clone label rewriting and visualization clone-plan sources."""

import unittest
from types import SimpleNamespace
from unittest import mock

import newton
import torch
from isaaclab_newton.cloner import newton_clone_utils as newton_clone_utils_module
from isaaclab_newton.cloner import replicate as replicate_module
from isaaclab_newton.cloner.newton_clone_utils import (
    _BUILTIN_LABEL_TYPES,
    rename_builder_labels,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager, NewtonMJWarpManager
from isaaclab_newton.physics import visualization_builder as visualization_builder_module
from newton.solvers import SolverMuJoCo

from pxr import Usd, UsdGeom

from isaaclab.cloner import ClonePlan

_VIS_LABEL_SUFFIXES = {
    "body_label": "Body",
    "joint_label": "Joint",
    "shape_label": "Shape",
    "articulation_label": "Articulation",
    "constraint_mimic_label": "ConstraintMimic",
    "equality_constraint_label": "EqualityConstraint",
}
_VIS_LABEL_ATTRS = tuple(_VIS_LABEL_SUFFIXES)

_TENDON_FREQ = "mujoco:tendon"
_SRC = "/Sources/protoA"
_DST = "/World/envs/env_{}"


class _FakeVisualizationModelBuilder:
    def __init__(self, up_axis=None):
        self.up_axis = up_axis
        for attr in _VIS_LABEL_ATTRS:
            setattr(self, attr, [])
            setattr(self, attr.replace("_label", "_world"), [])
        self.custom_attributes = {}
        self.geometry_sources = []
        self.world_slices = []
        self._current_world = None

    @property
    def shape_count(self):
        return len(self.shape_label)

    def begin_world(self):
        self._current_world = len(self.world_slices)
        self.world_slices.append([])

    def end_world(self):
        self._current_world = None

    def add_usd(self, stage, root_path=None, ignore_paths=None, schema_resolvers=None, **kwargs):
        del stage, ignore_paths, schema_resolvers, kwargs
        if root_path is None:
            return
        label_start = len(self.body_label)
        geometry_start = len(self.geometry_sources)
        for attr, suffix in _VIS_LABEL_SUFFIXES.items():
            getattr(self, attr).append(f"{root_path}/{suffix}")
            getattr(self, attr.replace("_label", "_world")).append(self._current_world or 0)
        self.geometry_sources.append(root_path)
        self._record_world_slice(label_start, len(self.body_label), geometry_start, len(self.geometry_sources))

    def add_builder(self, builder, xform=None):
        del xform
        label_start = len(self.body_label)
        geometry_start = len(self.geometry_sources)
        for attr in _VIS_LABEL_ATTRS:
            labels = getattr(builder, attr)
            getattr(self, attr).extend(labels)
            getattr(self, attr.replace("_label", "_world")).extend([self._current_world] * len(labels))
        self.geometry_sources.extend(builder.geometry_sources)
        self._record_world_slice(label_start, len(self.body_label), geometry_start, len(self.geometry_sources))

    def labels_for_world(self, world_id, attr):
        labels = getattr(self, attr)
        return [label for start, end, _, _ in self.world_slices[world_id] for label in labels[start:end]]

    def geometry_sources_for_world(self, world_id):
        return [
            source for _, _, start, end in self.world_slices[world_id] for source in self.geometry_sources[start:end]
        ]

    def _record_world_slice(self, label_start, label_end, geometry_start, geometry_end):
        if self._current_world is not None:
            self.world_slices[self._current_world].append((label_start, label_end, geometry_start, geometry_end))


def _inject_builtins(builder: newton.ModelBuilder, types: tuple[str, ...], src_path: str, worlds: list[int]) -> None:
    for kind in types:
        for world in worlds:
            if kind == "equality_constraint":
                builder.add_custom_values(
                    **{
                        "mujoco:equality_constraint_label": f"{src_path}/{kind}_{world}",
                        "mujoco:equality_constraint_world": world,
                    }
                )
            else:
                getattr(builder, f"{kind}_label").append(f"{src_path}/{kind}_{world}")
                getattr(builder, f"{kind}_world").append(world)


def _inject_tendons(builder: newton.ModelBuilder, src_path: str, worlds: list[int]) -> None:
    labels = builder.custom_attributes["mujoco:tendon_label"].values = []
    world_ids = builder.custom_attributes["mujoco:tendon_world"].values = []
    for world in worlds:
        labels.append(f"{src_path}/Tendon_{world}")
        world_ids.append(world)
    builder._custom_frequency_counts[_TENDON_FREQ] = len(worlds)


def _make_builder(worlds: list[int]) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    _inject_builtins(builder, _BUILTIN_LABEL_TYPES, _SRC, worlds)
    _inject_tendons(builder, _SRC, worlds)
    return builder


def _add_custom_frequency(builder, freq_name, string_columns):
    freq = f"syn:{freq_name}"
    builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=freq_name, namespace="syn"))
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name=f"{freq_name}_world", frequency=freq, dtype=int, default=0, namespace="syn", references="world"
        )
    )
    for column in string_columns:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(name=column, frequency=freq, dtype=str, default="", namespace="syn")
        )


def _populate_custom_frequency(builder, freq_name, string_columns, worlds):
    builder.custom_attributes[f"syn:{freq_name}_world"].values = list(worlds)
    for column in string_columns:
        builder.custom_attributes[f"syn:{column}"].values = [f"{_SRC}/{column}_{world}" for world in worlds]
    builder._custom_frequency_counts[f"syn:{freq_name}"] = len(worlds)


class TestRenameBuilderLabels(unittest.TestCase):
    def setUp(self):
        self.worlds = [0, 1, 2]
        self.env_ids = torch.tensor(self.worlds, dtype=torch.int32)
        self.mapping = torch.ones(1, len(self.worlds), dtype=torch.bool)

    def _rename(self, builder):
        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

    def _assert_builtins(self, builder, types=_BUILTIN_LABEL_TYPES):
        for kind in types:
            if kind == "equality_constraint":
                labels = builder.custom_attributes["mujoco:equality_constraint_label"].values
                worlds = builder.custom_attributes["mujoco:equality_constraint_world"].values
            else:
                labels = getattr(builder, f"{kind}_label")
                worlds = getattr(builder, f"{kind}_world")
            self.assertEqual(
                labels,
                [f"{_DST.format(int(w))}/{kind}_{int(w)}" for w in worlds],
            )

    def test_builtin_and_tendon_labels_rewritten_per_world(self):
        builder = _make_builder(self.worlds)
        self._rename(builder)
        self._assert_builtins(builder)
        tendon_worlds = builder.custom_attributes["mujoco:tendon_world"].values
        self.assertEqual(
            builder.custom_attributes["mujoco:tendon_label"].values,
            [f"{_DST.format(int(w))}/Tendon_{int(w)}" for w in tendon_worlds],
        )

    def test_source_root_boundary_cases(self):
        builder = _make_builder(self.worlds)
        builder.body_label.append(_SRC)
        builder.body_world.append(self.worlds[0])
        self._rename(builder)
        self.assertEqual(builder.body_label[-1], _DST.format(self.worlds[0]))

        builder = _make_builder(self.worlds)
        rename_builder_labels(builder, [f"{_SRC}/"], [_DST], self.env_ids, self.mapping)
        self._assert_builtins(builder)

    def test_unmatched_rows_left_untouched(self):
        builder = _make_builder(self.worlds)
        builder.body_label.append(f"{_SRC}/body_99")
        builder.body_world.append(99)
        builder.custom_attributes["mujoco:tendon_label"].values.append("named_tendon")
        builder.custom_attributes["mujoco:tendon_world"].values.append(self.worlds[0])
        self._rename(builder)
        self.assertEqual(builder.body_label[-1], f"{_SRC}/body_99")
        self.assertEqual(builder.custom_attributes["mujoco:tendon_label"].values[-1], "named_tendon")

    def test_sparse_env_ids(self):
        for worlds in ([10, 20, 30], [0, 1_000_000, 2_147_000_000]):
            builder = newton.ModelBuilder()
            SolverMuJoCo.register_custom_attributes(builder)
            _inject_builtins(builder, ("body",), _SRC, worlds)
            env_ids = torch.tensor(worlds, dtype=torch.int32)
            rename_builder_labels(builder, [_SRC], [_DST], env_ids, torch.ones(1, len(worlds), dtype=torch.bool))
            self._assert_builtins(builder, ("body",))


class TestRenameCustomAttributes(unittest.TestCase):
    def setUp(self):
        self.worlds = [0, 1]
        self.env_ids = torch.tensor(self.worlds, dtype=torch.int32)
        self.mapping = torch.ones(1, len(self.worlds), dtype=torch.bool)

    def test_custom_string_columns_follow_frequency_worlds(self):
        builder = newton.ModelBuilder()
        _add_custom_frequency(builder, "freqA", ["freqA_label", "freqA_alt"])
        _add_custom_frequency(builder, "freqB", ["freqB_label"])
        _populate_custom_frequency(builder, "freqA", ["freqA_label", "freqA_alt"], self.worlds)
        _populate_custom_frequency(builder, "freqB", ["freqB_label"], self.worlds)
        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

        for freq, columns in {"freqA": ("freqA_label", "freqA_alt"), "freqB": ("freqB_label",)}.items():
            worlds = builder.custom_attributes[f"syn:{freq}_world"].values
            for column in columns:
                self.assertEqual(
                    builder.custom_attributes[f"syn:{column}"].values,
                    [f"{_DST.format(int(w))}/{column}_{int(w)}" for w in worlds],
                )

    def test_empty_custom_string_column_passes_through(self):
        builder = newton.ModelBuilder()
        _add_custom_frequency(builder, "freqA", ["freqA_label"])
        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)
        _populate_custom_frequency(builder, "freqA", ["freqA_label"], self.worlds)
        self.assertEqual(len(builder.custom_attributes["syn:freqA_label"].values), len(self.worlds))

    def test_custom_string_columns_ignore_unset_world_rows(self):
        builder = newton.ModelBuilder()
        _add_custom_frequency(builder, "freqA", ["freqA_label"])
        builder.custom_attributes["syn:freqA_world"].values = [None, self.worlds[0]]
        builder.custom_attributes["syn:freqA_label"].values = ["unassigned", f"{_SRC}/freqA_label_{self.worlds[0]}"]
        builder._custom_frequency_counts["syn:freqA"] = 2

        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

        self.assertEqual(
            builder.custom_attributes["syn:freqA_label"].values,
            ["unassigned", f"{_DST.format(self.worlds[0])}/freqA_label_{self.worlds[0]}"],
        )


class TestRenameMultiSource(unittest.TestCase):
    def test_prefix_overlap_does_not_cross_contaminate(self):
        sources = ["/Sources/protoA", "/Sources/protoAB"]
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.body_label.extend([f"{sources[0]}/body", f"{sources[1]}/body"] * 2)
        builder.body_world.extend([0, 0, 1, 1])
        rename_builder_labels(
            builder,
            sources,
            ["/World/envs/env_{}", "/World/envs/env_{}"],
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([[1, 1], [1, 1]], dtype=torch.bool),
        )
        self.assertEqual(
            builder.body_label,
            ["/World/envs/env_0/body", "/World/envs/env_0/body", "/World/envs/env_1/body", "/World/envs/env_1/body"],
        )


class TestReplicateBuilderMapping(unittest.TestCase):
    @staticmethod
    def _source_builder(root_path: str):
        builder = _FakeVisualizationModelBuilder()
        builder.add_usd(None, root_path=root_path)
        return builder

    def test_inactive_source_rows_are_ignored(self):
        sources = ("/Sources/inactive", "/Sources/active")
        source_builders = {source: self._source_builder(source) for source in sources}
        builder = _FakeVisualizationModelBuilder()

        replicate_builder_mapping(
            builder,
            sources,
            torch.tensor([[False, False], [True, False]], dtype=torch.bool),
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            source_builders,
        )

        self.assertEqual(builder.geometry_sources_for_world(0), ["/Sources/active"])
        self.assertEqual(builder.geometry_sources_for_world(1), [])


class TestCloneBuilderImportOwnership(unittest.TestCase):
    def test_scoped_custom_frequency_filters_match_ignore_paths_and_restore(self):
        calls = []
        context = object()

        class _Prim:
            def __init__(self, path):
                self.path = path

            def GetPath(self):
                return self.path

        def original_filter(prim, received_context):
            calls.append((str(prim.GetPath()), received_context))
            return True

        frequency = SimpleNamespace(usd_prim_filter=original_filter)
        null_frequency = SimpleNamespace(usd_prim_filter=None)

        class _Builder:
            custom_frequencies = {"active": frequency, "inactive": null_frequency}

            def add_usd(self, stage, *, ignore_paths, **kwargs):
                del stage, kwargs
                self.filter_during_import = frequency.usd_prim_filter
                self.ignore_paths = ignore_paths
                return [
                    path
                    for path in ("/World/envs/env_0/item", "/World/envsibling/item", "/World/global/item")
                    if frequency.usd_prim_filter(_Prim(path), context)
                ]

        builder = _Builder()
        accepted = newton_clone_utils_module.add_usd_with_scoped_custom_frequencies(
            builder,
            object(),
            ignore_paths=("/World/envs",),
        )

        self.assertEqual(accepted, ["/World/global/item"])
        self.assertEqual(builder.ignore_paths, ("/World/envs",))
        self.assertIsNot(builder.filter_during_import, original_filter)
        self.assertIs(frequency.usd_prim_filter, original_filter)
        self.assertIsNone(null_frequency.usd_prim_filter)
        self.assertEqual(calls, [("/World/global/item", context)])

    def test_scoped_custom_frequency_filters_restore_after_import_failure(self):
        def original_filter(prim, context):
            del prim, context
            return True

        frequency = SimpleNamespace(usd_prim_filter=original_filter)

        class _Builder:
            custom_frequencies = {"active": frequency}

            def add_usd(self, stage, **kwargs):
                del stage, kwargs
                self.filter_during_import = frequency.usd_prim_filter
                raise RuntimeError("import failed")

        builder = _Builder()
        with self.assertRaisesRegex(RuntimeError, "import failed"):
            newton_clone_utils_module.add_usd_with_scoped_custom_frequencies(
                builder,
                object(),
                ignore_paths=("/World/envs",),
            )

        self.assertIsNot(builder.filter_during_import, original_filter)
        self.assertIs(frequency.usd_prim_filter, original_filter)

    def test_global_import_keeps_mujoco_shape_fields_and_excludes_clone_rows(self):
        class _Prim:
            def __init__(self, path):
                self.path = path

            def GetPath(self):
                return self.path

        def suffix_filter(suffix):
            return lambda prim, context: str(prim.GetPath()).endswith(suffix)

        original_filters = {
            "mujoco:actuator": suffix_filter("/actuator"),
            "mujoco:equality_constraint": suffix_filter("/equality"),
            "test:global": suffix_filter("/global_row"),
        }

        class _Builder:
            def __init__(self):
                self.custom_frequencies = {
                    name: SimpleNamespace(usd_prim_filter=callback) for name, callback in original_filters.items()
                }
                self._custom_frequency_counts = dict.fromkeys(original_filters, 0)
                self.custom_attributes = {}
                self.add_usd_kwargs = None
                self.geom_margin = None
                self.geom_solimp = None
                self.geom_solref = None

            def add_usd(self, stage, **kwargs):
                del stage
                self.add_usd_kwargs = kwargs
                self.assert_mujoco_resolver = any(
                    type(resolver).__name__ == "SchemaResolverMjc" for resolver in kwargs["schema_resolvers"]
                )
                if self.assert_mujoco_resolver:
                    self.geom_margin = 0.001
                    self.geom_solimp = (0.99, 0.99, 0.003, 0.5, 2.0)
                    self.geom_solref = (0.015, 1.0)
                context = {"builder": self}
                for path in (
                    "/World/envs/env_0/Robot/actuator",
                    "/World/envs/env_0/Robot/equality",
                    "/World/global/global_row",
                ):
                    prim = _Prim(path)
                    for name, frequency in self.custom_frequencies.items():
                        if frequency.usd_prim_filter(prim, context):
                            self._custom_frequency_counts[name] += 1
                return {"stage": "global"}

        builder = _Builder()
        source_builder = SimpleNamespace(_custom_frequency_counts={"mujoco:actuator": 69})
        source_builders = {"/World/envs/env_0": source_builder}

        class _Manager(NewtonMJWarpManager):
            @classmethod
            def create_builder(cls, **kwargs):
                self.assertEqual(kwargs, {"up_axis": "Z"})
                return builder

        mapping = torch.zeros((1, 1), dtype=torch.bool)
        with (
            mock.patch.object(replicate_module.PhysicsManager, "_sim", SimpleNamespace(physics_manager=_Manager)),
            mock.patch.object(
                replicate_module.PhysicsManager,
                "_cfg",
                NewtonCfg(solver_cfg=MJWarpSolverCfg()),
            ),
            mock.patch.object(replicate_module, "replace_newton_builder_shape_colors"),
            mock.patch.object(replicate_module, "build_source_builders", return_value=source_builders) as build_sources,
            mock.patch.object(replicate_module, "replicate_builder_mapping", return_value=({}, [])),
            mock.patch.object(replicate_module.NewtonManager, "_deformable_registry", ()),
            mock.patch.object(replicate_module.NewtonManager, "_post_replicate_hooks", ()),
            mock.patch.object(replicate_module.NewtonManager, "_per_world_builder_hooks", ()),
            mock.patch.object(replicate_module.NewtonManager, "_cl_inject_sites", return_value=({}, {}, {})),
            mock.patch(
                "isaaclab_newton.physics.newton_manager.add_native_mjcf_from_stage",
                return_value=(),
            ),
        ):
            result, _, _, _, returned_sources = replicate_module._build_newton_builder_from_mapping(
                stage=object(),
                sources=("/World/envs/env_0",),
                destinations=("/World/envs/env_{}",),
                env_ids=torch.tensor((0,), dtype=torch.int64),
                mapping=mapping,
            )

        self.assertIs(result, builder)
        self.assertTrue(builder.assert_mujoco_resolver)
        self.assertIs(build_sources.call_args.args[3].__func__, _Manager._import_stage.__func__)
        self.assertFalse(builder.add_usd_kwargs["convert_mjc_equality_constraints"])
        self.assertEqual(builder.geom_margin, 0.001)
        self.assertEqual(builder.geom_solimp, (0.99, 0.99, 0.003, 0.5, 2.0))
        self.assertEqual(builder.geom_solref, (0.015, 1.0))
        self.assertEqual(builder._custom_frequency_counts["mujoco:actuator"], 0)
        self.assertEqual(builder._custom_frequency_counts["mujoco:equality_constraint"], 0)
        self.assertEqual(builder._custom_frequency_counts["test:global"], 1)
        self.assertEqual(returned_sources["/World/envs/env_0"]._custom_frequency_counts["mujoco:actuator"], 69)
        for name, callback in original_filters.items():
            self.assertIs(builder.custom_frequencies[name].usd_prim_filter, callback)

    def test_non_mujoco_solver_keeps_newton_equality_conversion(self):
        builder = SimpleNamespace(custom_frequencies={}, add_usd=mock.Mock(return_value={}), up_axis="Z")

        class _Manager(NewtonManager):
            @classmethod
            def create_builder(cls, **kwargs):
                self.assertEqual(kwargs, {"up_axis": "Z"})
                return builder

        build_sources = mock.Mock(return_value={})
        with (
            mock.patch.object(replicate_module.PhysicsManager, "_sim", SimpleNamespace(physics_manager=_Manager)),
            mock.patch.object(replicate_module.PhysicsManager, "_cfg", object()),
            mock.patch.object(replicate_module, "replace_newton_builder_shape_colors"),
            mock.patch.object(replicate_module, "build_source_builders", build_sources),
            mock.patch.object(replicate_module, "replicate_builder_mapping", return_value=({}, [])),
            mock.patch.object(replicate_module.NewtonManager, "_deformable_registry", ()),
            mock.patch.object(replicate_module.NewtonManager, "_post_replicate_hooks", ()),
            mock.patch.object(replicate_module.NewtonManager, "_per_world_builder_hooks", ()),
            mock.patch.object(replicate_module.NewtonManager, "_cl_inject_sites", return_value=({}, {}, {})),
            mock.patch(
                "isaaclab_newton.physics.newton_manager.add_native_mjcf_from_stage",
                return_value=(),
            ),
        ):
            replicate_module._build_newton_builder_from_mapping(
                stage=object(),
                sources=("/World/envs/env_0",),
                destinations=("/World/envs/env_{}",),
                env_ids=torch.tensor((0,), dtype=torch.int64),
                mapping=torch.zeros((1, 1), dtype=torch.bool),
            )

        self.assertTrue(builder.add_usd.call_args.kwargs["convert_mjc_equality_constraints"])
        self.assertIs(build_sources.call_args.args[3].__func__, _Manager._import_stage.__func__)

    def test_source_builder_registration_belongs_to_factory(self):
        builder = SimpleNamespace(add_usd=mock.Mock(), up_axis="Z")
        import_stage = mock.Mock()
        with (
            mock.patch.object(newton.solvers.SolverMuJoCo, "register_custom_attributes") as register,
            mock.patch.object(newton_clone_utils_module, "replace_newton_builder_shape_colors"),
        ):
            result = newton_clone_utils_module.build_source_builders(
                stage=object(),
                sources=("/World/envs/env_0",),
                create_builder=lambda: builder,
                import_stage=import_stage,
                simplify_meshes=False,
            )

        self.assertEqual(result, {"/World/envs/env_0": builder})
        register.assert_not_called()

        import_stage.assert_called_once()

    def test_registered_mujoco_source_builder_preserves_native_equality(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/envs")
        UsdGeom.Xform.Define(stage, "/World/envs/env_0")

        def create_builder():
            builder = newton.ModelBuilder()
            SolverMuJoCo.register_custom_attributes(builder)
            return builder

        with mock.patch.object(newton_clone_utils_module, "replace_newton_builder_shape_colors"):
            builders = newton_clone_utils_module.build_source_builders(
                stage=stage,
                sources=("/World/envs/env_0",),
                create_builder=create_builder,
                import_stage=NewtonMJWarpManager._import_stage,
                simplify_meshes=False,
            )

        builder = builders["/World/envs/env_0"]
        self.assertIn("mujoco:actuator", builder.custom_frequencies)
        self.assertIn("mujoco:equality_constraint", builder.custom_frequencies)


class TestVisualizationClonePlan(unittest.TestCase):
    @staticmethod
    def _define_xform(stage, path, translation=None):
        xform = UsdGeom.Xform.Define(stage, path)
        if translation is not None:
            xform.AddTranslateOp().Set(translation)

    def test_visualization_builder_uses_clone_plan_sources_and_rewrites_labels(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self._define_xform(stage, "/World")
        self._define_xform(stage, "/World/envs")
        env_paths = [(env_id, f"/World/envs/env_{env_id}") for env_id in (0, 1, 2)]
        for env_id, env_path in env_paths:
            self._define_xform(stage, env_path, (float(env_id) * 3.0, 0.0, 0.0))
            self._define_xform(stage, f"{env_path}/Object")
        self._define_xform(stage, "/World/envs/env_0/Object/source_0_visual")
        self._define_xform(stage, "/World/envs/env_1/Object/source_1_visual")

        clone_plan = ClonePlan(
            sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
            destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
            clone_mask=torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool),
            env_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        )

        with (
            mock.patch.object(visualization_builder_module, "ModelBuilder", _FakeVisualizationModelBuilder),
            mock.patch.object(newton_clone_utils_module, "ModelBuilder", _FakeVisualizationModelBuilder),
            mock.patch.object(visualization_builder_module, "SchemaResolverNewton", lambda: object()),
            mock.patch.object(visualization_builder_module, "SchemaResolverPhysx", lambda: object()),
            mock.patch.object(visualization_builder_module, "add_native_mjcf_from_stage") as native_import,
        ):
            builder = visualization_builder_module.build_visualization_builder_from_stage_envs(
                stage, env_paths, clone_plan
            )

        self.assertTrue(native_import.call_args_list)
        for call in native_import.call_args_list:
            self.assertTrue(call.kwargs["skip_equality_constraints"])
            self.assertFalse(call.kwargs["convert_mjc_equality_constraints"])

        self.assertEqual(
            [builder.geometry_sources_for_world(i) for i in range(3)],
            [["/World/envs/env_0/Object"], ["/World/envs/env_1/Object"], ["/World/envs/env_0/Object"]],
        )
        for attr, suffix in _VIS_LABEL_SUFFIXES.items():
            self.assertEqual(
                [builder.labels_for_world(i, attr) for i in range(3)],
                [
                    [f"/World/envs/env_0/Object/{suffix}"],
                    [f"/World/envs/env_1/Object/{suffix}"],
                    [f"/World/envs/env_2/Object/{suffix}"],
                ],
            )


if __name__ == "__main__":
    unittest.main()
