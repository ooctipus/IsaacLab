# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX point-publication bindings."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    from isaaclab_ov.cloner import OvReplicateContext  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_scene import OvrtxScene  # noqa: E402
    from ovrtx import DataAccess  # noqa: E402

    from isaaclab.cloner import ClonePlan  # noqa: E402
    from isaaclab.cloner.clone_plan import CableLayout, DeformableLayout, PointCloudLayout  # noqa: E402
    from isaaclab.scene_data import SceneDataFormat  # noqa: E402
else:
    OvReplicateContext = None
    OvrtxScene = None
    SceneDataFormat = None


class _RecordingScene:
    point_format = SceneDataFormat.HostMeshPoints if SceneDataFormat is not None else None
    transform_format = SceneDataFormat.HostTransposedMatrix44d if SceneDataFormat is not None else None

    def __init__(self):
        self.bound_paths: list[list[str]] = []
        self.pinned: list[object] = []
        self.writes: list[tuple[list[tuple], np.ndarray]] = []

    def bind(self, paths, *_args, **_kwargs):
        handle = SimpleNamespace(paths=list(paths))
        self.bound_paths.append(handle.paths)
        return handle

    def pin_world_space(self, handle) -> None:
        self.pinned.append(handle)

    def write_points(self, entries, points) -> None:
        self.writes.append((entries, points))

    def write_xforms(self, *_args) -> None:
        pass


def _make_context() -> OvReplicateContext:
    context = OvReplicateContext.__new__(OvReplicateContext)
    context._ovrtx_scene = _RecordingScene()
    context._point_bindings = {}
    context._scene_data_provider = None
    context._object_xforms = None
    context._sdp_transform_generation = -1
    context._sdp_camera_transform_generations = {}
    context._sdp_point_generations = {}
    return context


class _Provider:
    def __init__(self, arrays: dict[str, wp.array]):
        self.arrays = arrays
        self.requests = []
        self.moves = []
        self.generations = dict.fromkeys(arrays, 0)
        self.dirty = dict.fromkeys(arrays, True)
        self.outputs = {}

    def request_points(self, output_format, name="points"):
        self.requests.append((output_format, name))
        if self.dirty[name]:
            self.generations[name] += 1
            self.dirty[name] = False
            output = self.outputs[name] = output_format()
            output.points = self.arrays[name].numpy()
            self.moves.append(name)
        return self.outputs[name]

    def point_generation(self, name="points"):
        return self.generations[name]


def test_bind_point_streams_uses_only_plan_destinations():
    context = _make_context()
    plan = ClonePlan(
        sources=(),
        destinations=(),
        clone_mask=torch.zeros((0, 1), dtype=torch.bool),
        env_ids=torch.arange(1),
        is_complete=True,
        deformables=(
            DeformableLayout(
                "/World/envs/env_0/Cloth",
                "/World/envs/env_0/Cloth/sim",
                "/World/envs/env_0/Cloth/mesh",
                "/World/envs/env_*/Cloth",
                "surface",
                3,
                3,
                None,
                None,
                0,
                0,
            ),
        ),
        point_clouds=(PointCloudLayout("/World/envs/env_0/Media/Particles", 5, 1, 0),),
        cables=(CableLayout("/World/envs/env_0/Cable", "/World/envs/env_*/Cable", 3, 2, 0),),
        _env_ids_cpu=(0,),
    )

    context._initialize_ovrtx(object(), plan)

    assert context.scene.bound_paths == [
        ["/World/envs/env_0/Cloth/mesh", "/World/envs/env_0/Media/Particles"],
        ["/World/envs/env_0/Cable"],
    ]
    assert len(context.scene.pinned) == 2
    assert context._point_bindings["points"][1:] == ([0, 3], [3, 5])
    assert context._point_bindings["cables"][1:] == ([0], [4])


def test_update_requests_one_host_publication_per_dirty_stream():
    context = _make_context()
    points = wp.array([wp.vec3f(float(i), 0.0, 0.0) for i in range(5)], dtype=wp.vec3f, device="cpu")
    cables = wp.array([wp.vec3f(9.0, 8.0, 7.0), wp.vec3f(6.0, 5.0, 4.0)], dtype=wp.vec3f, device="cpu")
    point_handle = SimpleNamespace(paths=["/points"])
    cable_handle = SimpleNamespace(paths=["/cable"])
    context._point_bindings = {
        "points": (point_handle, [1], [3]),
        "cables": (cable_handle, [0], [2]),
    }
    provider = context._scene_data_provider = _Provider({"points": points, "cables": cables})

    context._update_ovrtx()
    context._update_ovrtx()
    provider.dirty = dict.fromkeys(provider.dirty, True)
    context._update_ovrtx()

    assert (
        provider.requests
        == [
            (SceneDataFormat.HostMeshPoints, "points"),
            (SceneDataFormat.HostMeshPoints, "cables"),
        ]
        * 3
    )
    assert provider.moves == ["points", "cables", "points", "cables"]
    assert len(context.scene.writes) == 4
    np.testing.assert_array_equal(context.scene.writes[0][1][1:4], points.numpy()[1:4])
    np.testing.assert_array_equal(context.scene.writes[1][1][:2], cables.numpy())


def test_update_rejects_a_missing_bound_publication() -> None:
    context = _make_context()
    context._point_bindings = {"points": (object(), [0], [1])}
    context._scene_data_provider = SimpleNamespace(request_points=lambda *_args: None)

    with pytest.raises(RuntimeError, match="SDP did not publish"):
        context._update_ovrtx()


def test_scene_writes_host_point_slices_without_a_second_conversion():
    scene = OvrtxScene.__new__(OvrtxScene)
    source = np.arange(18, dtype=np.float32).reshape(6, 3)
    writes = []
    binding = SimpleNamespace(write=lambda *args, **kwargs: writes.append((args, kwargs)))

    scene.write_points((SimpleNamespace(binding=binding), [1, 4], [2, 1]), source)

    assert len(writes) == 1
    slices = writes[0][0][0]
    assert np.shares_memory(slices[0], source)
    assert np.shares_memory(slices[1], source)
    np.testing.assert_array_equal(slices[0], source[1:3])
    np.testing.assert_array_equal(slices[1], source[4:5])
    assert writes[0][1] == {"data_access": DataAccess.ASYNC}


def test_scene_rejects_inconsistent_point_slice_metadata():
    scene = OvrtxScene.__new__(OvrtxScene)

    with pytest.raises(ValueError, match="zip"):
        scene.write_points(
            (SimpleNamespace(binding=SimpleNamespace(write=lambda *_args, **_kwargs: None)), [0], [2, 2]),
            np.zeros((4, 3), dtype=np.float32),
        )
