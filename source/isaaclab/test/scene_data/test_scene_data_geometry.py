# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneData geometry (points) copy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from isaaclab.cloner.clone_plan import CableLayout, ClonePlan, DeformableLayout, PointCloudLayout
from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat, SceneDataPublication
from isaaclab.scene_data.scene_data_provider import SceneDataProvider


class _PointsBackend(SceneDataBackend):
    def __init__(
        self,
        points: np.ndarray | None = None,
    ):
        if points is None:
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        self._points = wp.array(points, dtype=wp.vec3f)
        self._points_data = SceneDataFormat.Points()
        self._points_data.points = self._points
        self._publication = SceneDataPublication(self._points_data, dirty=True)
        self._transform_publication = SceneDataPublication(SceneDataFormat.Transform(), dirty=False)
        self.point_reads = 0

    @property
    def transform_publication(self) -> SceneDataPublication:
        return self._transform_publication

    @property
    def point_publications(self) -> dict[str, SceneDataPublication]:
        if self._publication.dirty:
            self.point_reads += 1
        return {"points": self._publication}


def test_request_points_passes_through_and_caches_backend_buffer():
    backend = _PointsBackend()
    provider = SceneDataProvider(backend)

    first = provider.request_points(SceneDataFormat.Points)
    second = provider.request_points(SceneDataFormat.Points)

    assert first is second
    assert first.points is backend._points
    assert backend.point_reads == 1

    backend._publication.dirty = True
    third = provider.request_points(SceneDataFormat.Points)
    assert third is first
    assert backend.point_reads == 2


def test_request_host_points_moves_once_in_sdp_and_caches_the_result():
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    backend = _PointsBackend(points)
    provider = SceneDataProvider(backend)

    first = provider.request_points(SceneDataFormat.HostPoints)
    assert provider.request_points(SceneDataFormat.HostPoints) is first
    np.testing.assert_array_equal(first.points, points)
    assert backend.point_reads == 1

    backend._publication.dirty = True
    assert provider.request_points(SceneDataFormat.HostPoints) is first
    assert backend.point_reads == 2


def _completed_plan(num_instances: int = 0, **topology) -> ClonePlan:
    env_ids = tuple(range(num_instances))
    return ClonePlan(
        (),
        (),
        np.zeros((0, num_instances), dtype=np.bool_),
        np.arange(num_instances, dtype=np.int64),
        is_complete=True,
        _env_ids_cpu=env_ids,
        **topology,
    )


def _mapped_plan(num_instances: int = 1) -> ClonePlan:
    indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    weights = np.full((1, 4), 0.25, dtype=np.float32)
    return _completed_plan(
        num_instances,
        deformables=tuple(
            DeformableLayout(
                root_path=f"/World/envs/env_{env_id}/Soft",
                sim_mesh_path=f"/World/envs/env_{env_id}/Soft/sim",
                vis_mesh_path=f"/World/envs/env_{env_id}/Soft/visual",
                view_path="/World/envs/env_*/Soft",
                deformable_type="volume",
                vertex_count=4,
                vis_vertex_count=1,
                point_indices=indices,
                point_weights=weights,
                row=0,
                env_id=env_id,
            )
            for env_id in range(num_instances)
        ),
    )


def test_request_host_mesh_points_fuses_plan_mapping_and_keeps_native_points_aliased():
    backend = _PointsBackend(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    )
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(_mapped_plan())

    assert provider.request_points(SceneDataFormat.Points).points is backend._points
    mapped = provider.request_points(SceneDataFormat.HostMeshPoints)

    np.testing.assert_allclose(mapped.points, [[0.25, 0.25, 0.25]])


def test_request_host_mesh_points_accepts_an_all_direct_plan():
    backend = _PointsBackend()
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(
        _completed_plan(point_clouds=(PointCloudLayout("/World/Particles", backend._points.shape[0], 0, None),))
    )

    mapped = provider.request_points(SceneDataFormat.HostMeshPoints)

    np.testing.assert_array_equal(mapped.points, backend._points.numpy())


def test_mapped_sink_rejects_native_count_that_disagrees_with_plan():
    provider = SceneDataProvider(_PointsBackend(np.zeros((3, 3), dtype=np.float32)))
    provider._bind_point_plan(_mapped_plan())

    with pytest.raises(RuntimeError, match="has 3 native points.*requires 4"):
        provider.request_points(SceneDataFormat.HostMeshPoints)


def test_point_plan_deduplicates_shared_clone_mapping_tables():
    provider = SceneDataProvider(_PointsBackend(np.zeros((8, 3), dtype=np.float32)))
    provider._bind_point_plan(_mapped_plan(2))

    _, _, _, _, _, mapping_offsets, indices, weights = provider._point_maps["points"]
    assert mapping_offsets.tolist() == [0, 0]
    assert indices.shape == weights.shape == (1, 4)


def test_body_point_bundle_aliases_natively_and_gathers_only_when_requested(monkeypatch):
    plan = _completed_plan(
        deformables=(
            DeformableLayout("/A", "/A/sim", "/A/vis", "/A", "surface", 2, 2, None, None, 0, None),
            DeformableLayout("/B", "/B/sim", "/B/vis", "/B", "surface", 3, 3, None, None, 1, None),
        )
    )
    points = wp.array(
        [
            [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
        ],
        dtype=wp.vec3f,
    )
    native = SceneDataFormat.BodyPoints(
        points=(points,),
        binding_ids=(wp.array([1, 0], dtype=wp.int32),),
    )
    backend = _PointsBackend()
    backend._publication.data = native
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(plan)
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)

    assert provider.request_points(SceneDataFormat.BodyPoints) is native
    assert launches == []
    flat = provider.request_points(SceneDataFormat.Points)
    assert launches == ["body_points_to_points_kernel"]
    np.testing.assert_array_equal(flat.points.numpy()[:, 0], [10.0, 11.0, 20.0, 21.0, 22.0])
    assert provider.request_points(SceneDataFormat.Points) is flat
    assert launches == ["body_points_to_points_kernel"]


def test_cable_endpoints_are_derived_only_inside_requested_sdp_conversion(monkeypatch):
    native = SceneDataFormat.CablePoints(
        body_q=wp.array([wp.transform_identity()], dtype=wp.transformf),
        shape_body=wp.array([0], dtype=wp.int32),
        shape_transform=wp.array([wp.transform_identity()], dtype=wp.transformf),
        shape_scale=wp.array([[0.1, 1.0, 0.1]], dtype=wp.vec3f),
        shape_ids=wp.array([0], dtype=wp.int32),
        shape_offsets=wp.array([0], dtype=wp.int32),
        segment_counts=wp.array([1], dtype=wp.int32),
        binding_ids=wp.array([0], dtype=wp.int32),
    )
    publication = SceneDataPublication(native, dirty=True)
    backend = SimpleNamespace(point_publications={"cables": publication}, _materialize=lambda _publication: None)
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(_completed_plan(cables=(CableLayout("/Cable", "/Cable", 1, 0, None),)))
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)

    assert provider.request_points(SceneDataFormat.CablePoints, "cables") is native
    assert launches == []
    points = provider.request_points(SceneDataFormat.Points, "cables")
    assert launches == ["cable_points_to_points_kernel"]
    np.testing.assert_allclose(points.points.numpy(), [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])


def test_scene_data_publisher_is_private():
    provider = SceneDataProvider(_PointsBackend())

    assert not hasattr(provider, "backend")


@pytest.mark.parametrize("source_device", ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []))
def test_named_transforms_pass_native_pointer_or_convert_once_per_dirty_generation(source_device):
    current_device = "cuda:0" if source_device == "cpu" and wp.is_cuda_available() else "cpu"
    with wp.ScopedDevice(current_device):
        provider = SceneDataProvider(_PointsBackend())
        poses = SceneDataFormat.Vec3_Quat()
        poses.positions = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=wp.vec3f, device=source_device)
        poses.orientations = wp.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device=source_device
        )
        publication = SceneDataPublication(poses, True)
        provider.register_transforms("camera", publication)

        assert provider.request_transforms(SceneDataFormat.Vec3_Quat, name="camera") is poses
        assert provider.transform_generation("camera") == 1
        converted = provider.request_transforms(SceneDataFormat.Transform, name="camera")
        assert str(converted.transforms.device) == source_device
        assert provider.request_transforms(SceneDataFormat.Transform, name="camera") is converted
        assert provider.transform_generation("camera") == 1
        np.testing.assert_allclose(converted.transforms.numpy()[:, :3], poses.positions.numpy())

        poses.positions = wp.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=wp.vec3f, device=source_device)
        publication.dirty = True
        refreshed = provider.request_transforms(SceneDataFormat.Transform, name="camera")
        assert refreshed is converted
        assert provider.transform_generation("camera") == 2
        np.testing.assert_allclose(refreshed.transforms.numpy()[:, :3], poses.positions.numpy())


def test_named_transform_owner_can_unregister_its_publication():
    provider = SceneDataProvider(_PointsBackend())
    publication = SceneDataPublication(SceneDataFormat.Vec3_Quat(), True)
    provider.register_transforms("camera", publication)
    provider.unregister_transforms("camera", publication)

    with pytest.raises(KeyError):
        provider.request_transforms(SceneDataFormat.Vec3_Quat, name="camera")
