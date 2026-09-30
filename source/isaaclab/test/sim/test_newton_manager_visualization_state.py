# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instance-ownership tests for Newton scene-data resources."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import CableLayout, RigidBodyLayout

pytestmark = pytest.mark.integration


def _completed_plan(*body_paths: str, cables: tuple[CableLayout, ...] = ()) -> ClonePlan:
    return ClonePlan(
        sources=(),
        destinations=(),
        clone_mask=np.zeros((0, 0), dtype=np.bool_),
        is_complete=True,
        rigid_body_prototypes=tuple(
            RigidBodyLayout(path, path, row, None, path.rsplit("/", 1)[-1]) for row, path in enumerate(body_paths)
        ),
        cables=cables,
    )


def _simulation_context():
    return SimpleNamespace(cfg=SimpleNamespace(device="cpu"))


def _builder(*, body_count: int = 1):
    class Builder:
        particle_count = 0

        def __init__(self):
            self.body_count = body_count
            self.finalize_calls = []

        def finalize(self, *, device):
            self.finalize_calls.append(device)
            return SimpleNamespace(state=lambda: SimpleNamespace(body_q=None), num_envs=None)

    return Builder()


def test_visualization_resource_finalizes_the_clone_built_model():
    from isaaclab_newton.cloner import NewtonReplicateContext

    resource = NewtonReplicateContext(_simulation_context())
    resource._builder = builder = _builder(body_count=3)
    resource._num_envs = 4

    resource.finalize_visualization_model()

    assert builder.finalize_calls == ["cpu"]
    assert resource.get_model() is not None
    assert resource.get_state_0() is not None
    assert resource.get_model().num_envs == 4


def test_visualization_resource_supports_an_empty_clone_built_model():
    from isaaclab_newton.cloner import NewtonReplicateContext

    resource = NewtonReplicateContext(_simulation_context())
    resource._builder = _builder(body_count=0)

    resource.finalize_visualization_model()

    assert resource.get_model() is not None
    assert resource.get_state_0() is not None


def test_visualization_resource_rejects_missing_clone_builder():
    from isaaclab_newton.cloner import NewtonReplicateContext

    with pytest.raises(RuntimeError, match="replication did not produce"):
        NewtonReplicateContext(_simulation_context()).finalize_visualization_model()


def test_physics_owner_requests_registry_owned_native_pointer_through_sdp():
    import warp as wp
    from isaaclab_newton.cloner import NewtonReplicateContext

    from isaaclab.scene_data import SceneDataFormat

    body_q = wp.zeros(1, dtype=wp.transformf)
    state = SimpleNamespace(body_q=body_q, particle_q=None)
    calls = []

    class Provider:
        point_count = 0

        def transform_generation(self, _name=None):
            return 1

        def point_generation(self):
            return 0

        def request_transforms(self, output_format):
            calls.append(output_format)
            output = output_format()
            output.transforms = body_q
            return output

    resource = NewtonReplicateContext(_simulation_context())
    resource._physics_cfg = object()
    resource._state_0 = state
    resource._model = SimpleNamespace(body_count=1, body_label=["/World/Body"])
    provider = Provider()

    assert resource.request_visualization_state(provider) is state
    assert state.body_q is body_q
    assert calls == [SceneDataFormat.IndexedTransform]


@pytest.mark.parametrize("published_count", [0, 2])
def test_visualization_resource_rejects_transform_count_mismatch(published_count: int):
    import warp as wp
    from isaaclab_newton.cloner import NewtonReplicateContext

    state = SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf), particle_q=None)

    class Provider:
        def request_transforms(self, output_format):
            if not published_count:
                return None
            output = output_format()
            output.transforms = wp.zeros(published_count, dtype=wp.transformf)
            return output

    resource = NewtonReplicateContext(_simulation_context())
    resource._state_0 = state
    resource._model = SimpleNamespace(body_count=1)

    with pytest.raises(RuntimeError, match=rf"SDP published {published_count} transforms.*with 1 bodies"):
        resource.request_visualization_state(Provider())


def test_shadow_resource_requests_canonical_transforms_and_tracks_provider_generation():
    import warp as wp
    from isaaclab_newton.cloner import NewtonReplicateContext

    from isaaclab.scene_data import SceneDataFormat

    body_path = "/World/envs/env_0/Robot/forearm"
    state = SimpleNamespace(body_q=wp.zeros(1, dtype=wp.transformf, device="cpu"), particle_q=None)
    calls: list[tuple] = []

    class Provider:
        point_count = 0

        def __init__(self):
            self.generation = 1

        def transform_generation(self, _name=None):
            return self.generation

        def point_generation(self):
            return 0

        def request_transforms(self, output_format):
            calls.append(("transforms", output_format))
            output = output_format()
            output.transforms = state.body_q
            return output

    provider = Provider()
    resource = NewtonReplicateContext(_simulation_context())
    resource._model = SimpleNamespace(body_count=1, body_label=[body_path])
    resource._state_0 = state

    resource.request_visualization_state(provider)
    resource._sensor_state_dirty = False
    resource.request_visualization_state(provider)

    assert len(calls) == 2
    assert calls[0][1] is SceneDataFormat.IndexedTransform
    assert not resource._sensor_state_dirty

    provider.generation += 1
    resource.request_visualization_state(provider)
    assert len(calls) == 3
    assert resource._sensor_state_dirty


def test_shadow_resource_passes_canonical_point_pointer_through_sdp():
    import warp as wp
    from isaaclab_newton.cloner import NewtonReplicateContext

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider, SceneDataPublication

    source_points = SceneDataFormat.Points()
    source_points.points = wp.array([(float(index), 0.0, 0.0) for index in range(6)], dtype=wp.vec3f, device="cpu")
    source_transforms = SceneDataFormat.Transform()
    source_transforms.transforms = wp.empty(0, dtype=wp.transformf, device="cpu")
    provider = SceneDataProvider(
        SimpleNamespace(
            transform_publication=SceneDataPublication(source_transforms, True),
            point_publications={"points": SceneDataPublication(source_points, True)},
            _materialize=lambda publication: None,
        )
    )
    resource = NewtonReplicateContext(_simulation_context())
    resource._model = SimpleNamespace(body_count=0, body_label=[], particle_count=6)
    resource._state_0 = SimpleNamespace(body_q=None, particle_q=wp.zeros(6, dtype=wp.vec3f, device="cpu"))

    state = resource.request_visualization_state(provider)

    assert state.particle_q is source_points.points
    assert state.particle_q is resource.request_visualization_state(provider).particle_q


def test_shadow_resource_has_no_post_clone_stage_resolver():
    from isaaclab_newton.cloner import NewtonReplicateContext

    source = inspect.getsource(NewtonReplicateContext.request_visualization_state)
    assert "usd_stage" not in source
    assert "GetPrimAtPath" not in source
    assert not hasattr(NewtonReplicateContext, "_resolve_scene_data_body_paths")


def test_newton_scene_data_backend_binds_the_manager_instance_state_once():
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    body_q = wp.zeros(1, dtype=wp.transformf, device="cpu")
    resource = SimpleNamespace(
        _state_0=SimpleNamespace(body_q=body_q, particle_q=None),
        _deformable_registry=[],
    )
    forward_calls = []
    manager = SimpleNamespace(
        _newton=resource,
        forward=lambda: forward_calls.append(None),
        get_model=lambda: SimpleNamespace(body_count=1, body_label=["/World/Body"]),
    )
    backend = NewtonSceneDataBackend(manager)
    backend.setup(
        SimpleNamespace(body_label=["/World/Body"]),
        _completed_plan("/World/Body"),
        "cpu",
    )

    assert backend.transform_publication.data.transforms is body_q
    assert forward_calls == []
    assert backend.transform_publication.dirty
    publication = backend.point_publications["points"]
    publication.dirty = False
    assert not publication.dirty


def test_newton_scene_data_backend_maps_plan_order_without_reordering_native_state():
    """Generated Newton bodies may interleave, so the publication carries one setup-time gather map."""
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    planned_paths = ["/World/envs/env_0/Body", "/World/envs/env_1/Body"]
    model = SimpleNamespace(
        body_label=[
            "/World/envs/env_0/Cable/generated_segment",
            planned_paths[0],
            "/World/envs/env_1/Cable/generated_segment",
            planned_paths[1],
        ]
    )
    plan = _completed_plan(*planned_paths)
    body_q = wp.zeros(4, dtype=wp.transformf, device="cpu")
    state = SimpleNamespace(body_q=body_q, particle_q=None)
    backend = NewtonSceneDataBackend(SimpleNamespace(_newton=SimpleNamespace(_state_0=state)))

    backend.setup(model, plan, "cpu")

    assert backend.transform_publication.data.transforms is body_q
    assert backend.transform_publication.data.source_indices.numpy().tolist() == [1, 3]


def test_newton_point_publication_is_only_the_native_particle_pointer():
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    particle_q = wp.zeros(9, dtype=wp.vec3f, device="cpu")
    resource = SimpleNamespace(
        _state_0=SimpleNamespace(body_q=None, particle_q=particle_q),
        _deformable_registry=[],
    )
    manager = SimpleNamespace(
        _newton=resource,
        get_model=lambda: SimpleNamespace(body_count=0, body_label=[]),
    )

    backend = NewtonSceneDataBackend(manager)
    backend.setup(SimpleNamespace(body_label=[]), _completed_plan(), "cpu")
    publication = backend.point_publications["points"]

    assert publication.data.points is particle_q
    assert publication.data.points.shape == (9,)
    assert publication.dirty
    assert SceneDataProvider(backend).request_points(SceneDataFormat.Points) is publication.data
    assert not publication.dirty


def test_newton_cable_publishes_native_state_and_derives_endpoints_only_on_request(monkeypatch):
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    state = SimpleNamespace(body_q=wp.array([wp.transform_identity()], dtype=wp.transformf), particle_q=None)
    model = SimpleNamespace(
        shape_body=wp.array([0], dtype=wp.int32),
        shape_transform=wp.array([wp.transform_identity()], dtype=wp.transformf),
        shape_scale=wp.array([[0.1, 0.5, 0.1]], dtype=wp.vec3f),
    )
    resource = SimpleNamespace(_state_0=state, _model=model, _deformable_registry=[])
    manager = SimpleNamespace(
        _newton=resource,
        _fk_reset_mask=None,
        get_model=lambda: SimpleNamespace(body_count=1, body_label=["/World/Cable/segment"]),
        get_state_0=lambda: state,
    )
    backend = NewtonSceneDataBackend(manager)
    backend._cable_publication.data = SceneDataFormat.CablePoints(
        shape_body=model.shape_body,
        shape_transform=model.shape_transform,
        shape_scale=model.shape_scale,
        shape_ids=wp.array([0], dtype=wp.int32),
        shape_offsets=wp.array([0], dtype=wp.int32),
        segment_counts=wp.array([1], dtype=wp.int32),
        binding_ids=wp.array([0], dtype=wp.int32),
    )
    launches = []
    launch = wp.launch

    def traced_launch(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", traced_launch)
    plan = _completed_plan(cables=(CableLayout("/Cable", "/Cable", 1, 0, None),))
    backend.setup(SimpleNamespace(body_label=[]), plan, "cpu")
    provider = SceneDataProvider(backend)
    provider._bind_point_plan(plan)

    native = provider.request_points(SceneDataFormat.CablePoints, "cables")
    first = provider.request_points(SceneDataFormat.Points, "cables")
    second = provider.request_points(SceneDataFormat.Points, "cables")
    wp.synchronize()

    assert native is backend._cable_publication.data
    assert native.body_q is state.body_q
    assert launches == ["cable_points_to_points_kernel"]
    assert first is second
    assert first.points is not native.body_q
    assert first.points.numpy()[:, 2].tolist() == [-0.5, 0.5]
    assert provider.point_generation("cables") == 1
