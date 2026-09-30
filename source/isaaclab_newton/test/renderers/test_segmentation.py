# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Newton Warp renderer's segmentation mapping (no GPU / sim required)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("warp")

import isaaclab_newton.renderers.segmentation as segmentation
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
from isaaclab_newton.renderers.segmentation import NewtonSegmentationMapper

# The color palette / reserved ids live in core and are unit-tested there
# (``isaaclab/test/renderers/test_segmentation_colors.py``); here they are only an oracle for the
# mapper's info-dict keys.
from isaaclab.cloner import ClonePlan
from isaaclab.renderers.segmentation_colors import BACKGROUND_ID, UNLABELLED_ID, pack_rgba, random_color_from_id


def _empty_clone_plan() -> ClonePlan:
    """A clone plan owning nothing, standing in for scenes with no replicated shapes to fall back to."""
    return ClonePlan(sources=(), destinations=(), clone_mask=torch.zeros(0, 0, dtype=torch.bool))


def _cfg(**overrides):
    """Minimal renderer-cfg stand-in exposing only the fields the mapper reads."""
    base = {
        "semantic_filter": "*:*",
        "semantic_segmentation_mapping": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _scene():
    """Two planned cartpole instances plus a plan-owned, untagged ground shape.

    Returns the clone plan and the per-shape prim-path list (``model.shape_label``).
    """
    shape_paths = [
        "/World/envs/env_0/Robot/pole/geom",
        "/World/envs/env_0/Robot/cart/geom",
        "/World/envs/env_1/Robot/pole/geom",
        "/World/ground/geom",
    ]
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/ground"),
        destinations=("/World/envs/env_{}/Robot", "/World/ground"),
        clone_mask=torch.tensor([[True, True], [False, False]]),
        env_ids=torch.arange(2, dtype=torch.long),
        semantic_tags=((("class", "cartpole"),), ()),
    )
    return plan, shape_paths


def _model(shape_paths):
    return SimpleNamespace(shape_label=list(shape_paths), device="cpu")


def test_semantic_segmentation_shares_class_id_across_envs():
    """All cartpole shapes across envs share one class id; the unlabelled ground is UNLABELLED."""
    plan, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg())
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    # env_0 pole, env_0 cart, env_1 pole all share the cartpole class id; ground is UNLABELLED.
    assert ids[0] == ids[1] == ids[2]
    assert ids[0] >= 2
    assert ids[3] == UNLABELLED_ID
    labels = mapping.info["idToLabels"]
    assert labels[BACKGROUND_ID] == {"class": "BACKGROUND"}
    assert labels[UNLABELLED_ID] == {"class": "UNLABELLED"}
    assert labels[ids[0]] == {"class": "cartpole"}


def test_instance_segmentation_groups_by_labelled_ancestor():
    """Shapes group by concrete planned roots; idToSemantics carries the cfg label."""
    plan, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg())
    mapper.build_mapping("instance_segmentation", colorize=False)
    mapping = mapper.get_mapping("instance_segmentation", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    # env_0 pole and cart share the env_0/Robot instance; env_1 is a separate instance; ground unlabelled.
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
    assert ids[3] == UNLABELLED_ID
    assert mapping.info["idToLabels"][ids[0]] == "/World/envs/env_0/Robot"
    assert mapping.info["idToSemantics"][ids[0]] == {"class": "cartpole"}


def test_colorize_info_keys_are_color_tuples():
    """With colorization, info keys are ``(r, g, b, a)`` color tuples and a color palette is built."""
    plan, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg())
    mapper.build_mapping("semantic_segmentation", colorize=True)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=True)

    assert mapping.shape_to_color is not None
    assert random_color_from_id(BACKGROUND_ID) in mapping.info["idToLabels"]
    assert random_color_from_id(UNLABELLED_ID) in mapping.info["idToLabels"]


def test_semantic_filter_excludes_non_matching_types():
    """A filter restricted to an absent type marks every shape UNLABELLED."""
    plan, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg(semantic_filter=["shape"]))
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    assert mapping.shape_to_id.numpy().tolist() == [UNLABELLED_ID] * len(shape_paths)


def test_semantic_filter_comma_separated_type_clauses():
    """Comma-separated ``type:label`` pairs within one semicolon group each match independently.

    Filter ``"class:cartpole, material:wood"`` must label shapes annotated with ``class:cartpole``
    *and* shapes annotated with ``material:wood``; before the fix only the first type clause was
    parsed, leaving ``material:wood`` shapes UNLABELLED.
    """
    shape_paths = [
        "/World/robot/geom",
        "/World/shelf/geom",
        "/World/ground/geom",
    ]
    plan = ClonePlan(
        sources=("/World/robot", "/World/shelf", "/World/ground"),
        destinations=("/World/robot", "/World/shelf", "/World/ground"),
        clone_mask=torch.zeros((3, 1), dtype=torch.bool),
        semantic_tags=((("class", "cartpole"),), (("material", "wood"),), ()),
    )
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg(semantic_filter="class:cartpole, material:wood"))
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    assert ids[0] >= 2, "class:cartpole shape should be labelled"
    assert ids[1] >= 2, "material:wood shape should be labelled (was UNLABELLED before fix)"
    assert ids[0] != ids[1], "different label payloads must map to different ids"
    assert ids[2] == UNLABELLED_ID


def test_mapper_has_no_stage_or_global_plan_fallback():
    """Segmentation consumes its explicit plan and never discovers semantics from global state."""
    source = inspect.getsource(segmentation)
    for forbidden in ("GetPrimAtPath", "SimulationContext", "get_labels"):
        assert forbidden not in source


def test_mapper_rejects_a_rendered_shape_absent_from_the_plan():
    """Everything the Newton renderer draws is owned by the clone plan."""
    plan, _shape_paths = _scene()

    with pytest.raises(ValueError, match="'/World/unplanned/geom'.*not covered by the clone plan"):
        NewtonSegmentationMapper(_model(["/World/unplanned/geom"]), plan, _cfg())


def test_renderer_retains_the_exact_plan_passed_by_clone_lifecycle():
    """The renderer receives semantic metadata explicitly through ``prepare_stage``."""
    plan, _shape_paths = _scene()
    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)

    renderer.prepare_stage(None, plan)

    assert renderer._clone_plan is plan


def test_renderer_rejects_stage_preparation_without_plan():
    """Newton rendering exists only inside the shared cloning lifecycle."""
    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)

    with pytest.raises(ValueError, match="requires an active clone plan"):
        renderer.prepare_stage(None, None)


def test_renderer_normalizes_explicit_ppisp_without_stage_discovery():
    """Newton consumes only the PPISP cfg carried by the camera plan."""
    from isaaclab_ppisp import PpispCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    stage = MagicMock()
    spec = SimpleNamespace(
        cfg=SimpleNamespace(spawn=None, isp_cfg=PpispCfg(inputs={"responsivity": 3.0}), data_types=[]),
        camera_source_prim_paths=("/World/prototypes/Camera",),
    )

    renderer.prepare_cameras(stage, spec)

    assert spec.cfg.isp_cfg.inputs["responsivity"] == pytest.approx(3.0)
    assert spec.cfg.isp_cfg.inputs["exposureOffset"] == pytest.approx(0.0)
    stage.GetPrimAtPath.assert_not_called()


def test_renderer_rejects_unimplemented_lens_distortion():
    """Selecting Newton never silently renders a requested distorted camera as pinhole."""
    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    spec = SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(distortion=object()), isp_cfg=None))

    with pytest.raises(NotImplementedError, match="requested OpenCV lens-distortion"):
        renderer.prepare_cameras(None, spec)


def test_semantics_absent_from_plan_are_unlabelled():
    """An asset-file label absent from cfg metadata is outside the explicit plan contract."""
    plan, shape_paths = _scene()
    plan = ClonePlan(plan.sources, plan.destinations, plan.clone_mask, plan.env_ids)
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg())
    mapper.build_mapping("semantic_segmentation", colorize=False)

    assert mapper.get_mapping("semantic_segmentation", colorize=False).shape_to_id.numpy().tolist() == [
        UNLABELLED_ID
    ] * len(shape_paths)


def test_variant_masks_select_semantics_and_instance_roots():
    """Rows sharing a destination template retain their own tags and concrete instance roots."""
    shape_paths = ["/World/envs/env_0/Object/geom", "/World/envs/env_1/Object/geom"]
    plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        env_ids=torch.arange(2, dtype=torch.long),
        semantic_tags=((("class", "cone"),), (("class", "sphere"),)),
    )
    mapper = NewtonSegmentationMapper(_model(shape_paths), plan, _cfg())
    mapper.build_mapping("instance_segmentation", colorize=False)
    mapping = mapper.get_mapping("instance_segmentation", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    assert mapping.info["idToLabels"][ids[0]] == "/World/envs/env_0/Object"
    assert mapping.info["idToSemantics"][ids[0]] == {"class": "cone"}
    assert mapping.info["idToLabels"][ids[1]] == "/World/envs/env_1/Object"
    assert mapping.info["idToSemantics"][ids[1]] == {"class": "sphere"}


def test_inactive_variant_fallback_source_does_not_own_semantics():
    """An inactive row's placeholder source cannot shadow the active variant at that path."""
    plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[False], [True]], dtype=torch.bool),
        env_ids=torch.tensor([0]),
        semantic_tags=((("class", "inactive"),), (("class", "active"),)),
    )
    mapper = NewtonSegmentationMapper(_model(["/World/envs/env_0/Object/geom"]), plan, _cfg())
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    shape_id = mapping.shape_to_id.numpy().item()
    assert mapping.info["idToLabels"][shape_id] == {"class": "active"}


def test_semantic_segmentation_mapping_overrides_color():
    """``semantic_segmentation_mapping`` forces the class color and its info key."""
    plan, shape_paths = _scene()
    override = (255, 36, 66, 255)
    mapper = NewtonSegmentationMapper(
        _model(shape_paths), plan, _cfg(semantic_segmentation_mapping={"class:cartpole": override})
    )
    mapper.build_mapping("semantic_segmentation", colorize=True)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=True)

    # The cartpole class id must be colored with the override, and keyed by it in idToLabels.
    assert override in mapping.info["idToLabels"]
    assert mapping.info["idToLabels"][override] == {"class": "cartpole"}
    packed = pack_rgba(override)
    assert packed in mapping.shape_to_color.numpy().tolist()
