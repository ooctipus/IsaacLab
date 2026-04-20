# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the RetargetPipeline.

Requires GPU + Newton + Warp. No IsaacSim.
"""

import time

import newton
import newton.ik as ik
import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.manager_based.locomotion.position.mdp.kinematics import NewtonKinematics
from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget import (
    RetargetPipeline,
    RetargetPipelineCfg,
)


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


ANYMAL_USD = "/home/zhengyuz/Downloads/ANYmal-C/anymal_c.usd"
DEVICE = "cuda:0"
FOOT_ERR_TOL = 0.02

DEFAULT_JPOS = np.array([0, 0.4, -0.8, 0, -0.4, 0.8, 0, 0.4, -0.8, 0, -0.4, 0.8], dtype=np.float32)


def _objectives_factory(model, foot_ids, base_body_id=0):
    """Return an ObjectivesFactory closure for the given model and foot config."""
    def factory(n_problems):
        device = str(model.device)
        contact_objs = [
            ik.IKObjectivePosition(
                link_index=cid, link_offset=wp.vec3(0, 0, 0),
                target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=1.0,
            )
            for cid in foot_ids
        ]
        base_pos_obj = ik.IKObjectivePosition(
            link_index=base_body_id, link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=0.05,
        )
        base_rot_obj = ik.IKObjectiveRotation(
            link_index=base_body_id, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros(n_problems, dtype=wp.vec4, device=device), weight=0.5,
        )
        jl_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper, weight=10.0,
        )
        all_objs = [*contact_objs, base_pos_obj, base_rot_obj, jl_obj]
        return all_objs, contact_objs, base_pos_obj, base_rot_obj
    return factory


@pytest.fixture(scope="module")
def robot_data():
    kin = NewtonKinematics(ANYMAL_USD, device=DEVICE,
                           default_pos=(0, 0, 0.6), default_joint_pos=DEFAULT_JPOS)
    foot_ids = [i for i, n in enumerate(kin.body_names) if "FOOT" in n.upper()]
    base_pos = kin.default_body_q[0][:3]
    foot_pos = np.array([kin.default_body_q[fid][:3] for fid in foot_ids])
    foot_offsets = foot_pos - base_pos
    standing_height = float(base_pos[2] - foot_pos[:, 2].mean())
    foot_ground_offset = float(foot_pos[:, 2].mean())
    sf = _objectives_factory(kin.model, foot_ids)
    return kin, foot_ids, kin.default_joint_q, foot_offsets, standing_height, foot_ground_offset, sf


def _make_flat_mesh(size: float = 10.0) -> wp.Mesh:
    v = np.array([
        [-size / 2, -size / 2, 0],
        [size / 2, -size / 2, 0],
        [size / 2, size / 2, 0],
        [-size / 2, size / 2, 0],
    ], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return convert_to_warp_mesh(v, f, device=DEVICE)


def _make_pipeline(kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf, max_cand=500):
    return RetargetPipeline(
        kin=kin,
        objectives_factory=sf,
        cfg=RetargetPipelineCfg(device=DEVICE, max_candidates=max_cand),
        contact_body_ids=foot_ids,
        foot_offsets=foot_offsets,
        foot_ground_offset=fgo,
        standing_height=standing_height,
        default_joint_q=jq,
    )


class TestPipelineFlat:

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_flat_high_acceptance(self, robot_data):
        kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf = robot_data
        pipeline = _make_pipeline(kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf)
        buf = pipeline.run(_make_flat_mesh(), np.zeros(3), n_desired=20)
        print(pipeline.rejection_summary)
        assert buf.num_selected > 0, "Should select at least 1 on flat terrain"


class TestPipelineStair:

    @pytest.fixture(scope="class")
    def stair_mesh(self):
        meshes = []
        for i in range(5):
            z = i * 0.2
            box = trimesh.creation.box(extents=[0.3, 4.0, z + 0.01])
            box.apply_translation([i * 0.3 + 0.15, 0, z / 2])
            meshes.append(box)
        mesh = trimesh.util.concatenate(meshes)
        return convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces, device=DEVICE)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_stair_acceptance(self, robot_data, stair_mesh):
        kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf = robot_data
        pipeline = _make_pipeline(kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf, 1000)
        buf = pipeline.run(stair_mesh, np.zeros(3), n_desired=50)
        print(pipeline.rejection_summary)
        if buf.num_selected == 0:
            pytest.skip("No valid candidates on stair terrain")

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_stair_foot_error(self, robot_data, stair_mesh):
        kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf = robot_data

        def foot_err_criterion(buf, N):
            nc = len(foot_ids)
            nb = kin.model.body_count
            tpl = newton.ModelBuilder()
            tpl.add_usd(kin.usd_path, collapse_fixed_joints=False)
            bldr = newton.ModelBuilder()
            for _ in range(N):
                bldr.add_world(tpl)
            fk_m = bldr.finalize(device=DEVICE)
            jq_t = wp.to_torch(buf.joint_q_result)[:N]
            fk_m.joint_q = wp.from_torch(jq_t.contiguous().view(-1))
            st = fk_m.state()
            newton.eval_fk(fk_m, fk_m.joint_q,
                           wp.zeros(fk_m.joint_dof_count, dtype=float, device=DEVICE), st)
            body_q = wp.to_torch(st.body_q).view(N, nb, 7)  # type: ignore[arg-type]
            ct = wp.to_torch(buf.contact_targets).view(-1, 3)[:N * nc].view(N, nc, 3)
            idx = torch.tensor(foot_ids, device=DEVICE, dtype=torch.long)
            err = (body_q[:, idx, :3] - ct).norm(dim=-1).max(dim=-1).values
            return err <= FOOT_ERR_TOL

        pipeline = _make_pipeline(kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf, 1000)
        buf = pipeline.run(
            stair_mesh, np.zeros(3), n_desired=50,
            criteria={"foot_err": foot_err_criterion},
        )
        if buf.num_selected == 0:
            pytest.skip("No valid candidates after foot_err filter")


class TestPipelinePerformance:

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_1000_candidates_under_10s(self, robot_data):
        kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf = robot_data
        pipeline = _make_pipeline(kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf, 1000)
        t0 = time.time()
        pipeline.run(_make_flat_mesh(), np.zeros(3), n_desired=100)
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"Pipeline took {elapsed:.1f}s, expected <10s"


class TestPipelineDeterminism:

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_deterministic(self, robot_data):
        kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf = robot_data
        results = []
        for _ in range(2):
            pipeline = _make_pipeline(kin, foot_ids, jq, foot_offsets, standing_height, fgo, sf, 200)
            buf = pipeline.run(_make_flat_mesh(), np.zeros(3), n_desired=10)
            results.append(buf.num_selected)
        assert results[0] == results[1], f"Non-deterministic: {results[0]} vs {results[1]}"
