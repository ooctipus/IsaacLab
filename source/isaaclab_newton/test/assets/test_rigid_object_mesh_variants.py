# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for Newton rigid-object mesh variants."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, device="cuda:0").app

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


def _mesh_asset(name: str, mass: float) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Assets/Props/NIST/{name}.usd",
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
    )


@configclass
class _MeshVariantSceneCfg(InteractiveSceneCfg):
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_mesh_asset("round_peg_4mm", 1.0), _mesh_asset("rectangular_peg_16mm", 8.0)]
        ),
        mesh_variants_enabled=True,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )


@pytest.mark.isaacsim_ci
def test_write_mesh_variant_updates_selected_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch collision geometry and inertia through the rigid-object API."""
    sim_cfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=sim_cfg, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_MeshVariantSceneCfg(num_envs=2, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()

        asset = scene["object"]
        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([1.0, 8.0]))

        asset.write_mesh_variant_to_sim(
            torch.tensor([1, 0], dtype=torch.int32, device=sim.device),
            torch.tensor([0, 1], dtype=torch.int32, device=sim.device),
        )
        wp.synchronize_device(sim.device)

        torch.testing.assert_close(asset.mesh_variant_ids.torch.cpu(), torch.tensor([1, 0], dtype=torch.int32))
        torch.testing.assert_close(asset.data.body_mass.torch[:, 0].cpu(), torch.tensor([8.0, 1.0]))

        captured = {}

        def capture_indices(name, *, variant_ids, env_ids):
            captured[name] = variant_ids, env_ids

        monkeypatch.setattr(NewtonManager, "_set_mesh_variant_index", capture_indices)
        variant_ids = torch.tensor([0, 1], dtype=torch.int32, device=sim.device)
        env_ids = torch.tensor([1, 0], dtype=torch.int32, device=sim.device)
        asset.write_mesh_variant_to_sim(variant_ids, env_ids)
        captured_variants, captured_envs = captured[asset.cfg.prim_path]
        assert wp.to_torch(captured_variants).data_ptr() != variant_ids.data_ptr()
        assert wp.to_torch(captured_envs).data_ptr() != env_ids.data_ptr()
        torch.cuda.synchronize(sim.device)
        variant_ids.fill_(1)
        env_ids.fill_(0)

        torch.testing.assert_close(wp.to_torch(captured_variants).cpu(), torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(wp.to_torch(captured_envs).cpu(), torch.tensor([1, 0], dtype=torch.int32))

        asset.write_mesh_variant_to_sim(variant_ids, (0, 1))
        _, captured_envs = captured[asset.cfg.prim_path]
        torch.testing.assert_close(wp.to_torch(captured_envs).cpu(), torch.tensor([0, 1], dtype=torch.int32))
