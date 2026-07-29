# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the factory assembly contact material.

Binding a physics material that authors no ``physics:staticFriction`` /
``physics:dynamicFriction`` resolves the shape's friction to 0 on the Newton
backend (unauthored USD attributes read back as the schema default 0.0,
overriding the no-material default). mu=0 contacts with condim=3 zero the
pyramidal row invweight, so ``efc_D`` is floored to ``1/MJ_MINVAL`` (~1e15) and
the float32 Newton-solver Hessian degenerates to NaN on contact-rich states
(seated insertion). mjwarp warns "friction[0] < MJ_MINMU with condim=3 may
cause NaN" at model build for exactly this.
"""

from __future__ import annotations

from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

from isaaclab_tasks.core.multi_task.factory.factory_assets_cfg import ASSEMBLY_CONTACT_MATERIAL_CFG


class TestAssemblyContactMaterial:
    def test_newton_material_authors_nonzero_friction(self):
        """The newton_mjwarp material fragments must author nonzero static+dynamic friction."""
        fragments = ASSEMBLY_CONTACT_MATERIAL_CFG.newton_mjwarp
        assert isinstance(fragments, list), "newton material must be a fragment list (friction + newton contact)"
        friction_frags = [f for f in fragments if isinstance(f, UsdPhysicsRigidBodyMaterialCfg)]
        assert friction_frags, "missing UsdPhysicsRigidBodyMaterialCfg fragment: bound material would resolve mu=0"
        frag = friction_frags[0]
        assert frag.static_friction is not None and frag.static_friction > 1e-5
        assert frag.dynamic_friction is not None and frag.dynamic_friction > 1e-5
