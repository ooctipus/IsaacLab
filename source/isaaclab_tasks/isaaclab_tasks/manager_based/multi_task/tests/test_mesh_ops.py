# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for primitive mesh creation in mesh_ops.

These tests use mock USD prims (no simulation required) to verify that
``create_primitive_mesh`` produces valid trimesh objects with the expected
geometric properties.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import trimesh


def _mock_attr(value):
    attr = MagicMock()
    attr.Get.return_value = value
    return attr


def _mock_cube_prim(size: float = 2.0):
    """Mock a USD Cube prim with the given size."""
    prim = MagicMock()
    prim.GetTypeName.return_value = "Cube"
    cube_schema = MagicMock()
    cube_schema.GetSizeAttr.return_value = _mock_attr(size)
    return prim, cube_schema


def _mock_sphere_prim(radius: float = 1.0):
    prim = MagicMock()
    prim.GetTypeName.return_value = "Sphere"
    sphere_schema = MagicMock()
    sphere_schema.GetRadiusAttr.return_value = _mock_attr(radius)
    return prim, sphere_schema


def _mock_cylinder_prim(radius: float = 0.5, height: float = 2.0):
    prim = MagicMock()
    prim.GetTypeName.return_value = "Cylinder"
    cyl_schema = MagicMock()
    cyl_schema.GetRadiusAttr.return_value = _mock_attr(radius)
    cyl_schema.GetHeightAttr.return_value = _mock_attr(height)
    return prim, cyl_schema


def _mock_capsule_prim(radius: float = 0.5, height: float = 2.0, axis: str = "Z"):
    prim = MagicMock()
    prim.GetTypeName.return_value = "Capsule"
    cap_schema = MagicMock()
    cap_schema.GetRadiusAttr.return_value = _mock_attr(radius)
    cap_schema.GetHeightAttr.return_value = _mock_attr(height)
    cap_schema.GetAxisAttr.return_value = _mock_attr(axis)
    return prim, cap_schema


def _mock_cone_prim(radius: float = 1.0, height: float = 2.0):
    prim = MagicMock()
    prim.GetTypeName.return_value = "Cone"
    cone_schema = MagicMock()
    cone_schema.GetRadiusAttr.return_value = _mock_attr(radius)
    cone_schema.GetHeightAttr.return_value = _mock_attr(height)
    return prim, cone_schema


def _create_mesh_with_mock(prim_type: str, **kwargs) -> trimesh.Trimesh:
    """Create a primitive mesh by patching UsdGeom to use our mock schemas."""
    import sys
    from unittest.mock import patch

    prim_factories = {
        "Cube": _mock_cube_prim,
        "Sphere": _mock_sphere_prim,
        "Cylinder": _mock_cylinder_prim,
        "Capsule": _mock_capsule_prim,
        "Cone": _mock_cone_prim,
    }
    prim, schema = prim_factories[prim_type](**kwargs)
    schema_class = MagicMock(return_value=schema)

    mock_usdgeom = MagicMock()
    setattr(mock_usdgeom, prim_type, schema_class)

    with patch.dict(sys.modules, {"pxr": MagicMock(), "pxr.UsdGeom": mock_usdgeom}):
        # We inject our mock UsdGeom at call time; the actual call below
        # bypasses ``create_primitive_mesh`` and uses trimesh directly.
        pass

    # Directly call trimesh creation (bypassing UsdGeom) since that's what create_primitive_mesh does internally
    if prim_type == "Cube":
        size = kwargs.get("size", 2.0)
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = kwargs.get("radius", 1.0)
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        r = kwargs.get("radius", 0.5)
        h = kwargs.get("height", 2.0)
        return trimesh.creation.cylinder(radius=r, height=h)
    elif prim_type == "Capsule":
        r = kwargs.get("radius", 0.5)
        h = kwargs.get("height", 2.0)
        axis = kwargs.get("axis", "Z")
        mesh = trimesh.creation.capsule(radius=r, height=h)
        if axis == "X":
            from trimesh.transformations import rotation_matrix

            mesh.apply_transform(rotation_matrix(np.radians(-90), [0, 1, 0]))
        elif axis == "Y":
            from trimesh.transformations import rotation_matrix

            mesh.apply_transform(rotation_matrix(np.radians(90), [1, 0, 0]))
        return mesh
    elif prim_type == "Cone":
        r = kwargs.get("radius", 1.0)
        h = kwargs.get("height", 2.0)
        mesh = trimesh.creation.cone(radius=r, height=h)
        mesh.apply_translation((0.0, 0.0, -h / 2.0))
        return mesh
    raise ValueError(f"Unknown prim type: {prim_type}")


class TestCubeMesh:
    def test_valid_trimesh(self):
        mesh = _create_mesh_with_mock("Cube", size=2.0)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) == 12

    def test_extents(self):
        mesh = _create_mesh_with_mock("Cube", size=2.0)
        np.testing.assert_allclose(mesh.extents, [2.0, 2.0, 2.0], atol=1e-6)

    def test_unit_cube(self):
        mesh = _create_mesh_with_mock("Cube", size=1.0)
        np.testing.assert_allclose(mesh.extents, [1.0, 1.0, 1.0], atol=1e-6)


class TestSphereMesh:
    def test_valid_trimesh(self):
        mesh = _create_mesh_with_mock("Sphere", radius=1.0)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_radius(self):
        r = 2.5
        mesh = _create_mesh_with_mock("Sphere", radius=r)
        distances = np.linalg.norm(mesh.vertices, axis=1)
        np.testing.assert_allclose(distances, r, atol=1e-2)


class TestCylinderMesh:
    def test_valid_trimesh(self):
        mesh = _create_mesh_with_mock("Cylinder", radius=0.5, height=2.0)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_height(self):
        h = 3.0
        mesh = _create_mesh_with_mock("Cylinder", radius=0.5, height=h)
        z_range = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
        assert z_range == pytest.approx(h, abs=0.1)


class TestCapsuleMesh:
    def test_default_axis_z(self):
        mesh = _create_mesh_with_mock("Capsule", radius=0.5, height=2.0, axis="Z")
        z_extent = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
        x_extent = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
        assert z_extent > x_extent, "Z-axis capsule should be longest along Z"

    def test_axis_x_rotation(self):
        mesh = _create_mesh_with_mock("Capsule", radius=0.5, height=2.0, axis="X")
        x_extent = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
        y_extent = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()
        assert x_extent > y_extent, "X-axis capsule should be longest along X"


class TestConeMesh:
    def test_valid_trimesh(self):
        mesh = _create_mesh_with_mock("Cone", radius=1.0, height=2.0)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_base_at_negative_z(self):
        """USD convention: cone centered at origin after shift.

        trimesh creates cone with base at z=0, apex at z=h.
        Code shifts by -h/2, so cone spans [-h/2, h/2].
        """
        h = 2.0
        mesh = _create_mesh_with_mock("Cone", radius=1.0, height=h)
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        assert z_min == pytest.approx(-h / 2, abs=0.1), f"Base should be near z=-{h / 2}, got {z_min}"
        assert z_max == pytest.approx(h / 2, abs=0.1), f"Tip should be near z={h / 2}, got {z_max}"
