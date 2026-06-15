# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
from filelock import FileLock
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab.sim import converters, schemas
from isaaclab.sim.spawners.materials import SurfaceDeformableBodyMaterialBaseCfg
from isaaclab.sim.utils import (
    add_labels,
    bind_physics_material,
    bind_visual_material,
    change_prim_property,
    clone,
    create_prim,
    find_matching_prim_paths,
    get_current_stage,
    get_first_matching_child_prim,
    select_usd_variants,
    set_prim_visibility,
)
from isaaclab.utils.assets import check_file_path, retrieve_file_path
from isaaclab.utils.version import has_kit

if TYPE_CHECKING:
    import trimesh

    from pxr import Sdf, Usd, UsdGeom  # noqa: F401

    from . import from_files_cfg

# import logger
logger = logging.getLogger(__name__)


_VALID_PRIM_PATH_REGEX = re.compile(r"^[a-zA-Z0-9/_]+$")


def _is_regex_prim_path(path: str) -> bool:
    """Return whether ``path`` contains regex characters."""
    return _VALID_PRIM_PATH_REGEX.match(path) is None


def _ensure_prim_specs(root_layer: Sdf.Layer, prim_path: str) -> None:
    """Create prim specs for ``prim_path`` and its ancestors."""
    from pxr import Sdf  # noqa: PLC0415

    current_path = ""
    for path_part in prim_path.strip("/").split("/"):
        current_path = f"{current_path}/{path_part}"
        Sdf.CreatePrimInLayer(root_layer, current_path)


def _resolve_mesh_spawn_paths(prim_path: str) -> tuple[str, list[str]]:
    """Resolve source and destination prim paths for mesh spawning."""
    prim_path = str(prim_path)
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")

    root_path, asset_path = prim_path.rsplit("/", 1)
    asset_path = asset_path.replace(".*", "0")
    if not _is_regex_prim_path(root_path):
        return f"{root_path}/{asset_path}", []

    root_parts = root_path.strip("/").split("/")
    for prefix_len in range(len(root_parts), 0, -1):
        prefix_path = "/" + "/".join(root_parts[:prefix_len])
        if not _is_regex_prim_path(prefix_path):
            continue

        source_parent_paths = find_matching_prim_paths(prefix_path)
        if not source_parent_paths:
            continue

        suffix = "/".join(root_parts[prefix_len:])
        if suffix:
            source_path = f"{source_parent_paths[0]}/{suffix}/{asset_path}"
            destination_paths = [f"{parent}/{suffix}/{asset_path}" for parent in source_parent_paths[1:]]
        else:
            source_path = f"{source_parent_paths[0]}/{asset_path}"
            destination_paths = [f"{parent}/{asset_path}" for parent in source_parent_paths[1:]]
        return source_path, destination_paths

    raise RuntimeError(f"Unable to find source prim path for mesh spawn: '{root_path}'.")


def _apply_spawn_metadata(prim: Usd.Prim, cfg) -> None:
    """Apply common spawner metadata to a spawned prim."""
    from pxr import UsdGeom  # noqa: PLC0415

    if hasattr(cfg, "visible"):
        imageable = UsdGeom.Imageable(prim)
        if cfg.visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
        for semantic_type, semantic_value in cfg.semantic_tags:
            semantic_type_sanitized = semantic_type.replace(" ", "_")
            semantic_value_sanitized = semantic_value.replace(" ", "_")
            add_labels(prim, labels=[semantic_value_sanitized], instance_name=semantic_type_sanitized, overwrite=False)


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
    If a default prim is not specified, then the asset is spawned at the root prim.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # spawn asset from the given usd file
    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a URDF file and override the settings with the given config.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF. This file is then imported
    at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the URDF file does not exist at the given path.
    """
    # urdf loader to convert urdf to usd
    urdf_loader = converters.UrdfConverter(cfg)
    # spawn asset from the generated usd file
    return _spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)


@clone
def spawn_from_mjcf(
    prim_path: str,
    cfg: from_files_cfg.MjcfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a MJCF file and override the settings with the given config.

    It uses the :class:`MjcfConverter` class to create a USD file from MJCF. This file is then imported
    at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the MJCF file does not exist at the given path.
    """
    # mjcf loader to convert mjcf to usd
    mjcf_loader = converters.MjcfConverter(cfg)
    # spawn asset from the generated usd file
    return _spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation, orientation)


def spawn_from_mesh(
    prim_path: str,
    cfg: from_files_cfg.MeshFileCfg,
    mesh: trimesh.Trimesh,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an in-memory triangle mesh into the scene.

    The spawned prim hierarchy is ``prim_path`` as an ``Xform`` root with a
    child mesh at ``prim_path/mesh``. Regex prim paths are supported by
    spawning the source mesh under the first matching parent and copying the
    resulting prim subtree to the remaining matching parents.

    Args:
        prim_path: The prim path or pattern to spawn the mesh at.
        cfg: The mesh spawner configuration.
        mesh: Triangle mesh data to spawn.
        translation: Translation to apply to the mesh root [m]. Defaults to None.
        orientation: Orientation ``(x, y, z, w)`` to apply to the mesh root. Defaults to None.
        **kwargs: Additional keyword arguments for compatibility with other spawners.

    Returns:
        The source root prim of the spawned mesh.
    """
    del kwargs
    from pxr import Sdf, UsdGeom  # noqa: PLC0415

    stage = get_current_stage()
    source_path, destination_paths = _resolve_mesh_spawn_paths(prim_path)
    if stage.GetPrimAtPath(source_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{source_path}'.")

    root_prim = create_prim(source_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    mesh_prim = create_prim(
        f"{source_path}/mesh",
        "Mesh",
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(mesh_prim.GetPrimPath()), cfg.collision_props, stage=stage)

    if mesh.visual.vertex_colors is not None:
        rgba_colors = np.asarray(mesh.visual.vertex_colors).astype(np.float32) / 255.0
        color_prim_attr = mesh_prim.GetAttribute("primvars:displayColor")
        color_prim_var = UsdGeom.Primvar(color_prim_attr)
        color_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        color_prim_attr.Set(rgba_colors[:, :3])
        display_prim_attr = mesh_prim.GetAttribute("primvars:displayOpacity")
        display_prim_var = UsdGeom.Primvar(display_prim_attr)
        display_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        display_prim_attr.Set(rgba_colors[:, 3])

    if cfg.visual_material is not None:
        material_path = (
            f"{source_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(str(mesh_prim.GetPrimPath()), material_path, stage=stage)

    if cfg.physics_material is not None:
        material_path = (
            f"{source_path}/{cfg.physics_material_path}"
            if not cfg.physics_material_path.startswith("/")
            else cfg.physics_material_path
        )
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(str(mesh_prim.GetPrimPath()), material_path, stage=stage)

    _apply_spawn_metadata(root_prim, cfg)

    if destination_paths:
        root_layer = stage.GetRootLayer()
        with Sdf.ChangeBlock():
            for destination_path in destination_paths:
                _ensure_prim_specs(root_layer, destination_path)
                Sdf.CopySpec(root_layer, Sdf.Path(source_path), root_layer, Sdf.Path(destination_path))

    return root_prim


def spawn_ground_plane(
    prim_path: str,
    cfg: from_files_cfg.GroundPlaneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawns a ground plane into the scene.

    This function loads the USD file containing the grid plane asset from Isaac Sim. It may
    not work with other assets for ground planes. In those cases, please use the `spawn_from_usd`
    function.

    Note:
        This function takes keyword arguments to be compatible with other spawners. However, it does not
        use any of the kwargs.

    Args:
        prim_path: The path to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        ValueError: If the prim path already exists.
    """
    # Obtain current stage
    stage = get_current_stage()

    # Spawn Ground-plane
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, usd_path=cfg.usd_path, translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # Create physics material
    if cfg.physics_material is not None:
        cfg.physics_material.func(f"{prim_path}/physicsMaterial", cfg.physics_material)
        # Apply physics material to ground plane
        collision_prim = get_first_matching_child_prim(
            prim_path,
            predicate=lambda _prim: _prim.GetTypeName() == "Plane",
            stage=stage,
        )
        if collision_prim is None:
            raise ValueError(f"No collision prim found at path: '{prim_path}'.")
        # bind physics material to the collision prim
        collision_prim_path = str(collision_prim.GetPath())
        bind_physics_material(collision_prim_path, f"{prim_path}/physicsMaterial", stage=stage)

    # Obtain environment prim
    environment_prim = stage.GetPrimAtPath(f"{prim_path}/Environment")
    # Scale only the mesh
    # Warning: This is specific to the default grid plane asset.
    if environment_prim.IsValid():
        # compute scale from size
        scale = (cfg.size[0] / 100.0, cfg.size[1] / 100.0, 1.0)
        # apply scale to the mesh
        environment_prim.GetAttribute("xformOp:scale").Set(scale)

    # Change the color of the plane
    # Warning: This is specific to the default grid plane asset.
    if cfg.color is not None:
        from pxr import Gf, Sdf  # noqa: PLC0415

        # change the color
        change_prim_property(
            prop_path=f"{prim_path}/Looks/theGrid/Shader.inputs:diffuse_tint",
            value=Gf.Vec3f(*cfg.color),
            stage=stage,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
        )
    # Remove the light from the ground plane (USD API, works without Kit/Newton)
    # It isn't bright enough and messes up with the user's lighting settings
    light_prim = stage.GetPrimAtPath(f"{prim_path}/SphereLight")
    if light_prim.IsValid():
        from pxr import UsdGeom  # noqa: PLC0415

        imageable = UsdGeom.Imageable(light_prim)
        imageable.MakeInvisible()

    prim = stage.GetPrimAtPath(prim_path)
    # Apply semantic tags
    if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
        # note: taken from replicator scripts.utils.utils.py
        for semantic_type, semantic_value in cfg.semantic_tags:
            # deal with spaces by replacing them with underscores
            semantic_type_sanitized = semantic_type.replace(" ", "_")
            semantic_value_sanitized = semantic_value.replace(" ", "_")
            # add labels to the prim
            add_labels(prim, labels=[semantic_value_sanitized], instance_name=semantic_type_sanitized)

    # Apply visibility
    set_prim_visibility(prim, cfg.visible)

    # return the prim
    return prim


"""
Helper functions.
"""


def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: from_files_cfg.FileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        usd_path: The path to the USD file to spawn the asset from.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # In distributed training, serialize asset download and USD stage composition
    # across ranks to prevent file I/O races. Concurrent mmap reads/writes on
    # the same cached USD files cause segfaults in Sdf_CrateFile::_MmapStream::Read.
    _world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    file_status = check_file_path(usd_path)
    if file_status == 0:
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")

    if _world_size > 1:
        lock = FileLock(os.path.join(tempfile.gettempdir(), "isaaclab_usd_spawn.lock"))
    else:
        lock = nullcontext()
    with lock:
        if file_status == 2:
            usd_path = retrieve_file_path(usd_path, force_download=False)
        stage = get_current_stage()
        if not stage.GetPrimAtPath(prim_path).IsValid():
            create_prim(
                prim_path,
                usd_path=usd_path,
                translation=translation,
                orientation=orientation,
                scale=cfg.scale,
                stage=stage,
            )
        else:
            logger.warning(f"A prim already exists at prim path: '{prim_path}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    if cfg.spatial_tendons_props is not None:
        schemas.modify_spatial_tendon_properties(prim_path, cfg.spatial_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        # auto-enable body-level gravcomp if joint-level actuator gravcomp is requested
        # without it — actuatorgravcomp has no effect since there are no forces to route.
        # Only auto-populates when the user did not already set ``gravcomp`` themselves;
        # an explicit ``MujocoRigidBodyPropertiesCfg(gravcomp=0.5)`` is preserved as-is.
        from isaaclab_newton.sim.schemas.schemas_cfg import MujocoJointDrivePropertiesCfg, MujocoRigidBodyPropertiesCfg

        body_gravcomp_unset = (
            not isinstance(cfg.rigid_props, MujocoRigidBodyPropertiesCfg) or cfg.rigid_props.gravcomp is None
        )
        if (
            isinstance(cfg.joint_drive_props, MujocoJointDrivePropertiesCfg)
            and cfg.joint_drive_props.actuatorgravcomp
            and body_gravcomp_unset
        ):
            logger.info(
                "Joint-level actuator gravity compensation requires body-level gravcomp."
                " Auto-setting MujocoRigidBodyPropertiesCfg(gravcomp=1.0)."
            )
            schemas.modify_rigid_body_properties(prim_path, MujocoRigidBodyPropertiesCfg(gravcomp=1.0))
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # define deformable body properties, or modify if deformable body API is present (PhysX only)
    if cfg.deformable_props is not None:
        prim = stage.GetPrimAtPath(prim_path)
        deformable_type = (
            "surface" if isinstance(cfg.physics_material, SurfaceDeformableBodyMaterialBaseCfg) else "volume"
        )
        if "OmniPhysicsDeformableBodyAPI" in prim.GetAppliedSchemas():
            schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props, stage)
        else:
            schemas.define_deformable_body_properties(prim_path, cfg.deformable_props, stage, deformable_type)
        if cfg.mass_props is not None:
            raise ValueError(
                """MassPropertiesCfg are not supported for deformable bodies
                and should be set through deformable_props with mass=<value>."""
            )

    # apply visual material
    if cfg.visual_material is not None:
        if not has_kit():
            logger.warning("Skipping visual material application for '%s' in kitless mode.", prim_path)
            return stage.GetPrimAtPath(prim_path)
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path, stage=stage)

    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material
        cfg.physics_material.func(material_path, cfg.physics_material)
        # apply material
        bind_physics_material(prim_path, material_path, stage=stage)

    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_from_usd_with_compliant_contact_material(
    prim_path: str,
    cfg: from_files_cfg.UsdFileWithCompliantContactCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and apply physics material to specified prims.

    This function extends the :meth:`spawn_from_usd` function by allowing application of compliant contact
    physics materials to specified prims within the spawned asset. This is useful for configuring
    contact behavior of specific parts within the asset.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance containing the USD file path and physics material settings.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset with the physics material applied to the specified prims.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stiff = cfg.compliant_contact_stiffness
    damp = cfg.compliant_contact_damping
    if cfg.physics_material_prim_path is None:
        logger.warning("No physics material prim path specified. Skipping physics material application.")
        return prim

    if isinstance(cfg.physics_material_prim_path, str):
        prim_paths = [cfg.physics_material_prim_path]
    else:
        prim_paths = cfg.physics_material_prim_path

    if stiff is not None or damp is not None:
        material_kwargs = {}
        if stiff is not None:
            material_kwargs["compliant_contact_stiffness"] = stiff
        if damp is not None:
            material_kwargs["compliant_contact_damping"] = damp
        material_cfg = PhysxRigidBodyMaterialCfg(**material_kwargs)

        for path in prim_paths:
            if not path.startswith("/"):
                rigid_body_prim_path = f"{prim_path}/{path}"
            else:
                rigid_body_prim_path = path

            material_path = f"{rigid_body_prim_path}/compliant_material"

            # spawn physics material
            material_cfg.func(material_path, material_cfg)

            bind_physics_material(
                rigid_body_prim_path,
                material_path,
            )
            logger.info(
                f"Applied physics material to prim: {rigid_body_prim_path} with compliance stiffness: {stiff} and"
                f" compliance damping: {damp}."
            )

    return prim
