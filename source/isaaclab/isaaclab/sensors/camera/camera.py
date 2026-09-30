# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.sensors as sensor_utils
from isaaclab import cloner
from isaaclab.app.logging_utils import force_log_level
from isaaclab.renderers import BaseRenderer, CameraRenderSpec
from isaaclab.scene_data import SceneDataFormat, SceneDataPublication
from isaaclab.sim.views import FrameView
from isaaclab.utils import to_camel_case
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)
from isaaclab.utils.warp import ProxyArray

from ..sensor_base import SensorBase
from .camera_data import CameraData, RenderBufferKind

if TYPE_CHECKING:
    from .camera_cfg import CameraCfg

# import logger
logger = logging.getLogger(__name__)


@wp.kernel
def _camera_update_state_kernel(
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    intrinsics_src: wp.array(dtype=wp.mat33f),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_world_dst: wp.array(dtype=wp.quatf),
    intrinsics_dst: wp.array(dtype=wp.mat33f),
    frame: wp.array(dtype=wp.int64),
    env_mask: wp.array(dtype=wp.bool),
    env_ids: wp.array(dtype=wp.int32),
    use_env_ids: bool,
    use_env_mask: bool,
    update_pose: bool,
    update_intrinsics: bool,
    frame_op: int,
):
    """Update camera state for all, indexed, or masked cameras.

    ``frame_op`` uses 0 for no-op, 1 for increment, and 2 for reset.
    """
    src_id = wp.tid()
    dst_id = src_id
    if use_env_ids:
        dst_id = env_ids[src_id]
    if use_env_mask and not env_mask[dst_id]:
        return

    if update_pose:
        pos_dst[dst_id] = pos_src[src_id]
        quat_world_dst[dst_id] = quat_src[src_id] * wp.quatf(-0.5, 0.5, 0.5, 0.5)
    if update_intrinsics:
        intrinsics_dst[dst_id] = intrinsics_src[src_id]
    if frame_op == 1:
        frame[dst_id] = frame[dst_id] + wp.int64(1)
    elif frame_op == 2:
        frame[dst_id] = wp.int64(0)


class Camera(SensorBase):
    r"""The camera sensor for acquiring visual data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

    Summarizing from the `replicator extension`_, the following sensor types are supported:

    - ``"rgb"``: A 3-channel rendered color image.
    - ``"rgba"``: A 4-channel rendered color image with alpha channel.
    - ``"albedo"``: A 4-channel fast diffuse-albedo only path for color image.
      Note that this path will achieve the best performance when used alone or with depth only.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"depth"``: The same as ``"distance_to_image_plane"``.
    - ``"simple_shading_constant_diffuse"``: Simple shading (constant diffuse) RGB approximation.
    - ``"simple_shading_diffuse_mdl"``: Simple shading (diffuse MDL) RGB approximation.
    - ``"simple_shading_full_mdl"``: Simple shading (full MDL) RGB approximation.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"semantic_segmentation"``: The semantic segmentation data.
    - ``"instance_segmentation"``: The semantic instance segmentation data.
    - ``"instance_id_segmentation_fast"``: The instance id segmentation data.

    .. note::
        Currently the following sensor types are not supported in a "view" format:

        - ``"instance_segmentation"``: The instance segmentation data. Please use the fast counterparts instead.
        - ``"instance_id_segmentation"``: The instance id segmentation data. Please use the fast counterparts instead.
        - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_tight_fast"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_2d_loose_fast"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_3d"``: The 3D view space bounding box data.
        - ``"bounding_box_3d_fast"``: The 3D view space bounding box data.

    .. _replicator extension: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    cfg: CameraCfg
    """The configuration parameters."""

    UNSUPPORTED_TYPES: set[str] = {
        "instance_id_segmentation",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d_fast",
    }
    """The set of sensor types that are not supported by the camera class."""

    def __init__(self, cfg: CameraCfg):
        """Initializes the camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera or active renderer.
        """
        # These are the only fields cleanup owns, and construction can fail before SensorBase has
        # registered its callbacks. Seed them before any validation so ``__del__`` stays total.
        self._view: FrameView | None = None
        self._render_data = None
        self._render_pose_publication: SceneDataPublication | None = None
        self._scene_data_provider = None
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is None:
            raise RuntimeError("Camera construction requires an active SimulationContext.")
        self._scene_data_provider = sim_ctx.get_scene_data_provider()
        plan = sim_ctx.get_clone_plan()
        if plan is None or plan.is_complete:
            raise RuntimeError(
                f"Camera at {self.cfg.prim_path!r} requires an active clone plan before replication completes."
            )
        # Compute camera orientation (convention conversion) and spawn.
        rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=self.cfg.offset.convention, target="opengl"
        )
        rot_offset = rot_offset.squeeze(0).cpu().numpy()
        if self.cfg.spawn is not None and self.cfg.spawn.vertical_aperture is None:
            self.cfg.spawn.vertical_aperture = self.cfg.spawn.horizontal_aperture * self.cfg.height / self.cfg.width
        # Spawn exactly the prototype the plan assigned this camera cfg.
        spawn = self.cfg.spawn
        if spawn is not None:
            source_paths = cloner.query.cfg_source_paths(plan, self._source_cfg)
            spawn_target = next((path for path in source_paths if path is not None), None)
            if spawn_target is None:
                raise RuntimeError(f"Camera at {self.cfg.prim_path!r} has no populated clone-plan source.")
            source_prim = self.stage.GetPrimAtPath(spawn_target)
            if source_prim.IsValid() and (
                source_prim.HasAPI(UsdPhysics.ArticulationRootAPI) or source_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                logger.info(f" Spawning camera at '{self.cfg.prim_path}/camera'.")
                self.cfg.prim_path = f"{self.cfg.prim_path}/camera"
                spawn_target = f"{spawn_target}/camera"
            if not self.stage.GetPrimAtPath(spawn_target).IsValid():
                spawn.func(spawn_target, spawn, translation=self.cfg.offset.pos, orientation=rot_offset)
        self._view = FrameView(self.cfg.prim_path, simulation_context=sim_ctx, device=sim_ctx.device)
        # UsdGeom Camera prim for the sensor
        self._sensor_prims: list[UsdGeom.Camera] = list()
        # Allocated in :meth:`_create_buffers` once the renderer's output contract is known.
        self._data: CameraData | None = None
        # Construct the shared backend before reset; runtime resources initialize after cloning.
        self._renderer: BaseRenderer = sim_ctx.get_renderer(self.cfg.renderer_cfg)
        with force_log_level(logging.INFO):
            logger.info("Using renderer: %s", type(self._renderer).__name__)
        rows = tuple(cloner.query.iter_sources(plan, self.cfg.prim_path))
        if rows:
            source_paths = tuple(dict.fromkeys(source_path for _, _, source_path, _ in rows))
            source_paths_by_env = {env_id: source_path for _, _, source_path, env_ids in rows for env_id in env_ids}
            paths_by_env = {
                env_id: cloner.path.rebase(source_path, root, template.format(env_id))
                for root, template, source_path, env_ids in rows
                for env_id in env_ids
            }
            source_cameras = {path: UsdGeom.Camera.Get(self.stage, path) for path in source_paths}
            invalid_sources = [path for path, camera in source_cameras.items() if not camera]
            if invalid_sources:
                raise RuntimeError(f"Clone-plan camera source is not a UsdGeom.Camera: {invalid_sources}")
            # A global camera is one prim every env shares, so it is one render target, not N. Keep
            # the matching prototype handle beside each unique destination for stage-less backends.
            cameras_by_path = {
                paths_by_env[env_id]: source_cameras[source_paths_by_env[env_id]] for env_id in sorted(paths_by_env)
            }
            cam_paths = tuple(cameras_by_path)
            self._prototype_sensor_prims = tuple(cameras_by_path.values())
        else:
            raise RuntimeError(
                f"No clone-plan row covers the camera at '{self.cfg.prim_path}', so there is nothing to"
                " render into. Every asset the scene draws is declared to the plan and copied by the"
                " cloning pass, cameras included: build this camera inside a"
                " :class:`~isaaclab.cloner.ReplicateSession` that lists its config, or add it to the"
                " scene so the scene plans it."
            )
        self._render_spec = CameraRenderSpec(self.cfg, sim_ctx.device, source_paths, cam_paths)
        self._render_pose_publication = SceneDataPublication(SceneDataFormat.Vec3_Quat(), False)
        self._scene_data_provider.register_transforms(self.cfg.prim_path, self._render_pose_publication)
        self._renderer.prepare_cameras(self.stage, self._render_spec)
        sim_ctx.register_camera_sensor(self)

    def __del__(self):
        """Unsubscribes from callbacks and cleans up renderer resources."""
        if self._scene_data_provider is not None and self._render_pose_publication is not None:
            self._scene_data_provider.unregister_transforms(self.cfg.prim_path, self._render_pose_publication)
            self._render_pose_publication = None
        # unsubscribe callbacks
        super().__del__()

        # release the frame view's backend state, getattr covers partial initialization
        if getattr(self, "_view", None) is not None:
            self._view.close()
            self._view = None
        # release the render data; the backend itself belongs to SimulationContext
        if self._render_data is not None:
            self._renderer.cleanup(self._render_data)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> CameraData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def frame(self) -> ProxyArray:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.height, self.cfg.width)

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        """Allocated renderer output shapes without updating the camera."""
        outputs = None if self._data is None else self._data.output
        if outputs is None:
            raise RuntimeError("Camera output buffers are not available. Please call 'sim.play()' first.")
        return {name: buffer.shape for name, buffer in outputs.items()}

    @property
    def prim_paths(self) -> tuple[str, ...]:
        """USD path of this camera in each environment it covers, in ascending environment order.

        The clone plan says where the camera is cloned to, so these are known as soon as the
        prototype is spawned, whether or not the stage carries the clones.
        """
        return self._render_spec.camera_prim_paths

    """
    Configuration
    """

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor | wp.array, focal_length: float | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set parameters of the USD camera from its intrinsic matrix.

        The intrinsic matrix is used to set the following parameters to the USD camera:

        - ``focal_length``: The focal length of the camera.
        - ``horizontal_aperture``: The horizontal aperture of the camera.
        - ``vertical_aperture``: The vertical aperture of the camera.
        - ``horizontal_aperture_offset``: The horizontal offset of the camera.
        - ``vertical_aperture_offset``: The vertical offset of the camera.

        .. warning::

            Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
            i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
            is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None,
                focal_length will be calculated 1 / width.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            TypeError: If ``matrices`` is not a :class:`torch.Tensor` or a Warp array.
            ValueError: If the matrix count does not match the selected cameras, or a selected camera's
                calibration is owned by an OpenCV lens-distortion model.
        """
        if isinstance(matrices, torch.Tensor):
            if not matrices.is_contiguous():
                matrices = matrices.contiguous()
            matrices = wp.from_torch(matrices)
        elif not isinstance(matrices, wp.array):
            raise TypeError(f"Unsupported type for matrices: {type(matrices)}. Expected torch.Tensor or wp.array.")

        if env_ids is None:
            env_ids_np = np.arange(self._view.count)
        elif isinstance(env_ids, slice):
            env_ids_np = np.arange(self._view.count)[env_ids]
        else:
            env_ids_np = np.asarray(env_ids, dtype=np.int32).reshape(-1)

        matrices = matrices.numpy().astype(float, copy=False)
        if matrices.ndim == 2:
            matrices = matrices[None, ...]
        if matrices.shape != (len(env_ids_np), 3, 3):
            raise ValueError(
                f"Expected one 3x3 intrinsic matrix per selected camera, got {matrices.shape} for"
                f" {len(env_ids_np)} cameras."
            )
        distortion_ids = [
            int(i)
            for i in env_ids_np
            if self._sensor_prims[int(i)].GetPrim().GetAttribute("omni:lensdistortion:model").Get()
        ]
        if distortion_ids:
            raise ValueError(
                "set_intrinsic_matrices() cannot override cameras configured with an OpenCV lens-distortion"
                f" model: {distortion_ids}. Set their calibration in CameraCfg.spawn.distortion."
            )

        height, width = self.image_shape
        for i, intrinsic_matrix in zip(env_ids_np, matrices):
            # change data for corresponding camera index
            sensor_prim = self._sensor_prims[i]
            params = sensor_utils.convert_camera_intrinsics_to_usd(
                intrinsic_matrix=intrinsic_matrix.reshape(-1), height=height, width=width, focal_length=focal_length
            )
            # set parameters for camera
            for param_name, param_value in params.items():
                # convert to camel case (CC)
                param_name = to_camel_case(param_name, to="CC")
                # get attribute from the class
                param_attr = getattr(sensor_prim, f"Get{param_name}Attr")
                # convert numpy scalar to Python float for USD compatibility (NumPy 2.0+)
                if isinstance(param_value, np.floating):
                    param_value = float(param_value)
                # set value using pure USD API
                param_attr().Set(param_value)
        # update the internal buffers
        self._update_intrinsic_matrices(env_ids_np)

    """
    Operations - Set pose.
    """

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        r"""Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`isaaclab.sensors.camera.utils.convert_camera_frame_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        pos_wp = None
        if positions is not None:
            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions).to(device=self._device)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, device=self._device)
            positions = positions.to(device=self._device, dtype=torch.float32).reshape(-1, 3)
            pos_wp = wp.from_torch(positions.contiguous(), dtype=wp.vec3f)
        ori_wp = None
        if orientations is not None:
            if isinstance(orientations, np.ndarray):
                orientations = torch.from_numpy(orientations).to(device=self._device)
            elif not isinstance(orientations, torch.Tensor):
                orientations = torch.tensor(orientations, device=self._device)
            orientations = orientations.to(device=self._device, dtype=torch.float32).reshape(-1, 4)
            orientations = convert_camera_frame_orientation_convention(orientations, origin=convention, target="opengl")
            ori_wp = wp.from_torch(orientations.contiguous(), dtype=wp.vec4f)
        idx_wp = self._resolve_env_ids_wp(env_ids)
        with self._view.xform_world_space_writer() as writer:
            writer.set_poses(pos_wp, ori_wp, idx_wp)
        self._update_poses()

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is (N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            ValueError: If every eye position equals its target (look-at direction undefined for the
                whole batch). When only some rows are degenerate, those rows are skipped and the
                remaining poses are still applied; a warning is logged.
        """
        if isinstance(eyes, np.ndarray):
            eyes = torch.from_numpy(eyes).to(device=self._device)
        elif not isinstance(eyes, torch.Tensor):
            eyes = torch.tensor(eyes, device=self._device)
        eyes = eyes.to(device=self._device, dtype=torch.float32).reshape(-1, 3)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).to(device=self._device)
        elif not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self._device)
        targets = targets.to(device=self._device, dtype=torch.float32).reshape(-1, 3)
        if env_ids is None:
            env_ids_torch = torch.arange(self._view.count, dtype=torch.int32, device=self._device)
        elif isinstance(env_ids, slice):
            env_ids_torch = torch.arange(self._view.count, dtype=torch.int32, device=self._device)[env_ids]
        elif isinstance(env_ids, wp.array):
            env_ids_torch = wp.to_torch(env_ids).to(device=self._device, dtype=torch.int32).reshape(-1)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_torch = env_ids.to(device=self._device, dtype=torch.int32).reshape(-1)
        else:
            env_ids_torch = torch.tensor(env_ids, dtype=torch.int32, device=self._device).reshape(-1)
        # set camera poses using the view; degenerate rows (eye == target) come back as NaN
        rotation_matrix = create_rotation_matrix_from_view(eyes, targets, "Z", device=self._device)
        valid_indices = (~torch.isnan(rotation_matrix).any(dim=(-2, -1))).nonzero(as_tuple=True)[0]
        n_valid = valid_indices.numel()
        n_total = rotation_matrix.shape[0]
        if n_valid == 0:
            raise ValueError("look-at is undefined: every eye position equals its target")
        if n_valid < n_total:
            logger.warning(
                "set_world_poses_from_view: skipping %d pose(s) where eye equals target",
                n_total - n_valid,
            )
            rotation_matrix = rotation_matrix.index_select(0, valid_indices)
            eyes = eyes.index_select(0, valid_indices)
            env_ids_torch = env_ids_torch.index_select(0, valid_indices)
        orientations = quat_from_matrix(rotation_matrix)
        self.set_world_poses(eyes, orientations, env_ids_torch, convention="opengl")

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        if not self._is_initialized:
            raise RuntimeError("Camera could not be initialized. Check the renderer and simulation logs for details.")
        # reset the timestamps
        super().reset(env_ids, env_mask)
        update_state = self._update_poses if self._pose_follows_physics else self._update_camera_state
        if env_mask is not None:
            update_state(env_mask=env_mask, frame_op=2)
        elif env_ids is None:
            update_state(frame_op=2)
        else:
            env_ids_wp = self._resolve_env_ids_wp(env_ids)
            update_state(env_ids_wp, frame_op=2)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function delegates all render-product and annotator management to the
        :class:`~isaaclab.renderers.base_renderer.BaseRenderer` created in :meth:`__init__`. It also
        initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: Propagated from the renderer constructor when the active backend's runtime requirements
                are not satisfied.
        """
        # Initialize parent class
        super()._initialize_impl()

        self._view.initialize(self._clone_plan, self._scene_data_provider)
        frames = self._clone_plan.match_frames(self.cfg.prim_path)
        self._pose_follows_physics = any(frame.body_path is not None for frame in frames)
        # Check that sizes are correct
        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of camera prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # Create all env_ids buffer
        self._ALL_INDICES = wp.array(np.arange(self._view.count, dtype=np.int32), device=self._device)
        # Create frame count buffer
        self._frame = ProxyArray(wp.zeros(self._view.count, device=self._device, dtype=wp.int64))

        # Convert all encapsulated prims to Camera. A stage-less backend uses the exact prototype
        # handles retained when the plan was read, aligned with its clone destinations.
        self._sensor_prims.clear()
        view_prims = list(self._view.prims)
        if view_prims:
            for cam_prim in view_prims:
                if not cam_prim.IsA(UsdGeom.Camera):
                    raise RuntimeError(f"Prim at path '{cam_prim.GetPath()}' is not a Camera.")
                self._sensor_prims.append(UsdGeom.Camera(cam_prim))
        else:
            self._sensor_prims.extend(self._prototype_sensor_prims)

        self._render_data = self._renderer.create_render_data(self._render_spec)

        # Create internal buffers (includes intrinsic matrix and pose init)
        self._create_buffers()

    def _update_buffers_impl(self, env_mask: wp.array):
        if not self._env_mask_has_any(env_mask):
            return
        if self._pose_follows_physics:
            self._update_poses(env_mask=env_mask, frame_op=1, update_data_pose=self.cfg.update_latest_camera_pose)
        else:
            self._update_camera_state(env_mask=env_mask, frame_op=1)

        self._renderer.update(self._render_data, self._data.intrinsic_matrices)
        self._renderer.render(self._render_data)
        self._renderer.read_output(self._render_data, self._data)

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: CameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        if not cfg.data_types:
            raise ValueError("CameraCfg.data_types must request at least one renderer output.")
        if len(set(cfg.data_types)) != len(cfg.data_types):
            raise ValueError(f"CameraCfg.data_types contains duplicate outputs: {cfg.data_types}.")
        if "instance_segmentation_fast" in cfg.data_types:
            raise ValueError(
                "The data type 'instance_segmentation_fast' has been renamed to 'instance_segmentation'."
                " Please update your CameraCfg.data_types and any camera.data.output / camera.data.info key"
                " lookups to use 'instance_segmentation'."
            )
        # check if there is any intersection in unsupported types
        # reason: these use np structured data types which are not compatible with the camera buffer contract
        common_elements = set(cfg.data_types) & Camera.UNSUPPORTED_TYPES
        if common_elements:
            # provide alternative fast counterparts
            fast_common_elements = []
            for item in common_elements:
                if "instance_id_segmentation" in item:
                    fast_common_elements.append(item + "_fast")
            # raise error
            raise ValueError(
                f"Camera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be stored in the camera output buffers easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )
        simple_shading = [name for name in cfg.data_types if name.startswith("simple_shading_")]
        if len(simple_shading) > 1:
            raise ValueError(f"CameraCfg.data_types selects multiple simple shading modes: {simple_shading}.")

    def _create_buffers(self):
        """Create buffers for storing data."""
        specs = self._renderer.supported_output_types()
        # A requested output is part of the camera contract; silently dropping it would make the
        # selected renderer appear to work while returning a different interface.
        unknown: list[str] = []
        unsupported: list[str] = []
        for name in self.cfg.data_types:
            try:
                if RenderBufferKind(name) not in specs:
                    unsupported.append(name)
            except ValueError:
                unknown.append(name)
        errors = []
        if unknown:
            errors.append(f"Unknown camera data types: {unknown}.")
        if unsupported:
            errors.append(
                f"Renderer {type(self._renderer).__name__} does not support requested data types: {unsupported}."
            )
        if errors:
            raise ValueError("\n".join(errors))
        self._data = CameraData.allocate(
            data_types=self.cfg.data_types,
            height=self.cfg.height,
            width=self.cfg.width,
            num_views=self._view.count,
            device=self._device,
            supported_specs=specs,
        )
        # Camera-frame state (pose / intrinsics) is owned by the camera, not
        # the renderer: allocate warp buffers and populate them.
        self._data.create_buffers(self._view.count, self._device)
        self._update_intrinsic_matrices()
        self._update_poses()
        self._renderer.set_outputs(self._render_data, self._data.output)

    def _read_authored_opencv_intrinsics(
        self, prim: Usd.Prim, width: int, height: int, env_id: int
    ) -> tuple[float, float, float, float] | None:
        """Read the authored OpenCV lens-distortion intrinsics from a camera prim.

        Returns the calibrated ``(fx, fy, cx, cy)`` that the RTX/OVRTX renderer projects through when
        the prim carries an OpenCV lens-distortion model. Unlike the focal-length/aperture projection,
        these intrinsics may be non-square (``fx != fy``) or off-center.

        Args:
            prim: The camera prim to read the authored intrinsics from.
            width: The render width in pixels.
            height: The render height in pixels.
            env_id: The environment index, used in validation errors.

        Returns:
            The authored ``(fx, fy, cx, cy)`` in pixels, or ``None`` when no model is authored.
        """
        distortion_model = prim.GetAttribute("omni:lensdistortion:model").Get()
        if not distortion_model:
            return None
        prefix = f"omni:lensdistortion:{distortion_model}"
        intrinsics = tuple(prim.GetAttribute(f"{prefix}:{name}").Get() for name in ("fx", "fy", "cx", "cy"))
        if None in intrinsics:
            raise ValueError(
                f"Camera prim {prim.GetPath()!s} declares lens-distortion model {distortion_model!r}"
                " without complete fx/fy/cx/cy intrinsics."
            )
        image_size = prim.GetAttribute(f"{prefix}:imageSize").Get()
        if image_size is not None:
            authored_size = (int(image_size[0]), int(image_size[1]))
            if authored_size != (width, height):
                raise ValueError(
                    f"Camera prim {prim.GetPath()!s} (env {env_id}) lens-distortion imageSize"
                    f" {authored_size} does not match render resolution {(width, height)}."
                )
        return tuple(float(value) for value in intrinsics)

    def _update_intrinsic_matrices(self, env_ids: Sequence[int] | wp.array | None = None):
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. Without an authored
        OpenCV lens-distortion model the readback assumes square pixels and a centered principal point;
        when such a model is present its calibrated ``fx/fy/cx/cy`` are used instead.

        .. note::
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.
        """
        env_ids_np = self._resolve_env_ids_np(env_ids)
        if len(env_ids_np) == 0:
            return

        intrinsic_matrices = np.zeros((len(env_ids_np), 3, 3), dtype=np.float32)
        # viewport parameters are shared by every camera prim of this sensor
        height, width = self.image_shape
        # A full initialization reads the clone-plan prototypes. Cloned destinations inherit these
        # values, so reading each unique source once avoids an equivalent USD query per environment.
        sensor_prims = self._prototype_sensor_prims if env_ids is None else self._sensor_prims
        intrinsics_by_source = {}
        # iterate over all cameras
        for matrix_id, i in enumerate(env_ids_np):
            # Get corresponding sensor prim
            sensor_prim = sensor_prims[int(i)]
            source_path = sensor_prim.GetPath()
            if source_path not in intrinsics_by_source:
                # Prefer an authored OpenCV lens-distortion model when present: it carries the authoritative
                # fx/fy/cx/cy that the RTX/OVRTX renderer projects through, which may be non-square or off-center.
                authored = self._read_authored_opencv_intrinsics(sensor_prim.GetPrim(), width, height, int(i))
                if authored is not None:
                    intrinsics_by_source[source_path] = authored
                else:
                    # Rendering ignores aperture offsets and vertical aperture and assumes square pixels.
                    focal_length = sensor_prim.GetFocalLengthAttr().Get()
                    f_x = (width * focal_length) / sensor_prim.GetHorizontalApertureAttr().Get()
                    intrinsics_by_source[source_path] = (f_x, f_x, width * 0.5, height * 0.5)
            f_x, f_y, c_x, c_y = intrinsics_by_source[source_path]
            # create intrinsic matrix for depth linear
            intrinsic_matrices[matrix_id, 0, 0] = f_x
            intrinsic_matrices[matrix_id, 0, 2] = c_x
            intrinsic_matrices[matrix_id, 1, 1] = f_y
            intrinsic_matrices[matrix_id, 1, 2] = c_y
            intrinsic_matrices[matrix_id, 2, 2] = 1.0

        intrinsic_matrices_wp = wp.array(intrinsic_matrices, dtype=wp.mat33f, device=self._device)
        self._update_camera_state(
            env_ids=None if env_ids is None else self._resolve_env_ids_wp(env_ids_np),
            intrinsics_src=intrinsic_matrices_wp,
            update_intrinsics=True,
        )

    def _update_poses(
        self,
        env_ids: Sequence[int] | wp.array | None = None,
        env_mask: wp.array | None = None,
        frame_op: int = 0,
        update_data_pose: bool = True,
    ):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w).
        """
        if self._view is None or self._render_data is None:
            raise RuntimeError("Camera view is not available. Please call 'sim.play()' first.")

        # get the poses from the view (returns ProxyArray)
        env_ids_wp = None if env_mask is not None else self._resolve_env_ids_wp(env_ids)
        pos_w, quat_w = self._view.get_world_poses(env_ids_wp)
        pos_w_wp = pos_w.warp
        pos_w_wp = wp.array(
            ptr=pos_w_wp.ptr,
            dtype=wp.vec3f,
            shape=(pos_w_wp.shape[0],),
            device=pos_w_wp.device,
            copy=False,
        )
        quat_w_wp = quat_w.warp
        quat_w_wp = wp.array(
            ptr=quat_w_wp.ptr,
            dtype=wp.quatf,
            shape=(quat_w_wp.shape[0],),
            device=quat_w_wp.device,
            copy=False,
        )

        # FrameView publishes the camera prim's native OpenGL pose. Renderers request their exact
        # layout from SDP, avoiding CameraData's public world-convention conversion and its inverse.
        if env_ids_wp is None:
            publication = self._render_pose_publication
            publication.data.positions = pos_w_wp
            publication.data.orientations = quat_w_wp
            publication.dirty = True

        self._update_camera_state(
            env_ids=env_ids_wp,
            env_mask=env_mask,
            pos_src=pos_w_wp,
            quat_src=quat_w_wp,
            update_pose=update_data_pose,
            frame_op=frame_op,
        )

    def _update_camera_state(
        self,
        env_ids: wp.array | None = None,
        env_mask: wp.array | None = None,
        pos_src: wp.array | None = None,
        quat_src: wp.array | None = None,
        intrinsics_src: wp.array | None = None,
        update_pose: bool = False,
        update_intrinsics: bool = False,
        frame_op: int = 0,
    ):
        """Update camera pose, intrinsics, and frame counters through one Warp kernel."""
        count = env_ids.shape[0] if env_ids is not None else self._view.count
        if count == 0:
            return
        wp.launch(
            _camera_update_state_kernel,
            dim=count,
            inputs=[
                pos_src if pos_src is not None else self._data.pos_w.warp,
                quat_src if quat_src is not None else self._data.quat_w_world.warp,
                intrinsics_src if intrinsics_src is not None else self._data.intrinsic_matrices.warp,
                self._data.pos_w.warp,
                self._data.quat_w_world.warp,
                self._data.intrinsic_matrices.warp,
                self._frame.warp,
                env_mask if env_mask is not None else self._ALL_ENV_MASK,
                env_ids if env_ids is not None else self._ALL_INDICES,
                env_ids is not None,
                env_mask is not None,
                update_pose,
                update_intrinsics,
                frame_op,
            ],
            device=self._device,
        )

    def _resolve_env_ids_np(self, env_ids: Sequence[int] | wp.array | None) -> np.ndarray:
        """Resolve camera indices to a host ``int32`` array for USD metadata reads."""
        if env_ids is None:
            return np.arange(self._view.count, dtype=np.int32)
        if isinstance(env_ids, slice):
            return np.arange(self._view.count, dtype=np.int32)[env_ids]
        if isinstance(env_ids, wp.array):
            return env_ids.numpy().astype(np.int32, copy=False).reshape(-1)
        return np.asarray(env_ids, dtype=np.int32).reshape(-1)

    def _resolve_env_ids_wp(self, env_ids: Sequence[int] | torch.Tensor | wp.array | slice | None) -> wp.array | None:
        """Resolve camera indices to a Warp ``int32`` array."""
        if env_ids is None:
            return None
        if isinstance(env_ids, wp.array):
            if env_ids.dtype != wp.int32:
                raise TypeError(f"Unsupported wp.array dtype for env_ids: {env_ids.dtype}. Expected wp.int32.")
            if str(env_ids.device) == str(self._device):
                return env_ids
            env_ids = env_ids.numpy().astype(np.int32, copy=False).reshape(-1)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self._device, dtype=torch.int32).reshape(-1)
            if not env_ids.is_contiguous():
                env_ids = env_ids.contiguous()
            return wp.from_torch(env_ids, dtype=wp.int32)
        elif isinstance(env_ids, slice):
            env_ids = np.arange(self._view.count, dtype=np.int32)[env_ids]
        else:
            env_ids = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        return wp.array(env_ids, dtype=wp.int32, device=self._device)

    @staticmethod
    def _env_mask_has_any(env_mask: wp.array) -> bool:
        """Return whether the mask selects any camera."""
        return bool(np.any(env_mask.numpy()))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        if self._render_data is not None:
            self._renderer.cleanup(self._render_data)
        self._render_data = None
        # call parent
        super()._invalidate_initialize_callback(event)
