# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import numpy as np
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelState
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.cloner import expand_env_regex_ns
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase


class CuroboPlanner(MotionPlannerBase):
    """Motion planner for robot manipulation using cuRobo.

    This planner provides collision-aware motion planning capabilities for robotic manipulation tasks.
    It integrates with Isaac Lab environments to:

    - Build its collision world from the clone plan
    - Plan collision-free paths to target poses
    - Handle object attachment and detachment during manipulation
    - Execute planned motions with proper collision checking

    The planner uses cuRobo for fast motion generation and supports
    multi-phase planning for contact scenarios like grasping and placing objects.
    """

    def __init__(
        self,
        config: CuroboPlannerCfg,
        env: ManagerBasedEnv,
        env_id: int = 0,
    ) -> None:
        """Initialize CuRobo from one cfg and the environment's completed clone plan.

        Args:
            config: Declarative planner and collision-world configuration.
            env: Environment containing the planned scene.
            env_id: Environment whose robot and obstacle states the planner consumes.
        """
        robot = env.scene[config.robot_entity]
        if not isinstance(robot, Articulation):
            raise TypeError(f"CuRobo robot entity {config.robot_entity!r} is not an articulation.")
        super().__init__(env=env, robot=robot, env_id=env_id, debug=config.debug_planner)
        self.config = config
        log_level = logging.DEBUG if config.debug_planner else logging.INFO
        self.logger = logging.getLogger(f"CuroboPlanner_{env_id}")
        self.logger.setLevel(log_level)
        self.n_repeat = config.n_repeat
        self.step_size = config.motion_step_size

        self.attached_objects: dict[str, str] = {}
        setup_curobo_logger("warn")
        if not torch.cuda.is_available():
            raise RuntimeError("CuRobo motion planning requires CUDA.")
        idx = config.cuda_device
        self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
        self.logger.debug(f"cuRobo motion planner initialized on CUDA device {idx}")

        robot_cfg_file = join_path(get_robot_configs_path(), config.robot_config_file)
        robot_cfg: dict[str, Any] = load_yaml(robot_cfg_file)["robot_cfg"]
        robot_cfg["kinematics"]["urdf_path"] = retrieve_file_path(config.robot_urdf_file, force_download=True)
        self.logger.info(f"Loaded robot configuration from {robot_cfg_file}")
        robot_cfg["kinematics"]["collision_spheres"] = config.collision_spheres_file
        robot_cfg["kinematics"]["extra_collision_spheres"] = {
            config.attached_object_link_name: config.attached_object_sphere_count
        }
        self.robot_cfg = robot_cfg

        plan = env.sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("CuRobo requires a completed clone plan.")
        world_cfg = self._world_from_plan(plan)
        motion_gen_config: MotionGenConfig = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args=self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=config.num_trajopt_seeds,
            num_graph_seeds=config.num_graph_seeds,
            interpolation_dt=config.interpolation_dt,
            collision_cache=config.collision_cache_size,
            trajopt_tsteps=config.trajopt_tsteps,
            collision_activation_distance=config.collision_activation_distance,
            position_threshold=config.position_threshold,
            rotation_threshold=config.rotation_threshold,
        )
        self.motion_gen = MotionGen(motion_gen_config)

        self.plan_config = MotionGenPlanConfig(
            enable_graph=config.enable_graph,
            enable_graph_attempt=config.enable_graph_attempt,
            max_attempts=config.max_planning_attempts,
            enable_finetune_trajopt=config.enable_finetune_trajopt,
            time_dilation_factor=config.time_dilation_factor,
        )
        self._current_plan: JointState | None = None
        self._plan_index = 0
        self.logger.info("Warming up motion planner...")
        self.motion_gen.warmup(enable_graph=config.enable_graph, warmup_js_trajopt=False)

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to cuRobo device for isolated device management.

        Ensures all tensors used by cuRobo are on CUDA device, providing device isolation
        from Isaac Lab's potentially different device configuration. This prevents device
        mismatch errors and optimizes cuRobo performance.

        Args:
            tensor: Input tensor (may be on any device)

        Returns:
            Tensor converted to cuRobo's CUDA device with appropriate dtype
        """
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor back to environment device for Isaac Lab compatibility.

        Converts cuRobo tensors back to the environment's device to ensure compatibility
        with Isaac Lab operations that expect tensors on the environment's configured device.

        Args:
            tensor: Input tensor from cuRobo operations (typically on CUDA)

        Returns:
            Tensor converted to environment's device while preserving dtype
        """
        return tensor.to(device=self.env.device, dtype=tensor.dtype)

    def _world_from_plan(self, plan: ClonePlan) -> WorldConfig:
        """Build this environment's CuRobo mesh world from clone-plan geometry."""
        meshes = []
        self._object_obstacles: dict[str, str] = {}
        names: set[str] = set()
        env_template = self.env.scene.cfg.clone_cfg.clone_template
        for collision_cfg in self.config.mesh_prim_paths:
            if collision_cfg.name in names:
                raise ValueError(f"CuRobo obstacle name {collision_cfg.name!r} is declared more than once.")
            names.add(collision_cfg.name)
            prim_expr = expand_env_regex_ns(collision_cfg.prim_expr, env_template)
            matches = tuple(match for match in plan.match_geometry_targets(prim_expr) if match[0].env_id == self.env_id)
            if len(matches) != 1:
                raise ValueError(
                    f"CuRobo target {collision_cfg.prim_expr!r} has {len(matches)} clone-plan entries for env"
                    f" {self.env_id}; expected one."
                )
            target, geometries = matches[0]
            if collision_cfg.scene_entity is None:
                if any(geometry.frame.body_path is not None for geometry in geometries):
                    raise ValueError(
                        f"CuRobo target {collision_cfg.prim_expr!r} contains a rigid body but declares no"
                        " scene_entity for its pose."
                    )
                local_poses = [self._relative_pose(target.pose, geometry.frame.pose) for geometry in geometries]
                pose = self._pose_in_robot_frame(target.pose)
            else:
                try:
                    rigid_object = self.env.scene.rigid_objects[collision_cfg.scene_entity]
                except KeyError as exc:
                    raise ValueError(
                        f"CuRobo scene entity {collision_cfg.scene_entity!r} is not a rigid object."
                    ) from exc
                entity_expr = expand_env_regex_ns(rigid_object.cfg.prim_path, env_template)
                if entity_expr != prim_expr:
                    raise ValueError(
                        f"CuRobo target {collision_cfg.prim_expr!r} does not match scene entity"
                        f" {collision_cfg.scene_entity!r} at {rigid_object.cfg.prim_path!r}."
                    )
                bodies = tuple(body for body in plan.match_rigid_body_subtrees(prim_expr) if body.env_id == self.env_id)
                if len(bodies) != 1:
                    raise ValueError(
                        f"Dynamic CuRobo target {collision_cfg.prim_expr!r} must contain one planned rigid body."
                    )
                body = bodies[0]
                local_poses = [geometry.frame.pose for geometry in geometries]
                if any(geometry.frame.body_path != body.path for geometry in geometries):
                    raise ValueError(
                        f"Dynamic CuRobo target {collision_cfg.prim_expr!r} contains geometry outside its rigid body."
                    )
                position, quaternion = PoseUtils.subtract_frame_transforms(
                    self.robot.data.root_pos_w.torch[self.env_id],
                    self.robot.data.root_quat_w.torch[self.env_id],
                    rigid_object.data.root_pos_w.torch[self.env_id],
                    rigid_object.data.root_quat_w.torch[self.env_id],
                )
                pose = self._curobo_mesh_pose(position, quaternion)
                if collision_cfg.scene_entity in self._object_obstacles:
                    raise ValueError(f"Scene entity {collision_cfg.scene_entity!r} has multiple CuRobo obstacles.")
                self._object_obstacles[collision_cfg.scene_entity] = collision_cfg.name

            collision_geometries = tuple(
                (geometry, local_pose)
                for geometry, local_pose in zip(geometries, local_poses, strict=True)
                if geometry.collision
            )
            if not collision_geometries:
                raise ValueError(f"CuRobo target {collision_cfg.prim_expr!r} has no planned collision geometry.")
            vertices = []
            faces = []
            vertex_offset = 0
            for geometry, local_pose in collision_geometries:
                vertices.append(self._transform_vertices(geometry.vertices, local_pose))
                faces.append(geometry.faces + vertex_offset)
                vertex_offset += len(geometry.vertices)
            meshes.append(
                Mesh(
                    name=collision_cfg.name,
                    pose=pose,
                    vertices=np.concatenate(vertices).tolist(),
                    faces=np.concatenate(faces).reshape(-1).tolist(),
                )
            )
        return WorldConfig(mesh=meshes)

    def _pose_in_robot_frame(self, world_pose: tuple[float, ...]) -> list[float]:
        """Convert an ``xyzw`` world pose to a CuRobo ``wxyz`` robot-frame pose."""
        position = torch.tensor(world_pose[:3], device=self.robot.device)
        quaternion = torch.tensor(world_pose[3:], device=self.robot.device)
        position, quaternion = PoseUtils.subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[self.env_id],
            self.robot.data.root_quat_w.torch[self.env_id],
            position,
            quaternion,
        )
        return self._curobo_mesh_pose(position, quaternion)

    @staticmethod
    def _curobo_mesh_pose(position: torch.Tensor, quaternion: torch.Tensor) -> list[float]:
        """Return a CuRobo mesh pose list with ``wxyz`` quaternion ordering."""
        x, y, z, w = quaternion.tolist()
        return [*position.tolist(), w, x, y, z]

    @staticmethod
    def _relative_pose(reference: tuple[float, ...], pose: tuple[float, ...]) -> tuple[float, ...]:
        """Return ``pose`` relative to ``reference`` in clone-plan ``xyzw`` format."""
        position, quaternion = PoseUtils.subtract_frame_transforms(
            torch.tensor(reference[:3]),
            torch.tensor(reference[3:]),
            torch.tensor(pose[:3]),
            torch.tensor(pose[3:]),
        )
        return (*position.tolist(), *quaternion.tolist())

    @staticmethod
    def _transform_vertices(vertices: np.ndarray, pose: tuple[float, ...]) -> np.ndarray:
        """Transform scale-baked planned vertices by one ``xyzw`` pose."""
        rotation = PoseUtils.matrix_from_quat(torch.tensor(pose[3:])).numpy()
        return vertices @ rotation.T + np.asarray(pose[:3])

    @property
    def current_plan(self) -> JointState | None:
        """Current plan from cuRobo motion generator."""
        return self._current_plan

    def update_world(self) -> None:
        """Publish every configured rigid-object pose to CuRobo once."""
        for object_name, obstacle_name in self._object_obstacles.items():
            rigid_object = self.env.scene.rigid_objects[object_name]
            position, quaternion = PoseUtils.subtract_frame_transforms(
                self.robot.data.root_pos_w.torch[self.env_id],
                self.robot.data.root_quat_w.torch[self.env_id],
                rigid_object.data.root_pos_w.torch[self.env_id],
                rigid_object.data.root_quat_w.torch[self.env_id],
            )
            self.motion_gen.world_coll_checker.update_obstacle_pose(
                obstacle_name,
                self._make_pose(position=position, quaternion=quaternion),
                update_cpu_reference=True,
            )
        torch.cuda.synchronize(self.tensor_args.device)

    def _attach_object(self, object_name: str, obstacle_name: str) -> bool:
        """Attach an object to the robot for manipulation planning.

        Establishes an attachment between the specified object and the robot's end-effector
        or configured attachment link. This enables the robot to carry the object during
        motion planning while maintaining proper collision checking. The object's collision
        geometry is disabled in the world model since it's now part of the robot.

        Args:
            object_name: Isaac Lab scene key for the object.
            obstacle_name: Exact CuRobo world obstacle name declared by the planner cfg.

        Returns:
            True if attachment succeeded, False if attachment failed
        """
        current_joint_state = self._get_current_joint_state_for_curobo()

        self.logger.debug(f"Attaching {object_name} as obstacle {obstacle_name}")

        success = self.motion_gen.attach_objects_to_robot(
            joint_state=current_joint_state,
            object_names=[obstacle_name],
            link_name=self.config.attached_object_link_name,
            surface_sphere_radius=self.config.surface_sphere_radius,
            sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
            world_objects_pose_offset=None,
        )

        if success:
            self.attached_objects[object_name] = self.config.attached_object_link_name
            self.logger.debug(f"Successfully attached {object_name}")
            self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

            # Deactivate the original obstacle as it's now carried by the robot
            self.motion_gen.world_coll_checker.enable_obstacle(obstacle_name, enable=False)

            return True
        self.logger.error(f"cuRobo attach_objects_to_robot failed for {object_name}")
        return False

    def _detach_objects(self) -> None:
        """Detach every carried object and restore its world collision geometry."""
        link_names = set(self.attached_objects.values())
        for object_name in self.attached_objects:
            self.motion_gen.world_coll_checker.enable_obstacle(self._object_obstacles[object_name], enable=True)
        for link_name in link_names:
            self.motion_gen.kinematics.kinematics_config.enable_link_spheres(link_name)
            self.motion_gen.detach_object_from_robot(link_name=link_name)
        self.attached_objects.clear()

    def _get_current_joint_state_for_curobo(self) -> JointState:
        """
        Construct the current joint state for cuRobo with zero velocity and acceleration.

        This helper reads the robot's joint positions from Isaac Lab for the current environment
        and pairs them with zero velocities and accelerations as required by cuRobo planning.
        All tensors are moved to the cuRobo device and reordered to match the kinematic chain
        used by the cuRobo motion generator.

        Returns:
            JointState on the cuRobo device, ordered according to
            `self.motion_gen.kinematics.joint_names`, with position from the robot
            and zero velocity/acceleration.
        """
        # Fetch joint position (shape: [1, num_joints])
        joint_pos_raw: torch.Tensor = self.robot.data.joint_pos.torch[self.env_id, :].unsqueeze(0)
        joint_vel_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_acc_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)

        # Move to cuRobo device
        joint_pos: torch.Tensor = self._to_curobo_device(joint_pos_raw)
        joint_vel: torch.Tensor = self._to_curobo_device(joint_vel_raw)
        joint_acc: torch.Tensor = self._to_curobo_device(joint_acc_raw)

        cu_js: JointState = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc,
            joint_names=self.robot.data.joint_names,
            tensor_args=self.tensor_args,
        )
        return cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def get_ee_pose(self, joint_state: JointState) -> Pose:
        """Compute end-effector pose from joint configuration.

        Uses cuRobo's forward kinematics to calculate the end-effector pose
        at the specified joint configuration. Handles device conversion to ensure
        compatibility with cuRobo's CUDA-based computations.

        Args:
            joint_state: Robot joint configuration to compute end-effector pose from

        Returns:
            End-effector pose in world coordinates
        """
        cuda_position = self._to_curobo_device(joint_state.position)
        cuda_joint_state = JointState(
            position=cuda_position,
            velocity=(
                self._to_curobo_device(joint_state.velocity.detach().clone())
                if joint_state.velocity is not None
                else torch.zeros_like(cuda_position)
            ),
            acceleration=(
                self._to_curobo_device(joint_state.acceleration.detach().clone())
                if joint_state.acceleration is not None
                else torch.zeros_like(cuda_position)
            ),
            joint_names=joint_state.joint_names,
            tensor_args=self.tensor_args,
        )

        kin_state: Any = self.motion_gen.rollout_fn.compute_kinematics(cuda_joint_state)
        return kin_state.ee_pose

    def _make_pose(
        self,
        position: torch.Tensor | list[float],
        quaternion: torch.Tensor | None = None,
    ) -> Pose:
        """Create a CuRobo pose from an Isaac Lab ``xyzw`` pose.

        Args:
            position: Translation [m].
            quaternion: Quaternion in ``xyzw`` order, or ``None`` for identity.

        Returns:
            Pose on the configured CuRobo device.
        """
        position = torch.as_tensor(position, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        quaternion_wxyz = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device)
            if quaternion is None
            else torch.roll(self._to_curobo_device(quaternion), shifts=1, dims=-1)
        )
        return Pose(position=position, quaternion=quaternion_wxyz)

    def _set_active_links(self, links: list[str], active: bool) -> None:
        """Configure collision checking for specific robot links.

        Enables or disables collision sphere checking for the specified links.
        This is essential for contact scenarios where certain links (like fingers
        or attachment points) need collision checking disabled to allow contact
        with objects being grasped.

        Args:
            links: List of link names to enable or disable collision checking for
            active: True to enable collision checking, False to disable
        """
        for link in links:
            if active:
                self.motion_gen.kinematics.kinematics_config.enable_link_spheres(link)
            else:
                self.motion_gen.kinematics.kinematics_config.disable_link_spheres(link)

    def plan_motion(
        self,
        target_pose: torch.Tensor,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        """Plan collision-free motion to target pose.

        Plans a trajectory from the current robot configuration to the specified target pose.
        The method assumes that world updates and locked joint configurations have already
        been handled. Supports optional linear retiming for consistent execution speeds.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            step_size: Step size for linear retiming, enables retiming if provided
            enable_retiming: Whether to enable linear retiming, auto-detected from step_size if None

        Returns:
            True if planning succeeded and a valid trajectory was found, False otherwise
        """
        if enable_retiming is None:
            enable_retiming = step_size is not None

        # Ensure target pose is on cuRobo device (CUDA) for device isolation
        target_pose_cuda = self._to_curobo_device(target_pose)

        target_pos: torch.Tensor
        target_rot: torch.Tensor
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose_cuda)
        target_curobo_pose: Pose = self._make_pose(
            position=target_pos,
            quaternion=PoseUtils.quat_from_matrix(target_rot),
        )

        start_state: JointState = self._get_current_joint_state_for_curobo()

        self.logger.debug(f"Retiming enabled: {enable_retiming}, Step size: {step_size}")

        success: bool = self._plan_to_contact(
            start_state=start_state,
            goal_pose=target_curobo_pose,
            retreat_distance=self.config.retreat_distance,
            approach_distance=self.config.approach_distance,
            retime_plan=enable_retiming,
            step_size=step_size,
            contact=False,
        )

        return success

    def _plan_to_contact_pose(
        self,
        start_state: JointState,
        goal_pose: Pose,
        contact: bool = True,
    ) -> bool:
        """Plan motion with configurable collision checking for contact scenarios.

        Plans a trajectory while optionally disabling collision checking for hand links and
        attached objects. This is crucial for grasping and placing operations where contact
        is expected and collision checking would prevent successful planning.

        Args:
            start_state: Starting joint configuration for planning
            goal_pose: Target pose to reach in cuRobo coordinate frame
            contact: True to disable hand/attached object collisions for contact planning

        Returns:
            True if planning succeeded, False if no valid trajectory found
        """
        # Use configured hand link names instead of hardcoded ones
        disable_link_names: list[str] = self.config.hand_link_names.copy()
        link_spheres: dict[str, torch.Tensor] = {}

        if contact:
            # Store current spheres for the attached link so we can restore later
            attached_links = list(set(self.attached_objects.values()))
            for attached_link in attached_links:
                link_spheres[attached_link] = self.motion_gen.kinematics.kinematics_config.get_link_spheres(
                    attached_link
                ).clone()

            self.logger.debug(f"Attached link: {attached_links}")
            # Disable all specified links for contact planning
            self.logger.debug(f"Disable link names: {disable_link_names}")
            self._set_active_links(disable_link_names + attached_links, active=False)
        else:
            self.logger.debug(f"Disable link names: {disable_link_names}")

        try:
            result: Any = self.motion_gen.plan_single(start_state, goal_pose, self.plan_config)
            if not result.success.item():
                self.logger.debug(f"Contact planning failed: {result.status}")
                return False
            if result.optimized_plan is not None and len(result.optimized_plan.position) != 0:
                self._current_plan = result.optimized_plan
                self.logger.debug(f"Using optimized plan with {len(self._current_plan.position)} waypoints")
            else:
                self._current_plan = result.get_interpolated_plan()
                self.logger.debug(f"Using interpolated plan with {len(self._current_plan.position)} waypoints")

            self._current_plan = self.motion_gen.get_full_js(self._current_plan)
            common_js_names = [name for name in self.robot.data.joint_names if name in self._current_plan.joint_names]
            self._current_plan = self._current_plan.get_ordered_joint_state(common_js_names)
            self._plan_index = 0
            self.logger.debug(f"Contact planning succeeded with {len(self._current_plan.position)} waypoints")
            return True
        finally:
            if contact:
                self._set_active_links(disable_link_names, active=True)
                for attached_link, spheres in link_spheres.items():
                    self.motion_gen.kinematics.kinematics_config.update_link_spheres(attached_link, spheres)

    def _plan_to_contact(
        self,
        start_state: JointState,
        goal_pose: Pose,
        retreat_distance: float,
        approach_distance: float,
        contact: bool = False,
        retime_plan: bool = False,
        step_size: float | None = None,
    ) -> bool:
        """Execute multi-phase contact planning with approach and retreat phases.

        Implements a planning strategy for manipulation tasks that require approach and contact handling.
        Plans multiple trajectory segments with different collision checking configurations.

        Args:
            start_state: Starting joint state for planning
            goal_pose: Target pose to reach
            retreat_distance: Distance to retreat before transition to contact
            approach_distance: Distance to approach before final pose
            contact: Whether to enable contact planning mode
            retime_plan: Whether to retime the resulting plan
            step_size: Step size for retiming (only used if retime_plan is True)

        Returns:
            True if all planning phases succeeded, False if any phase failed
        """
        self.logger.debug(f"Multi-phase planning: retreat={retreat_distance}, approach={approach_distance}")

        target_poses: list[Pose] = []
        contacts: list[bool] = []

        if retreat_distance > 0:
            ee_pose: Pose = self.get_ee_pose(start_state)
            retreat_pose: Pose = ee_pose.multiply(
                self._make_pose(
                    position=[0.0, 0.0, -retreat_distance],
                )
            )
            target_poses.append(retreat_pose)
            contacts.append(True)
        contacts.append(contact)
        if approach_distance > 0:
            approach_pose: Pose = goal_pose.multiply(
                self._make_pose(
                    position=[0.0, 0.0, -approach_distance],
                )
            )
            target_poses.append(approach_pose)
            contacts.append(True)

        target_poses.append(goal_pose)

        current_state: JointState = start_state
        full_plan: JointState | None = None

        for i, (target_pose, contact_flag) in enumerate(zip(target_poses, contacts, strict=True)):
            self.logger.debug(
                f"Planning phase {i + 1} of {len(target_poses)}: contact={contact_flag} (collision"
                f" {'disabled' if contact_flag else 'enabled'})"
            )

            success: bool = self._plan_to_contact_pose(
                start_state=current_state,
                goal_pose=target_pose,
                contact=contact_flag,
            )

            if not success:
                self.logger.debug(f"Phase {i + 1} planning failed")
                return False

            if full_plan is None:
                full_plan = self._current_plan
            else:
                full_plan = full_plan.stack(self._current_plan)

            last_waypoint: torch.Tensor = self._current_plan.position[-1]
            current_state = JointState(
                position=last_waypoint.unsqueeze(0),
                velocity=torch.zeros_like(last_waypoint.unsqueeze(0)),
                acceleration=torch.zeros_like(last_waypoint.unsqueeze(0)),
                joint_names=self._current_plan.joint_names,
            )
            current_state = current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        self._current_plan = full_plan
        self._plan_index = 0

        if retime_plan and step_size is not None:
            original_length: int = len(self._current_plan.position)
            self._current_plan = self._linearly_retime_plan(step_size=step_size, plan=self._current_plan)
            self.logger.debug(
                f"Retimed complete plan from {original_length} to {len(self._current_plan.position)} waypoints"
            )

        self.logger.debug(f"Multi-phase planning succeeded with {len(self._current_plan.position)} total waypoints")

        return True

    def _linearly_retime_plan(
        self,
        plan: JointState,
        step_size: float,
    ) -> JointState:
        """Apply linear retiming to trajectory for consistent execution speed.

        Resamples the trajectory with uniform spacing between waypoints to ensure
        consistent motion speed during execution.

        Args:
            plan: Trajectory to retime.
            step_size: Desired spacing between waypoints in joint space [rad].

        Returns:
            Retimed trajectory with uniform waypoint spacing.
        """
        if len(plan.position) == 0:
            return plan

        path = plan.position

        if len(path) <= 1:
            return plan

        deltas = path[1:] - path[:-1]
        distances = torch.linalg.norm(deltas, dim=-1)

        waypoints = [path[0]]
        for distance, waypoint in zip(distances, path[1:], strict=True):
            if distance > 1e-6:
                waypoints.append(waypoint)

        if len(waypoints) <= 1:
            return plan

        waypoints = torch.stack(waypoints)

        deltas = waypoints[1:] - waypoints[:-1]
        distances = torch.linalg.norm(deltas, dim=-1)
        cum_distances = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, dim=0)])
        if cum_distances[-1] < 1e-6:
            return plan

        total_distance = cum_distances[-1]
        num_steps = int(torch.ceil(total_distance / step_size).item()) + 1

        # Create linearly spaced distances
        sampled_distances = torch.linspace(cum_distances[0], cum_distances[-1], num_steps, device=cum_distances.device)

        # Linear interpolation
        indices = torch.searchsorted(cum_distances, sampled_distances)
        indices = torch.clamp(indices, 1, len(cum_distances) - 1)

        # Get interpolation weights
        weights = (sampled_distances - cum_distances[indices - 1]) / (
            cum_distances[indices] - cum_distances[indices - 1]
        )
        weights = weights.unsqueeze(-1)

        # Interpolate waypoints
        sampled_waypoints = (1 - weights) * waypoints[indices - 1] + weights * waypoints[indices]

        self.logger.debug(
            f"Retiming: {len(path)} to {len(sampled_waypoints)} waypoints, "
            f"Distance: {total_distance:.3f}, Step size: {step_size}"
        )

        retimed_plan = JointState(
            position=sampled_waypoints,
            velocity=torch.zeros(
                (len(sampled_waypoints), plan.velocity.shape[-1]),
                device=plan.velocity.device,
                dtype=plan.velocity.dtype,
            ),
            acceleration=torch.zeros(
                (len(sampled_waypoints), plan.acceleration.shape[-1]),
                device=plan.acceleration.device,
                dtype=plan.acceleration.dtype,
            ),
            joint_names=plan.joint_names,
        )

        return retimed_plan

    def has_next_waypoint(self) -> bool:
        """Check if more waypoints remain in the current trajectory.

        Returns:
            True if there are unprocessed waypoints, False if trajectory is complete or empty
        """
        return self._current_plan is not None and self._plan_index < len(self._current_plan.position)

    def get_next_waypoint_ee_pose(self) -> Pose:
        """Get end-effector pose for the next waypoint in the trajectory.

        Advances the trajectory execution index and computes the end-effector pose
        for the next waypoint using forward kinematics.

        Returns:
            End-effector pose for the next waypoint in world coordinates

        Raises:
            IndexError: If no more waypoints remain in the trajectory
        """
        if not self.has_next_waypoint():
            raise IndexError("No more waypoints in the plan.")
        next_joint_state: JointState = self._current_plan[self._plan_index]
        self._plan_index += 1
        eef_state: CudaRobotModelState = self.motion_gen.compute_kinematics(next_joint_state)
        return eef_state.ee_pose

    def reset_plan(self) -> None:
        """Reset trajectory execution state.

        Clears the current trajectory and resets the execution index to zero.
        This prepares the planner for a new planning operation.
        """
        self._plan_index = 0
        self._current_plan = None

    def get_planned_poses(self) -> list[torch.Tensor]:
        """Extract all end-effector poses from current trajectory.

        Computes end-effector poses for all waypoints in the current trajectory without
        affecting the execution state. Optionally repeats the final pose multiple times
        if configured for stable goal reaching.

        Returns:
            List of end-effector poses as 4x4 transformation matrices, with optional repetition
        """
        if self._current_plan is None:
            return []

        planned_poses: list[torch.Tensor] = []
        for index in range(len(self._current_plan.position)):
            next_joint_state: JointState = self._current_plan[index]
            eef_state: CudaRobotModelState = self.motion_gen.compute_kinematics(next_joint_state)
            planned_pose = eef_state.ee_pose
            position = self._to_env_device(planned_pose.position)
            rotation = self._to_env_device(planned_pose.get_rotation())
            planned_poses.append(PoseUtils.make_pose(position, rotation)[0])

        if self.n_repeat is not None and self.n_repeat > 0 and len(planned_poses) > 0:
            self.logger.info(f"Repeating final pose {self.n_repeat} times")
            final_pose: torch.Tensor = planned_poses[-1]
            planned_poses.extend([final_pose] * self.n_repeat)

        return planned_poses

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        """Update the declared world and plan with the expected object attachment.

        Args:
            target_pose: Target end-effector pose as a 4-by-4 transformation matrix.
            expected_attached_object: Declared scene object to attach, or ``None``.
            step_size: Joint-space step size for linear retiming [rad].
            enable_retiming: Whether to linearly retime the trajectory.

        Returns:
            Whether planning succeeded.
        """
        self.reset_plan()
        self.update_world()
        self._set_gripper_state(expected_attached_object is not None)
        if expected_attached_object is None:
            return self.plan_motion(target_pose, step_size, enable_retiming)

        try:
            obstacle_name = self._object_obstacles[expected_attached_object]
        except KeyError as exc:
            raise ValueError(
                f"Attached object {expected_attached_object!r} is not declared in CuroboPlannerCfg.mesh_prim_paths."
            ) from exc
        gripper_position = self.robot.data.joint_pos.torch[self.env_id, -2:]
        if gripper_position[0].item() >= self.config.grasp_gripper_open_val:
            self.logger.info(f"Object {expected_attached_object} is not grasped")
            return False
        if not self._attach_object(expected_attached_object, obstacle_name):
            return False
        try:
            return self.plan_motion(target_pose, step_size, enable_retiming)
        finally:
            self._detach_objects()

    def _set_gripper_state(self, has_attached_objects: bool) -> None:
        """Configure gripper joint positions based on object attachment status.

        Sets the gripper to closed position when objects are attached and open position
        when no objects are attached. This ensures proper collision checking and planning
        with the correct gripper configuration.

        Args:
            has_attached_objects: True if robot currently has attached objects requiring closed gripper
        """
        if has_attached_objects:
            # Closed gripper for grasping
            locked_joints = self.config.gripper_closed_positions
        else:
            # Open gripper for manipulation
            locked_joints = self.config.gripper_open_positions

        self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)
