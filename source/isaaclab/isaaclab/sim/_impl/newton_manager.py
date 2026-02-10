# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import numpy as np
import re

import warp as wp
from newton import Axis, BroadPhaseMode, CollisionPipeline, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.sensors import SensorContact as NewtonContactSensor
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

logger = logging.getLogger(__name__)


# Debug functions disabled to avoid CUDA graph capture issues
# Use with use_cuda_graph=False for debugging
def _check_contacts_for_nan(contacts: Contacts, step_label: str = "") -> bool:
    """Debug function to check contacts for NaN or invalid values.
    
    NOTE: This function does GPU-to-CPU transfers which break CUDA graph capture.
    Only use with use_cuda_graph=False.
    
    Returns True if any issues found.
    """
    return False  # Disabled - uncomment below for debugging with use_cuda_graph=False
    # if contacts is None:
    #     return False
    # 
    # count = contacts.rigid_contact_count.numpy()[0]
    # if count == 0:
    #     return False
    # 
    # issues = []
    # 
    # # Check contact normals for zero vectors or NaN
    # normals = contacts.rigid_contact_normal.numpy()[:count]
    # norms = np.linalg.norm(normals, axis=1)
    # nan_normals = np.isnan(norms).sum()
    # zero_normals = (norms < 1e-10).sum()
    # if nan_normals > 0:
    #     issues.append(f"NaN normals: {nan_normals}")
    # if zero_normals > 0:
    #     issues.append(f"Zero normals: {zero_normals}")
    # 
    # # Check contact points for NaN
    # point0 = contacts.rigid_contact_point0.numpy()[:count]
    # point1 = contacts.rigid_contact_point1.numpy()[:count]
    # nan_points = np.isnan(point0).sum() + np.isnan(point1).sum()
    # if nan_points > 0:
    #     issues.append(f"NaN contact points: {nan_points}")
    # 
    # # Check for very large values (potential instability)
    # large_vals = (np.abs(point0) > 1000).sum() + (np.abs(point1) > 1000).sum()
    # if large_vals > 0:
    #     issues.append(f"Large contact positions (>1000): {large_vals}")
    # 
    # if issues:
    #     print(f"[CONTACT DEBUG {step_label}] Issues found in {count} contacts: {', '.join(issues)}")
    #     return True
    # return False


def _check_state_for_nan(state: State, step_label: str = "") -> bool:
    """Debug function to check state for NaN values.
    
    NOTE: This function does GPU-to-CPU transfers which break CUDA graph capture.
    Only use with use_cuda_graph=False.
    
    Returns True if any issues found.
    """
    return False  # Disabled - uncomment below for debugging with use_cuda_graph=False
    # issues = []
    # 
    # if state.joint_q is not None:
    #     q = state.joint_q.numpy()
    #     nan_q = np.isnan(q).sum()
    #     if nan_q > 0:
    #         issues.append(f"NaN joint_q: {nan_q}")
    # 
    # if state.joint_qd is not None:
    #     qd = state.joint_qd.numpy()
    #     nan_qd = np.isnan(qd).sum()
    #     if nan_qd > 0:
    #         issues.append(f"NaN joint_qd: {nan_qd}")
    # 
    # if state.body_q is not None:
    #     body_q = state.body_q.numpy()
    #     nan_body = np.isnan(body_q).sum()
    #     if nan_body > 0:
    #         issues.append(f"NaN body_q: {nan_body}")
    # 
    # if issues:
    #     print(f"[STATE DEBUG {step_label}] Issues found: {', '.join(issues)}")
    #     return True
    # return False


def flipped_match(x: str, y: str) -> re.Match | None:
    """Flipped match function.

    This function is used to match the contact partners' body/shape names with the body/shape names in the simulation.

    Args:
        x: The body/shape name in the simulation.
        y: The body/shape name in the contact view.

    Returns:
        The match object if the body/shape name is found in the contact view, otherwise None.
    """
    return re.match(y, x)


class NewtonManager:
    _builder: ModelBuilder = None
    _model: Model = None
    _device: str = "cuda:0"
    _dt: float = 1.0 / 200.0
    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _solver = None
    _state_0: State = None
    _state_1: State = None
    _state_temp: State = None
    _control: Control = None
    _on_init_callbacks: list = []
    _on_start_callbacks: list = []
    _contacts: Contacts = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _newton_contact_sensors: dict = {}  # Maps sensor_key to NewtonContactSensor
    _report_contacts: bool = False
    _graph = None
    _newton_stage_path = None
    _sim_time = 0.0
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _cfg: NewtonCfg = NewtonCfg()
    _solver_type: str = "mujoco_warp"
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"
    _num_envs: int = None
    _model_changes: set[int] = set()

    @classmethod
    def clear(cls):
        NewtonManager._builder = None
        NewtonManager._model = None
        NewtonManager._solver = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._state_temp = None
        NewtonManager._control = None
        NewtonManager._contacts = None
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._collision_pipeline = None
        NewtonManager._newton_contact_sensors = {}
        NewtonManager._report_contacts = False
        NewtonManager._graph = None
        NewtonManager._newton_stage_path = None
        NewtonManager._sim_time = 0.0
        NewtonManager._on_init_callbacks = []
        NewtonManager._on_start_callbacks = []
        NewtonManager._usdrt_stage = None
        # Only create new config if not during Python shutdown
        try:
            NewtonManager._cfg = NewtonCfg()
        except (ImportError, AttributeError, TypeError):
            NewtonManager._cfg = None
        NewtonManager._up_axis = "Z"
        NewtonManager._first_call = True
        NewtonManager._model_changes = set()

    @classmethod
    def set_builder(cls, builder):
        NewtonManager._builder = builder

    @classmethod
    def add_on_init_callback(cls, callback) -> None:
        NewtonManager._on_init_callbacks.append(callback)

    @classmethod
    def add_on_start_callback(cls, callback) -> None:
        NewtonManager._on_start_callbacks.append(callback)

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        NewtonManager._model_changes.add(change)

    @classmethod
    def start_simulation(cls) -> None:
        """Starts the simulation.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """

        print(f"[INFO] Builder: {NewtonManager._builder}")
        if NewtonManager._builder is None:
            NewtonManager.instantiate_builder_from_stage()
        print("[INFO] Running on init callbacks")
        for callback in NewtonManager._on_init_callbacks:
            callback()
        print(f"[INFO] Finalizing model on device: {NewtonManager._device}")
        NewtonManager._builder.up_axis = Axis.from_string(NewtonManager._up_axis)
        # Set smaller contact margin for manipulation examples (default 10cm is too large)
        NewtonManager._builder.default_shape_cfg.contact_margin = 0.01
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            NewtonManager._model = NewtonManager._builder.finalize(device=NewtonManager._device)
            NewtonManager._model.set_gravity(NewtonManager._gravity_vector)
            NewtonManager._model.num_envs = NewtonManager._num_envs
        NewtonManager._state_0 = NewtonManager._model.state()
        NewtonManager._state_1 = NewtonManager._model.state()
        NewtonManager._state_temp = NewtonManager._model.state()
        NewtonManager._control = NewtonManager._model.control()
        NewtonManager.forward_kinematics()
        # Initialize empty contacts - will be replaced in initialize_solver() if collision pipeline is needed
        NewtonManager._contacts = Contacts(0, 0)
        print("[INFO] Running on start callbacks")
        for callback in NewtonManager._on_start_callbacks:
            callback()
        if not NewtonManager._clone_physics_only:
            import usdrt

            NewtonManager._usdrt_stage = get_current_stage(fabric=True)
            for i, prim_path in enumerate(NewtonManager._model.body_key):
                prim = NewtonManager._usdrt_stage.GetPrimAtPath(prim_path)
                prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                prim.GetAttribute(NewtonManager._newton_index_attr).Set(i)
                xformable_prim = usdrt.Rt.Xformable(prim)
                if not xformable_prim.HasWorldXform():
                    xformable_prim.SetWorldXformFromUsd()

    @classmethod
    def instantiate_builder_from_stage(cls):
        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)
        builder = ModelBuilder(up_axis=up_axis)
        builder.add_usd(stage)
        NewtonManager.set_builder(builder)

    @classmethod
    def set_solver_settings(cls, newton_params: dict):
        NewtonManager._cfg = NewtonCfg(**newton_params)

    @classmethod
    def _create_collision_pipeline(cls) -> None:
        """Creates the unified collision pipeline and initial contacts.

        Uses EXPLICIT broadphase mode which properly honors excluded shape pairs
        (parent-child filtering). SAP/NXN modes don't correctly filter these pairs,
        causing instability in articulated systems.
        """
        NewtonManager._collision_pipeline = CollisionPipeline.from_model(
            NewtonManager._model, broad_phase_mode=BroadPhaseMode.EXPLICIT
        )
        NewtonManager._contacts = NewtonManager._model.collide(
            NewtonManager._state_0, collision_pipeline=NewtonManager._collision_pipeline
        )

    @classmethod
    def _create_mujoco_contacts(cls) -> None:
        """Creates a Contacts object for MuJoCo contact mode.

        When using MuJoCo's internal collision detection (use_mujoco_contacts=True),
        we still need a properly sized Contacts object for sensor evaluation.
        The solver's update_contacts() will populate this from MuJoCo data.
        """
        # Get the maximum contact capacity from the MuJoCo solver
        naconmax = NewtonManager._solver.mjw_data.naconmax
        # Create contacts with sufficient capacity and force attribute for sensors
        NewtonManager._contacts = Contacts(
            rigid_contact_max=naconmax,
            soft_contact_max=0,
            device=NewtonManager._device,
            requested_attributes={"force"},
        )

    @classmethod
    def initialize_solver(cls):
        """Initializes the solver and collision pipeline.

        This function initializes the solver based on the specified solver type. Currently, only XPBD and MuJoCoWarp
        are supported. If the solver requires external collision detection (i.e., not using MuJoCo's internal
        contacts), a unified collision pipeline is created.

        The graphing of the simulation is performed in this function if the simulation is ran using
        a CUDA enabled device.

        .. warning::
            When using a CUDA enabled device, the simulation will be graphed. This means that this function steps the
            simulation once to capture the graph. Hence, this function should only be called after everything else in
            the simulation is initialized.
        """
        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            NewtonManager._num_substeps = NewtonManager._cfg.num_substeps
            NewtonManager._solver_dt = NewtonManager._dt / NewtonManager._num_substeps
            NewtonManager._solver = NewtonManager._get_solver(NewtonManager._model, NewtonManager._cfg.solver_cfg)

            # Determine if we need external collision detection
            # - SolverMuJoCo with use_mujoco_contacts=True: uses internal MuJoCo collision detection
            # - SolverMuJoCo with use_mujoco_contacts=False: needs Newton's unified collision pipeline
            # - Other solvers (XPBD, Featherstone): always need Newton's unified collision pipeline
            if isinstance(NewtonManager._solver, SolverMuJoCo):
                use_mujoco_contacts = NewtonManager._cfg.solver_cfg.get("use_mujoco_contacts", False)
                NewtonManager._needs_collision_pipeline = not use_mujoco_contacts
            else:
                NewtonManager._needs_collision_pipeline = True

            # Create collision pipeline or MuJoCo contacts based on mode
            if NewtonManager._needs_collision_pipeline:
                NewtonManager._create_collision_pipeline()
            elif NewtonManager._report_contacts:
                # MuJoCo contacts mode with sensors registered: create proper Contacts object
                NewtonManager._create_mujoco_contacts()

        # Capture the graph if CUDA is enabled
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if NewtonManager._cfg.use_cuda_graph and NewtonManager._device.startswith("cuda"):
                with wp.ScopedCapture() as capture:
                    NewtonManager.simulate()
                NewtonManager._graph = capture.graph
            elif NewtonManager._cfg.use_cuda_graph and not NewtonManager._device.startswith("cuda"):
                logger.warning("CUDA graphs requested but device is CPU. Disabling CUDA graphs.")
                NewtonManager._cfg.use_cuda_graph = False

    @classmethod
    def simulate(cls) -> None:
        """Simulates the simulation.

        Performs one simulation step with the specified number of substeps. Depending on the solver type, this function
        may need to explicitly compute the collisions. This function also aggregates the contacts and evaluates the
        contact sensors. Finally, it performs the state swapping for Newton.
        """
        state_0_dict = NewtonManager._state_0.__dict__
        state_1_dict = NewtonManager._state_1.__dict__
        state_temp_dict = NewtonManager._state_temp.__dict__
        contacts = None

        # MJWarp computes its own collisions.
        if NewtonManager._needs_collision_pipeline:
            contacts = NewtonManager._model.collide(
                NewtonManager._state_0, collision_pipeline=NewtonManager._collision_pipeline
            )
            # Update class-level contacts for sensor evaluation
            NewtonManager._contacts = contacts
            
            # Debug: Check for invalid contacts
            if NewtonManager._cfg.debug_mode:
                _check_contacts_for_nan(contacts, "after_collide")
                _check_state_for_nan(NewtonManager._state_0, "before_step")

        if NewtonManager._num_substeps % 2 == 0:
            for i in range(NewtonManager._num_substeps):
                NewtonManager._solver.step(
                    NewtonManager._state_0,
                    NewtonManager._state_1,
                    NewtonManager._control,
                    contacts,
                    NewtonManager._solver_dt,
                )
                NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
                NewtonManager._state_0.clear_forces()
                
                # Debug: Check state after each substep
                if NewtonManager._cfg.debug_mode:
                    if _check_state_for_nan(NewtonManager._state_0, f"substep_{i}"):
                        print(f"[DEBUG] NaN appeared after substep {i}")
                        break
        else:
            for i in range(NewtonManager._num_substeps):
                NewtonManager._solver.step(
                    NewtonManager._state_0,
                    NewtonManager._state_1,
                    NewtonManager._control,
                    contacts,
                    NewtonManager._solver_dt,
                )

                # FIXME: Ask Lukasz help to deal with non-even number of substeps. This should not be needed.
                if i < NewtonManager._num_substeps - 1 or not NewtonManager._cfg.use_cuda_graph:
                    # we can just swap the state references
                    NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
                elif NewtonManager._cfg.use_cuda_graph:
                    # swap states by actually copying the state arrays to make sure the graph capture works
                    for key, value in state_0_dict.items():
                        if isinstance(value, wp.array):
                            if key not in state_temp_dict:
                                state_temp_dict[key] = wp.empty_like(value)
                            state_temp_dict[key].assign(value)
                            state_0_dict[key].assign(state_1_dict[key])
                            state_1_dict[key].assign(state_temp_dict[key])
                NewtonManager._state_0.clear_forces()

        # Transfer contact forces from solver to Newton contacts for sensor evaluation
        if NewtonManager._report_contacts:
            # For newton_contacts (unified pipeline): use locally computed contacts
            # For mujoco_contacts: use class-level _contacts, solver populates it from MuJoCo data
            eval_contacts = contacts if contacts is not None else NewtonManager._contacts
            NewtonManager._solver.update_contacts(eval_contacts, NewtonManager._state_0)
            for sensor in NewtonManager._newton_contact_sensors.values():
                sensor.eval(eval_contacts)

    @classmethod
    def set_device(cls, device: str) -> None:
        """Sets the device to use for the Newton simulation.

        Args:
            device (str): The device to use for the Newton simulation.
        """
        NewtonManager._device = device

    @classmethod
    def step(cls) -> None:
        """Steps the simulation.

        This function steps the simulation by the specified time step in the simulation configuration.
        """
        if NewtonManager._model_changes:
            for change in NewtonManager._model_changes:
                NewtonManager._solver.notify_model_changed(change)
            NewtonManager._model_changes = set()

        if NewtonManager._cfg.use_cuda_graph:
            wp.capture_launch(NewtonManager._graph)
        else:
            NewtonManager.simulate()

        if NewtonManager._cfg.debug_mode:
            convergence_data = NewtonManager.get_solver_convergence_steps()
            # print(f"solver niter: {convergence_data}")
            if convergence_data["max"] == NewtonManager._solver.mjw_model.opt.iterations:
                print("solver didn't converge!", convergence_data["max"])

        NewtonManager._sim_time += NewtonManager._solver_dt * NewtonManager._num_substeps

    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        niter = NewtonManager._solver.mjw_data.solver_niter.numpy()
        max_niter = np.max(niter)
        mean_niter = np.mean(niter)
        min_niter = np.min(niter)
        std_niter = np.std(niter)
        return {"max": max_niter, "mean": mean_niter, "min": min_niter, "std": std_niter}

    @classmethod
    def get_non_converged_env_ids(cls) -> np.ndarray | None:
        """Returns the environment IDs where the solver did not converge.
        
        This is useful for detecting simulation instability and taking corrective action
        (e.g., resetting those environments or terminating them early).
        
        Returns:
            numpy array of environment IDs where solver hit max iterations, or None if all converged.
        """
        if NewtonManager._solver is None or not hasattr(NewtonManager._solver, 'mjw_data'):
            return None
            
        niter = NewtonManager._solver.mjw_data.solver_niter.numpy()
        max_iter = NewtonManager._solver.mjw_model.opt.iterations
        
        # Find environments where solver hit max iterations
        non_converged = np.where(niter >= max_iter)[0]
        
        if len(non_converged) == 0:
            return None
        return non_converged

    @classmethod
    def set_simulation_dt(cls, dt: float) -> None:
        """Sets the simulation time step and the number of substeps.

        Args:
            dt (float): The simulation time step.
        """
        NewtonManager._dt = dt

    @classmethod
    def get_model(cls):
        return NewtonManager._model

    @classmethod
    def get_state_0(cls):
        return NewtonManager._state_0

    @classmethod
    def get_state_1(cls):
        return NewtonManager._state_1

    @classmethod
    def get_control(cls):
        return NewtonManager._control

    @classmethod
    def get_dt(cls):
        return NewtonManager._dt

    @classmethod
    def get_solver_dt(cls):
        return NewtonManager._solver_dt

    @classmethod
    def forward_kinematics(cls, mask: wp.array | None = None) -> None:
        """Evaluates the forward kinematics for the selected articulations.

        This function evaluates the forward kinematics for the selected articulations.
        """
        eval_fk(
            NewtonManager._model,
            NewtonManager._state_0.joint_q,
            NewtonManager._state_0.joint_qd,
            NewtonManager._state_0,
            None,
        )

    @classmethod
    def _get_solver(cls, model: Model, solver_cfg: dict) -> SolverBase:
        NewtonManager._solver_type = solver_cfg.pop("solver_type")

        if NewtonManager._solver_type == "mujoco_warp":
            return SolverMuJoCo(model, **solver_cfg)
        elif NewtonManager._solver_type == "xpbd":
            return SolverXPBD(model, **solver_cfg)
        elif NewtonManager._solver_type == "featherstone":
            return SolverFeatherstone(model, **solver_cfg)
        else:
            raise ValueError(f"Invalid solver type: {NewtonManager._solver_type}")

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        prune_noncolliding: bool = True,
        verbose: bool = False,
    ):
        """Adds a contact view.

        Adds a contact view to the simulation allowing to report contacts between the specified bodies/shapes and the
        contact partners. As of now, only one body/shape name expression can be provided. Similarly, only one contact
        partner body/shape expression can be provided. If no contact partner expression is provided, the contact view
        will report contacts with all bodies/shapes.

        Note that we make an explicit difference between a body and a shape. A body is a rigid body, while a shape
        is a collision shape. A body can have multiple shapes. The shape option allows a more fine-grained control
        over the contact reporting.

        Args:
            body_names_expr (str | None): The expression for the body names.
            shape_names_expr (str | None): The expression for the shape names.
            contact_partners_body_expr (str | None): The expression for the contact partners' body names.
            contact_partners_shape_expr (str | None): The expression for the contact partners' shape names.
            prune_noncolliding (bool): Make the force matrix sparse using the collision pairs in the model.
            verbose (bool): Whether to print verbose information.
        """
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")
        if contact_partners_body_expr is None and contact_partners_shape_expr is None:
            print(f"[INFO] Adding contact view for {body_names_expr}. It will report contacts with all bodies/shapes.")
        else:
            if body_names_expr is not None:
                if contact_partners_body_expr is not None:
                    print(f"[INFO] Adding contact view for {body_names_expr} with filter {contact_partners_body_expr}.")
                else:
                    print(f"[INFO] Adding contact view for {body_names_expr} with filter {shape_names_expr}.")
            else:
                if contact_partners_body_expr is not None:
                    print(
                        f"[INFO] Adding contact view for {shape_names_expr} with filter {contact_partners_body_expr}."
                    )
                else:
                    print(
                        f"[INFO] Adding contact view for {shape_names_expr} with filter {contact_partners_shape_expr}."
                    )

        # Create unique key for this sensor
        sensor_key = (body_names_expr, shape_names_expr, contact_partners_body_expr, contact_partners_shape_expr)

        # Create and store the sensor
        # Note: SensorContact constructor requests 'force' attribute from the model
        newton_sensor = NewtonContactSensor(
            NewtonManager._model,
            sensing_obj_bodies=body_names_expr,
            sensing_obj_shapes=shape_names_expr,
            counterpart_bodies=contact_partners_body_expr,
            counterpart_shapes=contact_partners_shape_expr,
            match_fn=flipped_match,
            include_total=True,
            prune_noncolliding=prune_noncolliding,
            verbose=verbose,
        )
        NewtonManager._newton_contact_sensors[sensor_key] = newton_sensor
        NewtonManager._report_contacts = True

        # Regenerate contacts to include force allocation requested by the sensor
        # The sensor requests 'force' attribute, so Contacts must be recreated
        if NewtonManager._collision_pipeline is not None:
            # Newton collision pipeline: regenerate contacts with force attribute
            NewtonManager._contacts = NewtonManager._model.collide(
                NewtonManager._state_0, collision_pipeline=NewtonManager._collision_pipeline
            )
        elif NewtonManager._solver is not None and isinstance(NewtonManager._solver, SolverMuJoCo):
            # MuJoCo contacts: create a properly sized Contacts object
            # Note: if solver not yet initialized, this will be done in initialize_solver()
            NewtonManager._create_mujoco_contacts()

        return sensor_key
