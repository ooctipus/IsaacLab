# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from dataclasses import MISSING

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

if typing.TYPE_CHECKING:
    from .curobo_planner import CuroboPlanner


@configclass
class CuroboCollisionCfg:
    """One clone-plan collision target consumed by CuRobo."""

    name: str = MISSING
    """Unique obstacle name in the CuRobo world."""

    prim_expr: str = MISSING
    """Full prim-path expression whose collision geometry the clone plan declares."""

    scene_entity: str | None = None
    """Rigid-object scene key for live pose updates, or ``None`` for static geometry."""


@configclass
class CuroboPlannerCfg:
    """Data required to construct a CuRobo planner for Franka cube stacking."""

    class_type: type[CuroboPlanner] | str = "{DIR}.curobo_planner:CuroboPlanner"
    """Planner implementation constructed as ``class_type(cfg, env, env_id)``."""

    robot_config_file: str = "franka.yml"
    """Robot configuration file in CuRobo's robot-config directory."""

    robot_urdf_file: str = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf"
    """Local or remote URDF used instead of the URDF named by ``robot_config_file``."""

    robot_entity: str = "robot"
    """Robot articulation key in the Isaac Lab scene."""

    mesh_prim_paths: list[CuroboCollisionCfg] = [
        CuroboCollisionCfg("table", "{ENV_REGEX_NS}/Table"),
        CuroboCollisionCfg("cube_1", "{ENV_REGEX_NS}/Cube_1", "cube_1"),
        CuroboCollisionCfg("cube_2", "{ENV_REGEX_NS}/Cube_2", "cube_2"),
        CuroboCollisionCfg("cube_3", "{ENV_REGEX_NS}/Cube_3", "cube_3"),
    ]
    """Collision targets declared by the clone plan and passed to CuRobo."""

    gripper_open_positions: dict[str, float] = {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
    """Open gripper positions used to update CuRobo collision spheres [rad]."""

    gripper_closed_positions: dict[str, float] = {"panda_finger_joint1": 0.024, "panda_finger_joint2": 0.024}
    """Closed gripper positions used to update CuRobo collision spheres [rad]."""

    hand_link_names: list[str] = ["panda_leftfinger", "panda_rightfinger", "panda_hand"]
    """Links whose collision spheres are disabled for contact planning."""

    attached_object_link_name: str = "attached_object"
    """Robot link used for attached-object collision spheres."""

    num_trajopt_seeds: int = 12
    """Number of trajectory-optimization seeds."""

    num_graph_seeds: int = 12
    """Number of graph-search seeds."""

    interpolation_dt: float = 0.05
    """Waypoint interpolation time step [s]."""

    collision_cache_size: dict[str, int] = {"obb": 150, "mesh": 150}
    """Obstacle capacity for each CuRobo collision representation."""

    trajopt_tsteps: int = 32
    """Number of trajectory-optimization time steps."""

    collision_activation_distance: float = 0.01
    """Distance at which collision constraints become active [m]."""

    position_threshold: float = 0.005
    """Planning success threshold for translation [m]."""

    rotation_threshold: float = 0.05
    """Planning success threshold for rotation [rad]."""

    approach_distance: float = 0.05
    """Pre-contact approach distance [m]."""

    retreat_distance: float = 0.05
    """Post-contact retreat distance [m]."""

    grasp_gripper_open_val: float = 0.04
    """Gripper position treated as open during grasp detection [rad]."""

    enable_graph: bool = True
    """Whether graph search is enabled."""

    enable_graph_attempt: int = 5
    """Number of graph-search attempts."""

    max_planning_attempts: int = 1
    """Maximum planning attempts."""

    enable_finetune_trajopt: bool = True
    """Whether to fine-tune graph solutions with trajectory optimization."""

    time_dilation_factor: float = 0.6
    """Trajectory time-dilation factor."""

    surface_sphere_radius: float = 0.01
    """Attached-object surface sphere radius [m]."""

    n_repeat: int | None = None
    """Number of repetitions of the final planned waypoint."""

    motion_step_size: float | None = None
    """Joint-space step size used when linearly retiming a plan [rad]."""

    debug_planner: bool = False
    """Whether detailed planner logging is enabled."""

    collision_spheres_file: str = "spheres/franka_mesh.yml"
    """CuRobo robot collision-sphere configuration file."""

    attached_object_sphere_count: int = 100
    """Collision spheres allocated to the attached-object link."""

    motion_noise_scale: float = 0.02
    """Gaussian noise scale applied to planned waypoints [rad]."""

    cuda_device: int = 0
    """CUDA device index."""
