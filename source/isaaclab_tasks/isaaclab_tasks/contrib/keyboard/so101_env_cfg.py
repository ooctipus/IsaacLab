# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .keyboards.keyboard_gen_cfg import KeyboardSpawnerCfg
from .keyboards.keyboard_geometry import generate_keyboard
from .keyboards.keyboard_pool import TYPING_KEYBOARD_VARIANTS
from .mdp.actions import NewtonRelativeJointPositionActionCfg
from .mdp.reset import KeyboardResetIKCfg
from .newton_selection import BODY, JOINT_COORD, JOINT_DOF, NewtonSelectorCfg

_REFERENCE_KEYBOARD = generate_keyboard(TYPING_KEYBOARD_VARIANTS[0])
_BACKSPACE_SLOT = next(key.slot for key in _REFERENCE_KEYBOARD.active_keys if key.label.lower() == "backspace")

ROBOT_Q = NewtonSelectorCfg(JOINT_COORD, path=".*/Robot/joints/.*", count_per_world=6)
ROBOT_QD = NewtonSelectorCfg(JOINT_DOF, path=".*/Robot/joints/.*", count_per_world=6)
KEY_Q = NewtonSelectorCfg(JOINT_COORD, path=".*/Keyboard/parts/part_.*/joints/key_.*_joint", count_per_world=108)
KEY_QD = NewtonSelectorCfg(JOINT_DOF, path=".*/Keyboard/parts/part_.*/joints/key_.*_joint", count_per_world=108)
KEY_BODIES = NewtonSelectorCfg(BODY, path=".*/Keyboard/parts/part_.*/keys/key_.*", count_per_world=108)
ROBOT_ROOT = NewtonSelectorCfg(BODY, path=".*/Robot/base", count_per_world=1)
KEYBOARD_ROOT = NewtonSelectorCfg(BODY, path=".*/Keyboard/parts/part_.*/base_link", count_per_world=18)
ARM_PATH = ".*/Robot/joints/(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"


##
# Pre-defined configs
##
from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """Authored robot and 18 six-key partitions, without runtime asset views."""

    robot: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.copy(),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.0, 0.0, 2**-0.5, 2**-0.5)),
    )
    keyboard: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Keyboard",
        spawn=TYPING_KEYBOARD_VARIANTS[0].copy(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.285, 0.0, 0.01), rot=(0.0, 0.0, -0.7071068, 0.7071068)),
    )

    # contact sensor
    robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_pose=False,
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    typing = mdp.LetterTypingCommandCfg(
        keys=KEY_Q,
        key_dofs=KEY_QD,
        key_bodies=KEY_BODIES,
        robot_joints=ROBOT_Q,
        robot_dofs=ROBOT_QD,
        reset_roots=NewtonSelectorCfg(BODY, path=(".*/Robot/base", KEYBOARD_ROOT.path), count_per_world=19),
        reset_coords=NewtonSelectorCfg(JOINT_COORD, path=(ROBOT_Q.path, KEY_Q.path), count_per_world=114),
        reset_dofs=NewtonSelectorCfg(JOINT_DOF, path=(ROBOT_QD.path, KEY_QD.path), count_per_world=114),
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        letter_length=(1, 5),
        max_len=5,
        command_mode="letter_full",
        typeable_slots=tuple(key.slot for key in _REFERENCE_KEYBOARD.active_keys if key.slot != _BACKSPACE_SLOT),
        backspace_slot=_BACKSPACE_SLOT,
        slot_labels=tuple(key.label for key in _REFERENCE_KEYBOARD.keys),
        reset=mdp.LetterTypingCommandCfg.ResetCfg(
            enabled=True,
            ik=KeyboardResetIKCfg(
                joints=NewtonSelectorCfg(JOINT_COORD, path=ARM_PATH, count_per_world=5),
                dofs=NewtonSelectorCfg(JOINT_DOF, path=ARM_PATH, count_per_world=5),
                body=NewtonSelectorCfg(BODY, path=".*/Robot/gripper", count_per_world=1),
                tip_offset=(-0.0079, -0.000218121, -0.0981274),
            ),
            ik_rpy_deg=(0.0, 45.0, 0.0),  # (roll, pitch, yaw) [deg]
            ik_hover_height=0.02,
            ik_iters=(1, 4),
            ik_seed_joint_noise=0.25,
            buffer_size=8192,
            normal_weight=0.1,
            pre_solve_reset=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": [-0.0, 0.0],
                        "y": [-0.0, 0.0],
                        "z": [0.015, 0.05],
                        "yaw": [-0.1, 0.1],
                        "roll": [0.0, 0.75],
                    },
                    "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                    "roots": KEYBOARD_ROOT,
                },
            ),
        ),
    )


@configclass
class SO101RelJointPosActionCfg:
    action = NewtonRelativeJointPositionActionCfg(asset_name="robot", joints=ROBOT_Q, dofs=ROBOT_QD, scale=0.02)


@configclass
class SO101ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_keys_onehot = ObsTerm(func=mdp.target_keys_onehot, params={"command_name": "typing"})
        typed_keys_onehot = ObsTerm(func=mdp.typed_keys_onehot, params={"command_name": "typing"})

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"joints": ROBOT_Q}, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"joints": ROBOT_QD}, noise=Unoise(n_min=-0.0, n_max=0.0))

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        key_positions = ObsTerm(
            func=mdp.key_positions_b,
            clip=(-2.0, 2.0),
            params={
                "keys": KEY_BODIES,
                "root": ROBOT_ROOT,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Reset-mode events (shared by all physics backends)."""

    reset_keyboard = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.0, 0.0],
                "y": [-0.0, 0.0],
                "z": [0.015, 0.05],
                "yaw": [-0.1, 0.1],
                "roll": [0.0, 0.75],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "roots": KEYBOARD_ROOT,
        },
    )


@configclass
class SO101ReorientRewardCfg:
    typing_progress = RewTerm(func=mdp.letter_typing_progress, weight=2.0, params={"command_name": "typing"})

    success = RewTerm(func=mdp.typing_success, weight=50.0, params={"command_name": "typing"})

    mechanical_power = RewTerm(func=mdp.mechanical_power, weight=-0.0005, params={"joints": ROBOT_QD})

    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-10, params={"term_keys": ["abnormal_robot"]})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=mdp.joint_vel_out_of_limit, params={"joints": ROBOT_QD})

    excessive_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "bodies": NewtonSelectorCfg(BODY, path=".*/Robot/.*", count_per_world=7),
            "sensor_name": "robot_contact",
            "threshold": 20.0,
        },
    )

    success = DoneTerm(func=mdp.typing_complete, params={"command_name": "typing"})


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=600,
            nconmax=600,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            use_mujoco_contacts=True,
            enable_sleeping=True,
        ),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp


@configclass
class SO101KeyboardEnvCfg(ManagerBasedRLEnvCfg):
    keyboard_variants: tuple[KeyboardSpawnerCfg, ...] = TYPING_KEYBOARD_VARIANTS
    """Registered reset variants; an empty tuple keeps the authored 108-key partitioned baseline."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    observations: SO101ObservationsCfg = SO101ObservationsCfg()
    actions: SO101RelJointPosActionCfg = SO101RelJointPosActionCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: SO101ReorientRewardCfg = SO101ReorientRewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg(), dt=0.01)

    def __post_init__(self):
        self.decimation = 4  # 100 Hz sim -> 25 Hz control
        self.episode_length_s = 6.0
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.85, -0.75, 1.0), lookat=(0.25, 0.0, 0.1), focal_length=28.0
        )

    def play_mode(self):
        """Enable typing markers for every playback environment."""
        super().play_mode()
        self.num_envs = 36
        self.commands.typing.debug_vis = True
