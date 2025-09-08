import gymnasium as gym

from . import agents

# State
gym.register(
    id="Pat-OneLeg-Ur5-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5:OneLegUr5IkAbs",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
        "state_machine_entry_point": f"{__name__}.state_machines.furniture_bench_ur5_state_machine:SmScrewTaskCfg",
        "state_machine_recorder_entry_point": (
            f"{__name__}.state_machines.recorders_cfg:EndEffectorStateRecorderManagerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5:OneLegUr5RelJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
        "state_machine_entry_point": f"{__name__}.state_machines.furniture_bench_ur5_state_machine:SmScrewTaskRelJointPositionAdapterCfg",
        "state_machine_recorder_entry_point": (
            f"{__name__}.state_machines.recorders_cfg:EndEffectorStateRecorderManagerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_finetune:OneLegUr5FinetuneRelJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Finetune-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_finetune:OneLegUr5FinetuneEvalRelJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5:OneLegUr5RelJointPositionDataCollection",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:OneLeg_DAggerRunnerCfg"
    },
    disable_env_checker=True,
)


gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5:OneLegUr5EvalRelJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
        "state_machine_entry_point": f"{__name__}.state_machines.furniture_bench_ur5_state_machine:SmScrewTaskRelJointPositionAdapterCfg",
        "state_machine_recorder_entry_point": (
            f"{__name__}.state_machines.recorders_cfg:EndEffectorStateRecorderManagerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRelUnscaled-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5:OneLegUr5EvalRelUnscaledJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

## Init States

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Reaching-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_init_states:OneLegUr5RelJointPositionReaching",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Grasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_init_states:OneLegUr5RelJointPositionGrasped",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Insertion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_init_states:OneLegUr5RelJointPositionInsertion",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Assembled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_init_states:OneLegUr5RelJointPositionAssembled",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-AssembledGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_init_states:OneLegUr5RelJointPositionAssembledGrasped",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

# RGB

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb:OneLegUr5RGBRelJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-DataCollection-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb:OneLegUr5DataCollectionRGBRelJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:OneLeg_DAggerRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRelUnscaled-Eval-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb:OneLegUr5EvalRGBRelUnscaledJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:OneLeg_DAggerRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosAbs-Eval-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb:OneLegUr5EvalRGBAbsJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

# RGB Init States

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Reaching-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_init_states:OneLegUr5RGBRelJointPositionReaching",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Grasped-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_init_states:OneLegUr5RGBRelJointPositionGrasped",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Insertion-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_init_states:OneLegUr5RGBRelJointPositionInsertion",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Assembled-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_init_states:OneLegUr5RGBRelJointPositionAssembled",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-AssembledGrasped-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_init_states:OneLegUr5RGBRelJointPositionAssembledGrasped",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

# RGB Init States Narrow

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Reaching-RGB-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_narrow_init_states:OneLegUr5RGBRelJointPositionReaching",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Grasped-RGB-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_narrow_init_states:OneLegUr5RGBRelJointPositionGrasped",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Insertion-RGB-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_narrow_init_states:OneLegUr5RGBRelJointPositionInsertion",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-Assembled-RGB-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_narrow_init_states:OneLegUr5RGBRelJointPositionAssembled",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Pat-OneLeg-Ur5-JointPosRel-AssembledGrasped-RGB-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_ur5_rgb_narrow_init_states:OneLegUr5RGBRelJointPositionAssembledGrasped",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)
