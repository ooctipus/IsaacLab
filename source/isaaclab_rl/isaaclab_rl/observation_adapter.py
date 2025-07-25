
import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv, VecEnvObs

def adapt_gym_single_space(obs_groups, observation_space: gym.spaces.Dict):
    num_actor_obs = 0
    for obs_group in obs_groups["policy"]:
        assert len(observation_space[obs_group].shape) == 1, "The Adapter module only supports 1D observations."
        num_actor_obs += observation_space[obs_group].shape[-1]

    num_critic_obs = 0
    for obs_group in obs_groups["critic"]:
        assert len(observation_space[obs_group].shape) == 1, "The Adapter module only supports 1D observations."
        num_critic_obs += observation_space[obs_group].shape[-1]

    new_gym_space_dict = gym.spaces.Dict()
    new_gym_space_dict["policy"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_actor_obs,))
    new_gym_space_dict["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_critic_obs,))
    return new_gym_space_dict

def process_observation(obs_groups, observation: VecEnvObs) -> VecEnvObs:
    policy_obs = []
    for obs_group in obs_groups["policy"]:
        policy_obs.append(observation[obs_group])
    
    critic_obs = []
    for obs_group in obs_groups["critic"]:
        critic_obs.append(observation[obs_group])

    return {"policy": torch.cat(policy_obs, dim=1), "critic": torch.cat(critic_obs, dim=1)}
    

class GroupToActorCriticGymObservationSpacePatch:

    def __init__(self, obs_groups: dict[str, list[str]]):
        self.obs_groups = obs_groups
        self.original_manager_based_configure_gym_env_spaces = ManagerBasedRLEnv._configure_gym_env_spaces
        self.original_manager_based_step = ManagerBasedRLEnv.step
        self.original_manager_based_reset = ManagerBasedRLEnv.reset
        self.original_direct_configure_gym_env_spaces = DirectRLEnv._configure_gym_env_spaces
        self.original_direct_based_step = DirectRLEnv.step
        self.original_direct_based_reset = DirectRLEnv.reset

    def apply_patch(self):
        
        def configure_group_obs_to_actor_ctiric_obs_gym_spaces(lab_env_self):
            if isinstance(lab_env_self, ManagerBasedRLEnv):
                self.original_manager_based_configure_gym_env_spaces(lab_env_self)
                lab_env_self.single_observation_space = adapt_gym_single_space(self.obs_groups, lab_env_self.single_observation_space)
                lab_env_self.observation_space = gym.vector.utils.batch_space(lab_env_self.single_observation_space, lab_env_self.num_envs)
            else:
                lab_env_self.single_observation_space = adapt_gym_single_space(self.obs_groups, lab_env_self.single_observation_space)
                lab_env_self.observation_space = gym.vector.utils.batch_space(lab_env_self.single_observation_space, lab_env_self.num_envs)
                if lab_env_self.state_space is not None:
                    lab_env_self.state_space = gym.vector.utils.batch_space(lab_env_self.single_observation_space["critic"], lab_env_self.num_envs)

        def observation_adapted_step(lab_env_self, action):
            if isinstance(lab_env_self, ManagerBasedRLEnv):
                obs_buf, rew_buf, terminated, time_out, extra = self.original_manager_based_step(lab_env_self, action=action)
                return process_observation(self.obs_groups, obs_buf), rew_buf, terminated, time_out, extra
            else:
                obs_buf, rew_buf, terminated, time_out, extra = self.original_direct_based_step(lab_env_self, action=action)
                return process_observation(self.obs_groups, obs_buf), rew_buf, terminated, time_out, extra
        
        def observation_adapted_reset(lab_env_self, *args, **kwargs):
            if isinstance(lab_env_self, ManagerBasedRLEnv):
                obs_buf, extra = self.original_manager_based_reset(lab_env_self, *args, **kwargs)
                return process_observation(self.obs_groups, obs_buf), extra
            else:
                obs_buf, extra = self.original_direct_based_reset(lab_env_self, *args, **kwargs)
                return process_observation(self.obs_groups, obs_buf), extra

        ManagerBasedRLEnv._configure_gym_env_spaces = configure_group_obs_to_actor_ctiric_obs_gym_spaces
        ManagerBasedRLEnv.step = observation_adapted_step
        ManagerBasedRLEnv.reset = observation_adapted_reset

        DirectRLEnv._configure_gym_env_spaces = configure_group_obs_to_actor_ctiric_obs_gym_spaces
        DirectRLEnv.step = observation_adapted_step
        DirectRLEnv.reset = observation_adapted_reset

    def remove_patch(self):

        ManagerBasedRLEnv._configure_gym_env_spaces = self.original_manager_based_configure_gym_env_spaces
        ManagerBasedRLEnv.step = self.original_manager_based_step
        ManagerBasedRLEnv.reset = self.original_manager_based_reset
        DirectRLEnv._configure_gym_env_spaces = self.original_direct_configure_gym_env_spaces
        DirectRLEnv.step = self.original_direct_based_step
        DirectRLEnv.reset = self.original_direct_based_reset
