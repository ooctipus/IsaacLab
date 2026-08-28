# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import (
        DirectRLEnv,
        ManagerBasedRLEnv,
    )

    with contextlib.suppress(ImportError):
        from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedRLEnvWarp


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the RSL-RL library

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        # NOTE: import here (not at module level) to avoid loading heavy env classes before Isaac Sim is initialized.
        from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv

        try:
            from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedEnvWarp, ManagerBasedRLEnvWarp
        except ImportError:
            DirectRLEnvWarp = None
            ManagerBasedEnvWarp = None
            ManagerBasedRLEnvWarp = None

        allowed_types = (ManagerBasedRLEnv, ManagerBasedEnv, DirectRLEnv)
        if DirectRLEnvWarp is not None:
            allowed_types += (DirectRLEnvWarp,)
        if ManagerBasedEnvWarp is not None:
            allowed_types += (ManagerBasedEnvWarp,)
        if ManagerBasedRLEnvWarp is not None:
            allowed_types += (ManagerBasedRLEnvWarp,)

        if not isinstance(env.unwrapped, allowed_types):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv / DirectRLEnv / DirectRLEnvWarp /"
                " ManagerBasedRLEnvWarp. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv | DirectRLEnvWarp | ManagerBasedRLEnvWarp:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf.copy_(value)

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        return TensorDict(self.unwrapped.obs_buf, batch_size=[self.num_envs])

    def get_state_curriculum(self) -> object | None:
        """Return the state curriculum exposed by the environment, if present."""
        return getattr(self.unwrapped, "state_curriculum", None)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    def print_nonfinite_diagnostics(self, max_envs: int = 4) -> None:
        """Print state associated with non-finite environment observations."""
        env = self.unwrapped
        obs_buf = env.obs_buf
        failed_env_ids: list[torch.Tensor] = []

        print(
            f"[nonfinite] rank={os.getenv('RANK', '0')} device={self.device} "
            f"step={getattr(env, 'common_step_counter', 'unknown')}",
            flush=True,
        )
        for group_name, group_obs in obs_buf.items():
            tensors = group_obs.items() if isinstance(group_obs, dict) else ((group_name, group_obs),)
            for tensor_name, tensor in tensors:
                invalid = ~torch.isfinite(tensor)
                if not invalid.any():
                    continue
                locations = invalid.nonzero()[:32]
                failed_env_ids.append(locations[:, 0])
                print(
                    f"[nonfinite] observation={group_name}/{tensor_name} shape={tuple(tensor.shape)} "
                    f"count={int(invalid.sum())} locations={locations.cpu().tolist()}",
                    flush=True,
                )

                if isinstance(group_obs, torch.Tensor):
                    self._print_observation_terms(group_name, tensor, invalid)

        if not failed_env_ids:
            print("[nonfinite] no non-finite values remain in the environment observation buffer", flush=True)
            return

        env_ids = torch.unique(torch.cat(failed_env_ids))[:max_envs]
        print(f"[nonfinite] affected_env_ids={env_ids.cpu().tolist()}", flush=True)
        for name in ("episode_length_buf", "reset_buf", "reset_terminated", "reset_time_outs", "reward_buf"):
            value = getattr(env, name, None)
            if isinstance(value, torch.Tensor):
                self._print_tensor(name, value, env_ids)

        action_manager = getattr(env, "action_manager", None)
        if action_manager is not None:
            self._print_tensor("action", action_manager.action, env_ids)
            self._print_tensor("previous_action", action_manager.prev_action, env_ids)

        termination_manager = getattr(env, "termination_manager", None)
        if termination_manager is not None:
            for env_id in env_ids.cpu().tolist():
                terms = {name: values[0] for name, values in termination_manager.get_active_iterable_terms(env_id)}
                print(f"[nonfinite] env={env_id} terminations={terms}", flush=True)

        context = getattr(env, "extras", {}).get("diagnostics", {})
        for name, value in context.items():
            if isinstance(value, torch.Tensor):
                self._print_tensor(f"context.{name}", value, env_ids)
            else:
                print(f"[nonfinite] context.{name}={value}", flush=True)

        scene = getattr(env, "scene", None)
        if scene is not None:
            for name, asset in (*scene.articulations.items(), *scene.rigid_objects.items()):
                variant_ids = getattr(asset, "mesh_variant_ids", None)
                if variant_ids is not None:
                    self._print_tensor(f"{name}.mesh_variant_ids", variant_ids.torch, env_ids)
                fields = ["root_state_w", "body_link_pose_w", "body_link_vel_w"]
                if name in scene.articulations:
                    fields += ["joint_pos", "joint_vel"]
                for field in fields:
                    try:
                        value = getattr(asset.data, field).torch
                        self._print_tensor(f"{name}.{field}", value, env_ids)
                        if field == "root_state_w":
                            self._print_tensor(f"{name}.root_quat_norm", value[:, 3:7].norm(dim=-1), env_ids)
                        elif field == "body_link_pose_w":
                            self._print_tensor(f"{name}.body_quat_norm", value[..., 3:7].norm(dim=-1), env_ids)
                    except Exception as exc:
                        print(f"[nonfinite] failed to read {name}.{field}: {exc}", flush=True)

        physics_manager = getattr(getattr(env, "sim", None), "physics_manager", None)
        get_solver_tensors = getattr(physics_manager, "_get_nonfinite_diagnostic_tensors", None)
        if get_solver_tensors is not None:
            try:
                for name, value in get_solver_tensors().items():
                    self._print_tensor(f"mjwarp.{name}", value, env_ids)
            except Exception as exc:
                print(f"[nonfinite] failed to read MJWarp state: {exc}", flush=True)
        get_reset_mask = getattr(physics_manager, "get_solver_reset_required", None)
        if get_reset_mask is not None:
            try:
                self._print_tensor("solver_reset_required", get_reset_mask(), env_ids)
            except Exception as exc:
                print(f"[nonfinite] failed to read solver reset mask: {exc}", flush=True)

    def _print_observation_terms(self, group_name: str, tensor: torch.Tensor, invalid: torch.Tensor) -> None:
        manager = getattr(self.unwrapped, "observation_manager", None)
        if manager is None or not manager.group_obs_concatenate.get(group_name, False) or tensor.ndim != 2:
            return

        start = 0
        for term_name, shape in zip(manager.active_terms[group_name], manager.group_obs_term_dim[group_name]):
            width = shape[-1]
            term_invalid = invalid[:, start : start + width]
            if term_invalid.any():
                locations = term_invalid.nonzero()[:32]
                env_ids = torch.unique(locations[:, 0])[:4]
                print(
                    f"[nonfinite] term={group_name}/{term_name} columns={start}:{start + width} "
                    f"count={int(term_invalid.sum())} locations={locations.cpu().tolist()}",
                    flush=True,
                )
                self._print_tensor(f"observation.{group_name}.{term_name}", tensor[:, start : start + width], env_ids)
            start += width

    @staticmethod
    def _print_tensor(name: str, tensor: torch.Tensor, env_ids: torch.Tensor) -> None:
        selected = tensor[env_ids].detach()
        flat = selected.reshape(selected.shape[0], -1)
        if flat.is_floating_point() or flat.is_complex():
            finite = torch.isfinite(flat)
            finite_values = flat[finite]
            value_range = (
                f" min={float(finite_values.min()):.7g} max={float(finite_values.max()):.7g}"
                f" abs_max={float(finite_values.abs().max()):.7g}"
                if finite_values.numel()
                else ""
            )
            invalid = (~finite).nonzero()[:32].cpu().tolist()
        else:
            value_range = ""
            invalid = []
        print(
            f"[nonfinite] {name} shape={tuple(selected.shape)}{value_range} invalid={invalid}",
            flush=True,
        )
        if flat.numel() <= 256:
            print(f"[nonfinite] {name}.values={selected.cpu().tolist()}", flush=True)

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
