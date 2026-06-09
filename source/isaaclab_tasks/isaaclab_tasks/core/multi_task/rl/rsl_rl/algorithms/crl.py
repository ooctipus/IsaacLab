# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contrastive Reinforcement Learning (CRL) algorithm.

Implements the CRL algorithm (Eysenbach et al. 2022) with Hindsight Experience
Replay (Andrychowicz et al. 2017). Architecture follows Wang et al. 2025
(*1000 Layer Networks for Self-Supervised RL*).

The public API mirrors :class:`~rsl_rl.algorithms.PPO` so that
:class:`~rsl_rl.runners.OnPolicyRunner` can drive CRL without modification:

- :meth:`act` — sample actions from the squashed-Gaussian actor.
- :meth:`process_env_step` — store transitions in the replay buffer.
- :meth:`compute_returns` — no-op (off-policy, no bootstrapping).
- :meth:`update` — sample from buffer, HER-relabel, run N SGD steps.
- :meth:`construct_algorithm` — factory that builds all components from config.
"""

from __future__ import annotations

from math import prod

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.env import VecEnv
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict

from ..extensions import HindsightRelabeling, resolve_her_config
from ..models.residual_mlp import ResidualMLP
from ..storage import ReplayBuffer

# ---------------------------------------------------------------------------
# Shared per-group encoder (used by both actor and critic, optionally tied)
# ---------------------------------------------------------------------------


class _SharedStateEncoder(nn.Module):
    """Per-group MLP encoders applied selectively to obs groups.

    Maintains an ``nn.ModuleDict`` of encoders keyed by obs-group name (only
    groups listed in ``encoder_cfg`` get an encoder; the rest pass through).
    Provides two forward forms so both inference and update paths can share the
    same parameters:

    - :meth:`encode_dict` — encode a :class:`TensorDict`'s groups in a given order.
      Used by the actor at inference time (env emits a TensorDict).
    - :meth:`encode_flat` — encode a flat tensor that is the concatenation of those
      groups in the given order along the last dim. Used by the actor and the
      critic during training updates, where observations come from the replay
      buffer as flat slices.

    Sharing the same instance between actor and critic ties the parameters so
    the contrastive Q's view of state matches the policy's. CRL's factory does
    this when ``share_encoders=True`` and registers the encoder parameters with
    the critic optimizer only (the actor's loss back-propagates through but does
    not step encoder parameters).

    Args:
        group_specs: Ordered list of ``(name, raw_flat_dim)`` pairs covering all
            active observation groups across actor and critic. Order is the
            sorted-by-name order used elsewhere in CRL so flat-buffer offsets
            line up.
        encoder_cfg: Mapping ``group_name -> kwargs`` forwarded to
            :class:`~rsl_rl.modules.MLP`. Required keys: ``hidden_dims``,
            ``output_dim``. Groups absent from this mapping pass through.
    """

    def __init__(
        self,
        group_specs: list[tuple[str, int]],
        encoder_cfg: dict[str, dict] | None = None,
    ) -> None:
        super().__init__()
        encoder_cfg = encoder_cfg or {}
        self._raw_dims: dict[str, int] = {name: dim for name, dim in group_specs}
        encoders: dict[str, nn.Module] = {}
        encoded_dims: dict[str, int] = {}
        for name, raw_dim in group_specs:
            if name in encoder_cfg:
                cfg = dict(encoder_cfg[name])
                enc = MLP(input_dim=raw_dim, **cfg)
                encoders[name] = enc
                # ``MLP`` is an ``nn.Sequential``; the last linear layer's
                # ``out_features`` is the encoded dim.
                last_linear = None
                for m in enc.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
                if last_linear is None:
                    raise ValueError(f"Encoder for obs group '{name}' has no nn.Linear layer.")
                encoded_dims[name] = last_linear.out_features
            else:
                encoded_dims[name] = raw_dim
        self.encoders = nn.ModuleDict(encoders)
        self._encoded_dims = encoded_dims

    def total_raw_dim(self, groups: list[str]) -> int:
        """Sum of raw flat dims for the given groups."""
        return sum(self._raw_dims[g] for g in groups)

    def total_encoded_dim(self, groups: list[str]) -> int:
        """Sum of encoder-output dims (or raw dims for passthrough) for the given groups."""
        return sum(self._encoded_dims[g] for g in groups)

    def encode_dict(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        """Encode groups from a :class:`TensorDict`, concatenated in the given order."""
        parts = []
        for g in groups:
            x = obs[g].flatten(start_dim=1)
            if g in self.encoders:
                parts.append(self.encoders[g](x))
            else:
                parts.append(x)
        return torch.cat(parts, dim=-1)

    def encode_flat(self, flat: torch.Tensor, groups: list[str]) -> torch.Tensor:
        """Encode a flat tensor that is the concatenation of ``groups`` in order.

        Splits ``flat`` along the last dim using known per-group raw sizes,
        applies each group's encoder (or passes through), and concatenates.
        """
        parts = []
        cursor = 0
        for g in groups:
            raw_d = self._raw_dims[g]
            slice_ = flat[..., cursor : cursor + raw_d]
            cursor += raw_d
            if g in self.encoders:
                parts.append(self.encoders[g](slice_))
            else:
                parts.append(slice_)
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# CRL model classes (CRL-specific, not shared with other algorithms)
# ---------------------------------------------------------------------------


class SquashedGaussianActor(nn.Module):
    """SAC-style actor: ResidualMLP backbone split into mean/log_std, tanh squashed.

    Routes the observation through an optional :class:`_SharedStateEncoder` before
    the residual MLP. Multi-dim obs groups (e.g. a CNN-shaped height-scan) are
    flattened by the encoder, so the actor can ingest the same obs term that
    feeds a CNN policy in PPO. When the encoder is shared with the critic, this
    actor's forward pass uses the same compressed view of state that the
    contrastive Q sees.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        num_layers_per_block: int = 4,
        expand: int = 1,
        activation: str = "swish",
        norm: bool = True,
        encoders: _SharedStateEncoder | None = None,
        encoder_cfg: dict[str, dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        active_groups = sorted(obs_groups[obs_set])
        self.obs_groups = active_groups
        # Per-group raw flat dim = product of all non-batch dims, so 4D groups
        # (e.g. ``(B, 1, 76, 126)``) get the right total size in one place.
        group_specs = [(g, int(prod(obs[g].shape[1:]))) for g in active_groups]
        if encoders is None:
            encoders = _SharedStateEncoder(group_specs, encoder_cfg)
        self.encoders = encoders
        self._action_dim = output_dim
        # Input dim to the residual MLP = encoded total of all active groups.
        self.obs_dim = encoders.total_encoded_dim(active_groups)
        self.raw_obs_dim = encoders.total_raw_dim(active_groups)

        self.mlp = ResidualMLP(
            input_dim=self.obs_dim,
            output_dim=output_dim * 2,
            hidden_dim=hidden_dim,
            num_blocks=depth // num_layers_per_block,
            expand=expand,
            num_layers_per_block=num_layers_per_block,
            activation=activation,
            norm=norm,
        )

    def flatten_raw(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate active groups into ``(B, raw_obs_dim)`` (no encoders applied).

        Used by ``CRL.act`` to produce the raw flat row stored in the replay
        buffer — encoders are trainable, so the buffer must hold raw obs and
        encoders re-run on every forward.
        """
        return torch.cat([obs[k].flatten(start_dim=1) for k in self.obs_groups], dim=-1)

    def encode(self, obs) -> torch.Tensor:
        """Apply the shared encoder. Accepts a :class:`TensorDict` or a raw flat tensor."""
        if isinstance(obs, TensorDict):
            return self.encoders.encode_dict(obs, self.obs_groups)
        return self.encoders.encode_flat(obs, self.obs_groups)

    def _gaussian_params(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(encoded)
        mean, log_std = out.split(self._action_dim, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return ``(action, log_prob)``. Accepts TensorDict or raw flat."""
        encoded = self.encode(obs)
        mean, log_std = self._gaussian_params(encoded)
        std = log_std.exp()
        x_t = mean + std * torch.randn_like(std)
        action = torch.tanh(x_t)
        log_prob = (
            -0.5 * ((x_t - mean) / (std + 1e-8)).square()
            - log_std
            - 0.5 * 1.8378770664093453
            - torch.log(1 - action.square() + 1e-6)
        ).sum(-1)
        return action, log_prob

    def deterministic(self, obs) -> torch.Tensor:
        """Return ``tanh(mean)`` — no exploration noise. Accepts TensorDict or raw flat."""
        encoded = self.encode(obs)
        mean, _ = self._gaussian_params(encoded)
        return torch.tanh(mean)

    def forward(self, obs):
        """Inference forward — deterministic action."""
        return self.deterministic(obs)

    @property
    def output_std(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def as_jit(self):
        return self

    def as_onnx(self, *args, **kwargs):
        return self

    def reset(self, dones=None):
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass


class BilinearCritic(nn.Module):
    """CRL bilinear critic: ``Q(s,a,g) = -||SA_enc(s,a) - G_enc(g)||``.

    Routes raw flat ``state`` and ``goal`` slices (sourced from the replay
    buffer during update) through an optional :class:`_SharedStateEncoder`
    before the SA / G residual MLPs. ``state_groups`` and ``goal_groups``
    declare which obs groups (in flat-buffer order) make up each slice; the
    encoder uses these to split, encode each group, and concatenate.

    Sharing the encoder with :class:`SquashedGaussianActor` ties the
    contrastive Q's view of state to the policy's view, halving wasted
    parameter count in the SA encoder's first layer and aligning the
    representations the policy gradient flows through.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        num_layers_per_block: int = 4,
        expand: int = 1,
        activation: str = "swish",
        norm: bool = True,
        repr_dim: int = 64,
        action_dim: int | None = None,
        encoders: _SharedStateEncoder | None = None,
        encoder_cfg: dict[str, dict] | None = None,
        state_groups: list[str] | None = None,
        goal_groups: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        active_groups = sorted(obs_groups[obs_set])
        self.obs_groups = active_groups
        # ``state_groups`` / ``goal_groups`` come from ``construct_algorithm``
        # which knows which group is the HER goal slice. Default fallback (for
        # tests / direct construction): treat ``target_state`` as the goal.
        if state_groups is None:
            state_groups = [g for g in active_groups if g != "target_state"]
        if goal_groups is None:
            goal_groups = [g for g in active_groups if g not in state_groups]
        self.state_groups = state_groups
        self.goal_groups = goal_groups

        # Build (or accept) the shared encoder. Group specs cover the full
        # active set so the encoder can serve actor (all groups), critic SA
        # (state groups), and critic G (goal groups) from a single ModuleDict.
        group_specs = [(g, int(prod(obs[g].shape[1:]))) for g in active_groups]
        if encoders is None:
            encoders = _SharedStateEncoder(group_specs, encoder_cfg)
        self.encoders = encoders

        encoded_state_dim = encoders.total_encoded_dim(state_groups)
        encoded_goal_dim = encoders.total_encoded_dim(goal_groups)
        _action_dim = action_dim if action_dim is not None else output_dim

        num_blocks = depth // num_layers_per_block
        self.sa_encoder = ResidualMLP(
            input_dim=encoded_state_dim + _action_dim,
            output_dim=repr_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expand=expand,
            num_layers_per_block=num_layers_per_block,
            activation=activation,
            norm=norm,
        )
        self.g_encoder = ResidualMLP(
            input_dim=encoded_goal_dim,
            output_dim=repr_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expand=expand,
            num_layers_per_block=num_layers_per_block,
            activation=activation,
            norm=norm,
        )

    def sa_forward(self, raw_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """SA branch: encode raw state through shared encoder, append action, run SA MLP."""
        encoded_state = self.encoders.encode_flat(raw_state, self.state_groups)
        return self.sa_encoder(torch.cat([encoded_state, action], dim=-1))

    def g_forward(self, raw_goal: torch.Tensor) -> torch.Tensor:
        """G branch: encode raw goal through shared encoder, run G MLP."""
        encoded_goal = self.encoders.encode_flat(raw_goal, self.goal_groups)
        return self.g_encoder(encoded_goal)

    def encode(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sa_forward(state, action), self.g_forward(goal)

    def forward(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        sa_repr, g_repr = self.encode(state, action, goal)
        return -torch.linalg.norm(sa_repr - g_repr, dim=-1)

    @property
    def output_std(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def as_jit(self):
        return self

    def as_onnx(self, *args, **kwargs):
        return self

    def reset(self, dones=None):
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass


# ---------------------------------------------------------------------------
# CRL algorithm
# ---------------------------------------------------------------------------


class CRL:
    """Contrastive Reinforcement Learning with optional HER.

    Implements the same public interface as :class:`~rsl_rl.algorithms.PPO` so
    :class:`~rsl_rl.runners.OnPolicyRunner` can drive it without modification.
    """

    actor: SquashedGaussianActor
    critic: BilinearCritic

    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: BilinearCritic,
        buffer: ReplayBuffer,
        *,
        obs_dim: int,
        goal_dim: int,
        goal_start_idx: int,
        goal_end_idx: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        logsumexp_penalty_coeff: float = 0.1,
        entropy_param: float = 0.5,
        replay_ratio: float = 0.1,
        num_sgd_steps: int = 800,
        min_replay_size: int = 1000,
        use_cuda_graph: bool = True,
        her: HindsightRelabeling | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.buffer = buffer
        self.her = her

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.goal_start_idx = goal_start_idx
        self.goal_end_idx = goal_end_idx
        self.action_dim = action_dim
        self.logsumexp_penalty_coeff = logsumexp_penalty_coeff
        self.replay_ratio = replay_ratio
        self.num_sgd_steps = num_sgd_steps
        self._min_replay_size = min_replay_size

        # Derive batch_size: total_samples = capacity * replay_ratio * num_envs.
        total_samples = int(buffer.capacity * replay_ratio * buffer.num_envs)
        self.batch_size = max(1, total_samples // num_sgd_steps)

        # Column offsets in the flat data_dim layout:
        #   [obs (obs_dim) | goal (goal_dim) | action (act_dim) | reward | done | seed | trunc]
        obs_goal_dim = obs_dim + goal_dim
        self._act_start = obs_goal_dim
        self._act_end = obs_goal_dim + action_dim
        self._reward_col = self._act_end
        self._done_col = self._act_end + 1
        self._seed_col = self._act_end + 2
        self._trunc_col = self._act_end + 3

        self.target_entropy = -entropy_param * action_dim
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        self._use_cuda_graph = use_cuda_graph and device != "cpu" and torch.cuda.is_available()
        capturable = self._use_cuda_graph

        # When the actor and critic share an encoder ModuleDict, register the
        # encoder parameters with the critic optimizer ONLY. The actor's loss
        # still back-propagates through the encoder (so gradients accumulate on
        # encoder params), but only the critic step uses them — i.e. the
        # contrastive loss is the encoder's training signal, the policy
        # gradient is not. Standard CRL practice; mirrors how SAC + shared
        # encoder is typically wired.
        actor_params = list(actor.parameters())
        critic_params = list(critic.parameters())
        if actor.encoders is critic.encoders:
            shared_ids = {id(p) for p in actor.encoders.parameters()}
            actor_params = [p for p in actor_params if id(p) not in shared_ids]
            # critic_params already includes the shared encoder params via
            # ``critic.parameters()`` traversal, so they get the contrastive
            # update. No extra step needed.

        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr, capturable=capturable, foreach=True)
        self.critic_optimizer = optim.Adam(critic_params, lr=critic_lr, capturable=capturable, foreach=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr, capturable=capturable)

        # Per-env episode seed counter for HER episode-boundary tracking.
        self._episode_seed: torch.Tensor | None = None

        # Multi-GPU (placeholder for parity with PPO).
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg else 1

        # Dummy attributes the runner may access.
        self.has_kinematic = False
        self.intrinsic_rewards = None
        self.kinematic_rewards = None

        goal_slice_dim = goal_end_idx - goal_start_idx
        B = self.batch_size

        # CUDA Graph: full-loop with index-based gathering from buffer.
        # Index tensors [N, B] store flat buffer addresses for each step.
        # O(N*B*8 bytes) = ~3 MB for N=400, B=512 — negligible vs old ~8 GB.
        N = num_sgd_steps
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._graph_captured = False
        self._all_idx_flat = torch.zeros(N, B, dtype=torch.long, device=device)
        self._all_idx_ft_flat = torch.zeros(N, B, dtype=torch.long, device=device)
        self._graph_critic_loss = torch.zeros(1, device=device)
        self._graph_actor_loss = torch.zeros(1, device=device)
        self._graph_alpha_loss = torch.zeros(1, device=device)
        self._graph_sum_critic = torch.zeros(1, device=device)
        self._graph_sum_actor = torch.zeros(1, device=device)
        self._graph_sum_alpha = torch.zeros(1, device=device)

        # Mixed precision: enabled on CUDA with bfloat16 support.
        self._use_amp = self._use_cuda_graph  # same gate as CUDA graphs (GPU only)

        # torch.compile on encoder modules for kernel fusion.
        if device != "cpu":
            self.critic.sa_encoder = torch.compile(self.critic.sa_encoder)
            self.critic.g_encoder = torch.compile(self.critic.g_encoder)
            self.actor.mlp = torch.compile(self.actor.mlp)

        self._compiled_critic_loss = self._critic_loss_fn
        self._compiled_actor_loss = self._actor_loss_fn

    # ------------------------------------------------------------------
    # PPO-compatible public API
    # ------------------------------------------------------------------

    @property
    def learning_rate(self) -> float:
        return self.actor_optimizer.param_groups[0]["lr"]

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def get_policy(self) -> SquashedGaussianActor:
        """Return the actor for inference (handles TensorDict via ``forward``)."""
        return self.actor

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions; store the *raw* flat obs for later ``process_env_step``.

        The replay buffer holds raw observations (not encoded) because the
        encoder is trainable — encoded values would go stale across SGD steps.
        Encoders re-run on every forward, both at inference time here and
        during the SGD loop on the buffer slice.
        """
        raw_flat = self.actor.flatten_raw(obs)
        self._last_obs = raw_flat
        with torch.no_grad():
            actions, _ = self.actor.sample(raw_flat)
        self._last_actions = actions
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Write one transition directly into the replay buffer."""
        if self._episode_seed is None:
            self._episode_seed = torch.zeros(dones.shape[0], device=self.device, dtype=torch.float32)

        done_float = dones.float().view(-1)
        truncated = extras.get("time_outs", torch.zeros_like(done_float))

        row = torch.cat(
            [
                self._last_obs,
                self._last_actions,
                rewards.view(-1, 1),
                done_float.unsqueeze(-1),
                self._episode_seed.unsqueeze(-1),
                truncated.unsqueeze(-1),
            ],
            dim=-1,
        )
        self.buffer.insert(row.unsqueeze(0))

        self._episode_seed = self._episode_seed + done_float

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op for off-policy algorithms."""

    def update(self) -> dict[str, float]:
        """Index-based update: sample (t,e,future_t) triples, gather, SGD."""
        if self.buffer.size < self._min_replay_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        self._update_iter = getattr(self, "_update_iter", 0) + 1

        # ---- Phase 1: refresh episode boundaries (one scan, ~1ms) ----
        if self.her is not None:
            self.her.update_episode_boundaries(self.buffer, self._seed_col)

        batch_size = self.batch_size

        # ---- Phase 2: batch sample + gather into pre-allocated buffers ----
        total = self.num_sgd_steps * batch_size
        if self.her is not None:
            t, e, ft = self.her.sample_indices(self.buffer, total)
        else:
            t = torch.randint(0, self.buffer.size - 1, (total,), device=self.device).int()
            e = torch.randint(0, self.buffer.num_envs, (total,), device=self.device).int()
            ft = t + 1

        # Write flat indices into pre-allocated [N, B] tensors (one copy each).
        num_envs = self.buffer.num_envs
        self._all_idx_flat.copy_((t.long() * num_envs + e.long()).reshape(self.num_sgd_steps, batch_size))
        self._all_idx_ft_flat.copy_((ft.long() * num_envs + e.long()).reshape(self.num_sgd_steps, batch_size))

        # ---- Phase 3: SGD loop ----
        if self._use_cuda_graph and not self._graph_captured:
            try:
                self._capture_full_loop_graph()
            except Exception as ex:
                print(f"  [CRL] CUDA Graph capture failed: {ex}. Using eager fallback.", flush=True)
                self._graph_captured = False
                self._cuda_graph = None

        if self._graph_captured:
            self._cuda_graph.replay()
            sum_critic = self._graph_sum_critic
            sum_actor = self._graph_sum_actor
            sum_alpha = self._graph_sum_alpha
        else:
            sum_critic = torch.zeros(1, device=self.device)
            sum_actor = torch.zeros(1, device=self.device)
            sum_alpha = torch.zeros(1, device=self.device)
            flat_buffer = self.buffer.data.reshape(-1, self.buffer.data_dim)
            for i in range(self.num_sgd_steps):
                idx = self._all_idx_flat[i]
                ft_idx = self._all_idx_ft_flat[i]
                mb_state = flat_buffer[idx, : self.obs_dim]
                mb_act = flat_buffer[idx, self._act_start : self._act_end]
                mb_goal = flat_buffer[ft_idx, self.goal_start_idx : self.goal_end_idx]

                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self._use_amp):
                    critic_loss, g_repr = self._compiled_critic_loss(mb_state, mb_act, mb_goal)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self._use_amp):
                    actor_loss, log_prob = self._compiled_actor_loss(mb_state, mb_goal, g_repr)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                sum_critic += critic_loss.detach()
                sum_actor += actor_loss.detach()
                sum_alpha += alpha_loss.detach()

        # ---- Phase 3: metrics ----
        n = float(self.num_sgd_steps)

        with torch.no_grad(), torch.random.fork_rng(devices=[self.device] if self.device != "cpu" else []):
            diag_n = min(256, batch_size)
            flat_buf = self.buffer.data.reshape(-1, self.buffer.data_dim)
            last_idx = self._all_idx_flat[-1, :diag_n]
            last_ft = self._all_idx_ft_flat[-1, :diag_n]
            ds = flat_buf[last_idx, : self.obs_dim]
            da = flat_buf[last_idx, self._act_start : self._act_end]
            dg = flat_buf[last_ft, self.goal_start_idx : self.goal_end_idx]
            dfg = dg

            sa_r, g_r = self.critic.encode(ds, da, dg)
            logits = -torch.sqrt(((sa_r[:, None, :] - g_r[None, :, :]) ** 2).sum(-1))
            infonce_acc = (logits.argmax(dim=1) == torch.arange(diag_n, device=self.device)).float().mean().item()
            infonce_gap = (logits.diag() - logits.mean(dim=1)).mean().item()

            actor_obs = torch.cat([ds, dfg], dim=-1)
            actor_act, lp = self.actor.sample(actor_obs)
            sa_pi, g_pi = self.critic.encode(ds, actor_act, dfg)
            qf_pi = -torch.sqrt(((sa_pi - g_pi) ** 2).sum(-1))

            act_mean_abs = actor_act.abs().mean().item()
            act_std = actor_act.std().item()
            achieved = ds[:, self.goal_start_idx : self.goal_end_idx]
            her_goal_dist = torch.linalg.norm(achieved - dg, dim=-1)
            policy_entropy = -lp.mean().item()

        result = {
            "Loss/actor": (sum_actor / n).item(),
            "Loss/critic": (sum_critic / n).item(),
            "Loss/alpha": (sum_alpha / n).item(),
            "Loss/log_alpha": self.log_alpha.item(),
            "Policy/entropy": policy_entropy,
            "Policy/action_abs_mean": act_mean_abs,
            "Policy/action_std": act_std,
            "Critic/qf_pi_mean": qf_pi.mean().item(),
            "Critic/qf_pi_std": qf_pi.std().item(),
            "Critic/infonce_acc": infonce_acc,
            "Critic/infonce_gap": infonce_gap,
            "Buffer/size": float(self.buffer.size),
            "Buffer/batch_size": float(batch_size),
            "Buffer/sgd_steps": float(self.num_sgd_steps),
        }
        if self.her is not None:
            result["HER/goal_dist_mean"] = her_goal_dist.mean().item()
            result["HER/goal_dist_max"] = her_goal_dist.max().item()
        return result

    def save(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data.clone(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> int:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}

        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer"])
            self.log_alpha.data.copy_(loaded_dict["log_alpha"])
        return load_cfg.get("iteration", False)

    def broadcast_parameters(self) -> None:
        pass

    def reduce_parameters(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_full_loop_graph(self) -> None:
        """Capture the entire SGD loop as one CUDA graph.

        Each step gathers from the buffer via pre-allocated index tensors.
        Memory is O(1) per step — the CUDA graph memory pool reuses
        intermediate tensors across iterations.
        """
        self._graph_flat_buffer = self.buffer.data.reshape(-1, self.buffer.data_dim)

        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._full_loop_body()
        torch.cuda.current_stream().wait_stream(s)

        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph, capture_error_mode="thread_local"):
            self._full_loop_body()
        self._graph_captured = True
        print(f"  [CRL] Full-loop CUDA Graph captured ({self.num_sgd_steps} steps, index-based)", flush=True)

    def _full_loop_body(self) -> None:
        """All SGD steps: gather from buffer → fwd → bwd → optimizer, each step."""
        self._graph_sum_critic.zero_()
        self._graph_sum_actor.zero_()
        self._graph_sum_alpha.zero_()
        flat_buf = self._graph_flat_buffer

        for i in range(self.num_sgd_steps):
            idx = self._all_idx_flat[i]
            ft_idx = self._all_idx_ft_flat[i]
            mb_state = flat_buf[idx, : self.obs_dim]
            mb_act = flat_buf[idx, self._act_start : self._act_end]
            mb_goal = flat_buf[ft_idx, self.goal_start_idx : self.goal_end_idx]

            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self._use_amp):
                sa_repr, g_repr = self.critic.encode(mb_state, mb_act, mb_goal)
                logits = -torch.cdist(sa_repr, g_repr, p=2)
                critic_loss = -(logits.diag() - torch.logsumexp(logits, dim=1)).mean()
                logsumexp_reg = torch.logsumexp(logits + 1e-6, dim=1)
                critic_loss = critic_loss + self.logsumexp_penalty_coeff * (logsumexp_reg**2).mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self._use_amp):
                actor_obs = torch.cat([mb_state, mb_goal], dim=-1)
                actor_actions, log_prob = self.actor.sample(actor_obs)
                # ``sa_forward`` runs the shared encoder on raw state, then the
                # SA encoder. When the encoder is shared with the actor, the
                # gradient flows back through it but the actor optimizer
                # excludes encoder params (see ``CRL.__init__``).
                sa_pi = self.critic.sa_forward(mb_state, actor_actions)
                qf_pi = -torch.sqrt(((sa_pi - g_repr.detach()) ** 2).sum(-1))
                alpha = self.log_alpha.exp().detach()
                actor_loss = (alpha * log_prob - qf_pi).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._graph_sum_critic += critic_loss.detach()
            self._graph_sum_actor += actor_loss.detach()
            self._graph_sum_alpha += alpha_loss.detach()

    def _critic_loss_fn(
        self, mb_state: torch.Tensor, mb_act: torch.Tensor, mb_goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (critic_loss, cached_g_repr) for reuse in actor loss."""
        sa_repr, g_repr = self.critic.encode(mb_state, mb_act, mb_goal)
        logits = -torch.cdist(sa_repr, g_repr, p=2)
        loss = -(logits.diag() - torch.logsumexp(logits, dim=1)).mean()
        logsumexp_reg = torch.logsumexp(logits + 1e-6, dim=1)
        loss = loss + self.logsumexp_penalty_coeff * (logsumexp_reg**2).mean()
        return loss, g_repr

    def _actor_loss_fn(
        self,
        mb_state: torch.Tensor,
        mb_goal: torch.Tensor,
        cached_g_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uses cached g_repr to skip redundant g_encoder forward pass."""
        actor_obs = torch.cat([mb_state, mb_goal], dim=-1)
        actor_actions, log_prob = self.actor.sample(actor_obs)
        sa_pi = self.critic.sa_forward(mb_state, actor_actions)
        qf_pi = -torch.sqrt(((sa_pi - cached_g_repr.detach()) ** 2).sum(-1))
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - qf_pi).mean()
        return actor_loss, log_prob

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> CRL:
        """Build CRL from config dict. Called by ``OnPolicyRunner.__init__``."""
        alg_cfg = dict(cfg["algorithm"])
        alg_cfg.pop("class_name", None)

        obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], default_sets=["actor", "critic"])

        # Resolve HER config (fills in obs_dim, goal_start_idx, goal_end_idx).
        alg_cfg = resolve_her_config(alg_cfg, obs, obs_groups)

        # Extract HER config from alg_cfg.
        her_cfg = alg_cfg.pop("her_cfg")
        active_keys = sorted(set(k for group_list in obs_groups.values() for k in group_list))
        # Per-group raw flat dim = product of all non-batch dims, so multi-dim
        # obs (e.g. height_scan ``(B, 1, 76, 126)``) get the right total size.
        group_specs = [(g, int(prod(obs[g].shape[1:]))) for g in active_keys]
        total_obs = sum(d for _, d in group_specs)
        obs_dim = her_cfg["obs_dim"]
        goal_start_idx = her_cfg["goal_start_idx"]
        goal_end_idx = her_cfg["goal_end_idx"]
        goal_dim = total_obs - obs_dim

        # Determine which active groups make up the state portion vs the goal
        # portion, in flat-buffer (sorted) order. Mirrors ``resolve_her_config``:
        # all non-goal groups are state; the ``target_state`` group is goal.
        goal_group_name = her_cfg.get("target_state", "target_state")
        state_groups = [g for g in active_keys if g != goal_group_name]
        goal_groups = [g for g in active_keys if g == goal_group_name]

        # Build actor and critic.
        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])
        actor_cfg.pop("class_name", None)
        critic_cfg.pop("class_name", None)
        # ``encoder_cfg`` lives on either the actor cfg or critic cfg in the
        # runner config; pull from whichever has it (or both — they should match).
        encoder_cfg = actor_cfg.pop("encoder_cfg", None) or critic_cfg.pop("encoder_cfg", None)
        critic_cfg.pop("encoder_cfg", None)  # ensure both popped if both set
        actor_cfg.pop("encoder_cfg", None)
        critic_cfg["action_dim"] = env.num_actions
        critic_cfg["state_groups"] = state_groups
        critic_cfg["goal_groups"] = goal_groups

        # Build the shared encoder once and pass it to both actor and critic
        # when ``share_encoders=True``. The actor sees all groups (state +
        # goal); the critic SA branch sees state groups; the critic G branch
        # sees goal groups. All three use the same ModuleDict — so encoder
        # parameters are tied across actor and both critic branches.
        share_encoders = alg_cfg.pop("share_encoders", True)
        shared_encoder = _SharedStateEncoder(group_specs, encoder_cfg)
        actor_encoder = shared_encoder
        critic_encoder = shared_encoder if share_encoders else _SharedStateEncoder(group_specs, encoder_cfg)

        actor: SquashedGaussianActor = SquashedGaussianActor(
            obs, obs_groups, "actor", env.num_actions, encoders=actor_encoder, **actor_cfg
        ).to(device)
        critic: BilinearCritic = BilinearCritic(
            obs, obs_groups, "critic", env.num_actions, encoders=critic_encoder, **critic_cfg
        ).to(device)
        print(f"Actor Model: {actor}")
        print(f"Critic Model: {critic}")

        # Build buffer (pop max_replay_size since __init__ doesn't take it).
        data_dim = total_obs + env.num_actions + 4
        buffer = ReplayBuffer(
            capacity=alg_cfg.pop("max_replay_size"),
            num_envs=env.num_envs,
            data_dim=data_dim,
            device=device,
        )

        # Build HER.
        her_obj = HindsightRelabeling(
            gamma=her_cfg["gamma"],
            goal_start_idx=her_cfg["goal_start_idx"],
            goal_end_idx=her_cfg["goal_end_idx"],
            obs_dim=her_cfg["obs_dim"],
        )

        return CRL(
            actor=actor,
            critic=critic,
            buffer=buffer,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            goal_start_idx=goal_start_idx,
            goal_end_idx=goal_end_idx,
            action_dim=env.num_actions,
            her=her_obj,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu_cfg"),
            **alg_cfg,
        )
