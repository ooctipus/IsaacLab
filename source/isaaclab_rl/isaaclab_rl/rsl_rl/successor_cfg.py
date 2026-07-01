# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass


@configclass
class RslRlSuccessorCfg:
    """Configuration for the forward-backward successor-representation critic.

    Replaces the scalar critic with a goal-conditioned successor-representation value ``V = <F(s, z), z>``
    (``w = z``, the goal embedding -- no learned read-out): the ``F``/``B`` maps (on the
    :class:`~...SuccessorFeatureCriticModel` critic) are learned **reward-free** from the forward-backward
    occupancy objective, and goal-reaching is intrinsic (the policy reward is the goal-alignment
    ``<B(s'), z>``). See :class:`~rsl_rl.extensions.SuccessorFeatures`. ``feature_dim`` must match the critic.
    """

    feature_dim: int = 128
    """Width ``d`` of ``F`` / ``B`` (must equal the critic model's ``feature_dim``)."""

    occupancy: str = "bilinear_fb"
    """Which reward-free occupancy objective trains ``F``/``B`` (the rest of the architecture -- the model,
    ``z``, the value ``V=<F(s,z),z>``, and the actor -- is shared, so this only switches
    :meth:`~rsl_rl.extensions.SuccessorFeatures.representation_loss`):

    * ``"bilinear_fb"`` -- the Touati-Ollivier / Meta-Motivo forward-backward measure: the ``[n, n]`` batch
      matrix ``M[i,j]=<F(s_i,z_i), B(s'_j)>`` with off-diagonal squared TD + a diagonal identity pull. Learns
      the full pairwise measure; bounds ``F`` only through ``<F, B>`` (fragile -- needs ``B`` to span the
      space; see the design doc).
    * ``"vector_td"`` -- the Barreto/Borsa successor-feature vector TD: regress the whole vector
      ``F(s,z) -> B(s) + gamma*(1-term)*F_bar(s',z)`` (``B(s)`` detached -> trains only ``F``; ``B`` owned by
      the orthonormality penalty). The gamma-contraction self-bounds ``F`` without a clamp; ``B`` is a thinner,
      decorrelated basis than the measure-shaped ``B`` of FB. This is the single-task special case of FB.
    """

    ortho_coef: float = 100.0
    """Weight of the backward orthonormality penalty ``E[B B^T] -> I`` (decorrelates states). Set to Meta-Motivo's
    HUMANOID value ``100`` (its DMC value is ``1.0``) -- this is a humanoid-scale locomotion task, and at ``1.0``
    the ``B`` Gram stayed correlated and collapsed under the FB loss in the value-off gate."""

    train_goal_ratio: float = 0.2
    """Fraction of the FB ``z`` drawn from goal embeddings ``project_z(B(s')[perm])``; the remaining
    ``1 - ratio`` are uniform on the ``sqrt(d)`` sphere. The random-sphere majority is what bounds ``F`` across
    the latent space (Meta-Motivo ``train_goal_ratio``: ``0.5`` for FB/DMC, ``0.2`` for FB-CPR humanoid)."""

    goal_command_name: str = "goal_point"
    """Deprecated command-manager lookup used when explicit goal bindings are absent."""

    fb_batch_size: int = 1024
    """Cap on the forward-backward batch-matrix size (Meta-Motivo trains FB at 1024). The full ``[n, n]`` measure
    matrix is what bounds ``F``; if the PPO minibatch exceeds this, the FB loss subsamples to this many states,
    otherwise it uses the full minibatch."""

    target_tau: float = 0.0001
    """Polyak rate for the forward-backward target network, applied per minibatch gradient step. Meta-Motivo uses
    ``0.01`` PER FRESH BATCH, but on-policy PPO reuses each rollout for ``num_learning_epochs * num_mini_batches``
    (~20) gradient steps, so a per-fresh-batch-equivalent rate needs a ~100x smaller per-minibatch ``tau``. The
    lagging target is what restrains the free forward map ``F`` (once the off-diagonal TD converges, only the
    target's lag keeps pulling the measure down); at ``0.01`` the target fully tracks within a rollout, the
    restraint vanishes, and ``F``/``M_diag`` run away. ``0.0001`` gave a bounded, decorrelated representation
    (``m_diag`` ~47, ``f_norm`` ~10 ~= the analytic ``1/(1-gamma)/sqrt(d)``) in the value-off gate."""

    goal_observation_bind: str | None = None
    """Expression resolving to the immutable per-task goal-observation TensorDict."""

    goal_indices_bind: str | None = None
    """Expression resolving to the stable per-environment goal-row tensor."""
