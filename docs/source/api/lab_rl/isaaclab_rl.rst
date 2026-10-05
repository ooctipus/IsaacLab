.. _api-isaaclab-rl:

isaaclab_rl
===========

.. automodule:: isaaclab_rl

Unified Entrypoints
-------------------

.. automodule:: isaaclab_rl.entrypoints

.. autoclass:: isaaclab_rl.entrypoints.TrainingRequest
   :members:
   :show-inheritance:

.. autoclass:: isaaclab_rl.entrypoints.PlaybackRequest
   :members:
   :show-inheritance:

.. autoclass:: isaaclab_rl.entrypoints.SimpleAgentRequest
   :members:
   :show-inheritance:

.. autofunction:: isaaclab_rl.entrypoints.train

.. autofunction:: isaaclab_rl.entrypoints.play

.. autofunction:: isaaclab_rl.entrypoints.zero_agent

.. autofunction:: isaaclab_rl.entrypoints.random_agent

.. autofunction:: isaaclab_rl.entrypoints.run_train_cli

.. autofunction:: isaaclab_rl.entrypoints.run_play_cli

.. autofunction:: isaaclab_rl.entrypoints.run_zero_agent_cli

.. autofunction:: isaaclab_rl.entrypoints.run_random_agent_cli

RL Utilities
------------

.. automodule:: isaaclab_rl.utils.wandb
   :members:
   :show-inheritance:

RL-Games Wrapper
----------------

.. automodule:: isaaclab_rl.rl_games
   :members:
   :show-inheritance:

RSL-RL Wrapper
--------------

The wrapper also accepts a :class:`gymnasium.Env` implementing
:class:`isaaclab_rl.rsl_rl.RslRlEnv`, including Gym wrappers around that environment.
Observation groups and transitions are batched Torch tensors. The environment resets
finished episodes within ``step`` and publishes the resulting groups in ``obs_buf``;
``cfg.is_finite_horizon`` controls timeout bootstrapping. No scene or asset ownership
is required by this boundary. The playback command additionally reads ``step_dt``
in seconds to pace inference.

.. automodule:: isaaclab_rl.rsl_rl
   :members:
   :imported-members:
   :show-inheritance:

SKRL Wrapper
------------

.. automodule:: isaaclab_rl.skrl
   :members:
   :show-inheritance:

Stable-Baselines3 Wrapper
-------------------------

.. automodule:: isaaclab_rl.sb3
   :members:
   :show-inheritance:

TorchRL Wrapper
---------------

.. automodule:: isaaclab_rl.torchrl
   :members:
   :imported-members:
   :show-inheritance:
