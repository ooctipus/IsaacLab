.. _how-to-env-wrappers:


Wrapping environments
=====================

.. currentmodule:: isaaclab

Environment wrappers are a way to modify the behavior of an environment without modifying the environment itself.
This can be used to apply functions to modify observations or rewards, record videos, enforce time limits, etc.
A detailed description of the API is available in the :class:`gymnasium.Wrapper` class.

At present, all RL environments inheriting from the :class:`~envs.ManagerBasedRLEnv` or :class:`~envs.DirectRLEnv` classes
are compatible with :class:`gymnasium.Wrapper`, since the base class implements the :class:`gymnasium.Env` interface.
In order to wrap an environment, you need to first initialize the base environment. After that, you can
wrap it with as many wrappers as you want by calling ``env = wrapper(env, *args, **kwargs)`` repeatedly.

For example, here is how you would wrap an environment to enforce that reset is called before step or render:

.. code-block:: python

    """Launch Isaac Sim Simulator first."""


    from isaaclab.app import AppLauncher

    # launch omniverse app in headless mode
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    """Rest everything follows."""

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    # create base environment
    cfg = load_cfg_from_registry("Isaac-Reach-Franka", "env_cfg_entry_point")
    env = gym.make("Isaac-Reach-Franka", cfg=cfg)
    # wrap environment to enforce that reset is called before step
    env = gym.wrappers.OrderEnforcing(env)


Recording videos
----------------

Configure video capture on the environment before constructing it. The visualizer owns the
viewpoint, and the recorder selects that visualizer as its source:

.. code-block:: python

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(eye=(1.0, 1.0, 1.0), lookat=(0.0, 0.0, 0.0))
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir="videos/train",
            video_interval=1500,
            video_length=200,
        )
    ]

The environment advances each recorder from ``env.step()``; no Gym wrapper or
``render_mode="rgb_array"`` is required. See :doc:`/source/how-to/record_video` for source
selection, headless requirements, and camera-sensor recording.


Wrapper for learning frameworks
-------------------------------

Every learning framework has its own API for interacting with environments. For example, the
`Stable-Baselines3`_ library uses the `gym.Env <https://gymnasium.farama.org/api/env/>`_
interface to interact with environments. However, libraries like `RL-Games`_, `RSL-RL`_ or `SKRL`_
use their own API for interfacing with a learning environments. Since there is no one-size-fits-all
solution, we do not base the :class:`~envs.ManagerBasedRLEnv` and :class:`~envs.DirectRLEnv` classes on any particular learning framework's
environment definition. Instead, we implement wrappers to make it compatible with the learning
framework's environment definition.

As an example of how to use the RL task environment with Stable-Baselines3:

.. code:: python

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    # create isaac-env instance
    env = gym.make(task_name, cfg=env_cfg)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)


.. caution::

  Wrapping the environment with the respective learning framework's wrapper should happen in the end,
  i.e. after all other wrappers have been applied. This is because the learning framework's wrapper
  modifies the interpretation of environment's APIs which may no longer be compatible with :class:`gymnasium.Env`.


Adding new wrappers
-------------------

All new wrappers should be added to the :mod:`isaaclab_rl` module.
They should check that the underlying environment is an instance of :class:`isaaclab.envs.ManagerBasedRLEnv`
or :class:`~envs.DirectRLEnv`
before applying the wrapper. This can be done by using the :func:`unwrapped` property.

We include a set of wrappers in this module that can be used as a reference to implement your own wrappers.
If you implement a new wrapper, please consider contributing it to the framework by opening a pull request.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _SKRL: https://skrl.readthedocs.io
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
