Optimize Stage Creation
=======================

Isaac Lab can construct a stage in memory to avoid disk I/O during scene creation. This is
particularly effective for large-scale RL setups with thousands of environments.

What This Feature Does
----------------------

**Stage in Memory**

- Constructs the stage in memory, rather than with a USD file, avoiding overhead from disk I/O
- After stage creation, if rendering is required, the stage is attached to the USD context, returning to the default stage configuration
- Not enabled by default

Usage Examples
--------------

Stage in memory can be toggled by setting the :attr:`isaaclab.sim.SimulationCfg.create_stage_in_memory` flag.

**Using Stage in Memory with a RL environment**

.. code-block:: python

    # create config and set flag
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = 1024
    cfg.sim.create_stage_in_memory = True
    # create env with stage in memory
    env = ManagerBasedRLEnv(cfg=cfg)

When using stage in memory without an existing RL environment class, wrap the stage creation steps
in a ``with`` statement to set the stage context. The stage is automatically attached
to the USD context when ``SimulationContext`` is created with ``create_stage_in_memory=True``.

**Using Stage in Memory with a manual scene setup**

.. code-block:: python

    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils import use_stage

    # init simulation context with stage in memory
    # Note: stage is automatically attached to USD context
    sim = SimulationContext(cfg=SimulationCfg(create_stage_in_memory=True))

    # grab stage and set stage context
    with use_stage(sim.stage):
        # create cartpole scene
        scene_cfg = CartpoleSceneCfg(num_envs=1024, env_spacing=2.0)
        scene = scene_cfg.class_type(scene_cfg)

    sim.play()


Limitations
-----------

**Stage in Memory**

- The stage is automatically attached to the USD context at ``SimulationContext`` creation, ensuring proper
  lifecycle events for viewport and physics systems.

- Certain low-level Kit APIs do not yet support stage in memory.

  - In one particular case, for some environments, the API call to color the ground plane is skipped, when stage in memory is enabled.
