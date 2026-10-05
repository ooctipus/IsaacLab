isaaclab\_newton.physics
========================

.. automodule:: isaaclab_newton.physics

  .. rubric:: Classes

  .. autosummary::

    NewtonManager
    NewtonCfg
    NewtonBackendCfg
    NewtonBuilderCfg
    NewtonSoftContactCfg
    NewtonCollisionPipelineCfg
    NewtonFeatherstoneManager
    NewtonKaminoManager
    NewtonMPMManager
    NewtonMJWarpManager
    NewtonVBDManager
    NewtonShapeCfg
    NewtonSolverCfg
    NewtonXPBDManager
    MJWarpSolverCfg
    VBDSolverCfg
    XPBDSolverCfg
    FeatherstoneSolverCfg
    KaminoCollisionDetectorCfg
    KaminoConstraintsCfg
    KaminoDVICfg
    KaminoDVISolverCfg
    KaminoDynamicsCfg
    KaminoFKCfg
    KaminoMaterialsCfg
    KaminoPADMMCfg
    KaminoPADMMSolverCfg
    MPMSolverCfg
    HydroelasticSDFCfg

.. currentmodule:: isaaclab_newton.physics

Physics Manager
---------------

.. autoclass:: NewtonManager
  :members:
  :inherited-members:
  :show-inheritance:

Physics Configuration
---------------------

.. autoclass:: NewtonCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonBackendCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonBuilderCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autofunction:: create_newton_builder

.. autoclass:: NewtonSoftContactCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: MJWarpSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: VBDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: XPBDSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: FeatherstoneSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoPADMMCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoDVICfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoDynamicsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoConstraintsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoFKCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoCollisionDetectorCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoMaterialsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoPADMMSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: KaminoDVISolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: MPMSolverCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonCollisionPipelineCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: HydroelasticSDFCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonShapeCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

Solver Managers
---------------

.. autoclass:: NewtonMJWarpManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: NewtonVBDManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: NewtonXPBDManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: NewtonFeatherstoneManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: NewtonKaminoManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: NewtonMPMManager
  :members:
  :inherited-members:
  :show-inheritance:

Experimental Homogeneous Populations
-----------------------------------

These explicitly composed resources support headless native-contact MuJoCo Warp
populations. Their lifecycle belongs to ``SimulationContext``; task assignments
and policy buffers belong to the calling environment.

.. autoclass:: isaaclab_newton.physics.population.NewtonPopulationCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: isaaclab_newton.physics.population.NewtonPopulationManager
  :members:
  :show-inheritance:

.. autoclass:: isaaclab_newton.physics.population.NewtonPopulationBackendCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: isaaclab_newton.physics.population.NewtonPopulationBackend
  :members:

.. autoclass:: isaaclab_newton.physics.population.NewtonPopulation
  :members:

Experimental Native Worlds
--------------------------

``NewtonWorldsBackend`` keeps native physics arrays and logical identities in one
``newton.solvers.MuJoCoWorlds`` runtime. Prepare immutable native prototypes once,
register the backend with ``SimulationContext``, record task callbacks, and install
it before resetting the simulation. GPU commands create, retype or destroy worlds;
``forward()`` applies coherent pending commands and refreshes current poses without
advancing physics. ``step()`` uses the same executable with physics enabled.

The caller owns complete reset payloads and publishes each new command sequence
only when its payload is ready. World capacities reserve virtual rows; live counts
and physical readiness remain separate. The initial backend is headless and exposes
native arrays through ``backend.runtime``; it provides no replicated Newton state.

.. autoclass:: NewtonWorldsCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonWorldsManager
  :members:
  :show-inheritance:

.. autoclass:: NewtonWorldsBackendCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonWorldsBackend
  :members:
