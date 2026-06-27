isaaclab\_newton.physics
========================

.. automodule:: isaaclab_newton.physics

  .. rubric:: Classes

  .. autosummary::

    NewtonManager
    NewtonCfg
    NewtonDebugCaptureCfg
    NewtonDebugReplayCfg
    NewtonSoftContactCfg
    MJWarpDebugOperationProvider
    ReplayAdapter
    ReplayCapability
    ReplayRequest
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

Physics Debugging
-----------------

.. autoclass:: MJWarpDebugOperationProvider
  :members:
  :show-inheritance:

.. autoclass:: ReplayAdapter
  :members:

.. autoclass:: ReplayCapability
  :members:

.. autoclass:: ReplayRequest
  :members:

.. autodata:: REPLAY_STAGES

.. autofunction:: register_replay_adapter

.. autofunction:: get_replay_adapter

.. autofunction:: get_replay_adapter_ids

Physics Configuration
---------------------

.. autoclass:: NewtonCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonSoftContactCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonDebugCaptureCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: NewtonDebugReplayCfg
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
  :exclude-members: DebugOperationProvider, DebugTriggerContext, DebugTriggerResult

.. autoclass:: NewtonVBDManager
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: NewtonXPBDManager
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: DebugOperationProvider, DebugTriggerContext, DebugTriggerResult

.. autoclass:: NewtonFeatherstoneManager
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: DebugOperationProvider, DebugTriggerContext, DebugTriggerResult

.. autoclass:: NewtonKaminoManager
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: DebugOperationProvider, DebugTriggerContext, DebugTriggerResult

.. autoclass:: NewtonMPMManager
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: DebugOperationProvider, DebugTriggerContext, DebugTriggerResult
