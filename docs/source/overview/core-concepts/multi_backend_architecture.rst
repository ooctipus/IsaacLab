Multi-Backend Architecture
==========================

.. seealso::

   This page is the source of truth for the ``isaaclab-selecting-backends`` and
   ``isaaclab-using-presets`` agent skills
   (`skills/user/select-backends/ <../../../../skills/user/select-backends/SKILL.md>`__,
   `skills/user/use-presets/ <../../../../skills/user/use-presets/SKILL.md>`__).
   When you change this page, update those skills so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

Isaac Lab 3.0 introduced a multi-backend architecture that enables running simulations with
different physics backends (PhysX, Newton, and OvPhysX) while maintaining a unified API.
This page explains how the backend system works and how to extend it.

Overview
--------

Instead of hard-coding a single physics engine, Isaac Lab makes configuration the composition
boundary. Every configurable component exposes ``class_type``; its owner invokes it with the same
convention:

.. code-block:: python

    component = cfg.class_type(cfg)

This applies across simulation components, though not every backend implements every component yet:

.. list-table::
   :header-rows: 1

   * - Component
     - Core API (``isaaclab``)
     - PhysX (``isaaclab_physx``)
     - Newton (``isaaclab_newton``)
     - OvPhysX (``isaaclab_ov``)
   * - Physics Manager
     - :class:`~isaaclab.physics.PhysicsManager`
     - :class:`~isaaclab_physx.physics.PhysxManager`
     - :class:`~isaaclab_newton.physics.NewtonManager`
     - :class:`~isaaclab_ov.physics.OvPhysxManager`
   * - Articulation
     - :class:`~isaaclab.assets.Articulation`
     - :class:`~isaaclab_physx.assets.Articulation`
     - :class:`~isaaclab_newton.assets.Articulation`
     - :class:`~isaaclab_ov.assets.Articulation`
   * - Rigid Object
     - :class:`~isaaclab.assets.RigidObject`
     - :class:`~isaaclab_physx.assets.RigidObject`
     - :class:`~isaaclab_newton.assets.RigidObject`
     - :class:`~isaaclab_ov.assets.RigidObject`
   * - Deformable Object
     - :class:`~isaaclab.assets.DeformableObject`
     - :class:`~isaaclab_physx.assets.DeformableObject`
     - :class:`~isaaclab_newton.assets.DeformableObject`
     - :class:`~isaaclab_ov.assets.DeformableObject`
   * - Cable Object
     - :class:`~isaaclab.assets.CableObject`
     - Not supported
     - :class:`~isaaclab_newton.assets.CableObject`
     - Not supported
   * - Contact Sensor
     - :class:`~isaaclab.sensors.ContactSensor`
     - :class:`~isaaclab_physx.sensors.ContactSensor`
     - :class:`~isaaclab_newton.sensors.ContactSensor`
     - :class:`~isaaclab_ov.sensors.ContactSensor`
   * - Renderer
     - :class:`~isaaclab.renderers.BaseRenderer`
     - :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`
     - :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`
     - :class:`~isaaclab_ov.renderers.OVRTXRenderer`
   * - Scene Data Backend
     - :class:`~isaaclab.scene_data.SceneDataBackend`
     - ``PhysxSceneDataBackend`` (in :mod:`isaaclab_physx.physics`)
     - ``NewtonSceneDataBackend`` (in :mod:`isaaclab_newton.physics`)
     - ``OvPhysxSceneDataBackend`` (in :mod:`isaaclab_ov.physics`)
   * - Cloner
     - :class:`~isaaclab.cloner.UsdReplicateContext`
     - ``PhysxReplicateContext``
     - ``NewtonReplicateContext``
     - ``OvReplicateContext`` (shared by OVPhysX and OVRTX)

Configuration-Driven Construction
---------------------------------

The resolved config owns the implementation choice for physics managers, renderers, and visualizers.
:class:`~isaaclab.scene.InteractiveScene` invokes the same config entry point for declared assets
and sensors:

.. code-block:: python

    physics_manager = physics_cfg.class_type(physics_cfg)
    asset = asset_cfg.class_type(asset_cfg)
    sensor = sensor_cfg.class_type(sensor_cfg)
    renderer = renderer_cfg.class_type(renderer_cfg)
    visualizer = visualizer_cfg.class_type(visualizer_cfg)

``class_type`` may be a lazy ``"{DIR}.module:Class"`` string in source; ``configclass`` resolves it
before construction. Physics, renderer, and visualizer configs name their concrete implementations.
Most core asset and sensor configs still name thin ``FactoryBase`` dispatch classes; removing that
extra runtime discovery is remaining migration debt. Configs contain data, not custom ``build()`` or
``build_visualizer()`` methods. Backend packages never import one another.

Backend Selection
-----------------

The physics backend is selected via the ``physics`` field in
:class:`~isaaclab.sim.SimulationCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    # Use PhysX
    sim_cfg = SimulationCfg(physics=PhysxCfg())

    # Use Newton with MuJoCo-Warp solver
    sim_cfg = SimulationCfg(physics=MJWarpSolverCfg(
        num_substeps=4,
    ))

    # Use OvPhysX
    sim_cfg = SimulationCfg(physics=OvPhysxCfg())

The :class:`~isaaclab.sim.SimulationContext` resolves this config and constructs exactly one manager
instance from it. :class:`~isaaclab.scene.InteractiveScene` likewise invokes each declared asset and
sensor config; most core asset and sensor configs currently reach their concrete implementation
through the dispatch layer described below.

Multi-Backend Environments with Presets
---------------------------------------

Environments can support multiple backends simultaneously using :doc:`backend and preset
selectors </source/concepts/backends_and_presets>`. Each backend gets its own configuration
variant. The example below shows only the physics-related fields:

.. code-block:: python

    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.physics import PhysxAutoCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.utils import PresetCfg

    @configclass
    class CartpolePhysicsCfg(PresetCfg):
        isaacsim_physx: PhysxCfg = PhysxCfg()
        ovphysx: OvPhysxCfg = OvPhysxCfg()
        physx: PhysxAutoCfg = PhysxAutoCfg(
            isaacsim_physx=isaacsim_physx,
            ovphysx=ovphysx,
        )
        default: PhysxCfg = isaacsim_physx
        newton_mjwarp: MJWarpSolverCfg = MJWarpSolverCfg(njmax=5, nconmax=3)

    @configclass
    class CartpoleEnvCfg(DirectRLEnvCfg):
        sim: SimulationCfg = SimulationCfg(physics=CartpolePhysicsCfg())

Users then select a physics backend at the command line:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          # Default (concrete Isaac Sim PhysX)
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct

          # Automatic PhysX-family selection
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=physx

          # MJWarp (Newton backend)
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp

          # OvPhysX backend
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=ovphysx

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          # Default (concrete Isaac Sim PhysX)
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct

          # Automatic PhysX-family selection
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=physx

          # MJWarp (Newton backend)
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp

          # OvPhysX backend
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=ovphysx

When a task's default would otherwise be automatic ``PhysxAutoCfg`` selection,
its ``default`` variant is the concrete ``isaacsim_physx`` configuration.
Explicit defaults such as Newton remain unchanged. The ``physics=physx``
selector is opt-in and chooses between Isaac Sim PhysX and OvPhysX at launch
time according to whether the resolved runtime requires Kit. This mirrors
renderer presets: the default is concrete ``isaacsim_rtx``, while
``renderer=rtx`` opts into automatic selection.

The Physics Manager
-------------------

Each backend implements :class:`~isaaclab.physics.PhysicsManager`, the abstract base class
that drives the simulation loop:

.. code-block:: python

    class PhysicsManager(ABC):
        def __init__(self, cfg) -> None: ...

        @abstractmethod
        def _bind_context(self, sim_context: SimulationContext) -> None: ...

        @abstractmethod
        def reset(self, soft: bool = False) -> None: ...

        @abstractmethod
        def forward(self) -> None: ...

        @abstractmethod
        def step(self) -> None: ...

        def close(self) -> None: ...  # concrete; dispatches STOP

The lifecycle is construction, private pre-clone ``_bind_context()``, one shared clone session,
post-clone hard ``reset()``, repeated stepping, and ``close()``. During binding, each consumer derives
its backend class from the resolved config and calls
:meth:`~isaaclab.sim.SimulationContext.get_or_create_backend`. Physics, renderers, and visualizers
that resolve the same backend class share one native resource rather than copying state between
resources. The class is the complete registry key. The simulation owns that shared resource's
lifetime; consumers close only their own state.

Asset and Sensor Interfaces
---------------------------

Asset and sensor configs carry ``class_type`` and
:class:`~isaaclab.scene.InteractiveScene` constructs each declared entity as
``asset_cfg.class_type(asset_cfg)``. Most core configs currently target a thin ``FactoryBase`` class,
which reads ``SimulationContext.physics_backend`` and loads the same-named implementation from
``isaaclab_<backend>`` (or ``isaaclab_ov`` for OvPhysX). This dispatch is migration debt and is not
used for physics managers, renderers, or visualizers.

Core base classes define the portable API contract; the PhysX, Newton, and OvPhysX packages
independently implement it where supported. Current implementations use ``wp.array`` (Warp arrays)
as their primary data type.

These base interfaces define the portable contract. Advanced code can also use
each engine's native low-level data API, but those APIs deliberately retain
different ownership and synchronization semantics. See
:doc:`physical-backends/direct-api-access/index` for PhysX typed views, Newton
live model/state arrays and generic selections, and OvPhysX tensor bindings.

Adding a New Physics Backend
----------------------------

To add a new physics backend (e.g., ``mybackend``), create a new extension package following
the established conventions:

**1. Package structure:**

.. code-block:: text

    source/isaaclab_mybackend/
    └── isaaclab_mybackend/
        ├── __init__.py
        ├── physics/
        │   ├── __init__.py           # lazy_export()
        │   ├── __init__.pyi          # public exports
        │   ├── mybackend_manager.py
        │   └── mybackend_manager_cfg.py
        ├── assets/
        │   └── ...
        ├── sensors/
        │   └── ...
        ├── renderers/
        │   └── ...
        └── cloner/
            └── ...

**2. Implement the physics manager:**

The manager exposes a :class:`~isaaclab.scene_data.SceneDataBackend` that publishes a pointer,
format, and dirty latch. A matching request passes the pointer through; another format is converted
once per dirty generation by :class:`~isaaclab.scene_data.SceneDataProvider`.

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager.py
    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataPublication


    class MyBackendSceneDataBackend(SceneDataBackend):
        def __init__(self, backend):
            self._backend = backend
            self._publication = SceneDataPublication(SceneDataFormat.Transform(), True)

        @property
        def transform_publication(self) -> SceneDataPublication:
            return self._publication

        @property
        def point_publications(self) -> dict[str, SceneDataPublication]:
            return {}

        def publish(self) -> None:
            self._publication.data.transforms = self._backend.transforms  # native pointer
            self._publication.dirty = True


    class MyBackendManager(PhysicsManager):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._backend = None
            self._state = None
            self._scene_data_backend = None

        def _bind_context(self, sim_context):
            super()._bind_context(sim_context)
            self._backend = sim_context.get_or_create_backend(
                MyBackendRuntime,
                self.cfg.runtime_variant,
                sim_context.stage,
                clone_role="physics",
            )
            self._scene_data_backend = MyBackendSceneDataBackend(self._backend)

        def get_scene_data_backend(self) -> SceneDataBackend:
            return self._scene_data_backend

        def reset(self, soft=False):
            if not soft:
                self._state = self._backend.initialize_after_clone()
                self.dispatch_event(PhysicsEvent.PHYSICS_READY)

        def step(self):
            self._backend.step(self._state)
            self._scene_data_backend.publish()

        def forward(self):
            self._backend.forward(self._state)
            self._scene_data_backend.publish()

        def close(self):
            super().close()
            self._state = None
            self._backend = None

**3. Create the physics config:**

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager_cfg.py
    from isaaclab.physics import PhysicsCfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class MyBackendCfg(PhysicsCfg):
        class_type = "{DIR}.mybackend_manager:MyBackendManager"
        runtime_variant: str = "default"

**4. Implement assets and sensors:**

Each asset or sensor extends the corresponding base class from ``isaaclab``. Until core asset and
sensor configs move off ``FactoryBase``, the implementation must use the module path and class name
expected by its core dispatch class. For example, ``isaaclab.assets.articulation.Articulation``
resolves ``isaaclab_mybackend.assets.articulation.Articulation``. Keep implementations inside this
package; do not import one from another backend package.

.. code-block:: python

    # isaaclab_mybackend/assets/articulation/articulation.py
    from isaaclab.assets.articulation import BaseArticulation

    class Articulation(BaseArticulation):
        def __init__(self, cfg):
            super().__init__(cfg)
            # Set up backend-specific simulation structures

Audit the Backend Lifecycle
---------------------------

``scripts/benchmarks/benchmark_backend_lifecycle.py`` builds a minimal direct configuration without
an interactive scene, constructs every declared asset inside one
:class:`~isaaclab.cloner.ReplicateSession`, and records construction time, initialization order,
registry sharing, clone/export counts, scene-data pointers, dirty generations, and conversions.

.. code-block:: bash

    # Inspect all 90 combinations and the 17 unsupported OV/Kit mixtures.
    uv run --extra all python scripts/benchmarks/benchmark_backend_lifecycle.py --list

    # Run all 73 supported rigid combinations in isolated worker processes.
    uv run --extra all python scripts/benchmarks/benchmark_backend_lifecycle.py \
        --output /tmp/backend_lifecycle.json

    # Exercise native and converted deformable-point publications across all visualizers.
    uv run --extra all python scripts/benchmarks/benchmark_backend_lifecycle.py \
        --points --output /tmp/backend_point_lifecycle.json

Each passing worker proves exact ``cfg.class_type(cfg)`` construction, one completed plan before
initialization, stable registry identities, shared Newton ``Model``/``State``/``Control`` objects
where applicable, and zero native-format conversions or exactly one non-native conversion per dirty
generation. The only excluded combinations mix OV components with Kit or Isaac Sim components in
one process.

Key Design Principles
---------------------

- **Declarative construction**: Every runtime component is constructed as ``cfg.class_type(cfg)``.
- **Explicit migration boundary**: Asset and sensor ``class_type`` values still use thin backend
  dispatch classes; physics, renderer, and visualizer configs do not.
- **Instance ownership**: Managers own instance state; matching backend classes share one native resource through
  the simulation registry.
- **One clone lifecycle**: Physics, renderers, and visualizers consume one clone plan and initialize
  after cloning.
- **One data boundary**: Physics publishes pointers, formats, and dirty state through the scene data
  provider; render consumers request their required format.
- **Independent packages**: Backend packages do not import one another, even when implementations
  are deliberately duplicated.
- **Independent selection**: Physics, renderer, and visualizer configs resolve independently;
  unsupported combinations fail explicitly.

See Also
--------

- :doc:`/source/migration/migrating_to_isaaclab_3-0` — migration guide from Isaac Lab 2.x to the
  multi-backend architecture
- :doc:`/source/concepts/backends_and_presets` — user guide to backend and preset selection
- :doc:`/source/features/hydra` — advanced configuration and preset authoring
- :doc:`physical-backends/index` — feature matrix and per-backend guides (PhysX, Newton, OvPhysX)
- :doc:`physical-backends/newton/index` — Newton backend guide
- :doc:`physical-backends/newton/newton-manager-abstraction` — adding Newton solver managers and
  coupled solvers
- :doc:`renderers` — renderer backend architecture
