.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Parallel simulation at scale needs many environments stepping side by side —
hundreds, sometimes tens of thousands per GPU — and authoring each of those envs
by hand would be hopelessly slow. Cloning is Isaac Lab's answer: you author a
small representative scene under ``/World/envs/env_n`` and the cloner expands it
across the rest of the env population for you, optionally with per-env variation.

The expansion itself is performed by USD and the active physics backend's native
replicator, wrapped by Isaac Lab's core :mod:`isaaclab.cloner` module behind a
single uniform surface.

.. contents:: On this page
   :local:
   :depth: 2


The Backend Layer
-----------------

At the bottom of the stack, each backend exposes a raw function that takes a flat
description of the world layout. These functions are useful for standalone tools
and tests and deliberately have parallel signatures:

.. code-block:: text

    backend_replicate(stage, sources, destinations, env_ids, selection, positions=None, quaternions=None, ...)

The parallel arrays describe the layout:

* ``sources`` — source prim paths already authored on the stage.
* ``destinations`` — destination templates containing ``"{}"``, formatted with each env id.
* ``env_ids`` — NumPy integer array of target env indices.
* ``selection`` — NumPy boolean array of shape ``[len(sources), num_envs]``;
  ``selection[i, j]`` is ``True`` when env ``j`` should be populated from source ``i``.
  The raw USD function names this argument ``mask``; physics functions name it ``mapping``.
* ``positions`` / ``quaternions`` — optional per-env world transforms.

Production scene construction stores those arrays once in a
:class:`~isaaclab.cloner.ClonePlan`. Simulation-owned backend contexts consume the
same value through ``context.replicate(plan)``; no backend rebuilds the mapping
from a second queue of array arguments.


Standalone Examples
~~~~~~~~~~~~~~~~~~~

Direct calls into the backend functions, for tooling or tests that need full
control. Production code reaches for one of the ways in
`Cloning in a Backend-Agnostic Way`_ instead.

**USD** — clone a visual cube across envs:

.. code-block:: python

    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.cloner import usd_replicate

    num_envs = 128
    stage = sim_utils.get_current_stage()
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/envs/env_0/Cube", cube_cfg)

    usd_replicate(
        stage,
        sources=("/World/envs/env_0/Cube",),
        destinations=("/World/envs/env_{}/Cube",),
        env_ids=np.arange(num_envs),
        mask=np.ones((1, num_envs), dtype=np.bool_),
    )

**PhysX** — call PhysX and USD on the same sources and destinations (either order):

.. code-block:: python

    from isaaclab_physx.cloner import physx_replicate

    sources = ("/World/envs/env_0/Cube",)
    destinations = ("/World/envs/env_{}/Cube",)
    env_ids = np.arange(num_envs)
    mapping = np.ones((1, num_envs), dtype=np.bool_)
    physx_replicate(stage, sources, destinations, env_ids, mapping=mapping)
    usd_replicate(stage, sources, destinations, env_ids, mask=mapping)

**Newton**:

.. code-block:: python

    from isaaclab_newton.cloner import newton_physics_replicate

    newton_physics_replicate(stage, sources, destinations, env_ids, mapping=mapping)

**OvPhysX**:

.. code-block:: python

    from isaaclab_ov.cloner import ovphysx_replicate

    ovphysx_replicate(stage, sources, destinations, env_ids, mapping=mapping)

Application code does not call those backend entry points independently. Doing
so would create multiple clone lifecycles and lets physics and rendering drift
onto different layouts. It publishes one :class:`~isaaclab.cloner.ClonePlan` and
dispatches that same plan once to the backends registered by the selected physics,
renderer, and visualizer cfgs.


Cloning in a Backend-Agnostic Way
---------------------------------

Authoring every prim in every env by hand would be prohibitively slow and would
also tie scene code to whichever physics engine happens to be active. Isaac Lab
sidesteps both problems with a single central abstraction:
:class:`~isaaclab.cloner.ClonePlan` — a compact description of how a small set of
prim-level prototypes maps onto the full population of envs, with each prototype
free to land in some envs and not others. A plan is built once, fed to each backend, and
lets every engine take its own fastest replication path: USD instancing for
visuals, PhysX's native replicator for rigid bodies and articulations, Newton's
world system for its parallel pipeline. The same plan drives all of them, so user
code never branches on the backend.

ClonePlan
~~~~~~~~~

A plan holds the parallel arrays used by production clone contexts — sources,
destinations, mask, env ids — in one place. Conceptually it is a small table
where each row describes one distinct prototype-to-destination mapping; the
fields listed below are that table's columns:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``sources``
     - Source prim paths, one per replication row.
   * - ``destinations``
     - Destination templates with ``"{}"`` for the env id, one per row.
   * - ``clone_mask``
     - NumPy boolean array ``[len(sources), num_envs]``; ``True`` when env ``j`` comes from row ``i``.
   * - ``env_ids``
     - Optional NumPy integer array of target env ids; execution requires it.
   * - ``positions``
     - Optional per-env world positions [m], shape ``[num_envs, 3]``.
   * - ``global_paths``
     - Unique prim paths for scene assets shared by every env and therefore not replicated.
   * - ``semantic_tags``
     - Cfg-declared semantic tags, one tuple per row; labels embedded only in USD are not inferred.
   * - ``*_prototypes`` and plan-owned geometry data
     - Completed frame, rigid-body, articulation, deformable, cable, and requested-geometry topology.

The plan does not own a stage. Simulation-owned contexts supply their own runtime
when they consume it.

A cartpole in every env is one row that reaches every env:

.. code-block:: text

    sources      = ("/World/envs/env_0/Cartpole",)
    destinations = ("/World/envs/env_{}/Cartpole",)
    clone_mask   = [[True, True, ..., True]]

Every asset gets its own row whether or not the envs differ, so the plan reads the same way in
every scene. Replication still copies whole envs where it can: when the rows together cover every
env from every prototype, dispatch hands the backends one copy of ``env_0`` instead of one copy
per asset. That is an internal shortcut — the plan itself, and everything that queries it, keeps
the per-asset rows.

When envs differ — say a cartpole in every env plus a 2-variant obstacle (box into
envs 0/1, sphere into envs 2/3):

.. code-block:: text

    sources      = ("/World/envs/env_0/Cartpole",
                    "/World/envs/env_0/Obstacle_0",     # box prototype
                    "/World/envs/env_0/Obstacle_1")     # sphere prototype
    destinations = ("/World/envs/env_{}/Cartpole",
                    "/World/envs/env_{}/Obstacle",
                    "/World/envs/env_{}/Obstacle")
    clone_mask   = [[1, 1, 1, 1],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1]]

Querying a plan
~~~~~~~~~~~~~~~

Anything that has to follow an asset between the two sides of that table — a sensor
resolving its ``prim_path`` back to the prototype it should read, a ray caster
loading one mesh per variant — asks :mod:`isaaclab.cloner.query` rather than
manipulating path strings itself:

.. code-block:: python

    from isaaclab import cloner

    # which prototype is env 2's obstacle cloned from?
    cloner.query.path_to_source(plan, "/World/envs/env_2/Obstacle")
    # -> ("/World/envs/env_0/Obstacle_1", "/World/envs/env_[^/]+/Obstacle", "")

    # the same question asked of a wildcard expression, naming the env it stands for
    cloner.query.path_to_source(plan, "/World/envs/env_[^/]+/Obstacle", env_id=0)
    # -> ("/World/envs/env_0/Obstacle_0", "/World/envs/env_[^/]+/Obstacle", "")

Two obstacle variants share one destination template, so the template alone does not
identify a prototype — the environment does. A concrete path carries it in the clone
slot; a wildcard does not, and resolves to one representative variant
unless you pass ``env_id``. Use :func:`~isaaclab.cloner.query.iter_sources` when you
need every variant behind a template. Note that environment ids are not mask columns:
column ``j`` stands for ``env_ids[j]``, and the queries speak ids throughout.

A plan is the *what*. Putting one together and handing it to the backends is
the *how*. Isaac Lab exposes two application-facing ways to do that:

* :class:`~isaaclab.scene.InteractiveScene` owns the lifecycle when assets and
  sensors are declared on its scene config. This is the normal workflow for both direct and
  manager-based environments, whether the scene is homogeneous or heterogeneous.
* :func:`~isaaclab.cloner.clone_plan_from_env_0` plus
  :func:`~isaaclab.cloner.replicate` is a cfg-first shortcut when every env is one copy
  of env_0. Reach for it in standalone tools and tests where depending on
  :class:`~isaaclab.scene.InteractiveScene` is unsuitable.

``InteractiveScene``
~~~~~~~~~~~~~~~~~~~~

Declare assets in an :class:`~isaaclab.scene.InteractiveSceneCfg` and construct
the scene normally. :class:`~isaaclab.scene.InteractiveScene` brackets asset
construction with :class:`~isaaclab.cloner.ReplicateSession` internally, so task
code does not expose clone-lifecycle ceremony. :class:`~isaaclab.envs.DirectRLEnv`
and :class:`~isaaclab.envs.DirectMARLEnv` construct their configured scene automatically.

.. code-block:: python

    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        )

    scene = InteractiveScene(MySceneCfg(num_envs=128, env_spacing=2.0))

When envs need to differ across the population, use
:class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` or
:class:`~isaaclab.sim.spawners.wrappers.MultiUsdFileCfg`; see
:doc:`multi_asset_spawning`.

:class:`~isaaclab.cloner.ReplicateSession` remains available to standalone tools
that need to author a heterogeneous scene without :class:`~isaaclab.scene.InteractiveScene`.

``clone_plan_from_env_0`` + ``replicate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shortcut for a standalone tool or test where every env is one copy of env_0 and
:class:`~isaaclab.scene.InteractiveScene` is not a suitable dependency. Pass a
:class:`~isaaclab.cloner.CloneCfg` and a flat tuple of asset and sensor cfgs;
:func:`~isaaclab.cloner.clone_plan_from_env_0` publishes the plan and assigns
their prototype spawn paths before construction:

.. code-block:: python

    from isaaclab import cloner

    asset_cfgs = (robot_cfg, ground_cfg, light_cfg)
    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), asset_cfgs, num_envs=128, env_spacing=2.0)
    robot, _, _ = [cfg.class_type(cfg) for cfg in asset_cfgs]
    cloner.replicate(plan)

Every env receives the same prototype. The tuple is deliberately flat: the cloner
does not inspect a task or scene cfg tree, and ``None`` is allowed for an optional
declared participant. Use :class:`~isaaclab.scene.InteractiveSceneCfg` in task
environments and whenever envs need to differ across the population.


Under the Hood
--------------

To see how the backend-agnostic surface works, follow one asset through the
system. :class:`~isaaclab.scene.InteractiveScene` builds and publishes one plan
before any asset constructor runs; the homogeneous direct helper does the same.
Constructors spawn only the prototype paths that plan assigns them. Replication
completes the plan's authored topology and hands that exact plan once to every
clone-capable backend in the simulation registry.

The story has to look like this because the engines underneath disagree about
*when* and *how* replication actually happens:

* **PhysX** registers its native replicator before scene construction and
  defers the copy work to physics runtime.
* **USD** is declarative and immediate — its registered
  :class:`~isaaclab.cloner.UsdReplicateContext` materializes the clones in place.
* **Newton** builds every asset row into one model builder, then finalizes one
  model after all rows have arrived.

Every engine supplies a context such as
:class:`~isaaclab.cloner.UsdReplicateContext`, ``PhysxReplicateContext`` or
``NewtonReplicateContext``. Physics managers, renderers and visualizers call
:meth:`~isaaclab.sim.SimulationContext.get_or_create_backend` as soon as they
know their cfg-derived backend class. That class is the complete registry key, so every consumer
of one backend type shares one simulation-owned context. The replication session dispatches
directly from that registry; asset and sensor cfgs contain no backend-selection or cloning
callbacks.

Registry dispatch
~~~~~~~~~~~~~~~~~

The Kit visualizer and Isaac RTX renderer each register the same
:class:`~isaaclab.cloner.UsdReplicateContext`; each physics manager registers its native context,
and a consumer with an independent scene registers the context that builds it. The replication
session sends the complete plan to each unique context in priority order. With
:attr:`~isaaclab.cloner.CloneCfg.replicate_physics` disabled, a context used
only by physics is skipped; a context shared with a renderer still runs.
In that USD-only mode, the session registers
:class:`~isaaclab.cloner.UsdReplicateContext` itself. A USD-backed renderer or visualizer resolves
the same type and therefore reuses that context automatically.

A renderer that draws into a scene of its own, such as OVRTX, registers its context the same way.
Its one simulation-owned clone context exports once and applies every row to the renderer scene.
Its camera has to be declared with the scene — an :class:`~isaaclab.scene.InteractiveSceneCfg`
field, or a camera built inside the session — because a camera added to an already replicated
scene is absent from the plan and therefore wrong.

Deferring the work like this buys three things at once:

* Replication can wait until the plan is fully built, so the final layout is
  known before any prims are spawned.
* Every asset's request is batched into a single backend call instead of one
  call per asset.
* Asset code stays free of backend selection and clone lifecycle state.

:attr:`~isaaclab.scene.InteractiveSceneCfg.replicate_physics` is piped into
:attr:`~isaaclab.cloner.CloneCfg.replicate_physics` and applied at dispatch.

Backend contexts
~~~~~~~~~~~~~~~~

Each backend ships a small adapter class — its *replicate context* — that
knows how to take a registered cfg and replicate it on the backend's specific
runtime:

.. code-block:: text

    UsdReplicateContext      # replicates USD prim subtrees
    PhysxReplicateContext    # replicates PhysX rigid bodies and articulations
    NewtonReplicateContext   # replicates Newton bodies in its parallel pipeline

One plan usually reaches more than one context — a Kit or Isaac RTX consumer pairs USD with the
active physics context so physics and visuals both follow, while kitless Newton skips USD authoring.
Swapping physics or renderers changes the registered contexts; asset
cfgs and scene code stay unchanged.

Running replication
~~~~~~~~~~~~~~~~~~~

:func:`~isaaclab.cloner.replicate` completes and publishes the plan, then runs the
registry's clone-capable backends in priority order. The private dispatch shape is roughly:

.. code-block:: python

    def replicate(plan):
        plan = declare_scene_layout(plan)
        publish(plan)
        for context in sorted(registered_clone_contexts, key=lambda item: item.replicate_priority):
            context.replicate(plan)

The plan is published to :class:`~isaaclab.sim.SimulationContext` so physics,
renderers, visualizers, and scene-data providers read the same layout. No fallback
context is constructed during dispatch.

Collision Filtering
-------------------

PhysX models per-env isolation through collision groups, so PhysX scenes need a
filtering pass after cloning to keep envs from colliding with each other while
still letting them collide with global prims (terrain, ground planes, lights).

:class:`~isaaclab.scene.InteractiveScene` runs that pass automatically when
``filter_collisions=True`` and the backend is PhysX. For standalone PhysX pipelines,
call :func:`~isaaclab.cloner.filter_collisions` after replication:

.. code-block:: python

    from isaaclab.cloner import filter_collisions

    filter_collisions(
        stage=stage,
        physicsscene_path="/physicsScene",
        collision_root_path="/World/collisions",
        prim_paths=[f"/World/envs/env_{i}" for i in range(num_envs)],
        global_paths=["/World/ground"],
    )

Newton isolates envs through its world system and does not need this pass.
