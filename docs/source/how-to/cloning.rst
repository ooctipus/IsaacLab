.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Isaac Lab describes a parallel scene once in a :class:`~isaaclab.cloner.ClonePlan`. Each registered
clone backend receives the plan-derived rows routed to it; no backend derives a second layout or
copies state from another backend instance.

A declarative :class:`~isaaclab.scene.InteractiveScene` owns this lifecycle. It builds and publishes
the plan before constructing scene entities, supplies each entity constructor with its exact
prototype path, dispatches the required clone contexts when construction finishes, and queues model
construction for the first hard reset after any intervening stage edits. A homogeneous direct
environment instead uses a task-owned two-phase sequence. Its plain
:class:`~isaaclab.scene.InteractiveScene` is only the runtime registry; ``_setup_scene()`` calls
:func:`~isaaclab.cloner.clone_plan_from_env_0` before cfg-owned constructors and
:func:`~isaaclab.cloner.replicate` afterward.

.. contents:: On this page
   :local:
   :depth: 2


ClonePlan
---------

A plan is a flat table with one row per independently copied asset variant:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``sources``
     - Exact authored prototype paths, one per row.
   * - ``destinations``
     - Destination templates with ``"{}"`` for the environment id. A shared world asset uses its
       exact path without a clone slot.
   * - ``clone_mask``
     - Boolean tensor ``[num_rows, num_envs]`` selecting the environments populated by each row.
   * - ``env_ids``
     - Environment ids represented by the mask columns.
   * - ``positions``
     - Environment origins [m], shape ``[num_envs, 3]``.
   * - ``replicate_physics``
     - Whether physics clone contexts may reuse plan prototypes across environments.
   * - ``filter_collisions``
     - Whether PhysX collision groups are authored after stage replication.
   * - ``cfg_rows``
     - Configuration identities mapped to their covering plan rows.
   * - ``context_rows``
     - Clone-context types mapped to the rows they consume.
   * - ``collision_paths``
     - Declared collision roots shared across environments.
   * - ``env_template``
     - Path template used for environment roots and namespace expansion.

For a homogeneous robot and one shared ground plane, the plan remains asset-wise:

.. code-block:: text

   sources      = ("/World/envs/env_0/Robot", "/World/Ground")
   destinations = ("/World/envs/env_{}/Robot", "/World/Ground")
   clone_mask   = [[1, 1, 1, 1],
                   [0, 0, 0, 0]]

A backend that can copy a whole homogeneous environment may request that representation. Its mapping
query collapses the asset rows only when doing so describes exactly the same layout; heterogeneous or
partially populated plans stay asset-wise.

A cfg nested below another authored asset maps to its nearest parent's rows. The parent copy already
contains that subtree, so the child does not create redundant replication work.


The single lifecycle
--------------------

Construct a simulation first, then construct a declarative scene with the standard
``cfg.class_type(cfg)`` convention. The scene owns its single clone lifecycle:

.. code-block:: python

   scene_cfg = MySceneCfg(num_envs=128, env_spacing=2.0)
   scene = scene_cfg.class_type(scene_cfg)

The lifecycle owns the order:

#. Build and publish the plan.
#. Author every environment root and its origin.
#. Construct every asset and sensor from cfg. Constructors author only the exact prototype paths
   assigned by the plan; input cfgs are not mutated.
#. Dispatch stage and native clone contexts; each reads its routed rows from the published plan.
#. Run collision filtering when ``CloneCfg`` enables it and PhysX requires it.
#. Apply any required stage edits before initialization.
#. On the first hard reset, construct model-role backends from the same plan before physics
   finalization. A soft reset cannot skip this phase.

Creating an asset outside this lifecycle is an ownership error: the plan would not contain it.
Likewise, a second lifecycle cannot replace the plan on one
:class:`~isaaclab.sim.SimulationContext`.

For a homogeneous workflow that authors one prototype directly, build and publish its plan before
the cfg-owned constructors, then dispatch that same plan afterward:

.. code-block:: python

   from isaaclab import cloner

   cfg = (cloner.CloneCfg(), robot_cfg)
   plan = cloner.clone_plan_from_env_0(cfg, num_envs=128, env_spacing=2.0)
   robot = robot_cfg.class_type(robot_cfg)
   cloner.replicate(plan)

:func:`~isaaclab.cloner.clone_plan_from_env_0` rejects multi-variant or partially populated layouts.
Put those entities on an :class:`~isaaclab.scene.InteractiveSceneCfg`, whose scene-owned lifecycle
keeps the asset-wise plan.


Declaring a scene
-----------------

Manager-based and heterogeneous environments put assets on an
:class:`~isaaclab.scene.InteractiveSceneCfg`:

.. code-block:: python

   @configclass
   class MySceneCfg(InteractiveSceneCfg):
       robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       ground = AssetBaseCfg(
           prim_path="/World/Ground",
           spawn=sim_utils.GroundPlaneCfg(),
       )
       light = AssetBaseCfg(
           prim_path="/World/Light",
           spawn=sim_utils.DomeLightCfg(intensity=2000.0),
       )


For a homogeneous direct environment, keep asset cfgs on the direct env cfg and construct its one
prototype in ``_setup_scene()``. The setup method owns the explicit plan, construction, and dispatch:

.. code-block:: python

   class CartpoleEnvCfg(DirectRLEnvCfg):
       robot_cfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       scene = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

   class CartpoleEnv(DirectRLEnv):

       def _setup_scene(self):
           plan = cloner.clone_plan_from_env_0(self.cfg, self.cfg.scene.num_envs, self.cfg.scene.env_spacing)
           self.robot = self.cfg.robot_cfg.class_type(self.cfg.robot_cfg)
           self.scene.articulations["robot"] = self.robot
           cloner.replicate(plan)

Do not clone by walking the completed stage or open another lifecycle after prototype dispatch.


Heterogeneous scenes
--------------------

Use :class:`~isaaclab.sim.MultiAssetSpawnerCfg` or :class:`~isaaclab.sim.MultiUsdFileCfg` to
declare variants. :attr:`isaaclab.cloner.CloneCfg.clone_combinations` declares which named scene
assets may coexist and their relative weights. The plan selects exact prototype paths for active
variants; inactive variants retain a row but are not spawned.

See :doc:`multi_asset_spawning` for configuration examples.


Querying the plan
-----------------

Consumers walk the prototype/clone relation through :mod:`isaaclab.cloner.query`, not by slicing
path strings or rediscovering the stage:

.. code-block:: python

   plan = sim.get_clone_plan()

   cloner.query.path_to_source(plan, "/World/envs/env_2/Obstacle")
   list(cloner.query.iter_sources(plan, "/World/envs/env_[^/]+/Obstacle"))

Use :func:`~isaaclab.cloner.query.iter_sources` when a destination template has several variants
and the consumer needs every populated source. Query functions speak environment ids throughout;
mask column ``j`` represents ``env_ids[j]``.


Backend ownership
-----------------

Each engine registers its clone context on :class:`~isaaclab.sim.SimulationContext`. The lifecycle
dispatches stage/native contexts once and queues model contexts for the first hard reset.

The core contexts are:

.. code-block:: text

   UsdReplicateContext      copies USD prim subtrees
   PhysxReplicateContext    registers PhysX native replication
   NewtonReplicateContext   builds the Newton model asset by asset
   OvReplicateContext       registers OvPhysX clone transforms

USD copying runs before native physics consumes its destinations. Isaac Sim PhysX registers one USD
scene context alongside its native PhysX context because native replication requires plan-authored
destination topology on the stage. Both consume the same plan, and renderer or visualizer requests
reuse that registered USD context, so the stage topology is copied once. Newton headless does not
request USD unless a renderer or visualizer requires the complete stage, or native physics
replication is disabled. In the latter mode Newton reads exact materialized per-environment paths
after stage edits rather than rebuilding scene ownership by walking the stage.

Collision filtering
-------------------

PhysX scenes need collision filtering after clone dispatch so environments do not collide with
one another. :func:`~isaaclab.cloner.replicate` derives collision roots from the plan and applies
the pass when :attr:`~isaaclab.cloner.CloneCfg.filter_collisions` enables it. Newton isolates
environments through its world model and does not need this pass.
