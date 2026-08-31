.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Isaac Lab describes a parallel scene once and gives every registered clone backend the same
:class:`~isaaclab.cloner.ClonePlan`. No clone backend derives a second layout or copies state from
another backend instance.

The public composition root is :class:`~isaaclab.cloner.ReplicateSession`. It builds and publishes
the plan before constructing scene entities, supplies each entity constructor with its exact
prototype path, and dispatches the plan once when construction finishes.

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

For a homogeneous robot and one shared ground plane, the plan remains asset-wise:

.. code-block:: text

   sources      = ("/World/envs/env_0/Robot", "/World/Ground")
   destinations = ("/World/envs/env_{}/Robot", "/World/Ground")
   clone_mask   = [[1, 1, 1, 1],
                   [0, 0, 0, 0]]

A backend that can copy a whole homogeneous environment may request that representation. The
session collapses the asset rows only when doing so describes exactly the same layout; heterogeneous
or partially populated plans stay asset-wise.

A cfg nested below another authored asset maps to its nearest parent's rows. The parent copy already
contains that subtree, so the child does not create redundant replication work.


The single lifecycle
--------------------

Construct a simulation first, then enter one session rooted in the configuration that owns the
scene. Construct cfg-owned objects with the standard ``cfg.class_type(cfg)`` convention:

.. code-block:: python

   from isaaclab import cloner

   scene_cfg = MySceneCfg(num_envs=128, env_spacing=2.0)
   with cloner.ReplicateSession(
       [scene_cfg],
       num_clones=scene_cfg.num_envs,
       env_spacing=scene_cfg.env_spacing,
       device=sim.device,
       env_template=scene_cfg.clone_cfg.clone_template,
       replicate_physics=scene_cfg.replicate_physics,
   ):
       scene = scene_cfg.class_type(scene_cfg)

   if scene_cfg.filter_collisions and "physx" in scene.physics_backend:
       scene.filter_collisions()

The session owns the order:

#. Build and publish the immutable plan.
#. Author every environment root and its origin.
#. Construct every asset and sensor from cfg. Constructors author only the exact prototype paths
   assigned by the plan; input cfgs are not mutated.
#. Dispatch the same plan once to each registered clone backend.
#. Run collision filtering after replication when PhysX requires it.

Creating an asset outside this lifecycle is an ownership error: the plan would not contain it.
Likewise, a second session cannot replace the plan on one
:class:`~isaaclab.sim.SimulationContext`.


Declaring a scene
-----------------

Manager-based and direct environments already provide the composition root. Put assets in the
scene cfg rather than constructing an env-0 scene manually:

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


For a direct environment, ``_setup_scene()`` binds the objects that the cfg-owned scene already
constructed:

.. code-block:: python

   def _setup_scene(self):
       self.robot = self.scene.articulations["robot"]

Do not clone by walking the completed stage or by building a second plan in ``_setup_scene()``.


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

   cloner.query.path_to_clone(plan, "/World/envs/env_0/Obstacle", env_id=2)
   cloner.query.path_env_ids(plan, "/World/envs/env_0/Obstacle")
   cloner.query.path_to_source(plan, "/World/envs/env_2/Obstacle")

Use :func:`~isaaclab.cloner.query.iter_sources` when a destination template has several variants
and the consumer needs every populated source. Query functions speak environment ids throughout;
mask column ``j`` represents ``env_ids[j]``.


Backend ownership
-----------------

Each engine registers its clone context on :class:`~isaaclab.sim.SimulationContext`. The session
dispatches each registered context once.

The core contexts are:

.. code-block:: text

   UsdReplicateContext      copies USD prim subtrees
   PhysxReplicateContext    registers PhysX native replication
   NewtonReplicateContext   builds the Newton model asset by asset
   OvReplicateContext       registers OvPhysX clone transforms

USD copying runs before native physics consumes its destinations. Under Kit, the session obtains
one shared USD context; a PhysX manager requesting the same type reuses it. Kitless Newton and
OvPhysX paths avoid that redundant USD copy.

Collision filtering
-------------------

PhysX scenes need collision filtering after clone dispatch so environments do not collide with
one another. :class:`~isaaclab.scene.InteractiveScene` derives shared collision roots from global
plan rows and applies the pass when requested by its environment root. Newton isolates environments
through its world model and does not need this pass.
