Changed
^^^^^^^

* **Breaking:** Changed the Newton cloner to approximate no collision geometry of its own.
  Colliders now follow the asset: shapes authoring ``physics:approximation`` are remeshed as
  before, and every other collider keeps the geometry it was authored with, since USD defaults
  the token to ``none``. Previously mesh colliders authoring no approximation were silently
  replaced with convex hulls. To restore a hull, author ``physics:approximation`` on the asset
  instead of relying on the cloner: add :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg`
  with ``mesh_approximation_name="convexHull"`` to the spawner's ``collision_props`` fragment
  list, pass it through
  :attr:`~isaaclab.sim.converters.MeshConverterCfg.mesh_collision_props` at conversion time, or
  author the token directly in the USD. Note that ``collision_props`` takes the fragment path
  only when every entry is a :class:`~isaaclab.sim.schemas.SchemaFragment`; mixing in the legacy
  ``CollisionPropertiesCfg`` falls back to the legacy writer and ignores the fragment.

Removed
^^^^^^^

* **Breaking:** Removed ``NewtonCfg.simplify_meshes`` and the ``simplify_meshes`` argument of
  :func:`~isaaclab_newton.cloner.newton_physics_replicate`,
  :class:`~isaaclab_newton.cloner.NewtonReplicateContext`, and ``build_source_builders``.
  Collision approximation is an asset-authoring concern.

* **Breaking:** Removed the clone-source shape-sequence check that discarded every authored
  ``physics:approximation`` mode in the scene when two sources resolved to different shape
  types. Distinct assets differ by design, so the check fired in any multi-source scene and
  silently replaced authored collision geometry. ``SolverMuJoCo``'s homogeneous-worlds
  requirement is now met by authoring the same approximation on every per-world variant of a
  slot, as :class:`~isaaclab_tasks.core.dexsuite.dexsuite_env_cfg.ObjectCfg` does.

Fixed
^^^^^

* Fixed the Newton cloner replacing ``NewtonSDFCollisionAPI`` colliders with convex hulls, which
  raised ``ValueError: method 'convex_hull' replaces the mesh with non-mesh geometry`` and
  prevented scenes with SDF colliders from building.
