Fixed
^^^^^

* Fixed Newton ray-caster site registration for sensors mounted on an environment root. It looked
  for a clone-plan row whose destination was the environment root, which only existed while a
  homogeneous scene was collapsed into one row, and now derives the environment template from the
  namespace the plan's destinations share.
* Fixed resource-owned BVH refits and ray-cast tasks running eagerly on every CUDA step. The Newton
  clone context now captures their conditional graph and invalidates it when task or native state
  pointers change.
* Fixed headless Newton runs without cameras importing visual-only geometry. Only a renderer or
  visualizer that draws the native Newton model now requests it before cloning.
* Fixed cloned articulations applying actuator stiffness, damping, and armature after Kamino had
  frozen its dynamic-constraint topology. Every clone now receives those values before finalization.
* Fixed repeated clone-plan source rows importing the same Newton source builder, and pinned the
  Newton importer fix that prevents mesh variants from decoding the same visual texture again.
* Changed Newton physics, the Newton Warp renderer and Newton-backed visualizers to share the same
  simulation-scoped :class:`~isaaclab_newton.cloner.NewtonReplicateContext` instance. Manager
  teardown no longer clears its model, state, or control while another consumer still uses them.
* Moved the Newton deformable object implementation from ``isaaclab_contrib.deformable`` into
  :mod:`isaaclab_newton.assets`, so the Newton backend no longer imports an optional package.

Changed
^^^^^^^

* **Breaking:** Removed ``NewtonWarpRendererCfg.create_default_light``. Declare lights as scene
  assets so they are covered by the clone plan instead of creating one per camera after cloning.
* **Breaking:** Removed the ``NewtonCfg`` wrapper. Pass a concrete solver config such as
  :class:`~isaaclab_newton.physics.MJWarpSolverCfg` directly to
  :attr:`~isaaclab.sim.SimulationCfg.physics`; shared Newton fields now live on
  :class:`~isaaclab_newton.physics.NewtonSolverCfg`.
* **Breaking:** Changed Newton Warp semantic and instance segmentation to use cfg-declared
  :attr:`~isaaclab.sim.spawners.SpawnerCfg.semantic_tags` retained by the clone plan. Labels embedded
  only in a USD asset are now unlabelled; declare them on the asset's spawner cfg.
* Changed the Newton scene-data backend to publish rigid transforms, particles, and a native cable
  state/topology pointer bundle. The provider derives cable points only in a requested destination
  format, so Newton no longer knows the renderer's layout. Cable topology and destination paths now
  come from the clone plan rather than a post-clone USD lookup.
* **Breaking:** Removed Newton's finished-stage model reconstruction fallback. Newton initialization
  now requires the registry-owned clone context to have built the model, and terrain heightfields are
  imported only from global roots declared by that same plan.
* **Breaking:** Removed the legacy ``visualization_builder`` and ``visualization_deformables``
  modules and their manager-owned PhysX-to-Newton particle mirror. A Newton renderer or visualizer
  now uses the clone-built registry model and requests its dynamic transform and point pointers
  through SDP. Environment count and placement come from the clone plan rather than a walk of
  ``/World/envs``.
* Changed the clone-built Newton visualization model to require exactly one SDP transform for every
  model body. Missing or mismatched publications now fail instead of retaining clone-time state.
* **Breaking:** Removed ``NewtonManager.sync_transforms_to_usd``, ``sync_cables_to_usd`` and
  ``sync_particles_to_usd``. None of them wrote USD -- they wrote Fabric -- and Newton should not
  know that Fabric is what reads it. The Newton scene data backend now only declares what it
  publishes and the consumer requests a :class:`~isaaclab.scene_data.SceneDataFormat`. Cable and
  MPM destinations are named point publications, so there is no separate synchronization escape
  hatch. Newton only marks its publications dirty; whichever consumer draws next requests them.
* Changed deformable clone hooks to publish exact plan-owned point bindings through the Newton
  registry entry. The backend no longer installs an aggregate hook or keeps a second particle
  visual registry.
* **Breaking:** Removed ``MPMParticleSpawnerCfg.visual_update_frequency``. MPM points now update on
  each dirty publication requested by a renderer rather than on an independent frame counter.
* **Breaking:** Removed ``NewtonManager._newton_fabric_ready``. Renderers now request the backend's
  transforms through :class:`~isaaclab.scene_data.SceneDataProvider`; the clone context owns the
  prepared Fabric destination.
* Changed Newton forward kinematics and publication to run at physics initialization, forward, and
  step boundaries instead of from a render callback. Every completed step marks the native
  publications dirty for all consumer formats; SDP requests only consume them and never invoke
  physics.
