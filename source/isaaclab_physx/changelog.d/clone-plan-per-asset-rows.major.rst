Changed
^^^^^^^

* **Breaking:** Changed ``PhysxManager`` to publish native tensors and dirty state without owning
  a second dynamic USD or Fabric synchronization path. PhysX no longer loads a Fabric extension,
  updates Fabric, or detaches and reattaches Fabric around timeline transitions. It now only
  publishes its transforms as
  :class:`~isaaclab.scene_data.SceneDataFormat`, and whatever draws the stage calls
  :meth:`~isaaclab.scene_data.SceneDataProvider.request_transforms`, so the conversion runs
  once at request time against the current pointer.
* **Breaking:** Removed ``SimulationCfg.use_fabric`` and the ``--disable_fabric`` CLI path.
  :class:`~isaaclab.sim.views.FrameView` now always selects the plan-owned
  :class:`~isaaclab_physx.sim.views.PhysxFrameView` for PhysX.
* Changed deferred PhysX articulation refresh to run from its scene-data backend when state is
  requested, instead of from a physics-manager render callback. Direct pose, joint-position, and
  deformable-node writes now dirty their affected scene-data publication at the producer boundary.
* Changed PhysX rigid and deformable scene-data views to bind only the clone plan's exact declared
  paths. Removed the post-clone USD traversal, wildcard inference, padded-count fallback and lazy
  rediscovery paths.
* Changed ``PhysxManager`` to register only its native PhysX replicator. Cameras, deformables and
  render consumers now own the shared USD/Fabric clone context when they require one.
* **Breaking:** Changed Isaac RTX scene-partition and render-product setup to trust the exact camera
  paths in :class:`~isaaclab.renderers.CameraRenderSpec`, instead of finding or validating camera
  prims by walking the USD stage. Removed the Isaac Sim 4.5 compatibility walk that disabled
  instanceability across the whole stage for segmentation cameras.
* Changed Isaac RTX cameras to share one simulation-owned runtime for Replicator, renderer settings,
  and Hydra attachment. All Isaac RTX camera configurations in one simulation must now declare the
  same process-global settings.
* **Breaking:** Removed ``IsaacEvents`` and ``PhysxManager.get_physics_sim_device``. Register
  lifecycle listeners with :class:`isaaclab.physics.PhysicsEvent` and use
  :meth:`isaaclab.physics.PhysicsManager.get_device`. PhysX now publishes ``PHYSICS_READY`` once
  when its view is created and ``PRIM_DELETION`` once during teardown, without a duplicate Carbonite
  compatibility bus.
