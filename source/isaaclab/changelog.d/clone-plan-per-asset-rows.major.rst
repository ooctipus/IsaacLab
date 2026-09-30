Changed
^^^^^^^

* **Breaking:** Changed ``isaaclab benchmark startup`` to measure unprofiled wall time by default.
  Pass ``--profile`` for dense cProfile attribution; its instrumented wall time is labeled separately
  and must not be used as a performance baseline. Results now include the tracked worktree-diff
  fingerprint so two dirty snapshots at the same commit cannot be mistaken for identical code.
* Changed :class:`~isaaclab.envs.ManagerBasedRLEnv` to construct vectorized Gym boxes directly
  from scalar bounds, avoiding a temporary tiled copy of every observation and action bound.
* **Breaking:** Changed the local remote-asset mirror to be authoritative on warm startup instead
  of revalidating every cached file against the server. Pass ``force_download=True`` to
  :func:`~isaaclab.utils.assets.retrieve_file_path` or delete the local cache to refresh an asset.
* **Breaking:** Changed :func:`~isaaclab.cloner.make_clone_plan` to always give each asset its own
  row. It previously collapsed a homogeneous scene into a single row for the environment root, so
  ``plan.sources`` was ``("/World/envs/env_0",)`` regardless of how many assets the environment
  held. Read per-asset rows through :mod:`isaaclab.cloner.query` instead of matching an environment
  root row; :func:`~isaaclab.cloner.replicate` still hands a backend one whole-environment copy
  when every environment is a copy of the same source, so backend behavior is unchanged.
* Changed :class:`~isaaclab.cloner.ReplicateSession` to publish its plan through
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plan` when the session opens rather than when it
  closes, so assets built inside the session can resolve where their copies will land.
  :class:`~isaaclab.scene.InteractiveScene` owns this lifecycle even when its cfg declares no assets.
* Added :attr:`~isaaclab.cloner.ClonePlan.semantic_tags`, retaining cfg-declared semantic tags per
  row so renderers can derive segmentation labels from the plan without discovering prims or labels
  on the cloned stage. Labels embedded only in a USD asset are intentionally outside this contract.
* Added :attr:`~isaaclab.scene.InteractiveSceneCfg.geometry_prim_paths` for scene consumers that
  need prototype geometry but are not themselves scene sensors. The scene forwards these paths to
  its clone plan explicitly; manager configurations are no longer searched during cloning.
* Changed :class:`~isaaclab.sensors.camera.Camera` to derive its
  :class:`~isaaclab.renderers.CameraRenderSpec` from the clone plan in ``__init__``, instead of
  globbing the stage for camera prims once replication had run. Renderers that clone the scene
  themselves therefore no longer need a fully cloned USD stage. A camera the plan does not cover
  now raises when it is constructed; declare every camera through the scene or a replication
  session. A camera whose prim comes from a referenced asset is covered by that asset's row, so
  ``spawn=None`` cameras need no row of their own.
* Added :attr:`~isaaclab.sensors.camera.Camera.prim_paths`, the camera's USD path in each
  environment it covers, derived from the clone plan rather than from the stage.
* Changed camera initialization to read each clone-plan prototype's authored calibration once;
  equivalent cloned destinations no longer repeat the same USD attribute reads per environment.
* Added :attr:`~isaaclab.renderers.CameraRenderSpec.camera_source_prim_paths` and removed
  ``camera_path_relative_to_env_0``. Renderers receive the plan's exact prototype and destination
  camera paths instead of reconstructing either from an assumed environment namespace.
* **Breaking:** Removed ``CameraRenderSpec.view_count`` and turned ``num_instances`` into a read-only
  property, since both restated ``len(camera_prim_paths)``. Read
  :attr:`~isaaclab.renderers.CameraRenderSpec.num_instances` and stop passing either to the
  constructor.
* Changed :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage` to a no-op hook
  instead of an abstract method, for renderers that build their scene while the cloner replicates.
* **Breaking:** Removed ``AssetBaseCfg.cloning_contexts``, ``SensorBaseCfg.cloning_contexts``,
  ``BaseRenderer.clone_context``, ``BaseVisualizer.clone_context``, ``REPLICATION_QUEUE``,
  ``queue_replication`` and the backend ``PHYSICS_CONTEXT`` aliases. Physics, renderers and
  visualizers now register clone-capable backends through
  :meth:`~isaaclab.sim.SimulationContext.get_or_create_backend`; the replication session dispatches
  its one plan directly from that registry instead of scraping consumers or routing per cfg.
* **Breaking:** Required :attr:`~isaaclab.sim.SimulationCfg.physics` and made
  :class:`~isaaclab.sim.SimulationContext` construct exactly ``cfg.physics.class_type(cfg.physics)``.
  Pass a concrete backend cfg such as ``SimulationCfg(physics=PhysxCfg())``. The context no longer
  accepts an omitted cfg, an unresolved ``PhysxAutoCfg``, or an implicit PhysX fallback. Pass that
  cfg as the first argument to :func:`~isaaclab.sim.build_simulation_context`; its ``dt`` and
  ``gravity_enabled`` shortcuts were removed. Its ``add_ground_plane``, ``add_lighting`` and
  ``auto_add_lighting`` scene-authoring shortcuts were also removed; declare ground and light cfgs
  in the replication session instead.

* Added :func:`~isaaclab.cloner.query.env_root_paths`, the environment root prim paths a plan
  replicates into.
* **Breaking:** Changed :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage` to take
  the :class:`~isaaclab.cloner.ClonePlan` instead of ``num_envs``, so a renderer authoring
  per-environment attributes reads where the environments are rather than assuming how they are
  named. :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` no longer hardcodes
  ``/World/envs/env_{i}`` when writing ``omni:scenePartition``.
* **Breaking:** Removed ``RenderContext``. :class:`~isaaclab.sim.SimulationContext` now owns camera
  renderers directly, the cloner calls each renderer's ``prepare_stage`` once after replication,
  and :class:`~isaaclab.sensors.Camera` performs its per-frame renderer calls directly. Renderer
  initialization still occurs exactly once at the camera's post-clone initialization point. Each
  camera constructs its exact renderer as ``renderer_cfg.class_type(renderer_cfg)``; matching
  native resources share independently through the simulation-scoped backend registry.
* **Breaking:** Removed the ``Renderer`` factory, ``InteractiveScene.initialize_renderers``,
  ``TiledCamera``, ``TiledCameraCfg``, deprecated RTX fields on ``CameraCfg``, and
  ``CameraISPMode``. Use :class:`~isaaclab.sensors.Camera` with an explicit
  ``CameraCfg.renderer_cfg`` and pass a concrete ``isp_cfg`` or ``None``.
* **Breaking:** Replaced the visualizer streaming-camera fields with a single
  :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_camera` that says where the panel's picture
  comes from: a ``str`` naming its planned prim-path expression is required when streaming is enabled.
  Visualizers no longer accept a :class:`~isaaclab.sensors.CameraCfg`, declare sensors through
  :class:`~isaaclab.scene.InteractiveScene`, move cameras, or drive camera updates. Declare the
  camera directly on the scene cfg so it enters the clone plan, then select its prim path.

  * ``streaming_sensor_prim_path="/World/envs/*/Camera"`` becomes
    ``streaming_camera="{ENV_REGEX_NS}/Camera"``.
  * ``streaming_cam_renderer="ovrtx"`` becomes ``scene.<camera>.renderer_cfg=OVRTXRendererCfg()``.
  * ``streaming_cam_target_prim_path`` and ``streaming_cam_eye`` become the scene camera's
    ``prim_path`` parent and ``offset`` respectively.
* **Breaking:** Removed the ``tiled_cam_*`` deprecation aliases on
  :class:`~isaaclab.visualizers.VisualizerCfg` and the ``__post_init__`` shim that forwarded them.
  Use the ``streaming_*`` fields directly.
* **Breaking:** Replaced :mod:`isaaclab.envs.utils.camera_view` with
  :mod:`isaaclab.visualizers.streaming_view`, which owns the streaming panel end to end as
  :class:`~isaaclab.visualizers.streaming_view.StreamingView`: it resolves the camera and the
  environment tiles once when a visualizer initializes and composites the tiles. Every backend now
  shares one implementation.

  * ``compose_streaming_grid``, ``camera_gt_batch`` and ``resolve_streaming_envs`` moved across
    unchanged; ``VISUALIZER_TILED_CAMERA_MAX_TILES`` is now
    :data:`~isaaclab.visualizers.streaming_view.MAX_STREAMING_TILES`.
  * Removed ``env_path_from_template``, ``find_camera_by_prim_path``, ``resolve_streaming_camera``
    and ``resolve_streaming_renderer_cfg``: a camera is named, not matched by prim-path pattern, and
    a renderer is a cfg rather than a string.
  * Removed ``prim_world_positions``, ``apply_camera_target_positions`` and
    ``apply_camera_view_from_origins``. Scene sensors own their poses instead.
  * Removed ``compute_tile_resolution``: the declared camera carries its own ``width``/``height``,
    so the tile size is no longer derived from the visualizer window (and no longer drawn from a
    second, independent random sample of the streamed environments).
  * Removed ``create_visualizer_camera``, ``evict_visualizer_camera``, ``remove_generated_prims``,
    ``ensure_camera_initialized``, ``resolve_tiled_env_indices``, ``resolve_mono_env_index``,
    ``camera_rgb_batch`` and ``compose_rgb_grid_tensor`` together with the shared streaming camera
    registry. Cameras come from the scene, so nothing spawns per-env prims, caches a renderer
    singleton, or deletes prims on close.
* **Breaking:** Removed the ``streaming_camera_cfg``, ``streaming_tile_size`` and
  ``streaming_env_count`` hooks from :class:`~isaaclab.visualizers.VisualizerCfg`. Declare a camera
  on the scene cfg and select it by name instead of overriding a visualizer hook.
* **Breaking:** Made visualizer configs declarative by removing ``clone_context``,
  ``create_visualizer`` and ``get_visualizer_type``. :class:`~isaaclab.sim.SimulationContext` now
  constructs each visualizer through its concrete ``cfg.class_type(cfg)`` before scene cloning and
  initializes it afterward.
* **Breaking:** Removed the ``newton`` visualizer CLI name and ``visualizer:newton`` recorder source.
  Use ``newton_gl`` and ``visualizer:newton_gl`` respectively.
* **Breaking:** Removed the deprecated ``isaaclab.devices`` exports and forwarding modules for
  ``OpenXRDevice`` and ``ManusVive``. Use :class:`isaaclab_teleop.IsaacTeleopDevice` with an
  ``XrCfg`` declared as ``env_cfg.scene.xr_anchor``.
* Added :meth:`~isaaclab.sim.SimulationContext.get_or_create_backend`, which gives physics,
  renderers and visualizers one simulation-scoped native resource when their configs resolve to
  the same backend class. The backend class is now the complete registry key; ``resource_key`` was
  removed from physics, renderer, visualizer and clone configs. Pass the class followed by its
  constructor arguments; zero-argument factory callbacks were removed. Its ``clone_role``
  records whether a clone backend is required by physics or independently by a scene consumer,
  preserving ``replicate_physics=False`` when both share one registered backend. Registry resources
  are cleared once after all consumers close.
* Changed USD clone ownership so PhysX cameras and deformables declare their scene resource and a direct
  :class:`~isaaclab.cloner.ReplicateSession` declares it by backend type when
  ``replicate_physics=False``. Headless PhysX simulations with native replication no longer run a
  redundant USD clone pass.
* Changed cfg-built sensors to register themselves with the scene-data provider. Streaming panels
  resolve their named camera there, without publishing an interactive-scene object or adding a
  second camera registry.
* Changed the streaming panel to fetch each ground-truth type for all of its tiles in one indexed
  read rather than one read per tile, which the Kit panel already did and the Newton, Rerun and
  Viser panels did not. Measured 1.3x faster per composited frame at 32 tiles of 320x240 rgb+depth.
* **Breaking:** Removed the source-side queries ``isaaclab.cloner.query.path_to_clone`` and
  ``isaaclab.cloner.query.path_env_ids``. Nothing walked the plan from the prototype side once
  cameras and sensors resolved themselves clone-side. Use
  :func:`~isaaclab.cloner.query.path_to_source` to go from a clone path back to its prototype, or
  :func:`~isaaclab.cloner.query.iter_sources` to visit every variant behind a destination template
  together with the environments it reaches.
* **Breaking:** Removed ``isaaclab.sim.resolve_matching_prims_from_source``. The unused helper
  retained a finished-stage discovery fallback when a path was absent from the clone plan; consumers
  must resolve their declared paths from the plan instead.
* **Breaking:** Removed the public pre-clone ``PhysicsManager.initialize(sim_context)`` lifecycle.
  :class:`~isaaclab.sim.SimulationContext` now binds each constructed manager internally before
  cloning; runtime model initialization remains in the hard reset after cloning.
* Fixed :func:`~isaaclab.cloner.make_clone_plan` skipping any asset cfg whose ``prim_path`` still
  held the ``{ENV_REGEX_NS}`` macro. :class:`~isaaclab.scene.InteractiveScene` expands the macro for
  the assets it collects, but an environment that builds its own assets hands the cfg over
  unexpanded, so the asset silently got no row and was never cloned. The plan now expands it through
  :func:`~isaaclab.cloner.expand_env_regex_ns` against the same ``env_template`` it matches with.
* Fixed clone-plan traversal stopping at a composite sensor's own plan-bearing fields and omitting
  nested assets such as a visuo-tactile sensor's camera. A cfg now contributes its own row and then
  continues normal dataclass traversal, so nested cameras enter the same plan without being passed
  separately to :class:`~isaaclab.cloner.ReplicateSession`.
* Fixed :class:`~isaaclab_ov.renderers.OVRTXRenderer` authoring an unparsable render product when
  its camera sat outside the clone plan. The camera's path below its environment root was empty, so
  the render product targeted ``/World/envs/env_0/`` and the exported USD failed to load with a
  parse error. OVRTX now says which declaration is missing instead.
* Changed :class:`~isaaclab_tasks.core.cartpole.CartpoleCameraEnv` and
  :class:`~isaaclab_tasks.core.reorient.ShadowHandCameraEnv` to build their assets inside a
  :class:`~isaaclab.cloner.ReplicateSession` rather than cloning from environment 0 afterwards. A
  camera resolves where its copies land from the plan, so the plan has to be published before the
  camera is built, which is the order every other scene already follows.
* **Breaking:** Changed :attr:`~isaaclab.cloner.UsdReplicateContext.replicate_priority` from ``100``
  to ``-100`` so USD authors the environment copies before any other backend runs, and gave
  :class:`~isaaclab_ov.cloner.OvReplicateContext` ``-200`` so it snapshots the plan prototypes once
  for both OVPhysX and OVRTX before those copies exist. A physics backend parses the stage to build
  its model, so it has to run after the copies are there; USD used to run last, and a backend that
  read the stage saw only environment zero. A custom context that must run before USD needs a
  priority below ``-100``.
* Fixed :func:`~isaaclab.cloner.usd_replicate` dropping an environment's origin when no row copied
  the environment root itself. The root was then created only as an ancestor of the asset below it,
  which carries no prim type, and only an ``Xformable`` prim honors ``xformOp:*`` -- so every
  environment composed at the prototype's origin, stacked on top of each other. The root it creates
  is now typed ``Xform`` before the transform is authored onto it.
* **Breaking:** Removed the ``XformPrimView`` compatibility alias and the ambiguous view-level
  ``get_scales`` / ``set_scales`` methods. Batched runtime frames now use the active backend's
  :class:`~isaaclab.sim.views.FrameView`; use its explicit local- or world-space scale APIs and
  writer scopes. :class:`~isaaclab.sim.views.UsdFrameView` remains available for explicit USD
  access. Static USD authoring retains the exact prim returned by its spawner instead of recovering
  it from the cloned stage.
  Construct a ``FrameView`` before replication with the active ``simulation_context``, then call
  ``initialize(plan, scene_data_provider)`` after cloning. The removed ``physics_manager``, ``stage``
  and ``clone_context`` constructor arguments no longer hide clone-resource ownership.
* **Breaking:** Changed :func:`~isaaclab.cloner.make_clone_plan` to describe global assets -- those
  outside the per-environment namespace, such as a ground plane -- instead of dropping them. A
  global row's destination equals its source and carries no ``"{}"``, and its mask is all ``False``,
  so nothing copies it; the plan now names every asset the scene declares, including cfgs such as
  terrain importers that author their global subtree without a spawner.
  :attr:`~isaaclab.scene.InteractiveScene.global_prim_paths` reads them from the plan, replacing the
  list the scene accumulated by globbing the stage. Code that walks ``plan.sources`` expecting a
  prototype to clone must skip rows whose destination has no clone slot.
* **Breaking:** Added :class:`~isaaclab.assets.Asset` as the default
  :attr:`~isaaclab.assets.AssetBaseCfg.class_type` and made
  :class:`~isaaclab.assets.AssetBase` inherit from it. The concrete ``Asset`` owns plan-backed
  spawning for authoring-only assets such as ground planes and lights, without adding physics views
  or data buffers. :attr:`~isaaclab.scene.InteractiveScene.extras` now stores an ``Asset`` instead
  of its ``AssetBaseCfg``; use ``scene.extras[name].cfg`` when configuration access is needed.
  The exact prim returned by the spawner is available as ``scene.extras[name].prim``.
  Custom composition roots should construct ``cfg.class_type(cfg)`` inside their replication
  session instead of calling ``cfg.spawn.func`` directly.
* **Breaking:** Changed :func:`~isaaclab.cloner.clone_plan_from_env_0` to accept one
  :class:`~isaaclab.cloner.CloneCfg` and an explicit flat asset/sensor cfg manifest. Standalone
  composition roots pass that manifest directly; the cloner does not inspect an environment or
  :class:`~isaaclab.scene.InteractiveSceneCfg` to discover participants. Direct environments
  declare their assets on ``scene`` and let :class:`~isaaclab.scene.InteractiveScene` own cloning.
* **Breaking:** Removed ``SpawnerCfg.spawn_path`` and the multi-asset ``spawn_paths`` fields.
  :func:`~isaaclab.cloner.make_clone_plan` no longer mutates configuration objects. Construction
  reads exact prototype sources from :func:`~isaaclab.cloner.query.cfg_source_paths`; multi-asset
  spawners accept those paths as their first argument, including ``None`` slots for inactive
  variants.
* Added an exact frame, rigid-body, deformable, and cable scene layout to
  :class:`~isaaclab.cloner.ClonePlan`. Replication declares it once from populated prototype rows,
  including heterogeneous masks, global bodies, cable segment counts, visual/simulation mesh paths,
  and unpadded node counts. Consumers now fail on a missing or conflicting declaration instead of
  rediscovering or approximating the cloned stage.
* **Breaking:** Removed ``NewtonActuatorAdapter.from_usd``. PhysX-family actuator initialization now
  consumes immutable per-joint declarations parsed into the clone plan, rather than traversing the
  finished USD stage.
* **Breaking:** Removed the unused ``RayCasterCfg.spawn`` field. Ray casters reference an existing
  planned body or frame and no longer claim a clone row for a spawner they never execute. To create
  a dedicated frame, declare an :class:`~isaaclab.assets.AssetBaseCfg` with
  :class:`~isaaclab.sim.SensorFrameCfg` beside the ray-caster cfg.
* **Breaking:** Replaced scene-data copy and synchronization methods with
  :meth:`~isaaclab.scene_data.SceneDataProvider.request_transforms` and
  :meth:`~isaaclab.scene_data.SceneDataProvider.request_points`. A native-format request aliases
  the published pointer; another format runs one conversion for the current dirty generation.
  Fabric destinations follow this same request path and carry their per-prim source indices.
* Added :class:`~isaaclab.scene_data.SceneDataFormat` variants ``TransposedMatrix44d``,
  ``FabricMatrix44`` and ``FabricMeshPoints``, which describe renderer sink layouts the same way
  the existing variants describe native physics pointers. Removed the unused generic ``Matrix44``
  format; request ``TransposedMatrix44d`` for a transposed double-precision matrix sink.
* Fixed Fabric transform requests writing only the cached world matrix. They now convert into the
  authoritative local matrix before hierarchy propagation, so Isaac RTX sees non-PhysX motion in
  body prims and their visual descendants.
* **Breaking:** Removed ``PhysicsManager.sync_transforms_to_stage`` and
  ``sync_geometries_to_stage``. Bringing the render representation up to date is a scene-data
  concern, and it is now pulled rather than pushed: a physics backend publishes its data and says
  when it changed, and the consumer asks :class:`~isaaclab.scene_data.SceneDataProvider` for that
  data in the format it draws from. Renderers and visualizers no longer reach through the
  simulation context into the physics manager.
* **Breaking:** Collapsed transform and point publications into
  :class:`~isaaclab.scene_data.SceneDataPublication`, containing only a native-format pointer or
  pointer bundle and dirty latch. Flat counts come from pointer shapes; padded deformable and cable
  counts, ordering, and destination mappings come only from :class:`~isaaclab.cloner.ClonePlan`.
  Removed the separate transform metadata properties, publication types, and SDP mapping APIs.
* **Breaking:** Removed ``PhysicsManager.pre_render``, ``after_visualizers_render`` and
  ``video_capture_backend``, plus ``BaseVisualizer.requires_forward_before_step``. A scene-data
  backend now publishes its pointers and dirty latches at reset, forward, and step boundaries.
  Scene-data requests only pass those pointers through or convert them, so neither the simulation
  render loop nor a visualizer invokes physics to prepare a frame.
* Added USD Fabric destination authoring to the shared USD clone context. Exact frame, transform,
  and point destinations are bound from the clone plan before consumers initialize; the scene-data
  provider only passes through or converts the prepared pointers.
* **Breaking:** Added :attr:`~isaaclab.physics.PhysicsEvent.PRIM_DELETION` and removed the
  platform-subscription hook from :class:`~isaaclab.physics.PhysicsManager`. Lifecycle callbacks
  are now weakly held and dispatched directly by their constructed manager.
* **Breaking:** Removed symbolic articulation ordering and its live-stage discovery. Set
  :attr:`~isaaclab.assets.ArticulationCfg.joint_ordering` and ``body_ordering`` to explicit,
  complete name permutations when a backend-independent public order is required.
* **Breaking:** Required ``randomize_visual_texture_material`` and ``randomize_visual_color``
  event terms to declare ``visual_prim_path`` relative to the asset. The terms no longer infer a
  ``visuals`` hierarchy or fall back to every descendant prim; the obsolete Replicator graph path
  and its unused ``event_name`` argument were removed.
* **Breaking:** Changed operational-space actions to reference plan-owned contact and
  frame-transformer sensors by ``contact_sensor_name`` and ``task_frame_sensor_name``. Declare
  those sensors on the scene instead of passing a relative frame path or constructing hidden
  sensors after cloning.
* Fixed Newton actuator properties being authored on only the first prototype of a heterogeneous
  articulation. Authoring now visits every populated prototype named by the clone plan.
  **Breaking:** Newton-native actuator authoring no longer searches for an articulation when no
  plan owns it; construct the articulation inside :class:`~isaaclab.cloner.ReplicateSession`.
* Fixed rigid-body layout order depending on USD authoring order. The clone plan now derives a
  canonical parent-before-child articulation order from its planned joints, so physics publication
  and renderer consumption share one order without a backend mapping or state copy.
* Made observation output shapes implementation-owned. Camera, ray-caster, and other expensive
  terms now derive their allocated shapes without evaluating lazy data during observation-manager
  construction; task configurations no longer duplicate output dimensions.
