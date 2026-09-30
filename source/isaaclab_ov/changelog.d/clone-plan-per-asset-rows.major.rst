Changed
^^^^^^^

* **Breaking:** Moved OVPhysX and OVRTX cloning into one simulation-scoped
  :class:`~isaaclab_ov.cloner.OvReplicateContext`. Both consumers resolve that same registry key,
  share one cached plan-pruned stage snapshot, and execute the plan into their native resources.
  The context owns the simulation's sole native OVRTX renderer and ``OvrtxScene``; cameras are
  per-product clients of those shared resources. The renderer no longer reads the clone plan,
  trims the snapshot, copies prims or places environment roots:
  :func:`~isaaclab.cloner.replicate` drives all of it. Declare every OVRTX camera with the scene it
  belongs to — as a
  :class:`~isaaclab.scene.InteractiveSceneCfg` field, or inside a
  :class:`~isaaclab.cloner.ReplicateSession` — rather than constructing it after replication.
  Cameras sharing the context must also resolve the same CUDA device, native logging configuration,
  and debug-stage directory; align those cfg values when migrating multi-camera scenes.
* **Breaking:** Removed ``OVRTXRenderer.prepare_stage``. The stage export it performed is
  now the cached :attr:`~isaaclab_ov.cloner.OvReplicateContext.stage_usda`. The context appends every
  planned camera product and populates the one OVRTX scene from the combined payload once. Nothing
  needs to request an export; it is created during replication.
* **Breaking:** Removed ``export_stage_to_string`` from :mod:`isaaclab_ov.renderers.ovrtx_usd`. Its
  plan pruning now belongs to the shared cloning context.
* **Breaking:** Moved the scene OVRTX draws into one concrete, ovstage-owned
  :class:`~isaaclab_ov.renderers.ovrtx_scene.OvrtxScene`. OVRTX now has one scene lifecycle and one
  host ingest format requested through the scene-data provider.
* Changed the shared OVRTX scene to be fully authored while the cloner replicates. The
  per-environment ``omni:scenePartition`` tokens, every render product's camera relationship and
  every camera transform binding all need the copies but read nothing from physics, so
  :class:`~isaaclab_ov.cloner.OvReplicateContext` finishes the combined scene in its one
  ``replicate``. :meth:`~isaaclab_ov.renderers.OVRTXRenderer.initialize` now asks that context to bind
  the authored rigid bodies, deformables and particles to the scene-data provider once and attach the
  shared scene once. Playing the simulation only connects an already-authored scene to physics.
* Changed the rigid-body, Newton deformable and Newton particle bindings so each one's discovery is
  written once, with the offsets and counts that slice Newton's
  ``particle_q`` kept as one list of ``(handle, offsets, counts)`` rows.
* Changed OVRTX population to open one combined scene, execute one clone pass and attach once,
  regardless of camera count. Each camera receives a unique render-product scope and output client;
  closing a camera leaves the shared native resources alive until the final client closes. A second
  replication or a camera joining after replication is rejected instead of mixing layouts.
* Changed OVRTX visual materials to use one clone-context-owned writer. Multi-camera scenes now
  submit each dirty material generation once to the shared native scene instead of once per camera.
* Changed OVRTX camera and scene-partition authoring to use the exact prototype and destination
  paths in :class:`~isaaclab.renderers.CameraRenderSpec`. The one required whole-stage transport
  export remains, but it no longer walks that stage to discover camera prims.
* Changed direct OvPhysX pose, joint-position, and deformable-node writes to dirty their affected
  scene-data publication immediately, so consumers request the new pointer without a physics step.
* Changed OvPhysX scene-data bindings to consume the clone plan's exact rigid/deformable paths and
  unpadded counts. Heterogeneous deformables now use count-specific native bindings, and missing or
  extra native bodies fail instead of falling back to a stage-derived layout.

Removed
^^^^^^^

* **Breaking:** Removed the module-level ``OvPhysxViewError`` alias. Catch
  :class:`~isaaclab_ov.sim.views.OvPhysxView.OvPhysxViewError` from its owning view type.
* **Breaking:** Removed the ``pre_ovrtx_renderer_stage.usda`` debug artifact written by
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.temp_usd_dir`, avoiding a second full serialization
  of the source stage. Inspect the retained ``ovrtx_renderer_stage.usda`` payload instead.
* **Breaking:** Removed the deprecated renderer-owned OVRTX scene, ``OvrtxLegacyScene``,
  ``OvstageScene``, ``make_ovrtx_scene``, ``ovrtx_use_ovstage_enabled`` and the
  ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` selector. OVRTX always uses its required ovstage dependency.

Fixed
^^^^^

* Fixed :attr:`~isaaclab.sensors.CameraCfg.background_color` being ignored by the OVRTX renderer.
* Fixed OVRTX rigid bodies being frozen under OvPhysX by binding and updating them through the
  physics-agnostic :class:`~isaaclab.scene_data.SceneDataProvider`.
* Changed rigid transforms to request OVRTX's exact host transposed double-precision matrix layout from
  the scene-data provider. OVRTX no longer owns a transform conversion kernel or source-index copy.
* Fixed OVRTX rendering on nonzero CUDA devices by selecting the camera's CUDA device before
  constructing the native renderer.
* Fixed final-client teardown after a post-clone failure by closing an OVRTX scene without detaching
  when its native renderer was never attached.
