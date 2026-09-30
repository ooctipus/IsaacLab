Changed
^^^^^^^

* **Breaking:** Rejected ``max_visible_envs`` and ``visible_env_indices`` on
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`. Kit previously implemented the filter by
  changing visibility on the shared cloned USD stage, which affected every renderer. Use a
  Newton, Rerun, or Viser visualizer for view-local environment filtering.
* **Breaking:** Changed the Kit, Newton, Rerun and Viser visualizers to stream from a camera the
  scene cfg owns, instead of spawning or declaring camera prims themselves. Each cfg names one
  through :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_camera`. Camera pose, updates,
  renderer selection, and clone ownership remain with the scene sensor. See the ``isaaclab``
  changelog for migration guidance.
* **Breaking:** Removed the four copies of the streaming pipeline. Every backend now builds its
  panel through :class:`~isaaclab.visualizers.streaming_view.StreamingView` and differs only in
  where it puts the resulting image, so ``_setup_streaming_view``, ``_compose_streaming_frame``,
  ``_apply_streaming_camera_pose``, ``_update_owned_camera_poses``, ``_resolve_streaming_renderer_cfg``
  and their ``libosdCPU.so`` preload copies are gone, along with the ``_camera_sensor``,
  ``_camera_sensor_indices``, ``_camera_env_indices``, ``_camera_is_owned``,
  ``_generated_camera_prim_paths``, ``_streaming_camera_key`` and ``_last_streaming_composite``
  attributes on each visualizer. Read the panel through ``visualizer._streaming`` instead.
* **Breaking:** Removed the ``streaming_camera_cfg`` and ``streaming_tile_size`` overrides from
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` and
  :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`. The streaming camera's renderer and
  resolution are now stated on the camera cfg itself rather than derived from the visualizer window.
* **Breaking:** Changed ``NewtonRTXVisualizer`` from a second native OVRTX/Newton scene owner into
  an image-only presenter for the camera named by ``streaming_camera``. Declare that camera on the
  scene cfg and select its renderer through ``CameraCfg.renderer_cfg``. The presenter no longer
  accepts Newton model-display fields such as ``show_particles`` or ``particle_color``.
* Changed Newton, Rerun and Viser visualizers to acquire their Newton replication context when the
  active :class:`~isaaclab.sim.SimulationContext` constructs them before cloning. Their runtime
  viewer and server resources still initialize only after cloning.
* Removed ``BaseVisualizer.requires_forward_before_step`` use from Kit. Kit already requests the
  Fabric transform format through the scene-data provider before pumping its viewport.

Fixed
^^^^^

* Fixed the streamed environment tiles being drawn from a different random sample than the one the
  streaming camera was sized for. The tiles are resolved once, when the visualizer initializes.
* Fixed the Kit streaming panel re-rendering its camera on every step while training was paused.
  The picture cannot change, so the panel re-uploads the last composite instead.
* Fixed headless Kit visualizers claiming they pump the Kit app loop, which prevented Isaac RTX
  cameras from updating after their first frame.
* Fixed the Newton RTX camera presenter causing visual-only geometry to enter a native Newton model
  it does not draw. Native Newton viewers still request that geometry before cloning.
