Changed
^^^^^^^

* **Breaking:** Moved the Factory, Forge, and AutoMate direct-task assets under ``env_cfg.scene``
  and delegated their construction and cloning to the shared scene lifecycle. Use ``env_cfg.task``
  instead of ``env_cfg.tasks[env_cfg.task_name]`` and access the robot configuration through
  ``env_cfg.scene.robot``.
* Added the ``newton_vbd`` physics preset to the direct and manager-based Cartpole task families,
  and allowed direct Cartpole camera tasks to request auxiliary image outputs while using the first
  output for observations.
* Added the typed ``visualizer=NAME`` preset selector and public ``resolve_config`` utility for
  resolving presets and dotted scalar overrides on data-only config roots.
* Changed :class:`~isaaclab_tasks.utils.PresetCfg` alternatives to remain shared until selection;
  resolution copies and wraps only the active branch instead of every inactive backend choice.
* Added ``MultiBackendCameraCfg`` and ``MultiBackendSceneCfg`` so selecting ``newton_rtx`` also
  selects a plan-owned camera while every other visualizer keeps the camera absent by default.
* Fixed direct tasks authoring ``/World/Light`` after replication. Cartpole, Pendulum, Handover,
  Reorient, locomotion, Anymal-C, Humanoid-AMP, Factory, Assembly, and Disassembly now declare the
  light with the rest of their plan-owned scene. These tasks no longer override ``_setup_scene``;
  their base environment constructs the declared scene automatically.
* Fixed direct Cartpole camera tasks constructing assets and cameras outside the common scene
  lifecycle; their direct cfg now owns the complete cloned scene. Use
  ``env_cfg.scene.camera`` instead of the removed ``env_cfg.tiled_camera`` field.
* **Breaking:** Renamed ``ObjectUniformPoseCommandCfg.success_visualizer_cfg`` to
  ``success_marker_cfg``. The marker is always active rather than controlled by ``debug_vis``, so
  the new name makes it part of the original clone plan before command managers initialize.
* Fixed Shadow Hand, Handover, and velocity task configs still mutating removed spawner, viewer, or
  nested Newton solver fields after the backend cfgs became declarative and flat.
* Removed construction-time camera renders and lazy contact, ray-cast, and joint-wrench updates from
  representative observation terms by declaring their allocated or fixed output-shape contracts.
* Removed the Blueprint stacking task's duplicate camera observation and per-environment PNG writes;
  camera observations now use the shared non-updating shape contract and recorder-owned tensors.
