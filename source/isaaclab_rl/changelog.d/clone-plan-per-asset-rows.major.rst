Changed
^^^^^^^

* **Breaking:** Removed the ``--disable_fabric`` option from RL training, playback, and simple-agent
  entry points. Rendering now requests its representation through the scene-data provider; remove
  the option from launch commands instead of selecting a separate USD synchronization path.
* **Breaking:** Changed ``--video`` to require a capture-capable visualizer already declared in
  ``env_cfg.sim.visualizer_cfgs``, or a preconfigured
  ``VideoRecorderCfg(source="sensor:<name>")``. RL entry points and simple agents no longer inject a
  Kit visualizer implicitly; declare the renderer, visualizer, and camera in the environment config.
