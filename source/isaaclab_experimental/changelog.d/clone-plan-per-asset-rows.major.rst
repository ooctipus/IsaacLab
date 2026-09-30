Changed
^^^^^^^

* **Breaking:** Changed Warp environments to construct ``cfg.scene.class_type(cfg.scene)`` inside
  the common replication lifecycle. Custom direct Warp environments must declare spawned assets on
  ``cfg.scene``; ``_setup_scene`` may bind task-local references but must not construct or clone
  scene entities.
* **Breaking:** Removed the ``rgb_array`` render mode and its implicit Kit viewport capture from
  Warp environments. Declare a planned camera and ``VideoRecorderCfg(source="sensor:<name>")`` to
  record frames.
