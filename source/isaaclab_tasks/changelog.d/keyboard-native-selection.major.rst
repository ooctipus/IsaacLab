Changed
^^^^^^^

* Migrated the SO101 keyboard task to MJWarp-only AssetBase authoring and task-local Newton selectors,
  preserving the single-articulation 108-key observation/action layout and reset curriculum.
* Added flat, compact episode-aware body, coordinate, and DOF selections without Newton articulation views.
* **Breaking:** Replaced keyboard command asset-name/reset-asset fields and asset-based MDP parameters with
  ``NewtonSelectorCfg`` bindings. Custom keyboard configurations must use the selectors in ``so101_env_cfg.py``;
  reset IK now uses ``KeyboardResetIKCfg``, and ``reach_key`` accepts selected bodies and ordered tip offsets.
  The keyboard PhysX presets were removed. Other tasks and shared MDP terms were unchanged.
