* **Breaking:** Migrated SO101 keyboard tasks to MJWarp, ``AssetBase`` authoring, and task-local
  numeric selections. Use ``NewtonSelectorCfg`` from ``selection_paths`` and the reset/action
  configuration in ``so101_env_cfg.py`` instead of asset-view MDP terms or the removed PhysX presets.
  Symbolic paths resolve to integer indices before numeric binding.
* Added six-to-108-key variants, six-key articulation partitions, and growable-world and
  exact-population tasks sharing the typing MDP. Reset switches restore geometry, inertia, joint
  properties, visibility, and active-key masks together. Use ``TYPING_KEYBOARD_VARIANTS`` instead
  of ``TYPING_KEYBOARD_POOL``; partition selectors use ``Keyboard/parts/part_*/keys`` and
  ``Keyboard/parts/part_*/joints``. ``keyboard_variants=()`` selects the all-active 108-key baseline.
* Added prepared constant caching for the sleeping baseline, variant-specific reset snapshots,
  matched training presets, and fused device MDP operations. Resets select another registered
  variant when alternatives exist; snapshots must match the selected variant.
* Preserved continuing worlds across exact-population replacement and added configurable
  redistribution with explicit administrative truncation. Growable worlds stage variant changes
  through world handles. Successor observations include typing updates and history without
  advancing live command state or random-number generators.
