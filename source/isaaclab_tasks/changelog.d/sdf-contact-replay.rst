Changed
^^^^^^^

* Changed multi-slot NIST Factory Board tasks using Newton MJWarp to enable
  solver sleeping and bounded SDF contact replay. Custom configurations that
  require the previous behavior should disable sleeping and set the per-world
  replay capacity to zero after layout-dependent task configuration.
