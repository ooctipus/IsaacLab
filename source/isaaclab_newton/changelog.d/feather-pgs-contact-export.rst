Changed
^^^^^^^

* Added automatic reduced-contact export selection through
  :attr:`FeatherPGSSolverCfg.contact_export`. Supported CUDA mesh and heightfield
  pipelines use compact export. Set ``tiled`` to retain the original mapping or
  ``compact`` to require admission. This field overrides the legacy
  ``NEWTON_REDUCER_COMPACT_EXPORT`` environment variable.
