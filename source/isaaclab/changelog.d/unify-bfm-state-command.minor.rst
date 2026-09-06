Added
^^^^^

* Added an explicit ``rng`` argument to :class:`~isaaclab.terrains.TerrainGenerator` and a
  ``function_rng`` hook on :class:`~isaaclab.terrains.SubTerrainBaseCfg` so stochastic
  sub-terrain functions can consume a caller-owned :class:`numpy.random.Generator` instead
  of the global NumPy state. The built-in height-field and mesh terrains implement it.
