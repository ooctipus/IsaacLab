Added
^^^^^

* Added per-environment visual-material support to the Newton color write path: shape → material
  bindings captured at stage import are remapped through the clone plan alongside the bound
  shapes, so each environment's clone of a per-environment
  :class:`~isaaclab.assets.VisualMaterial` owns its own ``model.shape_color`` rows and partial
  ``env_ids`` writes stay on one memoized scatter plan.
