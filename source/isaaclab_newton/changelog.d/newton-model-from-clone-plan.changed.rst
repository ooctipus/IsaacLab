* Changed Newton replication to build the model directly from the clone plan: the builder now holds the shared
  assets and one unplaced seed world per world prototype, Newton finalizes only that seed, and
  :class:`~isaaclab_newton.physics.NewtonWorldLayout` expands every seed array to all worlds by Newton's attribute
  metadata, placing each world with the arithmetic of :meth:`newton.ModelBuilder.replicate`. The model is bitwise
  equal to finalizing every replicated world, except for particle and body colorings, which now pair each world's
  prototype coloring instead of coloring all worlds at once.
* Changed :class:`~isaaclab_newton.physics.NewtonBackend` to build its model through the registered
  :class:`~isaaclab_newton.physics.NewtonWorldLayoutCfg` layout and to read deformable ranges from the model.
