Changed
^^^^^^^

* Changed Newton replication to build the model directly from the clone plan: the builder now holds the shared
  assets and one unplaced seed world per world prototype, Newton finalizes only that seed, and
  :class:`~isaaclab_newton.physics.NewtonWorldLayout` expands every seed array to all worlds by Newton's attribute
  metadata, placing each world with the arithmetic of :meth:`newton.ModelBuilder.replicate`. The model is bitwise
  equal to finalizing every replicated world.
* Changed :class:`~isaaclab_newton.physics.NewtonBackend` to build its model through the registered
  :class:`~isaaclab_newton.physics.NewtonWorldLayoutCfg` layout and to read deformable ranges from the model.

Removed
^^^^^^^

* Removed ``newton_builder_world_hook``. Use :func:`~isaaclab_newton.cloner.newton_world_prototype_hook`, whose
  callback receives the builder, the seed world index and the environment id labeling it, once per world prototype
  in its prototype frame.
