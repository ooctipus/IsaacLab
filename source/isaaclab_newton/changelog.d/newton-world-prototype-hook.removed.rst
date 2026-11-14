* Removed ``newton_builder_world_hook``. Use :func:`~isaaclab_newton.cloner.newton_world_prototype_hook`, whose
  callback runs once per world prototype, in its prototype frame, and receives the builder, the seed world index and
  the environment id labeling it.
