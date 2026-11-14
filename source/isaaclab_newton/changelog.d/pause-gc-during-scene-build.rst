Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonManager` to pause Python's cyclic garbage collector from the
  creation of its builder until the first hard reset has built the scene: the collector's automatic full collections
  walked the growing replicated scene, about 1 s per start-up at 16,384 environments. The reset collects once before
  it freezes the scene; the collection after each CUDA graph capture now also runs while the collector is paused.
