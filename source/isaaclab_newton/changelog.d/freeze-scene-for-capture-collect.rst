Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonManager` to freeze the garbage collector's objects once a hard reset
  has built the scene, model and solver, and to unfreeze them when the backend closes. The full collection after every
  CUDA graph capture, which prevents the capture-stream segfaults, now walks only the objects allocated since instead
  of the whole replicated scene.
