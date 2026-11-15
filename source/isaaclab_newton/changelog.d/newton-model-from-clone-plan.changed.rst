* Changed Newton replication to build the model directly from the clone plan: the builder now holds the shared
  assets and one unplaced seed world per world prototype, Newton finalizes only that seed, and
  :class:`~isaaclab_newton.physics.NewtonWorldLayout` expands every seed array to all worlds by Newton's attribute
  metadata (each domain's world, kind-dependent references, labels), placing each world with
  :func:`newton.utils.place_copies`, the placement :meth:`newton.ModelBuilder.replicate` uses. Tendons, legacy
  muscles, batteries and deformable visual meshes are expanded. The model is bitwise equal to finalizing every
  replicated world, except for particle and body colorings, which now pair each world's prototype coloring instead
  of coloring all worlds at once.
* Removed the particle and tetrahedron rest-frame rotation the cloner applied after composing a rotated asset;
  Newton's composition now rotates them.
* Changed :class:`~isaaclab_newton.physics.NewtonBackend` to build its model through the registered
  :class:`~isaaclab_newton.physics.NewtonWorldLayoutCfg` layout and to read deformable ranges from the model.
