Added
^^^^^

* Added contact-capacity servicing to :class:`~isaaclab_newton.physics.NewtonBackend`: when the collision pipeline
  offers ``service_capacity``, the backend sizes contact rows from the state the first replay of a new captured step
  starts from, grows the buffers from the measured demand once per step, and recaptures the step after a growth.
* Added :attr:`~isaaclab_newton.physics.FeatherPGSSolver.contact_drop_worlds`, the worlds whose rigid contacts the
  last step dropped for want of rows, reported after each step without a synchronization.
