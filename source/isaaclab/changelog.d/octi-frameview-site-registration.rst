Added
^^^^^

* Added :meth:`~isaaclab.sim.views.FrameView.register_frame` so sensors can pre-register
  frame prims with the active physics backend before model finalization. The camera
  sensor now pre-registers its frame during construction, letting backends that inject
  frame sites during replication (Newton) resolve the camera frame without matching
  paths against the finalized model. Backends without pre-registration ignore the call.
