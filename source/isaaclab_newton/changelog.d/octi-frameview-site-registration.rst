Added
^^^^^

* Added :meth:`~isaaclab_newton.sim.views.NewtonSiteFrameView.register_frame` to
  pre-register a frame's site requests before model finalization, mirroring direct
  :meth:`~isaaclab_newton.physics.NewtonManager.cl_register_site` users such as the
  IMU. Views constructed after finalization for a registered frame initialize from
  the sites injected during replication instead of resolving bodies against the
  finalized model.
